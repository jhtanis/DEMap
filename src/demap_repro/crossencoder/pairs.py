#!/usr/bin/env python3
"""Build supervised cross-encoder training/dev pairs from the candidate union.

Reads the HGBC candidate-union feature table and emits pointwise (query,
candidate) training pairs for cross-encoder fine-tuning.

- TRAIN  (default ``train`` — the FULL canonical train split): positives =
  ``is_label`` (includes injected-gold rows, which are true query->CDE matches
  and are train-only); negatives = ``~is_label & ~is_injected_gold``
  (retrieval-hard union candidates). Full ``train`` is the default because it
  yields materially stronger cross-encoders than the small ``val_train`` slice
  (see artifacts_v3_cdisc fulltrain evidence: +~0.057 R@5 / +0.12 R@1). Override
  with ``--train-split val_train`` to reproduce the legacy slice-trained CE.
  NOTE: ``train`` is not reachability-filtered, so some train golds are
  off-production — they remain valid training positives (text from the full
  catalogue); the feature table is expected to have dropped any train query left
  with no positive.
- DEV    (default ``val_dev``): the deployable union only (``~is_injected_gold``),
  labelled by ``is_label``. This reproduces the standard val_dev deployment
  denominator: queries whose gold is injected-only appear with negatives only
  and correctly count as misses. val_dev stays held out for CE selection.

Query text = ``query_text_q3``; candidate text via the shared recipe logic in
``score_crossencoder.py``. NEVER reads test/external holdouts. Pass ``--out-dir``
(the default is a legacy path); the final-reranker workflow writes under
``artifacts/final_reranker/crossencoder*/train_pairs/``.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_ce_training_pairs.py \
        --feature-table <pool with a `train` split> \
        --out-dir <dir> --cde-text-recipe SN_DEC_DEF_PQT_PV
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

REPO_ROOT = data_root()

from demap_repro.crossencoder import score as sc

# Default CE training universe = FULL canonical `train` (not the val_train slice).
# Configurable via --train-split / --dev-split; main() sets these globals from args
# before build_split / assert_integrity run.
TRAIN_SPLIT = "train"
DEV_SPLIT = "val_dev"
FORBIDDEN_SPLITS = {
    "test", "external_holdout_org", "external_holdout_gdc_altnames",
    "external_holdout_gdc_questiontext", "cimac_appendix_a_eval",
}
OUT_DATA_DIR = REPO_ROOT / "artifacts_v3_cdisc/crossencoder/finetune/data"
OUT_COLS = ["winner_id", "split", "query_id", "cde_id", "label",
            "is_injected_gold", "query_text", "candidate_text"]


def build_split(ft, split, query_col, cde_text_map, is_dev, neg_per_query):
    g = ft[ft["split"] == split].copy()
    g["query_id"] = g["query_id"].astype(str)
    g["cde_id"] = g["cde_id"].astype(str)
    g["is_injected_gold"] = g["is_injected_gold"].astype(bool)
    g["is_label"] = g["is_label"].astype(bool)

    if is_dev:
        # Deployable union only; positives are non-injected by construction.
        rows = g[~g["is_injected_gold"]].copy()
    else:
        # Train: all deployable rows + injected-gold positives (train-only).
        # Injected rows are always is_label, so they are never negatives.
        rows = g[(~g["is_injected_gold"]) | (g["is_label"])].copy()
        if neg_per_query is not None:
            pos = rows[rows["is_label"]]
            neg = rows[~rows["is_label"]].copy()
            neg["_r"] = neg[["biencoder_rank", "cdematch_rank"]].astype("Float64").min(axis=1)
            neg = (neg.sort_values(["query_id", "_r"])
                      .groupby("query_id", sort=False).head(neg_per_query)
                      .drop(columns="_r"))
            rows = pd.concat([pos, neg], ignore_index=True)

    rows["label"] = rows["is_label"].astype(float)
    rows["query_text"] = rows[query_col].fillna("").astype(str)
    rows["candidate_text"] = rows["cde_id"].map(cde_text_map).fillna("")
    return rows[OUT_COLS].reset_index(drop=True)


def assert_integrity(train: pd.DataFrame, dev: pd.DataFrame):
    # Split integrity / no holdout-or-test leakage.
    assert set(train["split"].unique()) == {TRAIN_SPLIT}, \
        f"train has rows outside the configured train split {TRAIN_SPLIT!r}"
    assert set(dev["split"].unique()) == {DEV_SPLIT}, \
        f"dev has rows outside the configured dev split {DEV_SPLIT!r}"
    assert not (set(train["split"]) & FORBIDDEN_SPLITS), "forbidden split in train"
    assert not (set(dev["split"]) & FORBIDDEN_SPLITS), "forbidden split in dev"
    # Injected golds: train-only, always positive, never negative; never in dev.
    assert bool((train.loc[train["is_injected_gold"], "label"] == 1.0).all()), \
        "injected-gold train row is not a positive"
    assert int((train["is_injected_gold"] & (train["label"] == 0.0)).sum()) == 0, \
        "injected gold appears as a negative in train"
    assert int(dev["is_injected_gold"].sum()) == 0, "injected gold leaked into dev"
    # Negatives are never injected (in either split).
    assert int((train["label"] == 0.0).sum()) == int(((train["label"] == 0.0) & ~train["is_injected_gold"]).sum())
    assert int((dev["label"] == 0.0).sum()) == int(((dev["label"] == 0.0) & ~dev["is_injected_gold"]).sum())
    # Every TRAIN query has >=1 positive (dev intentionally may not).
    tpos = train.groupby("query_id")["label"].max()
    assert bool((tpos == 1.0).all()), "a train query has no positive"
    # Key/group integrity: unique (split, query_id, cde_id).
    for name, df in [("train", train), ("dev", dev)]:
        dup = df.duplicated(["split", "query_id", "cde_id"]).sum()
        assert dup == 0, f"{name} has {dup} duplicate (split,query_id,cde_id) rows"


def _summ(df, name, cde_text_map):
    pos = df[df["label"] == 1.0]
    return {
        f"{name}_rows": int(len(df)),
        f"{name}_queries": int(df["query_id"].nunique()),
        f"{name}_positives": int(len(pos)),
        f"{name}_injected_positives": int((pos["is_injected_gold"]).sum()),
        f"{name}_negatives": int((df["label"] == 0.0).sum()),
        f"{name}_queries_without_positive": int((df.groupby("query_id")["label"].max() == 0.0).sum()),
        f"{name}_positives_missing_cand_text": int(((pos["candidate_text"] == "")).sum()),
    }


def main(argv=None):
    global TRAIN_SPLIT, DEV_SPLIT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-table", default=str(sc.DEFAULT_FT))
    ap.add_argument("--cde-master", default=str(sc.DEFAULT_CDE_MASTER))
    ap.add_argument("--cde-text-recipe", default="SN_DEC_DEF_PQT_PV", choices=sorted(sc.RECIPES.keys()))
    ap.add_argument("--query-text-col", default=sc.DEFAULT_QUERY_COL)
    ap.add_argument("--neg-per-query", type=int, default=None,
                    help="cap negatives per TRAIN query (by best bienc/CM rank); default = all")
    ap.add_argument("--train-split", default=TRAIN_SPLIT,
                    help="split used for CE TRAIN pairs (default: full canonical 'train')")
    ap.add_argument("--dev-split", default=DEV_SPLIT,
                    help="split used for CE DEV/selection pairs (default: 'val_dev')")
    ap.add_argument("--out-dir", default=str(OUT_DATA_DIR))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    TRAIN_SPLIT, DEV_SPLIT = args.train_split, args.dev_split
    assert TRAIN_SPLIT not in FORBIDDEN_SPLITS and DEV_SPLIT not in FORBIDDEN_SPLITS, \
        f"train/dev split may not be a held-out eval split: {FORBIDDEN_SPLITS}"

    sc._set_safe_tempdir()
    recipe = args.cde_text_recipe
    out_dir = Path(args.out_dir)
    train_out = out_dir / f"train_{recipe}.parquet"
    dev_out = out_dir / f"dev_{recipe}.parquet"
    summary_out = out_dir / f"pairs_summary_{recipe}.json"
    if (train_out.exists() or dev_out.exists()) and not args.overwrite:
        sys.exit(f"ERROR: outputs exist (use --overwrite):\n  {train_out}\n  {dev_out}")

    ft = pd.read_parquet(args.feature_table, columns=list(dict.fromkeys(
        sc.SCORE_KEYS + ["is_label", "is_injected_gold", "biencoder_rank",
                         "cdematch_rank", args.query_text_col])))
    cde_text_map = sc.build_cde_text_map(Path(args.cde_master), recipe)

    present = set(ft["split"].unique())
    for role, sp in (("train", TRAIN_SPLIT), ("dev", DEV_SPLIT)):
        if sp not in present:
            sys.exit(f"ERROR: --{role}-split {sp!r} absent from feature table "
                     f"{args.feature_table} (present splits: {sorted(present)}). "
                     f"For full-train CE, pass a candidate pool that includes a "
                     f"{sp!r} split (e.g. the crossencoder_fulltrain pool).")

    train = build_split(ft, TRAIN_SPLIT, args.query_text_col, cde_text_map, False, args.neg_per_query)
    dev = build_split(ft, DEV_SPLIT, args.query_text_col, cde_text_map, True, None)

    assert_integrity(train, dev)

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(train_out, index=False)
    dev.to_parquet(dev_out, index=False)

    summary = {"recipe": recipe, "query_text_col": args.query_text_col,
               "neg_per_query": args.neg_per_query,
               "train_path": str(train_out), "dev_path": str(dev_out)}
    summary.update(_summ(train, "train", cde_text_map))
    summary.update(_summ(dev, "dev", cde_text_map))
    summary_out.write_text(json.dumps(summary, indent=2))

    print(f"recipe={recipe}  neg_per_query={args.neg_per_query}")
    print(f"TRAIN: {summary['train_rows']} rows  {summary['train_queries']} queries  "
          f"pos={summary['train_positives']} (injected={summary['train_injected_positives']})  "
          f"neg={summary['train_negatives']}")
    print(f"DEV:   {summary['dev_rows']} rows  {summary['dev_queries']} queries  "
          f"pos={summary['dev_positives']}  neg={summary['dev_negatives']}  "
          f"queries_without_positive={summary['dev_queries_without_positive']}")
    print(f"positives missing candidate text: train={summary['train_positives_missing_cand_text']} "
          f"dev={summary['dev_positives_missing_cand_text']}")
    print(f"\nWrote:\n  {train_out}\n  {dev_out}\n  {summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
