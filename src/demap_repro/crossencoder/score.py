#!/usr/bin/env python3
"""Score query x candidate-CDE pairs with a Hugging Face cross-encoder.

Scaffold for the MVP cross-encoder reranking experiment. Reads the existing
HGBC candidate-union feature table, builds (query_text, candidate_text) pairs
for the *deployable* union (injected-gold rows excluded), scores them with a
``sentence_transformers.CrossEncoder``, and writes a per-candidate score table.

The candidate-text side is configurable via ``--cde-text-recipe`` (see RECIPES).
The query side defaults to ``query_text_q3`` (the representation the bi-encoder
and HGBC already use).

This script does NOT modify or overwrite the feature table or any existing HGBC
artifact; it only writes under ``artifacts_v3_cdisc/crossencoder/``.

Offline-safe: defaults to HF offline mode (compute nodes have no internet). If
the model is not in the HF cache, the script stops and prints the exact
pre-cache command instead of attempting a download.

Usage:
    # No-model data-path check (build + preview pairs, no scoring):
    PYTHONPATH=src .venv/bin/python scripts/score_crossencoder.py \
        --cde-text-recipe LN --splits test --preview-pairs 5

    # Dry-run scoring on CPU, small whole-query budget:
    PYTHONPATH=src .venv/bin/python scripts/score_crossencoder.py \
        --cde-text-recipe LN --splits test --device cpu --limit 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from demap_repro.utils.paths import data_root

REPO_ROOT = data_root()

DEFAULT_FT = REPO_ROOT / "artifacts_v3_cdisc/hgbc_reranker/feature_table.parquet"
DEFAULT_CDE_MASTER = REPO_ROOT / "data/processed/cde_master_enriched.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "artifacts_v3_cdisc/crossencoder"
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_QUERY_COL = "query_text_q3"

# Candidate-text recipes: ordered cde_master_enriched columns concatenated with
# RECIPE_SEP, skipping empty fields. Extend here to add recipes.
RECIPES = {
    "LN": ["LONG_NAME"],
    "SN_DEC_DEF_PQT_PV": [
        "SHORT_NAME", "DEC_LONG_NAME", "DEFINITION",
        "PREFERRED_QUESTION_TEXT", "PV_SUMMARY",
    ],
}
RECIPE_SEP = " | "

SCORE_KEYS = ["winner_id", "split", "query_id", "cde_id"]


def _set_safe_tempdir() -> Path:
    """Biowulf-safe temp dir: /lscratch/$SLURM_JOB_ID if available, else an
    artifact-local (gitignored) tmp dir. Never shared /tmp."""
    job = os.environ.get("SLURM_JOB_ID")
    cand = Path(f"/lscratch/{job}") if job and Path(f"/lscratch/{job}").is_dir() else None
    if cand is None:
        cand = DEFAULT_OUTDIR / "tmp"
        cand.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(cand)
    os.environ["TMPDIR"] = str(cand)
    return cand


def build_cde_text_map(cde_master_path: Path, recipe: str) -> dict:
    cols = RECIPES[recipe]
    master = pd.read_parquet(cde_master_path, columns=["cde_id"] + cols)
    master = master.dropna(subset=["cde_id"]).drop_duplicates("cde_id", keep="first")

    def _mk(row) -> str:
        parts = []
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            s = str(v).strip()
            if s:
                parts.append(s)
        return RECIPE_SEP.join(parts)

    texts = master[cols].apply(_mk, axis=1)
    return dict(zip(master["cde_id"].astype(str), texts))


def select_deployable_rows(ft: pd.DataFrame, splits, limit) -> pd.DataFrame:
    """Deployable union rows (injected gold excluded) for the requested splits.
    When ``limit`` is set it caps the number of candidate rows but always keeps
    *whole queries* so per-query rankings stay coherent for the dry-run."""
    df = ft[ft["split"].isin(splits)].copy()
    df = df[~df["is_injected_gold"].astype(bool)].copy()
    df["query_id"] = df["query_id"].astype(str)
    df["cde_id"] = df["cde_id"].astype(str)
    df = df.sort_values(["split", "query_id", "cde_id"]).reset_index(drop=True)
    if limit:
        sizes = df.groupby(["split", "query_id"], sort=False).size()
        cum_start = sizes.cumsum().shift(fill_value=0)
        keep = sizes.index[cum_start < limit]
        df = df.set_index(["split", "query_id"])
        df = df.loc[df.index.isin(keep)].reset_index()
    return df


def build_pairs(df: pd.DataFrame, query_col: str, cde_text_map: dict):
    queries = df[query_col].fillna("").astype(str).tolist()
    cde_ids = df["cde_id"].astype(str).tolist()
    cand_texts, missing = [], 0
    for cid in cde_ids:
        t = cde_text_map.get(cid, "")
        if not t:
            missing += 1
        cand_texts.append(t)
    return list(zip(queries, cand_texts)), missing


def _precache_cmd(model: str) -> str:
    cache = os.environ.get("HF_HUB_CACHE", "<HF_HUB_CACHE>")
    return (
        f"HF_HUB_CACHE={cache} .venv/bin/python -c "
        f"\"from sentence_transformers import CrossEncoder; CrossEncoder('{model}')\"   "
        f"# run on a LOGIN node with internet"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 ...")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--splits", default="test", help="comma-separated split names")
    ap.add_argument("--limit", type=int, default=None,
                    help="approx max candidate rows; applied at whole-query granularity")
    ap.add_argument("--cde-text-recipe", default="LN", choices=sorted(RECIPES.keys()))
    ap.add_argument("--query-text-col", default=DEFAULT_QUERY_COL)
    ap.add_argument("--feature-table", default=str(DEFAULT_FT))
    ap.add_argument("--cde-master", default=str(DEFAULT_CDE_MASTER))
    ap.add_argument("--out", default=None,
                    help="output parquet (default: artifacts_v3_cdisc/crossencoder/"
                         "crossenc_scores_<recipe>.parquet)")
    ap.add_argument("--preview-pairs", type=int, default=0,
                    help="if >0, build and print N example pairs, then exit (no model load)")
    ap.add_argument("--require-candidate-text", action="store_true",
                    help="hard-error (instead of counting) if any candidate row has "
                         "empty assembled CDE text under --cde-master; an empty "
                         "candidate string means the pool contains a CDE outside "
                         "the scoring catalog (the paper-v13 eligibility defect)")
    ap.add_argument("--allow-download", action="store_true",
                    help="permit HF download (otherwise offline; only on a node with internet)")
    args = ap.parse_args(argv)

    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _set_safe_tempdir()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    recipe = args.cde_text_recipe
    ft_path = Path(args.feature_table)
    if not ft_path.exists():
        sys.exit(f"ERROR: feature table not found: {ft_path}")

    need_cols = list(dict.fromkeys(
        SCORE_KEYS + ["is_injected_gold", args.query_text_col]))
    # ``is_injected_gold`` is optional: deployable feature tables (e.g. fixed-K) omit it.
    # Read only the columns that exist, then default the flag to False (all deployable),
    # mirroring scripts/train_hgbc_reranker.py.
    import pyarrow.parquet as _pq
    avail = set(_pq.ParquetFile(ft_path).schema.names)
    ft = pd.read_parquet(ft_path, columns=[c for c in need_cols if c in avail])
    if "is_injected_gold" not in ft.columns:
        print("WARNING: no is_injected_gold column; all rows treated as deployable")
        ft["is_injected_gold"] = False
    if args.query_text_col not in ft.columns:
        sys.exit(f"ERROR: query text column '{args.query_text_col}' not in feature table")

    present = set(ft["split"].unique())
    unknown = [s for s in splits if s not in present]
    if unknown:
        print(f"WARNING: splits not present in feature table (skipped): {unknown}")
    splits = [s for s in splits if s in present]
    if not splits:
        sys.exit("ERROR: no requested splits present in feature table")

    df = select_deployable_rows(ft, splits, args.limit)
    if df.empty:
        sys.exit("ERROR: no deployable rows selected")

    cde_text_map = build_cde_text_map(Path(args.cde_master), recipe)
    pairs, n_missing = build_pairs(df, args.query_text_col, cde_text_map)

    n_rows = len(df)
    n_queries = df.groupby(["split", "query_id"]).ngroups
    print(f"recipe={recipe}  splits={splits}")
    print(f"deployable rows selected: {n_rows}  (queries: {n_queries})")
    print(f"candidate rows with empty/missing CDE text: {n_missing}")
    if args.require_candidate_text and n_missing:
        bad = df.loc[[not t for _, t in pairs], ["split", "query_id", "cde_id"]]
        print(bad.head(20).to_string())
        sys.exit(f"ERROR: --require-candidate-text: {n_missing} candidate rows have "
                 f"empty assembled CDE text (first 20 above); the candidate pool "
                 f"contains CDEs absent from --cde-master")
    if args.limit:
        print(f"(--limit={args.limit} applied at whole-query granularity)")

    if args.preview_pairs > 0:
        print(f"\n=== preview {min(args.preview_pairs, n_rows)} pairs (recipe={recipe}) ===")
        for i in range(min(args.preview_pairs, n_rows)):
            q, c = pairs[i]
            print(f"\n[{i}] split={df.iloc[i]['split']} query_id={df.iloc[i]['query_id'][:10]} "
                  f"cde_id={df.iloc[i]['cde_id']}")
            print(f"    QUERY: {q[:200]}")
            print(f"    CAND : {c[:200]}")
        print("\n(preview mode: no model loaded, nothing scored)")
        return 0

    # ---- cache check + model load (offline unless --allow-download) ----
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(args.model, device=args.device, max_length=512)
    except Exception as e:  # uncached-offline or load failure
        print("\nERROR: could not load the cross-encoder model "
              f"(offline={'no' if args.allow_download else 'yes'}).")
        print(f"  reason: {type(e).__name__}: {e}")
        print("\nThe model does not appear to be in the HF cache. Pre-cache it first:\n")
        print("  " + _precache_cmd(args.model))
        print("\nThen re-run this script. Not submitting any GPU job.")
        return 2

    print(f"\nscoring {n_rows} pairs with {args.model} on device={args.device} "
          f"(batch_size={args.batch_size})...")
    scores = model.predict(pairs, batch_size=args.batch_size,
                           convert_to_numpy=True, show_progress_bar=False)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    out = df[SCORE_KEYS].copy()
    out["crossenc_score"] = scores
    out["crossenc_model"] = args.model
    out["cde_text_recipe"] = recipe

    out_path = Path(args.out) if args.out else DEFAULT_OUTDIR / f"crossenc_scores_{recipe}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    summary = {
        "model": args.model, "recipe": recipe, "device": args.device,
        "splits": splits, "query_text_col": args.query_text_col,
        "n_rows_scored": int(n_rows), "n_queries": int(n_queries),
        "n_missing_cde_text": int(n_missing),
        "limit": args.limit, "out": str(out_path),
        "injected_gold_excluded": True,
    }
    sidecar = out_path.with_suffix(".summary.json")
    sidecar.write_text(json.dumps(summary, indent=2))

    print(f"\nWrote scores: {out_path}  ({n_rows} rows)")
    print(f"Wrote summary: {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
