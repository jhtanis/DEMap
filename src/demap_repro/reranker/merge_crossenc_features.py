#!/usr/bin/env python3
"""Merge cross-encoder scores into a DERIVED HGBC feature table.

Builds ``feature_table + crossenc_*`` for the CE-as-HGBC-feature baseline
ablation. The base ``feature_table.parquet`` is opened READ-ONLY and is never
modified; all CE-derived columns are written to a NEW parquet at ``--out``.

CE scores are left-joined on ``["split", "query_id", "cde_id"]`` only (NOT on
``winner_id`` — the CE run's winner_id is a scratch tag and must be ignored).
The CE string columns (``winner_id``, ``crossenc_model``, ``cde_text_recipe``)
are dropped before the join so they never become HGBC features.

Per-query CE features (computed within each ``(split, query_id)`` group over the
rows that have a CE score):
  - crossenc_score             raw CE score (NaN for injected-gold rows)
  - crossenc_rank              1-based rank by DESCENDING score (1 = best)
  - crossenc_margin_to_top1    score - max(score in query)   (0 for the top row)
  - crossenc_margin_to_next    score(rank r) - score(rank r+1); 0 at the last rank
  - crossenc_score_z_by_query  (score - mean)/std within query; 0 if std==0 or n<2
  - crossenc_missing           1 if the row had no CE score (injected gold), else 0

Injected-gold rows carry NaN for all score-derived features and crossenc_missing=1.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_hgbc_feature_table_with_crossenc.py \
        --feature-table artifacts_v3_cdisc/hgbc_reranker/feature_table.parquet \
        --crossenc-scores artifacts_v3_cdisc/crossencoder/crossenc_scores_SN_DEC_DEF_PQT_PV_ftminilm_valtrain.parquet \
        --out artifacts_v3_cdisc/hgbc_reranker/feature_table_crossenc_ftminilm_valtrain.parquet
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

# Reuse the trainer's exact forbidden-name set so we can assert no collision.
from demap_repro.reranker.train import FORBIDDEN_FEATURE_COLS

KEY = ["split", "query_id", "cde_id"]
QKEY = ["split", "query_id"]
CE_DROP_COLS = ["winner_id", "crossenc_model", "cde_text_recipe"]
CE_FEATURES = [
    "crossenc_score", "crossenc_rank", "crossenc_margin_to_top1",
    "crossenc_margin_to_next", "crossenc_score_z_by_query", "crossenc_missing",
]


def _set_safe_tempdir(output_dir: Path | None = None) -> Path:
    """Biowulf-safe temp: per-job lscratch, else repo-local .scratch/. Never /tmp."""
    job = os.environ.get("SLURM_JOB_ID")
    lscratch = Path(f"/lscratch/{job}") if job and Path(f"/lscratch/{job}").is_dir() else None
    cand = lscratch if lscratch is not None else ((output_dir or REPO_ROOT) / "tmp")
    cand.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(cand)
    os.environ["TMPDIR"] = str(cand)
    return cand


def main(argv=None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-table")
    ap.add_argument("--crossenc-scores")
    ap.add_argument("--out")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--paper-config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv_list)

    if args.paper_config:
        from demap_repro.config.paper import (
            PaperConfigError, reject_conflicting_flags, resolve_paper_artifact_path,
            stage_job,
        )
        try:
            reject_conflicting_flags(
                argv_list, ["--feature-table", "--crossenc-scores", "--out"],
                context="merge-ce-features --paper-config")
            job = stage_job("merge_ft_medcpt_features", args.paper_config)
        except PaperConfigError as exc:
            ap.error(str(exc))
        args.feature_table = str(resolve_paper_artifact_path(job["feature_table"]))
        args.crossenc_scores = str(resolve_paper_artifact_path(job["crossenc_scores"]))
        args.out = str(resolve_paper_artifact_path(job["out"]))
        if args.dry_run:
            print(json.dumps({**job, "feature_table": args.feature_table,
                              "crossenc_scores": args.crossenc_scores,
                              "out": args.out}, indent=2, sort_keys=True))
            return 0
    else:
        missing = [flag for flag, value in (
            ("--feature-table", args.feature_table),
            ("--crossenc-scores", args.crossenc_scores), ("--out", args.out),
        ) if not value]
        if missing:
            ap.error("required outside paper mode: " + ", ".join(missing))

    ft_path = Path(args.feature_table)
    ce_path = Path(args.crossenc_scores)
    out_path = Path(args.out)
    tmp = _set_safe_tempdir(out_path.parent)
    print(f"TMPDIR={tmp}")
    if not ft_path.exists():
        sys.exit(f"ERROR: feature table not found: {ft_path}")
    if not ce_path.exists():
        sys.exit(f"ERROR: CE scores not found: {ce_path}")
    if out_path.exists() and not args.overwrite:
        sys.exit(f"ERROR: output exists (use --overwrite): {out_path}")
    if out_path.resolve() == ft_path.resolve():
        sys.exit("ERROR: --out must differ from --feature-table (never overwrite the base).")

    # ---- load (base read-only) ----
    base = pd.read_parquet(ft_path)
    ce = pd.read_parquet(ce_path)
    print(f"base rows: {len(base)}  | CE rows: {len(ce)}")

    # ---- normalize key dtypes for a safe join ----
    for df in (base, ce):
        for k in KEY:
            df[k] = df[k].astype(str)

    # ---- assert: both inputs unique on the join key ----
    assert not base.duplicated(KEY).any(), "base feature table not unique on (split,query_id,cde_id)"
    assert not ce.duplicated(KEY).any(), "CE scores not unique on (split,query_id,cde_id)"

    # ---- drop CE string columns from the CE side; keep key + crossenc_score ----
    ce_keep = ce[KEY + ["crossenc_score"]].copy()
    dropped = [c for c in CE_DROP_COLS if c in ce.columns]
    print(f"dropped from CE side: {dropped}")

    # ---- left-join on KEY only (preserve every base row) ----
    n_base = len(base)
    merged = base.merge(ce_keep, on=KEY, how="left")
    assert len(merged) == n_base, f"merge changed row count: {len(merged)} != {n_base}"

    # ---- assert: missing CE score <=> injected gold ----
    missing = merged["crossenc_score"].isna()
    injected = merged["is_injected_gold"].astype(bool)
    assert bool((missing == injected).all()), (
        "rows missing crossenc_score are not exactly the injected-gold rows "
        f"(missing={int(missing.sum())}, injected={int(injected.sum())}, "
        f"missing&~inj={int((missing & ~injected).sum())}, "
        f"inj&~missing={int((injected & ~missing).sum())})"
    )

    # ---- compute per-query CE features over the rows WITH a CE score ----
    dep = merged.loc[~missing, KEY + ["crossenc_score"]].copy()
    grp = dep.groupby(QKEY, sort=False)["crossenc_score"]

    crossenc_rank = grp.rank(ascending=False, method="first")
    crossenc_margin_to_top1 = dep["crossenc_score"] - grp.transform("max")

    mean = grp.transform("mean")
    std = grp.transform("std")  # ddof=1; NaN when n < 2
    z = (dep["crossenc_score"] - mean) / std
    crossenc_score_z = z.where(std.notna() & (std != 0.0), 0.0)

    # margin_to_next requires within-group ordering by descending score
    dep_sorted = dep.sort_values(QKEY + ["crossenc_score"], ascending=[True, True, False])
    next_score = dep_sorted.groupby(QKEY, sort=False)["crossenc_score"].shift(-1)
    margin_next = (dep_sorted["crossenc_score"] - next_score).fillna(0.0)
    crossenc_margin_to_next = margin_next.reindex(dep.index)

    # ---- write features back into merged (NaN stays for injected rows) ----
    for col in ["crossenc_rank", "crossenc_margin_to_top1",
                "crossenc_margin_to_next", "crossenc_score_z_by_query"]:
        merged[col] = np.nan
    merged.loc[dep.index, "crossenc_rank"] = crossenc_rank.astype(float).values
    merged.loc[dep.index, "crossenc_margin_to_top1"] = crossenc_margin_to_top1.values
    merged.loc[dep.index, "crossenc_margin_to_next"] = crossenc_margin_to_next.values
    merged.loc[dep.index, "crossenc_score_z_by_query"] = crossenc_score_z.values
    # crossenc_missing is defined for ALL rows (1 for injected/missing, else 0)
    merged["crossenc_missing"] = missing.astype("int64")

    # ---- assertions on the CE feature columns ----
    present = [c for c in merged.columns if c.startswith("crossenc_")]
    for c in CE_FEATURES:
        assert c in merged.columns, f"missing expected CE feature column: {c}"
    for c in present:
        assert pd.api.types.is_numeric_dtype(merged[c]), f"CE feature {c} is not numeric ({merged[c].dtype})"
    collide = sorted(set(present) & set(FORBIDDEN_FEATURE_COLS))
    assert not collide, f"crossenc_* columns collide with FORBIDDEN_FEATURE_COLS: {collide}"
    for c in CE_DROP_COLS:
        if c == "winner_id":
            continue  # base may legitimately keep its own winner_id
        assert c not in merged.columns, f"CE string column leaked into output: {c}"
    # the merge must not have produced suffixed duplicates
    assert not any(c.endswith(("_x", "_y")) for c in merged.columns), \
        "merge produced suffixed duplicate columns"

    # ---- per-split coverage report ----
    cov = (merged.assign(_has=(~missing).astype(int))
                 .groupby("split")
                 .agg(n_rows=("split", "size"),
                      n_with_ce=("_has", "sum"),
                      n_queries=("query_id", "nunique"))
                 .reset_index())
    cov["n_missing"] = cov["n_rows"] - cov["n_with_ce"]
    cov["coverage_pct"] = (100.0 * cov["n_with_ce"] / cov["n_rows"]).round(3)

    # ---- write derived table (NEW path) ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    # ---- report ----
    print("\n=== build_hgbc_feature_table_with_crossenc: OK ===")
    print(f"output: {out_path}")
    print(f"rows: {len(merged)}  (base rows: {n_base})")
    print(f"total columns: {merged.shape[1]}  (base: {base.shape[1]})")
    print(f"CE feature columns added: {present}")
    print(f"missing crossenc_score (== injected gold): {int(missing.sum())}")
    print("\nper-split coverage:")
    print(cov[["split", "n_rows", "n_queries", "n_with_ce", "n_missing", "coverage_pct"]]
          .to_string(index=False))
    print("\nassertions passed: uniqueness, row-count, missing==injected, "
          "numeric CE features, no FORBIDDEN collision, no CE string leak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
