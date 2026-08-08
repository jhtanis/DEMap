#!/usr/bin/env python3
"""
split_policy.py

Step 4 of the pipeline:
  - Read data/processed/pairs.parquet (string-string training examples + provenance)
  - Create external holdouts (org, standard, REF slice)
  - Create internal train/val/test split (80/10/10 by default)
    * split unit: query_id (prevents identical query_text leakage across splits)
    * stratified by: (query_source, family)
  - Warn + drop duplicate pair_id rows, write data/processed/splits/dedupe_report.csv
  - Write split manifests for reproducibility/audit

Outputs (default):
  data/processed/splits/train.parquet
  data/processed/splits/val.parquet
  data/processed/splits/test.parquet
  data/processed/splits/external_holdout_org.parquet
  data/processed/splits/external_holdout_standard.parquet
  data/processed/splits/external_holdout_refslice.parquet
  data/processed/splits/dedupe_report.csv
  data/processed/splits/manifest.json

Design notes:
  - External holdout rules are currently hard-coded (easy to move to a config file later).
  - This script does NOT alter texts; it only assigns split membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


# -----------------------------
# Holdout policy (current plan)
# -----------------------------
ORG_HOLDOUT_ALT_CONTEXTS: Set[str] = {"CCTG"}
ORG_HOLDOUT_REF_FAMILIES: Set[str] = {"CCTG"}

STANDARD_HOLDOUT_ALT_FAMILIES: Set[str] = {"CDISC"}

REFSLICE_HOLDOUT_REF_FAMILIES: Set[str] = {"CDASH"}


REQUIRED_COLUMNS: List[str] = [
    "pair_id",
    "query_id",
    "query_source",  # ALT / REF
    "family",
    "cde_id",
    "query_text",
    "query_text_raw",
]


@dataclass(frozen=True)
class SplitParams:
    seed: int = 1234
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    stratify_cols: Tuple[str, str] = ("query_source", "family")


def _sha1_int(s: str) -> int:
    """Stable SHA1 -> int for deterministic RNG seeding."""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    # Use first 16 hex chars (64 bits) to fit comfortably in Python int
    return int(h[:16], 16)


def _rng_for_stratum(base_seed: int, stratum_key: str) -> random.Random:
    return random.Random(_sha1_int(f"{base_seed}::{stratum_key}"))


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"pairs.parquet missing required columns: {missing}")


def _safe_str(x: object) -> str:
    return "" if x is None else str(x)


def _select_external_holdouts(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (internal_pool, holdout_org, holdout_standard, holdout_refslice)."""
    # Normalize key fields to strings for comparisons
    qsrc = df["query_source"].astype(str)
    fam = df["family"].astype(str)

    # Some columns exist only for ALT/REF; treat missing as empty
    context = df["context_name"].astype(str) if "context_name" in df.columns else pd.Series([""] * len(df), index=df.index)
    # document_type/name not needed for holdout criteria; included for reporting only

    mask_org = ((qsrc == "ALT") & (context.isin(ORG_HOLDOUT_ALT_CONTEXTS))) | (
        (qsrc == "REF") & (fam.isin(ORG_HOLDOUT_REF_FAMILIES))
    )
    mask_standard = (qsrc == "ALT") & (fam.isin(STANDARD_HOLDOUT_ALT_FAMILIES))
    mask_refslice = (qsrc == "REF") & (fam.isin(REFSLICE_HOLDOUT_REF_FAMILIES))

    # Precedence if overlaps occur (rare): org > standard > refslice
    holdout_org = df[mask_org].copy()
    holdout_standard = df[~mask_org & mask_standard].copy()
    holdout_refslice = df[~mask_org & ~mask_standard & mask_refslice].copy()

    internal_pool = df[~mask_org & ~mask_standard & ~mask_refslice].copy()
    return internal_pool, holdout_org, holdout_standard, holdout_refslice


def _assign_query_stratum(internal_pool: pd.DataFrame, params: SplitParams) -> pd.DataFrame:
    """
    Build a query-level table with one row per query_id and a chosen stratum key.

    Note: query_id is text-based (e.g., SOURCE::raw_text). The same query_id may
    appear across multiple families if identical query text appears in multiple provenance
    buckets. To support stratification, we assign each query_id to its *majority* family
    within the internal_pool. This preserves leakage control (query_id stays intact) while
    approximating the family distribution.
    """
    qcols = ["query_id", params.stratify_cols[0], params.stratify_cols[1]]
    qdf = internal_pool[qcols].copy()
    qdf[params.stratify_cols[0]] = qdf[params.stratify_cols[0]].astype(str)
    qdf[params.stratify_cols[1]] = qdf[params.stratify_cols[1]].astype(str)

    # Compute majority family per query_id
    # For each query_id, count occurrences per (source,family); pick the most frequent
    grp = qdf.groupby(["query_id", params.stratify_cols[0], params.stratify_cols[1]]).size().reset_index(name="n")
    # Sort so idxmax is deterministic: n desc, then source asc, family asc
    grp = grp.sort_values(["query_id", "n", params.stratify_cols[0], params.stratify_cols[1]], ascending=[True, False, True, True])
    majority = grp.groupby("query_id", as_index=False).first()
    majority = majority.rename(columns={params.stratify_cols[0]: "query_source", params.stratify_cols[1]: "family"})
    majority["stratum"] = majority["query_source"].astype(str) + "||" + majority["family"].astype(str)

    # Flag multi-family (query_id appears with >1 family in internal_pool)
    fam_counts = grp.groupby("query_id").size().reset_index(name="n_families_for_query_id")
    majority = majority.merge(fam_counts, on="query_id", how="left")
    majority["is_multi_family_query"] = majority["n_families_for_query_id"] > 1
    return majority[["query_id", "query_source", "family", "stratum", "is_multi_family_query", "n_families_for_query_id"]]


def _stratified_split_query_ids(
    query_table: pd.DataFrame,
    params: SplitParams,
) -> Dict[str, str]:
    """
    Returns mapping: query_id -> split_name in {"train","val","test"}.
    Stratifies by query_table["stratum"].

    Allocation per stratum:
      train_n = int(n * train_frac)
      val_n   = int(n * val_frac)
      test_n  = n - train_n - val_n
    """
    if abs((params.train_frac + params.val_frac + params.test_frac) - 1.0) > 1e-9:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    mapping: Dict[str, str] = {}
    for stratum in sorted(query_table["stratum"].unique()):
        qids = query_table.loc[query_table["stratum"] == stratum, "query_id"].astype(str).tolist()
        rng = _rng_for_stratum(params.seed, stratum)
        rng.shuffle(qids)

        n = len(qids)
        train_n = int(n * params.train_frac)
        val_n = int(n * params.val_frac)
        test_n = n - train_n - val_n

        train_ids = qids[:train_n]
        val_ids = qids[train_n : train_n + val_n]
        test_ids = qids[train_n + val_n :]

        for qid in train_ids:
            mapping[qid] = "train"
        for qid in val_ids:
            mapping[qid] = "val"
        for qid in test_ids:
            mapping[qid] = "test"

    # Sanity check: all query_ids assigned
    missing = set(query_table["query_id"].astype(str)) - set(mapping.keys())
    if missing:
        raise RuntimeError(f"Internal split failed to assign {len(missing)} query_id(s).")
    return mapping


def _counts_by(df: pd.DataFrame, cols: List[str]) -> List[Dict[str, object]]:
    """Return list of dict rows for group counts."""
    if df.empty:
        return []
    g = df.groupby(cols).size().reset_index(name="n_rows")
    # Add unique query_id and cde_id counts if columns exist
    if "query_id" in df.columns:
        q = df.groupby(cols)["query_id"].nunique().reset_index(name="n_query_ids")
        g = g.merge(q, on=cols, how="left")
    if "cde_id" in df.columns:
        c = df.groupby(cols)["cde_id"].nunique().reset_index(name="n_cde_ids")
        g = g.merge(c, on=cols, how="left")
    return g.sort_values("n_rows", ascending=False).to_dict(orient="records")


def _write_manifest(
    out_path: str,
    params: SplitParams,
    df_all: pd.DataFrame,
    internal_pool: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    holdouts: Dict[str, pd.DataFrame],
    dedupe_dropped: int,
    overlap_stats: Dict[str, int],
    multi_family_query_ids: int,
) -> None:
    manifest = {
        "params": asdict(params),
        "policy": {
            "org_holdout_alt_contexts": sorted(ORG_HOLDOUT_ALT_CONTEXTS),
            "org_holdout_ref_families": sorted(ORG_HOLDOUT_REF_FAMILIES),
            "standard_holdout_alt_families": sorted(STANDARD_HOLDOUT_ALT_FAMILIES),
            "refslice_holdout_ref_families": sorted(REFSLICE_HOLDOUT_REF_FAMILIES),
            "split_unit": "query_id",
            "stratify_by": list(params.stratify_cols),
            "duplicate_pair_id_policy": "warn+drop (writes dedupe_report.csv)",
        },
        "counts": {
            "rows_total_input": int(len(df_all) + dedupe_dropped),
            "rows_total_after_dedupe": int(len(df_all)),
            "rows_deduped_dropped": int(dedupe_dropped),
            "rows_internal_pool": int(len(internal_pool)),
            "unique_query_ids_internal_pool": int(internal_pool["query_id"].nunique()) if not internal_pool.empty else 0,
            "multi_family_query_ids_internal_pool": int(multi_family_query_ids),
        },
        "splits": {},
        "holdouts": {},
        "leakage_checks": overlap_stats,
    }

    def _split_summary(df: pd.DataFrame) -> Dict[str, object]:
        return {
            "rows": int(len(df)),
            "unique_query_ids": int(df["query_id"].nunique()) if not df.empty else 0,
            "unique_cde_ids": int(df["cde_id"].nunique()) if not df.empty else 0,
            "by_source_family": _counts_by(df, ["query_source", "family"]),
        }

    for name, sdf in splits.items():
        manifest["splits"][name] = _split_summary(sdf)

    for name, hdf in holdouts.items():
        manifest["holdouts"][name] = _split_summary(hdf)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _compute_overlap_stats(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    holdouts: Dict[str, pd.DataFrame],
) -> Dict[str, int]:
    """Compute query_id overlaps across splits/holdouts for transparency."""
    def qset(df: pd.DataFrame) -> Set[str]:
        return set(df["query_id"].astype(str).unique()) if not df.empty else set()

    train_q, val_q, test_q = qset(train), qset(val), qset(test)
    stats = {
        "train_val_query_id_overlap": len(train_q & val_q),
        "train_test_query_id_overlap": len(train_q & test_q),
        "val_test_query_id_overlap": len(val_q & test_q),
    }
    for hname, hdf in holdouts.items():
        hq = qset(hdf)
        stats[f"train_{hname}_query_id_overlap"] = len(train_q & hq)
        stats[f"val_{hname}_query_id_overlap"] = len(val_q & hq)
        stats[f"test_{hname}_query_id_overlap"] = len(test_q & hq)
    return stats


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Create internal splits and external holdouts from pairs.parquet.")
    ap.add_argument("--pairs-parquet", default=os.path.join("out", "pairs.parquet"), help="Input pairs parquet.")
    ap.add_argument("--out-dir", default=os.path.join("out", "splits"), help="Output directory for splits.")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed for splitting.")
    ap.add_argument("--train-frac", type=float, default=None, help="Train fraction (by query_id, within strata). If not set, computed as 1 - val_frac - test_frac.")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Val fraction (by query_id, within strata).")
    ap.add_argument("--test-frac", type=float, default=0.1, help="Test fraction (by query_id, within strata).")
    ap.add_argument(
        "--allow-empty-val-test",
        action="store_true",
        help="If set, allow val/test splits to be empty for very small datasets (mostly relevant for unit tests).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    train_frac = args.train_frac
    if train_frac is None:
        train_frac = 1.0 - float(args.val_frac) - float(args.test_frac)
    if train_frac < 0:
        raise ValueError("train_frac must be >= 0; check val_frac/test_frac")

    params = SplitParams(seed=args.seed, train_frac=float(train_frac), val_frac=float(args.val_frac), test_frac=float(args.test_frac))

    os.makedirs(args.out_dir, exist_ok=True)

    # Load pairs
    df = pd.read_parquet(args.pairs_parquet)
    _require_columns(df, REQUIRED_COLUMNS)

    # Ensure key columns are strings for safe comparisons
    for col in ["pair_id", "query_id", "query_source", "family", "cde_id"]:
        df[col] = df[col].astype(str)

    # Warn + drop duplicate pair_id
    dedupe_report_path = os.path.join(args.out_dir, "dedupe_report.csv")
    dup_mask = df.duplicated(subset=["pair_id"], keep="first")
    dedupe_dropped = int(dup_mask.sum())
    keep_cols = [c for c in ["pair_id", "query_id", "query_source", "family", "cde_id", "query_text_raw"] if c in df.columns]
    if dedupe_dropped > 0:
        dropped = df.loc[dup_mask].copy()
        dropped[keep_cols].to_csv(dedupe_report_path, index=False)
        print(f"WARNING: Dropped {dedupe_dropped} duplicate pair_id row(s). See: {dedupe_report_path}")
        df = df.loc[~dup_mask].copy()
    else:
        # Always write an empty dedupe report for consistency (header only).
        pd.DataFrame(columns=keep_cols).to_csv(dedupe_report_path, index=False)

    # External holdouts
    internal_pool, holdout_org, holdout_standard, holdout_refslice = _select_external_holdouts(df)

    # Internal query-level table for stratification
    query_table = _assign_query_stratum(internal_pool, params)
    multi_family_query_ids = int(query_table["is_multi_family_query"].sum())

    # Split internal pool by query_id (stratified)
    mapping = _stratified_split_query_ids(query_table, params)
    internal_pool["split"] = internal_pool["query_id"].map(mapping)

    train = internal_pool[internal_pool["split"] == "train"].drop(columns=["split"]).copy()
    val = internal_pool[internal_pool["split"] == "val"].drop(columns=["split"]).copy()
    test = internal_pool[internal_pool["split"] == "test"].drop(columns=["split"]).copy()

    # Optional guard: for real runs we typically expect non-empty train
    if train.empty:
        raise RuntimeError("Train split is empty. Check inputs or split fractions.")
    if (val.empty or test.empty) and not args.allow_empty_val_test:
        print("WARNING: val or test split is empty. This can happen for small datasets/strata.")

    # Write outputs
    def _w(name: str, sdf: pd.DataFrame) -> str:
        path = os.path.join(args.out_dir, f"{name}.parquet")
        sdf.to_parquet(path, index=False)
        return path

    out_paths = {
        "train": _w("train", train),
        "val": _w("val", val),
        "test": _w("test", test),
        "external_holdout_org": _w("external_holdout_org", holdout_org),
        "external_holdout_standard": _w("external_holdout_standard", holdout_standard),
        "external_holdout_refslice": _w("external_holdout_refslice", holdout_refslice),
    }

    # Leakage/overlap checks (query_id overlaps)
    holdouts = {
        "external_holdout_org": holdout_org,
        "external_holdout_standard": holdout_standard,
        "external_holdout_refslice": holdout_refslice,
    }
    overlap_stats = _compute_overlap_stats(train, val, test, holdouts)

    # Write manifest
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    _write_manifest(
        out_path=manifest_path,
        params=params,
        df_all=df,
        internal_pool=internal_pool,
        splits={"train": train, "val": val, "test": test},
        holdouts=holdouts,
        dedupe_dropped=dedupe_dropped,
        overlap_stats=overlap_stats,
        multi_family_query_ids=multi_family_query_ids,
    )

    print("Wrote splits to:", args.out_dir)
    print("Manifest:", manifest_path)
    for k, v in out_paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
