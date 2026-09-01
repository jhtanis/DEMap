#!/usr/bin/env python3
"""import_gdc_testset.py

Import the externally curated GDC CDE-match CSV tables as demap-compatible split
parquets.

This script produces two split files under --splits-dir:

  - external_holdout_gdc_altnames.parquet
  - external_holdout_gdc_questiontext.parquet

The output schema matches the repo's query split contract (query_text_raw,
query_text, query_text_q3, query_text_q4, cde_id, etc.) so it can be evaluated
by baseline-grid and dataset diagnostics.

It also performs a lightweight overlap audit against train.parquet to detect
instance-level leakage (exact/near-exact duplicate query strings). If overlaps
are found and --drop-overlaps is enabled, overlapping rows are removed from the
new split(s) and recorded in the manifest.

Usage (example)
---------------
python scripts/import_gdc_testset.py \
  --alt-csv  "/path/to/S69 ... Alt Names ... .csv" \
  --qtext-csv "/path/to/S69 ... Preferred Quest Text ... .csv" \
  --splits-dir data/processed/splits \
  --cde-master-enriched data/processed/cde_master_enriched.parquet \
  --drop-overlaps

Notes
-----
- The provided CSVs do not include CDE version. We resolve version by looking up
  Preferred CDE ID (public_id) in cde_master_enriched and choosing the maximum
  version for that public_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from demap_repro.data.gdc import (
    ImportManifest,
    audit_query_overlap,
    build_gdc_split_df,
    load_master_for_version_resolution,
    resolve_publicid_to_version_map,
    write_import_manifest,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    # Be conservative about dtype inference; keep as strings where possible.
    return pd.read_csv(path)


def import_one(
    *,
    csv_path: Path,
    query_kind: str,
    cde_publicid_to_version: dict,
    splits_dir: Path,
    train_split_parquet: Path | None,
    out_name: str,
    pv_col: str,
    drop_overlaps: bool,
    overwrite: bool,
) -> ImportManifest:
    df_csv = _read_csv(csv_path)

    df_split, stats = build_gdc_split_df(
        df_csv,
        query_kind=query_kind,
        cde_publicid_to_version=cde_publicid_to_version,
        pv_col=pv_col,
    )


    # Fail fast if any gold CDE IDs cannot be resolved to a (public_id, version).
    # If this triggers, it usually means the master catalog and the annotation CSV
    # are using different ID namespaces.
    if int(stats.get("n_missing_cde_version", 0)) > 0:
        bad = df_split[df_split["cde_version"].fillna("").astype(str).str.strip() == ""]
        missing_ids = sorted(bad["cde_publicid"].astype(str).unique().tolist())
        raise ValueError(
            "Some Preferred CDE IDs could not be resolved to a version using cde_master_enriched. "
            f"Missing version for {len(missing_ids)} public_id(s). Sample: {missing_ids[:10]}"
        )

    # Overlap audit: compare query_text_raw against train split query_text_raw.
    n_overlap, overlap_sample = audit_query_overlap(df_split, train_split_parquet=train_split_parquet)
    dropped = False
    if n_overlap > 0 and drop_overlaps:
        # Remove rows whose normalized query appears in train.
        # audit_query_overlap returns count with multiplicity, so we recompute mask here.
        from demap_repro.data.gdc import _norm_overlap  # type: ignore

        if train_split_parquet is not None and train_split_parquet.exists():
            try:
                df_train = pd.read_parquet(train_split_parquet, columns=["query_text_raw"])
            except Exception:
                df_train = pd.read_parquet(train_split_parquet)
            if "query_text_raw" in df_train.columns:
                train_set = set(df_train["query_text_raw"].fillna("").astype(str).map(_norm_overlap).tolist())
                mask = df_split["query_text_raw"].fillna("").astype(str).map(_norm_overlap).map(lambda x: x in train_set)
                df_split = df_split.loc[~mask].copy()
                dropped = True

    out_path = splits_dir / out_name
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output split already exists: {out_path} (use --overwrite to replace)")

    df_split.to_parquet(out_path, index=False)

    notes = ""
    if overlap_sample:
        notes = "query_overlap_sample=" + json.dumps(overlap_sample[:10])

    return ImportManifest(
        source_csv=str(csv_path),
        query_kind=query_kind,
        batch_names=stats.get("batch_names", []),
        n_rows_csv=int(stats.get("n_rows_csv", len(df_csv))),
        n_unique_seq=int(stats.get("n_unique_seq", df_csv["Seq ID"].nunique())),
        n_output_rows=int(len(df_split)),
        n_unique_cde_publicid=int(stats.get("n_unique_cde_publicid", df_split["cde_publicid"].nunique())),
        n_missing_cde_version=int(stats.get("n_missing_cde_version", 0)),
        n_query_overlap_with_train=int(n_overlap),
        dropped_overlaps=bool(dropped),
        pv_col=str(pv_col),
        notes=notes,
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt-csv", type=str, required=True, help="CSV file for ALT-name batch.")
    ap.add_argument("--qtext-csv", type=str, required=True, help="CSV file for Question Text batch.")
    ap.add_argument("--splits-dir", type=str, default="data/processed/splits", help="Output splits directory.")
    ap.add_argument(
        "--cde-master-enriched",
        type=str,
        default="data/processed/cde_master_enriched.parquet",
        help="Path to cde_master_enriched.parquet (for version resolution).",
    )
    ap.add_argument("--pv-col", type=str, default="Perm Val", help="PV column name in CSV (default: Perm Val).")
    ap.add_argument(
        "--drop-overlaps",
        action="store_true",
        help="If set, remove queries whose normalized text overlaps train.parquet query_text_raw.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output parquets.")
    ap.add_argument(
        "--manifest-path",
        type=str,
        default="artifacts/summaries/gdc_import_manifest.json",
        help="Where to write an import manifest JSON.",
    )
    args = ap.parse_args(argv)

    splits_dir = Path(args.splits_dir)
    _ensure_dir(splits_dir)

    master_path = Path(args.cde_master_enriched)
    if not master_path.exists():
        raise SystemExit(
            f"cde_master_enriched not found: {master_path}. "
            "Provide --cde-master-enriched or build the dataset first."
        )
    master_df = load_master_for_version_resolution(master_path)
    pid_to_ver = resolve_publicid_to_version_map(master_df)

    train_p = splits_dir / "train.parquet"
    train_split_parquet = train_p if train_p.exists() else None

    manifests: List[ImportManifest] = []
    manifests.append(
        import_one(
            csv_path=Path(args.alt_csv),
            query_kind="alt_names",
            cde_publicid_to_version=pid_to_ver,
            splits_dir=splits_dir,
            train_split_parquet=train_split_parquet,
            out_name="external_holdout_gdc_altnames.parquet",
            pv_col=args.pv_col,
            drop_overlaps=bool(args.drop_overlaps),
            overwrite=bool(args.overwrite),
        )
    )
    manifests.append(
        import_one(
            csv_path=Path(args.qtext_csv),
            query_kind="question_text",
            cde_publicid_to_version=pid_to_ver,
            splits_dir=splits_dir,
            train_split_parquet=train_split_parquet,
            out_name="external_holdout_gdc_questiontext.parquet",
            pv_col=args.pv_col,
            drop_overlaps=bool(args.drop_overlaps),
            overwrite=bool(args.overwrite),
        )
    )

    manifest_path = Path(args.manifest_path)
    _ensure_dir(manifest_path.parent)
    write_import_manifest(manifest_path, manifests)

    # Print a short summary for convenience.
    for m in manifests:
        print(
            f"[{m.query_kind}] wrote {m.n_output_rows} rows; "
            f"missing_version={m.n_missing_cde_version}; "
            f"train_overlap={m.n_query_overlap_with_train} (dropped={m.dropped_overlaps})"
        )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
