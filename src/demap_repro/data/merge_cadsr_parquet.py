#!/usr/bin/env python3
"""merge_and_summarize_cadsr_parquet.py

Phase 3-4 merger + summarizer for caDSR extraction outputs.

This script expects you ran extract_cadsr_xml_to_parquet.py for each XML file and
produced:
  <out_root>/<file_stem>/*.parquet

It will:
  Phase 3: Merge per-file Parquets into consolidated tables (one per table)
    under: <merged_dir>/<table>.parquet

  Phase 4: Produce inventory/summarization outputs for deciding:
    - which ReferenceDocument DocumentTypes/Names are "source-like"
    - which AlternateNameType/ContextName families are big enough for held-out tests

It focuses summaries on:
  - cde_reference_documents.parquet
  - cde_alternate_names.parquet

Outputs:
  - merged/*.parquet (consolidated)
  - summaries/*.csv and summaries/*.jsonl
  - summaries/merge_report.json

Usage:
  python merge_and_summarize_cadsr_parquet.py \
    --out-root out \
    --merged-dir merged \
    --summaries-dir summaries \
    --sample-size 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except ImportError:  # pragma: no cover
    # Keep import-time failure non-fatal so the rest of the package/tests can
    # still import. We raise a clear error at runtime when parquet I/O is used.
    pa = None  # type: ignore
    pq = None  # type: ignore


TABLES_DEFAULT = [
    'cde_master',
    'cde_iso11179_dec',
    'cde_iso11179_value_domain',
    'cde_permissible_values',
    'cde_classifications',
    'cde_reference_documents',
    'cde_alternate_names',
]


def _list_per_file_table_paths(out_root: Path, table: str) -> List[Path]:
    paths: List[Path] = []
    for sub in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        cand = sub / f'{table}.parquet'
        if cand.exists():
            paths.append(cand)
    return paths


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _merge_parquets(input_paths: List[Path], merged_path: Path) -> Dict[str, object]:
    """Stream-merge a list of Parquet files into one Parquet file."""
    if not input_paths:
        return {"merged_rows": 0, "input_files": 0, "merged_path": str(merged_path)}

    _ensure_dir(merged_path.parent)

    # Use schema from first file.
    first_pf = pq.ParquetFile(str(input_paths[0]))
    schema: pa.Schema = first_pf.schema_arrow

    # Overwrite existing.
    if merged_path.exists():
        merged_path.unlink()

    writer = pq.ParquetWriter(str(merged_path), schema=schema, compression='snappy')
    total_rows = 0
    try:
        for p in input_paths:
            pf = pq.ParquetFile(str(p))
            total_rows += pf.metadata.num_rows
            for batch in pf.iter_batches(batch_size=65536):
                table = pa.Table.from_batches([batch], schema=schema)
                writer.write_table(table)
    finally:
        writer.close()

    return {
        "merged_rows": int(total_rows),
        "input_files": int(len(input_paths)),
        "merged_path": str(merged_path),
    }


def _cde_key(df: pd.DataFrame) -> pd.Series:
    return df['cde_publicid'].astype(str) + 'v' + df['cde_version'].astype(str)


def _group_counts_and_coverage(
    df: pd.DataFrame,
    group_cols: List[str],
    cde_key_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (item_counts, cde_coverage) DataFrames."""
    item_counts = (
        df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name='n_items')
        .sort_values('n_items', ascending=False)
    )
    cde_coverage = (
        df.groupby(group_cols, dropna=False)[cde_key_col]
        .nunique()
        .reset_index(name='n_cdes')
        .sort_values('n_cdes', ascending=False)
    )
    return item_counts, cde_coverage


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _write_jsonl(records: Iterable[dict], path: Path) -> None:
    _ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _samples_by_group(
    df: pd.DataFrame,
    group_col: str,
    text_col: str,
    sample_size: int,
    seed: int,
) -> List[dict]:
    out: List[dict] = []

    # Keep only non-empty text.
    tmp = df[[group_col, text_col]].copy()
    tmp[text_col] = tmp[text_col].fillna('').astype(str).str.strip()
    tmp = tmp[tmp[text_col] != '']

    for val, g in tmp.groupby(group_col, dropna=False):
        # Deduplicate texts before sampling.
        texts = g[text_col].drop_duplicates().tolist()
        n = min(sample_size, len(texts))
        if n == 0:
            continue
        # Deterministic sampling: use hash of group value to vary seed per group.
        local_seed = (seed + (hash(str(val)) & 0xFFFFFFFF)) % 2**32
        sampled = pd.Series(texts).sample(n=n, random_state=local_seed).tolist()
        out.append({
            group_col: None if (isinstance(val, float) and pd.isna(val)) else val,
            "n_unique_texts": int(len(texts)),
            "n_samples": int(n),
            "samples": sampled,
        })

    # Sort by n_unique_texts desc.
    out.sort(key=lambda r: r['n_unique_texts'], reverse=True)
    return out


def _summarize_reference_documents(
    merged_refdoc_path: Path,
    summaries_dir: Path,
    sample_size: int,
    seed: int,
) -> None:
    if not merged_refdoc_path.exists():
        print(f"[WARN] Missing {merged_refdoc_path}; skipping refdoc summaries")
        return

    cols = ['cde_publicid', 'cde_version', 'document_type', 'name', 'document_text', 'source_file']
    tbl = pq.read_table(str(merged_refdoc_path), columns=[c for c in cols if c in pq.read_schema(str(merged_refdoc_path)).names])
    df = tbl.to_pandas()

    df['cde_key'] = _cde_key(df)

    # Counts
    item_counts, cde_cov = _group_counts_and_coverage(df, ['document_type'], 'cde_key')
    _write_csv(item_counts, summaries_dir / 'refdoc_document_type_item_counts.csv')
    _write_csv(cde_cov, summaries_dir / 'refdoc_document_type_cde_coverage.csv')

    item_counts, cde_cov = _group_counts_and_coverage(df, ['name'], 'cde_key')
    _write_csv(item_counts, summaries_dir / 'refdoc_name_item_counts.csv')
    _write_csv(cde_cov, summaries_dir / 'refdoc_name_cde_coverage.csv')

    item_counts, cde_cov = _group_counts_and_coverage(df, ['document_type', 'name'], 'cde_key')
    _write_csv(item_counts, summaries_dir / 'refdoc_document_type_name_item_counts.csv')
    _write_csv(cde_cov, summaries_dir / 'refdoc_document_type_name_cde_coverage.csv')

    # Samples
    samples_dt = _samples_by_group(df, 'document_type', 'document_text', sample_size=sample_size, seed=seed)
    _write_jsonl(samples_dt, summaries_dir / 'refdoc_document_type_samples.jsonl')

    # Optional: samples by Name (can be large); keep top N names by frequency.
    top_names = df['name'].value_counts(dropna=False).head(200).index.tolist()
    df_top = df[df['name'].isin(top_names)]
    samples_name = _samples_by_group(df_top, 'name', 'document_text', sample_size=sample_size, seed=seed)
    _write_jsonl(samples_name, summaries_dir / 'refdoc_name_samples_top200.jsonl')


def _summarize_alternate_names(
    merged_alt_path: Path,
    summaries_dir: Path,
    sample_size: int,
    seed: int,
) -> None:
    if not merged_alt_path.exists():
        print(f"[WARN] Missing {merged_alt_path}; skipping alternate-name summaries")
        return

    cols = ['cde_publicid', 'cde_version', 'alternate_name_type', 'context_name', 'alternate_name', 'source_file']
    tbl = pq.read_table(str(merged_alt_path), columns=[c for c in cols if c in pq.read_schema(str(merged_alt_path)).names])
    df = tbl.to_pandas()

    df['cde_key'] = _cde_key(df)

    # Counts
    item_counts, cde_cov = _group_counts_and_coverage(df, ['alternate_name_type'], 'cde_key')
    _write_csv(item_counts, summaries_dir / 'altname_type_item_counts.csv')
    _write_csv(cde_cov, summaries_dir / 'altname_type_cde_coverage.csv')

    item_counts, cde_cov = _group_counts_and_coverage(df, ['context_name'], 'cde_key')
    _write_csv(item_counts, summaries_dir / 'altname_context_item_counts.csv')
    _write_csv(cde_cov, summaries_dir / 'altname_context_cde_coverage.csv')

    item_counts, cde_cov = _group_counts_and_coverage(df, ['alternate_name_type', 'context_name'], 'cde_key')
    _write_csv(item_counts, summaries_dir / 'altname_type_context_item_counts.csv')
    _write_csv(cde_cov, summaries_dir / 'altname_type_context_cde_coverage.csv')

    # Samples
    samples_type = _samples_by_group(df, 'alternate_name_type', 'alternate_name', sample_size=sample_size, seed=seed)
    _write_jsonl(samples_type, summaries_dir / 'altname_type_samples.jsonl')

    top_ctx = df['context_name'].value_counts(dropna=False).head(200).index.tolist()
    df_top = df[df['context_name'].isin(top_ctx)]
    samples_ctx = _samples_by_group(df_top, 'context_name', 'alternate_name', sample_size=sample_size, seed=seed)
    _write_jsonl(samples_ctx, summaries_dir / 'altname_context_samples_top200.jsonl')


def _check_cde_master_uniqueness(merged_master_path: Path, summaries_dir: Path) -> Dict[str, object]:
    if not merged_master_path.exists():
        return {"master_present": False}

    cols = ['cde_publicid', 'cde_version', 'source_file', 'cde_xml_row_id']
    schema_names = pq.read_schema(str(merged_master_path)).names
    use_cols = [c for c in cols if c in schema_names]

    df = pq.read_table(str(merged_master_path), columns=use_cols).to_pandas()
    df['cde_key'] = _cde_key(df)

    dup_mask = df.duplicated(subset=['cde_key'], keep=False)
    n_dups = int(dup_mask.sum())
    n_unique = int(df['cde_key'].nunique())
    n_total = int(len(df))

    result = {
        "master_present": True,
        "n_rows": n_total,
        "n_unique_cde_keys": n_unique,
        "n_duplicate_rows": n_dups,
    }

    if n_dups:
        dups = df.loc[dup_mask].sort_values('cde_key')
        _write_csv(dups, summaries_dir / 'cde_master_duplicate_rows.csv')
        result["duplicates_csv"] = str(summaries_dir / 'cde_master_duplicate_rows.csv')

    return result


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True, help='Root directory containing per-file outputs')
    ap.add_argument('--merged-dir', default='merged', help='Where to write consolidated Parquet tables')
    ap.add_argument('--summaries-dir', default='summaries', help='Where to write summary CSV/JSONL outputs')
    ap.add_argument('--sample-size', type=int, default=25, help='Sample size per category (DocumentType, ContextName, etc.)')
    ap.add_argument('--seed', type=int, default=13, help='Random seed for sampling (deterministic)')
    ap.add_argument('--tables', nargs='*', default=TABLES_DEFAULT, help='Subset of tables to merge')

    args = ap.parse_args(argv)

    # Defer the dependency check until after argparse has a chance to handle
    # `--help`. This allows `demap merge-cadsr-parquet --help` to work even in
    # environments without pyarrow.
    if pa is None or pq is None:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'pyarrow'. This script reads/writes Parquet and requires it.\n"
            "Install with: pip install pyarrow\n"
        )

    out_root = Path(args.out_root)
    merged_dir = Path(args.merged_dir)
    summaries_dir = Path(args.summaries_dir)

    _ensure_dir(merged_dir)
    _ensure_dir(summaries_dir)

    merge_report: Dict[str, object] = {
        "out_root": str(out_root),
        "merged_dir": str(merged_dir),
        "tables": {},
    }

    # Phase 3: Merge
    for table in args.tables:
        input_paths = _list_per_file_table_paths(out_root, table)
        merged_path = merged_dir / f'{table}.parquet'
        info = _merge_parquets(input_paths, merged_path)
        merge_report['tables'][table] = info
        print(f"[merge] {table}: {info['merged_rows']} rows from {info['input_files']} files -> {merged_path}")

    # Phase 4: Summaries
    _summarize_reference_documents(merged_dir / 'cde_reference_documents.parquet', summaries_dir, sample_size=args.sample_size, seed=args.seed)
    _summarize_alternate_names(merged_dir / 'cde_alternate_names.parquet', summaries_dir, sample_size=args.sample_size, seed=args.seed)

    # Uniqueness checks
    master_check = _check_cde_master_uniqueness(merged_dir / 'cde_master.parquet', summaries_dir)
    merge_report['cde_master_check'] = master_check

    # Save report
    report_path = summaries_dir / 'merge_report.json'
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(merge_report, f, indent=2, ensure_ascii=False)

    print(f"[done] Wrote merge report: {report_path}")


if __name__ == '__main__':
    main()
