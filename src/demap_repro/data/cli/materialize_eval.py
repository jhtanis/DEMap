#!/usr/bin/env python3
"""Materialize the 6 canonical evaluation datasets (reachable-filtered) into a stable location.

Non-destructive: writes NEW files under data/processed/eval_canonical/ and dropped-query reports
under artifacts/manifests/eval_canonical/. Never overwrites or deletes existing splits.

Rule: for every canonical dataset, drop any query row whose gold CDE is not present in the
production CDE catalog (unreachable golds are removed, NOT counted as misses). GDC is the
concat of question-text + alt-names, deduplicated on pair_id, THEN reachability-filtered.

  PYTHONPATH=src .venv/bin/python scripts/materialize_canonical_eval.py [--dry-run]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

SPLITS = Path('data/processed/splits')
PROD = Path('data/processed/cadsr_xml_2026-06-18/cde_master_enriched_eval_production_cde_match.parquet')
OUT_DATA = Path('data/processed/eval_canonical')
OUT_MANIFEST = Path('artifacts/manifests/eval_canonical')


def _source_frames():
    org = pd.read_parquet(SPLITS / 'external_holdout_org.parquet')
    q = pd.read_parquet(SPLITS / 'external_holdout_gdc_questiontext.parquet')
    a = pd.read_parquet(SPLITS / 'external_holdout_gdc_altnames.parquet')
    gdc = pd.concat([q, a], ignore_index=True).drop_duplicates(subset=['pair_id'], keep='first')
    return {
        'test': (pd.read_parquet(SPLITS / 'test.parquet'), ['data/processed/splits/test.parquet'], 'none'),
        'cctg': (org[org['family'] == 'CCTG'].copy(), ['data/processed/splits/external_holdout_org.parquet'], "family == 'CCTG'"),
        'oid_alt': (org[org['family'] == 'OID'].copy(), ['data/processed/splits/external_holdout_org.parquet'], "family == 'OID'"),
        'cdash': (pd.read_parquet(SPLITS / 'external_holdout_refslice.parquet'), ['data/processed/splits/external_holdout_refslice.parquet'], 'none'),
        'gdc_combined': (gdc, ['data/processed/splits/external_holdout_gdc_questiontext.parquet',
                               'data/processed/splits/external_holdout_gdc_altnames.parquet'],
                         'concat + drop_duplicates(subset=[pair_id])'),
        'cimac_v2': (pd.read_parquet(SPLITS / 'cimac_v2_reachable.parquet'), ['data/processed/splits/cimac_v2_reachable.parquet'], 'none'),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)

    prod_ids = set(pd.read_parquet(PROD)['cde_id'].astype(str))
    frames = _source_frames()
    summary = []
    for name, (df, source, rule) in frames.items():
        assert 'query_text_q3' in df.columns, f'{name}: missing query_text_q3'
        assert 'cde_id' in df.columns, f'{name}: missing cde_id'
        orig = len(df)
        reach_mask = df['cde_id'].astype(str).isin(prod_ids)
        kept = df[reach_mask].reset_index(drop=True)
        dropped = df[~reach_mask].reset_index(drop=True)
        rec = dict(dataset=name, original_rows=orig, reachable_rows=len(kept),
                   dropped_rows=len(dropped), query_text_q3=True, source=source, rule=rule)
        summary.append(rec)
        print(f'{name:14s} original={orig:5d}  reachable={len(kept):5d}  dropped={len(dropped):4d}')
        if args.dry_run:
            continue
        OUT_DATA.mkdir(parents=True, exist_ok=True)
        OUT_MANIFEST.mkdir(parents=True, exist_ok=True)
        dst = OUT_DATA / f'{name}.parquet'
        assert not dst.exists(), f'refusing to overwrite {dst}'
        kept.to_parquet(dst, index=False)
        # dropped report (only when rows removed)
        if len(dropped):
            qid = 'query_id' if 'query_id' in dropped.columns else ('pair_id' if 'pair_id' in dropped.columns else dropped.columns[0])
            rep = dropped[[c for c in [qid, 'query_source', 'family', 'cde_id', 'cde_publicid'] if c in dropped.columns]].copy()
            rep.insert(0, 'dataset', name)
            rep['reason'] = 'gold_not_in_production_catalog'
            rep.to_csv(OUT_MANIFEST / f'{name}_dropped_queries.csv', index=False)

    if not args.dry_run:
        OUT_MANIFEST.mkdir(parents=True, exist_ok=True)
        (OUT_MANIFEST / 'reachability_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        pd.DataFrame([{k: v for k, v in r.items() if k not in ('source',)} for r in summary]).to_csv(
            OUT_MANIFEST / 'reachability_summary.csv', index=False)
    print('\nDRY-RUN (nothing written).' if args.dry_run else f'\nWrote 6 parquets -> {OUT_DATA}/  + reports -> {OUT_MANIFEST}/')


if __name__ == '__main__':
    main()
