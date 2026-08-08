#!/usr/bin/env python3
"""Build a REACHABLE-gold filtered copy of the evaluation split set.

Drops only queries whose gold CDE is unreachable in the June-18 ELIGIBLE catalogue
(any-gold convention: a query is kept iff AT LEAST ONE of its gold public ids is in
the eligible catalogue). Multi-gold queries with >=1 reachable gold are kept intact
(all their rows, including any inert archived-gold rows). All columns / metadata are
preserved -- only whole-query rows are removed. Source split files are NOT modified;
output goes to a new directory.

Reachability source = the eligible catalogue used to build the fixedk30 candidate pool:
  data/processed/cadsr_xml_2026-06-18/cde_master_enriched_eval_production_cde_match.parquet

Usage:
  PYTHONPATH=src .venv/bin/python scripts/build_reachable_eval_splits.py \
    --splits-dir .scratch/demap/keyword_attribution_2026-06-18/splits_symlinked_s2 \
    --eligible   data/processed/cadsr_xml_2026-06-18/cde_master_enriched_eval_production_cde_match.parquet \
    --out-dir    data/processed/splits_v3_cdisc_reachable_2026-06-18
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

SPLITS = ["test", "val_dev", "val_train", "external_holdout_org", "external_holdout_refslice",
          "cimac_v2", "theradex6_test", "external_holdout_gdc_questiontext",
          "external_holdout_gdc_altnames"]


def _pub(x):
    return str(x).split("::")[0].strip() if pd.notna(x) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--eligible", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    sd = Path(args.splits_dir); out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f"ERROR: {out} exists and is non-empty; pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    elig = pd.read_parquet(args.eligible, columns=["cde_publicid"])
    elig_ids = set(elig["cde_publicid"].astype(str).map(_pub))
    print(f"eligible public ids: {len(elig_ids)}")

    manifest = []
    dropped_rows = []
    for sp in SPLITS:
        df = pd.read_parquet(sd / f"{sp}.parquet")
        pidcol = "cde_publicid" if "cde_publicid" in df.columns else "cde_id"
        qid = df["query_id"].astype(str)
        # per-query any-gold reachability
        gold = {}
        for q, p in zip(qid, df[pidcol].map(_pub)):
            gold.setdefault(q, set())
            if p and p != "None":
                gold[q].add(p)
        reachable_q = {q for q, s in gold.items() if s & elig_ids}
        all_q = set(gold)
        drop_q = all_q - reachable_q
        keep_mask = qid.isin(reachable_q)
        kept = df[keep_mask].copy()
        kept.to_parquet(out / f"{sp}.parquet", index=False)
        manifest.append(dict(split=sp, orig_queries=len(all_q), filtered_queries=len(reachable_q),
                             dropped_queries=len(drop_q), orig_rows=len(df), filtered_rows=len(kept)))
        for q in sorted(drop_q):
            dropped_rows.append(dict(split=sp, query_id=q, gold_public_ids=";".join(sorted(gold[q]))))
        print(f"  {sp}: {len(all_q)} -> {len(reachable_q)} queries (dropped {len(drop_q)}); "
              f"rows {len(df)} -> {len(kept)}")

    mf = pd.DataFrame(manifest)
    mf.to_csv(out / "_filter_manifest.csv", index=False)
    if dropped_rows:
        pd.DataFrame(dropped_rows).to_csv(out / "_dropped_queries.csv", index=False)
    print(f"\nTotals: dropped {mf['dropped_queries'].sum()} queries across {len(SPLITS)} splits.")
    print(f"Wrote filtered splits + _filter_manifest.csv + _dropped_queries.csv to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
