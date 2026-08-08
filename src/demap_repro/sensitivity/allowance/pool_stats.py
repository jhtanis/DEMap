#!/usr/bin/env python3
"""Task J: candidate-pool statistics for one allowance rate's fixed-K30 base table.

Per split: n_queries, ceiling recall (gold public id present anywhere in the pool
-- the upper bound on any reranker Recall@k, reported as ceiling Recall@5),
mean/median realized pool size, %% queries with < 30 distinct candidates,
n queries whose accepted gold CDE is absent from the pool.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.table,
                         columns=["split", "query_id", "cde_publicid", "is_label"])
    out = {"allow_rate": args.rate, "table": args.table, "splits": {}}
    for split, g in df.groupby("split"):
        per_q = g.groupby("query_id").agg(
            pool_size=("cde_publicid", "nunique"),
            gold_in_pool=("is_label", "any"))
        n = len(per_q)
        out["splits"][split] = {
            "n_queries": int(n),
            "ceiling_recall_gold_in_pool": float(per_q["gold_in_pool"].mean()),
            "n_gold_absent": int((~per_q["gold_in_pool"]).sum()),
            "pool_size_mean": float(per_q["pool_size"].mean()),
            "pool_size_median": float(per_q["pool_size"].median()),
            "pool_size_min": int(per_q["pool_size"].min()),
            "pool_size_max": int(per_q["pool_size"].max()),
            "pct_queries_lt30_candidates": float((per_q["pool_size"] < 30).mean() * 100.0),
        }
        s = out["splits"][split]
        print(f"[pool {args.rate} {split}] n={n} ceiling={s['ceiling_recall_gold_in_pool']:.4f} "
              f"absent={s['n_gold_absent']} mean={s['pool_size_mean']:.2f} "
              f"median={s['pool_size_median']:.1f} lt30={s['pct_queries_lt30_candidates']:.1f}%")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
