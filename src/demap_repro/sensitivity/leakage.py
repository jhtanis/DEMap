#!/usr/bin/env python3
"""Paper v13 Table S7 (leakage sensitivity) — regenerated from per-query rankings.

Methods (author decision, 2026-07-27 handoff §8):
  1. Final corrected HGBC reranker  (per-query rankings from chain E; pass --hgbc-rankings)
  2. Python approximation to NCI CDE Match (clone; existing per-query candidates)
  3. BM25 (existing per-query rankings)
FT-MPNet is intentionally dropped.

Original canonical sets = data/processed/eval_canonical (primary; unchanged).
Leakage-filtered sets   = data/processed/eval_canonical_v2 (sensitivity only).
Filtered values are computed by re-scoring the SAME per-query rankings against the
v2 query lists — never derived from aggregate scores. Public-ID matching,
reachable-gold rules and deterministic ordering identical to the primary eval
(validated: the clone harness reproduces every published v1 metric exactly).

Outputs: table_s7_v13.csv (+ printed markdown).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

#: Data and artifact tree. This was an absolute path into the research
#: repository, which made the module unusable anywhere else; see
#: ``demap_repro.utils.paths`` and ``DEMAP_DATA_ROOT``.
REPO = data_root()
DATASETS = ["test", "cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2"]
ALLOW = {"test": "0.70", "cctg": "0.70", "oid_alt": "0.70", "cdash": "0.70",
         "gdc_combined": "1.0", "cimac_v2": "1.0"}


def pub(c) -> str:
    return str(c).split("::")[0]


def gold_sets(ds: str, version: str) -> pd.Series:
    base = "eval_canonical" if version == "v1" else "eval_canonical_v2"
    g = pd.read_parquet(REPO / f"data/processed/{base}/{ds}.parquet",
                        columns=["query_id", "cde_id"])
    g["query_id"] = g["query_id"].astype(str)
    g["pub"] = g["cde_id"].map(pub)
    return g.groupby("query_id")["pub"].apply(set)


def metrics_from_ranking(rank_df: pd.DataFrame, golds: pd.Series,
                         rank_col: str, id_col: str) -> dict:
    d = rank_df.copy()
    d["query_id"] = d["query_id"].astype(str)
    d["pub"] = d[id_col].map(pub)
    d = d.sort_values(rank_col, kind="stable").drop_duplicates(["query_id", "pub"])
    best = {}
    for q, sub in d.groupby("query_id"):
        if q not in golds.index:
            continue
        hit = sub[sub["pub"].isin(golds.loc[q])]
        if len(hit):
            best[q] = float(hit[rank_col].min())
    n = len(golds)
    out = {"n_queries": n}
    for k in (1, 5, 10):
        out[f"recall@{k}"] = sum(1 for q in golds.index
                                 if q in best and best[q] <= k) / n
    out["mrr@100"] = sum(1.0 / best[q] for q in golds.index
                         if q in best and best[q] <= 100) / n
    return out


def clone_ranking(ds: str) -> pd.DataFrame:
    f = REPO / ("artifacts/final_reranker/cde_match_clone/"
                f"cde_match_clone_candidates_{ds}__allow{ALLOW[ds]}.parquet")
    return pd.read_parquet(f, columns=["query_id", "cde_id", "cdematch_rank"])


def bm25_ranking(ds: str) -> pd.DataFrame:
    f = REPO / f"artifacts/bm25_canonical_v1/rankings/bm25_rankings_{ds}.parquet"
    d = pd.read_parquet(f)
    return d.rename(columns={"pub": "cde_pub"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hgbc-scored-rankings", default=None,
                    help="hgbc_scored_rankings.parquet from the corrected Step H "
                         "(with split/query_id/cde_id/hgbc_rank); omit to build "
                         "the lexical rows only")
    ap.add_argument("--out", default=str(REPO / ".scratch/demap/paper_v13_scientific_audit/table_s7_v13.csv"))
    args = ap.parse_args()

    global HGBC
    if args.hgbc_scored_rankings:
        HGBC = pd.read_parquet(args.hgbc_scored_rankings,
                               columns=["split", "query_id", "cde_id",
                                        "hgbc_rank", "is_injected_gold"])
        HGBC = HGBC[~HGBC["is_injected_gold"].astype(bool)]
        args.out = str(Path(args.out).with_name("table_s7_v13_with_hgbc.csv"))

    rows = []
    for ds in DATASETS:
        for version in ("v1", "v2"):
            golds = gold_sets(ds, version)
            # Python approximation (clone)
            m = metrics_from_ranking(clone_ranking(ds), golds, "cdematch_rank", "cde_id")
            rows.append({"method": "python_cde_match_approx", "dataset": ds,
                         "eval_set": version, **m})
            # BM25
            b = bm25_ranking(ds)
            m = metrics_from_ranking(b, golds, "rank", "cde_pub")
            rows.append({"method": "bm25", "dataset": ds, "eval_set": version, **m})
            # corrected HGBC (optional until chain E lands)
            if args.hgbc_scored_rankings:
                d = HGBC[HGBC["split"] == ds]
                m = metrics_from_ranking(d, golds, "hgbc_rank", "cde_id")
                rows.append({"method": "hgbc_corrected", "dataset": ds,
                             "eval_set": version, **m})

    t = pd.DataFrame(rows)
    wide = t.pivot_table(index=["method", "dataset"], columns="eval_set",
                         values=["recall@5", "n_queries"])
    wide[("recall@5", "delta")] = wide[("recall@5", "v2")] - wide[("recall@5", "v1")]
    print(wide.round(4).to_string())
    t.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
