#!/usr/bin/env python3
"""Task J: HGBC evaluation metrics for one allowance rate.

Reads <run-dir>/hgbc_noprov/hgbc_scored_rankings.parquet and computes, per split,
Recall@1/5/10 and MRR@100 under the deployment ranking convention
(hgbc_score descending, exact ties broken by ascending CDE public identifier --
FINAL_SCORE_TIE_BREAK_POLICY_VERSION final_rank_v1_score_desc_then_public_id_asc).
Every query counts in the denominator; queries whose gold is absent from the pool
count as misses. A cross-check block recomputes Recall@k from the trainer's own
hgbc_rank (rank method='first') and, at 0.70, compares against the archived
manuscript hgbc_eval_by_split.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/vf/users/nextgen2/james/tasks/cde_project/demap")
EVAL_SPLITS = ["test", "cctg", "oid_alt", "cdash", "val_dev", "val_train",
               "gdc_combined", "cimac_v2"]
MANUSCRIPT_R5 = {"test": 0.9715, "cctg": 0.9088, "oid_alt": 0.8324, "cdash": 0.9198}


def _pub_sort_key(pub: pd.Series) -> pd.Series:
    as_num = pd.to_numeric(pub, errors="coerce")
    return as_num.fillna(np.inf)


def metrics_for_split(g: pd.DataFrame) -> dict:
    g = g.copy()
    g["_pubnum"] = _pub_sort_key(g["cde_publicid"].astype(str))
    g = g.sort_values(["query_id", "hgbc_score", "_pubnum", "cde_publicid"],
                      ascending=[True, False, True, True], kind="mergesort")
    g["_rank"] = g.groupby("query_id").cumcount() + 1
    gold = g[g["is_label"]]
    best = gold.groupby("query_id")["_rank"].min()
    qids = g["query_id"].unique()
    n = len(qids)
    out = {"n_queries": int(n),
           "n_gold_in_pool": int(best.index.nunique())}
    for k in (1, 5, 10):
        out[f"recall@{k}"] = float((best <= k).sum() / n)
    out["mrr@100"] = float((1.0 / best[best <= 100]).sum() / n)
    # trainer-convention cross-check (row-order rank method='first')
    bt = gold.groupby("query_id")["hgbc_rank"].min()
    out["crosscheck_trainer_rank"] = {f"recall@{k}": float((bt <= k).sum() / n)
                                      for k in (1, 5, 10)}
    out["crosscheck_trainer_rank"]["mrr@100"] = float((1.0 / bt[bt <= 100]).sum() / n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-subdir", default="hgbc_noprov")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    md = rd / args.model_subdir

    df = pd.read_parquet(md / "hgbc_scored_rankings.parquet")
    df = df[~df["is_injected_gold"].astype(bool)]
    res = {"allow_rate": args.rate, "model_dir": str(md),
           "ranking_convention": "hgbc_score desc, tie-break cde_publicid asc "
                                 "(final_rank_v1_score_desc_then_public_id_asc)",
           "splits": {}}
    for split in EVAL_SPLITS:
        g = df[df["split"] == split]
        if not len(g):
            continue
        res["splits"][split] = metrics_for_split(g)
        m = res["splits"][split]
        print(f"[hgbc a{args.rate} {split}] n={m['n_queries']} "
              f"R@1={m['recall@1']:.4f} R@5={m['recall@5']:.4f} "
              f"R@10={m['recall@10']:.4f} MRR@100={m['mrr@100']:.4f}")

    # trainer's own eval CSV, for the record
    ev = pd.read_csv(md / "hgbc_eval_by_split.csv")
    res["trainer_eval_by_split"] = ev.to_dict(orient="records")

    if args.rate in ("0.70", "0.7"):
        arch = pd.read_csv(REPO / "artifacts/final_reranker/hgbc_reranker_v2_eligible/"
                                  "with_ce_noprov/hgbc_eval_by_split.csv")
        comp = {}
        for _, row in arch.iterrows():
            s = row["split"]
            new = ev[ev["split"] == s]
            if not len(new):
                continue
            comp[s] = {c: {"archived": float(row[c]), "rerun": float(new.iloc[0][c]),
                           "delta": float(new.iloc[0][c] - row[c])}
                       for c in ("recall@1", "recall@5", "recall@10")
                       if c in ev.columns and c in row.index}
        res["repro_vs_archived_manuscript_model"] = comp
        res["repro_vs_manuscript_r5_targets"] = {
            s: {"target": MANUSCRIPT_R5[s],
                "rerun": float(ev[ev["split"] == s].iloc[0]["recall@5"]),
                "match_4dp": bool(abs(float(ev[ev["split"] == s].iloc[0]["recall@5"])
                                      - MANUSCRIPT_R5[s]) < 5e-5)}
            for s in MANUSCRIPT_R5 if len(ev[ev["split"] == s])}
        for s, v in res["repro_vs_manuscript_r5_targets"].items():
            print(f"[repro {s}] target R@5={v['target']} rerun={v['rerun']:.4f} "
                  f"match_4dp={v['match_4dp']}")

    out = rd / f"metrics_hgbc_a{args.rate}.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
