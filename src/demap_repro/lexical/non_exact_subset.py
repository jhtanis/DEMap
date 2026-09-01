#!/usr/bin/env python3
"""Non-exact-subset evaluation: clone vs keyword_v1 fuzzy vs FT-MPNet.

Non-exact-under-policy (APPROVED, query-level): a query is NON-EXACT iff the Python
CDE-Match clone's POST-POLICY output (under the dataset-specific allowrate) contains
NO allowed exact-to-gold candidate (mtch_type=='exact' AND candidate publicid ∈ gold
publicids). This intentionally includes queries whose gold-exact was SUPPRESSED by the
allowrate=0.70 control (test/cctg/oid_alt/cdash) as well as queries with no exact match
at all and queries with no clone candidates.

Allowrate policy: gdc_combined/cimac_v2 -> 1.0 ; test/cctg/oid_alt/cdash -> 0.70.

Methods on the SAME non-exact query-id set per dataset (denominator = #non-exact qids):
  - clone       : clone post-policy candidates, ranked by cdematch_rank (depth<=10).
  - keyword_v1  : keyword_v1 fuzzy candidates (SAME exact control/mask as clone), ranked
                  by cdematch_rank. FAIRNESS: on the non-exact subset, any RESIDUAL
                  exact-to-gold rows are dropped (count reported); the fuzzy fallback is
                  kept (its intended, fair advantage on non-exact queries).
  - ft_mpnet    : phase2 all-MPNet deep rankings (top-1000), ranked by biencoder_rank;
                  no exact tier, evaluated as-is.
Gold match is PUBLICID-level, ANY-GOLD (multi-gold: hit if any gold pub in top-k).

Read-only; writes only under --out-root. No /tmp, no artifacts_v3_cdisc.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

#: Data and artifact tree. This was an absolute path into the research
#: repository, which made the module unusable anywhere else; see
#: ``demap_repro.utils.paths`` and ``DEMAP_DATA_ROOT``.
REPO = data_root()
from demap_repro.lexical.cde_match_interface import exact_match_control

SEED = 42
DATASETS = {  # dataset -> allowrate
    "test": 0.70, "cctg": 0.70, "oid_alt": 0.70, "cdash": 0.70,
    "gdc_combined": 1.0, "cimac_v2": 1.0,
}
CLONE_DIR = REPO / "artifacts/final_reranker/cde_match_clone"
KW_DIR = REPO / "artifacts/final_reranker/keyword_fuzzy_v2_eligible"
BIENC = REPO / "artifacts/final_reranker/biencoder_deep/eval_canonical/biencoder_deep_rankings_top1000.parquet"
EVAL_DIR = REPO / "data/processed/eval_canonical"


def _allow_tag(rate: float) -> str:
    return "allow1.0" if rate >= 1.0 else f"allow{rate:.2f}"


def _pub(s: pd.Series) -> pd.Series:
    return s.astype(str).str.split("::").str[0]


def build_gold_map(eval_path) -> dict:
    """query_id -> set of gold CDE public identifiers, from an evaluation parquet.

    Public-identifier level, any-gold: the paper never matches version-exact, and
    a query may carry more than one accepted gold CDE. Callers outside this module
    (BM25 evaluation) pass a path; :func:`_gold_map` is the by-name convenience.
    """
    e = pd.read_parquet(eval_path, columns=["query_id", "cde_id"])
    e["query_id"] = e["query_id"].astype(str)
    e["pub"] = _pub(e["cde_id"])
    gm: dict = {}
    for q, p in zip(e["query_id"], e["pub"]):
        gm.setdefault(q, set()).add(p)
    return gm


def _gold_map(ds: str) -> dict:
    """Gold map for a canonical dataset by name."""
    return build_gold_map(EVAL_DIR / f"{ds}.parquet")


def eval_method(cands: pd.DataFrame, subset_qids: set, gold_map: dict) -> dict:
    """cands: columns query_id(str), pub(str), rank(int, 1-based). Metrics over subset."""
    N = len(subset_qids)
    c = cands[cands["query_id"].isin(subset_qids)].copy()
    covered = set(c["query_id"].unique())
    best: dict = {}
    for q, g in c.groupby("query_id", sort=False):
        golds = gold_map.get(q, set())
        hit = g[g["pub"].isin(golds)]
        if len(hit):
            best[q] = int(hit["rank"].min())

    def recall(k: int) -> float:
        return sum(1 for q in subset_qids if best.get(q, 10**9) <= k) / N if N else 0.0

    mrr = (sum(1.0 / best[q] for q in subset_qids if q in best and best[q] <= 100) / N) if N else 0.0
    return {
        "recall@1": recall(1), "recall@5": recall(5), "recall@10": recall(10),
        "recall@100": recall(100), "mrr@100": mrr,
        "coverage": len(covered & subset_qids) / N if N else 0.0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(REPO / "artifacts/final_reranker/non_exact_subset_eval_v2_eligible"))
    args = ap.parse_args(argv)
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    subset_rows, summary_rows, metric_rows = [], [], []

    for ds, rate in DATASETS.items():
        tag = _allow_tag(rate)
        gold_map = _gold_map(ds)
        all_qids = set(gold_map.keys())

        # --- authoritative subset from CLONE post-policy output ---
        clone = pd.read_parquet(CLONE_DIR / f"cde_match_clone_candidates_{ds}__{tag}.parquet")
        clone["query_id"] = clone["query_id"].astype(str)
        clone["pub"] = _pub(clone["cde_id"])
        # exact-to-gold per query
        cl_ex = clone[clone["mtch_type"] == "exact"].copy()
        cl_ex["is_gold"] = [p in gold_map.get(q, set()) for q, p in zip(cl_ex["query_id"], cl_ex["pub"])]
        exact_qids = set(cl_ex.loc[cl_ex["is_gold"], "query_id"].unique())
        non_exact_qids = all_qids - exact_qids

        # hash "blocked region" among non-exact (informative; suppression happens here)
        ctrl = exact_match_control(allow_rate=rate, seed=SEED)
        n_blocked_region = sum(1 for q in non_exact_qids if not ctrl.query_allowed(q))

        summary_rows.append({
            "dataset": ds, "allowrate": rate,
            "n_queries_total": len(all_qids),
            "n_exact_under_policy": len(exact_qids),
            "n_non_exact": len(non_exact_qids),
            "pct_non_exact": round(100.0 * len(non_exact_qids) / len(all_qids), 2) if all_qids else 0.0,
            "n_non_exact_in_blocked_hash_region": n_blocked_region,
        })
        for q in sorted(all_qids):
            subset_rows.append({
                "dataset": ds, "query_id": q,
                "exact_under_policy": q in exact_qids,
                "had_clone_candidates": q in set(clone["query_id"].unique()),
                "n_gold": len(gold_map.get(q, set())),
            })

        # --- method: clone ---
        m = eval_method(clone.rename(columns={"cdematch_rank": "rank"})[["query_id", "pub", "rank"]],
                        non_exact_qids, gold_map)
        m.update({"dataset": ds, "method": "clone", "n_non_exact": len(non_exact_qids),
                  "residual_exact_removed": 0, "depth_cap": 10})
        metric_rows.append(m)

        # --- method: keyword_v1 fuzzy (defensive residual-exact drop on subset) ---
        kw = pd.read_parquet(KW_DIR / f"cde_match_clone_candidates_{ds}__{tag}.parquet")
        kw["query_id"] = kw["query_id"].astype(str)
        kw["pub"] = _pub(kw["cde_id"])
        in_subset = kw["query_id"].isin(non_exact_qids)
        is_gold = pd.Series([p in gold_map.get(q, set()) for q, p in zip(kw["query_id"], kw["pub"])],
                            index=kw.index)
        residual = in_subset & (kw["mtch_type"] == "exact") & is_gold
        n_residual = int(residual.sum())
        kw_fair = kw[~residual].copy()
        m = eval_method(kw_fair.rename(columns={"cdematch_rank": "rank"})[["query_id", "pub", "rank"]],
                        non_exact_qids, gold_map)
        m.update({"dataset": ds, "method": "keyword_v1_fuzzy", "n_non_exact": len(non_exact_qids),
                  "residual_exact_removed": n_residual, "depth_cap": 1000})
        metric_rows.append(m)

        # --- method: FT-MPNet (phase2 all-MPNet) deep rankings ---
        be = pd.read_parquet(BIENC)
        be = be[be["split"] == ds].copy()
        be["query_id"] = be["query_id"].astype(str)
        be["pub"] = _pub(be["cde_id"])
        m = eval_method(be.rename(columns={"biencoder_rank": "rank"})[["query_id", "pub", "rank"]],
                        non_exact_qids, gold_map)
        m.update({"dataset": ds, "method": "ft_mpnet", "n_non_exact": len(non_exact_qids),
                  "residual_exact_removed": 0, "depth_cap": 1000})
        metric_rows.append(m)

        print(f"[{ds}] allowrate={rate} total={len(all_qids)} exact={len(exact_qids)} "
              f"non_exact={len(non_exact_qids)} ({summary_rows[-1]['pct_non_exact']}%) "
              f"kw_residual_exact_removed={n_residual}")

    # --- write outputs ---
    sub_df = pd.DataFrame(subset_rows)
    sum_df = pd.DataFrame(summary_rows)
    met_df = pd.DataFrame(metric_rows)[
        ["dataset", "method", "n_non_exact", "coverage", "recall@1", "recall@5",
         "recall@10", "recall@100", "mrr@100", "residual_exact_removed", "depth_cap"]]
    sub_df.to_csv(out / "non_exact_query_ids_by_dataset.csv", index=False)
    sum_df.to_csv(out / "exact_policy_summary_by_dataset.csv", index=False)
    met_df.to_csv(out / "method_metrics_non_exact.csv", index=False)

    # markdown
    lines = ["# Non-exact-subset evaluation (clone / keyword_v1 fuzzy / FT-MPNet)", ""]
    lines += ["## Exact/non-exact policy summary", "",
              "| dataset | allowrate | n_total | n_exact | n_non_exact | %non_exact | n_in_blocked_hash_region |",
              "|---|---|---|---|---|---|---|"]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['allowrate']} | {r['n_queries_total']} | "
                     f"{r['n_exact_under_policy']} | {r['n_non_exact']} | {r['pct_non_exact']} | "
                     f"{r['n_non_exact_in_blocked_hash_region']} |")
    lines += ["", "## Metrics on the non-exact subset (denominator = n_non_exact)", ""]
    for ds in DATASETS:
        lines += [f"### {ds}", "",
                  "| method | n_non_exact | cov | R@1 | R@5 | R@10 | R@100 | MRR@100 | kw_resid_exact_removed |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in metric_rows if x["dataset"] == ds]:
            lines.append(f"| {r['method']} | {r['n_non_exact']} | {r['coverage']:.3f} | "
                         f"{r['recall@1']:.4f} | {r['recall@5']:.4f} | {r['recall@10']:.4f} | "
                         f"{r['recall@100']:.4f} | {r['mrr@100']:.4f} | {r['residual_exact_removed']} |")
        lines.append("")
    lines += ["## Caveats",
              "- Clone candidate depth is capped at match_limit=10, so clone **R@100 == R@10** (no candidates past rank 10).",
              "- keyword_v1 fuzzy (fuzzy_top_n=1000) and FT-MPNet (top-1000) support true @100.",
              "- Non-exact subset defined from the CLONE post-policy output; keyword_v1 fuzzy used the IDENTICAL "
              "exact control (allow_rate, seed=42, scope=gold) so residual exact-to-gold removals should be ~0.",
              "- `n_in_blocked_hash_region` = non-exact queries where hash01(seed,qid) >= allowrate (where the 0.70 "
              "control CAN suppress); it is an upper bound on suppressed-exact queries, not a proven count "
              "(distinguishing 'no exact ever' from 'exact suppressed' would need the allow1.0 clone output).",
              "- Gold matching is publicid-level, any-gold; denominator is unique non-exact query_ids per dataset."]
    (out / "method_metrics_non_exact.md").write_text("\n".join(lines) + "\n")

    print(f"\nWrote:\n  {out/'non_exact_query_ids_by_dataset.csv'}\n  {out/'exact_policy_summary_by_dataset.csv'}"
          f"\n  {out/'method_metrics_non_exact.csv'}\n  {out/'method_metrics_non_exact.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
