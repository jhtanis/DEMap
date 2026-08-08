#!/usr/bin/env python3
"""Evaluate standalone cross-encoder reranking of the candidate union.

Loads a cross-encoder score table (from ``score_crossencoder.py``) and the HGBC
feature table, ranks each query's *deployable* candidate union by cross-encoder
score, and reports deployment-safe Recall@K / MRR@100 by split.

Deployment-safe metrics are the existing ones from ``train_hgbc_reranker.py``:
injected-gold rows are excluded from ranking and queries whose gold CDE is
absent from the deployable candidate set count as misses (in the denominator).

Evaluation is restricted to the set of queries actually present in the score
table (so partial dry-run score files still yield coherent per-query metrics).
For context, bi-encoder and CDE Match baselines are computed on the same query
subset.

Writes only under ``artifacts_v3_cdisc/crossencoder/``. Does not touch the
feature table or any existing HGBC artifact.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/eval_crossencoder_rerank.py \
        --scores artifacts_v3_cdisc/crossencoder/crossenc_scores_LN.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

REPO_ROOT = data_root()

from demap_repro.reranker import train as thr

DEFAULT_FT = REPO_ROOT / "artifacts_v3_cdisc/hgbc_reranker/feature_table.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "artifacts_v3_cdisc/crossencoder"

SCORE_KEYS = ["winner_id", "split", "query_id", "cde_id"]
MK = ["recall@1", "recall@5", "recall@10", "mrr@100"]


def restrict_and_merge(ft: pd.DataFrame, scores: pd.DataFrame, splits) -> pd.DataFrame:
    """Return feature-table rows for the scored queries (in ``splits``) with the
    cross-encoder score merged on the row key."""
    sc = scores[scores["split"].isin(splits)].copy()
    for c in SCORE_KEYS:
        sc[c] = sc[c].astype(str)
        ft[c] = ft[c].astype(str)
    scored_q = sc[["split", "query_id"]].drop_duplicates()
    ftsub = ft.merge(scored_q, on=["split", "query_id"], how="inner")
    merged = ftsub.merge(sc[SCORE_KEYS + ["crossenc_score"]], on=SCORE_KEYS, how="left")
    return merged


def evaluate(merged: pd.DataFrame, splits) -> pd.DataFrame:
    """Deployment-safe cross-encoder metrics per split."""
    rows = []
    for s in splits:
        g = merged[merged["split"] == s]
        if g.empty:
            continue
        m = thr._deployment_metrics(g, "crossenc_score")
        deployable = g[~g["is_injected_gold"].astype(bool)]
        rows.append({
            "split": s, "method": "crossencoder",
            **{k: m[k] for k in MK},
            "n_queries": m["n_queries"],
            "n_queries_with_gold_in_candidates": m["n_queries_with_gold_in_candidates"],
            "n_injected_excluded": int(g["is_injected_gold"].astype(bool).sum()),
            "n_deployable_rows": int(len(deployable)),
            "n_unscored_deployable": int(deployable["crossenc_score"].isna().sum()),
        })
    return pd.DataFrame(rows)


def baseline_rows(merged: pd.DataFrame, splits) -> pd.DataFrame:
    """Bi-encoder / CDE Match deployment metrics on the same query subset."""
    bl = thr.eval_baselines(merged[merged["split"].isin(splits)])
    if bl.empty:
        return bl
    keep = ["split", "method"] + MK + ["n_queries", "n_queries_with_gold_in_candidates"]
    return bl[[c for c in keep if c in bl.columns]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="cross-encoder score parquet")
    ap.add_argument("--feature-table", default=str(DEFAULT_FT))
    ap.add_argument("--splits", default=None,
                    help="comma-separated; default = all splits present in the score file")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--tag", default="", help="optional filename suffix")
    args = ap.parse_args(argv)

    scores_path = Path(args.scores)
    if not scores_path.exists():
        sys.exit(f"ERROR: scores not found: {scores_path}")
    scores = pd.read_parquet(scores_path)
    recipe = str(scores["cde_text_recipe"].iloc[0]) if "cde_text_recipe" in scores else "NA"
    model = str(scores["crossenc_model"].iloc[0]) if "crossenc_model" in scores else "NA"

    ft = pd.read_parquet(
        args.feature_table,
        columns=SCORE_KEYS + ["is_injected_gold", "is_label", "family", "query_source",
                              "in_biencoder_topk", "in_cdematch_topk",
                              "biencoder_score", "cdematch_score"],
    )

    splits = ([s.strip() for s in args.splits.split(",") if s.strip()]
              if args.splits else sorted(scores["split"].unique().tolist()))

    merged = restrict_and_merge(ft, scores, splits)
    ce_eval = evaluate(merged, splits)
    bl = baseline_rows(merged, splits)

    combined = pd.concat([ce_eval, bl], ignore_index=True) if not bl.empty else ce_eval

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{recipe}" + (f"_{args.tag}" if args.tag else "")
    eval_csv = out_dir / f"crossenc_eval_by_split{suffix}.csv"
    strat_csv = out_dir / f"crossenc_eval_by_split_stratum{suffix}.csv"
    summary_md = out_dir / f"crossenc_summary{suffix}.md"

    combined.to_csv(eval_csv, index=False)

    strata = thr.compute_stratum_metrics(
        merged, "crossenc_score", ["split", "family", "query_source"]
    )
    strata.to_csv(strat_csv, index=False)

    # ---- summary md ----
    lines = [
        "# Cross-encoder standalone reranking — eval summary",
        "",
        f"- Model: `{model}`  ·  CDE-text recipe: `{recipe}`",
        f"- Scores: `{scores_path.name}`  ·  splits: {splits}",
        "- Deployment-safe metrics: injected gold excluded from ranking; "
        "gold-absent queries counted as misses (in denominator).",
        "- Evaluation restricted to queries present in the score file "
        "(coherent for partial dry-run score sets).",
        "",
        "## Cross-encoder by split",
        "",
        "| split | n_queries (denom) | gold_in_cands | injected_excluded | R@1 | R@5 | R@10 | MRR@100 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in ce_eval.iterrows():
        lines.append(
            f"| {r['split']} | {int(r['n_queries'])} "
            f"| {int(r['n_queries_with_gold_in_candidates'])} "
            f"| {int(r['n_injected_excluded'])} "
            f"| {r['recall@1']:.4f} | {r['recall@5']:.4f} "
            f"| {r['recall@10']:.4f} | {r['mrr@100']:.4f} |"
        )
    if not bl.empty:
        lines += ["", "## Baselines on the same query subset (context)", "",
                  "| split | method | R@1 | R@5 | R@10 | MRR@100 |",
                  "| --- | --- | --- | --- | --- |"]
        for _, r in bl.iterrows():
            lines.append(
                f"| {r['split']} | {r['method']} | {r['recall@1']:.4f} "
                f"| {r['recall@5']:.4f} | {r['recall@10']:.4f} | {r['mrr@100']:.4f} |"
            )
    lines.append("")
    summary_md.write_text("\n".join(lines))

    # ---- console report ----
    print(f"model={model}  recipe={recipe}  splits={splits}")
    for _, r in ce_eval.iterrows():
        print(f"  {r['split']:>22s}: denom={int(r['n_queries'])} "
              f"gold_in_cands={int(r['n_queries_with_gold_in_candidates'])} "
              f"injected_excluded={int(r['n_injected_excluded'])} "
              f"unscored_deployable={int(r['n_unscored_deployable'])} | "
              f"R@1={r['recall@1']:.4f} R@5={r['recall@5']:.4f} "
              f"R@10={r['recall@10']:.4f} MRR={r['mrr@100']:.4f}")
    print(f"\nWrote: {eval_csv}\n       {strat_csv}\n       {summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
