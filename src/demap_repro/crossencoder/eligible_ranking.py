#!/usr/bin/env python3
"""Build a slim cross-encoder ranking parquet for eligible-universe evaluation.

The eligible-universe evaluator (``scripts/eval_eligible_universe.py``,
``--source crossencoder``) needs a per-candidate ranking with columns::

    query_id, cde_id, split, crossencoder_rank

This script derives that ranking from the confirmed standalone cross-encoder
scores (MedCPT, listwise, full-train) over the deployable candidate union, using
the SAME logic as ``scripts/eval_crossencoder_rerank.py``:

  * restrict to the *deployable* candidate union (``is_injected_gold=False``) from
    the HGBC feature table -- injected-gold rows are NEVER included;
  * restrict to queries actually present in the score file;
  * attach the cross-encoder score on the (winner_id, split, query_id, cde_id) key;
  * dense-rank candidates within each (split, query_id) by cross-encoder score
    descending (unscored candidates sort last, mirroring deployment metrics).

Report-only: writes a single parquet under --out; never overwrites unless --overwrite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

REPO_ROOT = data_root()

DEFAULT_SCORES = (REPO_ROOT / "artifacts_v3_cdisc/crossencoder/"
                  "crossenc_scores_SN_DEC_DEF_PQT_PV_ftfull_medcpt_listwise.parquet")
DEFAULT_FT = REPO_ROOT / "artifacts_v3_cdisc/hgbc_reranker/feature_table.parquet"
DEFAULT_OUT = (REPO_ROOT / "artifacts_v3_cdisc/crossencoder/"
               "crossenc_eligible_ranking_medcpt_listwise.parquet")

KEY = ["winner_id", "split", "query_id", "cde_id"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scores", default=str(DEFAULT_SCORES),
                    help="standalone cross-encoder score parquet (crossenc_score column)")
    ap.add_argument("--feature-table", default=str(DEFAULT_FT),
                    help="HGBC candidate-union feature table (provides is_injected_gold)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f"ERROR: {out} exists; pass --overwrite to replace.", file=sys.stderr)
        return 2

    scores_path = Path(args.scores)
    ft_path = Path(args.feature_table)
    for p in (scores_path, ft_path):
        if not p.exists():
            print(f"ERROR: missing input: {p}", file=sys.stderr)
            return 2

    sc = pd.read_parquet(scores_path, columns=KEY + ["crossenc_score"])
    ft = pd.read_parquet(ft_path, columns=KEY + ["is_injected_gold"])

    for c in KEY:
        sc[c] = sc[c].astype(str)
        ft[c] = ft[c].astype(str)

    n_inj = int(ft["is_injected_gold"].astype(bool).sum())
    dep = ft[~ft["is_injected_gold"].astype(bool)].copy()

    # restrict to queries present in the score file (deployment-faithful)
    scored_q = sc[["split", "query_id"]].drop_duplicates()
    dep = dep.merge(scored_q, on=["split", "query_id"], how="inner")

    # attach cross-encoder score on the row key (unscored deployable -> NaN)
    merged = dep.merge(sc, on=KEY, how="left")

    # dense rank within (split, query_id) by CE score desc; NaN (unscored) last
    merged = merged.sort_values(
        ["split", "query_id", "crossenc_score"],
        ascending=[True, True, False], na_position="last", kind="mergesort",
    )
    merged["crossencoder_rank"] = merged.groupby(["split", "query_id"]).cumcount() + 1

    rank = merged[["query_id", "cde_id", "split", "crossencoder_rank"]].reset_index(drop=True)

    # invariants
    assert "is_injected_gold" not in rank.columns
    n_unscored = int(merged["crossenc_score"].isna().sum())

    out.parent.mkdir(parents=True, exist_ok=True)
    rank.to_parquet(out, index=False)

    print(f"scores      : {scores_path}")
    print(f"feature_tbl : {ft_path}  (injected-gold rows excluded: {n_inj})")
    print(f"rows        : {len(rank)}  queries: {rank['query_id'].nunique()}  "
          f"splits: {sorted(rank['split'].unique())}")
    print(f"unscored deployable candidates (ranked last): {n_unscored}")
    print(f"wrote       : {out}  cols={rank.columns.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
