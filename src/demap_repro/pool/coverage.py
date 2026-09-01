#!/usr/bin/env python3
"""Table S5 — Stage-1 candidate-pool gold coverage, by retrieval arm.

Coverage is *does an accepted gold CDE appear anywhere in the candidate set*.
It is not a Recall@k measure and not reranker performance: the gold entering
the pool is a precondition for correct ranking, not a measure of it. No
reranker can recover a gold CDE that is not in the pool, so this bounds
everything downstream.

Computed from the frozen fixed-K feature table, which is the pool the final
reranker actually scored, using the source flags written by
``build_candidate_union_publicid``:

===================  =========================================================
FT-MPNet@20          ``in_biencoder_topk``
CDE Match-Fuzzy@10   ``in_cdematch_topk`` **OR** ``in_keyword_topk``
Union                every row in the pool
===================  =========================================================

The lexical arm is an OR, and getting that wrong badly understates it. The
fixed top-10 CDE Match-Fuzzy list is *decomposed at build time* into two flags:
``in_cdematch_topk`` for the clone tier and ``in_keyword_topk`` for the
fuzzy-fallback tier. ``in_cdematch_topk`` alone covers about 3.5% of rows.

Why not the K-selection grid
----------------------------
``artifacts/final_reranker/k_selection/k_selection_grid.csv`` also carries union
coverage, and disagrees on exactly three queries. It predates the
production-catalog eligibility correction and **must not be cited**. The
divergences are traced individually in ``candidate_coverage_v1/summary/
discrepant_queries.csv``; the GDC one is instructive, being a false positive
where a retired version sharing the gold's public identifier sat at rank 1.

The union coverage computed here equals the shipped HGBC's
``n_queries_with_gold_in_candidates`` and its ``recall@100`` on every dataset,
so S5.6 and S5.7 provably rest on one identical pool.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import artifact_root

#: Evaluation splits, in manuscript order, with their display labels.
EVAL_SPLITS = ("test", "cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2")
DISPLAY = {"test": "Test", "cctg": "CCTG", "oid_alt": "OID ALT",
           "cdash": "CDASH", "gdc_combined": "GDC", "cimac_v2": "CIMAC"}

#: Flags read from the feature table.
FLAG_COLUMNS = ("in_biencoder_topk", "in_cdematch_topk", "in_keyword_topk")
REQUIRED_COLUMNS = ("split", "query_id", "is_label") + FLAG_COLUMNS

DEFAULT_FEATURE_TABLE = (
    "artifacts/final_reranker/hgbc_features_v2_eligible/"
    "feature_table_fixedk30_crossenc_v2.parquet")


def compute_coverage(features: pd.DataFrame, splits=EVAL_SPLITS) -> pd.DataFrame:
    """One row per evaluation dataset: covered counts and proportions per arm.

    ``features`` needs only the columns in :data:`REQUIRED_COLUMNS`.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")

    df = features.copy()
    for col in FLAG_COLUMNS + ("is_label",):
        df[col] = df[col].astype(bool)
    # The lexical arm is the OR of the two decomposed tiers.
    df["in_fuzzy10"] = df["in_cdematch_topk"] | df["in_keyword_topk"]

    rows = []
    for split in splits:
        d = df[df["split"] == split]
        if d.empty:
            raise ValueError(f"no rows for split {split!r}")
        n = d["query_id"].nunique()

        def covered(mask_col: str | None) -> int:
            """Queries with a gold row flagged into the given arm."""
            hit = d["is_label"] if mask_col is None else (d["is_label"] & d[mask_col])
            return int(hit.groupby(d["query_id"]).any().sum())

        cov_be, cov_fz, cov_un = (covered("in_biencoder_topk"),
                                  covered("in_fuzzy10"), covered(None))
        rows.append({
            "split": split, "dataset": DISPLAY[split], "n_queries": n,
            "ftmpnet20_n": cov_be, "ftmpnet20_prop": round(cov_be / n, 4),
            "fuzzy10_n": cov_fz, "fuzzy10_prop": round(cov_fz / n, 4),
            "union_n": cov_un, "union_prop": round(cov_un / n, 4),
        })
    return pd.DataFrame(rows)


def compute_gains(coverage: pd.DataFrame) -> pd.DataFrame:
    """What the union adds over each arm alone, and over the stronger of the two."""
    g = coverage[["dataset", "n_queries"]].copy()
    g["union_minus_ftmpnet_n"] = coverage.union_n - coverage.ftmpnet20_n
    g["union_minus_ftmpnet_prop"] = (coverage.union_prop - coverage.ftmpnet20_prop).round(4)
    g["union_minus_fuzzy_n"] = coverage.union_n - coverage.fuzzy10_n
    g["union_minus_fuzzy_prop"] = (coverage.union_prop - coverage.fuzzy10_prop).round(4)
    g["stronger_arm"] = ["FT-MPNet" if a > b else ("CDE Match-Fuzzy" if b > a else "tie")
                         for a, b in zip(coverage.ftmpnet20_n, coverage.fuzzy10_n)]
    stronger_n = coverage[["ftmpnet20_n", "fuzzy10_n"]].max(axis=1)
    stronger_p = coverage[["ftmpnet20_prop", "fuzzy10_prop"]].max(axis=1)
    g["union_minus_stronger_n"] = coverage.union_n - stronger_n
    g["union_minus_stronger_prop"] = (coverage.union_prop - stronger_p).round(4)
    # Inclusion-exclusion: |A| + |B| - |A OR B|.
    g["both_arms_n"] = coverage.ftmpnet20_n + coverage.fuzzy10_n - coverage.union_n
    g["ftmpnet_only_n"] = coverage.union_n - coverage.fuzzy10_n
    g["fuzzy_only_n"] = coverage.union_n - coverage.ftmpnet20_n
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-table", default=None,
                    help=f"fixed-K feature table (default: $DEMAP_ARTIFACT_ROOT/{DEFAULT_FEATURE_TABLE})")
    ap.add_argument("--out-dir", default=None,
                    help="write candidate_coverage_by_dataset.csv and _gains.csv here")
    args = ap.parse_args(argv)

    table = Path(args.feature_table) if args.feature_table else artifact_root() / DEFAULT_FEATURE_TABLE
    if not table.is_file():
        sys.exit(f"ERROR: feature table not found: {table}\n"
                 f"Set DEMAP_ARTIFACT_ROOT or pass --feature-table.")

    features = pd.read_parquet(table, columns=list(REQUIRED_COLUMNS))
    coverage = compute_coverage(features)
    gains = compute_gains(coverage)

    print(coverage.to_string(index=False))
    print()
    print(gains.to_string(index=False))

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(out / "candidate_coverage_by_dataset.csv", index=False)
        gains.to_csv(out / "candidate_coverage_gains.csv", index=False)
        print(f"\nwrote {out}/candidate_coverage_by_dataset.csv")
        print(f"wrote {out}/candidate_coverage_gains.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
