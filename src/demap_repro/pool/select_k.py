#!/usr/bin/env python3
"""Choose the candidate-pool size K.

The pool is a union of two ranked lists, deduplicated by CDE public identifier:
the FT-MPNet bi-encoder top ``se_k`` and CDE Match-Fuzzy top ``kw_k``. The keyword
depth is fixed at 10 and the bi-encoder depth is swept, giving nominal
K = 20, 30, 40, 60.

What is being measured is a **ceiling**: the fraction of queries whose gold CDE is
anywhere in the pool. No reranker can exceed it, so K is chosen where the ceiling
stops paying for the extra candidates the cross-encoder and HGBC must then score.

S5.3: the ceiling rises from 0.9828 at K = 20 to 0.9888 at K = 30 on Validation
Training, then flattens — only 0.0036 more between K = 30 and K = 60. K = 30 is
selected on Validation Training; Validation Dev is reported for description only
and gives the same 0.9888.

Branch discipline
-----------------
The grid also carries ``clone`` and ``clone_or_fuzzy`` branches, explored during
development. **They are not the paper pool.** The final system uses ``kwfuzzy``
only, deliberately: the Python approximation to NCI CDE Match is one of the
methods being *compared against* in Table 4, so building the pool from it would
compromise that comparison. Selecting on the wrong branch would change K and the
reported ceilings, so the branch is an explicit argument with no permissive
default.

Migration
---------
Three near-duplicate selectors existed in the research repository. The executed
producer of ``k_selection_grid.csv`` was
``.scratch/demap/final_reranker_steps_cd/k_selection_recheck.py``; the other two
(``reranker_k_selection_val_train/select_k.py``,
``reranker_k_ceiling_audit/build_k_ceiling.py``) were earlier variants and are not
migrated. This module keeps one implementation: :func:`compute_k_grid` builds the
grid from candidate rankings, :func:`select_k` makes the decision from a grid.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

import pandas as pd

__all__ = [
    "PAPER_BRANCH", "PAPER_KEYWORD_DEPTH", "PAPER_SELECTION_SPLIT",
    "PAPER_SELECTED_K", "SE_DEPTHS", "nominal_k",
    "compute_k_grid", "select_k",
]

#: The keyword arm the paper pool is built from.
PAPER_BRANCH = "kwfuzzy"

#: Keyword depth is held fixed while the bi-encoder depth is swept.
PAPER_KEYWORD_DEPTH = 10

#: K is selected on Validation Training; Validation Dev is descriptive only.
PAPER_SELECTION_SPLIT = "val_train"

#: The selected nominal pool size.
PAPER_SELECTED_K = 30

#: Bi-encoder depths swept, giving nominal K of 20/30/40/60 at keyword depth 10.
SE_DEPTHS = (10, 20, 30, 50)


def nominal_k(se_k: int, kw_k: int) -> int:
    """Nominal pool size: the two depths summed, before dedup by public id.

    The realized pool is smaller — about 27.6 candidates at nominal K = 30 —
    because the arms overlap.
    """
    return int(se_k) + int(kw_k)


def _publicid(cde_id: object) -> str:
    return str(cde_id).split("::")[0]


def compute_k_grid(
    gold: Mapping[str, Set[str]],
    biencoder_ranks: Mapping[str, Mapping[str, int]],
    keyword_ranks: Mapping[str, Mapping[str, int]],
    *,
    dataset: str,
    branch: str = PAPER_BRANCH,
    se_depths: Sequence[int] = SE_DEPTHS,
    kw_depths: Iterable[int] = (PAPER_KEYWORD_DEPTH,),
) -> pd.DataFrame:
    """Ceiling recall over a depth grid for one dataset.

    Parameters
    ----------
    gold
        ``query_id -> set of gold CDE public identifiers``. A query counts as
        covered when ANY of its golds is in the pool, matching the "any gold"
        convention used by every reported recall number.
    biencoder_ranks, keyword_ranks
        ``query_id -> {public_id: rank}``, ranks 1-based, already deduplicated to
        the best rank per public identifier.
    """
    qids = [q for q, g in gold.items() if g]
    n = len(qids)
    if not n:
        raise ValueError(f"{dataset}: no queries with gold")

    rows = []
    for se_k in se_depths:
        for kw_k in kw_depths:
            covered = 0
            total_size = 0
            for q in qids:
                pool = {p for p, r in biencoder_ranks.get(q, {}).items() if r <= se_k}
                if kw_k:
                    pool |= {p for p, r in keyword_ranks.get(q, {}).items() if r <= kw_k}
                total_size += len(pool)
                if gold[q] & pool:
                    covered += 1
            rows.append({
                "dataset": dataset,
                "branch": branch if kw_k else "none",
                "se_k": se_k, "kw_k": kw_k,
                "n_queries": n,
                "gold_in_pool": covered,
                "pool_recall": round(covered / n, 4),
                "mean_pool_size": round(total_size / n, 1),
            })
    return pd.DataFrame(rows)


def select_k(
    grid: pd.DataFrame,
    *,
    branch: str = PAPER_BRANCH,
    keyword_depth: int = PAPER_KEYWORD_DEPTH,
    selection_split: str = PAPER_SELECTION_SPLIT,
    descriptive_split: Optional[str] = "val_dev",
    plateau_tolerance: float = 0.005,
) -> Dict:
    """Choose K from a candidate-pool ceiling grid.

    The rule is the smallest nominal K whose ceiling is within
    ``plateau_tolerance`` of the best ceiling on ``selection_split`` — "take the
    smallest pool that has stopped improving", which is what the S5.3 narrative
    describes. Every candidate row is returned so the decision can be inspected
    rather than trusted.
    """
    sel = grid[(grid["dataset"] == selection_split)
               & (grid["branch"] == branch)
               & (grid["kw_k"] == keyword_depth)].copy()
    if sel.empty:
        raise ValueError(
            f"no rows for dataset={selection_split!r} branch={branch!r} "
            f"kw_k={keyword_depth}; available branches "
            f"{sorted(grid['branch'].unique())}")
    sel["k_nominal"] = [nominal_k(a, b) for a, b in zip(sel["se_k"], sel["kw_k"])]
    sel = sel.sort_values("k_nominal").reset_index(drop=True)

    best = float(sel["pool_recall"].max())
    within = sel[sel["pool_recall"] >= best - plateau_tolerance]
    chosen = within.iloc[0]

    result = {
        "branch": branch,
        "keyword_depth": keyword_depth,
        "selection_split": selection_split,
        "selection_metric": "ceiling recall (gold anywhere in pool)",
        "plateau_tolerance": plateau_tolerance,
        "selected_k": int(chosen["k_nominal"]),
        "selected_se_k": int(chosen["se_k"]),
        "selected_ceiling_recall": float(chosen["pool_recall"]),
        "selected_mean_pool_size": float(chosen["mean_pool_size"]),
        "best_ceiling_recall": best,
        "grid": [
            {"k_nominal": int(r["k_nominal"]), "se_k": int(r["se_k"]),
             "kw_k": int(r["kw_k"]), "pool_recall": float(r["pool_recall"]),
             "mean_pool_size": float(r["mean_pool_size"]),
             "n_queries": int(r["n_queries"]), "gold_in_pool": int(r["gold_in_pool"])}
            for _, r in sel.iterrows()
        ],
    }

    if descriptive_split:
        desc = grid[(grid["dataset"] == descriptive_split)
                    & (grid["branch"] == branch)
                    & (grid["kw_k"] == keyword_depth)
                    & (grid["se_k"] == chosen["se_k"])]
        if len(desc):
            result["descriptive_split"] = descriptive_split
            result["descriptive_ceiling_recall"] = float(desc.iloc[0]["pool_recall"])
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, help="k_selection_grid.csv")
    ap.add_argument("--branch", default=PAPER_BRANCH)
    ap.add_argument("--keyword-depth", type=int, default=PAPER_KEYWORD_DEPTH)
    ap.add_argument("--selection-split", default=PAPER_SELECTION_SPLIT)
    args = ap.parse_args(argv)

    import json
    grid = pd.read_csv(Path(args.grid))
    result = select_k(grid, branch=args.branch, keyword_depth=args.keyword_depth,
                      selection_split=args.selection_split)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
