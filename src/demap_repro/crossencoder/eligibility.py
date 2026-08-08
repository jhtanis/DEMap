"""Training-query eligibility for the cross-encoder.

The Train split is deliberately **not** reachability-filtered: it uses all pairs,
including ones whose gold CDE is absent from the June production catalog. Those
remain valid training positives, because candidate text still resolves against
the full enriched catalog.

A small number of train queries are different: their gold is neither present in
the deployable candidate pool nor injectable from production. They have no
obtainable positive at all, so a pointwise binary cross-encoder has nothing to
learn from them — every row is a negative, and the query contributes only label
noise. The paper removes them.

In the research repository this removal existed as five lines inside a Slurm
chain job (``chain_F2_pool_pairs_fulltrain_v2.sbatch`` ->
``build_ce_pool_train_v2.py``), visible only as a log line:

    dropped 172 train queries with no obtainable positive
    (fully off-production gold, not in pool)

Here it is a named step with a regression test, because "172 queries silently
vanished from training" is exactly the kind of detail that should not live in a
job script. S5.4 refers to it as the "corrected, production-eligible candidate
pool".

The removal is **train-only** by construction: evaluation and dev splits are
already reachability-filtered upstream, and dropping an evaluation query would
change a reported denominator.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

__all__ = ["drop_train_queries_without_obtainable_positive"]

#: Number of train queries removed for the canonical paper inputs.
PAPER_N_DROPPED = 172

#: Train queries before and after the removal, for the canonical paper inputs.
PAPER_N_TRAIN_QUERIES_BEFORE = 47645
PAPER_N_TRAIN_QUERIES_AFTER = 47473


def drop_train_queries_without_obtainable_positive(
    pool: pd.DataFrame,
    *,
    train_split: str = "train",
    split_col: str = "split",
    query_col: str = "query_id",
    label_col: str = "is_label",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove train queries that have no positive candidate anywhere in ``pool``.

    Parameters
    ----------
    pool
        The candidate pool, one row per (split, query, candidate), carrying a
        boolean ``is_label`` marking gold candidates.
    train_split
        Name of the split to filter. Only rows of this split are ever removed.

    Returns
    -------
    (filtered_pool, dropped_query_ids)
        ``dropped_query_ids`` is sorted, so callers get a stable record.

    Notes
    -----
    Rows of every other split pass through untouched — including their row count,
    order and dtypes — so this can never move an evaluation denominator.
    """
    for col in (split_col, query_col, label_col):
        if col not in pool.columns:
            raise KeyError(f"pool is missing required column {col!r}")

    is_train = pool[split_col] == train_split
    if not is_train.any():
        if verbose:
            print(f"  no {train_split!r} rows in pool; nothing to filter")
        return pool, []

    train_queries = set(pool.loc[is_train, query_col])
    with_positive = set(pool.loc[is_train & pool[label_col].astype(bool), query_col])
    dropped = sorted(train_queries - with_positive, key=str)

    if not dropped:
        if verbose:
            print(f"  all {len(train_queries)} {train_split!r} queries have an "
                  "obtainable positive; nothing dropped")
        return pool, []

    filtered = pool[~(is_train & pool[query_col].isin(dropped))].copy()
    if verbose:
        print(f"  dropped {len(dropped)} {train_split!r} queries with no obtainable "
              f"positive (fully off-production gold, not in pool): "
              f"{len(train_queries)} -> {len(train_queries) - len(dropped)} queries")
    return filtered, dropped
