"""CDE Match derived features for the candidate-union/XGBoost stack (PR-C).

Reads CDE Match-observable inputs (``cdematch_rank``, ``cdematch_score``,
``in_cdematch_topk``) from a candidate-union DataFrame and emits 7 derived
feature columns that downstream PRs (PR-D bi-encoder features, PR-F
agreement features, PR-G calibration, PR-H XGBoost trainer) will consume.

Every feature in this module is inference-computable and label-free:
  - No ``is_label``, ``true_*``, ``gold_*``, ``is_correct_top1`` is read.
  - Per-row features use only this row's ``cdematch_score`` and
    ``cdematch_rank`` (NaN where ``in_cdematch_topk`` is False).
  - Per-query broadcasts use only the query's CDE Match top-K rows.

NaN policy:
  - Per-row features (``cdematch_score_normalized``, ``log1p_cdematch_rank``,
    ``cdematch_margin_to_top1``, ``cdematch_score_z_within_query``) are NaN
    whenever ``in_cdematch_topk`` is False.
  - ``cdematch_top1_score`` is NaN if the query has zero CDE Match candidates.
  - ``cdematch_margin_1_2`` is NaN if the query has fewer than two CDE Match
    candidates.
  - ``cdematch_score_z_within_query`` is also NaN when the within-query
    sample std (ddof=1) is undefined (n < 2) or zero (all scores identical).
  - ``cdematch_candidate_count`` is always defined (0 if the query has no
    CDE Match candidates).

The function does not mutate its input. Re-running it on its own output is
an error (raises) — drop the existing feature columns first.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from demap_repro.pool.candidate_union import FORBIDDEN_LEAKAGE_COLUMNS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CDEMATCH_SCORE_MAX: float = 100.0


CDEMATCH_FEATURE_COLUMNS: Tuple[str, ...] = (
    "cdematch_score_normalized",
    "log1p_cdematch_rank",
    "cdematch_top1_score",
    "cdematch_margin_to_top1",
    "cdematch_margin_1_2",
    "cdematch_score_z_within_query",
    "cdematch_candidate_count",
)


CDEMATCH_REQUIRED_INPUT_COLUMNS: Tuple[str, ...] = (
    "query_id",
    "cdematch_rank",
    "cdematch_score",
    "in_cdematch_topk",
)


# Optional grouping prefix used when present. Per-query stats are scoped to
# the smallest meaningful partition: when both ``winner_id`` and ``split``
# are in the input (typical for candidate-union output), grouping is
# ``(winner_id, split, query_id)``; otherwise it falls back gracefully.
_OPTIONAL_GROUP_PREFIX_COLUMNS: Tuple[str, ...] = ("winner_id", "split")


__all__ = [
    "CDEMATCH_SCORE_MAX",
    "CDEMATCH_FEATURE_COLUMNS",
    "CDEMATCH_REQUIRED_INPUT_COLUMNS",
    "compute_cdematch_features",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_group_keys(df: pd.DataFrame) -> List[str]:
    """Return the per-query grouping keys to use.

    Prefers ``(winner_id, split, query_id)`` when all three are present;
    otherwise drops the missing prefix columns and uses what remains.
    ``query_id`` is always included.
    """
    keys = [c for c in _OPTIONAL_GROUP_PREFIX_COLUMNS if c in df.columns]
    keys.append("query_id")
    return keys


def _empty_topk_summary(group_keys: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Typed empty frames for the empty-in_topk corner case.

    Returning these allows the main code path to use the same merge pattern
    whether or not any rows have ``in_cdematch_topk == True``.
    """
    top1 = pd.DataFrame({**{k: pd.Series(dtype="object") for k in group_keys},
                         "cdematch_top1_score": pd.Series(dtype="float64")})
    top2 = pd.DataFrame({**{k: pd.Series(dtype="object") for k in group_keys},
                         "_top2_score": pd.Series(dtype="float64")})
    agg = pd.DataFrame({**{k: pd.Series(dtype="object") for k in group_keys},
                        "cdematch_candidate_count": pd.Series(dtype="Int64"),
                        "_score_mean": pd.Series(dtype="float64"),
                        "_score_std": pd.Series(dtype="float64")})
    return top1, top2, agg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_cdematch_features(union_df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with 7 CDE Match-derived feature columns.

    Required input columns:
      ``query_id``, ``cdematch_rank``, ``cdematch_score``, ``in_cdematch_topk``.

    Optional grouping columns honored when present: ``winner_id``, ``split``.

    Output: a copy of ``union_df`` with the columns listed in
    ``CDEMATCH_FEATURE_COLUMNS`` appended in that order. Row order is
    preserved. The input frame is not mutated.

    Raises ``ValueError`` if any required input column is missing, or if any
    of the output feature columns is already present in the input (the
    function is not idempotent; drop existing feature columns first).
    """
    # 1. Required-input check.
    missing = [c for c in CDEMATCH_REQUIRED_INPUT_COLUMNS if c not in union_df.columns]
    if missing:
        raise ValueError(
            f"compute_cdematch_features missing required input columns: {missing}"
        )

    # 2. Idempotence guard.
    already_present = [c for c in CDEMATCH_FEATURE_COLUMNS if c in union_df.columns]
    if already_present:
        raise ValueError(
            f"compute_cdematch_features: output columns already present in input: "
            f"{already_present}. Drop them first."
        )

    out = union_df.copy()
    group_keys = _resolve_group_keys(out)

    # 3. Per-row features (NaN-safe via NaN propagation).
    score = out["cdematch_score"].astype(float)
    # cdematch_rank may be Int64 nullable — cast through float so NA → NaN.
    rank_float = out["cdematch_rank"].astype(float)
    out["cdematch_score_normalized"] = score / CDEMATCH_SCORE_MAX
    out["log1p_cdematch_rank"] = np.log1p(rank_float - 1.0)

    # 4. Per-query stats from in_cdematch_topk rows.
    in_topk = out.loc[
        out["in_cdematch_topk"],
        group_keys + ["cdematch_score", "cdematch_rank"],
    ].copy()

    if len(in_topk) > 0:
        in_topk_sorted = (
            in_topk
            .sort_values(group_keys + ["cdematch_rank"], kind="mergesort")
            .reset_index(drop=True)
        )
        # _pos is the within-group positional index (0 = rank 1, 1 = rank 2, ...).
        in_topk_sorted["_pos"] = in_topk_sorted.groupby(group_keys, sort=False).cumcount()

        top1 = (
            in_topk_sorted.loc[in_topk_sorted["_pos"] == 0,
                               group_keys + ["cdematch_score"]]
            .rename(columns={"cdematch_score": "cdematch_top1_score"})
        )
        top2 = (
            in_topk_sorted.loc[in_topk_sorted["_pos"] == 1,
                               group_keys + ["cdematch_score"]]
            .rename(columns={"cdematch_score": "_top2_score"})
        )
        agg = (
            in_topk_sorted
            .groupby(group_keys, sort=False, as_index=False)
            .agg(
                cdematch_candidate_count=("cdematch_score", "size"),
                _score_mean=("cdematch_score", "mean"),
                _score_std=("cdematch_score", lambda s: s.std(ddof=1)),
            )
        )
    else:
        top1, top2, agg = _empty_topk_summary(group_keys)

    # 5. Left-merges (preserve out's row order; right-side keys are unique).
    out = out.merge(top1, on=group_keys, how="left")
    out = out.merge(top2, on=group_keys, how="left")
    out = out.merge(agg, on=group_keys, how="left")

    # 6. Candidate count is always defined (queries with no in_topk get 0).
    out["cdematch_candidate_count"] = (
        out["cdematch_candidate_count"].fillna(0).astype("Int64")
    )

    # 7. Derived per-query / per-row features.
    out["cdematch_margin_1_2"] = out["cdematch_top1_score"] - out["_top2_score"]
    out["cdematch_margin_to_top1"] = (
        out["cdematch_top1_score"] - out["cdematch_score"]
    )
    safe_std = out["_score_std"].where(out["_score_std"] > 0)
    out["cdematch_score_z_within_query"] = (
        (out["cdematch_score"] - out["_score_mean"]) / safe_std
    )

    # 8. Drop helper columns.
    out = out.drop(columns=["_top2_score", "_score_mean", "_score_std"])

    # 9. Final column order: original columns first, then new in spec order.
    new_cols = list(CDEMATCH_FEATURE_COLUMNS)
    final_cols = [c for c in union_df.columns if c not in new_cols] + new_cols
    out = out[final_cols]

    # 10. Leakage guard (defensive; should never fire).
    bad = sorted(set(out.columns) & FORBIDDEN_LEAKAGE_COLUMNS)
    if bad:  # pragma: no cover — defensive
        raise RuntimeError(
            f"compute_cdematch_features produced forbidden leakage columns: {bad}"
        )

    return out
