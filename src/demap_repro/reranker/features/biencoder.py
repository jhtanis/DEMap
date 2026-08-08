"""Bi-encoder derived features for the candidate-union/XGBoost stack (PR-D).

Reads bi-encoder-observable inputs (``biencoder_rank``, ``biencoder_score``,
``in_biencoder_topk``) from a candidate-union DataFrame and emits derived
feature columns that downstream PRs (PR-F agreement features, PR-G
calibration, PR-H XGBoost trainer) will consume.

Every feature in this module is inference-computable and label-free:
  - No ``is_label``, ``true_*``, ``gold_*``, ``is_correct_top1`` is read.
  - Per-row features use only this row's ``biencoder_score`` and
    ``biencoder_rank`` (NaN where ``in_biencoder_topk`` is False).
  - Per-query broadcasts use only the query's bi-encoder top-K rows.

NaN policy:
  - Per-row score/rank-derived features are NaN whenever
    ``in_biencoder_topk`` is False.
  - Boolean indicators (``in_top5``, ``in_top10``) are always defined
    (False when ``in_biencoder_topk`` is False or rank exceeds threshold).
  - ``top1_score`` is NaN if the query has zero bi-encoder candidates.
  - ``margin_1_2`` is NaN if the query has fewer than two candidates.
  - ``score_z`` is NaN when the within-query sample std (ddof=1) is
    undefined (n < 2) or zero (all scores identical).
  - ``candidate_count`` is always defined (0 if the query has no
    bi-encoder candidates).

The function does not mutate its input. Re-running it on its own output is
an error (raises) — drop the existing feature columns first.

Multi-model support: the ``prefix`` parameter (default ``"bienc_"``) is
prepended to every output column name. Callers using multiple bi-encoder
models can pass e.g. ``prefix="bienc_mpnet_"`` to avoid column collisions.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from demap_repro.pool.candidate_union import FORBIDDEN_LEAKAGE_COLUMNS


BIENCODER_REQUIRED_INPUT_COLUMNS: Tuple[str, ...] = (
    "query_id",
    "biencoder_rank",
    "biencoder_score",
    "in_biencoder_topk",
)

_OPTIONAL_GROUP_PREFIX_COLUMNS: Tuple[str, ...] = ("winner_id", "split")

_FEATURE_SUFFIXES: Tuple[str, ...] = (
    "score_normalized",
    "log1p_rank",
    "reciprocal_rank",
    "in_top5",
    "in_top10",
    "top1_score",
    "margin_to_top1",
    "margin_1_2",
    "score_z",
    "candidate_count",
)


def biencoder_feature_columns(prefix: str = "bienc_") -> Tuple[str, ...]:
    return tuple(f"{prefix}{s}" for s in _FEATURE_SUFFIXES)


__all__ = [
    "BIENCODER_REQUIRED_INPUT_COLUMNS",
    "biencoder_feature_columns",
    "compute_biencoder_features",
]


def _resolve_group_keys(df: pd.DataFrame) -> List[str]:
    keys = [c for c in _OPTIONAL_GROUP_PREFIX_COLUMNS if c in df.columns]
    keys.append("query_id")
    return keys


def compute_biencoder_features(
    union_df: pd.DataFrame,
    *,
    prefix: str = "bienc_",
) -> pd.DataFrame:
    """Return a new DataFrame with bi-encoder-derived feature columns appended.

    Required input columns:
      ``query_id``, ``biencoder_rank``, ``biencoder_score``,
      ``in_biencoder_topk``.

    Optional grouping columns honored when present: ``winner_id``, ``split``.
    """
    feature_cols = biencoder_feature_columns(prefix)

    missing = [c for c in BIENCODER_REQUIRED_INPUT_COLUMNS if c not in union_df.columns]
    if missing:
        raise ValueError(
            f"compute_biencoder_features missing required input columns: {missing}"
        )

    already_present = [c for c in feature_cols if c in union_df.columns]
    if already_present:
        raise ValueError(
            f"compute_biencoder_features: output columns already present in input: "
            f"{already_present}. Drop them first."
        )

    out = union_df.copy()
    group_keys = _resolve_group_keys(out)

    rank_float = out["biencoder_rank"].astype(float)
    score = out["biencoder_score"].astype(float)
    in_topk = out["in_biencoder_topk"].fillna(False).astype(bool)

    # --- Per-query stats from in_biencoder_topk rows ---
    topk_rows = out.loc[
        in_topk,
        group_keys + ["biencoder_score", "biencoder_rank"],
    ].copy()

    if len(topk_rows) > 0:
        topk_sorted = (
            topk_rows
            .sort_values(group_keys + ["biencoder_rank"], kind="mergesort")
            .reset_index(drop=True)
        )
        topk_sorted["_pos"] = topk_sorted.groupby(group_keys, sort=False).cumcount()

        top1 = (
            topk_sorted.loc[topk_sorted["_pos"] == 0, group_keys + ["biencoder_score"]]
            .rename(columns={"biencoder_score": f"{prefix}top1_score"})
        )
        top2 = (
            topk_sorted.loc[topk_sorted["_pos"] == 1, group_keys + ["biencoder_score"]]
            .rename(columns={"biencoder_score": "_top2_score"})
        )
        agg = (
            topk_sorted
            .groupby(group_keys, sort=False, as_index=False)
            .agg(
                _count=("biencoder_score", "size"),
                _score_mean=("biencoder_score", "mean"),
                _score_std=("biencoder_score", lambda s: s.std(ddof=1)),
                _score_max=("biencoder_score", "max"),
            )
        )
    else:
        top1 = pd.DataFrame(
            {**{k: pd.Series(dtype="object") for k in group_keys},
             f"{prefix}top1_score": pd.Series(dtype="float64")}
        )
        top2 = pd.DataFrame(
            {**{k: pd.Series(dtype="object") for k in group_keys},
             "_top2_score": pd.Series(dtype="float64")}
        )
        agg = pd.DataFrame(
            {**{k: pd.Series(dtype="object") for k in group_keys},
             "_count": pd.Series(dtype="Int64"),
             "_score_mean": pd.Series(dtype="float64"),
             "_score_std": pd.Series(dtype="float64"),
             "_score_max": pd.Series(dtype="float64")}
        )

    out = out.merge(top1, on=group_keys, how="left")
    out = out.merge(top2, on=group_keys, how="left")
    out = out.merge(agg, on=group_keys, how="left")

    # --- Per-row features ---
    safe_max = out["_score_max"].where(out["_score_max"] > 0)
    out[f"{prefix}score_normalized"] = score / safe_max

    out[f"{prefix}log1p_rank"] = np.log1p(rank_float - 1.0)

    out[f"{prefix}reciprocal_rank"] = np.where(
        rank_float.notna() & (rank_float > 0),
        1.0 / rank_float,
        np.nan,
    )

    out[f"{prefix}in_top5"] = in_topk & (rank_float <= 5)
    out[f"{prefix}in_top10"] = in_topk & (rank_float <= 10)

    out[f"{prefix}margin_to_top1"] = (
        out[f"{prefix}top1_score"] - score
    )

    out[f"{prefix}margin_1_2"] = (
        out[f"{prefix}top1_score"] - out["_top2_score"]
    )

    out[f"{prefix}candidate_count"] = out["_count"].fillna(0).astype("Int64")

    safe_std = out["_score_std"].where(out["_score_std"] > 0)
    out[f"{prefix}score_z"] = (score - out["_score_mean"]) / safe_std

    # --- Cleanup ---
    out = out.drop(columns=["_top2_score", "_score_mean", "_score_std", "_score_max", "_count"])

    new_cols = list(feature_cols)
    final_cols = [c for c in union_df.columns if c not in new_cols] + new_cols
    out = out[final_cols]

    bad = sorted(set(out.columns) & FORBIDDEN_LEAKAGE_COLUMNS)
    if bad:
        raise RuntimeError(
            f"compute_biencoder_features produced forbidden leakage columns: {bad}"
        )

    return out
