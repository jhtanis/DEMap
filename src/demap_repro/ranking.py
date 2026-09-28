"""Shared deterministic ranking policies used by paper verification and runtime."""
from __future__ import annotations

import math
from typing import Any, Iterable

import pandas as pd


def public_id_sort_key(value: Any) -> tuple[int, int, str]:
    """Numeric public IDs first numerically, then nonnumeric IDs lexically."""
    text = str(value)
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


def canonical_score_order(
    public_ids: Iterable[Any], scores: Iterable[Any]
) -> tuple[list[str], list[float]]:
    """Return score-desc/public-ID-ascending order, independent of input order."""
    ids = [str(value) for value in public_ids]
    numeric_scores = [float(value) for value in scores]
    if len(ids) != len(numeric_scores):
        raise ValueError("candidate ID/score list-length mismatch")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate public identifier")
    if any(not math.isfinite(value) for value in numeric_scores):
        raise ValueError("non-finite candidate score")
    order = sorted(
        range(len(ids)),
        key=lambda index: (-numeric_scores[index], public_id_sort_key(ids[index])),
    )
    return [ids[index] for index in order], [numeric_scores[index] for index in order]


def assign_canonical_score_ranks(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    score_col: str,
    public_id_col: str,
    rank_col: str,
) -> pd.DataFrame:
    """Assign 1-based canonical ranks while preserving the caller's row layout."""
    required = set(group_cols + [score_col, public_id_col])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"canonical ranking missing columns: {missing}")
    out = frame.copy()
    duplicate = out.duplicated(group_cols + [public_id_col], keep=False)
    if duplicate.any():
        raise ValueError(
            "duplicate public identifier within ranking group: "
            f"{out.loc[duplicate, group_cols + [public_id_col]].head().to_dict('records')}"
        )
    scores = pd.to_numeric(out[score_col], errors="coerce")
    if scores.isna().any() or (~scores.map(math.isfinite)).any():
        raise ValueError(f"{score_col} contains missing or non-finite values")
    keys = out[public_id_col].map(public_id_sort_key)
    out["__paper_id_kind"] = keys.map(lambda key: key[0])
    out["__paper_id_number"] = keys.map(lambda key: key[1])
    out["__paper_id_text"] = keys.map(lambda key: key[2])
    out["__paper_original_index"] = range(len(out))
    ordered = out.sort_values(
        group_cols + [score_col, "__paper_id_kind", "__paper_id_number", "__paper_id_text"],
        ascending=[True] * len(group_cols) + [False, True, True, True],
        kind="mergesort",
    )
    ordered[rank_col] = ordered.groupby(group_cols, sort=False).cumcount() + 1
    rank_by_index = ordered.set_index("__paper_original_index")[rank_col]
    out[rank_col] = out["__paper_original_index"].map(rank_by_index).astype("Int64")
    return out.drop(columns=[
        "__paper_id_kind", "__paper_id_number", "__paper_id_text",
        "__paper_original_index",
    ])
