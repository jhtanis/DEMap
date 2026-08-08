from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from demap_repro.text.recipes import parse_recipe

DEFAULT_PRIMARY_METRIC = 'recall@5'
DEFAULT_SECONDARY_METRIC = 'mrr@100'
DEFAULT_TERTIARY_METRIC = 'recall@10'
DEFAULT_TIE_MARGIN = 0.005

# Screening tie-break policy agreed for this project.
# Lower is simpler / more interpretable: Q1 < Q3 < Q4 < Q2.
QUESTION_COMPLEXITY_RANK = {
    'Q1': 0,
    'Q3': 1,
    'Q4': 2,
    'Q2': 3,
}

# Atomic CDE-side recipe-component complexity. Lower is simpler / more interpretable.
# User-agreed anchor: LN < SN < DEF ~ PQT < PV-Summary.
# We place DEC and VALUE_DOMAIN in the middle tier to keep deterministic behavior
# for recipes that include them, while still favoring PV-heavy recipes as richer.
ATOMIC_RECIPE_COMPLEXITY_RANK = {
    'v2a': 0,  # LONG_NAME
    'v1': 1,   # SHORT_NAME
    'v2b': 2,  # DEFINITION
    'v3': 2,   # PREFERRED_QUESTION_TEXT
    'v6': 2,   # DEC_LONG_NAME
    'v4': 2,   # VALUE_DOMAIN_TYPE | VALUE_DOMAIN_DATATYPE
    'v5': 3,   # PV_SUMMARY
}

LOSS_COMPLEXITY_RANK = {
    'mnrl': 0,
    'symmetric_mnrl': 1,
    'cached_mnrl': 2,
}


def _as_list(xs: Optional[Iterable[str]]) -> list[str]:
    if xs is None:
        return []
    return [str(x) for x in xs]


def _normalize_query_variant(qv: object) -> str:
    return str(qv or '').strip().upper()


def _normalize_loss_name(loss_name: object) -> str:
    return str(loss_name or '').strip().lower()


def _expanded_recipe_atoms(recipe: object) -> list[str]:
    parts = parse_recipe(str(recipe))
    atoms: list[str] = []
    for part in parts:
        if part == 'v2':
            atoms.extend(['v2a', 'v2b'])
        else:
            atoms.append(part)
    return atoms


def question_complexity_rank(query_variant: object) -> int:
    qv = _normalize_query_variant(query_variant)
    return int(QUESTION_COMPLEXITY_RANK.get(qv, 99))


def loss_complexity_rank(loss_name: object) -> int:
    return int(LOSS_COMPLEXITY_RANK.get(_normalize_loss_name(loss_name), 99))


def recipe_complexity_tuple(recipe: object) -> tuple:
    """Return a deterministic recipe-complexity key.

    Lower tuples are preferred. The comparison is intentionally simple and
    transparent:
      1. avoid the richest atomic component tiers when possible (e.g., PV)
      2. prefer fewer atomic fields
      3. prefer lower overall component-complexity sum
      4. prefer lower ordered atomic complexity profile
      5. fall back to the recipe string for determinism

    This is used *only* for near-tie breaking.
    """

    atoms = _expanded_recipe_atoms(recipe)
    ranks = [int(ATOMIC_RECIPE_COMPLEXITY_RANK.get(atom, 99)) for atom in atoms]
    ordered = tuple(ranks)
    if not ranks:
        return (99, 99, 99, ordered, str(recipe))
    return (
        max(ranks),
        len(ranks),
        sum(ranks),
        ordered,
        str(recipe),
    )


# Flattened numeric pieces for pandas sort fallback columns.
def recipe_complexity_columns(recipe: object) -> dict[str, object]:
    max_rank, n_atoms, sum_rank, ordered, recipe_name = recipe_complexity_tuple(recipe)
    cols: dict[str, object] = {
        'selection_recipe_complexity_max_rank': int(max_rank),
        'selection_recipe_complexity_n_atoms': int(n_atoms),
        'selection_recipe_complexity_sum_rank': int(sum_rank),
        'selection_recipe_complexity_profile': '|'.join(str(x) for x in ordered),
        'selection_recipe_complexity_recipe': str(recipe_name),
    }
    return cols


def add_representation_complexity_columns(
    df: pd.DataFrame,
    *,
    query_col: str = 'query_variant',
    recipe_col: str = 'recipe',
    loss_col: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    out['selection_question_complexity_rank'] = out[query_col].map(question_complexity_rank)
    recipe_cols = out[recipe_col].map(recipe_complexity_columns).apply(pd.Series)
    out = pd.concat([out, recipe_cols], axis=1)
    if loss_col is not None:
        out['selection_loss_complexity_rank'] = out[loss_col].map(loss_complexity_rank)
    return out


def order_candidates_with_tie_margin(
    df: pd.DataFrame,
    *,
    primary_col: str,
    secondary_col: Optional[str] = None,
    tertiary_col: Optional[str] = None,
    tie_margin: float = DEFAULT_TIE_MARGIN,
    fallback_cols: Optional[Sequence[str]] = None,
    fallback_ascending: Optional[Sequence[bool]] = None,
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Order candidates under the paper's near-tie policy.

    Policy:
      1. Highest primary metric wins.
      2. Candidates within ``tie_margin`` of the best primary are treated as tied.
      3. Ties are broken by the secondary metric.
      4. Exact secondary ties are broken by the tertiary metric.
      5. Any remaining exact ties fall through to deterministic fallback columns.

    The returned frame contains these helper columns:
      - ``selection_primary_best``
      - ``selection_primary_gap``
      - ``selection_within_tie_margin``
      - ``selection_tie_margin``
      - ``selection_rank``
    """
    if df.empty:
        out = df.copy()
        out['selection_primary_best'] = pd.Series(dtype=float)
        out['selection_primary_gap'] = pd.Series(dtype=float)
        out['selection_within_tie_margin'] = pd.Series(dtype=bool)
        out['selection_tie_margin'] = pd.Series(dtype=float)
        out['selection_rank'] = pd.Series(dtype=int)
        return out

    fallback_cols_list = _as_list(fallback_cols)
    if fallback_ascending is None:
        fallback_ascending_list = [True] * len(fallback_cols_list)
    else:
        fallback_ascending_list = [bool(x) for x in fallback_ascending]
        if len(fallback_ascending_list) != len(fallback_cols_list):
            raise ValueError('fallback_ascending must match fallback_cols length')

    def _metric_series(frame: pd.DataFrame, col: Optional[str]) -> pd.Series:
        if col is None:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[col], errors='coerce')

    def _order_one(frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy()
        work['selection_primary_best'] = float(_metric_series(work, primary_col).max(skipna=True))
        work['selection_primary_gap'] = work['selection_primary_best'] - _metric_series(work, primary_col)
        work['selection_within_tie_margin'] = work['selection_primary_gap'] <= (float(tie_margin) + 1e-12)
        work['selection_tie_margin'] = float(tie_margin)

        near_cols = [c for c in [secondary_col, tertiary_col] if c is not None] + fallback_cols_list
        near_ascending = [False] * len([c for c in [secondary_col, tertiary_col] if c is not None]) + fallback_ascending_list
        far_cols = [c for c in [primary_col, secondary_col, tertiary_col] if c is not None] + fallback_cols_list
        far_ascending = [False] * len([c for c in [primary_col, secondary_col, tertiary_col] if c is not None]) + fallback_ascending_list

        near = work[work['selection_within_tie_margin']].sort_values(near_cols, ascending=near_ascending, na_position='last')
        far = work[~work['selection_within_tie_margin']].sort_values(far_cols, ascending=far_ascending, na_position='last')
        out = pd.concat([near, far], axis=0, ignore_index=True)
        out['selection_rank'] = np.arange(1, len(out) + 1, dtype=int)
        return out

    group_cols_list = _as_list(group_cols)
    if not group_cols_list:
        return _order_one(df)

    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(group_cols_list, dropna=False, sort=False):
        pieces.append(_order_one(group))
    out = pd.concat(pieces, axis=0, ignore_index=True)
    sort_cols = group_cols_list + ['selection_rank']
    sort_ascending = [True] * len(group_cols_list) + [True]
    out = out.sort_values(sort_cols, ascending=sort_ascending, na_position='last').reset_index(drop=True)
    return out


def order_phase_tuning_configs(
    df: pd.DataFrame,
    *,
    primary_col: str = 'val_recall5_mean',
    secondary_col: str = 'val_mrr100_mean',
    primary_std_col: str = 'val_recall5_std',
    secondary_std_col: str = 'val_mrr100_std',
    tie_margin: float = DEFAULT_TIE_MARGIN,
    fallback_cols: Optional[Sequence[str]] = None,
    fallback_ascending: Optional[Sequence[bool]] = None,
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Order Phase-1 / Phase-2 tuning configs under the paper's near-tie policy.

    Policy:
      1. Highest primary metric wins.
      2. Candidates within ``tie_margin`` of the best primary are treated as tied.
      3. Ties are broken by the secondary metric.
      4. Remaining ties are broken by lower primary-metric variability.
      5. Then by lower secondary-metric variability.
      6. Any remaining ties fall through to deterministic fallback columns.

    The returned frame contains these helper columns:
      - ``selection_primary_best``
      - ``selection_primary_gap``
      - ``selection_within_tie_margin``
      - ``selection_tie_margin``
      - ``selection_rank``
    """
    if df.empty:
        out = df.copy()
        out['selection_primary_best'] = pd.Series(dtype=float)
        out['selection_primary_gap'] = pd.Series(dtype=float)
        out['selection_within_tie_margin'] = pd.Series(dtype=bool)
        out['selection_tie_margin'] = pd.Series(dtype=float)
        out['selection_rank'] = pd.Series(dtype=int)
        return out

    fallback_cols_list = _as_list(fallback_cols)
    if fallback_ascending is None:
        fallback_ascending_list = [True] * len(fallback_cols_list)
    else:
        fallback_ascending_list = [bool(x) for x in fallback_ascending]
        if len(fallback_ascending_list) != len(fallback_cols_list):
            raise ValueError('fallback_ascending must match fallback_cols length')

    def _metric_series(frame: pd.DataFrame, col: Optional[str]) -> pd.Series:
        if col is None:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[col], errors='coerce')

    def _order_one(frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy()
        work['selection_primary_best'] = float(_metric_series(work, primary_col).max(skipna=True))
        work['selection_primary_gap'] = work['selection_primary_best'] - _metric_series(work, primary_col)
        work['selection_within_tie_margin'] = work['selection_primary_gap'] <= (float(tie_margin) + 1e-12)
        work['selection_tie_margin'] = float(tie_margin)

        near_cols = [c for c in [secondary_col] if c is not None]
        near_cols += [c for c in [primary_std_col, secondary_std_col] if c is not None]
        near_cols += fallback_cols_list
        near_ascending = [False] * len([c for c in [secondary_col] if c is not None])
        near_ascending += [True] * len([c for c in [primary_std_col, secondary_std_col] if c is not None])
        near_ascending += fallback_ascending_list

        far_cols = [c for c in [primary_col, secondary_col] if c is not None]
        far_cols += [c for c in [primary_std_col, secondary_std_col] if c is not None]
        far_cols += fallback_cols_list
        far_ascending = [False] * len([c for c in [primary_col, secondary_col] if c is not None])
        far_ascending += [True] * len([c for c in [primary_std_col, secondary_std_col] if c is not None])
        far_ascending += fallback_ascending_list

        near = work[work['selection_within_tie_margin']].sort_values(near_cols, ascending=near_ascending, na_position='last')
        far = work[~work['selection_within_tie_margin']].sort_values(far_cols, ascending=far_ascending, na_position='last')
        out = pd.concat([near, far], axis=0, ignore_index=True)
        out['selection_rank'] = np.arange(1, len(out) + 1, dtype=int)
        return out

    group_cols_list = _as_list(group_cols)
    if not group_cols_list:
        return _order_one(df)

    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(group_cols_list, dropna=False, sort=False):
        pieces.append(_order_one(group))
    out = pd.concat(pieces, axis=0, ignore_index=True)
    sort_cols = group_cols_list + ['selection_rank']
    sort_ascending = [True] * len(group_cols_list) + [True]
    out = out.sort_values(sort_cols, ascending=sort_ascending, na_position='last').reset_index(drop=True)
    return out


__all__ = [
    'ATOMIC_RECIPE_COMPLEXITY_RANK',
    'DEFAULT_PRIMARY_METRIC',
    'DEFAULT_SECONDARY_METRIC',
    'DEFAULT_TERTIARY_METRIC',
    'DEFAULT_TIE_MARGIN',
    'LOSS_COMPLEXITY_RANK',
    'QUESTION_COMPLEXITY_RANK',
    'add_representation_complexity_columns',
    'loss_complexity_rank',
    'order_candidates_with_tie_margin',
    'order_phase_tuning_configs',
    'question_complexity_rank',
    'recipe_complexity_columns',
    'recipe_complexity_tuple',
]
