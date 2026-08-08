"""Table S6, Figure S6 and Figure S7 from the allowance sensitivity sweep.

All three manuscript items read one aggregate table — 120 rows of
``(allow_rate, dataset, method, tier, recall@1, recall@5, recall@10, mrr@100)`` —
and differ only in which slice they take:

===========  ==========================================  =========================
item         methods                                     rows
===========  ==========================================  =========================
Table S6     Python approximation, CDE Match-Fuzzy,       72 = 3 x 4 datasets x
             fixed HGBC                                   6 rates
Figure S6    the same three                               3 curves x 4 panels
Figure S7    fixed HGBC vs retrained-per-rate HGBC        2 curves x 4 panels
===========  ==========================================  =========================

Everything reported is the ``primary`` tier: the four caDSR-derived evaluation
sets, the only ones the allowance mask applies to.

Migrated from ``K5_make_table.py`` / ``K6_make_tableS6_v19.py`` (Table S6),
``K4_make_figure.py`` and ``manuscript/figures/make_figureS6_S7_allowance.py``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from demap_repro.sensitivity.allowance.arms import (
    ALLOWANCE_RATES,
    ARM_FIXED,
    ARM_RETRAINED,
    FIGURE_S7_METHODS,
    OPERATING_RATE,
    PRIMARY_DATASETS,
    TABLE_S6_METHODS,
    TIER_PRIMARY,
)

__all__ = [
    "load_sensitivity", "primary_slice", "table_s6",
    "figure_s6_series", "figure_s7_series", "robustness_summary",
]

METRICS = ("recall@1", "recall@5", "recall@10", "mrr@100")
REQUIRED_COLUMNS = ("allow_rate", "dataset", "method", "tier", *METRICS)


def load_sensitivity(path) -> pd.DataFrame:
    """Read the aggregate sensitivity table and check its shape."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def primary_slice(df: pd.DataFrame, methods: Sequence[str],
                  datasets: Sequence[str] = PRIMARY_DATASETS) -> pd.DataFrame:
    """Primary-tier rows for the given methods and datasets, in report order.

    Restricting to ``primary`` is deliberate: the in-sample validation rows and the
    always-unmasked external rows are diagnostics, not evidence, and must never be
    averaged in with the reported curves.
    """
    sub = df[(df["tier"] == TIER_PRIMARY)
             & (df["method"].isin(methods))
             & (df["dataset"].isin(datasets))].copy()
    sub["_m"] = sub["method"].map({m: i for i, m in enumerate(methods)})
    sub["_d"] = sub["dataset"].map({d: i for i, d in enumerate(datasets)})
    return (sub.sort_values(["_m", "_d", "allow_rate"])
               .drop(columns=["_m", "_d"]).reset_index(drop=True))


def table_s6(df: pd.DataFrame) -> pd.DataFrame:
    """Table S6: three methods x four datasets x six rates, all four metrics."""
    return primary_slice(df, TABLE_S6_METHODS)[list(REQUIRED_COLUMNS)]


def figure_s6_series(df: pd.DataFrame, metric: str = "recall@5") -> Dict[str, Dict[str, List]]:
    """Figure S6 curves: ``{dataset: {method: [(rate, value), ...]}}``."""
    return _series(primary_slice(df, TABLE_S6_METHODS), metric)


def figure_s7_series(df: pd.DataFrame, metric: str = "recall@5") -> Dict[str, Dict[str, List]]:
    """Figure S7 curves: fixed-at-70% versus retrained-at-each-rate."""
    return _series(primary_slice(df, FIGURE_S7_METHODS), metric)


def _series(sub: pd.DataFrame, metric: str) -> Dict[str, Dict[str, List]]:
    out: Dict[str, Dict[str, List]] = {}
    for (dataset, method), g in sub.groupby(["dataset", "method"], sort=False):
        g = g.sort_values("allow_rate")
        out.setdefault(dataset, {})[method] = list(
            zip(g["allow_rate"].tolist(), g[metric].tolist()))
    return out


def robustness_summary(df: pd.DataFrame, metric: str = "recall@5") -> pd.DataFrame:
    """Per-method, per-dataset variation across the full allowance range.

    Backs the S6.2 claim that the fixed HGBC varies far less than either lexical
    method: for each series, the spread between its best and worst rate, its value
    at 0% allowance (no exact evidence at all), and its value at the 70% operating
    point.
    """
    rows = []
    sub = df[df["tier"] == TIER_PRIMARY]
    for (method, dataset), g in sub.groupby(["method", "dataset"]):
        g = g.sort_values("allow_rate")
        at = dict(zip(g["allow_rate"], g[metric]))
        rows.append({
            "method": method,
            "dataset": dataset,
            "metric": metric,
            "min": float(g[metric].min()),
            "max": float(g[metric].max()),
            "range": float(g[metric].max() - g[metric].min()),
            "at_0.0": float(at.get(0.0, float("nan"))),
            "at_operating_rate": float(at.get(OPERATING_RATE, float("nan"))),
            "n_rates": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values(["method", "dataset"]).reset_index(drop=True)
