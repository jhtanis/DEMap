#!/usr/bin/env python3
"""Production-CDE-Match-style CDE eligibility filter (``getDSFilterString``).

Production CDE Match restricts its candidate universe with ``getDSFilterString``
(scripts/cde_match/S74_NCI_DS.sql, ln 704-737), whose default branch applies:

    de.ADMIN_STUS_ID not in (77,81,95,96,80)
    and de.CNTXT_NM_DN not in ('TEST','Training')
    [and de.currnt_ver_ind = 1]

The five excluded ``ADMIN_STUS_ID`` values map (data/raw/cadsr_xml/mapping/
admin_statuses.xls, sheet ``od_001 (18)``) **exactly** to the five RETIRED-family
status names:

    77 = RETIRED ARCHIVED
    80 = RETIRED UNLOADED
    81 = RETIRED WITHDRAWN
    95 = RETIRED DELETED
    96 = RETIRED PHASED OUT

so the ID list is equivalent to the SQL's own commented universe condition
``upper(ADMIN_STUS_NM_DN) not like '%RETIRED%'`` (S74_NCI_DS.sql ln 541/551/...).
``cde_master_enriched.workflow_status`` IS that status name (sourced from the XML
``WORKFLOWSTATUS`` element, extract_cadsr_xml.py:377), so this filter reproduces the
SQL admin-status exclusion EXACTLY via a case-insensitive "contains RETIRED" test —
robust even if a future caDSR snapshot adds the other four retired names (only
``RETIRED ARCHIVED`` is present today).

NOT implemented here (separate, documented deviations):
  * strict current-version (``currnt_ver_ind = 1``) -- the enriched master carries no
    current-version indicator; the pipelines approximate it by keeping the highest
    version per publicid.
  * ``registration_status`` is NOT an eligibility field -- it is ``regstr_stus_id``,
    used only in CDE Match scoring/tie-breaking, never as a filter.

The filter is **default OFF**. Callers opt in with ``mode='production_cde_match'``;
``mode='none'`` returns the frame unchanged so existing artifacts are preserved.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

# ADMIN_STUS_ID -> status name, for the five IDs CDE Match excludes
# (data/raw/cadsr_xml/mapping/admin_statuses.xls).
EXCLUDED_ADMIN_STUS_ID: Dict[int, str] = {
    77: "RETIRED ARCHIVED",
    80: "RETIRED UNLOADED",
    81: "RETIRED WITHDRAWN",
    95: "RETIRED DELETED",
    96: "RETIRED PHASED OUT",
}

# The substring that identifies a RETIRED-family workflow status (case-insensitive).
RETIRED_TOKEN = "RETIRED"

# Contexts CDE Match removes (CNTXT_NM_DN not in ('TEST','Training')); compared upper-cased.
EXCLUDED_CONTEXTS: Tuple[str, ...] = ("TEST", "TRAINING")

ELIGIBILITY_MODES: Tuple[str, ...] = ("none", "production_cde_match")

DEFAULT_WORKFLOW_STATUS_COL = "workflow_status"
DEFAULT_CONTEXT_COL = "context_name"


def retired_mask(workflow_status: pd.Series) -> pd.Series:
    """Boolean mask: rows whose workflow status is RETIRED-family (contains 'RETIRED')."""
    return workflow_status.astype(str).str.upper().str.contains(RETIRED_TOKEN, na=False)


def excluded_context_mask(context_name: pd.Series) -> pd.Series:
    """Boolean mask: rows whose context is TEST or Training."""
    return context_name.astype(str).str.upper().isin(EXCLUDED_CONTEXTS)


def eligibility_mask(
    df: pd.DataFrame,
    *,
    workflow_status_col: str = DEFAULT_WORKFLOW_STATUS_COL,
    context_col: str = DEFAULT_CONTEXT_COL,
) -> pd.Series:
    """Boolean mask of production-CDE-Match-eligible rows.

    Eligible = NOT RETIRED-family workflow status AND context not in {TEST, Training}.
    """
    for col in (workflow_status_col, context_col):
        if col not in df.columns:
            raise KeyError(
                f"eligibility filter needs column '{col}'; present columns: "
                f"{list(df.columns)}"
            )
    retired = retired_mask(df[workflow_status_col])
    ctx = excluded_context_mask(df[context_col])
    return ~(retired | ctx)


def eligibility_report(
    df: pd.DataFrame,
    *,
    workflow_status_col: str = DEFAULT_WORKFLOW_STATUS_COL,
    context_col: str = DEFAULT_CONTEXT_COL,
) -> Dict[str, object]:
    """Before/after + removed-by-reason counts for the eligibility filter.

    ``removed_*_only`` / ``removed_both`` are a disjoint partition of ``n_removed``;
    ``removed_retired`` / ``removed_test_training`` are the (overlapping) per-reason
    totals.
    """
    retired = retired_mask(df[workflow_status_col])
    ctx = excluded_context_mask(df[context_col])
    removed = retired | ctx
    n = int(len(df))
    n_removed = int(removed.sum())
    excl_status: List[str] = sorted(
        {str(s) for s in df.loc[retired, workflow_status_col].astype(str).unique()}
    )
    excl_ctx: List[str] = sorted(
        {str(c) for c in df.loc[ctx, context_col].astype(str).unique()}
    )
    return {
        "n_before": n,
        "n_after": n - n_removed,
        "n_removed": n_removed,
        "removed_retired": int(retired.sum()),
        "removed_test_training": int(ctx.sum()),
        "removed_retired_only": int((retired & ~ctx).sum()),
        "removed_context_only": int((ctx & ~retired).sum()),
        "removed_both": int((retired & ctx).sum()),
        "workflow_status_values_excluded": excl_status,
        "context_values_excluded": excl_ctx,
        "excluded_admin_stus_id_map": dict(EXCLUDED_ADMIN_STUS_ID),
    }


def filter_eligible(
    df: pd.DataFrame,
    *,
    mode: str = "none",
    workflow_status_col: str = DEFAULT_WORKFLOW_STATUS_COL,
    context_col: str = DEFAULT_CONTEXT_COL,
    return_report: bool = False,
):
    """Apply the production-CDE-Match eligibility filter.

    ``mode='none'`` (DEFAULT) returns ``df`` unchanged (no-op; existing behavior).
    ``mode='production_cde_match'`` drops RETIRED-family + TEST/Training rows.

    Returns the (filtered) DataFrame, or ``(df, report)`` when ``return_report=True``.
    """
    if mode not in ELIGIBILITY_MODES:
        raise ValueError(f"unknown eligibility mode {mode!r}; expected one of {ELIGIBILITY_MODES}")

    if mode == "none":
        out = df
        report: Dict[str, object] = {
            "mode": "none",
            "n_before": int(len(df)),
            "n_after": int(len(df)),
            "n_removed": 0,
            "removed_retired": 0,
            "removed_test_training": 0,
            "removed_retired_only": 0,
            "removed_context_only": 0,
            "removed_both": 0,
            "workflow_status_values_excluded": [],
            "context_values_excluded": [],
        }
    else:
        mask = eligibility_mask(
            df, workflow_status_col=workflow_status_col, context_col=context_col
        )
        out = df[mask].copy()
        report = {"mode": mode, **eligibility_report(
            df, workflow_status_col=workflow_status_col, context_col=context_col)}

    return (out, report) if return_report else out


def format_report(report: Dict[str, object]) -> str:
    """One-line-per-fact human summary of an eligibility report (for CLI logs)."""
    if report.get("mode") == "none":
        return "eligibility=none (no filter applied; catalog unchanged)"
    lines = [
        f"eligibility={report['mode']}: "
        f"{report['n_before']} -> {report['n_after']} CDEs "
        f"(removed {report['n_removed']})",
        f"  removed_retired={report['removed_retired']} "
        f"removed_test_training={report['removed_test_training']}",
        f"  disjoint: retired_only={report['removed_retired_only']} "
        f"context_only={report['removed_context_only']} both={report['removed_both']}",
        f"  retired statuses excluded: {report['workflow_status_values_excluded']}",
        f"  contexts excluded: {report['context_values_excluded']}",
    ]
    return "\n".join(lines)
