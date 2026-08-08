"""demap_repro.text.pv_summary

Deterministic construction of *permissible value (PV) summaries*.

This module is used in two places:

1) **Target-side (CDE) enrichment** via :mod:`demap_repro.data.enrich_master`.
2) **Query-side (SDE proxy) augmentation** via :mod:`demap_repro.data.queries`.

Design constraints
------------------
- Deterministic (no randomness).
- Conservative normalization (trim/collapse whitespace, Unicode quotes/dashes,
  NFKC); **no lowercasing**.
- No synthetic PV creation; we only re-order / select from observed PV rows.
- Cap the number of rendered PV examples (default 10) to avoid overly long
  representations.

Terminology
-----------
caDSR PV extracts typically provide both:

- ``valid_value``: a code-ish token (e.g., ``Y``, ``N``, ``1``)
- ``value_meaning``: a label/meaning (e.g., ``Yes``, ``No``, ``Mild``)

We render:

- **CDE-side PV examples** from ``value_meaning`` (formal).
- **SDE-side PV examples** from ``valid_value`` (code-ish), with an optional
  deterministic mixture for *generic* concepts (e.g., sometimes render "Yes"
  instead of "Y") controlled by ``sde_generic_label_p``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from demap_repro.text.normalize import normalize_query_text


# -----------------------------------------------------------------------------
# Generic concept detection (for SDE label/code mixing)
# -----------------------------------------------------------------------------


def _cf(s: str) -> str:
    return normalize_query_text(s).casefold()


# A small, publishable set of "generic" meanings.
#
# NOTE: this list is only used for *classification*; output text preserves
# original casing (post-safe-normalization) and we never lowercase rendered PVs.
_GENERIC_MEANINGS: Dict[str, Sequence[str]] = {
    "UNKNOWN": ("unknown", "unk", "not known", "don't know", "do not know"),
    "NA": ("not applicable", "n/a", "na", "notapplicable"),
    "NOT_REPORTED": ("not reported", "notreported"),
    "NOT_TESTED": ("not tested", "nottested"),
    "YES": ("yes", "true"),
    "NO": ("no", "false"),
    "NORMAL": ("normal",),
    "ABNORMAL": ("abnormal",),
    "POSITIVE": ("positive", "pos"),
    "NEGATIVE": ("negative", "neg"),
    "PRESENT": ("present",),
    "ABSENT": ("absent",),
}


_GENERIC_LOOKUP: Dict[str, str] = {}
for _concept, _vals in _GENERIC_MEANINGS.items():
    for _v in _vals:
        _GENERIC_LOOKUP[_v] = _concept


# Priority order for generic concepts when rendering examples.
_GENERIC_PRIORITY: List[str] = [
    "UNKNOWN",
    "NA",
    "NOT_REPORTED",
    "NOT_TESTED",
    "YES",
    "NO",
    "NORMAL",
    "ABNORMAL",
    "POSITIVE",
    "NEGATIVE",
    "PRESENT",
    "ABSENT",
]


def generic_concept(value_meaning: str) -> Optional[str]:
    """Return a canonical generic concept name for a meaning string, else None."""
    m = _cf(value_meaning)
    return _GENERIC_LOOKUP.get(m)


# -----------------------------------------------------------------------------
# PV type signatures
# -----------------------------------------------------------------------------


_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+)(?:\.\d+)?$")


def _dedupe_preserve_order(xs: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def pv_type_signature(value_meanings: Sequence[str]) -> str:
    """Compute a compact, deterministic type signature.

    The signature is intentionally coarse; it is meant to provide a small
    structural hint (binary vs ordinal vs large enum), not to perfectly model
    every caDSR value set.
    """

    vals = _dedupe_preserve_order([_cf(v) for v in value_meanings if normalize_query_text(v)])
    s = set(vals)
    if not s:
        return "EMPTY"

    # Binary patterns
    yes_set = {"yes", "true"}
    no_set = {"no", "false"}
    unknown_set = {"unknown", "unk", "not known", "don't know", "do not know"}
    na_set = {"not applicable", "n/a", "na", "notapplicable"}
    not_rep_set = {"not reported", "notreported"}
    not_test_set = {"not tested", "nottested"}
    allowed_binary = yes_set | no_set | unknown_set | na_set | not_rep_set | not_test_set

    has_yes = any(v in yes_set for v in s)
    has_no = any(v in no_set for v in s)
    if has_yes and has_no and s.issubset(allowed_binary):
        extras: List[str] = []
        if any(v in unknown_set for v in s):
            extras.append("UNKNOWN")
        if any(v in na_set for v in s):
            extras.append("NA")
        if any(v in not_rep_set for v in s):
            extras.append("NOT_REPORTED")
        if any(v in not_test_set for v in s):
            extras.append("NOT_TESTED")
        if not extras:
            return "BINARY"
        return "BINARY_WITH_" + "_".join(extras)

    # Likert (common sets)
    likert_freq = {"never", "rarely", "sometimes", "often", "always"}
    if s == likert_freq:
        return "LIKERT_5_FREQUENCY"

    likert_agree = {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"}
    if s == likert_agree:
        return "LIKERT_5_AGREEMENT"

    # Numeric ordinal (e.g., 0..4, 1..5)
    if all(_NUMERIC_RE.fullmatch(v) for v in s):
        # try integer ordinal
        try:
            ints = sorted({int(float(v)) for v in s})
            if len(ints) == len(s):
                # contiguous
                if ints and all(ints[i] + 1 == ints[i + 1] for i in range(len(ints) - 1)):
                    return f"ORDINAL_INT_{ints[0]}_{ints[-1]}"
        except Exception:
            pass

    return "ENUM"


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def _sha1_int(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)


def _sde_use_label(*, cde_publicid: str, cde_version: str, concept: str, p: int, salt: str) -> bool:
    """Deterministically choose whether SDE renders a generic concept as a label."""
    p2 = int(p)
    if p2 <= 0:
        return False
    if p2 >= 100:
        return True
    key = f"{cde_publicid}::{cde_version}::{concept}::{salt}"
    return (_sha1_int(key) % 100) < p2


def _evenly_spaced(items: Sequence[str], k: int) -> List[str]:
    """Deterministically pick k items spanning the list, preserving order."""
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[0]]
    step = float(len(items)) / float(k)
    idxs: List[int] = []
    for i in range(k):
        idx = int(i * step)
        if idx >= len(items):
            idx = len(items) - 1
        idxs.append(idx)
    # Ensure strictly increasing indices (rare rounding collisions).
    out: List[str] = []
    used = set()
    for idx in idxs:
        j = idx
        while j in used and j + 1 < len(items):
            j += 1
        used.add(j)
        out.append(items[j])
    return out


def _format_block(sig: str, pv_n: int, examples: Sequence[str]) -> str:
    head = f"PV_TYPE: {sig}(n={int(pv_n)})"
    ex = [normalize_query_text(x) for x in examples if normalize_query_text(x)]
    ex = _dedupe_preserve_order(ex)
    if not ex:
        return head
    return head + "; PV: " + " | ".join(ex)


def build_pv_summary_with_diagnostics(
    pv: pd.DataFrame,
    *,
    profile: str = "cadsr",
    pv_max_n: int = 10,
    pv_huge_threshold: int = 20,
    sde_generic_label_p: float = 0.30,
    salt: str = "demap",
    sep: str = " | ",
    pv_max_n_query: Optional[int] = None,
    pv_max_n_cde: Optional[int] = None,
    pv_center_fraction: float = 0.5,
    pv_min_n: int = 2,
    pv_size_jitter: int = 1,
    pv_generic_boost: float = 0.10,
    pv_generic_cap: float = 0.70,
    pv_small_n_generic_threshold: float = 0.5,
    pv_max_resample_attempts: int = 3,
    pv_placeholder_token: str = "<MISSING_PV_SUMMARY>",
    use_placeholder_for_query_omission: bool = False,
    diagnostics_summary_path: str | Path | None = None,
    diagnostics_records_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from demap_repro.text.pv_summary_v2 import build_pv_summary_tables

    summary_df, diagnostics_df = build_pv_summary_tables(
        pv,
        profile=str(profile),
        salt=str(salt),
        sep=str(sep),
        pv_max_n=int(pv_max_n),
        pv_huge_threshold=int(pv_huge_threshold),
        sde_generic_label_p=sde_generic_label_p,
        pv_max_n_query=pv_max_n_query,
        pv_max_n_cde=pv_max_n_cde,
        pv_center_fraction=float(pv_center_fraction),
        pv_min_n=int(pv_min_n),
        pv_size_jitter=int(pv_size_jitter),
        pv_generic_boost=float(pv_generic_boost),
        pv_generic_cap=float(pv_generic_cap),
        pv_small_n_generic_threshold=float(pv_small_n_generic_threshold),
        pv_max_resample_attempts=int(pv_max_resample_attempts),
        pv_placeholder_token=str(pv_placeholder_token),
        use_placeholder_for_query_omission=bool(use_placeholder_for_query_omission),
        diagnostics_summary_path=diagnostics_summary_path,
        diagnostics_records_path=diagnostics_records_path,
    )
    return summary_df, diagnostics_df


def build_pv_summary_table(
    pv: pd.DataFrame,
    *,
    profile: str = "cadsr",
    pv_max_n: int = 10,
    pv_huge_threshold: int = 20,
    sde_generic_label_p: float = 0.30,
    salt: str = "demap",
    sep: str = " | ",
    pv_max_n_query: Optional[int] = None,
    pv_max_n_cde: Optional[int] = None,
    pv_center_fraction: float = 0.5,
    pv_min_n: int = 2,
    pv_size_jitter: int = 1,
    pv_generic_boost: float = 0.10,
    pv_generic_cap: float = 0.70,
    pv_small_n_generic_threshold: float = 0.5,
    pv_max_resample_attempts: int = 3,
    pv_placeholder_token: str = "<MISSING_PV_SUMMARY>",
    use_placeholder_for_query_omission: bool = False,
    diagnostics_summary_path: str | Path | None = None,
    diagnostics_records_path: str | Path | None = None,
) -> pd.DataFrame:
    """Aggregate row-level PVs into a per-CDE PV summary table.

    The public adapter and returned schema are intentionally stable. The new
    conservative dual-summary logic lives behind this function in
    :mod:`demap_repro.text.pv_summary_v2`.

    Legacy parameters such as ``pv_max_n`` and ``pv_huge_threshold`` are still
    accepted for compatibility. The new split caps are authoritative when
    supplied; otherwise the adapter defaults to a slightly sparser query side
    and a canonical CDE side.
    """

    summary_df, _diagnostics_df = build_pv_summary_with_diagnostics(
        pv,
        profile=profile,
        pv_max_n=pv_max_n,
        pv_huge_threshold=pv_huge_threshold,
        sde_generic_label_p=sde_generic_label_p,
        salt=salt,
        sep=sep,
        pv_max_n_query=pv_max_n_query,
        pv_max_n_cde=pv_max_n_cde,
        pv_center_fraction=pv_center_fraction,
        pv_min_n=pv_min_n,
        pv_size_jitter=pv_size_jitter,
        pv_generic_boost=pv_generic_boost,
        pv_generic_cap=pv_generic_cap,
        pv_small_n_generic_threshold=pv_small_n_generic_threshold,
        pv_max_resample_attempts=pv_max_resample_attempts,
        pv_placeholder_token=pv_placeholder_token,
        use_placeholder_for_query_omission=use_placeholder_for_query_omission,
        diagnostics_summary_path=diagnostics_summary_path,
        diagnostics_records_path=diagnostics_records_path,
    )
    return summary_df
