"""Natural-language text features for the candidate-union/XGBoost stack (PR-D.2).

Derives text-surface features from query text, optional CDE master metadata,
and query × CDE pairs. All features are inference-computable and label-free.

Tiers:
  - Tier 1 (query-level): derived from ``query_text_q3``, ``pv_attached``,
    ``PV_BLOCK_SDE``, ``family``, ``query_source``. Always computed.
  - Tier 2 (candidate-CDE-level): derived from ``cde_master`` joined by
    ``cde_id``. Computed only when ``cde_master`` is provided.
  - Tier 3 (query × CDE pair): derived from Tier 1 + Tier 2 columns.
    Computed only when ``cde_master`` is provided.

NaN policy:
  - Missing/empty query text produces zero lengths, zero ratios, False
    booleans — never NaN.
  - Missing CDE master rows (cde_id not found) produce zero lengths, False
    booleans, zero overlap.
  - ``text_query_cde_char_len_ratio`` is NaN only when the CDE short-name
    length is zero (division by zero).

Leakage policy:
  - No ``is_label``, ``true_*``, ``gold_*`` columns are read.
  - CDE metadata comes from the public catalog (``cde_master_enriched``),
    keyed by each candidate's ``cde_id`` — not the gold CDE.
  - ``family`` and ``query_source`` are query provenance, known at inference.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from demap_repro.pool.candidate_union import FORBIDDEN_LEAKAGE_COLUMNS
from demap_repro.text.pv_parser import parse_pv_block


# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------

TEXT_FEATURE_COLUMNS_TIER1: Tuple[str, ...] = (
    "text_query_char_len",
    "text_query_word_count",
    "text_query_has_pv",
    "text_query_pv_n",
    "text_query_has_question_mark",
    "text_query_is_code_like",
    "text_query_upper_ratio",
    "text_query_digit_ratio",
    "text_query_underscore_count",
    "text_query_pipe_count",
    "text_family",
    "text_query_source",
)

TEXT_FEATURE_COLUMNS_TIER2: Tuple[str, ...] = (
    "text_cde_short_name_len",
    "text_cde_long_name_len",
    "text_cde_definition_len",
    "text_cde_has_pv",
    "text_cde_pv_n",
)

TEXT_FEATURE_COLUMNS_TIER3: Tuple[str, ...] = (
    "text_query_cde_char_len_ratio",
    "text_query_cde_word_overlap",
    "text_query_cde_pv_type_match",
)

REQUIRED_INPUT_COLUMNS: Tuple[str, ...] = (
    "query_id",
    "cde_id",
    "query_text_q3",
)


def text_feature_columns(
    include_tier2: bool = False,
    include_tier3: bool = False,
) -> Tuple[str, ...]:
    cols = list(TEXT_FEATURE_COLUMNS_TIER1)
    if include_tier2:
        cols.extend(TEXT_FEATURE_COLUMNS_TIER2)
    if include_tier3:
        cols.extend(TEXT_FEATURE_COLUMNS_TIER3)
    return tuple(cols)


__all__ = [
    "TEXT_FEATURE_COLUMNS_TIER1",
    "TEXT_FEATURE_COLUMNS_TIER2",
    "TEXT_FEATURE_COLUMNS_TIER3",
    "REQUIRED_INPUT_COLUMNS",
    "text_feature_columns",
    "compute_text_features",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PV_SEP_RE = re.compile(r"\s*\|\s*PV_TYPE:")
_CODE_LIKE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _extract_query_body(q3: str) -> str:
    """Strip the appended PV block from a Q3 text string."""
    parts = _PV_SEP_RE.split(q3, maxsplit=1)
    return parts[0].strip()


def _safe_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v).strip()


def _word_set(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _char_len(text: str) -> int:
    return len(text)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _is_code_like(text: str) -> bool:
    """Single token with underscores or camelCase, no spaces."""
    body = _extract_query_body(text)
    if not body or " " in body:
        return False
    return bool(_CODE_LIKE_RE.match(body)) and ("_" in body or (body != body.lower() and body != body.upper()))


def _upper_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)


def _pv_n_from_sde(pv_block_sde: str) -> int:
    parsed = parse_pv_block(pv_block_sde)
    if parsed.parsed_n is not None:
        return parsed.parsed_n
    return parsed.n_values


def _pv_type_from_sde(pv_block_sde: str) -> str:
    return parse_pv_block(pv_block_sde).pv_type


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_text_features(
    union_df: pd.DataFrame,
    *,
    cde_master: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Append text-derived feature columns to a candidate-union DataFrame.

    Args:
        union_df: Candidate-union DataFrame. Must contain at minimum
            ``query_id``, ``cde_id``, ``query_text_q3``.
        cde_master: Optional ``cde_master_enriched`` DataFrame with columns
            ``cde_id``, ``SHORT_NAME``, ``LONG_NAME``, ``DEFINITION``,
            ``PV_N``, ``PV_TYPE``. When provided, Tier 2 + Tier 3 features
            are emitted.

    Returns:
        A copy of ``union_df`` with feature columns appended.
    """
    include_master = cde_master is not None

    feature_cols = text_feature_columns(
        include_tier2=include_master,
        include_tier3=include_master,
    )

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in union_df.columns]
    if missing:
        raise ValueError(
            f"compute_text_features missing required input columns: {missing}"
        )

    already = [c for c in feature_cols if c in union_df.columns]
    if already:
        raise ValueError(
            f"compute_text_features: output columns already present: "
            f"{already}. Drop them first."
        )

    out = union_df.copy()

    # --- Tier 1: query-level features ---
    q3 = out["query_text_q3"].apply(_safe_str)
    body = q3.apply(_extract_query_body)

    out["text_query_char_len"] = body.apply(_char_len).astype(int)
    out["text_query_word_count"] = body.apply(_word_count).astype(int)

    if "pv_attached" in out.columns:
        out["text_query_has_pv"] = out["pv_attached"].fillna(False).astype(bool)
    else:
        out["text_query_has_pv"] = q3.str.contains(r"\| PV_TYPE:", regex=True, na=False)

    if "PV_BLOCK_SDE" in out.columns:
        pv_sde = out["PV_BLOCK_SDE"].apply(_safe_str)
        out["text_query_pv_n"] = pv_sde.apply(_pv_n_from_sde).astype("Int64")
    else:
        out["text_query_pv_n"] = pd.array([0] * len(out), dtype="Int64")

    out["text_query_has_question_mark"] = body.str.contains(r"\?", regex=True, na=False)
    out["text_query_is_code_like"] = q3.apply(_is_code_like)
    out["text_query_upper_ratio"] = body.apply(_upper_ratio).astype(float)
    out["text_query_digit_ratio"] = body.apply(_digit_ratio).astype(float)
    out["text_query_underscore_count"] = body.str.count("_").fillna(0).astype(int)
    out["text_query_pipe_count"] = q3.str.count(r"\|").fillna(0).astype(int)

    out["text_family"] = (
        out["family"].apply(_safe_str) if "family" in out.columns
        else ""
    )
    out["text_query_source"] = (
        out["query_source"].apply(_safe_str) if "query_source" in out.columns
        else ""
    )

    # --- Tier 2 + 3: CDE master join ---
    if include_master:
        master_cols = ["cde_id"]
        for c in ("SHORT_NAME", "LONG_NAME", "DEFINITION", "PV_N", "PV_TYPE"):
            if c in cde_master.columns:
                master_cols.append(c)
        m = cde_master[master_cols].drop_duplicates(subset=["cde_id"]).copy()
        m["cde_id"] = m["cde_id"].astype(str)

        out["cde_id"] = out["cde_id"].astype(str)
        out = out.merge(
            m.rename(columns=lambda c: f"_m_{c}" if c != "cde_id" else c),
            on="cde_id",
            how="left",
        )

        def _master_str(col: str) -> pd.Series:
            mc = f"_m_{col}"
            if mc in out.columns:
                return out[mc].apply(_safe_str)
            return pd.Series("", index=out.index)

        sn = _master_str("SHORT_NAME")
        ln = _master_str("LONG_NAME")
        defn = _master_str("DEFINITION")

        out["text_cde_short_name_len"] = sn.apply(_char_len).astype(int)
        out["text_cde_long_name_len"] = ln.apply(_char_len).astype(int)
        out["text_cde_definition_len"] = defn.apply(_char_len).astype(int)

        if "_m_PV_N" in out.columns:
            out["text_cde_has_pv"] = pd.to_numeric(out["_m_PV_N"], errors="coerce").fillna(0).astype(int) > 0
            out["text_cde_pv_n"] = pd.to_numeric(out["_m_PV_N"], errors="coerce").fillna(0).astype("Int64")
        else:
            out["text_cde_has_pv"] = False
            out["text_cde_pv_n"] = pd.array([0] * len(out), dtype="Int64")

        # Tier 3: pair features
        sn_len = out["text_cde_short_name_len"].astype(float)
        out["text_query_cde_char_len_ratio"] = np.where(
            sn_len > 0,
            out["text_query_char_len"].astype(float) / sn_len,
            np.nan,
        )

        body_words = body.apply(_word_set)
        sn_words = sn.apply(_word_set)
        out["text_query_cde_word_overlap"] = [
            _jaccard(bw, sw) for bw, sw in zip(body_words, sn_words)
        ]

        if "_m_PV_TYPE" in out.columns and "PV_BLOCK_SDE" in union_df.columns:
            query_pv_type = out["PV_BLOCK_SDE"].apply(_safe_str).apply(_pv_type_from_sde)
            cde_pv_type = out["_m_PV_TYPE"].apply(_safe_str).str.upper()
            out["text_query_cde_pv_type_match"] = (
                (query_pv_type != "") & (cde_pv_type != "") & (query_pv_type == cde_pv_type)
            )
        else:
            out["text_query_cde_pv_type_match"] = False

        helper_cols = [c for c in out.columns if c.startswith("_m_")]
        out = out.drop(columns=helper_cols)

    # --- Final column order ---
    new_cols = list(feature_cols)
    final_cols = [c for c in union_df.columns if c not in new_cols] + new_cols
    out = out[final_cols]

    bad = sorted(set(out.columns) & FORBIDDEN_LEAKAGE_COLUMNS)
    if bad:
        raise RuntimeError(
            f"compute_text_features produced forbidden leakage columns: {bad}"
        )

    return out
