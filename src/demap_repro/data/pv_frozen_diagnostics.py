from __future__ import annotations

"""PV-only diagnostics from frozen paired query/CDE artifacts.

This module prefers the *paired frozen benchmark* artifact (typically
``data/processed/pairs.parquet``) as the source of truth for what downstream
retrieval models actually saw on the PV side.

Primary input:
- pairs.parquet (query-side PV block + paired gold-CDE PV block)

Secondary validation input:
- cde_master_enriched.parquet (catalog-side CDE PV summary)

The analysis does **not** recompute PV summaries. It describes the PV text already
materialized in the frozen artifacts and provides:
- query-side PV summary stats
- paired CDE-side PV summary stats
- catalog CDE PV summary stats
- query↔paired-CDE overlap stats
- integrity/consistency checks against the catalog CDE PV summary
- a normalized CSV export of paired PV fields
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from demap_repro.text.normalize import normalize_query_text
from demap_repro.text.pv_summary import generic_concept


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_PV_HEAD_RE = re.compile(
    r"^\s*PV_TYPE:\s*(?P<pv_type>.+?)\(n=(?P<declared_n>\d+)\)\s*(?:;\s*PV:\s*(?P<items>.*))?$"
)
_INPUT_QUERY_TEXT_CANDIDATES: Tuple[str, ...] = (
    "query_text_raw",
    "query_text_q3",
    "query_text",
    "query_text_q4",
)


def _norm_text(x: Any) -> str:
    return normalize_query_text(x)


def _norm_cf(x: Any) -> str:
    return _norm_text(x).casefold()


def _tokenize(x: Any) -> List[str]:
    s = _norm_cf(x)
    if not s:
        return []
    return _WORD_RE.findall(s)


@dataclass(frozen=True)
class ParsedPVBlock:
    raw_text: str
    norm_text: str
    norm_cf: str
    present: bool
    placeholder: bool
    parse_error: bool
    pv_type: str
    declared_n: Optional[int]
    items: Tuple[str, ...]
    items_cf: Tuple[str, ...]
    item_count: int
    char_len: int
    token_count: int
    generic_item_count: int
    code_like_item_count: int



def _is_code_like_item(item: str) -> bool:
    s = str(item or "").strip()
    if not s:
        return False
    if generic_concept(s) is not None:
        return False
    if any(ch.isspace() for ch in s):
        return False
    if any(ch.isdigit() for ch in s):
        return True
    if any(ch in "_.-:/" for ch in s):
        return True
    if len(s) <= 3:
        return True
    if s.isupper():
        return True
    return False



def _is_missing_scalar(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False



def _has_nonempty_text(x: Any) -> bool:
    if _is_missing_scalar(x):
        return False
    return bool(str(x).strip())



def _first_existing(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None



def _split_versioned_cde_id(x: Any) -> Tuple[str, str]:
    if _is_missing_scalar(x):
        return "", ""
    s = str(x).strip()
    if not s:
        return "", ""
    if "::" not in s:
        return s, ""
    left, right = s.split("::", 1)
    return left.strip(), right.strip()



def _versioned_cde_id_from_frame(df: pd.DataFrame) -> pd.Series:
    if "cde_id" in df.columns:
        s = df["cde_id"].astype(str)
        has_version = s.str.contains(r"::", regex=True, na=False)
        if has_version.any():
            return s
    if "cde_publicid" in df.columns and "cde_version" in df.columns:
        left = df["cde_publicid"].astype(str)
        right = df["cde_version"].astype(str)
        return left + "::" + right
    if "cde_id" in df.columns:
        return df["cde_id"].astype(str)
    raise ValueError("Input rows must contain either versioned cde_id or cde_publicid + cde_version.")



def parse_pv_block(block: Any, *, placeholder_token: str = "<MISSING_PV_SUMMARY>") -> ParsedPVBlock:
    raw = "" if _is_missing_scalar(block) else str(block).strip()
    norm = _norm_text(raw)
    norm_cf = norm.casefold()
    placeholder_norm = _norm_text(placeholder_token)
    placeholder = bool(norm and norm == placeholder_norm)
    if not norm or placeholder:
        return ParsedPVBlock(
            raw_text=raw,
            norm_text=norm,
            norm_cf=norm_cf,
            present=False,
            placeholder=placeholder,
            parse_error=False,
            pv_type="",
            declared_n=None,
            items=tuple(),
            items_cf=tuple(),
            item_count=0,
            char_len=0,
            token_count=0,
            generic_item_count=0,
            code_like_item_count=0,
        )

    m = _PV_HEAD_RE.match(raw)
    parse_error = m is None
    pv_type = ""
    declared_n: Optional[int] = None
    items: List[str] = []
    if m is not None:
        pv_type = str(m.group("pv_type") or "").strip()
        try:
            declared_n = int(m.group("declared_n"))
        except Exception:
            declared_n = None
        items_blob = str(m.group("items") or "").strip()
        if items_blob:
            items = [_norm_text(part) for part in items_blob.split(" | ")]
            items = [item for item in items if item]

    items_cf = tuple(_norm_cf(x) for x in items)
    generic_item_count = sum(1 for item in items if generic_concept(item) is not None)
    code_like_item_count = sum(1 for item in items if _is_code_like_item(item))
    return ParsedPVBlock(
        raw_text=raw,
        norm_text=norm,
        norm_cf=norm_cf,
        present=True,
        placeholder=False,
        parse_error=parse_error,
        pv_type=pv_type,
        declared_n=declared_n,
        items=tuple(items),
        items_cf=items_cf,
        item_count=len(items),
        char_len=len(norm),
        token_count=len(_tokenize(norm)),
        generic_item_count=generic_item_count,
        code_like_item_count=code_like_item_count,
    )



def _parsed_records_to_frame(records: Sequence[ParsedPVBlock]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        rows.append(
            {
                "pv_present": bool(rec.present),
                "pv_placeholder": bool(rec.placeholder),
                "pv_parse_error": bool(rec.parse_error),
                "pv_type": rec.pv_type,
                "pv_declared_n": rec.declared_n,
                "pv_item_count": int(rec.item_count),
                "pv_char_len": int(rec.char_len),
                "pv_token_count": int(rec.token_count),
                "pv_generic_item_count": int(rec.generic_item_count),
                "pv_code_like_item_count": int(rec.code_like_item_count),
                "pv_generic_item_fraction": float(rec.generic_item_count / rec.item_count) if rec.item_count > 0 else float("nan"),
                "pv_code_like_item_fraction": float(rec.code_like_item_count / rec.item_count) if rec.item_count > 0 else float("nan"),
                "pv_items": list(rec.items),
                "pv_items_cf": list(rec.items_cf),
                "pv_block_text": rec.raw_text,
                "pv_block_text_norm": rec.norm_text,
                "pv_block_text_cf": rec.norm_cf,
            }
        )
    return pd.DataFrame(rows)



def _pct(numer: float, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return float(numer) / float(denom)



def _safe_numeric_summary(values: Sequence[float], prefix: str) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_n": int(arr.size),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(arr.max()),
    }



def summarize_parsed_blocks(df: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    if df.empty:
        out: Dict[str, Any] = {
            "scope": scope,
            "n_rows": 0,
            "pv_present_n": 0,
            "pv_present_rate": 0.0,
            "pv_missing_n": 0,
            "pv_missing_rate": 0.0,
            "pv_placeholder_n": 0,
            "pv_placeholder_rate": 0.0,
            "pv_parse_error_n": 0,
            "pv_parse_error_rate": 0.0,
            "pv_any_generic_item_rate": 0.0,
            "pv_any_code_like_item_rate": 0.0,
        }
        out.update(_safe_numeric_summary([], "pv_item_count_present"))
        out.update(_safe_numeric_summary([], "pv_char_len_present"))
        out.update(_safe_numeric_summary([], "pv_token_count_present"))
        out.update(_safe_numeric_summary([], "pv_generic_item_fraction_present"))
        out.update(_safe_numeric_summary([], "pv_code_like_item_fraction_present"))
        out.update(_safe_numeric_summary([], "pv_declared_n_present"))
        return pd.DataFrame([out])

    total = int(len(df))
    present = df[df["pv_present"]].copy()

    out: Dict[str, Any] = {
        "scope": scope,
        "n_rows": total,
        "pv_present_n": int(df["pv_present"].sum()),
        "pv_present_rate": _pct(float(df["pv_present"].sum()), float(total)),
        "pv_missing_n": int((~df["pv_present"]).sum()),
        "pv_missing_rate": _pct(float((~df["pv_present"]).sum()), float(total)),
        "pv_placeholder_n": int(df["pv_placeholder"].sum()),
        "pv_placeholder_rate": _pct(float(df["pv_placeholder"].sum()), float(total)),
        "pv_parse_error_n": int(df["pv_parse_error"].sum()),
        "pv_parse_error_rate": _pct(float(df["pv_parse_error"].sum()), float(total)),
        "pv_any_generic_item_rate": _pct(float((df["pv_generic_item_count"] > 0).sum()), float(total)),
        "pv_any_code_like_item_rate": _pct(float((df["pv_code_like_item_count"] > 0).sum()), float(total)),
    }
    if not present.empty:
        out.update(_safe_numeric_summary(present["pv_item_count"].tolist(), "pv_item_count_present"))
        out.update(_safe_numeric_summary(present["pv_char_len"].tolist(), "pv_char_len_present"))
        out.update(_safe_numeric_summary(present["pv_token_count"].tolist(), "pv_token_count_present"))
        out.update(_safe_numeric_summary(present["pv_generic_item_fraction"].tolist(), "pv_generic_item_fraction_present"))
        out.update(_safe_numeric_summary(present["pv_code_like_item_fraction"].tolist(), "pv_code_like_item_fraction_present"))
        declared_nonnull = present["pv_declared_n"].dropna().tolist()
        out.update(_safe_numeric_summary(declared_nonnull, "pv_declared_n_present"))
    else:
        out.update(_safe_numeric_summary([], "pv_item_count_present"))
        out.update(_safe_numeric_summary([], "pv_char_len_present"))
        out.update(_safe_numeric_summary([], "pv_token_count_present"))
        out.update(_safe_numeric_summary([], "pv_generic_item_fraction_present"))
        out.update(_safe_numeric_summary([], "pv_code_like_item_fraction_present"))
        out.update(_safe_numeric_summary([], "pv_declared_n_present"))
    return pd.DataFrame([out])



def _set_jaccard(xs: Sequence[str], ys: Sequence[str]) -> float:
    sx = set(x for x in xs if x)
    sy = set(y for y in ys if y)
    union = sx | sy
    if not union:
        return float("nan")
    return float(len(sx & sy)) / float(len(union))



def _token_jaccard_from_items(xs: Sequence[str], ys: Sequence[str]) -> float:
    x_tokens: List[str] = []
    y_tokens: List[str] = []
    for x in xs:
        x_tokens.extend(_tokenize(x))
    for y in ys:
        y_tokens.extend(_tokenize(y))
    return _set_jaccard(x_tokens, y_tokens)



def _proper_subset(xs: Sequence[str], ys: Sequence[str]) -> bool:
    sx = {x for x in xs if x}
    sy = {y for y in ys if y}
    return bool(sx) and sx < sy



def _subset_or_equal(xs: Sequence[str], ys: Sequence[str]) -> bool:
    sx = {x for x in xs if x}
    sy = {y for y in ys if y}
    return bool(sx) and sx.issubset(sy)



def _prepare_catalog_cdes(cdes: pd.DataFrame, *, cde_pv_col: str) -> pd.DataFrame:
    if cde_pv_col not in cdes.columns:
        raise ValueError(f"cde parquet missing required PV column: {cde_pv_col}")
    cat = cdes.copy()
    cat["cde_id_versioned"] = _versioned_cde_id_from_frame(cat)
    if "cde_publicid" in cat.columns:
        cat["cde_id"] = cat["cde_publicid"].astype(str)
    else:
        cat["cde_id"] = cat["cde_id_versioned"].map(lambda x: _split_versioned_cde_id(x)[0])
    keep = [col for col in ["cde_id", "cde_id_versioned", cde_pv_col] if col in cat.columns]
    cat = cat[keep].copy()
    cat = cat.rename(columns={cde_pv_col: "master_cde_pv_summary"})
    cat["cde_id"] = cat["cde_id"].astype(str)
    cat["cde_id_versioned"] = cat["cde_id_versioned"].astype(str)
    cat = cat.drop_duplicates(subset=["cde_id_versioned"], keep="first")
    return cat



def _prepare_paired_rows(
    rows: pd.DataFrame,
    *,
    query_pv_col: str,
    pair_cde_pv_col: Optional[str],
) -> pd.DataFrame:
    if query_pv_col not in rows.columns:
        raise ValueError(f"paired artifact missing required query PV column: {query_pv_col}")

    text_col = _first_existing(rows, _INPUT_QUERY_TEXT_CANDIDATES)
    if text_col is None:
        raise ValueError(
            "paired artifact missing a usable query text column; expected one of "
            f"{list(_INPUT_QUERY_TEXT_CANDIDATES)}"
        )
    if "query_id" not in rows.columns:
        raise ValueError("paired artifact missing required column: 'query_id'")

    out = rows.copy()
    out["query_id"] = out["query_id"].astype(str)
    out["query_text"] = out[text_col].fillna("").astype(str)
    out["query_pv_summary"] = out[query_pv_col]
    out["cde_id_versioned"] = _versioned_cde_id_from_frame(out)
    if "cde_publicid" in out.columns:
        out["cde_id"] = out["cde_publicid"].astype(str)
    else:
        out["cde_id"] = out["cde_id_versioned"].map(lambda x: _split_versioned_cde_id(x)[0])

    if pair_cde_pv_col and pair_cde_pv_col in out.columns:
        out["cde_pv_summary"] = out[pair_cde_pv_col]
    else:
        out["cde_pv_summary"] = ""

    keep = [
        col
        for col in [
            "pair_id",
            "query_id",
            "query_source",
            "query_field",
            "query_text",
            "query_pv_summary",
            "cde_id",
            "cde_id_versioned",
            "cde_pv_summary",
        ]
        if col in out.columns
    ]
    return out[keep].copy()



def _build_export_and_pair_records(
    paired_rows: pd.DataFrame,
    catalog_cdes: pd.DataFrame,
    *,
    placeholder_token: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    export = paired_rows.merge(
        catalog_cdes[["cde_id_versioned", "master_cde_pv_summary"]],
        on="cde_id_versioned",
        how="left",
        validate="m:1",
        indicator="_catalog_merge",
    )
    export["cde_master_row_found"] = export["_catalog_merge"].eq("both")
    export = export.drop(columns=["_catalog_merge"])

    q_parsed = [parse_pv_block(x, placeholder_token=placeholder_token) for x in export["query_pv_summary"].tolist()]
    c_pair_parsed = [parse_pv_block(x, placeholder_token=placeholder_token) for x in export["cde_pv_summary"].tolist()]
    c_master_parsed = [parse_pv_block(x, placeholder_token=placeholder_token) for x in export["master_cde_pv_summary"].tolist()]

    q_df = _parsed_records_to_frame(q_parsed).add_prefix("query_")
    c_pair_df = _parsed_records_to_frame(c_pair_parsed).add_prefix("cde_")
    c_master_df = _parsed_records_to_frame(c_master_parsed).add_prefix("master_cde_")

    pair = pd.concat(
        [
            export.reset_index(drop=True),
            q_df.reset_index(drop=True),
            c_pair_df.reset_index(drop=True),
            c_master_df.reset_index(drop=True),
        ],
        axis=1,
    )
    pair["both_present"] = pair["query_pv_present"] & pair["cde_pv_present"]
    pair["query_absent_cde_present"] = (~pair["query_pv_present"]) & pair["cde_pv_present"]
    pair["exact_block_match"] = pair["both_present"] & (pair["query_pv_block_text"] == pair["cde_pv_block_text"])
    pair["normalized_block_match"] = pair["both_present"] & (pair["query_pv_block_text_cf"] == pair["cde_pv_block_text_cf"])
    pair["exact_item_match"] = pair["both_present"] & (pair["query_pv_items"] == pair["cde_pv_items"])
    pair["normalized_item_match"] = pair["both_present"] & (pair["query_pv_items_cf"] == pair["cde_pv_items_cf"])

    item_jaccards: List[float] = []
    token_jaccards: List[float] = []
    strict_subsets: List[bool] = []
    subset_or_equals: List[bool] = []
    exact_overlap_counts: List[int] = []
    query_shorter: List[bool] = []
    for q_items, c_items, q_n, c_n in zip(
        pair["query_pv_items_cf"].tolist(),
        pair["cde_pv_items_cf"].tolist(),
        pair["query_pv_item_count"].tolist(),
        pair["cde_pv_item_count"].tolist(),
    ):
        q_seq = list(q_items) if isinstance(q_items, (list, tuple)) else []
        c_seq = list(c_items) if isinstance(c_items, (list, tuple)) else []
        item_jaccards.append(_set_jaccard(q_seq, c_seq))
        token_jaccards.append(_token_jaccard_from_items(q_seq, c_seq))
        strict_subsets.append(_proper_subset(q_seq, c_seq))
        subset_or_equals.append(_subset_or_equal(q_seq, c_seq))
        exact_overlap_counts.append(int(len(set(q_seq) & set(c_seq))))
        try:
            query_shorter.append(int(q_n) < int(c_n))
        except Exception:
            query_shorter.append(False)

    pair["item_jaccard"] = item_jaccards
    pair["token_jaccard"] = token_jaccards
    pair["strict_subset"] = strict_subsets
    pair["subset_or_equal"] = subset_or_equals
    pair["exact_overlap_count"] = exact_overlap_counts
    pair["query_shorter_than_cde"] = query_shorter

    pair["pair_cde_matches_master"] = (
        pair["cde_pv_present"]
        & pair["master_cde_pv_present"]
        & (pair["cde_pv_block_text_cf"] == pair["master_cde_pv_block_text_cf"])
    )
    pair["pair_cde_missing_master_present"] = (~pair["cde_pv_present"]) & pair["master_cde_pv_present"]
    pair["pair_cde_present_master_missing"] = pair["cde_pv_present"] & (~pair["master_cde_pv_present"])
    return export, pair



def summarize_pairwise_overlap(pair: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    total = int(len(pair))
    both = pair[pair["both_present"]].copy()
    out: Dict[str, Any] = {
        "scope": scope,
        "n_pairs": total,
        "both_present_n": int(pair["both_present"].sum()),
        "both_present_rate": _pct(float(pair["both_present"].sum()), float(total)),
        "query_absent_cde_present_n": int(pair["query_absent_cde_present"].sum()),
        "query_absent_cde_present_rate": _pct(float(pair["query_absent_cde_present"].sum()), float(total)),
        "cde_master_row_found_n": int(pair["cde_master_row_found"].sum()),
        "cde_master_row_found_rate": _pct(float(pair["cde_master_row_found"].sum()), float(total)),
        "pair_cde_matches_master_n": int(pair["pair_cde_matches_master"].sum()),
        "pair_cde_matches_master_rate": _pct(float(pair["pair_cde_matches_master"].sum()), float(total)),
        "pair_cde_missing_master_present_n": int(pair["pair_cde_missing_master_present"].sum()),
        "pair_cde_missing_master_present_rate": _pct(float(pair["pair_cde_missing_master_present"].sum()), float(total)),
        "pair_cde_present_master_missing_n": int(pair["pair_cde_present_master_missing"].sum()),
        "pair_cde_present_master_missing_rate": _pct(float(pair["pair_cde_present_master_missing"].sum()), float(total)),
    }
    if not both.empty:
        denom = float(len(both))
        out.update(
            {
                "exact_block_match_n": int(both["exact_block_match"].sum()),
                "exact_block_match_rate": _pct(float(both["exact_block_match"].sum()), denom),
                "normalized_block_match_n": int(both["normalized_block_match"].sum()),
                "normalized_block_match_rate": _pct(float(both["normalized_block_match"].sum()), denom),
                "exact_item_match_n": int(both["exact_item_match"].sum()),
                "exact_item_match_rate": _pct(float(both["exact_item_match"].sum()), denom),
                "normalized_item_match_n": int(both["normalized_item_match"].sum()),
                "normalized_item_match_rate": _pct(float(both["normalized_item_match"].sum()), denom),
                "strict_subset_n": int(both["strict_subset"].sum()),
                "strict_subset_rate": _pct(float(both["strict_subset"].sum()), denom),
                "subset_or_equal_n": int(both["subset_or_equal"].sum()),
                "subset_or_equal_rate": _pct(float(both["subset_or_equal"].sum()), denom),
                "query_shorter_than_cde_n": int(both["query_shorter_than_cde"].sum()),
                "query_shorter_than_cde_rate": _pct(float(both["query_shorter_than_cde"].sum()), denom),
            }
        )
        out.update(_safe_numeric_summary(both["exact_overlap_count"].tolist(), "exact_overlap_count"))
        out.update(_safe_numeric_summary(both["item_jaccard"].tolist(), "item_jaccard"))
        out.update(_safe_numeric_summary(both["token_jaccard"].tolist(), "token_jaccard"))
    else:
        out.update(
            {
                "exact_block_match_n": 0,
                "exact_block_match_rate": float("nan"),
                "normalized_block_match_n": 0,
                "normalized_block_match_rate": float("nan"),
                "exact_item_match_n": 0,
                "exact_item_match_rate": float("nan"),
                "normalized_item_match_n": 0,
                "normalized_item_match_rate": float("nan"),
                "strict_subset_n": 0,
                "strict_subset_rate": float("nan"),
                "subset_or_equal_n": 0,
                "subset_or_equal_rate": float("nan"),
                "query_shorter_than_cde_n": 0,
                "query_shorter_than_cde_rate": float("nan"),
            }
        )
        out.update(_safe_numeric_summary([], "exact_overlap_count"))
        out.update(_safe_numeric_summary([], "item_jaccard"))
        out.update(_safe_numeric_summary([], "token_jaccard"))
    return pd.DataFrame([out])



def summarize_integrity_checks(export: pd.DataFrame, pair: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    total = int(len(export))
    pair_id_series = export["pair_id"].astype(str) if "pair_id" in export.columns else pd.Series([], dtype=str)
    query_id_series = export["query_id"].astype(str) if "query_id" in export.columns else pd.Series([], dtype=str)
    comparable = pair[pair["cde_pv_present"] & pair["master_cde_pv_present"]].copy()
    comparable_n = int(len(comparable))
    out: Dict[str, Any] = {
        "scope": scope,
        "n_rows": total,
        "n_unique_query_id": int(query_id_series.nunique()) if not query_id_series.empty else 0,
        "duplicate_query_id_n": int(query_id_series.duplicated().sum()) if not query_id_series.empty else 0,
        "missing_query_id_n": int((query_id_series.str.strip() == "").sum()) if not query_id_series.empty else 0,
        "n_unique_pair_id": int(pair_id_series.nunique()) if not pair_id_series.empty else 0,
        "duplicate_pair_id_n": int(pair_id_series.duplicated().sum()) if not pair_id_series.empty else 0,
        "missing_pair_id_n": int((pair_id_series.str.strip() == "").sum()) if not pair_id_series.empty else 0,
        "missing_cde_id_n": int(export["cde_id"].astype(str).str.strip().eq("").sum()),
        "missing_cde_id_versioned_n": int(export["cde_id_versioned"].astype(str).str.strip().eq("").sum()),
        "cde_master_row_found_n": int(export["cde_master_row_found"].sum()),
        "cde_master_row_found_rate": _pct(float(export["cde_master_row_found"].sum()), float(total)),
        "pair_cde_present_n": int(pair["cde_pv_present"].sum()),
        "pair_cde_present_rate": _pct(float(pair["cde_pv_present"].sum()), float(total)),
        "master_cde_present_n": int(pair["master_cde_pv_present"].sum()),
        "master_cde_present_rate": _pct(float(pair["master_cde_pv_present"].sum()), float(total)),
        "pair_cde_matches_master_n": int(pair["pair_cde_matches_master"].sum()),
        "pair_cde_matches_master_rate_all_rows": _pct(float(pair["pair_cde_matches_master"].sum()), float(total)),
        "pair_cde_matches_master_rate_comparable": _pct(float(pair["pair_cde_matches_master"].sum()), float(comparable_n)),
        "pair_cde_missing_master_present_n": int(pair["pair_cde_missing_master_present"].sum()),
        "pair_cde_missing_master_present_rate": _pct(float(pair["pair_cde_missing_master_present"].sum()), float(total)),
        "pair_cde_present_master_missing_n": int(pair["pair_cde_present_master_missing"].sum()),
        "pair_cde_present_master_missing_rate": _pct(float(pair["pair_cde_present_master_missing"].sum()), float(total)),
    }
    return pd.DataFrame([out])



def compute_frozen_pv_diagnostics_from_pairs(
    pairs: pd.DataFrame,
    cdes: pd.DataFrame,
    *,
    query_pv_col: str = "PV_BLOCK_SDE",
    pair_cde_pv_col: Optional[str] = "PV_BLOCK_CDE",
    cde_pv_col: str = "PV_SUMMARY",
    placeholder_token: str = "<MISSING_PV_SUMMARY>",
) -> Dict[str, pd.DataFrame]:
    paired_rows = _prepare_paired_rows(pairs, query_pv_col=query_pv_col, pair_cde_pv_col=pair_cde_pv_col)
    catalog_cdes = _prepare_catalog_cdes(cdes, cde_pv_col=cde_pv_col)
    export, pair_records = _build_export_and_pair_records(
        paired_rows,
        catalog_cdes,
        placeholder_token=placeholder_token,
    )

    query_parsed = _parsed_records_to_frame(
        [parse_pv_block(x, placeholder_token=placeholder_token) for x in export["query_pv_summary"].tolist()]
    )
    query_summary = summarize_parsed_blocks(query_parsed, scope="query_rows")

    matched_cde_parsed = _parsed_records_to_frame(
        [parse_pv_block(x, placeholder_token=placeholder_token) for x in export["cde_pv_summary"].tolist()]
    )
    matched_cde_summary = summarize_parsed_blocks(matched_cde_parsed, scope="matched_cde_rows")

    catalog_parsed = _parsed_records_to_frame(
        [parse_pv_block(x, placeholder_token=placeholder_token) for x in catalog_cdes["master_cde_pv_summary"].tolist()]
    )
    catalog_cde_summary = summarize_parsed_blocks(catalog_parsed, scope="catalog_cdes")

    overlap_summary = summarize_pairwise_overlap(pair_records, scope="query_rows_joined_to_gold_cde")
    integrity_checks = summarize_integrity_checks(export, pair_records, scope="paired_rows_vs_catalog")

    export_csv = export[[
        col for col in [
            "query_id",
            "query_text",
            "query_pv_summary",
            "cde_id",
            "cde_id_versioned",
            "cde_pv_summary",
        ] if col in export.columns
    ]].copy()

    return {
        "paired_pv_export": export_csv,
        "query_summary": query_summary,
        "matched_cde_summary": matched_cde_summary,
        "catalog_cde_summary": catalog_cde_summary,
        "pair_overlap_summary": overlap_summary,
        "integrity_checks": integrity_checks,
        "pair_records": pair_records,
    }



def _records_preview(pair: pd.DataFrame, *, n_each: int = 10) -> pd.DataFrame:
    if pair.empty:
        return pair.head(0).copy()
    blocks: List[pd.DataFrame] = []
    sortable = pair.copy()
    for col in ["token_jaccard", "item_jaccard"]:
        if col not in sortable.columns:
            sortable[col] = float("nan")
    blocks.append(sortable.sort_values(["token_jaccard", "item_jaccard"], ascending=[False, False]).head(n_each))
    exact = pair[pair.get("normalized_item_match", False)].head(n_each)
    if not exact.empty:
        blocks.append(exact)
    subset = pair[pair.get("strict_subset", False)].sort_values(["token_jaccard", "item_jaccard"], ascending=[False, False]).head(n_each)
    if not subset.empty:
        blocks.append(subset)
    absent = pair[pair.get("query_absent_cde_present", False)].head(n_each)
    if not absent.empty:
        blocks.append(absent)
    mismatch = pair[pair.get("pair_cde_missing_master_present", False)].head(n_each)
    if not mismatch.empty:
        blocks.append(mismatch)
    out = pd.concat(blocks, ignore_index=True).drop_duplicates(subset=[c for c in ["query_id", "cde_id_versioned"] if c in pair.columns])
    keep_cols = [
        c
        for c in [
            "pair_id",
            "query_id",
            "query_source",
            "query_text",
            "cde_id",
            "cde_id_versioned",
            "query_pv_block_text",
            "cde_pv_block_text",
            "master_cde_pv_block_text",
            "both_present",
            "query_absent_cde_present",
            "normalized_block_match",
            "normalized_item_match",
            "strict_subset",
            "query_shorter_than_cde",
            "pair_cde_matches_master",
            "pair_cde_missing_master_present",
            "exact_overlap_count",
            "item_jaccard",
            "token_jaccard",
        ]
        if c in out.columns
    ]
    return out[keep_cols].copy()



def _frame_to_serializable_dict(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) == 1:
        return df.iloc[0].to_dict()
    return {"rows": df.to_dict(orient="records")}



def _load_row_artifact(*, pairs_parquet: Optional[str], queries_parquet: Optional[str]) -> Tuple[pd.DataFrame, str, Path]:
    if pairs_parquet:
        p = Path(pairs_parquet)
        if p.exists():
            return pd.read_parquet(p), "pairs", p
    if queries_parquet:
        q = Path(queries_parquet)
        if q.exists():
            return pd.read_parquet(q), "queries", q
    raise SystemExit(
        "No usable row-level artifact found. Provide --pairs-parquet (preferred) or --queries-parquet."
    )



def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Compute PV-only diagnostics from frozen paired query/CDE artifacts (prefers pairs.parquet)."
    )
    ap.add_argument(
        "--pairs-parquet",
        default=os.path.join("data", "processed", "pairs.parquet"),
        help="Frozen pairs parquet path (preferred source of truth).",
    )
    ap.add_argument(
        "--queries-parquet",
        default=os.path.join("data", "processed", "queries.parquet"),
        help="Legacy fallback row artifact path used only when pairs.parquet is unavailable.",
    )
    ap.add_argument(
        "--cde-parquet",
        default=os.path.join("data", "processed", "cde_master_enriched.parquet"),
        help="Frozen enriched CDE parquet path.",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join("artifacts", "summaries", "pv_frozen_diagnostics"),
        help="Directory for summary outputs.",
    )
    ap.add_argument(
        "--query-pv-col",
        default="PV_BLOCK_SDE",
        help="Query-side PV block column in the row-level artifact.",
    )
    ap.add_argument(
        "--query-cde-pv-col",
        default="PV_BLOCK_CDE",
        help="Paired CDE-side PV block column in the row-level artifact.",
    )
    ap.add_argument(
        "--cde-pv-col",
        default="PV_SUMMARY",
        help="Catalog CDE-side PV summary column in cde_master_enriched.parquet.",
    )
    ap.add_argument(
        "--placeholder-token",
        default="<MISSING_PV_SUMMARY>",
        help="Placeholder token used to represent omitted query PV summaries.",
    )
    ap.add_argument("--no-pair-records", action="store_true", help="Do not write the full per-query pair-records CSV.")
    ap.add_argument("--preview-n", type=int, default=10, help="Rows per category to include in the preview CSV.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    rows, source_name, source_path = _load_row_artifact(
        pairs_parquet=args.pairs_parquet,
        queries_parquet=args.queries_parquet,
    )
    cdes = pd.read_parquet(args.cde_parquet)
    outputs = compute_frozen_pv_diagnostics_from_pairs(
        rows,
        cdes,
        query_pv_col=args.query_pv_col,
        pair_cde_pv_col=args.query_cde_pv_col if args.query_cde_pv_col else None,
        cde_pv_col=args.cde_pv_col,
        placeholder_token=args.placeholder_token,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_export_path = out_dir / "paired_pv_export.csv"
    query_summary_path = out_dir / "query_pv_summary.csv"
    matched_cde_summary_path = out_dir / "matched_cde_pv_summary.csv"
    catalog_cde_summary_path = out_dir / "catalog_cde_pv_summary.csv"
    overlap_summary_path = out_dir / "query_cde_pv_overlap_summary.csv"
    integrity_path = out_dir / "pv_integrity_checks.csv"
    pair_preview_path = out_dir / "query_cde_pv_pair_preview.csv"
    pair_records_path = out_dir / "query_cde_pv_pair_records.csv"
    json_path = out_dir / "pv_frozen_diagnostics.json"

    outputs["paired_pv_export"].to_csv(paired_export_path, index=False)
    outputs["query_summary"].to_csv(query_summary_path, index=False)
    outputs["matched_cde_summary"].to_csv(matched_cde_summary_path, index=False)
    outputs["catalog_cde_summary"].to_csv(catalog_cde_summary_path, index=False)
    outputs["pair_overlap_summary"].to_csv(overlap_summary_path, index=False)
    outputs["integrity_checks"].to_csv(integrity_path, index=False)
    _records_preview(outputs["pair_records"], n_each=int(args.preview_n)).to_csv(pair_preview_path, index=False)
    if not bool(args.no_pair_records):
        outputs["pair_records"].to_csv(pair_records_path, index=False)

    payload = {
        "row_artifact_source": source_name,
        "row_artifact_path": str(source_path),
        "paired_pv_export": _frame_to_serializable_dict(outputs["paired_pv_export"].head(10)),
        "query_summary": _frame_to_serializable_dict(outputs["query_summary"]),
        "matched_cde_summary": _frame_to_serializable_dict(outputs["matched_cde_summary"]),
        "catalog_cde_summary": _frame_to_serializable_dict(outputs["catalog_cde_summary"]),
        "pair_overlap_summary": _frame_to_serializable_dict(outputs["pair_overlap_summary"]),
        "integrity_checks": _frame_to_serializable_dict(outputs["integrity_checks"]),
        "paths": {
            "paired_export_csv": str(paired_export_path),
            "query_summary_csv": str(query_summary_path),
            "matched_cde_summary_csv": str(matched_cde_summary_path),
            "catalog_cde_summary_csv": str(catalog_cde_summary_path),
            "pair_overlap_summary_csv": str(overlap_summary_path),
            "integrity_checks_csv": str(integrity_path),
            "pair_preview_csv": str(pair_preview_path),
            "pair_records_csv": str(pair_records_path) if not bool(args.no_pair_records) else "",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[pv-frozen-diagnostics] using {source_name} artifact: {source_path}")
    print("Wrote:", paired_export_path)
    print("Wrote:", query_summary_path)
    print("Wrote:", matched_cde_summary_path)
    print("Wrote:", catalog_cde_summary_path)
    print("Wrote:", overlap_summary_path)
    print("Wrote:", integrity_path)
    print("Wrote:", pair_preview_path)
    if not bool(args.no_pair_records):
        print("Wrote:", pair_records_path)
    print("Wrote:", json_path)


if __name__ == "__main__":  # pragma: no cover
    main()
