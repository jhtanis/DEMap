#!/usr/bin/env python
"""Analyze 'in-the-wild' SDE-like query instances from merged caDSR Parquet extracts.

Inputs (merged Parquet tables):
  - cde_alternate_names.parquet
  - cde_reference_documents.parquet

Outputs (CSV + JSON):
  - inwild_summary.json: headline totals
  - audit_filter_stages.csv: row/CDE counts after each filter stage
  - alt_candidate_inventory.csv: candidate ALT buckets for curator review (pre-allowlist)
  - top10_excluded_alt_buckets.csv: top excluded ALT buckets (pre-allowlist minus allowlist+safety)
  - alt_allowlist_unmatched.csv: allowlist rows with zero matches in ALT candidates
  - alt_rule_excluded_summary.csv: summary of ALT rows excluded by hard rules + not-allowlisted
  - alt_rule_excluded_samples.csv: stratified samples (by bucket) of excluded ALT rows
  - alt_dedupe_report.csv: within-CDE ALT de-duplication report (normalized-text key, provenance preserved)
  - refdoc_type_inventory.csv: refdoc DocumentType counts (pre-allowlist)
  - top10_excluded_refdoc_types.csv: top excluded refdoc DocumentTypes (by count)
  - refdoc_allowlist_unmatched.csv: allowlist DocumentTypes with zero matches
  - ref_rule_excluded_summary.csv: summary of REF rows excluded by hard rules + DocumentType allowlist
  - ref_rule_excluded_samples.csv: stratified samples (by bucket) of excluded REF rows
  - ref_dedupe_report.csv: within-CDE REF de-duplication report (normalized-text key, provenance preserved)
  - alt_inwild_by_bucket.csv: alternate-name query stats by (type, context)
  - alt_inwild_by_family.csv: alternate-name query stats by family (COG, CDISC, ...)
  - alt_inwild_by_context.csv: alternate-name stats by context (org/study)
  - ref_inwild_by_bucket.csv: refdoc query stats by (document_type, name)
  - ref_inwild_by_family.csv: refdoc query stats by family (COG, CRF, ...)
  - ref_inwild_by_name.csv: refdoc stats by name (useful for debugging)
  - holdout_candidates_alt_families.csv / holdout_candidates_ref_families.csv

Design notes
- "Query->CDE pair" means one SDE-like query instance mapped to one target CDE.
  This is NOT the cross-product of alternate names and question texts within a CDE.
- We use strict inclusion rules to maximize precision.
- Families are *provenance groupings* to support held-out external validation.

Example:
  demap build-queries \
    --alt-parquet data/interim/cadsr_merged/cde_alternate_names.parquet \
    --ref-parquet data/interim/cadsr_merged/cde_reference_documents.parquet \
    --out-dir artifacts/summaries/inwild_strict \
    --out-parquet data/processed/queries.parquet

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from demap_repro.text.normalize import normalize_query_text
from demap_repro.text.pv_summary import build_pv_summary_table


# -----------------------------------------------------------------------------
# Strict inclusion rules
# -----------------------------------------------------------------------------

# Alternate-name allowlist.
#
# This is a strict, curator-editable CSV of exact (alternate_name_type, context_name)
# pairs, with a provenance family label.
#
# See:
#   - configs/allowlists/alt_allowlist_current.csv
#   - configs/allowlists/alt_allowlist_vYYYYMMDD.csv
DEFAULT_ALT_ALLOWLIST = str((Path(__file__).resolve().parents[3] / "configs" / "allowlists" / "alt_allowlist_current.csv"))


# Alternate-name recipe allowlist (concatenation rules).
#
# This is a strict, curator-editable CSV describing how to build composite ALT
# queries (e.g., Table + Name) while preserving fidelity to caDSR source
# provenance: concatenation only occurs when both components appear within the
# same CDE's AlternateNameList.
#
# See:
#   - configs/allowlists/alt_query_recipes_current.csv
#   - configs/allowlists/alt_query_recipes_vYYYYMMDD.csv
DEFAULT_ALT_RECIPE_ALLOWLIST = str((Path(__file__).resolve().parents[3] / "configs" / "allowlists" / "alt_query_recipes_current.csv"))


# Reference-document allowlist.
#
# This is a strict, curator-editable CSV of literal DocumentType strings.
#
# See:
#   - configs/allowlists/refdoc_allowlist_current.csv
#   - configs/allowlists/refdoc_allowlist_vYYYYMMDD.csv
DEFAULT_REFDOC_ALLOWLIST = str((Path(__file__).resolve().parents[3] / "configs" / "allowlists" / "refdoc_allowlist_current.csv"))


# Alternate-name hard excludes (extra safety; you are primarily including by the allowlist).
ALT_EXCLUDE_TYPE_PREFIXES = (
    "UML ",
)
ALT_EXCLUDE_TYPE_EXACT = {
    "USED_BY",
    "HISTORICAL_CDE_ID",
    "GME_XMLLocReference",
    "HCT_BRIDG",
}
ALT_EXCLUDE_TYPE_REGEX = [
    re.compile(r"^Map:", re.IGNORECASE),
]


# NOTE: DocumentType allowlisting is now driven by the CSV allowlist above.

# Reference-document include Name patterns (strict, high-signal).
# These are matched case-insensitively AFTER whitespace normalization.
REFDOC_NAME_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Generic CRF text
    (re.compile(r"^(CRF\s*Text(\s*\d+)?|CRF\s*Text\d+)$", re.IGNORECASE), "CRF"),
    # COG CRF text
    (re.compile(r"^COG\s*CRF\s*TEXT\s*\d+$", re.IGNORECASE), "COG"),
    (re.compile(r"^COG\s*CRF\s*Text\s*\d+$", re.IGNORECASE), "COG"),
    # Theradex variants
    (re.compile(r"^Theradex\s*-\s*\d+$", re.IGNORECASE), "THERADEX"),
    # CTEP
    (re.compile(r"^CTEP\s*Text\s*\d+$", re.IGNORECASE), "CTEP"),
    # CCTG
    (re.compile(r"^CCTG[_-]?\d+$", re.IGNORECASE), "CCTG"),
    # DCP
    (re.compile(r"^DCP[-\s]?Text[-\s]?\d+$", re.IGNORECASE), "DCP"),
    (re.compile(r"^DCP-Text-\d+$", re.IGNORECASE), "DCP"),
    # NMDP
    (re.compile(r"^NMDP\s*Text\s*\d+$", re.IGNORECASE), "NMDP"),
    # CDASH
    (re.compile(r"^CDASH\s*PROMPT$", re.IGNORECASE), "CDASH"),
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _norm_str(x) -> str:
    """Normalize strings for robust matching.

    - Convert None -> ""
    - Replace NBSP with space
    - Strip ends
    - Collapse internal whitespace to a single space
    """
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\u00a0", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _sha1_hex(s: str) -> str:
    """Stable SHA1 hex digest for IDs.

    We avoid Python's built-in hash() because it is intentionally randomized between
    interpreter sessions (unless PYTHONHASHSEED is fixed).
    """
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _is_nonempty(x) -> bool:
    return _norm_str(x) != ""


def _cde_key(df: pd.DataFrame) -> pd.Series:
    return df["cde_publicid"].astype(str) + "::" + df["cde_version"].astype(str)


def _type_is_excluded(alt_type: str) -> bool:
    t = _norm_str(alt_type)
    if any(t.startswith(pfx) for pfx in ALT_EXCLUDE_TYPE_PREFIXES):
        return True
    if t in ALT_EXCLUDE_TYPE_EXACT:
        return True
    for rgx in ALT_EXCLUDE_TYPE_REGEX:
        if rgx.search(t):
            return True
    return False


def _assign_ref_family(name: str) -> str:
    n = _norm_str(name)
    for pat, fam in REFDOC_NAME_PATTERNS:
        if pat.match(n):
            return fam
    return "OTHER"


def _load_alt_allowlist_csv(path: str) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], str]]:
    """Load strict ALT allowlist CSV.

    The allowlist is **exact-only and literal**:

    - No wildcards: both ``alternate_name_type`` and ``context_name`` are required.
    - Case-sensitive: ``cog`` and ``COG`` are treated as different buckets.
    - Whitespace-sensitive: leading/trailing spaces are treated as meaningful characters.

    Notes
    -----
    We *do* normalize the ALT source tables (via ``_norm_str``) before matching. For
    reproducibility, curators should copy/paste bucket labels from the
    ``alt_candidate_inventory.csv`` produced by ``demap build-queries``.

    Returns
    -------
    allow_df
        Allowlist dataframe (string columns), as read from CSV (no normalization for matching).
    bucket_to_family
        Mapping (alternate_name_type, context_name) -> family.
    """
    p = os.path.expanduser(path)

    # Allow both repo-root relative paths ("configs/...") and absolute paths.
    # If the provided path doesn't exist as-is, also try interpreting it relative
    # to the repo root inferred from this module's location.
    if not os.path.exists(p):
        alt_p = str((Path(__file__).resolve().parents[3] / p))
        if os.path.exists(alt_p):
            p = alt_p
        else:
            raise FileNotFoundError(
                f"ALT allowlist CSV not found: {path}. "
                f"Expected by default at: {DEFAULT_ALT_ALLOWLIST}"
            )

    allow_df = pd.read_csv(p, dtype=str).fillna("")

    expected = {"alternate_name_type", "context_name", "family"}
    missing = sorted(list(expected - set(allow_df.columns)))
    if missing:
        raise ValueError(
            f"ALT allowlist CSV missing required columns: {missing}. "
            f"Found columns: {sorted(list(allow_df.columns))}"
        )

    # Ensure string dtype (but do NOT normalize values for matching).
    for c in ["alternate_name_type", "context_name", "family"]:
        allow_df[c] = allow_df[c].astype(str)

    # Strict validation: exact-only requires the fields to be present and non-blank.
    # (Validation uses .strip() only to detect blank fields; matching remains literal.)
    def _is_blank(v: Any) -> bool:
        s = "" if v is None else str(v)
        s2 = s.strip()
        return (s2 == "") or (s2.casefold() == "nan")

    if len(allow_df) > 0:
        bad = allow_df[
            allow_df["alternate_name_type"].map(_is_blank)
            | allow_df["context_name"].map(_is_blank)
            | allow_df["family"].map(_is_blank)
        ]
        if not bad.empty:
            preview = bad.head(5).copy()
            preview.insert(0, "row", preview.index + 2)  # +2 accounts for header and 0-index
            raise ValueError(
                "ALT allowlist CSV contains blank required fields (exact-only requires all 3). "
                f"First offending rows:\n{preview.to_string(index=False)}"
            )

    bucket_to_family: Dict[Tuple[str, str], str] = {}
    for _i, r in allow_df.iterrows():
        t = str(r["alternate_name_type"])
        c = str(r["context_name"])
        fam = str(r["family"])
        k = (t, c)
        if k in bucket_to_family and bucket_to_family[k] != fam:
            raise ValueError(f"ALT allowlist has conflicting families for {k}: '{bucket_to_family[k]}' vs '{fam}'")
        bucket_to_family[k] = fam

    return allow_df, bucket_to_family

def _load_alt_recipe_allowlist_csv(path: str) -> pd.DataFrame:
    """Load strict ALT recipe allowlist CSV (for composite/concatenated ALT queries).

    The recipe allowlist is **exact-only and literal** (like the bucket allowlist):

    - No wildcards.
    - Case-sensitive ("cog" and "COG" are different).
    - Whitespace-sensitive (leading/trailing spaces are meaningful characters).

    Expected columns (required)
    --------------------------
    - recipe_id: stable identifier for the recipe (used in synthetic bucket labels)
    - context_name: caDSR context_name (literal-exact match)
    - alt_type_a: component A alternate_name_type (typically Table/Domain)
    - alt_type_b: component B alternate_name_type (typically Name/Variable)
    - joiner: string used to join A and B (e.g. ".")
    - family: provenance family label to assign to the generated composite queries

    Optional columns
    ----------------
    - max_a: per-CDE cap on number of A values to include (default comes from CLI)
    - max_b: per-CDE cap on number of B values to include (default comes from CLI)
    - enabled: if present and falsey (0/false/False), the row is ignored

    Notes
    -----
    Recipes preserve fidelity to source provenance: we only concatenate A+B when
    both appear for the *same* (cde_publicid, cde_version) within the caDSR
    AlternateNameList.
    """
    p = os.path.expanduser(path)

    # Allow both repo-root relative paths ("configs/...") and absolute paths.
    if not os.path.exists(p):
        alt_p = str((Path(__file__).resolve().parents[3] / p))
        if os.path.exists(alt_p):
            p = alt_p
        else:
            raise FileNotFoundError(
                f"ALT recipe allowlist CSV not found: {path}. "
                f"Expected by default at: {DEFAULT_ALT_RECIPE_ALLOWLIST}"
            )

    df = pd.read_csv(p, dtype=str).fillna("")

    required = {"recipe_id", "context_name", "alt_type_a", "alt_type_b", "joiner", "family"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(
            f"ALT recipe allowlist CSV missing required columns: {missing}. "
            f"Found columns: {sorted(list(df.columns))}"
        )

    for c in sorted(list(required | {"max_a", "max_b", "enabled"} & set(df.columns))):
        df[c] = df[c].astype(str)

    # Filter enabled if present
    if "enabled" in df.columns:
        def _is_enabled(v: str) -> bool:
            s = "" if v is None else str(v).strip().casefold()
            if s in ("", "1", "true", "yes", "y"):
                return True
            if s in ("0", "false", "no", "n"):
                return False
            # Default: treat unknown as enabled, but be conservative in logging.
            return True

        df = df[df["enabled"].map(_is_enabled)].copy()

    # Validation: required fields must be non-blank.
    def _is_blank(v: Any) -> bool:
        s = "" if v is None else str(v)
        s2 = s.strip()
        return (s2 == "") or (s2.casefold() == "nan")

    if len(df) > 0:
        bad = df[
            df["recipe_id"].map(_is_blank)
            | df["context_name"].map(_is_blank)
            | df["alt_type_a"].map(_is_blank)
            | df["alt_type_b"].map(_is_blank)
            | df["joiner"].map(_is_blank)
            | df["family"].map(_is_blank)
        ].copy()
        if len(bad) > 0:
            raise ValueError(
                "ALT recipe allowlist CSV contains blank required fields. "
                f"Bad rows: {len(bad)}. "
                "(Exact-only matching requires literal strings.)"
            )

    # recipe_id must be unique
    if len(df) > 0 and df["recipe_id"].duplicated().any():
        dups = df[df["recipe_id"].duplicated()]["recipe_id"].tolist()
        raise ValueError(f"ALT recipe allowlist has duplicate recipe_id values: {sorted(set(dups))}")

    return df


def _table_domain_quality_score(alt_type: str, value_norm: str) -> Tuple[int, str]:
    """Heuristic, publishable quality score for Table/Domain-like tokens.

    This is used ONLY to deterministically choose which A-values to keep when
    a CDE has multiple candidates and max_a < n_candidates.

    Score meanings:
      2 = strongly matches expected convention for that family
      1 = weak/partial match
      0 = does not look like a table/domain token

    Important: this function must be deterministic and must NOT lowercase.
    It assumes `value_norm` has already been safe-normalized (NFKC + whitespace
    + common quotes/dashes) via normalize_query_text.
    """
    t = str(alt_type or "").casefold()
    v = str(value_norm or "")

    # CDISC SDTM domains are typically 2 letters (AE, DM, LB...), sometimes 3-4.
    if "sdtm" in t and "domain" in t:
        if re.fullmatch(r"[A-Z]{2,4}", v):
            return 2, "matches SDTM domain pattern: [A-Z]{2,4}"
        if re.fullmatch(r"[A-Za-z]{2,6}", v) and v.isupper():
            return 1, "all-caps alpha token (weak SDTM-domain-like)"
        return 0, "not SDTM-domain-like"

    # OMOP table names are uppercase snake_case (e.g., CONDITION_OCCURRENCE).
    if "omop" in t and "table" in t:
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", v) and ("_" in v):
            return 2, "matches OMOP table pattern: uppercase snake_case"
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", v):
            return 1, "all-caps token (weak OMOP-table-like)"
        return 0, "not OMOP-table-like"

    # PCORnet table names are typically uppercase (often with underscores).
    if "pcornet" in t and "table" in t:
        if " " in v:
            return 0, "contains spaces (not PCORnet-table-like)"
        if re.fullmatch(r"[A-Z][A-Z0-9_()]*", v) and ("_" in v or "(" in v or ")" in v):
            return 2, "matches PCORnet table pattern (uppercase token with separators)"
        if re.fullmatch(r"[A-Z][A-Z0-9_()]*", v):
            return 1, "uppercase token (weak PCORnet-table-like)"
        return 0, "not PCORnet-table-like"

    # FHIR US Core resource/table tokens are typically CamelCase with no spaces.
    if "fhir" in t and "table" in t:
        if " " in v:
            return 0, "contains spaces (not FHIR-resource-like)"
        if re.fullmatch(r"[A-Z][A-Za-z0-9]*", v):
            return 2, "matches FHIR resource token pattern: CamelCase"
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", v):
            return 1, "alphanumeric token (weak FHIR-resource-like)"
        return 0, "not FHIR-resource-like"

    # CCDI table tokens tend to be lower snake_case in practice.
    if "ccdi" in t and "table" in t:
        if re.fullmatch(r"[a-z][a-z0-9_]*", v) and ("_" in v):
            return 2, "matches CCDI table pattern: lower snake_case"
        if re.fullmatch(r"[a-z][a-z0-9_]*", v):
            return 1, "lowercase token (weak CCDI-table-like)"
        return 0, "not CCDI-table-like"

    # Generic fallback (should be rare).
    if " " not in v and re.fullmatch(r"[A-Za-z0-9_\.:-]+", v):
        return 1, "generic token-like (fallback)"
    return 0, "not token-like"


def _compute_global_norm_counts(df: pd.DataFrame, *, alt_types: List[str]) -> Dict[Tuple[str, str], int]:
    """Compute global counts of safe-normalized alternate_name values by alt_type.

    Returns
    -------
    Dict[(alt_type, value_norm)] -> count
    """
    if df is None or df.empty:
        return {}

    d = df[df["alternate_name_type"].isin(list(alt_types))].copy()
    if d.empty:
        return {}

    d["_norm"] = d["alternate_name"].fillna("").astype(str).map(normalize_query_text)
    g = d.groupby(["alternate_name_type", "_norm"], dropna=False).size().reset_index(name="n")
    out: Dict[Tuple[str, str], int] = {}
    for _, r in g.iterrows():
        out[(str(r["alternate_name_type"]), str(r["_norm"]))] = int(r["n"])
    return out


def _build_alt_recipe_concat_queries(
    *,
    alt_candidates: pd.DataFrame,
    recipes: pd.DataFrame,
    global_counts: Dict[Tuple[str, str], int],
    default_max_a: int,
    default_max_b: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build composite ALT query rows from recipe allowlist.

    Parameters
    ----------
    alt_candidates
        ALT rows after language + non-empty filtering (and usually after hard
        type-excludes). Must include:
          - cde_publicid, cde_version, cde_key
          - alternate_name, alternate_name_type, context_name, context_version, language
    recipes
        Recipe allowlist rows (literal-exact matching).
    global_counts
        Global corpus counts of normalized A-values by alt_type (for tie-breaking).
    default_max_a, default_max_b
        CLI defaults if per-recipe max_a/max_b are not provided.

    Returns
    -------
    recipe_rows
        DataFrame of synthetic ALT rows representing composite queries.
    report_df
        Per-recipe diagnostics (rows generated, coverage).
    """
    if recipes is None or recipes.empty:
        empty = alt_candidates.head(0).copy()
        # Return empty recipe rows with required columns.
        cols_needed = [
            "cde_publicid",
            "cde_version",
            "cde_key",
            "alternate_name",
            "alternate_name_type",
            "context_name",
            "context_version",
            "language",
            "family",
            "alt_query_mode",
            "alt_recipe_id",
            "alt_component_a_type",
            "alt_component_b_type",
            "alt_component_a",
            "alt_component_b",
            "source_file",
            "cde_xml_row_id",
        ]
        for c in cols_needed:
            if c not in empty.columns:
                empty[c] = ""
        empty = empty[cols_needed].head(0)
        report = pd.DataFrame([], columns=["recipe_id", "context_name", "alt_type_a", "alt_type_b", "family", "n_rows", "n_cdes_both"])
        return empty, report

    # Build outputs per recipe and concat at end.
    out_rows: List[pd.DataFrame] = []
    report_rows: List[Dict[str, Any]] = []

    # Only consider non-empty alternate_name text.
    base = alt_candidates.copy()
    base = base[base["alternate_name"].map(_is_nonempty).astype(bool)].copy()

    for _, r in recipes.iterrows():
        recipe_id = str(r["recipe_id"])
        ctx = str(r["context_name"])
        a_type = str(r["alt_type_a"])
        b_type = str(r["alt_type_b"])
        joiner = str(r["joiner"])
        family = str(r["family"])

        try:
            max_a = int(str(r.get("max_a", "") or "").strip() or default_max_a)
        except Exception:
            max_a = int(default_max_a)
        try:
            max_b = int(str(r.get("max_b", "") or "").strip() or default_max_b)
        except Exception:
            max_b = int(default_max_b)

        # Subset to this context and the two component types.
        sub = base[(base["context_name"] == ctx) & (base["alternate_name_type"].isin([a_type, b_type]))].copy()
        if sub.empty:
            report_rows.append(
                {
                    "recipe_id": recipe_id,
                    "context_name": ctx,
                    "alt_type_a": a_type,
                    "alt_type_b": b_type,
                    "family": family,
                    "joiner": joiner,
                    "max_a": max_a,
                    "max_b": max_b,
                    "n_rows": 0,
                    "n_cdes_both": 0,
                    "n_cdes_with_a": 0,
                    "n_cdes_with_b": 0,
                    "reason": "no_rows_for_context_or_types",
                }
            )
            continue

        dfA = sub[sub["alternate_name_type"] == a_type].copy()
        dfB = sub[sub["alternate_name_type"] == b_type].copy()

        # Compute safe normalized forms for deterministic ordering and for the synthetic query text.
        dfA["a_norm"] = dfA["alternate_name"].fillna("").astype(str).map(normalize_query_text)
        dfB["b_norm"] = dfB["alternate_name"].fillna("").astype(str).map(normalize_query_text)

        # A-value ranking: quality_score desc, corpus_count desc, norm asc
        a_unique = pd.DataFrame({"a_norm": sorted(dfA["a_norm"].dropna().astype(str).unique().tolist())})
        if len(a_unique) > 0:
            scores = a_unique["a_norm"].map(lambda v: _table_domain_quality_score(a_type, v)[0])
            reasons = a_unique["a_norm"].map(lambda v: _table_domain_quality_score(a_type, v)[1])
            a_unique["quality_score"] = scores.astype(int)
            a_unique["quality_reason"] = reasons.astype(str)
            a_unique["corpus_count"] = a_unique["a_norm"].map(lambda v: int(global_counts.get((a_type, v), 0)))
            a_unique = a_unique.sort_values(["quality_score", "corpus_count", "a_norm"], ascending=[False, False, True]).reset_index(drop=True)
            a_unique["a_rank"] = a_unique.index.astype(int)
        else:
            a_unique["quality_score"] = []
            a_unique["quality_reason"] = []
            a_unique["corpus_count"] = []
            a_unique["a_rank"] = []

        dfA = dfA.merge(a_unique[["a_norm", "a_rank"]], on="a_norm", how="left")
        dfA["a_rank"] = dfA["a_rank"].fillna(10**9).astype(int)

        # Keep top max_a unique a_norm per CDE.
        dfA = dfA.sort_values(["cde_key", "a_rank", "a_norm"], ascending=[True, True, True])
        dfA = dfA.drop_duplicates(subset=["cde_key", "a_norm"], keep="first")
        dfA["a_i"] = dfA.groupby("cde_key").cumcount()
        dfA_top = dfA[dfA["a_i"] < int(max_a)].copy()

        # B-value ranking: lexicographic by b_norm (deterministic).
        dfB = dfB.sort_values(["cde_key", "b_norm"], ascending=[True, True])
        dfB = dfB.drop_duplicates(subset=["cde_key", "b_norm"], keep="first")
        dfB["b_i"] = dfB.groupby("cde_key").cumcount()
        dfB_top = dfB[dfB["b_i"] < int(max_b)].copy()

        n_cdes_with_a = int(dfA_top["cde_key"].nunique())
        n_cdes_with_b = int(dfB_top["cde_key"].nunique())

        # Inner join on cde_key -> cross product of selected A and selected B within each CDE.
        dfAB = dfA_top.merge(dfB_top, on=["cde_key", "cde_publicid", "cde_version"], how="inner", suffixes=("_a", "_b"))
        if dfAB.empty:
            report_rows.append(
                {
                    "recipe_id": recipe_id,
                    "context_name": ctx,
                    "alt_type_a": a_type,
                    "alt_type_b": b_type,
                    "family": family,
                    "joiner": joiner,
                    "max_a": max_a,
                    "max_b": max_b,
                    "n_rows": 0,
                    "n_cdes_both": 0,
                    "n_cdes_with_a": n_cdes_with_a,
                    "n_cdes_with_b": n_cdes_with_b,
                    "reason": "no_cdes_with_both_components",
                }
            )
            continue

        dfAB["alternate_name"] = dfAB["a_norm"].astype(str) + joiner + dfAB["b_norm"].astype(str)

        # Build synthetic ALT rows.
        out = pd.DataFrame(
            {
                "cde_publicid": dfAB["cde_publicid"].astype(str),
                "cde_version": dfAB["cde_version"].astype(str),
                "cde_key": dfAB["cde_key"].astype(str),
                "alternate_name": dfAB["alternate_name"].astype(str),
                "alternate_name_type": f"RECIPE::{recipe_id}",
                "context_name": ctx,
                # Prefer the context_version from component rows when present.
                "context_version": dfAB.get("context_version_a", pd.Series([""] * len(dfAB))).fillna("").astype(str),
                "language": dfAB.get("language_a", pd.Series([""] * len(dfAB))).fillna("English").astype(str),
                "family": family,
                "alt_query_mode": "recipe",
                "alt_recipe_id": recipe_id,
                "alt_component_a_type": a_type,
                "alt_component_b_type": b_type,
                "alt_component_a": dfAB["a_norm"].astype(str),
                "alt_component_b": dfAB["b_norm"].astype(str),
                # No single source_file/xml_row_id: these are synthetic composites.
                "source_file": "",
                "cde_xml_row_id": "",
            }
        )

        out_rows.append(out)

        report_rows.append(
            {
                "recipe_id": recipe_id,
                "context_name": ctx,
                "alt_type_a": a_type,
                "alt_type_b": b_type,
                "family": family,
                "joiner": joiner,
                "max_a": max_a,
                "max_b": max_b,
                "n_rows": int(len(out)),
                "n_cdes_both": int(out["cde_key"].nunique()),
                "n_cdes_with_a": n_cdes_with_a,
                "n_cdes_with_b": n_cdes_with_b,
                "reason": "ok",
            }
        )

    recipe_rows = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame([], columns=[])
    report_df = pd.DataFrame(report_rows)

    return recipe_rows, report_df


def _write_alt_table_domain_ordering_evidence_from_xml(
    *,
    xml_path: str,
    out_csv: str,
    top_n: int = 50,
    alt_types: Optional[List[str]] = None,
) -> None:
    """Generate a small, reproducible 'ordering evidence' CSV from a caDSR CDE XML file.

    This is intended to justify why we use a quality-first ordering for Table/Domain
    components (max_a selection) instead of alphabetical ordering.

    The evidence file is deliberately small: per alt_type we emit the top-N most
    frequent normalized tokens observed in the XML.

    Parameters
    ----------
    xml_path
        Path to a caDSR CDE export XML file.
    out_csv
        Output CSV path.
    top_n
        Max rows per alt_type (default 50).
    alt_types
        AlternateNameType values to consider. If None, uses the known Table/Domain
        types used by the default recipe allowlist.
    """
    if alt_types is None:
        alt_types = [
            "FHIR US CORE Table",
            "CDISC SDTM Domain",
            "PCORnet Table",
            "OMOP Table",
            "CCDI Table",
        ]

    try:
        from lxml import etree
    except Exception as e:
        raise ImportError("lxml is required for --ordering-evidence-xml") from e

    counts: Dict[Tuple[str, str], int] = {}
    example_raw: Dict[Tuple[str, str], str] = {}

    def _child_text(el, name: str) -> str:
        if el is None:
            return ""
        c = el.find(name)
        if c is None or c.text is None:
            return ""
        return str(c.text)

    # Stream DataElement nodes to keep memory bounded.
    ctx = etree.iterparse(str(xml_path), events=("end",), tag="DataElement", huge_tree=True)
    for _, elem in ctx:
        an_list = elem.find("ALTERNATENAMELIST")
        if an_list is not None:
            for an in an_list.findall("ALTERNATENAMELIST_ITEM"):
                t = _child_text(an, "AlternateNameType")
                if t not in set(alt_types):
                    continue
                val = _child_text(an, "AlternateName")
                if val is None:
                    continue
                val = str(val)
                if val.strip() == "":
                    continue
                norm = normalize_query_text(val)
                key = (t, norm)
                counts[key] = int(counts.get(key, 0)) + 1
                if key not in example_raw:
                    example_raw[key] = val

        # Free memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    rows: List[Dict[str, Any]] = []
    for (t, norm), n in counts.items():
        score, reason = _table_domain_quality_score(t, norm)
        rows.append(
            {
                "alternate_name_type": t,
                "value_norm": norm,
                "example_raw": example_raw.get((t, norm), ""),
                "count_in_xml": int(n),
                "quality_score": int(score),
                "quality_reason": str(reason),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
        return

    # Rank order used by the implementation: quality_score desc, count desc, value_norm asc
    df = df.sort_values(
        ["alternate_name_type", "quality_score", "count_in_xml", "value_norm"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    df["rank_within_type"] = df.groupby("alternate_name_type").cumcount() + 1

    # Emit top-N by count per type for a compact evidence file.
    df_top = (
        df.sort_values(["alternate_name_type", "count_in_xml", "value_norm"], ascending=[True, False, True])
        .groupby("alternate_name_type", as_index=False, group_keys=False)
        .head(int(top_n))
        .copy()
    )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df_top.to_csv(out_csv, index=False)



def _load_refdoc_allowlist_csv(path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load strict REF document-type allowlist CSV.

    The allowlist is **literal-exact**:

    - No wildcards: ``document_type`` values are matched exactly.
    - Case-sensitive: ``preferred`` and ``Preferred`` are treated as different.
    - Whitespace-sensitive: leading/trailing spaces are treated as meaningful characters.

    Notes
    -----
    We normalize REF source tables via ``_norm_str`` before matching (e.g., trimming
    ends and collapsing internal whitespace). For reproducibility, curators should
    copy/paste DocumentType values from ``refdoc_type_inventory.csv`` produced by
    ``demap build-queries``.

    Returns
    -------
    allow_df
        Allowlist dataframe (no normalization for matching).
    allowed_types
        List of allowed document_type strings.
    """
    p = os.path.expanduser(path)

    # Allow both repo-root relative paths ("configs/...") and absolute paths.
    if not os.path.exists(p):
        alt_p = str((Path(__file__).resolve().parents[3] / p))
        if os.path.exists(alt_p):
            p = alt_p
        else:
            raise FileNotFoundError(
                f"REFDOC allowlist CSV not found: {path}. "
                f"Expected by default at: {DEFAULT_REFDOC_ALLOWLIST}"
            )

    allow_df = pd.read_csv(p, dtype=str).fillna("")

    expected = {"document_type"}
    missing = sorted(list(expected - set(allow_df.columns)))
    if missing:
        raise ValueError(
            f"REFDOC allowlist CSV missing required columns: {missing}. "
            f"Found columns: {sorted(list(allow_df.columns))}"
        )

    allow_df["document_type"] = allow_df["document_type"].astype(str)

    def _is_blank(v: Any) -> bool:
        s = "" if v is None else str(v)
        s2 = s.strip()
        return (s2 == "") or (s2.casefold() == "nan")

    if len(allow_df) > 0:
        bad = allow_df[allow_df["document_type"].map(_is_blank)]
        if not bad.empty:
            preview = bad.head(5).copy()
            preview.insert(0, "row", preview.index + 2)
            raise ValueError(
                "REFDOC allowlist CSV contains blank required fields. "
                f"First offending rows:\n{preview.to_string(index=False)}"
            )

    # Preserve order (useful for curator review / diffs).
    allowed_types: List[str] = []
    seen: set[str] = set()
    for _i, r in allow_df.iterrows():
        t = str(r["document_type"])
        if t in seen:
            continue
        allowed_types.append(t)
        seen.add(t)

    return allow_df, allowed_types


def _build_alt_candidate_inventory(alt_pre: pd.DataFrame) -> pd.DataFrame:
    """Build curator-facing inventory of candidate ALT buckets.

    This is computed on **English + non-empty** alternate names, *pre-allowlist*.
    """
    if alt_pre.empty:
        return pd.DataFrame(
            columns=[
                "alternate_name_type",
                "context_name",
                "n_rows",
                "n_unique_cdes",
                "n_unique_names",
                "name_len_min",
                "name_len_median",
                "name_len_max",
                "pct_with_spaces",
                "pct_snake_case",
                "pct_all_caps",
                "type_excluded_by_rules",
            ]
        )

    d = alt_pre.copy()

    d["name_len"] = d["alternate_name"].astype(str).str.len()
    d["has_space"] = d["alternate_name"].astype(str).str.contains(r"\s", regex=True)
    d["snake_case"] = d["alternate_name"].astype(str).str.match(r"^[A-Za-z0-9]+(_[A-Za-z0-9]+)+$")
    d["all_caps"] = d["alternate_name"].astype(str).str.match(r"^[A-Z0-9_]+$")

    grp = d.groupby(["alternate_name_type", "context_name"], dropna=False)
    inv = grp.agg(
        n_rows=("cde_key", "size"),
        n_unique_cdes=("cde_key", "nunique"),
        n_unique_names=("alternate_name", "nunique"),
        name_len_min=("name_len", "min"),
        name_len_median=("name_len", "median"),
        name_len_max=("name_len", "max"),
        pct_with_spaces=("has_space", "mean"),
        pct_snake_case=("snake_case", "mean"),
        pct_all_caps=("all_caps", "mean"),
        type_excluded_by_rules=("type_excluded", "max"),
    ).reset_index()

    # Tidy
    for c in ["pct_with_spaces", "pct_snake_case", "pct_all_caps"]:
        inv[c] = inv[c].fillna(0.0).astype(float).round(3)
    inv["type_excluded_by_rules"] = inv["type_excluded_by_rules"].fillna(False).astype(bool)

    inv = inv.sort_values(["n_rows", "n_unique_cdes"], ascending=False)
    return inv


def _stable_int_from_str(s: str) -> int:
    """Stable integer derived from a string.

    We avoid Python's built-in ``hash()`` because it is intentionally randomized
    between interpreter sessions.

    This returns a non-negative integer in the range [0, 2**31-1].
    """
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    # Use 8 hex chars (~32 bits), then clamp to 31-bit for pandas/random_state.
    v = int(h[:8], 16)
    return int(v % (2**31 - 1))


def _sample_stratified_by_bucket(
    df: pd.DataFrame,
    *,
    bucket_cols: List[str],
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Sample up to ``n`` rows, stratified by bucket.

    Strategy
    --------
    - Compute bucket sizes and sort buckets by size (descending).
    - Allocate at least 1 sample to each of the top-K buckets where K=min(n, nbuckets).
    - Distribute remaining samples round-robin across those buckets (bounded by each
      bucket's size).
    - Sample deterministically within each bucket using a stable per-bucket seed.

    This yields a diverse sample that still emphasizes large buckets.
    """

    if df.empty or n <= 0:
        return df.head(0).copy()

    missing = [c for c in bucket_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot stratify-sample: missing bucket columns: {missing}")

    counts = (
        df.groupby(bucket_cols, dropna=False)
        .size()
        .reset_index(name="bucket_n")
        .sort_values(["bucket_n"], ascending=False)
        .reset_index(drop=True)
    )

    bucket_keys: List[Tuple[Any, ...]] = [
        tuple(r[c] for c in bucket_cols) for _i, r in counts.iterrows()
    ]
    bucket_sizes: Dict[Tuple[Any, ...], int] = {
        tuple(r[c] for c in bucket_cols): int(r["bucket_n"]) for _i, r in counts.iterrows()
    }

    # Only allocate across the top-K buckets if there are more buckets than samples.
    K = int(min(len(bucket_keys), n))
    bucket_keys = bucket_keys[:K]

    alloc: Dict[Tuple[Any, ...], int] = {k: 1 for k in bucket_keys}
    remaining = int(n - K)

    # Round-robin distribute remaining samples, bounded by bucket size.
    if remaining > 0 and K > 0:
        idx = 0
        made_progress = True
        while remaining > 0 and made_progress:
            made_progress = False
            k = bucket_keys[idx % K]
            if alloc[k] < bucket_sizes.get(k, 0):
                alloc[k] += 1
                remaining -= 1
                made_progress = True
            idx += 1

    parts: List[pd.DataFrame] = []
    for k in bucket_keys:
        d = df
        for c, v in zip(bucket_cols, k):
            d = d[d[c] == v]
        take = int(min(len(d), alloc.get(k, 0)))
        if take <= 0:
            continue
        # Stable per-bucket seed.
        seed_i = int((seed + _stable_int_from_str("|".join(map(str, k)))) % (2**31 - 1))
        if len(d) <= take:
            s = d.copy()
        else:
            s = d.sample(n=take, random_state=seed_i)
        parts.append(s)

    if not parts:
        return df.head(0).copy()

    out = pd.concat(parts, ignore_index=True)
    return out.reset_index(drop=True)


def _write_rule_exclusion_artifacts(
    *,
    out_dir: str,
    source: str,
    rule_order: List[str],
    rule_notes: Dict[str, str],
    excluded_groups: Dict[str, pd.DataFrame],
    text_col: str,
    bucket_cols: List[str],
    sample_cols: List[str],
    sample_n: int,
    sample_seed: int,
    summary_filename: str,
    samples_filename: str,
) -> Tuple[str, str]:
    """Write exclusion summary + sampled examples for a query source (ALT or REF)."""

    os.makedirs(out_dir, exist_ok=True)

    # Summary
    summary_rows: List[Dict[str, Any]] = []
    for rule in rule_order:
        df = excluded_groups.get(rule)
        if df is None or df.empty:
            n_rows = 0
            n_cdes = 0
            n_texts = 0
            n_buckets = 0
        else:
            n_rows = int(len(df))
            if "cde_key" in df.columns:
                n_cdes = int(df["cde_key"].nunique())
            else:
                n_cdes = int(df[["cde_publicid", "cde_version"]].drop_duplicates().shape[0])
            n_texts = int(df[text_col].nunique()) if text_col in df.columns else 0
            if all(c in df.columns for c in bucket_cols):
                n_buckets = int(df[bucket_cols].drop_duplicates().shape[0])
            else:
                n_buckets = 0

        summary_rows.append(
            {
                "query_source": source,
                "rule_name": rule,
                "n_rows": n_rows,
                "n_unique_cdes": n_cdes,
                "n_unique_texts": n_texts,
                "n_unique_buckets": n_buckets,
                "notes": rule_notes.get(rule, ""),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False)

    # Samples
    sample_parts: List[pd.DataFrame] = []
    for rule in rule_order:
        df = excluded_groups.get(rule)
        if df is None or df.empty:
            continue
        d = df.copy()
        if "raw_text" not in d.columns:
            d["raw_text"] = d[text_col].astype(str)
        if "normalized_text" not in d.columns:
            d["normalized_text"] = d[text_col].map(normalize_query_text)

        sampled = _sample_stratified_by_bucket(d, bucket_cols=bucket_cols, n=int(sample_n), seed=int(sample_seed))
        sampled = sampled.copy()
        sampled.insert(0, "rule_name", rule)
        sampled.insert(0, "query_source", source)

        keep = [c for c in ["query_source", "rule_name", *sample_cols] if c in sampled.columns]
        sample_parts.append(sampled[keep].copy())

    samples_df = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame(columns=["query_source", "rule_name", *sample_cols])
    samples_path = os.path.join(out_dir, samples_filename)
    samples_df.to_csv(samples_path, index=False)

    return summary_path, samples_path


def _dedupe_within_cde_keep_provenance(
    df: pd.DataFrame,
    *,
    text_col: str,
    key_cols: List[str],
    report_path: str,
    max_collision_rows: int = 200,
) -> pd.DataFrame:
    """Deduplicate within-CDE query instances using conservative normalization.

    This collapses duplicates that differ only by formatting noise (whitespace,
    common Unicode quotes, and dash/minus variants), while **preserving
    provenance buckets**.

    Duplicate definition
    --------------------
    Two rows are duplicates iff the following are identical:

    - CDE identity: (cde_publicid, cde_version)
    - Provenance identity: ``key_cols`` (e.g., ALT type/context; REF doctype/name)
    - Normalized text: ``normalize_query_text(df[text_col])``

    We intentionally do **not** lowercase during normalization.

    Notes
    -----
    - This function is intended for *within-CDE* de-duplication only.
    - It is deterministic (stable sorting + keep='first').
    - It writes a lightweight CSV report for auditability.
    """

    # Always write a report for reproducibility, even if empty input.
    if df.empty:
        report_df = pd.DataFrame(
            [
                {
                    "report_type": "summary",
                    "total_rows_before": 0,
                    "total_rows_after": 0,
                    "n_removed": 0,
                    "n_collision_groups": 0,
                }
            ]
        )
        report_df.to_csv(report_path, index=False)
        return df

    d = df.copy()
    norm_col = "_normalized_query_text"
    d[norm_col] = d[text_col].map(normalize_query_text)

    # Deterministic ordering so `drop_duplicates(keep='first')` is stable.
    sort_cols = [
        c
        for c in [
            "cde_publicid",
            "cde_version",
            *key_cols,
            norm_col,
            text_col,
        ]
        if c in d.columns
    ]
    d = d.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    dedupe_cols = [c for c in ["cde_publicid", "cde_version", *key_cols, norm_col] if c in d.columns]
    before_n = int(len(d))

    # Collision groups (before dropping).
    grp = d.groupby(dedupe_cols, dropna=False)
    sizes = grp.size().reset_index(name="group_size")
    collisions = sizes[sizes["group_size"] > 1].copy()
    n_collision_groups = int(len(collisions))

    # Keep first row per group.
    d_deduped = d.drop_duplicates(subset=dedupe_cols, keep="first").copy()
    after_n = int(len(d_deduped))
    removed_n = before_n - after_n

    # Report: one summary row + up to max_collision_rows collision rows.
    report_rows: List[Dict[str, Any]] = [
        {
            "report_type": "summary",
            "total_rows_before": before_n,
            "total_rows_after": after_n,
            "n_removed": removed_n,
            "n_collision_groups": n_collision_groups,
        }
    ]

    if n_collision_groups > 0:
        top = collisions.sort_values("group_size", ascending=False).head(max_collision_rows)
        # Columns for provenance (exclude internal normalized key col).
        report_key_cols = [c for c in dedupe_cols if c != norm_col]
        for _, r in top.iterrows():
            key = {c: r[c] for c in dedupe_cols}
            g = d
            for c in dedupe_cols:
                g = g[g[c] == key[c]]
            raw_texts = g[text_col].astype(str).tolist()
            kept_raw = raw_texts[0] if raw_texts else ""
            dropped = raw_texts[1:]
            report_rows.append(
                {
                    "report_type": "collision",
                    "total_rows_before": before_n,
                    "total_rows_after": after_n,
                    "n_removed": removed_n,
                    "n_collision_groups": n_collision_groups,
                    **{c: key.get(c, "") for c in report_key_cols},
                    "normalized_query_text": key.get(norm_col, ""),
                    "group_size": int(r["group_size"]),
                    "kept_raw_text": kept_raw,
                    "dropped_count": max(0, len(raw_texts) - 1),
                    "dropped_raw_texts_sample": " | ".join(dropped[:5]),
                }
            )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False)

    # Remove internal column.
    d_deduped = d_deduped.drop(columns=[norm_col])
    return d_deduped


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--alt-parquet",
        default=os.path.join("data", "interim", "cadsr_merged", "cde_alternate_names.parquet"),
        help="Path to merged cde_alternate_names.parquet",
    )
    ap.add_argument(
        "--ref-parquet", "--refdoc-parquet",
        default=os.path.join("data", "interim", "cadsr_merged", "cde_reference_documents.parquet"),
        help="Path to merged cde_reference_documents.parquet (alias: --refdoc-parquet)",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join("artifacts", "summaries", "inwild_strict"),
        help="Output directory for audit artifacts (CSVs, tsvs, debugging outputs)",
    )
    ap.add_argument(
        "--out-parquet",
        default=os.path.join("data", "processed", "queries.parquet"),
        help="Write row-level query records to this parquet (for downstream training/splitting)",
    )
    ap.add_argument(
        "--language",
        default="English",
        help="Only retain rows with Language == this value (case-insensitive)",
    )
    ap.add_argument(
        "--alt-allowlist",
        default=DEFAULT_ALT_ALLOWLIST,
        help=(
            "CSV allowlist of exact (alternate_name_type, context_name) ALT buckets, with family labels. "
            "Default is configs/allowlists/alt_allowlist_current.csv"
        ),
    )

    ap.add_argument(
        "--alt-recipe-allowlist",
        default=DEFAULT_ALT_RECIPE_ALLOWLIST,
        help=(
            "CSV allowlist of ALT composite query recipes (e.g., Table + Name). "
            "Default is configs/allowlists/alt_query_recipes_current.csv. "
            "Set --no-alt-recipes to disable."
        ),
    )
    ap.add_argument(
        "--no-alt-recipes",
        action="store_true",
        help="Disable building composite/concatenated ALT queries from the recipe allowlist.",
    )
    ap.add_argument(
        "--alt-concat-max-a",
        type=int,
        default=1,
        help=(
            "Default max number of component-A tokens (Table/Domain) to include per CDE when "
            "building composite ALT queries (default: 1)."
        ),
    )
    ap.add_argument(
        "--alt-concat-max-b",
        type=int,
        default=10,
        help=(
            "Default max number of component-B tokens (Name/Variable) to include per CDE when "
            "building composite ALT queries (default: 10)."
        ),
    )
    ap.add_argument(
        "--ordering-evidence-xml",
        default="",
        help=(
            "Optional path to a caDSR CDE XML file used to generate an 'ordering evidence' CSV "
            "for Table/Domain tokens (quality-first ordering justification). "
            "If blank, no evidence CSV is generated."
        ),
    )
    ap.add_argument(
        "--ordering-evidence-top-n",
        type=int,
        default=50,
        help="Top-N most frequent tokens per alt_type to include in the ordering evidence CSV (default: 50).",
    )
    ap.add_argument(
        "--ordering-evidence-out-csv",
        default="",
        help=(
            "Where to write the ordering evidence CSV. "
            "Default: <out-dir>/ordering_evidence_alt_table_domain.csv"
        ),
    )

    ap.add_argument(
        "--refdoc-allowlist",
        default=DEFAULT_REFDOC_ALLOWLIST,
        help=(
            "CSV allowlist of literal REF document_type strings (question/prompt text only). "
            "Default is configs/allowlists/refdoc_allowlist_current.csv"
        ),
    )
    ap.add_argument(
        "--excluded-top-n",
        type=int,
        default=10,
        help="Number of excluded ALT buckets to write to top10_excluded_alt_buckets.csv (default: 10)",
    )
    ap.add_argument(
        "--skip-alt-candidate-inventory",
        action="store_true",
        help="If set, do not write alt_candidate_inventory.csv (still writes top10_excluded_alt_buckets.csv)",
    )

    ap.add_argument(
        "--excluded-sample-n",
        type=int,
        default=100,
        help=(
            "Maximum number of excluded-row samples to write per exclusion group, in the "
            "*_rule_excluded_samples.csv artifacts (default: 100)."
        ),
    )
    ap.add_argument(
        "--excluded-sample-seed",
        type=int,
        default=0,
        help="Random seed for excluded-row sampling (deterministic; default: 0)",
    )

    ap.add_argument("--min-cdes", type=int, default=500, help="Min unique CDEs for a holdout candidate")
    ap.add_argument(
        "--min-queries",
        type=int,
        default=1000,
        help="Min query->CDE pairs for a holdout candidate (after dedup)",
    )
    ap.add_argument(
        "--allow-other-ref-names",
        action="store_true",
        help=(
            "If set, do not drop refdoc rows whose Name does not match a strict provenance pattern. "
            "(Useful for debugging or exploratory runs; strict mode is default.)"
        ),
    )

    # Optional PV attachment (default-on if a PV parquet exists).
    #
    # If a merged cde_permissible_values.parquet is available, we can append a
    # deterministic PV summary to the *prefixed* query_text (Q2) for both ALT
    # and REF. We intentionally leave query_text_raw (Q1) unchanged so the
    # dataset continues to provide a PV-free baseline query variant.
    ap.add_argument(
        "--pv-parquet",
        default=None,
        help=(
            "Optional path to merged cde_permissible_values.parquet. "
            "If omitted, build-queries will auto-detect a sibling file named 'cde_permissible_values.parquet' "
            "next to --alt-parquet/--ref-parquet and, if found, attach PV summaries to query_text (Q2)."
        ),
    )
    ap.add_argument(
        "--no-pv-attachment",
        action="store_true",
        help=(
            "Disable attaching PV summaries to query_text (Q2), even if a merged PV parquet is present. "
            "Default behavior is to attach when cde_permissible_values.parquet can be found."
        ),
    )
    ap.add_argument(
        "--pv-max-n",
        type=int,
        default=10,
        help="Max number of PV examples to include in the PV summary block (cap; default: 10)",
    )
    ap.add_argument(
        "--pv-huge-threshold",
        type=int,
        default=20,
        help=(
            "Treat PV lists larger than this as 'huge' for non-generic example selection "
            "(still capped by --pv-max-n; default: 20)"
        ),
    )
    ap.add_argument(
        "--sde-generic-label-p",
        type=float,
        default=0.30,
        help=(
            "Probability (0..1; deterministic) that SDE PV summaries render a generic concept " 
	    "using the label (value_meaning) instead of the code (valid_value). Default: 0.30."
        ),
    )
    ap.add_argument(
        "--pv-hash-salt",
        default="demap",
        help="Salt for deterministic PV rendering decisions (default: 'demap')",
    )
    ap.add_argument("--pv-summary-profile", default="cadsr", help="PV summary profile (default: cadsr).")
    ap.add_argument("--pv-max-n-query", type=int, default=None, help="Optional query-side PV cap.")
    ap.add_argument("--pv-max-n-cde", type=int, default=None, help="Optional CDE-side PV cap.")
    ap.add_argument("--pv-center-fraction", type=float, default=0.5)
    ap.add_argument("--pv-min-n", type=int, default=2)
    ap.add_argument("--pv-size-jitter", type=int, default=1)
    ap.add_argument("--pv-generic-boost", type=float, default=0.10)
    ap.add_argument("--pv-generic-cap", type=float, default=0.70)
    ap.add_argument("--pv-small-n-generic-threshold", type=float, default=0.5)
    ap.add_argument("--pv-max-resample-attempts", type=int, default=3)
    ap.add_argument("--pv-placeholder-token", default="<MISSING_PV_SUMMARY>")
    ap.add_argument("--pv-diagnostics-summary", default=None, help="Optional JSON path for PV summary diagnostics.")
    ap.add_argument("--pv-diagnostics-records", default=None, help="Optional CSV path for per-CDE PV diagnostics.")

    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    # -----------------------------------------------------------------
    # Optional: generate a compact, reproducible 'ordering evidence' CSV
    # from a representative caDSR XML file (publishable justification for
    # quality-first ordering when selecting Table/Domain tokens).
    # -----------------------------------------------------------------
    if str(getattr(args, "ordering_evidence_xml", "") or "").strip() != "":
        oe_xml = str(args.ordering_evidence_xml)
        oe_out = str(getattr(args, "ordering_evidence_out_csv", "") or "").strip()
        if oe_out == "":
            oe_out = os.path.join(args.out_dir, "ordering_evidence_alt_table_domain.csv")
        try:
            _write_alt_table_domain_ordering_evidence_from_xml(
                xml_path=oe_xml,
                out_csv=oe_out,
                top_n=int(getattr(args, "ordering_evidence_top_n", 50) or 50),
            )
            print("Wrote ordering evidence CSV to:", oe_out)
        except Exception as e:
            print("WARNING: failed to generate ordering evidence CSV:", e)


    # -----------------------------------------------------------------
    # PV attachment (default-on when a merged PV parquet is available)
    # -----------------------------------------------------------------
    pv_parquet_path: Optional[str] = None
    if bool(getattr(args, "no_pv_attachment", False)):
        pv_parquet_path = None
    elif args.pv_parquet is not None and str(args.pv_parquet).strip() != "":
        pv_parquet_path = str(args.pv_parquet)
    else:
        # Auto-detect a sibling merged PV parquet next to the ALT/REF inputs.
        # This keeps the default workflow simple:
        #   data/interim/cadsr_merged/{cde_alternate_names,cde_reference_documents,cde_permissible_values}.parquet
        for base in [args.alt_parquet, args.ref_parquet]:
            try:
                cand = Path(str(base)).expanduser().resolve().parent / "cde_permissible_values.parquet"
            except Exception:
                continue
            if cand.exists():
                pv_parquet_path = str(cand)
                break

    # ---------------------
    # Load (read only needed columns)
    # ---------------------
    alt_cols = [
        "cde_publicid",
        "cde_version",
        "alternate_name",
        "alternate_name_type",
        "context_name",
        "context_version",
        "language",
        "source_file",
        "cde_xml_row_id",
    ]
    ref_cols = [
        "cde_publicid",
        "cde_version",
        "document_type",
        "name",
        "document_text",
        "language",
        "source_file",
        "cde_xml_row_id",
        "url",
        "display_order",
    ]

    # Some merged files may not have all optional columns; we will load what exists.
    alt = pd.read_parquet(args.alt_parquet)
    ref = pd.read_parquet(args.ref_parquet)

    # Basic column normalization: tolerate slightly different column names.
    alt = alt.rename(
        columns={
            "alternateName": "alternate_name",
            "alternateNameType": "alternate_name_type",
            "contextName": "context_name",
            "contextVersion": "context_version",
        }
    )
    # Refdoc parquet column normalization: tolerate slightly different schemas
    # across merged exports and historical pipeline versions.
    #
    # Historically we've seen:
    # - documentType / documentText / documentName (camelCase)
    # - document_type / document_text / document_name (snake_case)
    # - name as the canonical refdoc name column used by allowlists
    ref = ref.rename(
        columns={
            "documentType": "document_type",
            "documentText": "document_text",
            "documentName": "name",
            "document_name": "name",
            # Optional columns
            "URL": "url",
            "displayOrder": "display_order",
        }
    )

    # Ensure required columns exist.
    required_alt = ["cde_publicid", "cde_version", "alternate_name", "alternate_name_type", "context_name", "language"]
    required_ref = ["cde_publicid", "cde_version", "document_type", "name", "document_text", "language"]
    for col in required_alt:
        if col not in alt.columns:
            raise KeyError(f"Alternate-name parquet missing required column: {col}")
    for col in required_ref:
        if col not in ref.columns:
            raise KeyError(f"Refdoc parquet missing required column: {col}")

    # Normalize strings
    for col in ["alternate_name", "alternate_name_type", "context_name", "language"]:
        alt[col] = alt[col].map(_norm_str)
    for col in ["document_type", "name", "document_text", "language"]:
        ref[col] = ref[col].map(_norm_str)

    # Compute CDE keys early so exclusion reports can include non-English rows.
    alt["cde_key"] = _cde_key(alt)
    ref["cde_key"] = _cde_key(ref)

    # Keep pre-language-filter frames for exclusion reporting.
    alt_all = alt.copy()
    ref_all = ref.copy()

    # Language filter (strict English-only by default)
    lang_cf = str(args.language).strip().casefold()
    alt_non_english = alt_all.head(0).copy()
    ref_non_english = ref_all.head(0).copy()
    if lang_cf:
        alt_lang_mask = alt_all["language"].astype(str).str.casefold() == lang_cf
        ref_lang_mask = ref_all["language"].astype(str).str.casefold() == lang_cf
        alt_non_english = alt_all[~alt_lang_mask].copy()
        ref_non_english = ref_all[~ref_lang_mask].copy()
        alt = alt_all[alt_lang_mask].copy()
        ref = ref_all[ref_lang_mask].copy()

    # Exclusion groups accumulate rows that are filtered out by hard rules or
    # by strict allowlists.
    #
    # Each row is assigned to a single primary reason (rule_name) so counts are
    # interpretable.
    alt_excluded_groups: Dict[str, pd.DataFrame] = {"non_english": alt_non_english}
    ref_excluded_groups: Dict[str, pd.DataFrame] = {"non_english": ref_non_english}

    audit_rows: List[Dict[str, Any]] = []

    # ---------------------
    # Filter (ALT)
    # ---------------------
    alt0 = alt
    audit_rows.append({"stage": "alt_raw", "n_rows": int(len(alt0)), "n_cdes": int(alt0["cde_key"].nunique())})

    # pandas can infer a non-bool dtype for an empty Series returned by
    # .map(...). If used as df[mask], a non-bool Series is treated as a column
    # selector, which can yield an empty DataFrame with *no columns*.
    # Force boolean dtype so empty ALT/REF inputs are handled correctly.
    alt_nonempty_mask = alt0["alternate_name"].map(_is_nonempty).astype(bool)
    alt_empty = alt0[~alt_nonempty_mask].copy()
    alt_excluded_groups["empty_or_null_text"] = alt_empty

    alt1 = alt0[alt_nonempty_mask].copy()
    audit_rows.append({"stage": "alt_nonempty_name", "n_rows": int(len(alt1)), "n_cdes": int(alt1["cde_key"].nunique())})

    # Precompute safety exclude flag (for inventory + excluded buckets)
    alt1["type_excluded"] = alt1["alternate_name_type"].map(_type_is_excluded)

    # Hard type-based exclusion group (pre-allowlist, but English + non-empty).
    alt_excluded_groups["type_excluded_by_rules"] = alt1[alt1["type_excluded"].astype(bool)].copy()

    # Candidate inventory (pre-allowlist)
    inv_path = os.path.join(args.out_dir, "alt_candidate_inventory.csv")
    if not args.skip_alt_candidate_inventory:
        inv = _build_alt_candidate_inventory(alt1)
        inv.to_csv(inv_path, index=False)

    # Load strict allowlist (exact-only)
    allow_df, bucket_to_family = _load_alt_allowlist_csv(args.alt_allowlist)
    allowed_index = pd.MultiIndex.from_frame(allow_df[["alternate_name_type", "context_name"]])

    # Report allowlist rows with zero matches in the pre-allowlist ALT candidates
    present_buckets = set(
        map(
            tuple,
            alt1[["alternate_name_type", "context_name"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
        )
    )
    unmatched_rows: List[Dict[str, Any]] = []
    for _i, r in allow_df.iterrows():
        t = str(r["alternate_name_type"])
        c = str(r["context_name"])
        if (t, c) not in present_buckets:
            unmatched_rows.append(
                {
                    "alternate_name_type": t,
                    "context_name": c,
                    "family": str(r["family"]),
                    "reason": "no_matches_in_alt_candidates",
                }
            )

    unmatched_df = pd.DataFrame(unmatched_rows, columns=["alternate_name_type", "context_name", "family", "reason"])
    unmatched_path = os.path.join(args.out_dir, "alt_allowlist_unmatched.csv")
    unmatched_df.to_csv(unmatched_path, index=False)
    if len(unmatched_df) > 0:
        print(f"WARNING: {len(unmatched_df)} ALT allowlist rows had no matches. See: {unmatched_path}")

    # Literal exact matching on (alternate_name_type, context_name)
    alt_keys = pd.MultiIndex.from_frame(alt1[["alternate_name_type", "context_name"]])
    mask_allow = alt_keys.isin(allowed_index)

    # Not allowlisted (but otherwise eligible: English + non-empty + not type-excluded)
    alt_excluded_groups["not_allowlisted"] = alt1[(~mask_allow) & (~alt1["type_excluded"].astype(bool))].copy()

    alt2 = alt1[mask_allow].copy()
    audit_rows.append({"stage": "alt_in_allowlist_buckets", "n_rows": int(len(alt2)), "n_cdes": int(alt2["cde_key"].nunique())})

    # Safety excludes (type-level)
    # NOTE: For empty frames, pandas may infer a non-bool dtype for the
    # `type_excluded` Series (e.g., object). If used directly as a mask,
    # df[mask] can be interpreted as a column selector and drop all columns.
    # Cast to bool to ensure stable row-filter semantics.
    alt3 = alt2[~alt2["type_excluded"].astype(bool)].copy()
    audit_rows.append({"stage": "alt_after_excludes", "n_rows": int(len(alt3)), "n_cdes": int(alt3["cde_key"].nunique())})

    # Top excluded buckets (pre-allowlist minus rows that pass allowlist + safety excludes)
    # Definition: excluded = (English + non-empty ALT) minus (allowlist pass AND not type-excluded)
    mask_pass = mask_allow & (~alt1["type_excluded"].astype(bool))
    excluded = alt1[~mask_pass].copy()

    def _bucket_is_allowlisted(t: str, c: str) -> bool:
        return (str(t), str(c)) in bucket_to_family

    excl_by_bucket = (
        excluded.groupby(["alternate_name_type", "context_name"], dropna=False)
        .agg(
            n_rows=("cde_key", "size"),
            n_unique_cdes=("cde_key", "nunique"),
            n_unique_names=("alternate_name", "nunique"),
            any_type_excluded=("type_excluded", "max"),
        )
        .reset_index()
    )
    excl_by_bucket["is_allowlisted_bucket"] = excl_by_bucket.apply(
        lambda r: _bucket_is_allowlisted(r["alternate_name_type"], r["context_name"]), axis=1
    )
    excl_by_bucket = excl_by_bucket.sort_values(["n_rows", "n_unique_cdes"], ascending=False)
    top_n = int(args.excluded_top_n)
    if top_n <= 0:
        top_n = 10
    excl_path = os.path.join(args.out_dir, "top10_excluded_alt_buckets.csv")
    excl_by_bucket.head(top_n).to_csv(excl_path, index=False)

    # Family assignment for included ALT rows
    keys_included = list(zip(alt3["alternate_name_type"], alt3["context_name"]))
    alt3["family"] = [bucket_to_family[k] for k in keys_included]

    # Mark included ALT rows as 'bucket' queries by default (for provenance/audit).
    alt3["alt_query_mode"] = "bucket"
    alt3["alt_recipe_id"] = ""
    alt3["alt_component_a_type"] = ""
    alt3["alt_component_b_type"] = ""
    alt3["alt_component_a"] = ""
    alt3["alt_component_b"] = ""

    # -----------------------------------------------------------------
    # ALT composite query recipes (Table/Domain + Name/Variable)
    # -----------------------------------------------------------------
    recipe_rows = alt3.head(0).copy()
    recipe_report = pd.DataFrame([], columns=[])
    if not bool(getattr(args, "no_alt_recipes", False)):
        try:
            recipe_allow_df = _load_alt_recipe_allowlist_csv(args.alt_recipe_allowlist)

            # Global counts for A-types (Table/Domain) used for tie-breaking within max_a.
            a_types = sorted(recipe_allow_df["alt_type_a"].drop_duplicates().astype(str).tolist()) if len(recipe_allow_df) > 0 else []
            global_counts = _compute_global_norm_counts(
                alt1[~alt1["type_excluded"].astype(bool)].copy(),
                alt_types=a_types,
            )

            recipe_rows, recipe_report = _build_alt_recipe_concat_queries(
                alt_candidates=alt1[~alt1["type_excluded"].astype(bool)].copy(),
                recipes=recipe_allow_df,
                global_counts=global_counts,
                default_max_a=int(getattr(args, "alt_concat_max_a", 1) or 1),
                default_max_b=int(getattr(args, "alt_concat_max_b", 10) or 10),
            )

            # Write per-recipe diagnostics
            recipe_report_path = os.path.join(args.out_dir, "alt_recipe_allowlist_report.csv")
            if recipe_report is not None:
                recipe_report.to_csv(recipe_report_path, index=False)

                n_zero = int((recipe_report.get("n_rows", 0) == 0).sum()) if "n_rows" in recipe_report.columns else 0
                if n_zero > 0:
                    print(
                        f"WARNING: {n_zero} ALT recipe allowlist rows produced zero composite queries. "
                        f"See: {recipe_report_path}"
                    )

            # Append composite queries to included ALT set.
            if recipe_rows is not None and len(recipe_rows) > 0:
                audit_rows.append(
                    {
                        "stage": "alt_recipe_queries_generated",
                        "n_rows": int(len(recipe_rows)),
                        "n_cdes": int(recipe_rows["cde_key"].nunique()) if "cde_key" in recipe_rows.columns else 0,
                    }
                )
                alt3 = pd.concat([alt3, recipe_rows], ignore_index=True, sort=False)
                audit_rows.append(
                    {
                        "stage": "alt_after_recipes_appended",
                        "n_rows": int(len(alt3)),
                        "n_cdes": int(alt3["cde_key"].nunique()),
                    }
                )

        except Exception as e:
            print("WARNING: failed to build ALT composite queries from recipe allowlist:", e)


    # -----------------------------------------------------------------
    # ALT: rule-based exclusion artifacts (curator-facing)
    # -----------------------------------------------------------------
    def _add_alt_allowlist_meta(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return df_in
        d = df_in.copy()
        keys = list(zip(d["alternate_name_type"], d["context_name"]))
        d["is_allowlisted_bucket"] = [(t, c) in bucket_to_family for t, c in keys]
        d["family_if_allowlisted"] = [bucket_to_family.get((t, c), "") for t, c in keys]
        return d

    for _k in list(alt_excluded_groups.keys()):
        alt_excluded_groups[_k] = _add_alt_allowlist_meta(alt_excluded_groups[_k])

    alt_rule_order = [
        "non_english",
        "empty_or_null_text",
        "type_excluded_by_rules",
        "not_allowlisted",
    ]
    alt_rule_notes = {
        "non_english": "Filtered out by Language != requested language.",
        "empty_or_null_text": "Filtered out because alternate_name was empty or null.",
        "type_excluded_by_rules": "Filtered out because alternate_name_type matched hard exclusion rules.",
        "not_allowlisted": "Filtered out because (alternate_name_type, context_name) was not in the strict ALT allowlist.",
    }

    alt_rule_excl_summary_path, alt_rule_excl_samples_path = _write_rule_exclusion_artifacts(
        out_dir=args.out_dir,
        source="ALT",
        rule_order=alt_rule_order,
        rule_notes=alt_rule_notes,
        excluded_groups=alt_excluded_groups,
        text_col="alternate_name",
        bucket_cols=["alternate_name_type", "context_name"],
        sample_cols=[
            "cde_publicid",
            "cde_version",
            "cde_key",
            "alternate_name_type",
            "context_name",
            "language",
            "raw_text",
            "normalized_text",
            "is_allowlisted_bucket",
            "family_if_allowlisted",
        ],
        sample_n=int(args.excluded_sample_n),
        sample_seed=int(args.excluded_sample_seed),
        summary_filename="alt_rule_excluded_summary.csv",
        samples_filename="alt_rule_excluded_samples.csv",
    )

    # Deduplicate within-CDE query instances.
    #
    # Curator-aligned policy:
    # - Collapse duplicates that differ only by whitespace and common Unicode
    #   quotes/dashes (via normalize_query_text)
    # - Do NOT lowercase
    # - Do NOT dedupe across provenance buckets (keep ALT type/context/language)
    alt_before = int(len(alt3))
    alt_pairs = _dedupe_within_cde_keep_provenance(
        alt3,
        text_col="alternate_name",
        key_cols=["alternate_name_type", "context_name", "language"],
        report_path=os.path.join(args.out_dir, "alt_dedupe_report.csv"),
    )
    alt_after = int(len(alt_pairs))
    audit_rows.append({"stage": "alt_after_dedupe", "n_rows": alt_after, "n_cdes": int(alt_pairs["cde_key"].nunique())})

    # ---------------------
    # Filter (REFDOC)
    # ---------------------
    ref0 = ref
    audit_rows.append({"stage": "ref_raw", "n_rows": int(len(ref0)), "n_cdes": int(ref0["cde_key"].nunique())})

    # See note above for alt_nonempty_mask.
    ref_nonempty_mask = ref0["document_text"].map(_is_nonempty).astype(bool)
    ref_empty = ref0[~ref_nonempty_mask].copy()
    ref_excluded_groups["empty_or_null_text"] = ref_empty

    ref1 = ref0[ref_nonempty_mask].copy()
    audit_rows.append({"stage": "ref_nonempty_text", "n_rows": int(len(ref1)), "n_cdes": int(ref1["cde_key"].nunique())})

    # Curator-facing inventory of DocumentType values (pre-allowlist).
    ref_inv = (
        ref1.groupby(["document_type"], dropna=False)
        .agg(
            n_rows=("cde_key", "size"),
            n_unique_cdes=("cde_key", "nunique"),
            n_unique_names=("name", "nunique"),
            n_unique_texts=("document_text", "nunique"),
        )
        .reset_index()
        .sort_values(["n_rows", "n_unique_cdes"], ascending=False)
    )
    ref_inv_path = os.path.join(args.out_dir, "refdoc_type_inventory.csv")
    ref_inv.to_csv(ref_inv_path, index=False)

    # Load strict REF DocumentType allowlist (literal-exact).
    ref_allow_df, ref_allowed_types = _load_refdoc_allowlist_csv(args.refdoc_allowlist)
    ref_allowed_set = set(ref_allowed_types)

    # Report allowlist rows with zero matches in pre-allowlist REF candidates.
    present_doc_types = set(ref1["document_type"].drop_duplicates().astype(str))
    ref_unmatched_rows: List[Dict[str, Any]] = []
    for _i, r in ref_allow_df.iterrows():
        t = str(r["document_type"])
        if t not in present_doc_types:
            ref_unmatched_rows.append({"document_type": t, "reason": "no_matches_in_ref_candidates"})
    ref_unmatched_df = pd.DataFrame(ref_unmatched_rows, columns=["document_type", "reason"])
    ref_unmatched_path = os.path.join(args.out_dir, "refdoc_allowlist_unmatched.csv")
    ref_unmatched_df.to_csv(ref_unmatched_path, index=False)
    if len(ref_unmatched_df) > 0:
        print(f"WARNING: {len(ref_unmatched_df)} REF allowlist rows had no matches. See: {ref_unmatched_path}")

    # DocumentType allowlist filter (literal-exact).
    ref2 = ref1[ref1["document_type"].isin(ref_allowed_set)].copy()
    audit_rows.append({"stage": "ref_in_doc_type_allowlist", "n_rows": int(len(ref2)), "n_cdes": int(ref2["cde_key"].nunique())})

    # Top excluded DocumentTypes (pre-allowlist minus allowlist).
    ref_excluded = ref1[~ref1["document_type"].isin(ref_allowed_set)].copy()

    # Exclusion group: DocumentType not allowlisted (English + non-empty).
    ref_excluded_groups["doctype_not_allowlisted"] = ref_excluded.copy()
    ref_excl_by_type = (
        ref_excluded.groupby(["document_type"], dropna=False)
        .agg(
            n_rows=("cde_key", "size"),
            n_unique_cdes=("cde_key", "nunique"),
            n_unique_names=("name", "nunique"),
            n_unique_texts=("document_text", "nunique"),
        )
        .reset_index()
        .sort_values(["n_rows", "n_unique_cdes"], ascending=False)
    )
    ref_excl_path = os.path.join(args.out_dir, "top10_excluded_refdoc_types.csv")
    ref_excl_by_type.head(int(top_n)).to_csv(ref_excl_path, index=False)

    ref2["family"] = ref2["name"].map(_assign_ref_family)

    # Exclusion group: name-family pattern mismatch (only in strict mode).
    if args.allow_other_ref_names:
        ref_excluded_groups["name_family_excluded_by_rules"] = ref2.head(0).copy()
    else:
        ref_excluded_groups["name_family_excluded_by_rules"] = ref2[ref2["family"] == "OTHER"].copy()

    if args.allow_other_ref_names:
        ref3 = ref2.copy()
        audit_rows.append({"stage": "ref_allow_other_names", "n_rows": int(len(ref3)), "n_cdes": int(ref3["cde_key"].nunique())})
    else:
        ref3 = ref2[ref2["family"] != "OTHER"].copy()
        audit_rows.append({"stage": "ref_in_name_family_allowlist", "n_rows": int(len(ref3)), "n_cdes": int(ref3["cde_key"].nunique())})

        # If this unexpectedly drops everything, write debugging artifacts.
        if len(ref2) > 0 and len(ref3) == 0:
            # Top names in ref2
            top_names = ref2["name"].value_counts().head(200).reset_index()
            top_names.columns = ["name", "n_rows"]
            top_names.to_csv(os.path.join(args.out_dir, "DEBUG_top_ref_names_after_doctype_allowlist.csv"), index=False)

    # -----------------------------------------------------------------
    # REF: rule-based exclusion artifacts (curator-facing)
    # -----------------------------------------------------------------
    def _add_ref_meta(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return df_in
        d = df_in.copy()
        d["is_doctype_allowlisted"] = d["document_type"].isin(ref_allowed_set)
        d["assigned_family"] = d["name"].map(_assign_ref_family)
        return d

    for _k in list(ref_excluded_groups.keys()):
        ref_excluded_groups[_k] = _add_ref_meta(ref_excluded_groups[_k])

    ref_rule_order = [
        "non_english",
        "empty_or_null_text",
        "doctype_not_allowlisted",
        "name_family_excluded_by_rules",
    ]
    ref_rule_notes = {
        "non_english": "Filtered out by Language != requested language.",
        "empty_or_null_text": "Filtered out because document_text was empty or null.",
        "doctype_not_allowlisted": "Filtered out because document_type was not in the strict REF document_type allowlist.",
        "name_family_excluded_by_rules": "Filtered out because Name did not match a strict provenance family pattern (family=OTHER).",
    }

    ref_rule_excl_summary_path, ref_rule_excl_samples_path = _write_rule_exclusion_artifacts(
        out_dir=args.out_dir,
        source="REF",
        rule_order=ref_rule_order,
        rule_notes=ref_rule_notes,
        excluded_groups=ref_excluded_groups,
        text_col="document_text",
        bucket_cols=["document_type", "name"],
        sample_cols=[
            "cde_publicid",
            "cde_version",
            "cde_key",
            "document_type",
            "name",
            "language",
            "raw_text",
            "normalized_text",
            "is_doctype_allowlisted",
            "assigned_family",
        ],
        sample_n=int(args.excluded_sample_n),
        sample_seed=int(args.excluded_sample_seed),
        summary_filename="ref_rule_excluded_summary.csv",
        samples_filename="ref_rule_excluded_samples.csv",
    )

    # Deduplicate within-CDE query instances (REFDOC).
    #
    # Safe default provenance key includes: DocumentType + Name + language.
    ref_before = int(len(ref3))
    ref_pairs = _dedupe_within_cde_keep_provenance(
        ref3,
        text_col="document_text",
        key_cols=["document_type", "name", "language"],
        report_path=os.path.join(args.out_dir, "ref_dedupe_report.csv"),
    )
    ref_after = int(len(ref_pairs))
    audit_rows.append({"stage": "ref_after_dedupe", "n_rows": ref_after, "n_cdes": int(ref_pairs["cde_key"].nunique())})

    # -----------------------------------------------------------------
    # Optional PV summaries to attach to query_text (Q2)
    # -----------------------------------------------------------------
    pv_summary: Optional[pd.DataFrame] = None
    if pv_parquet_path is not None and os.path.exists(pv_parquet_path):
        # Only summarize PVs for CDEs that survived strict filtering.
        keep_cde_ids = set(
            pd.concat(
                [
                    alt_pairs[["cde_publicid", "cde_version"]],
                    ref_pairs[["cde_publicid", "cde_version"]],
                ],
                ignore_index=True,
            )
            .astype(str)
            .agg("::".join, axis=1)
            .unique()
            .tolist()
        )

        # Read a narrow PV schema; tolerate missing optional columns.
        pv_cols = [
            "cde_publicid",
            "cde_version",
            "valid_value",
            "value_meaning",
            "value_meaning_long_name",
            "meaning_description",
            "meaning_concept_display_order",
        ]
        try:
            pv_df = pd.read_parquet(pv_parquet_path, columns=pv_cols)
        except Exception:
            # Fallback: read without column filtering (older parquet engines / drift).
            pv_df = pd.read_parquet(pv_parquet_path)
        for c in ["cde_publicid", "cde_version"]:
            if c in pv_df.columns:
                pv_df[c] = pv_df[c].astype(str)
        pv_df["_cde_id"] = pv_df["cde_publicid"].astype(str) + "::" + pv_df["cde_version"].astype(str)
        pv_df = pv_df[pv_df["_cde_id"].isin(keep_cde_ids)].copy()
        pv_df = pv_df.drop(columns=["_cde_id"])

        if not pv_df.empty:
            pv_summary = build_pv_summary_table(
                pv_df,
                profile=str(args.pv_summary_profile),
                pv_max_n=int(args.pv_max_n),
                pv_huge_threshold=int(args.pv_huge_threshold),
                sde_generic_label_p=float(args.sde_generic_label_p),
                salt=str(args.pv_hash_salt),
                pv_max_n_query=args.pv_max_n_query,
                pv_max_n_cde=args.pv_max_n_cde,
                pv_center_fraction=float(args.pv_center_fraction),
                pv_min_n=int(args.pv_min_n),
                pv_size_jitter=int(args.pv_size_jitter),
                pv_generic_boost=float(args.pv_generic_boost),
                pv_generic_cap=float(args.pv_generic_cap),
                pv_small_n_generic_threshold=float(args.pv_small_n_generic_threshold),
                pv_max_resample_attempts=int(args.pv_max_resample_attempts),
                pv_placeholder_token=str(args.pv_placeholder_token),
                use_placeholder_for_query_omission=False,
                diagnostics_summary_path=args.pv_diagnostics_summary,
                diagnostics_records_path=args.pv_diagnostics_records,
            )

    def _attach_pv_block(df_q: pd.DataFrame, *, label: str) -> pd.DataFrame:
        """Attach PV summary to the *prefixed* query_text (Q2), keeping Q1 unchanged.

        Query variants emitted by this module:
          - Q1: query_text_raw (no prefixes, no PV)
          - Q2: query_text (prefix-labeled, PV attached with provenance labels)
          - Q3: query_text_q3 (no prefixes, PV attached without provenance labels)
          - Q4: query_text_q4 (no prefixes, PV always present; uses placeholder when missing)
        """
        d = df_q.copy()
        if pv_summary is None or pv_summary.empty:
            d["PV_N"] = 0
            d["PV_TYPE"] = ""
            d["PV_BLOCK_SDE"] = ""
            d["PV_BLOCK_CDE"] = ""
            d["pv_attached"] = False
            return d

        d = d.merge(pv_summary, on=["cde_publicid", "cde_version"], how="left")
        d["PV_N"] = pd.to_numeric(d["PV_N"], errors="coerce").fillna(0).astype(int)
        d["PV_TYPE"] = d.get("PV_TYPE", "").fillna("").astype(str)
        d["PV_BLOCK_SDE"] = d.get("PV_BLOCK_SDE", "").fillna("").astype(str)
        d["PV_BLOCK_CDE"] = d.get("PV_BLOCK_CDE", "").fillna("").astype(str)

        d["pv_attached"] = (d["PV_N"] > 0) & d["PV_BLOCK_SDE"].map(lambda x: normalize_query_text(x) != "")
        m = d["pv_attached"]
        if m.any():
            d.loc[m, "query_text"] = d.loc[m, "query_text"].astype(str) + " | " + str(label) + ": " + d.loc[m, "PV_BLOCK_SDE"].astype(str)
        return d

    def _add_q3_nolabel(df_q: pd.DataFrame) -> pd.DataFrame:
        """Create Q3 (label-free) query text with PV block.

        Q3 is intended for embedding models that may be harmed by artificial
        prefixes like "ALT_NAME:"/"REF_TEXT:" and labels like
        "VALUE_DOMAIN:"/"ANSWER_CHOICES:".

        - If PV is attached, Q3 becomes: "<query_text_raw> | <PV_BLOCK_SDE>"
        - Otherwise, Q3 equals Q1 (query_text_raw).
        """
        d = df_q.copy()
        d["query_text_q3"] = d["query_text_raw"].fillna("").astype(str)
        if "pv_attached" not in d.columns:
            return d
        m = d["pv_attached"].fillna(False)
        if "PV_BLOCK_SDE" in d.columns:
            m = m & d["PV_BLOCK_SDE"].map(lambda x: normalize_query_text(x) != "")
        if m.any():
            d.loc[m, "query_text_q3"] = d.loc[m, "query_text_raw"].fillna("").astype(str) + " | " + d.loc[m, "PV_BLOCK_SDE"].fillna("").astype(str)
        return d

    def _add_q4_nolabel_placeholder(df_q: pd.DataFrame, *, pv_placeholder: str = "<MISSING_PV_SUMMARY>") -> pd.DataFrame:
        """Create Q4 (label-free) query text with a PV slot *always present*.

        Q4 is like Q3, but when PV is not attached we append a placeholder token.

        - If PV is attached: Q4 = "<query_text_raw> | <PV_BLOCK_SDE>"
        - If PV is not attached (or PV block is empty after normalization):
              Q4 = "<query_text_raw> | <MISSING_PV_SUMMARY>"
        """
        d = df_q.copy()
        raw = d["query_text_raw"].fillna("").astype(str)
        if "PV_BLOCK_SDE" in d.columns:
            pv_block = d["PV_BLOCK_SDE"].fillna("").astype(str)
        else:  # defensive fallback
            pv_block = pd.Series([""] * len(d), index=d.index).astype(str)

        # Default: placeholder appended.
        # If raw is empty, fall back to the placeholder alone.
        base = raw.map(lambda s: s.strip())
        d["query_text_q4"] = base.map(lambda s: (s + " | " + str(pv_placeholder)) if s else str(pv_placeholder))

        if "pv_attached" not in d.columns:
            return d

        m = d["pv_attached"].fillna(False)
        # Only treat PV as attached if PV block is non-empty after normalization.
        m = m & pv_block.map(lambda x: normalize_query_text(x) != "")
        if m.any():
            d.loc[m, "query_text_q4"] = base.loc[m].astype(str) + " | " + pv_block.loc[m].astype(str)
        return d

    # ---------------------
    # Write row-level query records for downstream training/splitting
    # ---------------------
    # ALT rows -> unified schema
    alt_q = alt_pairs.copy()
    alt_q["query_source"] = "ALT"
    alt_q["query_field"] = "alternate_name"
    alt_q["query_text_raw"] = alt_q["alternate_name"]
    alt_q["query_text"] = "ALT_NAME: " + alt_q["alternate_name"]
    alt_q["cde_id"] = alt_q["cde_publicid"].astype(str) + "::" + alt_q["cde_version"].astype(str)
    alt_q["query_id"] = ("ALT::" + alt_q["query_text_raw"]).map(_sha1_hex)
    alt_q["pair_id"] = (
        "ALT::"
        + alt_q["query_text_raw"]
        + "::"
        + alt_q["cde_publicid"].astype(str)
        + "::"
        + alt_q["cde_version"].astype(str)
    ).map(_sha1_hex)

    # REF rows -> unified schema
    ref_q = ref_pairs.copy()
    ref_q["query_source"] = "REF"
    ref_q["query_field"] = "document_text"
    ref_q["query_text_raw"] = ref_q["document_text"]
    ref_q["query_text"] = "REF_TEXT: " + ref_q["document_text"]
    ref_q["cde_id"] = ref_q["cde_publicid"].astype(str) + "::" + ref_q["cde_version"].astype(str)
    ref_q["query_id"] = ("REF::" + ref_q["query_text_raw"]).map(_sha1_hex)
    ref_q["pair_id"] = (
        "REF::"
        + ref_q["query_text_raw"]
        + "::"
        + ref_q["cde_publicid"].astype(str)
        + "::"
        + ref_q["cde_version"].astype(str)
    ).map(_sha1_hex)

    # Attach PV summaries to query_text (Q2) if available.
    # Q1 (query_text_raw) remains unchanged.
    alt_q = _attach_pv_block(alt_q, label="VALUE_DOMAIN")
    ref_q = _attach_pv_block(ref_q, label="ANSWER_CHOICES")

    # Create Q3 (label-free PV-augmented text).
    alt_q = _add_q3_nolabel(alt_q)
    ref_q = _add_q3_nolabel(ref_q)

    # Create Q4 (label-free PV-augmented text with placeholder when PV missing).
    alt_q = _add_q4_nolabel_placeholder(alt_q)
    ref_q = _add_q4_nolabel_placeholder(ref_q)

    # Column order (keep provenance columns when present)
    common_cols = [
        "pair_id",
        "query_id",
        "query_source",
        "query_field",
        "query_text_raw",
        "query_text",
        "query_text_q3",
        "query_text_q4",
        "cde_publicid",
        "cde_version",
        # Backwards-compatible convenience key (used in tests and downstream analysis).
        # This is identical to cde_id (publicid::version), but we keep both names
        # to avoid breaking older workflows.
        "cde_key",
        "cde_id",
        "family",
        "language",
        "PV_N",
        "PV_TYPE",
        "PV_BLOCK_SDE",
        "PV_BLOCK_CDE",
        "pv_attached",
    ]
    alt_extra = [
        "context_name",
        "context_version",
        "alternate_name_type",
        "alt_query_mode",
        "alt_recipe_id",
        "alt_component_a_type",
        "alt_component_b_type",
        "alt_component_a",
        "alt_component_b",
        "source_file",
        "cde_xml_row_id",
    ]
    ref_extra = [
        "document_type",
        "name",
        "url",
        "display_order",
        "source_file",
        "cde_xml_row_id",
    ]

    def _select_cols(df, cols):
        keep = [c for c in cols if c in df.columns]
        return df[keep].copy()

    alt_out = _select_cols(alt_q, common_cols + alt_extra)
    ref_out = _select_cols(ref_q, common_cols + ref_extra)

    queries = pd.concat([alt_out, ref_out], ignore_index=True)

    # Write row-level queries to out_dir/queries.parquet (always).
    out_dir_parquet = os.path.join(args.out_dir, "queries.parquet")
    queries.to_parquet(out_dir_parquet, index=False)
    print("Wrote row-level queries to:", out_dir_parquet)

    # Optionally also write to a separate parquet path (for downstream pipelines).
    # This preserves backward compatibility with workflows that expect queries in data/processed/.
    if args.out_parquet:
        if os.path.abspath(args.out_parquet) != os.path.abspath(out_dir_parquet):
            os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
            queries.to_parquet(args.out_parquet, index=False)
            print("Also wrote row-level queries to:", args.out_parquet)

    # ---------------------
    # Headline totals
    # ---------------------
    def _summarize_pairs(df_pairs: pd.DataFrame, text_col: str) -> Dict[str, int]:
        return {
            "n_query_cde_pairs": int(len(df_pairs)),
            "n_unique_query_texts": int(df_pairs[text_col].nunique()),
            "n_unique_cdes": int(df_pairs[["cde_publicid", "cde_version"]].drop_duplicates().shape[0]),
        }

    alt_summary = _summarize_pairs(alt_pairs, "alternate_name")
    ref_summary = _summarize_pairs(ref_pairs, "document_text")

    # Union summary (treat alt and ref as separate query modalities)
    total_pairs = alt_summary["n_query_cde_pairs"] + ref_summary["n_query_cde_pairs"]

    # CDE overlap stats
    alt_cdes = set(alt_pairs["cde_key"].unique())
    ref_cdes = set(ref_pairs["cde_key"].unique())
    both_cdes = alt_cdes.intersection(ref_cdes)

    overlap_stats = {
        "n_cdes_with_alt_queries": int(len(alt_cdes)),
        "n_cdes_with_ref_queries": int(len(ref_cdes)),
        "n_cdes_with_both": int(len(both_cdes)),
    }

    # ---------------------
    # Summaries by bucket
    # ---------------------
    alt_by_bucket = (
        alt_pairs.groupby(["alternate_name_type", "context_name", "family"], dropna=False)
        .agg(
            n_query_cde_pairs=("cde_key", "size"),
            n_unique_cdes=("cde_key", "nunique"),
            n_unique_texts=("alternate_name", "nunique"),
        )
        .reset_index()
        .sort_values(["n_unique_cdes", "n_query_cde_pairs"], ascending=False)
    )

    ref_by_bucket = (
        ref_pairs.groupby(["document_type", "name", "family"], dropna=False)
        .agg(
            n_query_cde_pairs=("cde_key", "size"),
            n_unique_cdes=("cde_key", "nunique"),
            n_unique_texts=("document_text", "nunique"),
        )
        .reset_index()
        .sort_values(["n_unique_cdes", "n_query_cde_pairs"], ascending=False)
    )

    # ---------------------
    # Summaries by family
    # ---------------------
    alt_by_family = (
        alt_pairs.groupby(["family"], dropna=False)
        .agg(
            n_query_cde_pairs=("cde_key", "size"),
            n_unique_cdes=("cde_key", "nunique"),
            n_unique_texts=("alternate_name", "nunique"),
            n_contexts=("context_name", "nunique"),
        )
        .reset_index()
        .sort_values(["n_unique_cdes", "n_query_cde_pairs"], ascending=False)
    )

    ref_by_family = (
        ref_pairs.groupby(["family"], dropna=False)
        .agg(
            n_query_cde_pairs=("cde_key", "size"),
            n_unique_cdes=("cde_key", "nunique"),
            n_unique_texts=("document_text", "nunique"),
            n_doc_types=("document_type", "nunique"),
        )
        .reset_index()
        .sort_values(["n_unique_cdes", "n_query_cde_pairs"], ascending=False)
    )

    # Ref by Name (debugging/exploration)
    ref_by_name = (
        ref_pairs.groupby(["name"], dropna=False)
        .agg(
            n_query_cde_pairs=("cde_key", "size"),
            n_unique_cdes=("cde_key", "nunique"),
            n_unique_texts=("document_text", "nunique"),
            n_families=("family", "nunique"),
        )
        .reset_index()
        .sort_values(["n_unique_cdes", "n_query_cde_pairs"], ascending=False)
    )

    # Alternate by Context (org/study size)
    alt_by_context = (
        alt_pairs.groupby(["context_name"], dropna=False)
        .agg(
            n_query_cde_pairs=("cde_key", "size"),
            n_unique_cdes=("cde_key", "nunique"),
            n_unique_texts=("alternate_name", "nunique"),
            n_types=("alternate_name_type", "nunique"),
        )
        .reset_index()
        .sort_values(["n_unique_cdes", "n_query_cde_pairs"], ascending=False)
    )

    # ---------------------
    # Holdout candidates (by family)
    # ---------------------
    alt_holdout = alt_by_family[
        (alt_by_family["n_unique_cdes"] >= args.min_cdes) & (alt_by_family["n_query_cde_pairs"] >= args.min_queries)
    ].copy()
    ref_holdout = ref_by_family[
        (ref_by_family["n_unique_cdes"] >= args.min_cdes) & (ref_by_family["n_query_cde_pairs"] >= args.min_queries)
    ].copy()

    # ---------------------
    # Write outputs
    # ---------------------
    pd.DataFrame(audit_rows).to_csv(os.path.join(args.out_dir, "audit_filter_stages.csv"), index=False)

    alt_by_bucket.to_csv(os.path.join(args.out_dir, "alt_inwild_by_bucket.csv"), index=False)
    alt_by_family.to_csv(os.path.join(args.out_dir, "alt_inwild_by_family.csv"), index=False)
    alt_by_context.to_csv(os.path.join(args.out_dir, "alt_inwild_by_context.csv"), index=False)
    alt_holdout.to_csv(os.path.join(args.out_dir, "holdout_candidates_alt_families.csv"), index=False)

    ref_by_bucket.to_csv(os.path.join(args.out_dir, "ref_inwild_by_bucket.csv"), index=False)
    ref_by_family.to_csv(os.path.join(args.out_dir, "ref_inwild_by_family.csv"), index=False)
    ref_by_name.to_csv(os.path.join(args.out_dir, "ref_inwild_by_name.csv"), index=False)
    ref_holdout.to_csv(os.path.join(args.out_dir, "holdout_candidates_ref_families.csv"), index=False)

    summary = {
        "inputs": {"alt_parquet": args.alt_parquet, "ref_parquet": args.ref_parquet},
        "alt_allowlist": {
            "path": args.alt_allowlist,
            "n_rows": int(len(allow_df)),
            "exact_only": True,
            "literal_exact": True,
            "case_sensitive": True,
            "whitespace_sensitive": True,
            "n_unmatched": int(len(unmatched_df)),
        },
        "refdoc_allowlist": {
            "path": args.refdoc_allowlist,
            "n_rows": int(len(ref_allow_df)),
            "exact_only": True,
            "literal_exact": True,
            "case_sensitive": True,
            "whitespace_sensitive": True,
            "n_unmatched": int(len(ref_unmatched_df)),
        },
        "thresholds": {"min_cdes": args.min_cdes, "min_queries": args.min_queries, "excluded_top_n": int(top_n)},
        "outputs": {
            "alt_candidate_inventory_csv": inv_path if not args.skip_alt_candidate_inventory else None,
            "top_excluded_alt_buckets_csv": excl_path,
            "alt_allowlist_unmatched_csv": unmatched_path,
            "alt_rule_excluded_summary_csv": alt_rule_excl_summary_path,
            "alt_rule_excluded_samples_csv": alt_rule_excl_samples_path,
            "alt_dedupe_report_csv": os.path.join(args.out_dir, "alt_dedupe_report.csv"),
            "refdoc_type_inventory_csv": ref_inv_path,
            "top_excluded_refdoc_types_csv": ref_excl_path,
            "refdoc_allowlist_unmatched_csv": ref_unmatched_path,
            "ref_rule_excluded_summary_csv": ref_rule_excl_summary_path,
            "ref_rule_excluded_samples_csv": ref_rule_excl_samples_path,
            "ref_dedupe_report_csv": os.path.join(args.out_dir, "ref_dedupe_report.csv"),
        },
        "dedupe": {
            "alt": {"n_rows_before": int(alt_before), "n_rows_after": int(alt_after), "n_removed": int(alt_before - alt_after)},
            "ref": {"n_rows_before": int(ref_before), "n_rows_after": int(ref_after), "n_removed": int(ref_before - ref_after)},
        },
        "alt_summary": alt_summary,
        "ref_summary": ref_summary,
        "n_total_query_cde_pairs": int(total_pairs),
        "overlap_cde_stats": overlap_stats,
        "n_alt_contexts": int(alt_pairs["context_name"].nunique()),
        "ref_families": sorted([f for f in ref_pairs["family"].unique() if f and f != "OTHER"]),
        "alt_families": sorted([f for f in alt_pairs["family"].unique() if f and f != "OTHER"]),
    }

    with open(os.path.join(args.out_dir, "inwild_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote outputs to:", args.out_dir)
    print("Headline totals:")
    print("  ALT query->CDE pairs:", alt_summary["n_query_cde_pairs"], "(unique CDEs:", alt_summary["n_unique_cdes"], ")")
    print("  REF query->CDE pairs:", ref_summary["n_query_cde_pairs"], "(unique CDEs:", ref_summary["n_unique_cdes"], ")")
    print("  TOTAL query->CDE pairs:", total_pairs)


if __name__ == "__main__":
    main()