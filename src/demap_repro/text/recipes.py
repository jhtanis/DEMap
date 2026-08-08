"""CDE text recipe construction.

This module defines the *target-side* CDE text representations (v1-v6) and
utilities to build a CDE catalog (cde_id -> cde_text) for retrieval.

Atomic recipes
--------------
The atomic recipes are:

  v1  SHORT_NAME
  v2  LONG_NAME | DEFINITION
  v2a LONG_NAME
  v2b DEFINITION
  v3  PREFERRED_QUESTION_TEXT
  v4  VALUE_DOMAIN_TYPE | VALUE_DOMAIN_DATATYPE
  v5  PV_SUMMARY
  v6  DEC_LONG_NAME

Composite recipes
-----------------
In addition to atomic recipes, the repo supports *composite* recipes expressed
as an underscore-joined list of atomic recipe names, e.g.:

  v1_v2_v3
  v1_v3_v5_v6

The ordering is significant and preserved left-to-right.

Formatting
----------
Formatting is controlled by (cde_format, sep):

  labeled : include field labels (e.g., "LONG_NAME: ...")
  raw     : values only

The baseline experiment runner (:mod:`demap_repro.biencoder.engine.baseline_grid`) imports
two public symbols from this module:

  - RECIPE_FIELDS (atomic recipes only)
  - build_catalog(...)
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Mapping, Optional

import pandas as pd


def norm_str(x: object) -> str:
    """Normalize a value to a compact string."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", s)


# Numeric-only SHORT_NAME filtering
_NUMERIC_ONLY_RE = re.compile(r"^[0-9._-]+$")


def is_numeric_only_short_name(s: str) -> bool:
    """Return True if s contains *no* alphabet characters and is only digits/._-."""
    s2 = norm_str(s)
    if not s2:
        return False
    return bool(_NUMERIC_ONLY_RE.fullmatch(s2))

# Versioned numeric ID-pair SHORT_NAME filtering (e.g., "3196264v1.0:4867450v1.0")
_VERSIONED_ID_PAIR_RE = re.compile(r"^\d+v\d+(?:\.\d+)+(?::\d+v\d+(?:\.\d+)+)+$")


def is_versioned_id_pair_short_name(s: str) -> bool:
    """Return True if s looks like a versioned numeric ID pair (e.g., '123v1.0:456v1.0').

    This is intended to filter out SHORT_NAME values that are mostly numeric/gibberish but
    include version markers and delimiters (and therefore are not caught by the
    numeric-only filter).
    """
    s2 = norm_str(s)
    if not s2:
        return False
    return bool(_VERSIONED_ID_PAIR_RE.fullmatch(s2))



# -----------------------------------------------------------------------------
# Recipes (atomic)
# -----------------------------------------------------------------------------

RECIPE_FIELDS: Dict[str, List[str]] = {
    "v1": ["SHORT_NAME"],
    "v2": ["LONG_NAME", "DEFINITION"],
    "v2a": ["LONG_NAME"],
    "v2b": ["DEFINITION"],
    "v3": ["PREFERRED_QUESTION_TEXT"],
    "v4": ["VALUE_DOMAIN_TYPE", "VALUE_DOMAIN_DATATYPE"],
    "v5": ["PV_SUMMARY"],
    "v6": ["DEC_LONG_NAME"],
}


# Human-readable aliases for the cde_ai collaborator quickstart. These resolve to
# existing vN composite recipes inside parse_recipe (the single parse chokepoint),
# so every consumer (build_catalog / is_valid_recipe / finetune) accepts them.
# Additive only: vN recipes are unaffected (no vN string matches an alias key).
RECIPE_ALIASES: Dict[str, str] = {
    # SN + DEC + DEFINITION (v2b, NOT LONG_NAME|DEFINITION) + PQT + PV. DEC already carries the
    # DEC long name, so LONG_NAME (v2) is redundant here; the verified paper finalist used v2b.
    "SN_DEC_DEF_PQT_PV": "v1_v6_v2b_v3_v5",  # SHORT_NAME, DEC, DEFINITION, PQT, PV_SUMMARY
    "SN_DEC_DEF_PQT": "v1_v6_v2b_v3",        # same minus PV_SUMMARY (no PV)
}


def parse_recipe(recipe: str) -> List[str]:
    """Parse an atomic or composite recipe name into atomic components.

    Examples
    --------
    - "v2" -> ["v2"]
    - "v1_v2_v3" -> ["v1", "v2", "v3"]

    Rules
    -----
    - Composite recipes are underscore-joined lists of atomic recipe names.
    - All parts must be known atomic recipes (keys of RECIPE_FIELDS).
    - No duplicate parts are allowed (e.g., "v1_v1" is invalid).
    - Ordering is preserved exactly as written left-to-right.
    """
    r = str(recipe).strip()
    r = RECIPE_ALIASES.get(r, r)  # resolve cde_ai human-readable aliases (e.g. SN_DEC_DEF_PQT_PV)
    if r in RECIPE_FIELDS:
        return [r]
    if "_" not in r:
        raise ValueError(f"Unknown CDE recipe: {r}. Expected one of {sorted(RECIPE_FIELDS)} or a composite like v1_v2_v3")

    parts = [p.strip() for p in r.split("_") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid recipe string: {recipe!r}")

    unknown = [p for p in parts if p not in RECIPE_FIELDS]
    if unknown:
        raise ValueError(
            f"Unknown CDE recipe part(s): {unknown} in {r}. "
            f"Expected parts from {sorted(RECIPE_FIELDS)}"
        )

    if len(parts) != len(set(parts)):
        raise ValueError(f"Composite recipe has duplicate parts (not allowed): {r}")

    return parts


def is_valid_recipe(recipe: str) -> bool:
    try:
        parse_recipe(recipe)
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class CdeTextBuildConfig:
    """Configuration for building CDE text."""

    cde_format: str = "labeled"  # "labeled" or "raw"
    sep: str = " | "
    v1_filter_numeric_only: bool = True
    v1_filter_versioned_id_short_name: bool = False
    # When a field is empty/missing after cleaning (or filtered), either omit it
    # entirely (default behavior) or insert a placeholder token.
    #
    # This policy currently applies to:
    #   - SHORT_NAME (v1)
    #   - PV_SUMMARY (v5)
    placeholder_policy: str = "omit"  # "omit" | "placeholder"
    short_name_placeholder: str = "<MISSING_SHORT_NAME>"
    pv_placeholder: str = "<MISSING_PV_SUMMARY>"


def required_fields_for_recipe(recipe: str) -> List[str]:
    """Return required master-table columns for an atomic or composite recipe."""
    parts = parse_recipe(recipe)
    out: List[str] = []
    seen = set()
    for p in parts:
        for f in RECIPE_FIELDS[p]:
            if f in seen:
                continue
            seen.add(f)
            out.append(f)
    return out


def build_cde_text_from_row(
    row: Mapping[str, object],
    *,
    recipe: str,
    cde_format: str = "labeled",
    sep: str = " | ",
    v1_filter_numeric_only: bool = True,
    v1_filter_versioned_id_short_name: bool = False,
    placeholder_policy: str = "omit",
    short_name_placeholder: str = "<MISSING_SHORT_NAME>",
    pv_placeholder: str = "<MISSING_PV_SUMMARY>",
) -> str:
    """Build a single CDE text string from one row of the enriched master.

    Supports both atomic and composite recipes.
    """
    fmt = str(cde_format).strip().lower()
    if fmt not in {"labeled", "raw"}:
        raise ValueError("cde_format must be 'labeled' or 'raw'")

    parts = parse_recipe(recipe)
    fields = required_fields_for_recipe(recipe)

    pol = str(placeholder_policy).strip().lower()
    if pol not in {"omit", "placeholder"}:
        raise ValueError("placeholder_policy must be 'omit' or 'placeholder'")

    out_parts: List[str] = []
    for field in fields:
        val = norm_str(row.get(field, ""))

        # v1 SHORT_NAME filtering options (applies to SHORT_NAME field)
        if field == "SHORT_NAME":
            if v1_filter_numeric_only and is_numeric_only_short_name(val):
                val = ""
            if v1_filter_versioned_id_short_name and is_versioned_id_pair_short_name(val):
                val = ""

        # Placeholder policy (CDE-side only):
        # If a field is empty after cleaning/filtering, either omit it (default)
        # or insert a stable placeholder token so the downstream embedding model
        # still receives a "slot" for that field. Placeholders are intentionally limited
        # to SHORT_NAME and PV_SUMMARY; all other fields (DEC/LONG_NAME/DEFINITION/PQT)
        # follow the pre-existing omit-when-missing behavior.
        if not val and pol == "placeholder":
            if field == "SHORT_NAME":
                val = norm_str(short_name_placeholder)
            elif field == "PV_SUMMARY":
                val = norm_str(pv_placeholder)

        if not val:
            continue

        if fmt == "labeled":
            out_parts.append(f"{field}: {val}")
        else:
            out_parts.append(val)

    return str(sep).join(out_parts)


def build_catalog(
    master: pd.DataFrame,
    *,
    recipe: str,
    cde_format: str = "labeled",
    sep: str = " | ",
    recipe_configs: Optional[Dict[str, Dict]] = None,
    id_col: str = "cde_id",
) -> pd.DataFrame:
    """Build a catalog with columns [cde_id, cde_text] for a recipe.

    Parameters
    ----------
    master:
        Enriched CDE master table.
    recipe:
        An atomic recipe (v1..v6) or a composite recipe (e.g., v1_v2_v3).
    cde_format:
        "labeled" or "raw".
    sep:
        Separator used between fields.
    recipe_configs:
        Optional dict of per-recipe configs. Currently supports:
          - recipe_configs["v1"]["filter_numeric_only"] (bool)
          - recipe_configs["v1"]["filter_versioned_id_short_name"] (bool)
          - recipe_configs["placeholder_policy"] (str: omit|placeholder)
          - recipe_configs["short_name_placeholder"] (str)
          - recipe_configs["pv_placeholder"] (str)
    id_col:
        Column containing unique CDE IDs (default: "cde_id" = "<PUBLICID>::<VERSION>").
    """
    if id_col not in master.columns:
        raise ValueError(f"master is missing required id column: {id_col}")

    # Validate recipe and determine required fields
    needed = required_fields_for_recipe(recipe)
    missing = [c for c in needed if c not in master.columns]
    if missing:
        raise ValueError(f"master missing required fields for {recipe}: {missing}")

    # Per-recipe config (only v1 currently)
    v1_filter_numeric_only = True
    v1_filter_versioned_id_short_name = False
    placeholder_policy = "omit"
    short_name_placeholder = "<MISSING_SHORT_NAME>"
    pv_placeholder = "<MISSING_PV_SUMMARY>"

    if recipe_configs and isinstance(recipe_configs.get("v1"), dict):
        v1 = recipe_configs["v1"]
        v1_filter_numeric_only = bool(v1.get("filter_numeric_only", True))
        v1_filter_versioned_id_short_name = bool(v1.get("filter_versioned_id_short_name", False))

    # Global placeholder policy (applies to SHORT_NAME + PV_SUMMARY).
    # Keep this permissive: ignore unknown keys.
    if recipe_configs and isinstance(recipe_configs, dict):
        if "placeholder_policy" in recipe_configs:
            placeholder_policy = str(recipe_configs.get("placeholder_policy") or "omit")
        if "short_name_placeholder" in recipe_configs:
            short_name_placeholder = str(recipe_configs.get("short_name_placeholder") or short_name_placeholder)
        if "pv_placeholder" in recipe_configs:
            pv_placeholder = str(recipe_configs.get("pv_placeholder") or pv_placeholder)

    ids = master[id_col].astype(str).tolist()

    texts: List[str] = []
    for _, row in master.iterrows():
        texts.append(
            build_cde_text_from_row(
                row,
                recipe=str(recipe),
                cde_format=cde_format,
                sep=sep,
                v1_filter_numeric_only=v1_filter_numeric_only,
                v1_filter_versioned_id_short_name=v1_filter_versioned_id_short_name,
                placeholder_policy=placeholder_policy,
                short_name_placeholder=short_name_placeholder,
                pv_placeholder=pv_placeholder,
            )
        )

    out = pd.DataFrame({id_col: ids, "cde_text": texts})
    out["cde_text"] = out["cde_text"].fillna("").astype(str)

    # Drop empty texts (cannot embed)
    out = out[out["cde_text"].str.strip().ne("")].copy()

    # Keep one row per CDE id
    out = out.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)

    return out
