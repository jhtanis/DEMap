"""Deterministic PV-overlap feature builder (PR-A, tiers 1-4 only).

Design note: ``artifacts_v2_balanced/pv_synonym_overlap/PV_SYNONYM_AND_CALIBRATION_DESIGN.md``
plus the "PV-overlap feature builder — design proposal" turn that
specified column names and the NaN policy.

Scope:
  - Tier 1: availability / counts (always present, never NaN — except
    `pv_count_ratio_min_over_max` when both sides are empty).
  - Tier 2: exact normalized overlap.
  - Tier 3: numeric overlap (built on ``pv_parser.numeric_pv_set``).
  - Tier 4: composition fractions (generic / code-like / numeric per
    side).
  - Tier 5 (embedding / SapBERT similarity): NOT computed; columns are
    optionally emitted as NaN stubs when ``include_embedding_stubs`` is
    True so the schema is forward-compatible.

NaN policy:
  - Counts and booleans are always present, never NaN.
  - Jaccard is NaN ONLY when both sides are empty (union empty); a
    one-sided empty pair yields Jaccard = 0.0.
  - Side-specific fractions are NaN when that side has zero items
    (denominator zero).

Leakage warning — CDE PV hydration is OPT-IN:
  All core DEMAP splits (train / val_* / test / external_holdout_* /
  CCTG REF / OID ALT-style) had their CDE-side PVs intentionally
  thinned to prevent the matching CDE's PV string from leaking into
  the model. Replacing the empty / thinned ``PV_BLOCK_CDE`` with the
  master's full ``PV_SUMMARY`` would reverse that protection and
  inflate every PV-overlap feature on those splits.

  Therefore:
    - ``hydrate_cde_from_master_when_empty`` defaults to **False**.
    - Providing ``cde_master`` to the batch function is necessary but
      NOT sufficient — the caller must also explicitly set
      ``hydrate_cde_from_master_when_empty=True`` (or
      ``prefer_cde_source="pv_summary"``).
    - The single-pair function never hydrates.
    - Hydration is appropriate only for explicit external / curated
      evaluation contexts where PV thinning was NOT part of leakage
      prevention — e.g. GDC REF / ALT and CIMAC ablations.
    - Use of hydration on core splits is a leakage bug; downstream
      code should label hydrated outputs (path, column name, report
      file) clearly so they cannot be mixed with non-hydrated ones.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from demap_repro.text.pv_parser import (
    GENERIC_PV_TOKENS,
    ParsedPV,
    is_code_like_pv,
    is_numeric_pv,
    numeric_pv_set,
    parse_pv_block,
)

__all__ = [
    "PVOverlapConfig",
    "compute_pv_overlap_features",
    "compute_pv_overlap_features_batch",
    "compute_pv_overlap_for_union",
    "feature_columns",
]


# Stable union of column names emitted by tier-5 embedding features
# (excluding per-tau columns, which depend on config). Used to define
# the stub schema and to filter columns out of test-time comparisons.
_EMBEDDING_AGGREGATE_COLUMNS: Tuple[str, ...] = (
    "pv_sim_max",
    "pv_sim_p90",
    "pv_sim_mean_best_sde_to_cde",
    "pv_sim_mean_best_cde_to_sde",
    "pv_sim_min_best_sde_to_cde",
)


@dataclass(frozen=True)
class PVOverlapConfig:
    """Configuration for ``compute_pv_overlap_features``.

    All flags default to a v1 "tiers 1-4 only" build that produces a
    deterministic, embedding-free feature row. Embedding / SapBERT
    columns are emitted only when ``enable_embedding`` or
    ``include_embedding_stubs`` is True.
    """

    enable_availability: bool = True
    enable_exact: bool = True
    enable_numeric: bool = True
    enable_composition: bool = True
    enable_embedding: bool = False
    include_embedding_stubs: bool = False
    similarity_thresholds: Tuple[float, ...] = (0.82,)
    # CDE hydration (batch-only). Default OFF — hydrating thinned
    # PV_BLOCK_CDE from PV_SUMMARY would reverse leakage protection on
    # core DEMAP splits. See the module docstring for the full caveat.
    # Hydration is appropriate only for explicit external / curated
    # ablations (GDC REF/ALT, CIMAC).
    hydrate_cde_from_master_when_empty: bool = False
    prefer_cde_source: str = "pv_block_cde"  # "pv_block_cde" | "pv_summary"
    warn_on_cde_source_disagreement: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_parsed(value: Union[str, ParsedPV, None]) -> ParsedPV:
    """Normalize the per-pair input to a ``ParsedPV``."""
    if isinstance(value, ParsedPV):
        return value
    return parse_pv_block(value)


def _is_blank_str(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _tau_label(tau: float) -> str:
    """Deterministic, compact label for a similarity threshold.

    0.82 -> "0.82", 0.8 -> "0.8", 0.825 -> "0.825".
    """
    return f"{tau:g}"


def _safe_fraction(num: int, den: int) -> float:
    """Return num/den, or NaN when den == 0."""
    if den == 0:
        return float("nan")
    return num / den


# ---------------------------------------------------------------------------
# Per-tier feature dicts
# ---------------------------------------------------------------------------


def _availability_features(sde: ParsedPV, cde: ParsedPV) -> Dict[str, object]:
    n_sde = sde.n_values
    n_cde = cde.n_values
    count_min = min(n_sde, n_cde)
    count_max = max(n_sde, n_cde)
    # ratio_min_over_max is NaN when both sides are empty (max == 0);
    # otherwise it's well-defined in [0, 1].
    ratio = float("nan") if count_max == 0 else count_min / count_max
    pv_type_match = bool(
        sde.pv_type and cde.pv_type and sde.pv_type == cde.pv_type
    )
    return {
        "pv_n_sde": n_sde,
        "pv_n_cde": n_cde,
        "pv_count_min": count_min,
        "pv_count_max": count_max,
        "pv_count_abs_diff": abs(n_sde - n_cde),
        "pv_count_ratio_min_over_max": ratio,
        "pv_both_have_pvs": (n_sde > 0) and (n_cde > 0),
        "pv_only_sde_has_pvs": (n_sde > 0) and (n_cde == 0),
        "pv_only_cde_has_pvs": (n_cde > 0) and (n_sde == 0),
        "pv_neither_has_pvs": (n_sde == 0) and (n_cde == 0),
        "pv_type_match": pv_type_match,
    }


def _exact_features(sde: ParsedPV, cde: ParsedPV) -> Dict[str, object]:
    sde_set = set(sde.pv_values_norm)
    cde_set = set(cde.pv_values_norm)
    overlap = sde_set & cde_set
    union = sde_set | cde_set
    n_sde = len(sde_set)
    n_cde = len(cde_set)
    overlap_count = len(overlap)
    # Jaccard: NaN only when both sides empty. One-sided empty -> 0.0.
    if not union:
        jaccard: float = float("nan")
    else:
        jaccard = overlap_count / len(union)
    # Non-generic overlap: same intersection minus generic tokens.
    overlap_count_nongeneric = sum(1 for v in overlap if v not in GENERIC_PV_TOKENS)
    return {
        "pv_exact_overlap_count": overlap_count,
        "pv_exact_overlap_jaccard": jaccard,
        "pv_exact_overlap_frac_sde": _safe_fraction(overlap_count, n_sde),
        "pv_exact_overlap_frac_cde": _safe_fraction(overlap_count, n_cde),
        "pv_exact_overlap_count_nongeneric": overlap_count_nongeneric,
    }


def _numeric_features(sde: ParsedPV, cde: ParsedPV) -> Dict[str, object]:
    sde_set = numeric_pv_set(sde.pv_values_norm)
    cde_set = numeric_pv_set(cde.pv_values_norm)
    overlap = sde_set & cde_set
    union = sde_set | cde_set
    n_sde = len(sde_set)
    n_cde = len(cde_set)
    overlap_count = len(overlap)
    if not union:
        jaccard: float = float("nan")
    else:
        jaccard = overlap_count / len(union)
    return {
        "pv_numeric_n_sde": n_sde,
        "pv_numeric_n_cde": n_cde,
        "pv_numeric_overlap_count": overlap_count,
        "pv_numeric_jaccard": jaccard,
        "pv_numeric_sde_overlap_fraction": _safe_fraction(overlap_count, n_sde),
        "pv_numeric_cde_overlap_fraction": _safe_fraction(overlap_count, n_cde),
        "pv_numeric_both_have": (n_sde > 0) and (n_cde > 0),
        "pv_numeric_only_sde_has": (n_sde > 0) and (n_cde == 0),
        "pv_numeric_only_cde_has": (n_cde > 0) and (n_sde == 0),
    }


def _composition_features(sde: ParsedPV, cde: ParsedPV) -> Dict[str, object]:
    def _side(values: List[str], suffix: str) -> Dict[str, object]:
        n = len(values)
        if n == 0:
            return {
                f"pv_generic_frac_{suffix}": float("nan"),
                f"pv_code_like_frac_{suffix}": float("nan"),
                f"pv_numeric_frac_{suffix}": float("nan"),
            }
        generic = sum(1 for v in values if v in GENERIC_PV_TOKENS)
        code = sum(1 for v in values if is_code_like_pv(v))
        numeric = sum(1 for v in values if is_numeric_pv(v))
        return {
            f"pv_generic_frac_{suffix}": generic / n,
            f"pv_code_like_frac_{suffix}": code / n,
            f"pv_numeric_frac_{suffix}": numeric / n,
        }

    out: Dict[str, object] = {}
    out.update(_side(sde.pv_values_norm, "sde"))
    out.update(_side(cde.pv_values_norm, "cde"))
    return out


def _embedding_stub_features(config: PVOverlapConfig) -> Dict[str, object]:
    """Tier-5 placeholder columns. All values NaN / False until a real
    SapBERT cache exists. Column names are a deterministic function of
    ``similarity_thresholds`` so the schema is locked at config time."""
    out: Dict[str, object] = {}
    for k in _EMBEDDING_AGGREGATE_COLUMNS:
        out[k] = float("nan")
    for tau in config.similarity_thresholds:
        label = _tau_label(tau)
        out[f"pv_sim_at_{label}_count_sde"] = float("nan")
        out[f"pv_sim_at_{label}_frac_sde"] = float("nan")
        out[f"pv_sim_at_{label}_count_cde"] = float("nan")
        out[f"pv_sim_at_{label}_frac_cde"] = float("nan")
        out[f"pv_sim_at_{label}_f1"] = float("nan")
    out["pv_sim_available"] = False
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_pv_overlap_features(
    sde: Union[str, ParsedPV, None],
    cde: Union[str, ParsedPV, None],
    *,
    config: PVOverlapConfig = PVOverlapConfig(),
) -> Dict[str, object]:
    """Single-pair PV-overlap features.

    Pure: never touches disk, never hydrates from a master table.
    Accepts raw PV-block strings or pre-parsed ``ParsedPV`` instances
    (the latter is recommended in tight loops since parsing is the
    only non-trivial cost). Returns a flat dict of scalar features
    (int / float / bool). The output schema is a deterministic
    function of the config alone — same keys, same order, every call.
    """
    sde_parsed = _to_parsed(sde)
    cde_parsed = _to_parsed(cde)
    out: Dict[str, object] = {}
    if config.enable_availability:
        out.update(_availability_features(sde_parsed, cde_parsed))
    if config.enable_exact:
        out.update(_exact_features(sde_parsed, cde_parsed))
    if config.enable_numeric:
        out.update(_numeric_features(sde_parsed, cde_parsed))
    if config.enable_composition:
        out.update(_composition_features(sde_parsed, cde_parsed))
    if config.enable_embedding or config.include_embedding_stubs:
        # Real embedding values aren't wired yet — even when
        # enable_embedding=True we emit stubs. A future PR populates
        # them from a SapBERT cache.
        out.update(_embedding_stub_features(config))
    return out


def feature_columns(config: PVOverlapConfig = PVOverlapConfig()) -> List[str]:
    """Return the ordered list of feature column names this config
    will emit. Useful for building an empty DataFrame skeleton or for
    asserting a fixed feature order before XGBoost training.

    Implementation note: derived by sampling a non-empty pair so the
    column list is always consistent with what ``compute_pv_overlap_features``
    actually emits. A divergence between the two is a bug, caught by
    the test suite.
    """
    sample_sde = "PV_TYPE: ENUM(n=1); PV: alpha"
    sample_cde = "PV_TYPE: ENUM(n=1); PV: beta"
    sample = compute_pv_overlap_features(sample_sde, sample_cde, config=config)
    return list(sample.keys())


# ---------------------------------------------------------------------------
# Batch interface (CDE hydration handled here, not in single-pair)
# ---------------------------------------------------------------------------


def _resolve_cde_value(
    row_value: object,
    master_value: Optional[object],
    *,
    prefer: str,
) -> object:
    """Pick the CDE PV string for a single row given the row-level
    value and the master-table value (or None when no master).

    Returns the row value unmodified when no master lookup happened,
    so downstream tests can verify hydration is opt-in.
    """
    row_blank = _is_blank_str(row_value)
    master_blank = master_value is None or _is_blank_str(master_value)
    if prefer == "pv_summary":
        if not master_blank:
            return master_value
        return row_value
    # Default: prefer pv_block_cde
    if row_blank and not master_blank:
        return master_value
    return row_value


def _values_disagree(a: object, b: object) -> bool:
    """True iff both inputs parse to non-empty PV lists that differ as
    normalized sets. Used by the disagreement warning."""
    pa = parse_pv_block(a) if not isinstance(a, ParsedPV) else a
    pb = parse_pv_block(b) if not isinstance(b, ParsedPV) else b
    if not pa.pv_values_norm or not pb.pv_values_norm:
        return False
    return set(pa.pv_values_norm) != set(pb.pv_values_norm)


def compute_pv_overlap_features_batch(
    pairs: pd.DataFrame,
    *,
    sde_col: str = "PV_BLOCK_SDE",
    cde_col: str = "PV_BLOCK_CDE",
    config: PVOverlapConfig = PVOverlapConfig(),
    cde_master: Optional[pd.DataFrame] = None,
    cde_id_col: str = "cde_id",
) -> pd.DataFrame:
    """Compute PV-overlap features for a candidate-pair DataFrame.

    Returns a DataFrame with the same index as ``pairs`` and only the
    feature columns (the caller is responsible for ``pd.concat`` /
    ``join`` to attach them back).

    CDE-PV hydration (OPT-IN ONLY — see module docstring):
      - Providing ``cde_master`` alone does NOT trigger hydration.
      - Hydration runs only when the caller explicitly sets
        ``config.hydrate_cde_from_master_when_empty=True`` OR
        ``config.prefer_cde_source="pv_summary"``. Either is an
        affirmative leakage-aware decision; the default config does
        neither.
      - With ``hydrate_cde_from_master_when_empty=True``, an empty
        ``PV_BLOCK_CDE`` is filled from ``PV_SUMMARY`` keyed by the
        versioned ``cde_id_col``. The row's ``cde_id`` MUST be
        present in that case (hard error otherwise).
      - With ``prefer_cde_source="pv_summary"``, the master value is
        preferred whenever it is non-empty, regardless of whether the
        row value is blank.
      - When master is in use AND both row-level and master-level
        values are non-empty and disagree, the function emits a
        ``UserWarning`` (at most once per ``cde_id``) when
        ``config.warn_on_cde_source_disagreement`` is True. When
        hydration is OFF, the master is never consulted and no such
        warning fires — providing ``cde_master`` is a no-op.

    Output column order is determined by ``feature_columns(config)``.
    """
    if sde_col not in pairs.columns:
        raise KeyError(f"pairs missing required SDE column: {sde_col!r}")
    if cde_col not in pairs.columns:
        raise KeyError(f"pairs missing required CDE column: {cde_col!r}")

    # Decide whether to consult the master at all. Providing
    # ``cde_master`` is necessary but not sufficient — see the
    # leakage docstring at the top of the module.
    master_in_use = cde_master is not None and (
        config.hydrate_cde_from_master_when_empty
        or config.prefer_cde_source == "pv_summary"
    )

    columns = feature_columns(config)
    sde_values = pairs[sde_col].tolist()
    cde_values = pairs[cde_col].tolist()

    if not master_in_use:
        # Trivial path: row CDE values are used verbatim. This is the
        # leakage-safe default for core DEMAP splits.
        rows: List[Dict[str, object]] = [
            compute_pv_overlap_features(sde_v, cde_v, config=config)
            for sde_v, cde_v in zip(sde_values, cde_values)
        ]
        return pd.DataFrame(rows, columns=columns, index=pairs.index)

    # Master is in use. Validate master schema and pairs join key.
    if "PV_SUMMARY" not in cde_master.columns:
        raise KeyError(
            "cde_master must contain a 'PV_SUMMARY' column for hydration"
        )
    if cde_id_col not in cde_master.columns:
        raise KeyError(
            f"cde_master must contain the join key {cde_id_col!r} for hydration"
        )
    if cde_id_col not in pairs.columns:
        raise KeyError(
            f"pairs missing required join column {cde_id_col!r} for CDE hydration"
        )

    master_map: Dict[object, object] = dict(
        zip(cde_master[cde_id_col].tolist(), cde_master["PV_SUMMARY"].tolist())
    )
    cde_ids = pairs[cde_id_col].tolist()
    warned: set = set()
    rows = []

    for sde_v, cde_v, cde_id in zip(sde_values, cde_values, cde_ids):
        cde_id_blank = cde_id is None or (
            isinstance(cde_id, float) and math.isnan(cde_id)
        )
        if cde_id_blank:
            row_is_blank = _is_blank_str(cde_v)
            if config.prefer_cde_source == "pv_summary" or (
                config.hydrate_cde_from_master_when_empty and row_is_blank
            ):
                raise ValueError(
                    f"Row needs CDE hydration but {cde_id_col!r} is "
                    f"missing/NaN. Versioned CDE ids are required for "
                    f"master lookup."
                )
            master_v: Optional[object] = None
        else:
            master_v = master_map.get(cde_id)

        resolved_cde = _resolve_cde_value(
            cde_v, master_v, prefer=config.prefer_cde_source
        )

        if (
            config.warn_on_cde_source_disagreement
            and master_v is not None
            and not _is_blank_str(cde_v)
            and not _is_blank_str(master_v)
            and not cde_id_blank
            and cde_id not in warned
            and _values_disagree(cde_v, master_v)
        ):
            warned.add(cde_id)
            warnings.warn(
                f"PV_BLOCK_CDE and PV_SUMMARY disagree for cde_id={cde_id!r}; "
                f"keeping {config.prefer_cde_source!r} per config",
                UserWarning,
                stacklevel=2,
            )

        rows.append(compute_pv_overlap_features(sde_v, resolved_cde, config=config))

    return pd.DataFrame(rows, columns=columns, index=pairs.index)


# ---------------------------------------------------------------------------
# Candidate-union integration (PR-E)
# ---------------------------------------------------------------------------


def compute_pv_overlap_for_union(
    union: pd.DataFrame,
    *,
    cde_master: pd.DataFrame,
    config: Optional[PVOverlapConfig] = None,
) -> pd.DataFrame:
    """Compute candidate-level PV-overlap features for a candidate-union DataFrame.

    Joins each candidate row's ``cde_id`` to ``cde_master["PV_SUMMARY"]``
    to obtain CDE-side PV text, then computes tiers 1-4 PV-overlap
    features against the row's ``PV_BLOCK_SDE``.

    Every candidate CDE receives the same public/global PV summary from
    the master table — the same source the bi-encoder already uses for
    CDE catalog text.  Query-side ``PV_BLOCK_SDE`` is never altered.

    ``PV_SUMMARY`` is joined transiently and discarded after feature
    computation; it is not added to the candidate-union schema.

    Parameters
    ----------
    union : pd.DataFrame
        Candidate-union rows.  Must contain at least ``PV_BLOCK_SDE``
        and ``cde_id``.
    cde_master : pd.DataFrame
        Enriched CDE master table with ``cde_id`` and ``PV_SUMMARY``.
    config : PVOverlapConfig or None
        Feature tier configuration.  When *None*, uses the default
        ``PVOverlapConfig()`` (tiers 1-4, no embedding stubs).
        Hydration flags on the config are ignored because CDE-side PV
        text is resolved via the master join, not via the batch
        function's internal hydration path.

    Returns
    -------
    pd.DataFrame
        Feature-only DataFrame with the same index as *union*.
        Column order matches ``feature_columns(config)``.
    """
    for col in ("PV_BLOCK_SDE", "cde_id"):
        if col not in union.columns:
            raise KeyError(f"union missing required column: {col!r}")
    for col in ("cde_id", "PV_SUMMARY"):
        if col not in cde_master.columns:
            raise KeyError(f"cde_master missing required column: {col!r}")

    cfg = config if config is not None else PVOverlapConfig()

    master_pvs = (
        cde_master[["cde_id", "PV_SUMMARY"]]
        .drop_duplicates(subset="cde_id", keep="first")
    )
    enriched = union[["PV_BLOCK_SDE", "cde_id"]].copy()
    enriched = enriched.merge(master_pvs, on="cde_id", how="left")
    enriched["PV_SUMMARY"] = enriched["PV_SUMMARY"].fillna("").astype(str)
    enriched.index = union.index

    return compute_pv_overlap_features_batch(
        enriched,
        sde_col="PV_BLOCK_SDE",
        cde_col="PV_SUMMARY",
        config=cfg,
    )
