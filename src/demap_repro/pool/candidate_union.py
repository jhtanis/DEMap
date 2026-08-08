"""Candidate-union assembly for the DEMAP XGBoost stack (PR-B core).

Combines bi-encoder top-K rankings and CDE Match top-K rankings into a
deterministic per-(winner, split, query, candidate) DataFrame that downstream
PRs (C, D, E, F, G, H, I, J) will enrich with score-derived features,
PV-overlap features, agreement signals, calibration, and XGBoost/cross-encoder
scores.

This module is the *contract* PR. It deliberately stays minimal:
  - Thin score/rank pass-throughs from each candidate source.
  - Source flags (``in_biencoder_topk``, ``in_cdematch_topk``).
  - Stratum/context columns from the split parquet (passthrough).
  - The single label-derived column ``is_label``.
  - Per-split ``hydration_policy`` stamping (no PV resolution here).
  - Hard leakage-column guard at output time.

The CIMAC-specific CDE Match per-row fallback, PV-overlap integration,
text-quality features, lexical features, calibration, agreement features,
XGBoost training, and cross-encoder integration all land in later PRs and
read from this schema.

Row key: ``(winner_id, split, query_id, cde_id)``.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Frozen schema constants
# ---------------------------------------------------------------------------

CANDIDATE_UNION_COLUMNS: Tuple[str, ...] = (
    "winner_id",
    "split",
    "query_id",
    "cde_id",
    "pair_id",
    "family",
    "query_source",
    "query_text_q3",
    "PV_BLOCK_SDE",
    "pv_attached",
    "hydration_policy",
    "is_label",
    "in_biencoder_topk",
    "in_cdematch_topk",
    "biencoder_rank",
    "biencoder_score",
    "cdematch_rank",
    "cdematch_score",
)

# Outcome / label-derived columns that must never appear in the union output.
# Source rankings.parquet emits the first group (true_rank/true_score/
# is_correct_top1/p_correct are label-derived; top1_*/s1/s2/s10/margin_* are
# query-global and can be re-broadcast in PR-D — they are excluded here to
# keep the PR-B schema small and the leakage guard strict). The CIMAC CDE
# Match per-row file (PR-B.1) adds a second group: gold_*, top1_long_name,
# top1_rule, top1_public_id, top1_version — gold outcome diagnostics plus
# the CIMAC-specific top1 summary columns. The loaders drop both groups
# and a defensive check at output raises if any survive.
FORBIDDEN_LEAKAGE_COLUMNS: frozenset = frozenset({
    # Bi-encoder side (PR-B core).
    "true_rank",
    "true_score",
    "is_correct_top1",
    "top1_cde_id",
    "top1_score",
    "s1",
    "s2",
    "s10",
    "margin_1_2",
    "margin_1_10",
    "confidence_raw",
    "p_correct",
    # CIMAC CDE Match per-row outcome diagnostics (PR-B.1).
    "gold_public_id",
    "gold_version_from_comments",
    "gold_rank",
    "gold_in_top1",
    "gold_in_top5",
    "gold_in_top10",
    "gold_in_top20",
    "gold_score",
    "gold_version_returned",
    "gold_version_matches",
    "top1_long_name",
    "top1_rule",
    "top1_public_id",
    "top1_version",
})

# Default CDE-side PV hydration policy per split. caDSR-derived splits stay
# "thinned" (PV_BLOCK_CDE only — preserves leakage protection). Externally
# curated splits (GDC, CIMAC) are "hydrated" (PR-E will fall back to
# PV_SUMMARY from cde_master_enriched when PV_BLOCK_CDE is empty). PR-B only
# stamps the policy column; no PV text is resolved here.
DEFAULT_HYDRATION_POLICY: Mapping[str, str] = MappingProxyType({
    "train": "thinned",
    "val": "thinned",
    "val_train": "thinned",
    "val_dev": "thinned",
    "test": "thinned",
    "natural_internal_eval": "thinned",
    "external_holdout_org": "thinned",
    "external_holdout_refslice": "thinned",
    "external_holdout_gdc_altnames": "hydrated",
    "external_holdout_gdc_questiontext": "hydrated",
    "cimac_appendix_a_eval": "hydrated",
    # June-18 externally-curated eval splits.
    "cimac_v2": "hydrated",
    "theradex6_test": "hydrated",
    # Canonical final-reranker eval datasets (configs/evaluation/canonical_eval_datasets.yaml).
    # cctg/oid_alt derive from external_holdout_org and cdash from external_holdout_refslice
    # (all thinned, like their source splits); gdc_combined is the GDC altnames∪questiontext
    # union (hydrated, like both sources). cimac_v2 is already covered above.
    "cctg": "thinned",
    "oid_alt": "thinned",
    "cdash": "thinned",
    "gdc_combined": "hydrated",
})


# Columns the union joins out of each split parquet. ``cde_id`` is the gold
# label; everything else is passthrough context for downstream features.
_SPLIT_META_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "query_id",
    "query_source",
    "family",
    "query_text_q3",
    "PV_BLOCK_SDE",
    "pv_attached",
    "cde_id",  # the gold; renamed to cde_id_gold internally
)


__all__ = [
    "CANDIDATE_UNION_COLUMNS",
    "FORBIDDEN_LEAKAGE_COLUMNS",
    "DEFAULT_HYDRATION_POLICY",
    "load_biencoder_topk",
    "load_cdematch_topk_wide",
    "load_cdematch_topk_cimac",
    "build_candidate_union",
    "build_candidate_union_publicid",
    "CANDIDATE_UNION_PUBLICID_COLUMNS",
]


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


def _infer_winner_id_from_path(rankings_parquet: Path) -> str:
    """Winner id is the parent directory name of rankings.parquet."""
    return Path(rankings_parquet).resolve().parent.name


def load_biencoder_topk(
    rankings_parquet: Path,
    *,
    splits: Optional[Sequence[str]] = None,
    K: int = 20,
    winner_id: Optional[str] = None,
) -> pd.DataFrame:
    """Return long-format (query × candidate) rows from a v2 rankings.parquet.

    Output columns: ``winner_id, split, query_id, cde_id, biencoder_rank,
    biencoder_score, in_biencoder_topk``. Forbidden leakage columns are
    dropped at the source.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    rankings_parquet = Path(rankings_parquet)
    if not rankings_parquet.exists():
        raise FileNotFoundError(rankings_parquet)
    df = pd.read_parquet(
        rankings_parquet,
        columns=["split", "query_id", "topk_cde_ids", "topk_scores"],
    )
    if splits is not None:
        df = df[df["split"].isin(splits)]
    if winner_id is None:
        winner_id = _infer_winner_id_from_path(rankings_parquet)
    out_cols = [
        "winner_id", "split", "query_id", "cde_id",
        "biencoder_rank", "biencoder_score", "in_biencoder_topk",
    ]
    if len(df) == 0:
        return _empty_typed_df({c: _DTYPE_HINTS.get(c, "object") for c in out_cols})
    df = df.copy()
    df["topk_cde_ids"] = df["topk_cde_ids"].apply(lambda a: list(a[:K]) if a is not None else [])
    df["topk_scores"] = df["topk_scores"].apply(lambda a: list(a[:K]) if a is not None else [])
    long = df.explode(["topk_cde_ids", "topk_scores"], ignore_index=True)
    long = long.rename(columns={"topk_cde_ids": "cde_id", "topk_scores": "biencoder_score"})
    # Drop NaN candidate slots (queries that had empty topk arrays).
    long = long[long["cde_id"].notna()].reset_index(drop=True)
    if len(long) == 0:
        return _empty_typed_df({c: _DTYPE_HINTS.get(c, "object") for c in out_cols})
    long["biencoder_rank"] = long.groupby(["split", "query_id"], sort=False).cumcount() + 1
    long["biencoder_score"] = long["biencoder_score"].astype(float)
    long["biencoder_rank"] = long["biencoder_rank"].astype("Int64")
    long["in_biencoder_topk"] = True
    long["winner_id"] = winner_id
    long["cde_id"] = long["cde_id"].astype(str)
    long["query_id"] = long["query_id"].astype(str)
    long["split"] = long["split"].astype(str)
    return long[out_cols]


def load_cdematch_topk_wide(
    cdematch_wide_csv: Path,
    *,
    query_ids: Optional[Iterable[str]] = None,
    K: int = 10,
) -> pd.DataFrame:
    """Return long-format CDE Match top-K rows from a wide CSV.

    Output columns: ``query_id, cde_id, cdematch_rank, cdematch_score,
    in_cdematch_topk``.

    The wide CSV is expected to have a ``query_id`` column plus paired
    ``pred_cde_id_<k>`` / ``pred_score_<k>`` columns for k = 1..N. Empty
    slots (NaN or empty-string ``pred_cde_id_<k>``) are dropped.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    cdematch_wide_csv = Path(cdematch_wide_csv)
    if not cdematch_wide_csv.exists():
        raise FileNotFoundError(cdematch_wide_csv)
    df = pd.read_csv(cdematch_wide_csv, dtype=str)
    if "query_id" not in df.columns:
        raise ValueError(f"cdematch wide CSV missing 'query_id' column: {cdematch_wide_csv}")
    if query_ids is not None:
        wanted = set(map(str, query_ids))
        df = df[df["query_id"].astype(str).isin(wanted)]
    out_cols = ["query_id", "cde_id", "cdematch_rank", "cdematch_score", "in_cdematch_topk"]
    if len(df) == 0:
        return _empty_typed_df({c: _DTYPE_HINTS.get(c, "object") for c in out_cols})
    pred_id_cols = [f"pred_cde_id_{k}" for k in range(1, K + 1) if f"pred_cde_id_{k}" in df.columns]
    pred_score_cols = [f"pred_score_{k}" for k in range(1, K + 1) if f"pred_score_{k}" in df.columns]
    if not pred_id_cols or not pred_score_cols:
        return _empty_typed_df({c: _DTYPE_HINTS.get(c, "object") for c in out_cols})
    ids_long = df.melt(
        id_vars=["query_id"], value_vars=pred_id_cols,
        var_name="rank_col", value_name="cde_id",
    )
    ids_long["cdematch_rank"] = ids_long["rank_col"].str.extract(r"_(\d+)$")[0].astype(int)
    ids_long = ids_long.drop(columns="rank_col")
    scores_long = df.melt(
        id_vars=["query_id"], value_vars=pred_score_cols,
        var_name="rank_col", value_name="cdematch_score",
    )
    scores_long["cdematch_rank"] = scores_long["rank_col"].str.extract(r"_(\d+)$")[0].astype(int)
    scores_long = scores_long.drop(columns="rank_col")
    out = ids_long.merge(scores_long, on=["query_id", "cdematch_rank"], how="inner")
    # Drop empty slots (no cde_id or no parseable score).
    mask = out["cde_id"].notna() & (out["cde_id"].astype(str) != "")
    out = out[mask].copy()
    out["cdematch_score"] = pd.to_numeric(out["cdematch_score"], errors="coerce")
    out = out.dropna(subset=["cdematch_score"]).copy()
    out["cdematch_rank"] = out["cdematch_rank"].astype("Int64")
    out["in_cdematch_topk"] = True
    out["cde_id"] = out["cde_id"].astype(str)
    out["query_id"] = out["query_id"].astype(str)
    return out[out_cols]


# ---------------------------------------------------------------------------
# CIMAC fallback loader (PR-B.1)
# ---------------------------------------------------------------------------


# Columns the CIMAC per-row loader is allowed to read. Selecting only these
# keeps every gold/outcome column out of the in-memory frame as a defense
# in depth (the leakage guard at the end of build_candidate_union is the
# second line of defense).
_CIMAC_PER_ROW_NEEDED_COLS: Tuple[str, ...] = (
    "seq_id",
    "query_id",
    "is_evaluable",
    "data_element",
    "top20_public_ids",
    "top20_scores",
)


def _version_sort_key(version: str) -> Tuple[int, ...]:
    """Numeric tuple key for dotted version strings (``"5.1"`` → ``(5, 1)``).

    Non-numeric segments sort below any numeric segment (mapped to ``-1``).
    Shorter tuples compare less when prefixes match, so ``"5"`` < ``"5.1"``
    and ``"4.2"`` > ``"4.1"`` — matching the existing CIMAC
    highest-version-per-publicid policy from
    ``scripts/export_cimac_appendix_a_splits.py``.
    """
    parts: List[int] = []
    for piece in str(version).split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(-1)
    return tuple(parts)


def _build_publicid_to_versioned_cde_id(cde_master: pd.DataFrame) -> dict:
    """Return ``{publicid: "<publicid>::<highest_version>"}``.

    Uses the highest-version-per-publicid rule via ``_version_sort_key``.
    """
    out: dict = {}
    for publicid, group in cde_master.groupby("cde_publicid", sort=True):
        versions = group["cde_version"].astype(str).tolist()
        best = max(versions, key=_version_sort_key)
        out[str(publicid)] = f"{publicid}::{best}"
    return out


def _build_cimac_original_id_to_query_id(cimac_split: pd.DataFrame) -> dict:
    """Return ``{cimac_query_id_original (int): query_id (v2 hash, str)}``.

    Raises if ``cimac_query_id_original`` is not unique (defensive — the v2
    split builder guarantees uniqueness).
    """
    if not cimac_split["cimac_query_id_original"].is_unique:
        raise ValueError(
            "cimac_query_id_original must be unique in the CIMAC split parquet"
        )
    return dict(zip(
        cimac_split["cimac_query_id_original"].astype(int).tolist(),
        cimac_split["query_id"].astype(str).tolist(),
    ))


def _parse_csv_list_str(value) -> List[str]:
    """Parse a comma-separated string into a list of stripped tokens.
    Empty/NaN/None → ``[]``."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    s = str(value)
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip() != ""]


def _parse_csv_list_float(value) -> List[float]:
    parts = _parse_csv_list_str(value)
    return [float(p) for p in parts]


def load_cdematch_topk_cimac(
    cimac_per_row_csv: Path,
    cimac_split_parquet: Path,
    cde_master_enriched: Path,
    *,
    K: int = 10,
    require_complete_mapping: bool = True,
) -> pd.DataFrame:
    """Return long-format CDE Match top-K rows for CIMAC, with versioned cde_ids.

    Reads ``cimac_cdematch_per_row.csv`` (one row per CIMAC data element),
    filters to ``is_evaluable == True``, maps each row's 0-based ``query_id``
    to the v2 hash ``query_id`` via ``cimac_appendix_a_eval.parquet``'s
    ``cimac_query_id_original`` column, and resolves the unversioned
    ``top20_public_ids`` to versioned ``cde_id`` strings using the
    highest-version-per-publicid rule on ``cde_master_enriched.parquet``.

    Output schema is identical to :func:`load_cdematch_topk_wide`:
    ``[query_id, cde_id, cdematch_rank, cdematch_score, in_cdematch_topk]``.

    Mapping note:
      The per-row CSV has two integer ID columns: ``seq_id`` (1-based row
      index) and ``query_id`` (0-based original workbook index). The split
      parquet's ``cimac_query_id_original`` is derived from the workbook's
      0-based index, so the correct join key is the per-row ``query_id``
      column, **not** ``seq_id``.

    Notes:
      - Public IDs absent from ``cde_master_enriched`` are dropped silently
        (they cannot be resolved to a valid versioned ``cde_id``).
      - ``require_complete_mapping=True`` raises if any
        ``cimac_query_id_original`` in the split parquet has no matching
        per-row entry; ``False`` simply omits those.
      - When ``data_element`` is present in both the per-row CSV and the
        split parquet, an alignment check verifies the mapping is correct.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    cimac_per_row_csv = Path(cimac_per_row_csv)
    cimac_split_parquet = Path(cimac_split_parquet)
    cde_master_enriched = Path(cde_master_enriched)
    for p in (cimac_per_row_csv, cimac_split_parquet, cde_master_enriched):
        if not p.exists():
            raise FileNotFoundError(p)

    out_cols = ["query_id", "cde_id", "cdematch_rank", "cdematch_score", "in_cdematch_topk"]

    # 1. Read per-row CSV, selecting only the columns we are allowed to use.
    raw = pd.read_csv(cimac_per_row_csv, dtype=str)
    missing = [c for c in _CIMAC_PER_ROW_NEEDED_COLS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"CIMAC per-row CSV missing required columns: {missing} "
            f"(file: {cimac_per_row_csv})"
        )
    per_row = raw[list(_CIMAC_PER_ROW_NEEDED_COLS)].copy()

    # 2. Filter to evaluable rows.
    is_eval = per_row["is_evaluable"].astype(str).str.strip().str.lower() == "true"
    per_row = per_row.loc[is_eval].copy()
    if len(per_row) == 0:
        return _empty_typed_df({c: _DTYPE_HINTS.get(c, "object") for c in out_cols})

    # 3. Build cimac_query_id_original → v2 hash query_id map from the split.
    import pyarrow.parquet as pq
    split_schema = pq.read_schema(cimac_split_parquet)
    split_cols = ["query_id", "cimac_query_id_original"]
    if "data_element" in split_schema.names:
        split_cols = split_cols + ["data_element"]
    split = pd.read_parquet(cimac_split_parquet, columns=split_cols)
    original_id_to_qid = _build_cimac_original_id_to_query_id(split)

    # 4. Join key: per-row ``query_id`` (0-based workbook index) matches
    #    the split's ``cimac_query_id_original`` (also 0-based).
    per_row["_cimac_original_id"] = per_row["query_id"].astype(int)
    per_row_ids = set(per_row["_cimac_original_id"].tolist())
    split_ids = set(original_id_to_qid.keys())
    missing_ids = split_ids - per_row_ids
    if missing_ids:
        if require_complete_mapping:
            sample = sorted(missing_ids)[:5]
            raise ValueError(
                f"CIMAC per-row CSV missing entries for {len(missing_ids)} "
                f"split rows (by cimac_query_id_original): examples={sample}"
            )
    per_row = per_row.loc[per_row["_cimac_original_id"].isin(split_ids)].copy()
    if len(per_row) == 0:
        return _empty_typed_df({c: _DTYPE_HINTS.get(c, "object") for c in out_cols})

    # 4b. Defensive data-element alignment check.
    if "data_element" in split.columns and "data_element" in per_row.columns:
        split_elem = dict(zip(
            split["cimac_query_id_original"].astype(int),
            split["data_element"].astype(str),
        ))
        for _, r in per_row.iterrows():
            orig_id = int(r["_cimac_original_id"])
            expected = split_elem.get(orig_id)
            actual = str(r["data_element"]).strip()
            if expected is not None and actual and expected != actual:
                raise ValueError(
                    f"CIMAC data_element mismatch at cimac_query_id_original="
                    f"{orig_id}: split has '{expected}', per-row has '{actual}'. "
                    f"The per-row query_id→split cimac_query_id_original "
                    f"mapping may be misaligned."
                )

    # 5. Build publicid → versioned cde_id map from cde_master_enriched.
    master = pd.read_parquet(
        cde_master_enriched,
        columns=["cde_publicid", "cde_version"],
    )
    publicid_to_cde_id = _build_publicid_to_versioned_cde_id(master)

    # 6. Explode each per-row's top20_public_ids / top20_scores into long format.
    records: List[dict] = []
    for _, r in per_row.iterrows():
        orig_id = int(r["_cimac_original_id"])
        q_id = original_id_to_qid[orig_id]
        ids = _parse_csv_list_str(r["top20_public_ids"])
        scores = _parse_csv_list_float(r["top20_scores"])
        if len(ids) != len(scores):
            raise ValueError(
                f"CIMAC cimac_query_id_original={orig_id}: "
                f"top20_public_ids has {len(ids)} entries "
                f"but top20_scores has {len(scores)} "
                f"(file: {cimac_per_row_csv})"
            )
        if not ids:
            continue
        for rank, (publicid, score) in enumerate(zip(ids[:K], scores[:K]), start=1):
            cde_id = publicid_to_cde_id.get(publicid)
            if cde_id is None:
                continue
            records.append({
                "query_id": q_id,
                "cde_id": cde_id,
                "cdematch_rank": rank,
                "cdematch_score": float(score),
                "in_cdematch_topk": True,
            })

    if not records:
        return _empty_typed_df({c: _DTYPE_HINTS.get(c, "object") for c in out_cols})

    df = pd.DataFrame(records)
    df["query_id"] = df["query_id"].astype(str)
    df["cde_id"] = df["cde_id"].astype(str)
    df["cdematch_rank"] = df["cdematch_rank"].astype("Int64")
    df["cdematch_score"] = df["cdematch_score"].astype(float)
    df["in_cdematch_topk"] = df["in_cdematch_topk"].astype(bool)
    return df[out_cols]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_candidate_union(
    *,
    rankings_parquet: Path,
    splits_dir: Path,
    cdematch_wide_csv: Path,
    cimac_per_row_csv: Optional[Path] = None,
    cde_master_enriched: Optional[Path] = None,
    splits: Optional[Sequence[str]] = None,
    K_biencoder: int = 20,
    K_cdematch: int = 10,
    hydration_policy: Optional[Mapping[str, str]] = None,
    winner_id: Optional[str] = None,
) -> pd.DataFrame:
    """Assemble the candidate-union DataFrame for one (winner) × N splits.

    Output schema is exactly ``CANDIDATE_UNION_COLUMNS`` and the row key is
    ``(winner_id, split, query_id, cde_id)``. See module docstring for the
    full design.

    CIMAC fallback (PR-B.1): when ``cimac_per_row_csv`` is supplied and
    ``"cimac_appendix_a_eval"`` is among the requested splits, CDE Match
    candidates for CIMAC queries are loaded via :func:`load_cdematch_topk_cimac`
    and concatenated with the wide-CSV rows. ``cde_master_enriched`` is
    required in this case (it provides the publicid → versioned cde_id
    mapping). Without the per-row file, CIMAC degrades gracefully: CIMAC
    rows simply have ``in_cdematch_topk=False`` (PR-B-core behavior).
    """
    rankings_parquet = Path(rankings_parquet)
    splits_dir = Path(splits_dir)
    cdematch_wide_csv = Path(cdematch_wide_csv)

    if winner_id is None:
        winner_id = _infer_winner_id_from_path(rankings_parquet)

    # 1. Bi-encoder long-format rows.
    be = load_biencoder_topk(
        rankings_parquet, splits=splits, K=K_biencoder, winner_id=winner_id,
    )
    if len(be) == 0:
        return _empty_union()

    # 2. Splits actually present in the bi-encoder output.
    splits_in_be = sorted(be["split"].unique().tolist())
    if splits is None:
        splits_to_use = splits_in_be
    else:
        splits_to_use = [s for s in splits if s in splits_in_be]
        if not splits_to_use:
            return _empty_union()

    # 3. Load split metadata for those splits.
    split_meta = _load_split_meta(splits_dir, splits_to_use)

    # 4. CDE Match top-K, filtered to relevant query_ids.
    all_qids = sorted(
        set(be["query_id"].astype(str)) | set(split_meta["query_id"].astype(str))
    )
    cm = load_cdematch_topk_wide(cdematch_wide_csv, query_ids=all_qids, K=K_cdematch)

    # 4b. CIMAC fallback (PR-B.1): the wide CSV does not cover CIMAC queries
    # (they use a different query_id keying), so when the CIMAC split is
    # requested and a per-row file is supplied, load CIMAC CDE Match rows
    # separately and concatenate. The per-row loader emits the same five-
    # column schema as the wide loader, so the rest of the pipeline is
    # unchanged.
    if cimac_per_row_csv is not None and "cimac_appendix_a_eval" in splits_to_use:
        if cde_master_enriched is None:
            raise ValueError(
                "cde_master_enriched is required when cimac_per_row_csv is provided"
            )
        cimac_split_path = Path(splits_dir) / "cimac_appendix_a_eval.parquet"
        cm_cimac = load_cdematch_topk_cimac(
            cimac_per_row_csv,
            cimac_split_path,
            cde_master_enriched,
            K=K_cdematch,
        )
        if len(cm_cimac):
            cm_cimac = cm_cimac.loc[cm_cimac["query_id"].isin(all_qids)]
        if len(cm_cimac):
            cm = pd.concat([cm, cm_cimac], ignore_index=True)

    # 5. Outer-merge bi-encoder and CDE Match on (query_id, cde_id).
    union = be.merge(cm, on=["query_id", "cde_id"], how="outer")
    # ``.eq(True)`` collapses object-dtype {True, NaN} columns to bool
    # without triggering the pandas 3.x silent-downcast deprecation that
    # ``.fillna(False).astype(bool)`` would.
    union["in_biencoder_topk"] = union["in_biencoder_topk"].eq(True)
    union["in_cdematch_topk"] = union["in_cdematch_topk"].eq(True)
    union["winner_id"] = union["winner_id"].fillna(winner_id)

    # Backfill split for CDE-Match-only rows via query_id → split.
    qid_to_split = split_meta.drop_duplicates("query_id").set_index("query_id")["split"]
    union["split"] = union["split"].fillna(union["query_id"].map(qid_to_split))
    union = union.dropna(subset=["split"]).copy()

    # 6. Join split metadata.
    union = union.merge(split_meta, on=["split", "query_id"], how="left")
    union["is_label"] = (
        union["cde_id"].astype(str) == union["cde_id_gold"].astype(str)
    )
    union = union.drop(columns=["cde_id_gold"])

    # 7. Stamp hydration policy.
    policy_map = dict(DEFAULT_HYDRATION_POLICY)
    if hydration_policy:
        policy_map.update(hydration_policy)
    union["hydration_policy"] = union["split"].map(policy_map)
    if union["hydration_policy"].isna().any():
        missing = sorted(
            union.loc[union["hydration_policy"].isna(), "split"].unique()
        )
        raise ValueError(
            f"hydration_policy not defined for splits: {missing}. "
            "Either extend DEFAULT_HYDRATION_POLICY or pass "
            "hydration_policy={split: 'thinned'|'hydrated', ...} to "
            "build_candidate_union."
        )

    # 8. Deduplicate by row key, taking the union of source flags.
    key = ["winner_id", "split", "query_id", "cde_id"]
    agg_spec = {
        "pair_id": "first",
        "family": "first",
        "query_source": "first",
        "query_text_q3": "first",
        "PV_BLOCK_SDE": "first",
        "pv_attached": "first",
        "hydration_policy": "first",
        "is_label": "first",
        "in_biencoder_topk": "any",
        "in_cdematch_topk": "any",
        # Each (query, cde) is contributed by exactly one bi-encoder row and
        # at most one CDE Match row, so min/max collapse single-non-null
        # series without ambiguity.
        "biencoder_rank": "min",
        "biencoder_score": "max",
        "cdematch_rank": "min",
        "cdematch_score": "max",
    }
    union = union.groupby(key, as_index=False, sort=True).agg(agg_spec)

    # 9. Coerce final dtypes.
    union["winner_id"] = union["winner_id"].astype(str)
    union["split"] = union["split"].astype(str)
    union["query_id"] = union["query_id"].astype(str)
    union["cde_id"] = union["cde_id"].astype(str)
    union["pair_id"] = union["pair_id"].fillna("").astype(str)
    union["family"] = union["family"].fillna("").astype(str)
    union["query_source"] = union["query_source"].fillna("").astype(str)
    union["query_text_q3"] = union["query_text_q3"].fillna("").astype(str)
    union["PV_BLOCK_SDE"] = union["PV_BLOCK_SDE"].fillna("").astype(str)
    union["pv_attached"] = union["pv_attached"].fillna(False).astype(bool)
    union["hydration_policy"] = union["hydration_policy"].astype(str)
    union["is_label"] = union["is_label"].fillna(False).astype(bool)
    union["in_biencoder_topk"] = union["in_biencoder_topk"].astype(bool)
    union["in_cdematch_topk"] = union["in_cdematch_topk"].astype(bool)
    union["biencoder_rank"] = union["biencoder_rank"].astype("Int64")
    union["cdematch_rank"] = union["cdematch_rank"].astype("Int64")
    union["biencoder_score"] = union["biencoder_score"].astype(float)
    union["cdematch_score"] = union["cdematch_score"].astype(float)

    # 10. Final column order, stable sort, leakage guard.
    union = union[list(CANDIDATE_UNION_COLUMNS)]
    union = union.sort_values(key, kind="mergesort").reset_index(drop=True)

    bad = sorted(set(union.columns) & FORBIDDEN_LEAKAGE_COLUMNS)
    if bad:  # pragma: no cover — defensive guard, exercised by test
        raise RuntimeError(
            f"forbidden leakage columns present in candidate union: {bad}"
        )

    return union


# ---------------------------------------------------------------------------
# Public-ID-keyed union (June-18 deployable path)
# ---------------------------------------------------------------------------

# Schema of the public-id-keyed union = the frozen contract + two columns.
CANDIDATE_UNION_PUBLICID_COLUMNS: Tuple[str, ...] = (
    CANDIDATE_UNION_COLUMNS + ("cde_publicid", "in_keyword_topk")
)


def _pub(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split("::").str[0]


def build_candidate_union_publicid(
    *,
    be: pd.DataFrame,
    cm: pd.DataFrame,
    kw: Optional[pd.DataFrame],
    split_meta: pd.DataFrame,
    cde_master: pd.DataFrame,
    winner_id: str,
    hydration_policy: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Public-ID-keyed union of bi-encoder + CDE-Match(clone) + keyword sources.

    Merges the three candidate sources on ``(winner_id, split, query_id,
    cde_publicid)`` so the same public id arriving from DIFFERENT versioned
    ``cde_id`` strings (e.g. a clone-wide candidate at ``5`` and a bi-encoder
    candidate at ``5.1``) collapses to ONE row.

    Canonical versioned ``cde_id`` per ``(query_id, cde_publicid)`` — explicit rule:
      1. the query's GOLD versioned ``cde_id`` when this public id IS the gold
         public id (so version-sensitive joins/eval land on the exact gold), else
      2. the highest-version ``cde_id`` for that public id in ``cde_master``
         (``_build_publicid_to_versioned_cde_id``), else
      3. the first observed candidate ``cde_id`` (bi-encoder, then CDE-Match, then
         keyword) — deterministic fallback for public ids absent from the master.

    ``is_label`` is computed on PUBLIC ID. Output schema is
    :data:`CANDIDATE_UNION_PUBLICID_COLUMNS` (the frozen contract + ``cde_publicid``
    and ``in_keyword_topk``). No gold is injected here.
    """
    inputs = (be, cm, kw)
    if all(x is None or len(x) == 0 for x in inputs):
        return _empty_publicid_union()

    # --- per-source (query_id, publicid) aggregates --------------------------
    be = be.copy()
    be["cde_publicid"] = _pub(be["cde_id"])
    be_agg = be.groupby(["query_id", "cde_publicid"], as_index=False).agg(
        split=("split", "first"), winner_id=("winner_id", "first"),
        biencoder_rank=("biencoder_rank", "min"),
        biencoder_score=("biencoder_score", "max"),
        be_cde=("cde_id", "first"))
    be_agg["in_biencoder_topk"] = True

    if cm is not None and len(cm):
        cm = cm.copy()
        cm["cde_publicid"] = _pub(cm["cde_id"])
        cm_agg = cm.groupby(["query_id", "cde_publicid"], as_index=False).agg(
            cdematch_rank=("cdematch_rank", "min"),
            cdematch_score=("cdematch_score", "max"),
            cm_cde=("cde_id", "first"))
        cm_agg["in_cdematch_topk"] = True
    else:
        cm_agg = pd.DataFrame(columns=["query_id", "cde_publicid", "cdematch_rank",
                                       "cdematch_score", "cm_cde", "in_cdematch_topk"])

    if kw is not None and len(kw):
        kw = kw.copy()
        kw["query_id"] = kw["query_id"].astype(str)
        kw["cde_publicid"] = _pub(kw["cde_id"])
        kw_agg = kw.groupby(["query_id", "cde_publicid"], as_index=False).agg(
            kw_cde=("cde_id", "first"))
        kw_agg["in_keyword_topk"] = True
    else:
        kw_agg = pd.DataFrame(columns=["query_id", "cde_publicid", "kw_cde", "in_keyword_topk"])

    for f in (be_agg, cm_agg, kw_agg):
        f["query_id"] = f["query_id"].astype(str)
        f["cde_publicid"] = f["cde_publicid"].astype(str)

    # --- outer-merge the three sources on (query_id, publicid) ---------------
    u = be_agg.merge(cm_agg, on=["query_id", "cde_publicid"], how="outer")
    u = u.merge(kw_agg, on=["query_id", "cde_publicid"], how="outer")
    for flag in ("in_biencoder_topk", "in_cdematch_topk", "in_keyword_topk"):
        u[flag] = u[flag].eq(True) if flag in u.columns else False

    # backfill split / winner for cm/kw-only rows.
    qid_to_split = split_meta.drop_duplicates("query_id").set_index("query_id")["split"]
    u["split"] = u["split"].fillna(u["query_id"].map(qid_to_split))
    u["winner_id"] = u["winner_id"].fillna(winner_id)
    u = u.dropna(subset=["split"]).copy()

    # --- gold (public id) + is_label -----------------------------------------
    gm = split_meta.drop_duplicates("query_id").set_index("query_id")
    gold_cde = gm["cde_id_gold"].astype(str)
    gold_pub = _pub(gm["cde_id_gold"])
    u["gold_pub"] = u["query_id"].map(gold_pub)
    u["is_label"] = u["cde_publicid"].astype(str) == u["gold_pub"].astype(str)

    # --- canonical versioned cde_id ------------------------------------------
    master_map = _build_publicid_to_versioned_cde_id(cde_master)
    obs = u["be_cde"]
    for c in ("cm_cde", "kw_cde"):
        if c in u.columns:
            obs = obs.fillna(u[c])
    canonical = u["cde_publicid"].map(master_map)
    canonical = canonical.fillna(obs)
    gold_for_qid = u["query_id"].map(gold_cde)
    u["cde_id"] = [g if lab else c for lab, g, c in
                   zip(u["is_label"].tolist(), gold_for_qid.tolist(), canonical.tolist())]

    # --- join split metadata + hydration -------------------------------------
    meta_cols = ["split", "query_id", "pair_id", "family", "query_source",
                 "query_text_q3", "PV_BLOCK_SDE", "pv_attached"]
    u = u.merge(split_meta[meta_cols], on=["split", "query_id"], how="left")

    policy_map = dict(DEFAULT_HYDRATION_POLICY)
    if hydration_policy:
        policy_map.update(hydration_policy)
    u["hydration_policy"] = u["split"].map(policy_map)
    if u["hydration_policy"].isna().any():
        missing = sorted(u.loc[u["hydration_policy"].isna(), "split"].unique())
        raise ValueError(f"hydration_policy not defined for splits: {missing}")

    # --- dtypes + final schema -----------------------------------------------
    u["winner_id"] = u["winner_id"].astype(str)
    u["split"] = u["split"].astype(str)
    u["query_id"] = u["query_id"].astype(str)
    u["cde_id"] = u["cde_id"].astype(str)
    u["cde_publicid"] = u["cde_publicid"].astype(str)
    u["pair_id"] = u["pair_id"].fillna("").astype(str)
    u["family"] = u["family"].fillna("").astype(str)
    u["query_source"] = u["query_source"].fillna("").astype(str)
    u["query_text_q3"] = u["query_text_q3"].fillna("").astype(str)
    u["PV_BLOCK_SDE"] = u["PV_BLOCK_SDE"].fillna("").astype(str)
    u["pv_attached"] = u["pv_attached"].fillna(False).astype(bool)
    u["hydration_policy"] = u["hydration_policy"].astype(str)
    u["is_label"] = u["is_label"].fillna(False).astype(bool)
    for flag in ("in_biencoder_topk", "in_cdematch_topk", "in_keyword_topk"):
        u[flag] = u[flag].astype(bool)
    u["biencoder_rank"] = u["biencoder_rank"].astype("Int64")
    u["cdematch_rank"] = u["cdematch_rank"].astype("Int64")
    u["biencoder_score"] = u["biencoder_score"].astype(float)
    u["cdematch_score"] = u["cdematch_score"].astype(float)

    key = ["winner_id", "split", "query_id", "cde_publicid"]
    u = u[list(CANDIDATE_UNION_PUBLICID_COLUMNS)].sort_values(
        key, kind="mergesort").reset_index(drop=True)

    bad = sorted(set(u.columns) & FORBIDDEN_LEAKAGE_COLUMNS)
    if bad:  # pragma: no cover
        raise RuntimeError(f"forbidden leakage columns in publicid union: {bad}")
    return u


def _empty_publicid_union() -> pd.DataFrame:
    base = {col: _DTYPE_HINTS.get(col, "object") for col in CANDIDATE_UNION_COLUMNS}
    base["cde_publicid"] = "object"
    base["in_keyword_topk"] = "bool"
    return _empty_typed_df(base)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_split_meta(splits_dir: Path, splits_to_use: Sequence[str]) -> pd.DataFrame:
    """Load per-split metadata frames and concatenate. Missing optional
    columns are filled with empty strings or False."""
    frames: List[pd.DataFrame] = []
    for s in splits_to_use:
        p = Path(splits_dir) / f"{s}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"split parquet not found: {p}")
        avail = pd.read_parquet(p).columns
        cols = [c for c in _SPLIT_META_COLUMNS if c in avail]
        for required in ("query_id", "cde_id"):
            if required not in cols:
                raise ValueError(
                    f"split {s}.parquet missing required column: {required}"
                )
        sdf = pd.read_parquet(p, columns=cols).copy()
        if "pair_id" not in sdf.columns:
            sdf["pair_id"] = ""
        if "query_source" not in sdf.columns:
            sdf["query_source"] = ""
        if "family" not in sdf.columns:
            sdf["family"] = ""
        if "query_text_q3" not in sdf.columns:
            sdf["query_text_q3"] = ""
        if "PV_BLOCK_SDE" not in sdf.columns:
            sdf["PV_BLOCK_SDE"] = ""
        if "pv_attached" not in sdf.columns:
            sdf["pv_attached"] = False
        sdf["split"] = s
        sdf = sdf.rename(columns={"cde_id": "cde_id_gold"})
        sdf["query_id"] = sdf["query_id"].astype(str)
        sdf["cde_id_gold"] = sdf["cde_id_gold"].astype(str)
        sdf = sdf.drop_duplicates(subset=["split", "query_id"], keep="first")
        frames.append(sdf[[
            "split", "query_id", "cde_id_gold", "pair_id", "family",
            "query_source", "query_text_q3", "PV_BLOCK_SDE", "pv_attached",
        ]])
    return pd.concat(frames, ignore_index=True)


_DTYPE_HINTS = {
    "is_label": "bool",
    "in_biencoder_topk": "bool",
    "in_cdematch_topk": "bool",
    "pv_attached": "bool",
    "biencoder_score": "float64",
    "cdematch_score": "float64",
    "biencoder_rank": "Int64",
    "cdematch_rank": "Int64",
}


def _empty_typed_df(dtype_map: Mapping[str, str]) -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtype_map.items()})


def _empty_union() -> pd.DataFrame:
    """Typed empty DataFrame matching CANDIDATE_UNION_COLUMNS."""
    return _empty_typed_df({
        col: _DTYPE_HINTS.get(col, "object") for col in CANDIDATE_UNION_COLUMNS
    })
