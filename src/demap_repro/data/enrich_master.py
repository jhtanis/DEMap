#!/usr/bin/env python3
"""enrich_master.py

Build an enriched caDSR CDE master table for downstream *target-side*
construction.

Why this exists
---------------
The caDSR XML extractor writes multiple normalized Parquet tables
(master, DEC, value domain, permissible values, reference documents, etc.).
For modeling and analysis we want one wide table keyed by (cde_publicid, cde_version)
with the fields needed to build the CDE text representations (v1-v6).

Inputs (typical under data/interim/cadsr_merged/)
-------------------------------------------------
- cde_master.parquet
- cde_iso11179_dec.parquet
- cde_iso11179_value_domain.parquet
- cde_permissible_values.parquet
- cde_reference_documents.parquet (only needed for v3 Preferred Question Text)

Outputs
-------
- data/processed/cde_master_enriched.parquet (default)

Enriched columns used by CDE text recipes (v1-v6)
-------------------------------------------------
- SHORT_NAME           (from cde_master.preferred_name)
- LONG_NAME            (from cde_master.long_name)
- DEFINITION           (from cde_master.preferred_definition)
- PREFERRED_QUESTION_TEXT (from reference docs where DocumentType == Preferred Question Text)
- VALUE_DOMAIN_TYPE    (from value domain)
- VALUE_DOMAIN_DATATYPE (from value domain)
- PV_SUMMARY           (up to N=10 permissible value meaning names, ordered)
- DEC_LONG_NAME        (from DataElementConcept long name)

Notes on PV_SUMMARY
-------------------
- We include at most N=10 value meaning names.
- We sort by display order when available (meaning_concept_display_order).
- We *do not* include the PV "value" token (valid_value) to keep the representation
  easy to explain and stable across source conventions.

Language policy
---------------
- For Preferred Question Text (v3), we only retain rows with Language == English
  (case-insensitive).
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional

import pandas as pd

from demap_repro.text.recipes import norm_str
from demap_repro.text.pv_summary import build_pv_summary_table


def _ensure_dir_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _dedupe_keep_first(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def _is_english(s: str) -> bool:
    return norm_str(s).casefold() == "english"


def _summarize_permissible_values(
    pv: pd.DataFrame,
    *,
    pv_summary_profile: str = "cadsr",
    pv_max_n: int = 10,
    pv_huge_threshold: int = 20,
    pv_max_n_query: Optional[int] = None,
    pv_max_n_cde: Optional[int] = None,
    pv_center_fraction: float = 0.5,
    pv_min_n: int = 2,
    pv_size_jitter: int = 1,
    pv_generic_boost: float = 0.10,
    pv_generic_cap: float = 0.70,
    sde_generic_label_p: float = 0.30,
    pv_small_n_generic_threshold: float = 0.5,
    pv_max_resample_attempts: int = 3,
    pv_placeholder_token: str = "<MISSING_PV_SUMMARY>",
    diagnostics_summary_path: Optional[str] = None,
    diagnostics_records_path: Optional[str] = None,
    sep: str = " | ",
) -> pd.DataFrame:
    """Aggregate PV rows to per-CDE PV_N, PV_TYPE, and PV_SUMMARY.

    Implementation note
    -------------------
    We delegate to :func:`demap_repro.text.pv_summary.build_pv_summary_table` to ensure
    PV summaries are constructed consistently across:

    - CDE-side enrichment (this module)
    - SDE-side query augmentation (demap_repro.data.queries)
    """

    if pv is None or pv.empty:
        return pd.DataFrame(columns=["cde_publicid", "cde_version", "PV_N", "PV_TYPE", "PV_SUMMARY"])

    pv_sum = build_pv_summary_table(
        pv,
        profile=str(pv_summary_profile),
        pv_max_n=int(pv_max_n),
        pv_huge_threshold=int(pv_huge_threshold),
        pv_max_n_query=pv_max_n_query,
        pv_max_n_cde=pv_max_n_cde,
        pv_center_fraction=float(pv_center_fraction),
        pv_min_n=int(pv_min_n),
        pv_size_jitter=int(pv_size_jitter),
        pv_generic_boost=float(pv_generic_boost),
        pv_generic_cap=float(pv_generic_cap),
        sde_generic_label_p=float(sde_generic_label_p),
        pv_small_n_generic_threshold=float(pv_small_n_generic_threshold),
        pv_max_resample_attempts=int(pv_max_resample_attempts),
        pv_placeholder_token=str(pv_placeholder_token),
        diagnostics_summary_path=diagnostics_summary_path,
        diagnostics_records_path=diagnostics_records_path,
        # CDE enrichment consumes PV_BLOCK_CDE; query omission placeholders stay disabled here.
        use_placeholder_for_query_omission=False,
        sep=str(sep),
    )

    out = pv_sum[["cde_publicid", "cde_version", "PV_N", "PV_TYPE", "PV_BLOCK_CDE"]].copy()
    out = out.rename(columns={"PV_BLOCK_CDE": "PV_SUMMARY"})
    return out


def _preferred_question_text(
    refdocs: Optional[pd.DataFrame],
    language: str = "English",
) -> pd.DataFrame:
    """Pick one Preferred Question Text per CDE (if present)."""
    if refdocs is None or refdocs.empty:
        return pd.DataFrame(columns=["cde_publicid", "cde_version", "PREFERRED_QUESTION_TEXT"])

    required = ["cde_publicid", "cde_version", "document_type", "document_text"]
    missing = [c for c in required if c not in refdocs.columns]
    if missing:
        raise ValueError(f"cde_reference_documents missing required columns: {missing}")

    df = refdocs.copy()
    df["document_type"] = df["document_type"].map(norm_str)
    df["document_text"] = df["document_text"].map(norm_str)
    df = df[df["document_text"].map(lambda x: x != "")].copy()

    # Language filter (strict)
    if "language" in df.columns:
        lang_cf = norm_str(language).casefold()
        df["language"] = df["language"].map(norm_str)
        df = df[df["language"].str.casefold() == lang_cf].copy()

    # DocumentType filter (robust)
    dt_cf = df["document_type"].str.casefold()
    preferred_types = {"preferred question text", "preferredquestiontext"}
    df = df[dt_cf.isin(preferred_types)].copy()

    if df.empty:
        return pd.DataFrame(columns=["cde_publicid", "cde_version", "PREFERRED_QUESTION_TEXT"])

    # Choose "first" by display order when present.
    if "display_order" in df.columns:
        df["_order"] = pd.to_numeric(df["display_order"], errors="coerce")
    else:
        df["_order"] = pd.NA

    df = df.sort_values(
        ["cde_publicid", "cde_version", "_order"],
        ascending=[True, True, True],
        na_position="last",
        kind="mergesort",
    )
    df = df.drop_duplicates(subset=["cde_publicid", "cde_version"], keep="first")

    out = df[["cde_publicid", "cde_version", "document_text"]].rename(
        columns={"document_text": "PREFERRED_QUESTION_TEXT"}
    )
    return out.reset_index(drop=True)


def enrich_master(
    cde_master: pd.DataFrame,
    cde_dec: Optional[pd.DataFrame] = None,
    cde_value_domain: Optional[pd.DataFrame] = None,
    cde_permissible_values: Optional[pd.DataFrame] = None,
    cde_reference_documents: Optional[pd.DataFrame] = None,
    *,
    pv_summary_profile: str = "cadsr",
    pv_max_n: int = 10,
    pv_huge_threshold: int = 20,
    pv_max_n_query: Optional[int] = None,
    pv_max_n_cde: Optional[int] = None,
    pv_center_fraction: float = 0.5,
    pv_min_n: int = 2,
    pv_size_jitter: int = 1,
    pv_generic_boost: float = 0.10,
    pv_generic_cap: float = 0.70,
    sde_generic_label_p: float = 0.30,
    pv_small_n_generic_threshold: float = 0.5,
    pv_max_resample_attempts: int = 3,
    pv_placeholder_token: str = "<MISSING_PV_SUMMARY>",
    pv_diagnostics_summary_path: Optional[str] = None,
    pv_diagnostics_records_path: Optional[str] = None,
    language: str = "English",
) -> pd.DataFrame:
    """Join normalized tables into an enriched CDE master."""

    base = cde_master.copy()
    required = ["cde_publicid", "cde_version", "preferred_name", "long_name", "preferred_definition"]
    miss = [c for c in required if c not in base.columns]
    if miss:
        raise ValueError(f"cde_master missing required columns: {miss}")

    base = base.rename(
        columns={
            "preferred_name": "SHORT_NAME",
            "long_name": "LONG_NAME",
            "preferred_definition": "DEFINITION",
        }
    )

    # DEC long name
    if cde_dec is not None and not cde_dec.empty and "dec_long_name" in cde_dec.columns:
        dec = cde_dec[["cde_publicid", "cde_version", "dec_long_name"]].copy()
        dec = dec.rename(columns={"dec_long_name": "DEC_LONG_NAME"})
        dec["DEC_LONG_NAME"] = dec["DEC_LONG_NAME"].fillna("").astype(str)
        dec = _dedupe_keep_first(dec, ["cde_publicid", "cde_version"])
        base = base.merge(dec, on=["cde_publicid", "cde_version"], how="left")
    else:
        base["DEC_LONG_NAME"] = ""

    # Value domain
    if cde_value_domain is not None and not cde_value_domain.empty:
        cols = ["cde_publicid", "cde_version"]
        if "value_domain_type" in cde_value_domain.columns:
            cols.append("value_domain_type")
        if "datatype" in cde_value_domain.columns:
            cols.append("datatype")
        vd = cde_value_domain[cols].copy()
        vd = vd.rename(
            columns={
                "value_domain_type": "VALUE_DOMAIN_TYPE",
                "datatype": "VALUE_DOMAIN_DATATYPE",
            }
        )
        if "VALUE_DOMAIN_TYPE" not in vd.columns:
            vd["VALUE_DOMAIN_TYPE"] = ""
        if "VALUE_DOMAIN_DATATYPE" not in vd.columns:
            vd["VALUE_DOMAIN_DATATYPE"] = ""
        vd = _dedupe_keep_first(vd, ["cde_publicid", "cde_version"])
        base = base.merge(
            vd[["cde_publicid", "cde_version", "VALUE_DOMAIN_TYPE", "VALUE_DOMAIN_DATATYPE"]],
            on=["cde_publicid", "cde_version"],
            how="left",
        )
    else:
        base["VALUE_DOMAIN_TYPE"] = ""
        base["VALUE_DOMAIN_DATATYPE"] = ""

    # PV summary
    pv_sum = _summarize_permissible_values(
        cde_permissible_values,
        pv_summary_profile=pv_summary_profile,
        pv_max_n=pv_max_n,
        pv_huge_threshold=pv_huge_threshold,
        pv_max_n_query=pv_max_n_query,
        pv_max_n_cde=pv_max_n_cde,
        pv_center_fraction=pv_center_fraction,
        pv_min_n=pv_min_n,
        pv_size_jitter=pv_size_jitter,
        pv_generic_boost=pv_generic_boost,
        pv_generic_cap=pv_generic_cap,
        sde_generic_label_p=sde_generic_label_p,
        pv_small_n_generic_threshold=pv_small_n_generic_threshold,
        pv_max_resample_attempts=pv_max_resample_attempts,
        pv_placeholder_token=pv_placeholder_token,
        diagnostics_summary_path=pv_diagnostics_summary_path,
        diagnostics_records_path=pv_diagnostics_records_path,
    )
    base = base.merge(pv_sum, on=["cde_publicid", "cde_version"], how="left")
    if "PV_N" not in base.columns:
        base["PV_N"] = 0
    if "PV_TYPE" not in base.columns:
        base["PV_TYPE"] = ""
    if "PV_SUMMARY" not in base.columns:
        base["PV_SUMMARY"] = ""

    # Preferred question text
    pref_q = _preferred_question_text(cde_reference_documents, language=language)
    base = base.merge(pref_q, on=["cde_publicid", "cde_version"], how="left")
    if "PREFERRED_QUESTION_TEXT" not in base.columns:
        base["PREFERRED_QUESTION_TEXT"] = ""

    # Fill
    for c in [
        "SHORT_NAME",
        "LONG_NAME",
        "DEFINITION",
        "DEC_LONG_NAME",
        "VALUE_DOMAIN_TYPE",
        "VALUE_DOMAIN_DATATYPE",
        "PV_SUMMARY",
        "PV_TYPE",
        "PREFERRED_QUESTION_TEXT",
    ]:
        if c in base.columns:
            base[c] = base[c].fillna("").astype(str)

    base["PV_N"] = pd.to_numeric(base.get("PV_N", 0), errors="coerce").fillna(0).astype(int)

    base["cde_id"] = base["cde_publicid"].astype(str) + "::" + base["cde_version"].astype(str)
    base = _dedupe_keep_first(base, ["cde_id"])
    return base


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build an enriched cde_master parquet for v1-v6 recipes.")
    ap.add_argument("--merged-dir", default="data/interim/cadsr_merged", help="Directory containing merged caDSR tables.")
    ap.add_argument(
        "--out-parquet",
        default=os.path.join("data", "processed", "cde_master_enriched.parquet"),
        help="Output parquet path.",
    )
    ap.add_argument("--cde-master", default=None, help="Override path to merged cde_master.parquet")
    ap.add_argument("--cde-dec", default=None, help="Override path to merged cde_iso11179_dec.parquet")
    ap.add_argument("--cde-value-domain", default=None, help="Override path to merged cde_iso11179_value_domain.parquet")
    ap.add_argument("--cde-permissible-values", default=None, help="Override path to merged cde_permissible_values.parquet")
    ap.add_argument(
        "--refdocs-parquet",
        default=None,
        help="Override path to merged cde_reference_documents.parquet (for Preferred Question Text).",
    )
    ap.add_argument("--pv-max-n", type=int, default=10, help="Max number of PV meaning names to include in PV_SUMMARY.")
    ap.add_argument(
        "--pv-huge-threshold",
        type=int,
        default=20,
        help="Treat PV lists larger than this as 'huge' for example selection (accepted for compatibility).",
    )
    ap.add_argument("--pv-summary-profile", default="cadsr", help="PV summary profile (default: cadsr).")
    ap.add_argument("--pv-max-n-query", type=int, default=None, help="Optional query-side PV cap.")
    ap.add_argument("--pv-max-n-cde", type=int, default=None, help="Optional CDE-side PV cap.")
    ap.add_argument("--pv-center-fraction", type=float, default=0.5)
    ap.add_argument("--pv-min-n", type=int, default=2)
    ap.add_argument("--pv-size-jitter", type=int, default=1)
    ap.add_argument("--pv-generic-boost", type=float, default=0.10)
    ap.add_argument("--pv-generic-cap", type=float, default=0.70)
    ap.add_argument("--sde-generic-label-p", type=float, default=0.30)
    ap.add_argument("--pv-small-n-generic-threshold", type=float, default=0.5)
    ap.add_argument("--pv-max-resample-attempts", type=int, default=3)
    ap.add_argument("--pv-placeholder-token", default="<MISSING_PV_SUMMARY>")
    ap.add_argument("--pv-diagnostics-summary", default=None, help="Optional JSON path for PV summary diagnostics.")
    ap.add_argument("--pv-diagnostics-records", default=None, help="Optional CSV path for per-CDE PV diagnostics.")
    ap.add_argument("--language", default="English", help="Language filter for Preferred Question Text.")

    args = ap.parse_args(list(argv) if argv is not None else None)

    merged = args.merged_dir
    cde_master_path = args.cde_master or os.path.join(merged, "cde_master.parquet")
    cde_dec_path = args.cde_dec or os.path.join(merged, "cde_iso11179_dec.parquet")
    cde_vd_path = args.cde_value_domain or os.path.join(merged, "cde_iso11179_value_domain.parquet")
    cde_pv_path = args.cde_permissible_values or os.path.join(merged, "cde_permissible_values.parquet")
    refdocs_path = args.refdocs_parquet or os.path.join(merged, "cde_reference_documents.parquet")

    cde_master = pd.read_parquet(cde_master_path)
    cde_dec = pd.read_parquet(cde_dec_path) if os.path.exists(cde_dec_path) else None
    cde_vd = pd.read_parquet(cde_vd_path) if os.path.exists(cde_vd_path) else None
    cde_pv = pd.read_parquet(cde_pv_path) if os.path.exists(cde_pv_path) else None
    refdocs = pd.read_parquet(refdocs_path) if os.path.exists(refdocs_path) else None

    enriched = enrich_master(
        cde_master,
        cde_dec=cde_dec,
        cde_value_domain=cde_vd,
        cde_permissible_values=cde_pv,
        cde_reference_documents=refdocs,
        pv_summary_profile=args.pv_summary_profile,
        pv_max_n=args.pv_max_n,
        pv_huge_threshold=args.pv_huge_threshold,
        pv_max_n_query=args.pv_max_n_query,
        pv_max_n_cde=args.pv_max_n_cde,
        pv_center_fraction=args.pv_center_fraction,
        pv_min_n=args.pv_min_n,
        pv_size_jitter=args.pv_size_jitter,
        pv_generic_boost=args.pv_generic_boost,
        pv_generic_cap=args.pv_generic_cap,
        sde_generic_label_p=args.sde_generic_label_p,
        pv_small_n_generic_threshold=args.pv_small_n_generic_threshold,
        pv_max_resample_attempts=args.pv_max_resample_attempts,
        pv_placeholder_token=args.pv_placeholder_token,
        pv_diagnostics_summary_path=args.pv_diagnostics_summary,
        pv_diagnostics_records_path=args.pv_diagnostics_records,
        language=args.language,
    )

    _ensure_dir_for_file(args.out_parquet)
    enriched.to_parquet(args.out_parquet, index=False)
    print("Wrote:", args.out_parquet, "| rows:", len(enriched))


if __name__ == "__main__":
    main()
