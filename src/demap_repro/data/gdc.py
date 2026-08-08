"""demap_repro.data.gdc

Utilities for importing externally curated query-to-CDE match tables.

This module converts the two GDC annotation CSVs (ALT-name and Question-Text
batches) into the repo's standard split parquet schema so they can be evaluated
by:

  - demap_repro.biencoder.engine.baseline_grid (off-the-shelf embedding runs)
  - demap.experiments.eval_cdematch (external ranking exports)
  - demap.analysis.dataset_diagnostics

Input CSV format (as provided by the user)
-----------------------------------------
Required columns:
  - Batch Name
  - Seq ID
  - Entity
  - Perm Val
  - Preferred CDE ID

Notes
-----
- The source CSVs repeat each (Seq ID) row once per permissible value.
  We collapse to one row per (Batch Name, Seq ID).
- The CSVs do not include CDE version. We resolve version from the repo's
  cde_master_enriched.parquet by selecting the maximum version for each
  cde_publicid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from demap_repro.text.normalize import normalize_query_text
from demap_repro.text.pv_summary import build_pv_summary_table


PV_PLACEHOLDER_DEFAULT = "<MISSING_PV_SUMMARY>"


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _norm_overlap(x: Any) -> str:
    """Conservative normalization for overlap audits.

    We use the repo's query normalizer (NFKC + whitespace + punctuation
    normalization) and then casefold to make overlap detection robust to case.
    """
    return normalize_query_text(x).casefold()


def _dedupe_preserve_order(xs: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def resolve_publicid_to_version_map(cde_master_enriched: pd.DataFrame) -> Dict[str, str]:
    """Return map {cde_publicid -> max(cde_version)} as strings."""
    if not {"cde_publicid", "cde_version"}.issubset(cde_master_enriched.columns):
        raise KeyError("cde_master_enriched must contain cde_publicid and cde_version")

    df = cde_master_enriched[["cde_publicid", "cde_version"]].copy()
    df["cde_publicid"] = df["cde_publicid"].astype(str)

    # Robust conversion: versions are often numeric but may be strings.
    v_num = pd.to_numeric(df["cde_version"], errors="coerce")
    df["_v_num"] = v_num

    out: Dict[str, str] = {}
    for pid, g in df.groupby("cde_publicid", sort=False):
        if g["_v_num"].notna().any():
            vmax = float(g["_v_num"].max())
            if vmax.is_integer():
                out[pid] = str(int(vmax))
            else:
                out[pid] = str(vmax)
        else:
            out[pid] = str(g["cde_version"].astype(str).max())
    return out


def load_master_for_version_resolution(path: Path) -> pd.DataFrame:
    """Load the minimal master columns needed to resolve CDE versions."""
    try:
        return pd.read_parquet(path, columns=["cde_publicid", "cde_version"])
    except Exception:
        return pd.read_parquet(path)


@dataclass
class ImportManifest:
    """Lightweight import manifest suitable for paper reproducibility."""

    source_csv: str
    query_kind: str
    batch_names: List[str]
    n_rows_csv: int
    n_unique_seq: int
    n_output_rows: int
    n_unique_cde_publicid: int
    n_missing_cde_version: int
    n_query_overlap_with_train: int
    dropped_overlaps: bool
    pv_col: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_csv": self.source_csv,
            "query_kind": self.query_kind,
            "batch_names": list(self.batch_names),
            "n_rows_csv": int(self.n_rows_csv),
            "n_unique_seq": int(self.n_unique_seq),
            "n_output_rows": int(self.n_output_rows),
            "n_unique_cde_publicid": int(self.n_unique_cde_publicid),
            "n_missing_cde_version": int(self.n_missing_cde_version),
            "n_query_overlap_with_train": int(self.n_query_overlap_with_train),
            "dropped_overlaps": bool(self.dropped_overlaps),
            "pv_col": str(self.pv_col),
            "notes": str(self.notes),
        }


def build_gdc_split_df(
    df_csv: pd.DataFrame,
    *,
    query_kind: str,
    cde_publicid_to_version: Dict[str, str],
    pv_col: str = "Perm Val",
    pv_placeholder: str = PV_PLACEHOLDER_DEFAULT,
    pv_join: str = " | ",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convert a GDC CSV dataframe into a split dataframe.

    Returns (df_split, stats), where df_split is one row per Seq ID.
    """

    required = {"Batch Name", "Seq ID", "Entity", "Preferred CDE ID"}
    missing = [c for c in sorted(required) if c not in df_csv.columns]
    if missing:
        raise KeyError(f"Missing required columns in GDC CSV: {missing}")

    qk = str(query_kind).strip().lower()
    if qk not in {"alt_names", "question_text"}:
        raise ValueError("query_kind must be 'alt_names' or 'question_text'")

    df = df_csv.copy()
    df["Batch Name"] = df["Batch Name"].fillna("").astype(str)
    df["Entity"] = df["Entity"].fillna("").astype(str)

    # Normalize Preferred CDE ID to a clean string.
    pid_raw = df["Preferred CDE ID"]
    pid_num = pd.to_numeric(pid_raw, errors="coerce")
    pid_fallback = pid_raw.astype(str)

    pid_str: List[str] = []
    for x_num, x_fb in zip(pid_num.tolist(), pid_fallback.tolist()):
        if x_num is None or (isinstance(x_num, float) and pd.isna(x_num)):
            pid_str.append(str(x_fb))
            continue
        try:
            xf = float(x_num)
            pid_str.append(str(int(xf)) if xf.is_integer() else str(xf))
        except Exception:
            pid_str.append(str(x_fb))

    df["cde_publicid"] = pd.Series(pid_str, index=df.index).astype(str)

    # Resolve version from master map.
    df["cde_version"] = df["cde_publicid"].map(lambda x: cde_publicid_to_version.get(str(x), ""))
    df["cde_id"] = df["cde_publicid"].astype(str) + "::" + df["cde_version"].astype(str)

    # PV column is optional, but user requested Perm Val.
    if pv_col in df.columns:
        df[pv_col] = df[pv_col].fillna("").astype(str)
    else:
        df[pv_col] = ""

    rows: List[Dict[str, Any]] = []
    missing_version_n = 0

    def _pv_summary_from_values(
        *,
        cde_publicid: str,
        cde_version: str,
        pv_values: Sequence[str],
        sep: str,
    ) -> Tuple[int, str, str, str, bool]:
        """Build a PV summary block using the repo's shared PV-summary code.

        The in-wild datasets attach PV summaries produced by
        :func:`demap_repro.text.pv_summary.build_pv_summary_table`.

        GDC CSVs provide a per-query PV list (one row per PV). To keep the
        GDC split semantics intact while unifying the PV-summary behavior,
        we synthesize a row-level PV table from the provided PV strings and
        run the same summarizer.

        Returns
        -------
        (PV_N, PV_TYPE, PV_BLOCK_SDE, PV_BLOCK_CDE, pv_attached)
        """

        pv_vals = [normalize_query_text(x) for x in pv_values if normalize_query_text(x) != ""]
        pv_vals = _dedupe_preserve_order(pv_vals)
        if not pv_vals:
            return 0, "", "", "", False

        # Preserve the CSV PV order via an explicit display_order column.
        pv_df = pd.DataFrame(
            {
                "cde_publicid": [str(cde_publicid)] * len(pv_vals),
                "cde_version": [str(cde_version)] * len(pv_vals),
                # GDC provides only one PV string; treat it as both code and meaning.
                "valid_value": pv_vals,
                "value_meaning": pv_vals,
                "meaning_concept_display_order": list(range(len(pv_vals))),
            }
        )

        # Use default PV-summary parameters (pv_max_n=10, pv_huge_threshold=20,
        # sde_generic_label_p=30, salt='demap') to match the standard pipeline.
        summ = build_pv_summary_table(pv_df, profile="gdc_strict", sep=str(sep))
        if summ is None or summ.empty:
            return 0, "", "", "", False

        r0 = summ.iloc[0]
        pv_n_raw = pd.to_numeric(r0.get("PV_N", 0), errors="coerce")
        pv_n = int(0 if pv_n_raw is None or pd.isna(pv_n_raw) else pv_n_raw)
        pv_type = str(r0.get("PV_TYPE", "") or "")
        pv_block_sde = str(r0.get("PV_BLOCK_SDE", "") or "")
        pv_block_cde = str(r0.get("PV_BLOCK_CDE", "") or "")
        attached = (pv_n > 0) and (normalize_query_text(pv_block_sde) != "")
        return pv_n, pv_type, pv_block_sde, pv_block_cde, attached

    for (batch_name, seq_id), g in df.groupby(["Batch Name", "Seq ID"], sort=False):
        ent = ""
        for x in g["Entity"].tolist():
            if normalize_query_text(x) != "":
                ent = str(x)
                break

        cde_publicid = str(g["cde_publicid"].iloc[0])
        cde_version = str(g["cde_version"].iloc[0])
        if normalize_query_text(cde_version) == "":
            missing_version_n += 1

        PV_N, PV_TYPE, PV_BLOCK_SDE, PV_BLOCK_CDE, pv_attached = _pv_summary_from_values(
            cde_publicid=cde_publicid,
            cde_version=cde_version,
            pv_values=g[pv_col].tolist(),
            sep=pv_join,
        )

        # Backwards-compatible alias: older analyses referenced `pv_block`.
        pv_block = PV_BLOCK_SDE

        if qk == "alt_names":
            query_source = "GDC_ALT"
            query_field = "alternate_name"
            prefix = "ALT_NAME: "
            pv_label = "VALUE_DOMAIN"
            family = "external_holdout_gdc_altnames"
        else:
            query_source = "GDC_QTXT"
            query_field = "preferred_question_text"
            prefix = "QUESTION_TEXT: "
            pv_label = "ANSWER_CHOICES"
            family = "external_holdout_gdc_questiontext"

        query_text_raw = ent

        query_text = prefix + ent if ent else prefix.strip()
        if pv_attached:
            query_text = query_text + " | " + pv_label + ": " + pv_block

        query_text_q3 = query_text_raw
        if pv_attached:
            query_text_q3 = (query_text_raw.strip() + " | " + pv_block).strip()

        base = query_text_raw.strip()
        if pv_attached:
            query_text_q4 = (base + " | " + pv_block).strip()
        else:
            query_text_q4 = (base + " | " + pv_placeholder).strip() if base else str(pv_placeholder)

        query_id = _sha1_hex(f"{query_source}::{query_text_raw}")
        pair_id = _sha1_hex(f"{query_source}::{query_text_raw}::{cde_publicid}::{cde_version}")

        rows.append(
            {
                "split_source": "gdc_csv",
                "batch_name": str(batch_name),
                "seq_id": int(seq_id) if str(seq_id).isdigit() else str(seq_id),
                "query_source": query_source,
                "query_field": query_field,
                "family": family,
                "query_id": query_id,
                "pair_id": pair_id,
                "query_text_raw": query_text_raw,
                "query_text": query_text,
                "query_text_q3": query_text_q3,
                "query_text_q4": query_text_q4,
                # PV summary fields (match in-wild query schema where possible).
                "PV_N": int(PV_N),
                "PV_TYPE": str(PV_TYPE),
                "PV_BLOCK_SDE": str(PV_BLOCK_SDE),
                "PV_BLOCK_CDE": str(PV_BLOCK_CDE),
                "pv_attached": bool(pv_attached),
                # Backwards-compat alias used by older diagnostics.
                "pv_block": str(pv_block),
                "cde_publicid": cde_publicid,
                "cde_version": cde_version,
                "cde_id": f"{cde_publicid}::{cde_version}",
            }
        )

    out = pd.DataFrame(rows)
    stats = {
        "n_rows_csv": int(len(df_csv)),
        "n_unique_seq": int(df_csv["Seq ID"].nunique()),
        "n_output_rows": int(len(out)),
        "n_unique_cde_publicid": int(out["cde_publicid"].nunique()) if not out.empty else 0,
        "n_missing_cde_version": int(missing_version_n),
        "batch_names": sorted([b for b in df["Batch Name"].unique().tolist() if str(b).strip() != ""])[:50],
    }
    return out, stats


def audit_query_overlap(
    df_new: pd.DataFrame,
    *,
    train_split_parquet: Optional[Path],
    train_query_col: str = "query_text_raw",
    new_query_col: str = "query_text_raw",
) -> Tuple[int, List[str]]:
    """Return (n_overlaps, sample_overlaps) against train split query strings."""
    if train_split_parquet is None or not train_split_parquet.exists():
        return 0, []

    try:
        df_train = pd.read_parquet(train_split_parquet, columns=[train_query_col])
    except Exception:
        df_train = pd.read_parquet(train_split_parquet)

    if train_query_col not in df_train.columns:
        return 0, []

    train_norm = df_train[train_query_col].fillna("").astype(str).map(_norm_overlap)
    train_set = set([x for x in train_norm.tolist() if x != ""])

    new_norm = df_new[new_query_col].fillna("").astype(str).map(_norm_overlap)
    overlaps = [x for x in new_norm.tolist() if x in train_set and x != ""]

    # sample up to 20 distinct overlaps
    sample: List[str] = []
    seen = set()
    for x in overlaps:
        if x not in seen:
            sample.append(x)
            seen.add(x)
        if len(sample) >= 20:
            break

    return int(len(overlaps)), sample


def write_import_manifest(path: Path, manifests: Sequence[ImportManifest]) -> None:
    obj = {
        "manifest_version": 1,
        "imports": [m.to_dict() for m in manifests],
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
