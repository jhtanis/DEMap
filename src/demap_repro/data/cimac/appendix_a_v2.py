#!/usr/bin/env python3
"""Build a processed CIMAC Appendix A **v2** eval split from the updated raw workbook.

Source: ``data/raw/cimac/CIMAC-CIDC_Master_AppendixA_v2.xlsx`` , sheet ``Appendix A``.
The raw gold (``CDE`` column) is **public-ID only** (no version). Versions are resolved by
looking the public id up in the June-16 caDSR eval catalogue and picking the current/best version
(preferring an eligible RELEASED row, else the max version present); rows whose public id is absent
are kept with a ``gold_versioned=False`` flag (public-id-level only).

One row per Data Element (the CIMAC field). 3 data elements carry multiple CDEs (newline-separated)
— the first is the primary ``cde_id``; all are kept in ``gold_publicids_all`` with ``n_gold``.

Outputs (default):
  data/processed/splits/cimac_appendix_a_v2.parquet
  data/processed/splits/cimac_appendix_a_v2.csv            (inspection copy)
  data/processed/splits/cimac_appendix_a_v2.manifest.json

Usage:
  PYTHONPATH=src .venv/bin/python scripts/build_cimac_appendix_a_v2_split.py
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import List, Optional
import pandas as pd

from demap_repro.utils.paths import data_root

#: Data tree (see demap_repro.utils.paths). Formerly ``parents[1]``, which after
#: packaging resolves to ``src/demap_repro/data``.
REPO = data_root()
RAW = REPO / "data/raw/cimac/CIMAC-CIDC_Master_AppendixA_v2.xlsx"
CATALOG = REPO / "data/processed/cadsr_xml_2026-06-16/cde_master_enriched_eval.parquet"
OUT = REPO / "data/processed/splits/cimac_appendix_a_v2.parquet"
SHEET = "Appendix A"
HEADER_TOKEN = "Data Element"


def _find_header(raw: pd.DataFrame) -> int:
    for i in range(min(20, len(raw))):
        cells = [str(v).strip() for v in raw.iloc[i].tolist()]
        if any(c == HEADER_TOKEN for c in cells) and any(c == "CDE" for c in cells):
            return i
    raise RuntimeError("could not locate the Appendix A header row")


def _parse_golds(cde_cell: object) -> List[str]:
    """Return the list of public ids in a CDE cell (handles newline / whitespace separated)."""
    if cde_cell is None or (isinstance(cde_cell, float) and pd.isna(cde_cell)):
        return []
    return re.findall(r"\d+", str(cde_cell))


def _version_resolver(catalog_path: Path):
    """publicid(str) -> (version(str)|None, how)."""
    df = pd.read_parquet(catalog_path, columns=["cde_publicid", "cde_version", "workflow_status"])
    df["cde_publicid"] = df["cde_publicid"].astype(str)
    df["vnum"] = pd.to_numeric(df["cde_version"], errors="coerce")
    df["released"] = df["workflow_status"].astype(str).str.upper().eq("RELEASED")
    by_pid = {}
    for pid, g in df.groupby("cde_publicid"):
        by_pid[pid] = g
    def resolve(pid: str):
        g = by_pid.get(str(pid))
        if g is None:
            return None, "publicid_absent_from_catalog"
        rel = g[g["released"]]
        pick = rel if len(rel) else g
        row = pick.sort_values("vnum", ascending=False).iloc[0]
        how = "released_max_version" if len(rel) else "max_version_any_status"
        return str(row["cde_version"]), how
    return resolve


def build(raw_path: Path, catalog_path: Path, out_path: Path) -> dict:
    raw = pd.read_excel(raw_path, SHEET, header=None)
    hdr = _find_header(raw)
    cols = [str(v).strip() for v in raw.iloc[hdr].tolist()]
    df = raw.iloc[hdr + 1:].reset_index(drop=True)
    df.columns = cols
    df["_source_row"] = range(hdr + 2, hdr + 2 + len(df))  # 1-based Excel-ish row id
    df = df.dropna(how="all")

    de = df["Data Element"].astype(str).str.strip().replace("nan", "")
    data = df[de != ""].copy()  # rows that are actual data elements (drop category separators)
    n_dataelements = len(data)

    resolve = _version_resolver(catalog_path)
    rows = []
    dropped = []
    for i, (_, r) in enumerate(data.iterrows(), start=1):
        delem = str(r["Data Element"]).strip()
        cat = str(r.get("Data Category", "")).strip()
        golds = _parse_golds(r.get("CDE"))
        rec = {
            "query_id": f"cimac_appa_v2::{i:03d}",
            "source_row": int(r["_source_row"]),
            "query_source": "CIMAC_APPA_V2",
            "query_field": "data_element",
            "family": "cimac_appendix_a_v2",
            "data_category": cat,
            "data_element": delem,
            "query_text_raw": delem,                      # retrieval text = the CIMAC field name
            "description": str(r.get("Description", "")).strip(),
            "data_type": str(r.get("Data Type", "")).strip(),
            "required": str(r.get("Required", "")).strip(),
        }
        if not golds:
            rec.update(dict(cde_publicid="", cde_version="", cde_id="",
                            gold_publicids_all="", n_gold=0,
                            gold_versioned=False, gold_resolution="no_gold_in_raw"))
            dropped.append({"query_id": rec["query_id"], "data_element": delem, "reason": "no_gold_in_raw"})
        else:
            pid = golds[0]
            ver, how = resolve(pid)
            versioned = ver is not None
            rec.update(dict(
                cde_publicid=pid,
                cde_version=(ver or ""),
                cde_id=(f"{pid}::{ver}" if versioned else pid),
                gold_publicids_all="|".join(golds),
                n_gold=len(golds),
                gold_versioned=bool(versioned),
                gold_resolution=how,
            ))
        rows.append(rec)

    out = pd.DataFrame(rows)
    # The same data_element name can recur across data categories with DIFFERENT golds
    # (e.g. received_dose_units), so data_element alone is an ambiguous retrieval key; the
    # unique key is (data_category, data_element). Add a context-qualified query text and a
    # duplicate-name flag. query_text_raw stays = data_element (matches the existing 154-entity
    # CDE Match request Entity convention); query_text_context disambiguates by category.
    dup_counts = out["data_element"].value_counts()
    out["dup_data_element"] = out["data_element"].map(lambda d: bool(dup_counts.get(d, 0) > 1))
    out["query_text_context"] = (out["data_category"].astype(str).str.strip()
                                 + " | " + out["data_element"].astype(str).str.strip())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    out.to_csv(out_path.with_suffix(".csv"), index=False)

    labeled = out[out["n_gold"] > 0]
    manifest = {
        "source": str(raw_path), "sheet": SHEET, "catalog_for_version_resolution": str(catalog_path),
        "n_rows_total_sheet": int(len(raw) - hdr - 1),
        "n_data_elements": int(n_dataelements),
        "n_with_gold": int(len(labeled)),
        "n_without_gold": int((out["n_gold"] == 0).sum()),
        "n_multi_gold": int((out["n_gold"] > 1).sum()),
        "n_gold_versioned": int(out["gold_versioned"].sum()),
        "n_gold_publicid_only": int(((out["n_gold"] > 0) & (~out["gold_versioned"])).sum()),
        "unique_query_id": int(out["query_id"].nunique()),
        "unique_data_element_names": int(out["data_element"].nunique()),
        "n_rows_duplicate_name": int(out["dup_data_element"].sum()),
        "unique_category_element_pairs": int(out[["data_category", "data_element"]].drop_duplicates().shape[0]),
        "unique_gold_publicid": int(labeled["cde_publicid"].nunique()),
        "gold_resolution_counts": out["gold_resolution"].value_counts().to_dict() if "gold_resolution" in out else {},
        "dropped_no_gold": dropped,
        "out_parquet": str(out_path),
    }
    (out_path.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--catalog", default=str(CATALOG))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    m = build(Path(a.raw), Path(a.catalog), Path(a.out))
    print("== CIMAC Appendix A v2 split ==")
    for k, v in m.items():
        if k in ("dropped_no_gold",):
            print(f"  {k}: {len(v)} rows")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
