#!/usr/bin/env python3
"""Build the CANONICAL corrected CIMAC v2 evaluation split (PV-enriched), non-destructively.

The old CIMAC v2 split had an empty ``PV_BLOCK_SDE`` (an evaluation-input defect unique to CIMAC),
so ``query_text_q3`` lacked permissible values. This script rebuilds CIMAC v2 with enriched PVs and
``query_text_q3 = query_text_raw | PV_BLOCK_SDE`` (caDSR convention, PV_CAP=10), preserving query_ids,
gold, and denominators. The 8 non-CIMAC splits are symlinked unchanged so the output dir is a drop-in.

Canonical output (default): ``data/processed/splits_v3_cdisc_reachable_2026-06-18_cimacpv/`` — this is
the **default eval split path** going forward. The old empty-PV split
(``data/processed/splits_v3_cdisc_reachable_2026-06-18/``) is preserved AUDIT-ONLY.

Hard guardrail: asserts the CIMAC strata are exactly **full_131 / exact_92 / non_exact_39** so a stale
or mis-built split cannot silently become canonical. Writes a manifest JSON next to the output.

Usage:
    PYTHONPATH=src python scripts/build_corrected_cimac_pv_split.py            # default canonical path
    PYTHONPATH=src python scripts/build_corrected_cimac_pv_split.py --check-only   # verify, don't write
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

#: Data tree (see demap_repro.utils.paths). Formerly ``parents[1]``, which after
#: packaging resolves to ``src/demap_repro/data``. The two ``sys.path`` inserts
#: that followed named directories inside the package and are gone.
REPO = data_root()

# PV enrichment source (capped fixed-tabs, PV_CAP=10) + legacy split dirs.
DEF_ENR = REPO / ".scratch/demap/sapbert_pv_ceiling/cimac_pv_audit/cimac_pv_enriched_fixed_tabs.parquet"
DEF_SRC154 = REPO / ".scratch/demap/keyword_attribution_2026-06-18/splits_symlinked_s2"
DEF_REACH_LEGACY = REPO / "data/processed/splits_v3_cdisc_reachable_2026-06-18"          # old empty-PV (audit)
DEF_OUT_REACH = REPO / "data/processed/splits_v3_cdisc_reachable_2026-06-18_cimacpv"      # corrected canonical
SPLITS = ["test", "val_dev", "val_train", "external_holdout_org", "external_holdout_refslice",
          "cimac_v2", "theradex6_test", "external_holdout_gdc_questiontext", "external_holdout_gdc_altnames"]


def _pv_type(block: str) -> str:
    m = re.match(r"PV_TYPE:\s*([^()]+)\(n=", str(block))
    return m.group(1).strip() if m else ""


def correct_cimac(df: pd.DataFrame, pv_block: dict, pv_n: dict) -> pd.DataFrame:
    df = df.copy()
    qid = df["query_id"].astype(str)
    blk = qid.map(lambda q: pv_block.get(q, "") or "")
    df["PV_BLOCK_SDE"] = blk
    if "PV_N" in df.columns:
        df["PV_N"] = qid.map(lambda q: int(pv_n.get(q, 0) or 0)).astype(df["PV_N"].dtype)
    df["PV_TYPE"] = blk.map(_pv_type)
    df["pv_attached"] = blk.str.strip() != ""
    if "pv_block" in df.columns:
        df["pv_block"] = blk
    raw = df["query_text_raw"].astype(str)
    df["query_text_q3"] = [f"{r} | {b}" if str(b).strip() else r for r, b in zip(raw, blk)]
    return df


def assert_strata():
    """Authoritative 131/92/39 guardrail via the canonical strata definition."""
    from eval_hgbc_cimac131 import _strata, DEF_ELIG, DEF_INTERIM, DEF_SPLIT
    full, exa, nex = _strata(Path(DEF_ELIG), Path(DEF_INTERIM), Path(DEF_SPLIT))
    assert (len(full), len(exa), len(nex)) == (131, 92, 39), \
        f"CIMAC strata are {(len(full), len(exa), len(nex))}, expected (131, 92, 39)"
    return len(full), len(exa), len(nex)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pv-enrichment", default=str(DEF_ENR))
    ap.add_argument("--src154-dir", default=str(DEF_SRC154))
    ap.add_argument("--out-reach", default=str(DEF_OUT_REACH))
    ap.add_argument("--date", default="2026-06-29", help="build date recorded in the manifest")
    ap.add_argument("--check-only", action="store_true", help="verify strata + inputs; do not write")
    args = ap.parse_args(argv)

    full, exa, nex = assert_strata()
    print(f"strata guardrail OK: full_131={full} exact_92={exa} non_exact_39={nex}")
    if args.check_only:
        print("--check-only: strata verified; not writing.")
        return 0

    enr = pd.read_parquet(args.pv_enrichment).set_index("query_id")
    pv_block = enr["PV_BLOCK_SDE_enriched"].to_dict()
    pv_n = enr["PV_N_enriched"].to_dict()

    out_reach = Path(args.out_reach)
    out_reach.mkdir(parents=True, exist_ok=True)
    src_dir = Path(args.src154_dir)
    legacy_reach = DEF_REACH_LEGACY
    for sp in SPLITS:
        dst = out_reach / f"{sp}.parquet"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if sp == "cimac_v2":
            correct_cimac(pd.read_parquet(legacy_reach / "cimac_v2.parquet"), pv_block, pv_n).to_parquet(dst, index=False)
        else:
            src = (legacy_reach / f"{sp}.parquet").resolve()
            dst.symlink_to(os.path.relpath(src, out_reach))

    d = pd.read_parquet(out_reach / "cimac_v2.parquet")
    n_pv = int((d["PV_BLOCK_SDE"].fillna("").astype(str).str.strip() != "").sum())
    n_q3 = int((d["query_text_q3"].astype(str) != d["query_text_raw"].astype(str)).sum())
    manifest = {
        "name": "corrected CIMAC v2 PV-enriched eval split (canonical)",
        "build_date": args.date,
        "raw_inputs": {
            "pv_enrichment": str(args.pv_enrichment),
            "base_reachable_split (audit, empty-PV)": str(legacy_reach),
            "raw_cimac_appendix": "data/raw/cimac/CIMAC-CIDC_Master_AppendixA.xlsx",
        },
        "pv_method": "PV_BLOCK_SDE enriched from capped fixed-tabs (PV_CAP=10, caDSR convention); "
                     "query_text_q3 = query_text_raw | PV_BLOCK_SDE",
        "corrected_path (canonical, default)": str(out_reach),
        "old_audit_path (empty-PV, do NOT use)": str(legacy_reach),
        "cimac_v2_rows": int(len(d)), "cimac_v2_unique_queries": int(d["query_id"].nunique()),
        "queries_with_PV_BLOCK_SDE": n_pv, "queries_q3_differs_from_raw": n_q3,
        "reachable_gold_full_131": full, "exact_reachable_92": exa, "non_exact_reachable_39": nex,
        "regenerate_command": "PYTHONPATH=src python scripts/build_corrected_cimac_pv_split.py",
    }
    (out_reach / "CORRECTED_CIMAC_PV_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote corrected split + manifest under: {out_reach}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
