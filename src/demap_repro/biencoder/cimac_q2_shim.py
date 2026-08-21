#!/usr/bin/env python3
"""Deterministically materialize the Q2 (`query_text`) column for canonical CIMAC v2.

CIMAC v2 (`data/processed/eval_canonical/cimac_v2.parquet`) never passed through
`demap build-queries`, so it carries `query_text_raw` and `query_text_q3` but no
`query_text` (Q2). Every other canonical eval set has it.

This script reproduces the EXACT Q2 construction from
`src/demap/dataset/inwild_queries.py` (`build_queries`):

    ALT rows (line 2177):  query_text = "ALT_NAME: " + query_text_raw
    _attach_pv_block (2087/2114), called at 2208 with label="VALUE_DOMAIN":
        pv_attached = (PV_N > 0) & (normalize_query_text(PV_BLOCK_SDE) != "")
        query_text[pv_attached] += " | VALUE_DOMAIN: " + PV_BLOCK_SDE

CIMAC is classified ALT-style 100.0% / REF 0.0% in the manuscript's Table 1, so
the ALT convention is applied. ALT/REF are query STYLES, not provenance claims.

REGRESSION GATE: the same code path is used to rebuild `query_text_q3`
(`_add_q3_nolabel`, line 2135) and must reproduce the stored column byte-identically
for all 131 rows, otherwise this script aborts.

The canonical parquet is NEVER modified in place (its sha256 is recorded in every
published canonical_audit.csv). Output goes to a shim splits dir under .scratch/.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

#: Data and artifact tree. This was an absolute path into the research
#: repository, which made the module unusable anywhere else; see
#: ``demap_repro.utils.paths`` and ``DEMAP_DATA_ROOT``.
REPO = data_root()
from demap_repro.data.queries import normalize_query_text

SRC = REPO / "data/processed/eval_canonical/cimac_v2.parquet"
OUT_DIR = REPO / ".scratch/v20_claude/rerun/_q2splits"
OUT = OUT_DIR / "cimac_v2.parquet"

ALT_PREFIX = "ALT_NAME: "
PV_LABEL = "VALUE_DOMAIN"
EXPECTED_SRC_SHA256 = "1d0797330864cb8a73d277c1885fa5bba41a63d3a011cfaaa148e59c6dce4fe0"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    src_sha = sha256_file(SRC)
    print(f"[src] {SRC}")
    print(f"[src] sha256={src_sha}")
    if src_sha != EXPECTED_SRC_SHA256:
        raise SystemExit(f"[FAIL] source parquet sha256 changed: {src_sha} != {EXPECTED_SRC_SHA256}")

    d = pd.read_parquet(SRC)
    n = len(d)
    print(f"[src] rows={n} cols={len(d.columns)}")
    if n != 131:
        raise SystemExit(f"[FAIL] expected 131 rows, got {n}")
    if "query_text" in d.columns:
        raise SystemExit("[FAIL] source already has query_text; refusing to overwrite")
    for c in ("query_text_raw", "PV_N", "PV_BLOCK_SDE", "pv_attached", "query_text_q3"):
        if c not in d.columns:
            raise SystemExit(f"[FAIL] required column missing from CIMAC: {c}  -> Q2 CANNOT BE BUILT")

    raw = d["query_text_raw"].fillna("").astype(str)
    blk = d["PV_BLOCK_SDE"].fillna("").astype(str)
    pv_n = pd.to_numeric(d["PV_N"], errors="coerce").fillna(0).astype(int)

    # --- recompute pv_attached exactly as _attach_pv_block does (line 2111) ---
    pv_attached = (pv_n > 0) & blk.map(lambda x: normalize_query_text(x) != "")
    stored_attached = d["pv_attached"].fillna(False).astype(bool)
    n_mismatch = int((pv_attached != stored_attached).sum())
    print(f"[check] pv_attached recomputed vs stored: {n - n_mismatch}/{n} match")
    if n_mismatch:
        raise SystemExit(f"[FAIL] pv_attached mismatch on {n_mismatch} rows")

    # --- REGRESSION GATE: rebuild Q3 via _add_q3_nolabel logic (line 2135) ---
    q3 = raw.copy()
    m3 = pv_attached & blk.map(lambda x: normalize_query_text(x) != "")
    q3.loc[m3] = raw.loc[m3] + " | " + blk.loc[m3]
    stored_q3 = d["query_text_q3"].fillna("").astype(str)
    n_q3_ok = int((q3 == stored_q3).sum())
    print(f"[REGRESSION] query_text_q3 reproduced byte-identically: {n_q3_ok}/{n}")
    if n_q3_ok != n:
        bad = d.index[q3 != stored_q3][:5]
        for i in bad:
            print(f"  MISMATCH row {i}\n    built = {q3.loc[i]!r}\n    stored= {stored_q3.loc[i]!r}")
        raise SystemExit("[FAIL] Q3 regression check failed -> transformation is WRONG, aborting")

    # --- build Q2 with the same code path ---
    q2 = ALT_PREFIX + raw
    q2.loc[pv_attached] = q2.loc[pv_attached] + " | " + PV_LABEL + ": " + blk.loc[pv_attached]

    # Q2 must equal Q3 with exactly the label tokens inserted.
    reverted = q2.str.replace(f" | {PV_LABEL}: ", " | ", regex=False).str.slice(len(ALT_PREFIX))
    n_rev = int((reverted == stored_q3).sum())
    print(f"[check] Q2 minus label tokens == stored Q3: {n_rev}/{n}")
    if n_rev != n:
        raise SystemExit("[FAIL] Q2 is not Q3-plus-labels; construction is inconsistent")

    out = d.copy()
    out["query_text"] = q2.values

    n_pv = int(pv_attached.sum())
    print(f"[q2] pv_attached rows (get ' | {PV_LABEL}: '): {n_pv}/{n}; prefix-only: {n - n_pv}/{n}")
    print(f"[q2] all start with {ALT_PREFIX!r}: {int(out['query_text'].str.startswith(ALT_PREFIX).sum())}/{n}")
    print("[q2] examples:")
    for i in list(out.index[pv_attached])[:2] + list(out.index[~pv_attached])[:2]:
        print(f"  {out.loc[i, 'query_text']!r}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    out_sha = sha256_file(OUT)
    print(f"[out] {OUT}\n[out] rows={len(out)} cols={len(out.columns)} sha256={out_sha}")

    manifest = {
        "source_parquet": str(SRC.relative_to(REPO)),
        "source_sha256": src_sha,
        "output_parquet": str(OUT.relative_to(REPO)),
        "output_sha256": out_sha,
        "n_rows": n,
        "alt_prefix": ALT_PREFIX,
        "pv_label": PV_LABEL,
        "n_pv_attached": n_pv,
        "q3_regression_rows_matched": n_q3_ok,
        "pv_attached_rows_matched": n - n_mismatch,
        "construction": 'query_text = "ALT_NAME: " + query_text_raw '
                        '[+ " | VALUE_DOMAIN: " + PV_BLOCK_SDE if pv_attached]',
        "source_of_truth": "src/demap/dataset/inwild_queries.py::build_queries "
                           "(lines 2177 ALT, 2087/2114 _attach_pv_block, 2208 label=VALUE_DOMAIN)",
        "canonical_parquet_modified": False,
    }
    (OUT_DIR / "q2_build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[out] manifest -> {OUT_DIR / 'q2_build_manifest.json'}")
    print("[OK] Q2 shim built; canonical parquet untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
