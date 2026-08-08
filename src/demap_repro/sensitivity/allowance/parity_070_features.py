#!/usr/bin/env python3
"""Task J Phase-2 parity check: the rebuilt 0.70 tables must match the archived
manuscript tables.

  base:   rates/a0.70/fixedk30_base/feature_table.parquet
     vs   artifacts/final_reranker/hgbc_features_v2_eligible/fixedk30_base/feature_table.parquet
  merged: rates/a0.70/feature_table_fixedk30_crossenc.parquet
     vs   artifacts/final_reranker/hgbc_features_v2_eligible/feature_table_fixedk30_crossenc_v2.parquet

Non-CE columns are expected EXACTLY equal (deterministic CPU pipeline);
crossenc_* columns are compared with a float tolerance (GPU batch-composition
jitter is possible in principle). Writes rates/a0.70/PARITY_FEATURES.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/vf/users/nextgen2/james/tasks/cde_project/demap")
J = REPO / ".scratch/v17_claude/J_allowance"
KEY = ["split", "query_id", "cde_id"]

PAIRS = [
    ("base",
     J / "rates/a0.70/fixedk30_base/feature_table.parquet",
     REPO / "artifacts/final_reranker/hgbc_features_v2_eligible/fixedk30_base/feature_table.parquet"),
    ("merged",
     J / "rates/a0.70/feature_table_fixedk30_crossenc.parquet",
     REPO / "artifacts/final_reranker/hgbc_features_v2_eligible/feature_table_fixedk30_crossenc_v2.parquet"),
]

report = {}

def main(argv=None) -> int:
    """Prove the 70% rebuild matches the shipped pipeline."""
    for name, new_p, old_p in PAIRS:
        new = pd.read_parquet(new_p)
        old = pd.read_parquet(old_p)
        rec = {"n_new": len(new), "n_old": len(old),
               "cols_new_only": sorted(set(new.columns) - set(old.columns)),
               "cols_old_only": sorted(set(old.columns) - set(new.columns))}
        a = new.sort_values(KEY, kind="mergesort").reset_index(drop=True)
        b = old.sort_values(KEY, kind="mergesort").reset_index(drop=True)
        keys_equal = (len(a) == len(b)) and all(
            (a[k].astype(str).values == b[k].astype(str).values).all() for k in KEY)
        rec["row_keys_identical"] = bool(keys_equal)
        if keys_equal:
            exact_bad, ce_stats = {}, {}
            shared = [c for c in b.columns if c in a.columns and c not in KEY]
            for c in shared:
                xa, xb = a[c], b[c]
                is_ce = c.startswith("crossenc_")
                if xa.dtype.kind in "fciub" or xb.dtype.kind in "fciub":
                    va = pd.to_numeric(xa, errors="coerce").astype(float).values
                    vb = pd.to_numeric(xb, errors="coerce").astype(float).values
                    both_nan = np.isnan(va) & np.isnan(vb)
                    diff = np.abs(va - vb)
                    neq = ~(both_nan | (va == vb))
                    if is_ce:
                        with np.errstate(invalid="ignore"):
                            md = float(np.nanmax(np.where(both_nan, 0.0, diff))) if len(diff) else 0.0
                        ce_stats[c] = {"n_not_bitequal": int(neq.sum()), "max_abs_diff": md}
                    elif neq.sum():
                        exact_bad[c] = {"n_diff": int(neq.sum()),
                                        "max_abs_diff": float(np.nanmax(diff[neq]))}
                else:
                    neq = ~((xa.astype(str) == xb.astype(str)) | (xa.isna() & xb.isna()))
                    if int(neq.sum()):
                        exact_bad[c] = {"n_diff": int(neq.sum())}
            rec["non_ce_columns_exactly_equal"] = not exact_bad
            rec["non_ce_diffs"] = exact_bad
            rec["ce_column_stats"] = ce_stats
        report[name] = rec
        print(f"[parity {name}] keys_identical={rec['row_keys_identical']} "
              f"non_ce_exact={rec.get('non_ce_columns_exactly_equal')} ")
        if rec.get("ce_column_stats"):
            for c, s in rec["ce_column_stats"].items():
                print(f"    {c}: not_bitequal={s['n_not_bitequal']} max_abs_diff={s['max_abs_diff']:.3e}")

    out = J / "rates/a0.70/PARITY_FEATURES.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
