#!/usr/bin/env python3
"""build_pairs.py

Create the main *query→CDE* table used for splitting, evaluation, and
fine-tuning.

In this refactored repo, **pairs.parquet does not bake in a single CDE text
representation**. Instead, we keep:

- Query text variants (Q1: query_text_raw, Q2: query_text, optional Q3: query_text_q3)
- Ground-truth target identifiers (cde_publicid, cde_version, cde_id)
- Provenance metadata (family, query_source, alternate_name_type, document_type, name, ...)

Target-side CDE text representations (v1-v6) are constructed later from
`cde_master_enriched.parquet` based on experiment configuration.

Inputs
------
- queries.parquet (from `demap build-queries`)

Outputs
-------
- pairs.parquet (default: data/processed/pairs.parquet)

Notes
-----
Compatibility note
------------------
`demap build-queries` already emits stable identifiers (query_id, pair_id).
This step exists primarily to:
  - ensure `cde_id` is present
  - preserve / (if missing) create a stable `pair_id`
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional

import pandas as pd


def _ensure_dir_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def build_pairs(queries: pd.DataFrame) -> pd.DataFrame:
    required = [
        "query_id",
        "query_text_raw",
        "query_text",
        "cde_publicid",
        "cde_version",
    ]
    missing = [c for c in required if c not in queries.columns]
    if missing:
        raise ValueError(f"queries.parquet is missing required columns: {missing}")

    out = queries.copy().reset_index(drop=True)
    out["cde_publicid"] = out["cde_publicid"].astype(str)
    out["cde_version"] = out["cde_version"].astype(str)

    if "cde_id" not in out.columns:
        out["cde_id"] = out["cde_publicid"] + "::" + out["cde_version"]
    else:
        out["cde_id"] = out["cde_id"].astype(str)

    # Preserve stable pair_id if present; otherwise create one.
    if "pair_id" not in out.columns:
        import hashlib

        def _sha1_hex(s: str) -> str:
            return hashlib.sha1(s.encode("utf-8")).hexdigest()

        out["pair_id"] = (out["query_id"].astype(str) + "::" + out["cde_id"].astype(str)).map(_sha1_hex)
    else:
        out["pair_id"] = out["pair_id"].astype(str)

    return out


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build pairs.parquet from queries.parquet")
    ap.add_argument("--queries-parquet", default=os.path.join("data", "processed", "queries.parquet"))
    ap.add_argument("--out-parquet", default=os.path.join("data", "processed", "pairs.parquet"))

    args = ap.parse_args(list(argv) if argv is not None else None)

    q = pd.read_parquet(args.queries_parquet)
    pairs = build_pairs(q)

    _ensure_dir_for_file(args.out_parquet)
    pairs.to_parquet(args.out_parquet, index=False)
    print("Wrote:", args.out_parquet, "| rows:", len(pairs))


if __name__ == "__main__":
    main()
