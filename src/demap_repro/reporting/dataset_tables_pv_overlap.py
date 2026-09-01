#!/usr/bin/env python3
"""Table S3 — permissible-value overlap between the query side and the CDE side.

Second and final stage of the Table S3 chain. Stage one,
:mod:`demap_repro.data.pv_frozen_diagnostics`, walks the frozen benchmark pairs
and emits one diagnostic row per (query, gold CDE) pair. This module groups those
rows into the four figures the manuscript prints: how many pairs carry a PV
summary on both sides, how often the two blocks match as normalized strings, how
often the query block is shorter, and the mean item-level Jaccard overlap.

What the table is for
---------------------
Both sides are rendered from the same upstream permissible values, so the obvious
worry is that the query is handed a copy of the answer. Table S3 is the evidence
that it is not: whole-block agreement is 0.4%, and mean item overlap is 0.168.
Query-side permissible values contribute partial, not duplicated, evidence.

Grouping
--------
An ``overall`` row plus one row per ``cde_pv_type`` with at least 100 both-present
rows. The computation yields six such groups; the paper prints four
(``BINARY_WITH_UNKNOWN_NA`` and ``BINARY`` are omitted from the table).

Which input Table S3 came from
------------------------------
The manuscript prints 39,391 / 24,543 / 3,074 / 8,816: the frozen 2026-04-23 run
over a ``pairs.parquet`` build of 69,844 rows. That build was overwritten on
2026-05-13 by the paper-era 69,102-row build — the one every model stage
consumed — so **this code cannot regenerate the printed table**. Run over the
surviving benchmark it gives 38,964 / 24,158 / 3,040 / 8,815, with ENUM
query-shorter at 45.3% rather than 45.2%; eleven of the twelve rate cells are
unaffected either way.

Both are kept as fixtures, and the difference changes neither the table's
message nor any downstream result. This was never PV-code drift: the generation
code regenerates the current PV blocks with 100% row-level parity; only the
*input* to the frozen diagnostic was replaced. See
``manifests/expected_results.json`` under ``A_pv_overlap``.

Migrated from cell 13 of
``notebooks/07_dataset_statistics_tables_and_figure.ipynb``, the confirmed
producer. The grouping logic is unchanged.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from demap_repro.utils.paths import data_root

__all__ = [
    "MIN_GROUP_ROWS", "REPORTED_GROUPS", "REQUIRED_COLUMNS",
    "parse_items", "set_jaccard", "build_pv_overlap_table",
]

#: A cde_pv_type needs this many both-present rows to get its own row.
MIN_GROUP_ROWS = 100

#: The four groups v21 prints, in table order. The computation produces six.
REPORTED_GROUPS = ("overall", "ENUM", "BINARY_WITH_UNKNOWN", "BINARY_WITH_NA")

REQUIRED_COLUMNS = (
    "both_present", "cde_pv_type", "normalized_block_match",
    "normalized_item_match", "query_shorter_than_cde",
    "query_pv_items_cf", "cde_pv_items_cf",
)

DEFAULT_RECORDS = ("artifacts/summaries/pv_frozen_diagnostics/"
                   "query_cde_pv_pair_records.csv")


def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a CSV round-tripped boolean column back to bool.

    The diagnostics are written as CSV, so ``True`` may arrive as a string, and a
    naive ``astype(bool)`` would make the string ``"False"`` truthy — silently
    reporting every row as a match.
    """
    if series.dtype == bool:
        return series
    return (series.astype(str).str.strip().str.lower()
            .isin({"true", "t", "1", "yes"}))


def parse_items(value) -> List[str]:
    """Parse a repr'd list of PV items from the diagnostics CSV."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def set_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Item-level Jaccard overlap of two PV item lists."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return np.nan
    union = sa | sb
    return len(sa & sb) / len(union) if union else np.nan


def _summarize(df: pd.DataFrame, group_name: str) -> Dict[str, object]:
    sub = df.loc[df["both_present"]]
    n = len(sub)
    return {
        "group": group_name,
        "n_rows_both_present": int(n),
        "normalized_block_match_rate": float(sub["normalized_block_match"].mean()) if n else np.nan,
        "normalized_set_match_rate": float(sub["normalized_set_match"].mean()) if n else np.nan,
        "query_shorter_than_cde_rate": float(sub["query_shorter_than_cde"].mean()) if n else np.nan,
        "mean_jaccard_overlap": float(sub["jaccard_overlap"].mean()) if n else np.nan,
    }


def build_pv_overlap_table(records_csv: Path,
                           min_group_rows: int = MIN_GROUP_ROWS) -> pd.DataFrame:
    """Group the per-pair PV diagnostics into the Table S3 rows."""
    records = pd.read_csv(records_csv, usecols=list(REQUIRED_COLUMNS) + ["query_source"])

    for col in ("both_present", "normalized_block_match",
                "normalized_item_match", "query_shorter_than_cde"):
        records[col] = _to_bool(records[col])

    query_items = records["query_pv_items_cf"].map(parse_items)
    cde_items = records["cde_pv_items_cf"].map(parse_items)
    both = records["both_present"]

    records["normalized_set_match"] = [
        set(q) == set(c) if flag else False
        for q, c, flag in zip(query_items, cde_items, both)
    ]
    records["jaccard_overlap"] = [
        set_jaccard(q, c) if flag else np.nan
        for q, c, flag in zip(query_items, cde_items, both)
    ]

    counts = records.loc[both, "cde_pv_type"].fillna("").value_counts()
    major_types = [t for t, n in counts.items() if t and int(n) >= min_group_rows]

    rows = [_summarize(records, "overall")]
    for pv_type in major_types:
        rows.append(_summarize(records[records["cde_pv_type"] == pv_type], pv_type))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=None,
                    help=f"per-pair PV diagnostics CSV (default: <data root>/{DEFAULT_RECORDS})")
    ap.add_argument("--out", default=None, help="write the table here as CSV")
    ap.add_argument("--reported-only", action="store_true",
                    help="restrict to the four groups printed in v21")
    args = ap.parse_args(argv)

    records = Path(args.records) if args.records else data_root() / DEFAULT_RECORDS
    table = build_pv_overlap_table(records)
    if args.reported_only:
        table = (table[table["group"].isin(REPORTED_GROUPS)]
                 .set_index("group").reindex(REPORTED_GROUPS).reset_index())

    print(table.to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
