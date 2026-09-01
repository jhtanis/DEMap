"""Table S3 — permissible-value overlap, and its two computations.

The manuscript prints 39,391 / 24,543 / 3,074 / 8,816. That is the frozen
2026-04-23 diagnostics run, and it is what this repository now targets.

Two reviews reached different answers, and both were internally sound:

* The 2026-04-23 run computed the table over a ``pairs.parquet`` build of
  69,844 rows. That file was overwritten on 2026-05-13 by the 69,102-row
  paper-era build - the one every model stage consumed and the one 3.1 and
  S1.3 quote. Recomputing the table over the surviving build gives
  38,964 / 24,158 / 3,040 / 8,815, which this repository asserted from
  2026-08-08 to 2026-09-01.
* The manuscript revision reviewed the same discrepancy against the retained
  artifacts - ``artifacts/tables/table04_pv_overlap_query_vs_cde.csv``, and two
  independent siblings of the same run - and kept 39,391. 38,964 was never
  written to an artifact; it existed only as a recomputation, which is why that
  review could not find a source for it.

The author's decision is 39,391, and this file follows it. The numerical
question is closed; do not reopen it.

What follows from that decision is a provenance fact worth stating plainly:
**Table S3 is not regenerable from the inputs this repository can reach.** The
69,844-row build no longer exists, and the 2026-04-23 per-pair records were not
retained in a form this grouping can consume. So Table S3 sits in the same
class as the official NCI CDE Match column of Table 4 - a frozen historical
result, reproduced from a committed fixture rather than recomputed.

Both computations are kept as fixtures. The grouping code itself never drifted;
only the input it was run over differs, which is exactly what
``test_the_grouping_code_reproduces_the_69102_recomputation`` demonstrates.

Precision note. The manuscript fixture's ``overall`` row carries full precision,
taken from ``pv_frozen_diagnostics.json`` of the same 2026-04-23 run; its four
per-group rows are only available at 4 dp. This matters for one cell:
``query_shorter_than_cde_rate`` stored as 0.3015 is exactly on the boundary for
the 1-dp percentage the manuscript prints, and rounding it twice yields 30.2%
where the manuscript prints 30.1%. The full-precision 0.30146... rounds once,
unambiguously, to 30.1%. Every other cell rounds the same either way. This is
the same double-rounding failure that put 0.769 and 0.175 in Table 4.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from demap_repro.reporting.dataset_tables_pv_overlap import (
    MIN_GROUP_ROWS,
    REPORTED_GROUPS,
    build_pv_overlap_table,
    parse_items,
    set_jaccard,
)

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# Table S3 as the manuscript prints it: the frozen 2026-04-23 run.
PAPER_TABLE_S3 = {
    "overall":             {"rows": 39391, "block_pct": 0.4, "shorter_pct": 30.1, "jaccard": 0.168},
    "ENUM":                {"rows": 24543, "block_pct": 0.4, "shorter_pct": 45.2, "jaccard": 0.153},
    "BINARY_WITH_UNKNOWN": {"rows": 3074,  "block_pct": 0.1, "shorter_pct": 0.0,  "jaccard": 0.333},
    "BINARY_WITH_NA":      {"rows": 8816,  "block_pct": 0.0, "shorter_pct": 0.0,  "jaccard": 0.123},
}

# The same table recomputed over the surviving 69,102-row pairs.parquet. Not a
# parity target for the manuscript; retained as provenance for the divergence.
RECOMPUTED_ROWS = {"overall": 38964, "ENUM": 24158,
                   "BINARY_WITH_UNKNOWN": 3040, "BINARY_WITH_NA": 8815}


@pytest.fixture(scope="module")
def manuscript():
    """The frozen run the manuscript prints."""
    return pd.read_csv(FIXTURES / "table_s3_pv_overlap_manuscript.csv").set_index("group")


@pytest.fixture(scope="module")
def recomputed():
    """The same table over the surviving 69,102-row benchmark."""
    return pd.read_csv(FIXTURES / "table_s3_pv_overlap_69102_recomputed.csv").set_index("group")


# --------------------------------------------------------------------------
# the fixture the manuscript prints
# --------------------------------------------------------------------------

@pytest.mark.parametrize("group", list(PAPER_TABLE_S3))
def test_fixture_values_match_the_manuscript(manuscript, group):
    want = PAPER_TABLE_S3[group]
    row = manuscript.loc[group]
    assert int(row["n_rows_both_present"]) == want["rows"]
    assert round(100 * row["normalized_block_match_rate"], 1) == want["block_pct"]
    assert round(100 * row["query_shorter_than_cde_rate"], 1) == want["shorter_pct"]
    assert round(row["mean_jaccard_overlap"], 3) == want["jaccard"]


def test_prose_denominator_is_the_overall_row(manuscript):
    """S3.5 reads 'over the 39,391 rows in which both sides carry a
    permissible-value summary'. It must be the same number as the table."""
    assert int(manuscript.loc["overall", "n_rows_both_present"]) == 39391


def test_table_shows_pv_evidence_is_partial_not_duplicated(manuscript):
    """The point of the table: whole-block agreement is rare and item overlap is
    low, so query-side PV is not a copy of the gold CDE's values."""
    assert manuscript.loc["overall", "normalized_block_match_rate"] < 0.01
    assert manuscript.loc["overall", "mean_jaccard_overlap"] < 0.25


# --------------------------------------------------------------------------
# the divergence between the two computations
# --------------------------------------------------------------------------

def test_the_two_computations_are_genuinely_different(manuscript, recomputed):
    """Neither fixture is a copy of the other. If they ever coincided, one of
    the two would be mislabelled."""
    for group, other in RECOMPUTED_ROWS.items():
        assert int(recomputed.loc[group, "n_rows_both_present"]) == other
        assert int(manuscript.loc[group, "n_rows_both_present"]) != other


def test_only_one_rate_cell_moved(manuscript, recomputed):
    """11 of 12 rate cells round identically; only ENUM query-shorter differs,
    45.2% vs 45.3%. The divergence is a denominator difference, not a change in
    what the table says - which is why it does not affect the interpretation of
    the PV-overlap analysis."""
    moved = []
    for group in PAPER_TABLE_S3:
        for col, places in (("normalized_block_match_rate", 3),
                            ("query_shorter_than_cde_rate", 3),
                            ("mean_jaccard_overlap", 3)):
            a = round(float(manuscript.loc[group, col]), places)
            b = round(float(recomputed.loc[group, col]), places)
            if a != b:
                moved.append((group, col, a, b))
    assert len(moved) == 1, moved
    group, col, _a, _b = moved[0]
    assert group == "ENUM" and col == "query_shorter_than_cde_rate"


def test_the_difference_does_not_change_the_conclusion(manuscript, recomputed):
    """Both computations support the same statement: whole-block agreement is
    rare and item overlap is low."""
    for table in (manuscript, recomputed):
        assert table.loc["overall", "normalized_block_match_rate"] < 0.01
        assert table.loc["overall", "mean_jaccard_overlap"] < 0.25


# --------------------------------------------------------------------------
# the grouping logic
# --------------------------------------------------------------------------

def test_reported_groups_are_the_four_the_paper_prints():
    assert REPORTED_GROUPS == ("overall", "ENUM", "BINARY_WITH_UNKNOWN", "BINARY_WITH_NA")


@pytest.mark.parametrize("fixture_name", ["manuscript", "recomputed"])
def test_computation_carries_six_groups_but_the_paper_prints_four(request, fixture_name):
    """BINARY_WITH_UNKNOWN_NA and BINARY clear the 100-row threshold but are
    omitted from the table. True of both computations."""
    table = request.getfixturevalue(fixture_name)
    assert len(table) == 6
    assert set(table.index) - set(REPORTED_GROUPS) == {"BINARY_WITH_UNKNOWN_NA", "BINARY"}


def test_small_pv_types_are_excluded_by_the_row_threshold(tmp_path):
    records = pd.DataFrame({
        "query_source": ["REF"] * 12,
        "both_present": [True] * 12,
        "cde_pv_type": ["ENUM"] * 10 + ["RARE"] * 2,
        "normalized_block_match": [False] * 12,
        "normalized_item_match": [False] * 12,
        "query_shorter_than_cde": [True] * 12,
        "query_pv_items_cf": ["['a']"] * 12,
        "cde_pv_items_cf": ["['a', 'b']"] * 12,
    })
    csv = tmp_path / "records.csv"
    records.to_csv(csv, index=False)
    table = build_pv_overlap_table(csv, min_group_rows=5)
    assert set(table["group"]) == {"overall", "ENUM"}, "RARE has 2 rows, below the threshold"


def test_rows_without_both_sides_are_excluded(tmp_path):
    """A pair with PV on only one side cannot be an overlap observation."""
    records = pd.DataFrame({
        "query_source": ["REF"] * 4,
        "both_present": [True, True, False, False],
        "cde_pv_type": ["ENUM"] * 4,
        "normalized_block_match": [True, False, True, True],
        "normalized_item_match": [True, False, True, True],
        "query_shorter_than_cde": [True, False, True, True],
        "query_pv_items_cf": ["['a']"] * 4,
        "cde_pv_items_cf": ["['a']"] * 4,
    })
    csv = tmp_path / "records.csv"
    records.to_csv(csv, index=False)
    table = build_pv_overlap_table(csv, min_group_rows=1).set_index("group")
    assert int(table.loc["overall", "n_rows_both_present"]) == 2
    assert table.loc["overall", "normalized_block_match_rate"] == 0.5


def test_string_booleans_survive_the_csv_round_trip(tmp_path):
    """The diagnostics are CSV, so booleans arrive as strings. A naive
    astype(bool) would make 'False' truthy and report every row as a match."""
    records = pd.DataFrame({
        "query_source": ["REF"] * 2,
        "both_present": ["True", "True"],
        "cde_pv_type": ["ENUM"] * 2,
        "normalized_block_match": ["False", "False"],
        "normalized_item_match": ["False", "False"],
        "query_shorter_than_cde": ["False", "True"],
        "query_pv_items_cf": ["['a']"] * 2,
        "cde_pv_items_cf": ["['a', 'b']"] * 2,
    })
    csv = tmp_path / "records.csv"
    records.to_csv(csv, index=False)
    table = build_pv_overlap_table(csv, min_group_rows=1).set_index("group")
    assert table.loc["overall", "normalized_block_match_rate"] == 0.0
    assert table.loc["overall", "query_shorter_than_cde_rate"] == 0.5


@pytest.mark.parametrize("raw,expected", [
    ("['yes', 'no']", ["yes", "no"]),
    ("[]", []),
    ("", []),
    ("nan", []),
    (None, []),
])
def test_item_parsing(raw, expected):
    assert parse_items(raw) == expected


def test_jaccard_overlap():
    assert set_jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert set_jaccard(["a"], ["a", "b"]) == 0.5
    assert set_jaccard(["a"], ["b"]) == 0.0


def test_default_row_threshold_is_100():
    assert MIN_GROUP_ROWS == 100


# --------------------------------------------------------------------------
# full chain
# --------------------------------------------------------------------------

@pytest.mark.needs_artifacts
@pytest.mark.slow
def test_the_grouping_code_reproduces_the_69102_recomputation(artifact_root, tmp_path, recomputed):
    """pairs.parquet -> pv_frozen_diagnostics -> this grouping.

    This is the provenance claim, not a manuscript check. Running the current
    code over the benchmark this repository can actually reach - the surviving
    69,102-row pairs.parquet - yields 38,964, not the 39,391 the manuscript
    prints.

    That is expected and is the whole point of keeping both fixtures. The
    manuscript's table came from the 2026-04-23 run over a 69,844-row build
    that was later overwritten, so it cannot be regenerated here. What this
    test establishes is that the *grouping code* never drifted: given an input,
    it still produces the table that input implies.

    A failure here means the PV grouping logic changed, which would be a real
    defect. It does not mean the manuscript is wrong.
    """
    from demap_repro.data import pv_frozen_diagnostics

    pairs = artifact_root / "data/processed/pairs.parquet"
    catalog = artifact_root / "data/processed/cde_master_enriched.parquet"
    if not (pairs.exists() and catalog.exists()):
        pytest.skip("frozen benchmark inputs not present")

    out = tmp_path / "diag"
    pv_frozen_diagnostics.main([
        "--pairs-parquet", str(pairs),
        "--cde-parquet", str(catalog),
        "--out-dir", str(out),
    ])
    table = build_pv_overlap_table(out / "query_cde_pv_pair_records.csv").set_index("group")

    for group in REPORTED_GROUPS:
        assert int(table.loc[group, "n_rows_both_present"]) == RECOMPUTED_ROWS[group], group
        for col in ("normalized_block_match_rate", "query_shorter_than_cde_rate",
                    "mean_jaccard_overlap"):
            assert round(float(table.loc[group, col]), 6) == \
                   round(float(recomputed.loc[group, col]), 6), (group, col)
