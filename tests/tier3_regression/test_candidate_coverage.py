"""Table S5 — Stage-1 candidate-pool gold coverage.

The manuscript's claim is that combining the two retrievers raises gold
coverage above either alone on every dataset. That is a statement about
candidate *generation*, and it is the precondition for everything the reranker
does: no reranker can recover a gold CDE that never entered the pool.

Offline, the committed fixture is checked against the printed table. With
``DEMAP_ARTIFACT_ROOT`` set, the whole thing is recomputed from the frozen
feature table and must reproduce the fixture exactly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from demap_repro.pool.coverage import (
    DISPLAY,
    EVAL_SPLITS,
    REQUIRED_COLUMNS,
    compute_coverage,
    compute_gains,
)

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: Table S5 exactly as the manuscript prints it, at 3 dp.
MANUSCRIPT_TABLE_S5 = {
    "Test":    {"n": 3959, "ftmpnet": "0.958", "fuzzy": "0.795", "union": "0.986"},
    "CCTG":    {"n": 1097, "ftmpnet": "0.841", "fuzzy": "0.807", "union": "0.959"},
    "OID ALT": {"n": 1766, "ftmpnet": "0.555", "fuzzy": "0.710", "union": "0.865"},
    "CDASH":   {"n": 324,  "ftmpnet": "0.923", "fuzzy": "0.861", "union": "0.969"},
    "GDC":     {"n": 72,   "ftmpnet": "0.917", "fuzzy": "0.972", "union": "0.986"},
    "CIMAC":   {"n": 131,  "ftmpnet": "0.672", "fuzzy": "0.809", "union": "0.870"},
}


@pytest.fixture(scope="module")
def coverage():
    return pd.read_csv(FIXTURES / "candidate_coverage_by_dataset.csv").set_index("dataset")


@pytest.fixture(scope="module")
def gains():
    return pd.read_csv(FIXTURES / "candidate_coverage_gains.csv").set_index("dataset")


# --------------------------------------------------------------------------
# the fixture is what the manuscript prints
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", list(MANUSCRIPT_TABLE_S5))
def test_fixture_matches_the_manuscript(coverage, dataset):
    want = MANUSCRIPT_TABLE_S5[dataset]
    row = coverage.loc[dataset]
    assert int(row["n_queries"]) == want["n"]
    for col, key in (("ftmpnet20_prop", "ftmpnet"), ("fuzzy10_prop", "fuzzy"),
                     ("union_prop", "union")):
        assert f"{float(row[col]):.3f}" == want[key], f"{dataset}/{col}"


def test_denominators_are_the_paper_population(coverage):
    """The same six query counts Table 1 and Table 4 use."""
    from demap_repro.evaluation.canonical_datasets import PAPER_EVAL_QUERY_COUNTS

    for split, n in PAPER_EVAL_QUERY_COUNTS.items():
        assert int(coverage.loc[DISPLAY[split], "n_queries"]) == n, split


# --------------------------------------------------------------------------
# the claim the section makes
# --------------------------------------------------------------------------

def test_union_beats_both_arms_on_every_dataset(coverage):
    """S5.6's claim: 'their union improved on the better individual retriever
    in every dataset, reaching 0.865-0.986'."""
    for dataset, row in coverage.iterrows():
        assert row["union_n"] >= row["ftmpnet20_n"], dataset
        assert row["union_n"] >= row["fuzzy10_n"], dataset
        assert row["union_n"] > max(row["ftmpnet20_n"], row["fuzzy10_n"]), dataset
    assert round(coverage["union_prop"].min(), 3) == 0.865
    assert round(coverage["union_prop"].max(), 3) == 0.986


def test_neither_arm_dominates(coverage):
    """The complementarity argument: which arm is stronger flips by dataset.
    On OID ALT the lexical arm alone covers more gold than the bi-encoder; on
    Test the ordering reverses."""
    assert coverage.loc["OID ALT", "fuzzy10_n"] > coverage.loc["OID ALT", "ftmpnet20_n"]
    assert coverage.loc["Test", "ftmpnet20_n"] > coverage.loc["Test", "fuzzy10_n"]


def test_gains_are_internally_consistent(coverage, gains):
    """Inclusion-exclusion: |A| + |B| - |A OR B| queries are covered by both."""
    for dataset, row in coverage.iterrows():
        g = gains.loc[dataset]
        assert g["both_arms_n"] == row["ftmpnet20_n"] + row["fuzzy10_n"] - row["union_n"]
        assert g["union_minus_ftmpnet_n"] == row["union_n"] - row["ftmpnet20_n"]
        assert g["union_minus_fuzzy_n"] == row["union_n"] - row["fuzzy10_n"]


# --------------------------------------------------------------------------
# the arm definition, which is easy to get wrong
# --------------------------------------------------------------------------

def test_the_lexical_arm_is_the_or_of_both_tiers():
    """The fixed top-10 CDE Match-Fuzzy list is decomposed at build time into a
    clone tier and a fuzzy-fallback tier. Using either flag alone understates
    the arm, so coverage must OR them.

    Built so ``in_cdematch_topk`` alone would find nothing: the gold row is
    flagged only into the fallback tier.
    """
    features = pd.DataFrame({
        "split": ["test"] * 3,
        "query_id": ["q1", "q1", "q2"],
        "is_label": [True, False, True],
        "in_biencoder_topk": [False, True, False],
        "in_cdematch_topk": [False, True, False],
        "in_keyword_topk": [True, False, False],
    })
    cov = compute_coverage(features, splits=("test",)).iloc[0]
    assert cov["n_queries"] == 2
    assert cov["fuzzy10_n"] == 1, "the OR of both tiers must find the q1 gold"
    assert cov["ftmpnet20_n"] == 0
    assert cov["union_n"] == 2, "union counts every gold row in the pool"


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        compute_coverage(pd.DataFrame({"split": ["test"], "query_id": ["q"]}))


# --------------------------------------------------------------------------
# recomputation from the frozen pool
# --------------------------------------------------------------------------

@pytest.mark.needs_artifacts
@pytest.mark.slow
def test_recomputes_from_the_frozen_feature_table(artifact_root, coverage, gains):
    """The fixture must be exactly what the shipped pool yields."""
    table = (artifact_root / "artifacts/final_reranker/hgbc_features_v2_eligible"
             / "feature_table_fixedk30_crossenc_v2.parquet")
    if not table.is_file():
        pytest.skip(f"frozen feature table unavailable: {table}")

    features = pd.read_parquet(table, columns=list(REQUIRED_COLUMNS))
    got = compute_coverage(features).set_index("dataset")
    got_gains = compute_gains(compute_coverage(features)).set_index("dataset")

    for dataset in coverage.index:
        for col in ("n_queries", "ftmpnet20_n", "fuzzy10_n", "union_n"):
            assert int(got.loc[dataset, col]) == int(coverage.loc[dataset, col]), (dataset, col)
        for col in ("ftmpnet20_prop", "fuzzy10_prop", "union_prop"):
            assert abs(got.loc[dataset, col] - coverage.loc[dataset, col]) < 5e-5, (dataset, col)
        for col in ("both_arms_n", "union_minus_stronger_n"):
            assert int(got_gains.loc[dataset, col]) == int(gains.loc[dataset, col]), (dataset, col)


@pytest.mark.needs_artifacts
@pytest.mark.slow
def test_union_coverage_equals_the_shipped_models_pool(artifact_root, coverage):
    """S5.6 and S5.7 must rest on one identical pool.

    The union coverage is the shipped HGBC's own
    ``n_queries_with_gold_in_candidates``, so the coverage analysis and the
    ablation are demonstrably measuring the same candidate set.
    """
    shipped = (artifact_root / "artifacts/final_reranker/hgbc_reranker_v2_eligible"
               / "with_ce_noprov/hgbc_eval_by_split.csv")
    if not shipped.is_file():
        pytest.skip(f"shipped evaluation unavailable: {shipped}")

    ev = pd.read_csv(shipped)
    ev = ev[ev["method"] == "hgbc"].set_index("split")
    if "n_queries_with_gold_in_candidates" not in ev.columns:
        pytest.skip("shipped evaluation carries no pool-coverage column")

    for split in EVAL_SPLITS:
        assert int(ev.loc[split, "n_queries_with_gold_in_candidates"]) == \
               int(coverage.loc[DISPLAY[split], "union_n"]), split
