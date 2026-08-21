"""The minimal revalidation of the paper's core results, as executable tests.

These are the Level A and Level B checks defined in
``docs/handoff/2026-08-21-paper-repo-bug-backport-and-core-revalidation.md``.
They exist because the rest of the tier-3 suite checks *committed fixtures*, and
a fixture faithfully preserves whatever produced it — including, in principle, a
number produced by buggy code. These tests instead recompute from upstream:

**Level A** rebuilds the production catalog's eligibility filter from the raw
79,827-record export and checks the evaluation population, gold reachability and
the exact-match allowance against the manuscript.

**Level B** reloads the frozen HGBC and re-scores the frozen 117-feature table,
so every Table 4 number for the final reranker is recomputed from the model
rather than read from ``final_table4.csv``.

Both skip without ``DEMAP_DATA_ROOT`` (Level A) or ``DEMAP_ARTIFACT_ROOT``
(Level B), so the suite stays green on a clean clone. Level B is CPU-only and
takes seconds: no retraining and no GPU are needed to check these claims.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: Section 2.1 / Table 4 query-level denominators.
PAPER_QUERY_COUNTS = {
    "test": 3959, "cctg": 1097, "oid_alt": 1766,
    "cdash": 324, "gdc_combined": 72, "cimac_v2": 131,
}

#: Table 4 Recall@5 for the final reranker, to the manuscript's 3 decimals.
PAPER_FINAL_RERANKER_RECALL5 = {
    "test": 0.971, "cctg": 0.909, "oid_alt": 0.832,
    "cdash": 0.920, "gdc_combined": 0.972, "cimac_v2": 0.802,
}

DISPLAY = {"test": "Test", "cctg": "CCTG", "oid_alt": "OID ALT",
           "cdash": "CDASH", "gdc_combined": "GDC", "cimac_v2": "CIMAC"}

CATALOG_DIR = "data/processed/cadsr_xml_2026-06-18"
RAW_CATALOG = f"{CATALOG_DIR}/cde_master_enriched_eval.parquet"
PRODUCTION_CATALOG = f"{CATALOG_DIR}/cde_master_enriched_eval_production_cde_match.parquet"
HGBC_DIR = "artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov"
FEATURE_TABLE = ("artifacts/final_reranker/hgbc_features_v2_eligible/"
                 "feature_table_fixedk30_crossenc_v2.parquet")


def _require(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    return path


# ==========================================================================
# Level A — structural, offline
# ==========================================================================

@pytest.fixture(scope="module")
def evaluation_sets(data_root):
    directory = data_root / "data/processed/eval_canonical"
    _require(directory)
    return {name: pd.read_parquet(directory / f"{name}.parquet")
            for name in PAPER_QUERY_COUNTS}


@pytest.fixture(scope="module")
def production_catalog(data_root):
    return pd.read_parquet(_require(data_root / PRODUCTION_CATALOG))


def test_A1_canonical_evaluation_query_counts(evaluation_sets):
    """Section 2.1: 3,959 / 1,097 / 1,766 / 324 / 72 / 131 distinct queries."""
    actual = {name: int(df["query_id"].nunique()) for name, df in evaluation_sets.items()}
    assert actual == PAPER_QUERY_COUNTS


def test_A2_production_catalog_identity(production_catalog):
    """S1: 'The June production catalog contains 62,976 records representing
    62,858 unique CDE public identifiers.'"""
    assert len(production_catalog) == 62976
    assert int(production_catalog["cde_publicid"].nunique()) == 62858


def test_A3_eligibility_filter_reproduces_the_catalog(data_root):
    """S1: 'created from a 79,827-record export by removing 16,846 retired records
    and 21 records from administrative TEST or Training contexts; 16 records met
    both exclusion criteria.'

    Recomputed from the raw export with this repository's own filter, not read
    off the shipped catalog.
    """
    from demap_repro.data.eligibility import excluded_context_mask, retired_mask

    raw = pd.read_parquet(_require(data_root / RAW_CATALOG))
    assert len(raw) == 79827

    retired = retired_mask(raw["workflow_status"])
    excluded_context = excluded_context_mask(raw["context_name"])

    assert int(retired.sum()) == 16846
    assert int(excluded_context.sum()) == 21
    assert int((retired & excluded_context).sum()) == 16
    assert int((~(retired | excluded_context)).sum()) == 62976


def test_A4_every_gold_cde_is_reachable(evaluation_sets, production_catalog):
    """S1: evaluation sets are filtered to queries whose gold CDE is in the
    catalog, so a query can never be an unreachable-by-construction miss."""
    catalog_ids = set(production_catalog["cde_id"].astype(str))
    unreachable = {name: sorted(set(df["cde_id"].astype(str)) - catalog_ids)
                   for name, df in evaluation_sets.items()}
    assert all(not ids for ids in unreachable.values()), {
        k: v[:5] for k, v in unreachable.items() if v}


def test_A5_allowance_routing_matches_the_manuscript():
    """Section 2.3: 0.70 on the four caDSR-derived sets, 1.00 on GDC and CIMAC."""
    from demap_repro.reranker.split_routing import allowance_for_split

    expected = {"test": 0.70, "cctg": 0.70, "oid_alt": 0.70, "cdash": 0.70,
                "gdc_combined": 1.0, "cimac_v2": 1.0}
    assert {name: allowance_for_split(name) for name in expected} == expected


def test_A6_exact_match_mask_is_nested_and_deterministic(evaluation_sets):
    """S6.2: 'Queries were assigned to nested allowance subsets using a SHA-1 hash
    of the query identifier with seed 42.'"""
    from demap_repro.lexical.mask import allowed_set

    query_ids = list(evaluation_sets["test"]["query_id"].astype(str).unique())
    rates = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0]
    admitted = {rate: allowed_set(query_ids, rate) for rate in rates}

    for lower, higher in zip(rates, rates[1:]):
        assert admitted[lower] <= admitted[higher], (lower, higher)
    assert admitted[0.0] == set()
    assert admitted[1.0] == set(query_ids)
    # Same inputs, same answer — no RNG state, no PYTHONHASHSEED dependence.
    assert allowed_set(query_ids, 0.7) == admitted[0.7]
    assert abs(len(admitted[0.7]) / len(query_ids) - 0.70) < 0.02


def test_A7_pool_and_feature_contract():
    """Figure 6 / S5.4: FT-MPNet top 20 + CDE Match-Fuzzy top 10, deduplicated by
    public identifier, K = 30; 117 features."""
    from demap_repro.pool.select_k import (
        PAPER_BRANCH, PAPER_KEYWORD_DEPTH, PAPER_SELECTED_K, PAPER_SELECTION_SPLIT)

    assert PAPER_SELECTED_K == 30
    assert PAPER_KEYWORD_DEPTH == 10
    assert PAPER_BRANCH == "kwfuzzy"
    assert PAPER_SELECTION_SPLIT == "val_train"

    features = json.loads((FIXTURES / "hgbc_feature_set.json").read_text())
    assert features["n_features_used"] == 117


# ==========================================================================
# Level B — recompute the final reranker from the frozen model
# ==========================================================================

@pytest.fixture(scope="module")
def recomputed_reranker_metrics(artifact_root):
    """Re-score the frozen 117-feature table with the frozen HGBC.

    This is the whole of Level B: no retraining, no GPU, a few seconds of CPU.
    """
    joblib = pytest.importorskip("joblib")
    from demap_repro.reranker.features.categorical_vocab import CategoricalVocab
    from demap_repro.reranker.train import _deployment_metrics, prepare_features

    hgbc_dir = _require(artifact_root / HGBC_DIR)
    features = pd.read_parquet(_require(artifact_root / FEATURE_TABLE))
    feature_set = json.loads((hgbc_dir / "feature_set.json").read_text())

    vocab = CategoricalVocab.load(hgbc_dir / "categorical_vocab.json")
    matrix, _ = prepare_features(features, categorical_vocab=vocab)
    model = joblib.load(hgbc_dir / "hgbc_model.joblib")

    features = features.copy()
    features["hgbc_score"] = model.predict_proba(
        matrix[feature_set["included_features"]])[:, 1]

    return {name: _deployment_metrics(features[features["split"] == name], "hgbc_score")
            for name in PAPER_QUERY_COUNTS}


def test_B0_the_frozen_model_uses_the_117_feature_contract(artifact_root):
    joblib = pytest.importorskip("joblib")

    hgbc_dir = _require(artifact_root / HGBC_DIR)
    model = joblib.load(hgbc_dir / "hgbc_model.joblib")
    assert model.n_features_in_ == 117


def test_B1_denominators_are_the_paper_query_counts(recomputed_reranker_metrics):
    """A recomputation against the wrong evaluation population would show up
    here before it showed up in a recall value."""
    actual = {name: int(metrics["n_queries"])
              for name, metrics in recomputed_reranker_metrics.items()}
    assert actual == PAPER_QUERY_COUNTS


@pytest.mark.parametrize("dataset", list(PAPER_QUERY_COUNTS))
def test_B2_final_reranker_recall5_matches_table4(recomputed_reranker_metrics, dataset):
    """Table 4, recomputed from the model rather than read from the fixture."""
    recall5 = recomputed_reranker_metrics[dataset]["recall@5"]
    assert round(recall5, 3) == PAPER_FINAL_RERANKER_RECALL5[dataset], DISPLAY[dataset]


def test_B3_recomputation_matches_the_frozen_eval_table(recomputed_reranker_metrics,
                                                        fixtures_dir):
    """Level-1 tolerance from ``manifests/expected_results.json`` is 5e-4. This is
    deterministic inference over a frozen model and a frozen feature table, so the
    agreement should be far tighter than that; anything looser means the
    recomputation is not doing what the committed artifact did.
    """
    frozen = pd.read_csv(fixtures_dir / "hgbc_eval_by_split.csv").set_index("split")
    for dataset, metrics in recomputed_reranker_metrics.items():
        for metric in ("recall@1", "recall@5", "recall@10", "mrr@100"):
            assert metrics[metric] == pytest.approx(
                float(frozen.loc[dataset, metric]), abs=5e-4), (dataset, metric)


def test_B4_headline_abstract_claims(recomputed_reranker_metrics):
    """Abstract: 'Recall@5 of 0.971 internally and 0.802-0.972 externally'."""
    recall5 = {name: metrics["recall@5"]
               for name, metrics in recomputed_reranker_metrics.items()}
    assert round(recall5["test"], 3) == 0.971

    external = [value for name, value in recall5.items() if name != "test"]
    assert round(min(external), 3) == 0.802
    assert round(max(external), 3) == 0.972


def test_B5_gdc_is_seventy_of_seventytwo(recomputed_reranker_metrics):
    """3.8: 'compared with 70 of 72 for the final reranker'. An integer count, so
    it is asserted as one rather than as a rounded rate."""
    metrics = recomputed_reranker_metrics["gdc_combined"]
    total = int(metrics["n_queries"])
    assert total == 72
    assert round(metrics["recall@5"] * total) == 70
