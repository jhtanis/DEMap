"""Table 4, Table S6, the final HGBC configuration, and the BM25 baseline.

These are the reported end-to-end results. Every value here is checked against
the manuscript, so a rebuild that shifts a headline number fails loudly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# 3.8 / Table 4: the six methods the paper compares.
TABLE4_METHODS = {
    "final_reranker", "python_cde_match_approx", "bm25",
    "ft_mpnet", "cde_match_fuzzy", "ce_pool_rerank",
}
TABLE4_DATASETS = {"Test", "CCTG", "OID ALT", "CDASH", "GDC", "CIMAC"}

# S5.1 BM25 Recall@5, quoted in Table 4.
PAPER_BM25_RECALL5 = {
    "test": 0.361, "cctg": 0.392, "oid_alt": 0.175,
    "cdash": 0.710, "gdc_combined": 0.542, "cimac_v2": 0.160,
}


@pytest.fixture(scope="module")
def table4():
    return pd.read_csv(FIXTURES / "final_table4.csv")


@pytest.fixture(scope="module")
def leakage():
    return pd.read_csv(FIXTURES / "table_s5_leakage.csv")


# --------------------------------------------------------------------------
# Table 4
# --------------------------------------------------------------------------

def test_table4_reports_only_the_six_paper_methods(table4):
    """The artifact also carries old_* pre-correction rows. They are provenance,
    not results, and must never reach the table."""
    reported = {m for m in table4["method"].unique() if not m.startswith("old_")}
    assert reported == TABLE4_METHODS
    assert any(m.startswith("old_") for m in table4["method"].unique()), (
        "the old_* rows should still exist in the artifact as provenance")


def test_table4_is_six_methods_by_six_datasets(table4):
    reported = table4[~table4["method"].str.startswith("old_")]
    assert set(reported["dataset"]) == TABLE4_DATASETS
    assert len(reported) == 36


def test_headline_internal_recall(table4):
    """Abstract: 'Recall@5 0.971 internally'."""
    row = table4[(table4["method"] == "final_reranker") & (table4["dataset"] == "Test")]
    assert round(float(row.iloc[0]["recall@5"]), 3) == 0.971


def test_headline_external_range(table4):
    """Abstract: '0.802-0.972 externally'."""
    ext = table4[(table4["method"] == "final_reranker")
                 & (table4["dataset"] != "Test")]
    assert round(ext["recall@5"].min(), 3) == 0.802
    assert round(ext["recall@5"].max(), 3) == 0.972


def test_final_reranker_leads_on_five_of_six_datasets(table4):
    """Abstract: 'highest Recall@5 on five of six datasets'."""
    reported = table4[~table4["method"].str.startswith("old_")]
    wins = 0
    for dataset, g in reported.groupby("dataset"):
        best = g.loc[g["recall@5"].idxmax(), "method"]
        if best == "final_reranker":
            wins += 1
    assert wins == 5


def test_ft_medcpt_reranks_the_same_pool(table4):
    """Table 4 caption: FT-MedCPT reranks the same ~30-candidate pool, so it must
    never beat the pool ceiling."""
    ce = table4[table4["method"] == "ce_pool_rerank"]
    assert len(ce) == 6
    assert ce["recall@5"].max() <= 0.99


# --------------------------------------------------------------------------
# Table S5 — leakage sensitivity
# --------------------------------------------------------------------------

def test_leakage_table_covers_three_methods_and_five_external_sets(leakage):
    assert set(leakage["method"]) == {"python_cde_match_approx", "bm25", "hgbc_corrected"}
    assert set(leakage["eval_set"]) == {"v1", "v2"}
    # Test is present but in-distribution by design; the five holdouts are the point.
    assert {"cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2"} <= set(leakage["dataset"])


def test_ft_mpnet_is_not_a_table_s6_method(leakage):
    assert "ft_mpnet" not in set(leakage["method"])


def test_test_split_is_unchanged_by_filtering(leakage):
    """Test is in-distribution by design and passes through the filter untouched."""
    test = leakage[leakage["dataset"] == "test"]
    for method, g in test.groupby("method"):
        v1 = g[g["eval_set"] == "v1"]["recall@5"].iloc[0]
        v2 = g[g["eval_set"] == "v2"]["recall@5"].iloc[0]
        assert v1 == v2, method


def test_cimac_leakage_deltas_match_section_3_8(leakage):
    """3.8: 'On CIMAC ... reduced Recall@5 from 0.802 to 0.779 for the final
    reranker and from 0.679 to 0.637 for the Python approximation'."""
    cimac = leakage[leakage["dataset"] == "cimac_v2"]

    def r5(method, eval_set):
        return round(float(cimac[(cimac["method"] == method)
                                 & (cimac["eval_set"] == eval_set)]["recall@5"].iloc[0]), 3)

    assert r5("hgbc_corrected", "v1") == 0.802
    assert r5("hgbc_corrected", "v2") == 0.779
    assert r5("python_cde_match_approx", "v1") == 0.679
    assert r5("python_cde_match_approx", "v2") == 0.637


def test_method_ordering_survives_the_leakage_filter(leakage):
    """3.8: 'without changing method order' — the claim that matters more than the
    individual deltas."""
    for dataset, g in leakage.groupby("dataset"):
        v1 = g[g["eval_set"] == "v1"].sort_values("recall@5", ascending=False)["method"].tolist()
        v2 = g[g["eval_set"] == "v2"].sort_values("recall@5", ascending=False)["method"].tolist()
        assert v1 == v2, dataset


def test_filtered_denominators_match_the_leakage_filter(leakage):
    expected = {"cctg": (1097, 1089), "oid_alt": (1766, 1759), "cdash": (324, 310),
                "gdc_combined": (72, 68), "cimac_v2": (131, 113), "test": (3959, 3959)}
    for dataset, (v1_n, v2_n) in expected.items():
        g = leakage[leakage["dataset"] == dataset]
        assert set(g[g["eval_set"] == "v1"]["n_queries"]) == {v1_n}, dataset
        assert set(g[g["eval_set"] == "v2"]["n_queries"]) == {v2_n}, dataset


# --------------------------------------------------------------------------
# the final HGBC
# --------------------------------------------------------------------------

def test_final_model_uses_117_features_without_provenance():
    """S5.5 and Figure 6. The five excluded features are the two final-rank
    columns, their log transform, and the two query-provenance categoricals."""
    fs = json.loads((FIXTURES / "hgbc_feature_set.json").read_text())
    assert fs["n_features_used"] == 117
    assert len(fs["included_features"]) == 117
    assert set(fs["excluded_features"]) == {
        "keyword_rank", "cdematch_rank", "log1p_cdematch_rank",
        "text_family", "text_query_source",
    }


def test_final_model_has_no_categorical_features():
    """S5.5: 'no normalization and no categorical features in the final model'.
    The two categoricals are exactly the provenance columns that were excluded."""
    fs = json.loads((FIXTURES / "hgbc_feature_set.json").read_text())
    assert fs["categorical_columns"] == []
    assert fs["categorical_feature_indices"] == []


def test_no_provenance_feature_reaches_the_model():
    fs = json.loads((FIXTURES / "hgbc_feature_set.json").read_text())
    for feature in fs["included_features"]:
        assert "family" not in feature
        assert "query_source" not in feature


def test_provenance_dropout_was_not_used():
    """The dropout experiment is explicitly out of scope; the shipped model was
    trained at rate 0."""
    fs = json.loads((FIXTURES / "hgbc_feature_set.json").read_text())
    assert fs["provenance_dropout_rate"] == 0.0
    assert fs["provenance_dropout_policy_file"] is None


def test_selected_hyperparameters_match_s55():
    """S5.5: 'selected 200 / 3 / 0.05 / 30'."""
    cfg = json.loads((FIXTURES / "hgbc_selected_config.json").read_text())
    assert cfg == {"max_iter": 200, "max_depth": 3,
                   "learning_rate": 0.05, "min_samples_leaf": 30}


def test_hyperparameter_grid_had_16_combinations():
    """S5.5: 'from 16 combinations (iterations {200,500} x depth {3,5} x
    lr {0.05,0.1} x min_samples_leaf {10,30})'."""
    grid = pd.read_csv(FIXTURES / "hgbc_grid_results.csv")
    assert len(grid) == 16
    assert set(grid["max_iter"]) == {200, 500}
    assert set(grid["max_depth"]) == {3, 5}
    assert set(grid["learning_rate"]) == {0.05, 0.1}
    assert set(grid["min_samples_leaf"]) == {10, 30}


def test_selected_config_is_the_grid_winner_on_val_dev():
    grid = pd.read_csv(FIXTURES / "hgbc_grid_results.csv")
    cfg = json.loads((FIXTURES / "hgbc_selected_config.json").read_text())
    metric = next(c for c in grid.columns if "recall" in c.lower() and "5" in c)
    best = grid.loc[grid[metric].idxmax()]
    for key, value in cfg.items():
        assert best[key] == value, key


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

def test_bm25_recall5_matches_table4():
    bm25 = pd.read_csv(FIXTURES / "bm25_canonical_metrics.csv").set_index("dataset")
    for dataset, expected in PAPER_BM25_RECALL5.items():
        assert round(float(bm25.loc[dataset, "recall@5"]), 3) == expected, dataset


def test_bm25_selection_ran_over_the_full_representation_grid():
    """S5.1: the representation was selected on Validation Dev over the same 4 x 10
    grid used for the bi-encoder screen."""
    sel = pd.read_csv(FIXTURES / "bm25_val_dev_selection.csv")
    assert len(sel) == 40
    assert sel["query_variant"].nunique() == 4
    assert sel["recipe"].nunique() == 10


def test_bm25_selected_representation_is_q3_by_sn_ln_pqt_pv():
    """S5.1: 'raw query + PV summary (Q3) x SN + LN + PQT + PV'.

    Selection runs over the reportable cells only. ``v1_v2a_v3_v5`` decodes to
    SHORT_NAME + LONG_NAME + PREFERRED_QUESTION_TEXT + PV_SUMMARY.
    """
    from demap_repro.text.recipes import RECIPE_FIELDS

    sel = pd.read_csv(FIXTURES / "bm25_val_dev_selection.csv")
    best = sel[sel["reportable"]].loc[sel[sel["reportable"]]["recall@5"].idxmax()]
    assert best["query_variant"] == "Q3"
    assert best["recipe"] == "v1_v2a_v3_v5"

    fields = [f for token in best["recipe"].split("_") for f in RECIPE_FIELDS[token]]
    assert fields == ["SHORT_NAME", "LONG_NAME", "PREFERRED_QUESTION_TEXT", "PV_SUMMARY"]


def test_bm25_quarantine_gate_governs_selection():
    """Half the grid is not reportable: only Q1 and Q3 exist on all six evaluation
    datasets, so cells built on Q2/Q4 cannot produce canonical output. Ignoring the
    gate would select a Q2 cell that ties on Validation Dev but cannot be reported.
    """
    sel = pd.read_csv(FIXTURES / "bm25_val_dev_selection.csv")
    assert sel["reportable"].sum() == 20
    assert (~sel["reportable"]).sum() == 20
    # Q1 and Q3 are the freezable variants; Q2 and Q4 are quarantined wholesale.
    assert set(sel[sel["reportable"]]["query_variant"]) == {"Q1", "Q3"}
    assert set(sel[~sel["reportable"]]["query_variant"]) == {"Q2", "Q4"}

    ungated_best = sel.loc[sel["recall@5"].idxmax()]
    assert ungated_best["query_variant"] == "Q2", (
        "if this stops being true the gate no longer changes the outcome and the "
        "test has lost its point")
    assert not ungated_best["reportable"]


def test_bm25_indexed_the_production_catalog():
    """S5.1: 'Retrieved from the same production catalog' — 62,976 CDEs."""
    sel = pd.read_csv(FIXTURES / "bm25_val_dev_selection.csv")
    assert set(sel["n_index_docs"]) == {62976}


# ---------------------------------------------------------------------------
# Table 4 display precision
# ---------------------------------------------------------------------------

#: Table 4 exactly as the manuscript prints it, at 3 dp.
MANUSCRIPT_TABLE4_3DP = {
    "Test":    {"final_reranker": "0.971", "python_cde_match_approx": "0.719", "bm25": "0.361",
                "ft_mpnet": "0.912", "cde_match_fuzzy": "0.773", "ce_pool_rerank": "0.914"},
    "CCTG":    {"final_reranker": "0.909", "python_cde_match_approx": "0.739", "bm25": "0.392",
                "ft_mpnet": "0.705", "cde_match_fuzzy": "0.780", "ce_pool_rerank": "0.718"},
    "OID ALT": {"final_reranker": "0.832", "python_cde_match_approx": "0.770", "bm25": "0.176",
                "ft_mpnet": "0.428", "cde_match_fuzzy": "0.702", "ce_pool_rerank": "0.564"},
    "CDASH":   {"final_reranker": "0.920", "python_cde_match_approx": "0.750", "bm25": "0.710",
                "ft_mpnet": "0.840", "cde_match_fuzzy": "0.830", "ce_pool_rerank": "0.864"},
    "GDC":     {"final_reranker": "0.972", "python_cde_match_approx": "0.986", "bm25": "0.542",
                "ft_mpnet": "0.833", "cde_match_fuzzy": "0.972", "ce_pool_rerank": "0.903"},
    "CIMAC":   {"final_reranker": "0.802", "python_cde_match_approx": "0.679", "bm25": "0.160",
                "ft_mpnet": "0.565", "cde_match_fuzzy": "0.786", "ce_pool_rerank": "0.595"},
}


@pytest.mark.parametrize("dataset", sorted(MANUSCRIPT_TABLE4_3DP))
def test_table4_renders_the_manuscript_at_three_decimals(table4, dataset):
    """Rounding the fixture once must give the printed table, in all 36 cells."""
    idx = table4.set_index(["dataset", "method"])["recall@5"]
    for method, want in MANUSCRIPT_TABLE4_3DP[dataset].items():
        got = f"{float(idx.loc[(dataset, method)]):.3f}"
        assert got == want, f"{dataset}/{method}: fixture renders {got}, manuscript prints {want}"


def test_table4_values_are_not_stored_pre_rounded(table4):
    """The two cells that a 4-dp source would round the wrong way.

    Both sit exactly on a 4-dp boundary whose float representation falls
    marginally below it, so a value stored at 4 dp renders 0.769 / 0.175 while
    the full-precision value renders 0.770 / 0.176. Table S7 always printed the
    latter for the same two quantities. Storing either at 4 dp reintroduces the
    disagreement, so assert the fixture carries more precision than 4 dp.
    """
    idx = table4.set_index(["dataset", "method"])["recall@5"]
    for method, boundary in (("python_cde_match_approx", 0.7695), ("bm25", 0.1755)):
        v = float(idx.loc[("OID ALT", method)])
        assert abs(v - boundary) > 1e-9, (
            f"OID ALT/{method} is stored as the pre-rounded {boundary}; "
            f"Table 4 would render {boundary:.3f} instead of {v:.3f}")
        assert f"{v:.3f}" == f"{round(boundary + 1e-4, 3):.3f}"
