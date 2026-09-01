"""Figure S6 — the broad evidence-family ablation.

S5.7's claim is that with the candidate pool held fixed, removing
lexical/keyword-oriented evidence costs far more than removing either neural
family. That only means anything if the partition is a genuine partition, if
every condition scored the same pool, and if the hyperparameters were frozen
rather than re-selected — so those are what is tested, alongside the numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from demap_repro.reranker.family_ablation import (
    CONDITIONS,
    FAMILY_ORDER,
    FROZEN_HYPERPARAMETERS,
    drop_list,
    load_family_map,
    verify_partition,
    write_condition_specs,
)

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: Figure S6 panel A, as the manuscript prints it.
MANUSCRIPT_RECALL5_DELTAS = {
    "minus_lexical_keyword": {"Test": -0.038, "CCTG": -0.145, "OID ALT": -0.232,
                              "CDASH": -0.043, "GDC": -0.083, "CIMAC": -0.153},
    "minus_ft_mpnet":        {"Test": -0.011, "CCTG": -0.011, "OID ALT": 0.000,
                              "CDASH": -0.003, "GDC": 0.000, "CIMAC": +0.008},
    "minus_ft_medcpt":       {"Test": -0.004, "CCTG": -0.012, "OID ALT": -0.008,
                              "CDASH": -0.006, "GDC": 0.000, "CIMAC": -0.008},
}

#: Figure S6 panel B.
MANUSCRIPT_RECALL1_DELTAS = {
    "minus_lexical_keyword": {"Test": -0.136, "CCTG": -0.284, "OID ALT": -0.404,
                              "CDASH": -0.154, "GDC": -0.194, "CIMAC": -0.351},
    "minus_ft_mpnet":        {"Test": -0.019, "CCTG": -0.013, "OID ALT": -0.004,
                              "CDASH": -0.040, "GDC": 0.000, "CIMAC": -0.030},
    "minus_ft_medcpt":       {"Test": -0.009, "CCTG": +0.001, "OID ALT": -0.006,
                              "CDASH": -0.028, "GDC": 0.000, "CIMAC": -0.015},
}

DATASETS = ["Test", "CCTG", "OID ALT", "CDASH", "GDC", "CIMAC"]


@pytest.fixture(scope="module")
def family_map():
    return load_family_map()


@pytest.fixture(scope="module")
def deltas5():
    return pd.read_csv(FIXTURES / "broad_family_recall5_delta.csv").set_index("condition")


@pytest.fixture(scope="module")
def deltas1():
    return pd.read_csv(FIXTURES / "broad_family_recall1_delta.csv").set_index("condition")


@pytest.fixture(scope="module")
def display5():
    return pd.read_csv(FIXTURES / "broad_family_recall5_display.csv").set_index("condition")


# --------------------------------------------------------------------------
# the partition
# --------------------------------------------------------------------------

def test_the_partition_is_exhaustive_and_exclusive(family_map):
    report = verify_partition(family_map)
    assert report["total"] == 117
    assert report["overlap"] == 0
    assert report["sizes"] == {"lexical_keyword": 98, "ft_mpnet": 13, "ft_medcpt": 6}


def test_family_sizes_match_the_caption(family_map):
    """The Figure S6 caption states 98 / 13 / 6."""
    sizes = [len(family_map["families"][f]["members"]) for f in FAMILY_ORDER]
    assert sizes == [98, 13, 6]


def test_pv_and_text_features_are_in_the_lexical_family(family_map):
    """G4 and G5 are grouped by TYPE OF EVIDENCE, not by software provenance.

    They are computed for every pooled candidate from the public catalog and are
    not outputs of CDE Match-Fuzzy. That is why the family is named
    "Lexical/keyword-oriented evidence" and must not be renamed after the
    retriever.
    """
    lex = set(family_map["families"]["lexical_keyword"]["members"])
    assert any(f.startswith("pv_") for f in lex)
    assert any(f.startswith("text_") for f in lex)
    assert family_map["families"]["lexical_keyword"]["label"] == \
        "Lexical/keyword-oriented evidence"


def test_source_indicators_are_split_by_which_arm_they_describe(family_map):
    lex = set(family_map["families"]["lexical_keyword"]["members"])
    mpnet = set(family_map["families"]["ft_mpnet"]["members"])
    assert {"in_cdematch_topk", "in_keyword_topk"} <= lex
    assert "in_biencoder_topk" in mpnet


# --------------------------------------------------------------------------
# the conditions
# --------------------------------------------------------------------------

def test_full_condition_drops_only_the_baseline_five(family_map):
    """`full` must reproduce the shipped model, so it removes no family."""
    drop = drop_list(family_map, "full")
    assert sorted(drop) == sorted(family_map["baseline_excluded"])
    assert len(drop) == 5


@pytest.mark.parametrize("condition", [c for c in CONDITIONS if c != "full"])
def test_each_condition_drops_exactly_its_family(family_map, condition):
    baseline = set(family_map["baseline_excluded"])
    family = CONDITIONS[condition]
    members = set(family_map["families"][family]["members"])
    drop = set(drop_list(family_map, condition))
    assert drop == baseline | members
    assert not (members & baseline), "a family member is already baseline-excluded"


def test_unknown_condition_is_rejected(family_map):
    with pytest.raises(ValueError, match="unknown condition"):
        drop_list(family_map, "minus_everything")


def test_condition_specs_carry_the_frozen_hyperparameters(tmp_path, family_map):
    """Hyperparameters are frozen, not re-selected, so the feature set is the
    only design change between conditions."""
    written = write_condition_specs(tmp_path, family_map)
    assert set(written) == set(CONDITIONS)
    for condition, path in written.items():
        spec = json.loads(path.read_text())
        assert spec["fixed_config"] == FROZEN_HYPERPARAMETERS
        assert spec["fixed_config"] == {"max_iter": 200, "max_depth": 3,
                                        "learning_rate": 0.05, "min_samples_leaf": 30}


def test_the_frozen_config_is_the_shipped_one():
    """S5.5 states the selected configuration; the ablation must reuse it."""
    shipped = json.loads((FIXTURES / "hgbc_selected_config.json").read_text())
    for key, value in FROZEN_HYPERPARAMETERS.items():
        assert shipped[key] == value, key


# --------------------------------------------------------------------------
# the numbers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("condition", list(MANUSCRIPT_RECALL5_DELTAS))
def test_recall5_deltas_match_the_figure(deltas5, condition):
    for dataset, want in MANUSCRIPT_RECALL5_DELTAS[condition].items():
        got = round(float(deltas5.loc[condition, dataset]), 3)
        assert got == pytest.approx(want, abs=1e-9), f"{condition}/{dataset}"


@pytest.mark.parametrize("condition", list(MANUSCRIPT_RECALL1_DELTAS))
def test_recall1_deltas_match_the_figure(deltas1, condition):
    for dataset, want in MANUSCRIPT_RECALL1_DELTAS[condition].items():
        got = round(float(deltas1.loc[condition, dataset]), 3)
        assert got == pytest.approx(want, abs=1e-9), f"{condition}/{dataset}"


def test_the_full_model_reproduces_table_4(display5):
    """The control: the full condition must be the shipped final reranker."""
    table4 = pd.read_csv(FIXTURES / "final_table4.csv")
    final = table4[table4.method == "final_reranker"].set_index("dataset")["recall@5"]
    for dataset in DATASETS:
        shown = str(display5.loc["full", dataset])
        assert shown == f"{float(final.loc[dataset]):.3f}", dataset


def test_lexical_evidence_dominates_on_every_dataset(deltas5, deltas1):
    """S5.7's claim: removing lexical/keyword-oriented evidence costs more than
    removing either neural family, everywhere."""
    for dataset in DATASETS:
        lex = abs(float(deltas5.loc["minus_lexical_keyword", dataset]))
        for other in ("minus_ft_mpnet", "minus_ft_medcpt"):
            assert lex > abs(float(deltas5.loc[other, dataset])), f"Recall@5 {dataset}/{other}"
            assert abs(float(deltas1.loc["minus_lexical_keyword", dataset])) > \
                abs(float(deltas1.loc[other, dataset])), f"Recall@1 {dataset}/{other}"


def test_oid_alt_is_the_largest_lexical_effect(deltas5):
    """The schema-like external set loses most, which is the interpretation
    S5.7 draws."""
    row = deltas5.loc["minus_lexical_keyword", DATASETS].astype(float)
    assert row.idxmin() == "OID ALT"
    assert round(float(row["OID ALT"]), 3) == -0.232


def test_display_values_are_single_rounded(display5, deltas5):
    """Absolute values must come from the display CSV, not from re-rounding the
    4-dp absolute/delta pair. Re-rounding double-rounds - the same failure that
    put 0.769 and 0.175 into Table 4."""
    for dataset in DATASETS:
        cell = str(display5.loc["minus_lexical_keyword", dataset])
        assert "(" in cell and ")" in cell, "display cells carry value and delta"
        shown_delta = float(cell.split("(")[1].rstrip(")").replace("−", "-"))
        assert shown_delta == pytest.approx(
            round(float(deltas5.loc["minus_lexical_keyword", dataset]), 3), abs=1e-9), dataset


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_the_generator_reproduces_the_published_figure(tmp_path):
    """The migrated generator must produce the exact bytes the manuscript embeds.

    md5 537ca37d6f29901b7cc5063ee8398dec, recorded when the figure was built.
    """
    pytest.importorskip("matplotlib")
    import hashlib

    from demap_repro.reporting.figures import make_figureS6_broad_family_ablation as gen

    gen.build(fig_dir=tmp_path)
    png = tmp_path / "figureS6_broad_family_ablation.png"
    assert png.is_file()
    assert hashlib.md5(png.read_bytes()).hexdigest() == "537ca37d6f29901b7cc5063ee8398dec"
