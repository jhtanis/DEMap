"""Figure 4 and Figure 5 input tables.

Two things are checked. First, the committed reference tables carry the values
v21 actually prints — so the numbers in the paper are pinned regardless of whether
anyone can rebuild them. Second, when the corrected artifact roots are available,
the DST builders regenerate those tables exactly.

The second check is the one that matters for provenance. The research
repository's builder for these tables reads pre-correction roots and would
silently restore superseded values; it is retained as history and is never
executed. These builders were written fresh against the ``*_v2_eligible`` roots,
and this is where that claim is tested.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "figure_inputs"

# S5.4 / Figure 4, corrected values.
PAPER_FINETUNED = {"MedCPT": 0.912, "BGE": 0.904, "MiniLM": 0.874}
# Pre-correction values that must NOT reappear (from _build_data2.py's stale roots).
SUPERSEDED_FINETUNED = {"MedCPT": 0.9128, "BGE": 0.8940, "MiniLM": 0.8699}

# 3.6 / Figure 5A.
PAPER_KEYWORD_SELECTION = {
    "cdematch_fuzzy_keyword_v1": 0.782,
    "python_approximation_to_nci_cde_match": 0.728,
}


def _read(name):
    return pd.read_csv(FIXTURES / name)


# --------------------------------------------------------------------------
# Figure 4
# --------------------------------------------------------------------------

def test_finetuned_values_match_the_manuscript():
    df = _read("crossencoder_finetuned.csv")
    canonical = df[df["stage"] == "fulltrain"]
    got = dict(zip(canonical["backbone"],
                   canonical["val_dev_recall_at_5"].round(3)))
    assert got == PAPER_FINETUNED


def test_precorrection_values_are_absent():
    """A regression to the stale roots would put these numbers back."""
    df = _read("crossencoder_finetuned.csv")
    canonical = df[df["stage"] == "fulltrain"]
    got = dict(zip(canonical["backbone"], canonical["val_dev_recall_at_5"].round(4)))
    for backbone, stale in SUPERSEDED_FINETUNED.items():
        assert got[backbone] != stale, (
            f"{backbone} carries the pre-correction value {stale}; the Figure 4 "
            "inputs were rebuilt from the stale roots")


def test_medcpt_is_the_selected_backbone():
    df = _read("crossencoder_finetuned.csv")
    selected = df[df["selected"]]
    assert len(selected) == 1
    assert selected.iloc[0]["backbone"] == "MedCPT"
    assert selected.iloc[0]["stage"] == "fulltrain"


def test_finetuned_sources_are_the_corrected_roots():
    df = _read("crossencoder_finetuned.csv")
    assert df["source_file"].str.contains("_v2_eligible").all()


def test_single_run_per_backbone_is_recorded():
    """One seed per backbone, so Figure 4 carries no error bars. The reason is
    recorded in the table rather than left to the reader."""
    df = _read("crossencoder_finetuned.csv")
    assert (df["n_runs"] == 1).all()
    assert (df["training_seed"] == 20260527).all()
    assert df["error_bar_note"].str.contains("no error bars possible").all()


def test_offtheshelf_val_dev_values_are_present():
    df = _read("crossencoder_offtheshelf.csv")
    val_dev = df[df["split"] == "val_dev"]
    assert len(val_dev) == 3
    assert val_dev["recall_at_5"].astype(float).between(0.5, 0.7).all()


def test_finetuning_improves_every_backbone():
    """Figure 4 plots delta = fine-tuned minus off-the-shelf; all three positive."""
    ft = _read("crossencoder_finetuned.csv")
    ft = ft[ft["stage"] == "fulltrain"].set_index("backbone")["val_dev_recall_at_5"]
    ots = _read("crossencoder_offtheshelf.csv")
    ots = ots[ots["split"] == "val_dev"].set_index("backbone")["recall_at_5"].astype(float)
    for backbone in ft.index:
        assert ft[backbone] - ots[backbone] > 0.25, backbone


def test_canonical_offtheshelf_gap_is_recorded_not_imputed():
    """No untrained CE was ever scored on the six canonical datasets. Those cells
    say MISSING and explain why, rather than being filled in."""
    df = _read("crossencoder_offtheshelf.csv")
    missing = df[df["recall_at_5"] == "MISSING"]
    assert len(missing) == 18   # 3 backbones x 6 canonical datasets
    assert (missing["n_runs"] == 0).all()
    assert missing["note"].str.contains("canonical-score gap").all()


# --------------------------------------------------------------------------
# Figure 5
# --------------------------------------------------------------------------

def test_keyword_selection_matches_section_3_6():
    df = _read("keyword_selection.csv")
    primary = df[df["variant"] == "canonical_val_dev_n3934"]
    got = dict(zip(primary["method"], primary["recall_at_5"].round(3)))
    assert got == PAPER_KEYWORD_SELECTION


def test_fuzzy_wins_the_keyword_selection():
    df = _read("keyword_selection.csv")
    primary = df[df["variant"] == "canonical_val_dev_n3934"].set_index("method")
    margin = (primary.loc["cdematch_fuzzy_keyword_v1", "recall_at_5"]
              - primary.loc["python_approximation_to_nci_cde_match", "recall_at_5"])
    assert round(margin, 4) == 0.0544


def test_fullset_recall_covers_all_six_datasets():
    df = _read("keyword_fullset_recall5.csv")
    assert set(df["dataset"]) == {"test", "cctg", "oid_alt", "cdash",
                                  "gdc_combined", "cimac_v2"}
    assert (df["method"] == "cdematch_fuzzy_keyword_v1").all()


def test_non_exact_panel_omits_gdc():
    """Figure 5 caption: 'GDC omitted from panel C (only two such queries remain)'."""
    df = _read("nonexact_subset_recall5.csv")
    included = df[df["included_in_panel_c"]]
    assert "gdc_combined" not in set(included["dataset"])
    assert set(included["dataset"]) == {"test", "cctg", "oid_alt", "cdash", "cimac_v2"}


def test_biencoder_beats_both_lexical_methods_without_exact_evidence():
    """The point of panel C: with string equality unavailable, the lexical methods
    collapse and the bi-encoder does not."""
    df = _read("nonexact_subset_recall5.csv")
    df = df[df["included_in_panel_c"]]
    for dataset, g in df.groupby("dataset"):
        by_method = dict(zip(g["method"], g["recall_at_5"]))
        mpnet = by_method["ft_mpnet"]
        for lexical in ("python_approximation_to_nci_cde_match",
                        "cdematch_fuzzy_keyword_v1"):
            assert mpnet > by_method[lexical], f"{dataset}: {lexical}"


def test_non_exact_subsets_are_a_substantial_share_of_queries():
    df = _read("nonexact_subset_recall5.csv")
    assert (df["n_non_exact"] > 0).all()


# --------------------------------------------------------------------------
# rebuild from the corrected roots
# --------------------------------------------------------------------------

@pytest.mark.needs_artifacts
@pytest.mark.needs_gated
def test_builders_regenerate_figure4_inputs(artifact_root):
    from demap_repro.reporting.inputs.build_crossencoder import build_figure4_inputs

    root = artifact_root / "artifacts/final_reranker/crossencoder_fulltrain_v2_eligible"
    if not root.exists():
        pytest.skip("corrected cross-encoder roots not present")

    finetuned, offtheshelf = build_figure4_inputs(artifact_root)
    ref = _read("crossencoder_finetuned.csv")
    key = ["backbone", "stage"]
    cols = ["val_dev_recall_at_5", "selected", "rank_within_stage", "n_queries"]
    pd.testing.assert_frame_equal(
        finetuned.set_index(key)[cols].sort_index(),
        ref.set_index(key)[cols].sort_index(),
    )

    ref_ots = _read("crossencoder_offtheshelf.csv")
    got = offtheshelf[offtheshelf["split"] == "val_dev"].set_index("backbone")["recall_at_5"]
    want = ref_ots[ref_ots["split"] == "val_dev"].set_index("backbone")["recall_at_5"]
    pd.testing.assert_series_equal(got.astype(float).sort_index(),
                                   want.astype(float).sort_index())


@pytest.mark.needs_artifacts
@pytest.mark.needs_gated
def test_builders_regenerate_figure5_inputs(artifact_root):
    from demap_repro.reporting.inputs.build_keyword import (
        PANEL_C_DATASETS, build_figure5_inputs,
    )

    probe = (artifact_root / "artifacts/final_reranker"
             / "non_exact_subset_eval_v2_eligible/method_metrics_non_exact.csv")
    if not probe.exists():
        pytest.skip("corrected keyword roots not present")

    selection, fullset, nonexact = build_figure5_inputs(artifact_root)

    ref_sel = _read("keyword_selection.csv")
    ref_sel = ref_sel[ref_sel["variant"] == "canonical_val_dev_n3934"]
    pd.testing.assert_series_equal(
        selection.set_index("method")["recall_at_5"].sort_index(),
        ref_sel.set_index("method")["recall_at_5"].sort_index(),
    )

    pd.testing.assert_series_equal(
        fullset.set_index("dataset")["recall_at_5"].sort_index(),
        _read("keyword_fullset_recall5.csv").set_index("dataset")["recall_at_5"].sort_index(),
    )

    ref_ne = _read("nonexact_subset_recall5.csv")
    ref_ne = ref_ne[ref_ne["dataset"].isin(PANEL_C_DATASETS)]
    key = ["dataset", "method"]
    pd.testing.assert_series_equal(
        nonexact.set_index(key)["recall_at_5"].sort_index(),
        ref_ne.set_index(key)["recall_at_5"].sort_index(),
    )
