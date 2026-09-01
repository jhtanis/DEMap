"""Every result the manuscript reports has code here that produces it.

This is the invariant the migration exists to establish: no paper-essential
executable behaviour is left behind in the research repository's scratch
directories, untracked files or manuscript revision folders.

It is checked structurally rather than by inspection, because the failure is
quiet. During the migration audit four figures — 2, S1, S2, S4 and S5 — were
recorded as "superseded" when in fact the only thing that produced them was a
Jupyter notebook that had not been migrated. The paper scope said the figures
existed; nothing said the code did.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.tier1

REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "src" / "demap_repro" / "reporting" / "figures"
SCHEMATICS = REPO / "src" / "demap_repro" / "reporting" / "schematics"


@pytest.fixture(scope="module")
def scope():
    return yaml.safe_load((REPO / "manifests" / "paper_scope.yaml").read_text())


@pytest.fixture(scope="module")
def migration():
    return yaml.safe_load((REPO / "manifests" / "source_migration.yaml").read_text())


#: Manuscript figure -> the module that renders it.
FIGURE_GENERATORS = {
    "Figure 2":  "make_figure2_S1_S2_heatmaps.py",
    "Figure S1": "make_figure2_S1_S2_heatmaps.py",
    "Figure S2": "make_figure2_S1_S2_heatmaps.py",
    "Figure 3":  "make_figure3_phase1_phase2.py",
    "Figure 4":  "make_figure4_crossencoder.py",
    "Figure 5":  "make_figure5_keyword.py",
    "Figure S3": "make_figureS3_repxloss.py",
    "Figure S4": "make_figureS4_stage_comparison.py",
    "Figure S5": "make_figureS5_k_ceiling.py",
    "Figure S7": "make_figureS7_S8_allowance.py",
    "Figure S8": "make_figureS7_S8_allowance.py",
}

#: Manually maintained schematics. These are validated, not regenerated.
MANUAL_FIGURES = {"Figure 1", "Figure 6"}

#: Manuscript table -> the module that builds it.
TABLE_BUILDERS = {
    "Table 1":  "src/demap_repro/data/characterization.py",
    "Table 2":  "src/demap_repro/reporting/inputs/build_phase12.py",
    "Table 3":  "src/demap_repro/reporting/inputs/build_phase12.py",
    "Table 4":  "src/demap_repro/reporting/table4.py",
    "Table S1": "src/demap_repro/data/characterization.py",
    "Table S2": "src/demap_repro/data/characterization.py",
    "Table S3": "src/demap_repro/reporting/dataset_tables_pv_overlap.py",
    "Table S4": "src/demap_repro/reporting/inputs/build_stage_comparison.py",
    "Table S5": "src/demap_repro/pool/coverage.py",
    "Table S6": "src/demap_repro/sensitivity/leakage.py",
    "Table S7": "src/demap_repro/sensitivity/allowance/report.py",
}


def test_every_generated_figure_has_a_generator(scope):
    missing = []
    for item in scope["items"]:
        if item["item_type"] != "figure":
            continue
        name = item["item_id"]
        generator = FIGURE_GENERATORS.get(name)
        if generator is None:
            missing.append(f"{name}: no generator mapped")
        elif not (FIGURES / generator).exists():
            missing.append(f"{name}: {generator} missing")
    assert not missing, "figures with no code to produce them:\n  " + "\n  ".join(missing)


def test_manual_schematics_are_declared_as_such(scope):
    for item in scope["items"]:
        if item["item_id"] in MANUAL_FIGURES:
            assert item["item_type"] == "figure_manual"
            assert item["public_runnable"] == "manual_validate_only"


def test_manual_schematic_references_are_retained(scope):
    """Not regenerated, but the generators are kept so the claims stay auditable."""
    assert (SCHEMATICS / "_reference_figure1.py").exists()
    assert (SCHEMATICS / "_reference_figure6.py").exists()


def test_every_table_has_a_builder(scope):
    missing = []
    for item in scope["items"]:
        if item["item_type"] != "table":
            continue
        name = item["item_id"]
        builder = TABLE_BUILDERS.get(name)
        if builder is None:
            missing.append(f"{name}: no builder mapped")
        elif not (REPO / builder).exists():
            missing.append(f"{name}: {builder} missing")
    assert not missing, "tables with no code to produce them:\n  " + "\n  ".join(missing)


def test_no_migration_entry_is_left_undecided(migration):
    """Every candidate must be migrated or explicitly declined with a reason —
    'not_copied' means nobody looked."""
    undecided = [e["src"] for e in migration["entries"]
                 if e["migration_state"] == "not_copied"]
    assert not undecided, (
        "entries neither migrated nor explicitly declined:\n  " + "\n  ".join(undecided))


def test_declined_entries_carry_a_reason(migration):
    missing = [e["src"] for e in migration["entries"]
               if e["migration_state"] == "not_migrated_by_decision"
               and not e.get("exclusion_reason")]
    assert not missing, "declined without a reason:\n  " + "\n  ".join(missing)


def test_no_gated_material_is_marked_copied(migration):
    copied = [e["src"] for e in migration["gated_material"] if e.get("copied_to_dst")]
    assert not copied, f"gated material marked as copied: {copied}"
