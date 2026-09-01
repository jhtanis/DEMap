"""The frozen CIMAC evaluation input, and the gates it is no longer behind.

CIMAC's queries and gold CDE mappings come from the NCI CIMAC-CIDC
clinical-data-element template — template and data-element metadata, not
patient-level study data — which NCI publishes openly. The Appendix A workbook
served today is byte-identical to the copy this study used, so that half of the
provenance is independently verifiable.

Permissible-value metadata came from a second public workbook that has since
changed upstream, with no dated archive, so the exact PV-enriched 131-query
representation is shipped as a frozen artifact instead of regenerated. These
tests pin that artifact and check it still says what the manuscript reports.
"""
from __future__ import annotations

import hashlib
import pathlib

import pandas as pd
import pytest
import yaml

pytestmark = [pytest.mark.tier3, pytest.mark.parity]

REPO = pathlib.Path(__file__).resolve().parents[2]
FROZEN = REPO / "data" / "frozen" / "cimac_v2.parquet"

#: The artifact as shipped. A substitution must not go unnoticed.
SHA256 = "1d0797330864cb8a73d277c1885fa5bba41a63d3a011cfaaa148e59c6dce4fe0"

#: The public Appendix A workbook, verified byte-identical to the study copy.
PUBLIC_WORKBOOK_SHA256 = "6d657dfd3f7289e74137929836974f742136fea3d2cb753432499418a57c71a3"

REQUIRED_COLUMNS = {
    "query_id", "query_source", "query_field", "family", "query_text_raw",
    "cde_publicid", "cde_version", "cde_id", "gold_versioned",
    "PV_N", "PV_TYPE", "PV_BLOCK_SDE", "pv_attached", "pv_block", "query_text_q3",
}


@pytest.fixture(scope="module")
def cimac():
    return pd.read_parquet(FROZEN)


def test_the_frozen_artifact_ships():
    assert FROZEN.is_file(), f"{FROZEN} is missing; it is a required evaluation input"


def test_digest_is_pinned():
    got = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    assert got == SHA256, (
        f"data/frozen/cimac_v2.parquet changed.\n  expected {SHA256}\n  got      {got}\n"
        "This is the exact evaluation input the manuscript used. If it must change, "
        "the CIMAC column of every reported table changes with it.")


def test_it_is_the_131_query_paper_population(cimac):
    from demap_repro.evaluation.canonical_datasets import PAPER_EVAL_QUERY_COUNTS

    assert len(cimac) == 131
    assert cimac["query_id"].nunique() == 131
    assert PAPER_EVAL_QUERY_COUNTS["cimac_v2"] == 131


def test_schema_carries_everything_evaluation_needs(cimac):
    missing = REQUIRED_COLUMNS - set(cimac.columns)
    assert not missing, f"frozen CIMAC input is missing columns: {sorted(missing)}"


def test_query_ids_follow_the_canonical_scheme(cimac):
    ids = cimac["query_id"].astype(str)
    assert ids.str.match(r"^cimac_v2::\d{3}$").all()
    assert ids.min() == "cimac_v2::001"
    assert ids.max() == "cimac_v2::154", "ids run to 154; 131 survive reachable-gold filtering"


def test_the_pv_enriched_representation_is_present(cimac):
    """``query_text_q3`` is what evaluation consumes, and is the reason this
    artifact is frozen rather than regenerated."""
    q3 = cimac["query_text_q3"].astype(str)
    assert (q3.str.strip() != "").all(), "every query must carry a q3 representation"
    with_pv = cimac["PV_BLOCK_SDE"].astype(str).str.strip().replace("nan", "") != ""
    assert with_pv.sum() == 59, f"expected 59 PV-bearing queries, got {int(with_pv.sum())}"
    # Where a PV block exists, q3 must actually differ from the bare query text.
    differs = (q3[with_pv].str.strip() != cimac.loc[with_pv, "query_text_raw"].astype(str).str.strip())
    assert differs.all(), "PV-bearing queries must have q3 != raw"


def test_every_gold_is_a_versioned_cde(cimac):
    """``<public id>::<version>``, where the version may be decimal.

    Two CIMAC golds carry decimal caDSR versions - 88::5.1 and 2003853::4.2 -
    so a `\\d+::\\d+` pattern is wrong. caDSR versions are not integers.
    """
    pattern = r"^\d+::\d+(\.\d+)?$"
    ids = cimac["cde_id"].astype(str)
    ok = ids.str.match(pattern)
    assert ok.all(), f"malformed gold ids: {sorted(set(ids[~ok]))}"
    assert cimac["cde_publicid"].notna().all()
    assert {"88::5.1", "2003853::4.2"} <= set(ids), "the two decimal-version golds"


# --------------------------------------------------------------------------
# CIMAC is no longer gated
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ledger():
    return yaml.safe_load((REPO / "manifests" / "source_migration.yaml").read_text())


def test_the_cimac_gate_is_retired(ledger):
    gate = ledger["release_gates"]["cimac_unresolved"]
    assert gate["status"] == "resolved"
    assert not gate["affects_stages"]
    assert PUBLIC_WORKBOOK_SHA256 in gate["resolution"], (
        "the resolution must record the digest that makes the claim checkable")


def test_no_cimac_material_is_still_gated(ledger):
    still = [e["src"] for e in ledger["gated_material"]
             if e["release_gate"] == "cimac_unresolved"]
    assert not still, f"CIMAC entries still in gated_material: {still}"


def test_the_superseded_pv_workbook_is_declined_with_a_reason(ledger):
    """It is public but stale upstream, so it is deliberately not shipped."""
    entry = next((e for e in ledger["entries"]
                  if str(e.get("src", "")).endswith("CIMAC-CIDC_Permissible_Values.xlsx")), None)
    assert entry is not None, "the PV workbook decision must be recorded"
    assert entry["migration_state"] == "not_migrated_by_decision"
    assert "superseded_upstream" in entry["exclusion_reason"]


def test_no_manifest_references_a_retired_gate():
    """A result family pointing at a retired gate overstates what is restricted."""
    import json

    led = yaml.safe_load((REPO / "manifests" / "source_migration.yaml").read_text())
    retired = {k for k, v in led["release_gates"].items()
               if v.get("status") in ("resolved", "not_distributed_by_size")}
    stale = []

    results = json.loads((REPO / "manifests" / "expected_results.json").read_text())["results"]
    for key, fam in results.items():
        g = fam.get("release_gate")
        if isinstance(g, str):
            stale += [f"expected_results:{key} -> {s.strip()}"
                      for s in g.split(";") if s.strip() in retired]

    scope = yaml.safe_load((REPO / "manifests" / "paper_scope.yaml").read_text())
    for item in scope["items"]:
        g = item.get("release_gate")
        if isinstance(g, str):
            stale += [f"paper_scope:{item['id']} -> {s.strip()}"
                      for s in g.split(";") if s.strip() in retired]

    assert not stale, "manifests cite gates that are retired:\n  " + "\n  ".join(stale)
