"""Permissible-value summary generation.

Query-side and CDE-side PV summaries are the one place where a benchmark like
this can leak the answer into the question. Both sides are rendered from the same
upstream permissible values, so a careless implementation would hand the query a
verbatim copy of the gold CDE's value list — and the retrieval task would collapse
into string matching.

Several safeguards prevent that. The 2026-08-07 audit verified them empirically
but found no permanent tests, so they are written here: the query side is capped
below the CDE side, generation is deterministic from the CDE identity rather than
from RNG state, small value sets follow an explicit policy instead of being
emitted verbatim, and blocks that come out identical are resampled and then
dropped.

The audit's row-level result — 46,948 / 46,948 exact on both sides against the
frozen benchmark — is recorded as a fixture and asserted here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from demap_repro.text import pv_summary, pv_summary_v2 as pv2

pytestmark = pytest.mark.tier2

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# data/processed/dataset_build_manifest.json — the paper-era build parameters.
PAPER_PV_PARAMS = {
    "pv_max_n_query": 8,
    "pv_max_n_cde": 10,
    "sde_generic_label_p": 0.30,
    "pv_center_fraction": 0.5,
    "pv_min_n": 2,
    "pv_size_jitter": 1,
    "pv_generic_boost": 0.10,
    "pv_generic_cap": 0.70,
    "pv_small_n_generic_threshold": 0.5,
    "pv_max_resample_attempts": 3,
}

PROFILE = "cadsr"   # the profile used for the paper build


@pytest.fixture(scope="module")
def config():
    return pv2.default_config_from_kwargs()


def _pool(values, *, publicid="2002587", version="3", config=None):
    """Canonical PV pool from a list of (valid_value, preferred_meaning) pairs."""
    df = pd.DataFrame({
        "cde_publicid": [publicid] * len(values),
        "cde_version": [version] * len(values),
        "valid_value": [v for v, _ in values],
        "preferred_meaning_label": [m for _, m in values],
    })
    return pv2.canonicalize_pv_pool(
        df, profile=PROFILE, config=config or pv2.default_config_from_kwargs())


def _enum(n=20):
    return [(f"C{i:04d}", f"Concept {i}") for i in range(n)]


# --------------------------------------------------------------------------
# defaults match the paper build
# --------------------------------------------------------------------------

def test_default_config_matches_the_paper_era_build(config):
    for key, expected in PAPER_PV_PARAMS.items():
        assert config[key] == expected, key


def test_query_omission_uses_no_placeholder(config):
    """A placeholder token would itself be a signal. The paper build omits the
    block entirely instead."""
    assert config["use_placeholder_for_query_omission"] is False


# --------------------------------------------------------------------------
# the asymmetry that keeps the query from copying the answer
# --------------------------------------------------------------------------

def test_query_side_cap_is_lower_than_cde_side_cap(config):
    """The single most important safeguard. If the query could show as many values
    as the CDE, an exact block match would be achievable by construction."""
    assert config["pv_max_n_query"] < config["pv_max_n_cde"]


@pytest.mark.parametrize("n_values", [2, 3, 5, 12, 40, 200])
def test_sample_size_never_exceeds_the_side_cap(config, n_values):
    pool = _pool(_enum(n_values), config=config)
    for side, cap_key in (("query", "pv_max_n_query"), ("cde", "pv_max_n_cde")):
        seed = pv2._seed_base(pool, side=side, salt="demap")
        k = pv2.compute_side_sample_size(pool, side=side, config=config, seed=seed)
        assert k <= config[cap_key], (side, n_values, k)
        assert k <= pool.n


def test_the_asymmetry_is_a_cap_not_a_per_cde_guarantee(config):
    """The two sides are sampled independently, each with its own jitter, so for an
    individual CDE the query block can come out longer than the CDE block. The
    guarantee is distributional, enforced by the caps.

    This matters for reading Table S3: 'query shorter than CDE' is 30.1% overall,
    not 100%, and that is by design rather than a defect.
    """
    longer = 0
    total = 0
    for n_values in (3, 5, 8, 12, 40):
        for i in range(60):
            pool = _pool(_enum(n_values), publicid=f"{2000000 + i}", config=config)
            kq = pv2.compute_side_sample_size(
                pool, side="query", config=config,
                seed=pv2._seed_base(pool, side="query", salt="demap"))
            kc = pv2.compute_side_sample_size(
                pool, side="cde", config=config,
                seed=pv2._seed_base(pool, side="cde", salt="demap"))
            assert kq <= config["pv_max_n_query"]
            assert kc <= config["pv_max_n_cde"]
            longer += kq > kc
            total += 1
    assert 0 < longer < total, (
        "expected the query side to be longer sometimes but not usually; "
        f"got {longer}/{total}")


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def test_seed_derives_from_identity_not_rng_state(config):
    """Seeds come from a hash over (salt, public id, version, side), so a rebuild
    on another host or in another process order produces identical summaries."""
    a = _pool(_enum(), config=config)
    b = _pool(_enum(), config=config)
    assert (pv2._seed_base(a, side="query", salt="demap")
            == pv2._seed_base(b, side="query", salt="demap"))


def test_the_two_sides_get_different_seeds(config):
    """Same CDE, different side: a shared seed would make both sides select the
    same values and the blocks would tend to coincide."""
    pool = _pool(_enum(), config=config)
    assert (pv2._seed_base(pool, side="query", salt="demap")
            != pv2._seed_base(pool, side="cde", salt="demap"))


def test_different_cdes_get_different_seeds(config):
    a = _pool(_enum(), publicid="2002587", config=config)
    b = _pool(_enum(), publicid="2002588", config=config)
    assert (pv2._seed_base(a, side="query", salt="demap")
            != pv2._seed_base(b, side="query", salt="demap"))


def test_salt_changes_the_seed(config):
    pool = _pool(_enum(), config=config)
    assert (pv2._seed_base(pool, side="query", salt="demap")
            != pv2._seed_base(pool, side="query", salt="other"))


def test_deterministic_selection_is_stable_and_bounded():
    items = [f"value_{i}" for i in range(20)]
    first = pv2._deterministic_select(items, k=5, seed="s", key_fn=str)
    again = pv2._deterministic_select(items, k=5, seed="s", key_fn=str)
    assert first == again
    assert len(first) == 5
    assert set(first) <= set(items)


def test_a_different_seed_generally_selects_differently():
    items = [f"value_{i}" for i in range(20)]
    a = pv2._deterministic_select(items, k=5, seed="s1", key_fn=str)
    b = pv2._deterministic_select(items, k=5, seed="s2", key_fn=str)
    assert a != b


# --------------------------------------------------------------------------
# identical-block avoidance
# --------------------------------------------------------------------------

def test_strict_subset_detection():
    assert pv2._items_strict_subset(["a"], ["a", "b"])
    assert not pv2._items_strict_subset(["a", "b"], ["a", "b"])
    assert not pv2._items_strict_subset(["a", "c"], ["a", "b"])


def test_identical_blocks_are_resampled_then_dropped(config):
    """With only two values both sides must show the same items, so the anti-leakage
    pass has no escape and must empty the query side rather than emit a copy."""
    pool = _pool([("Y", "Yes"), ("N", "No")], config=config)
    items = ["Yes", "No"]
    out, diag = pv2.apply_anti_leakage(
        pool, items, items, profile=PROFILE, config=config,
        seed_base=pv2._seed_base(pool, side="query", salt="demap"))
    assert out == [] or out != items


def test_anti_leakage_leaves_distinct_blocks_alone(config):
    pool = _pool(_enum(30), config=config)
    query_items = [f"Concept {i}" for i in range(4)]
    cde_items = [f"Concept {i}" for i in range(10, 20)]
    out, _ = pv2.apply_anti_leakage(
        pool, query_items, cde_items, profile=PROFILE, config=config,
        seed_base=pv2._seed_base(pool, side="query", salt="demap"))
    assert out == query_items


def test_small_value_sets_follow_an_explicit_policy(config):
    """Two-value sets are where verbatim copying is most likely; the policy decides
    to omit or to keep-with-rule rather than defaulting to emit."""
    pool = _pool([("Y", "Yes"), ("N", "No")], config=config)
    out, diag = pv2.apply_small_n_policy(
        pool, ["Yes", "No"], ["Yes", "No"], profile=PROFILE, config=config)
    assert isinstance(out, list)
    assert diag, "the small-n policy must record which rule it applied"


# --------------------------------------------------------------------------
# typing and rendering
# --------------------------------------------------------------------------

@pytest.mark.parametrize("meaning,expected", [
    ("Yes", "YES"), ("yes", "YES"), ("No", "NO"), ("Unknown", "UNKNOWN"),
    ("Not Applicable", "NA"), ("n/a", "NA"), ("Positive", "POSITIVE"),
    ("Not Reported", "NOT_REPORTED"),
])
def test_generic_concepts_are_recognized_case_insensitively(meaning, expected):
    """Generic values (Yes/No/Unknown/NA and friends) carry no discriminative
    signal, so they are bucketed rather than treated as distinct concepts."""
    assert pv_summary.generic_concept(meaning) == expected


@pytest.mark.parametrize("meaning", [
    "Squamous cell carcinoma", "Stage IIIA", "Other", "Left upper lobe",
])
def test_substantive_values_are_not_generic(meaning):
    """'Other' is deliberately NOT in the generic vocabulary: it is a real,
    selectable permissible value in caDSR, not a missing-data marker."""
    assert pv_summary.generic_concept(meaning) is None


def test_pv_type_signature_is_stable():
    assert (pv_summary.pv_type_signature(["Yes", "No", "Unknown"])
            == pv_summary.pv_type_signature(["Yes", "No", "Unknown"]))


def test_binary_families_are_distinguished():
    """ENUM, BINARY_WITH_UNKNOWN and BINARY_WITH_NA are reported as separate rows
    in Table S3, so the signature must actually separate them."""
    signatures = {
        pv_summary.pv_type_signature(["Yes", "No", "Unknown"]),
        pv_summary.pv_type_signature(["Yes", "No", "Not Applicable"]),
        pv_summary.pv_type_signature(["Yes", "No"]),
        pv_summary.pv_type_signature(["Stage I", "Stage II", "Stage III"]),
    }
    assert len(signatures) == 4


def test_block_formatting_is_deterministic():
    a = pv_summary._format_block("ENUM", 3, ["Yes", "No", "Unknown"])
    b = pv_summary._format_block("ENUM", 3, ["Yes", "No", "Unknown"])
    assert a == b
    assert "Yes" in a


def test_query_normalization_is_idempotent():
    once = pv_summary.normalize_query_text("  Tumor   Size (mm) ")
    assert pv_summary.normalize_query_text(once) == once


# --------------------------------------------------------------------------
# the audited row-level result
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def parity():
    return json.loads((FIXTURES / "pv_row_level_parity.json").read_text())


def test_audited_row_level_parity_was_exact(parity):
    """2026-08-07 audit: the current implementation regenerates the frozen
    benchmark's PV blocks exactly, on both sides, over the whole dataset."""
    for side in ("query", "cde"):
        assert parity[side]["mismatches"] == 0
        assert parity[side]["pct_identical"] == 100.0
        assert parity[side]["exact_matches"] == parity[side]["comparable_rows"] == 46948
        assert parity[side]["missing_to_present"] == 0
        assert parity[side]["present_to_missing"] == 0


def test_audit_covered_the_canonical_69102_pair_benchmark(parity):
    assert parity["query"]["total_rows"] == parity["cde"]["total_rows"] == 69102


def test_pv_type_also_reproduced_exactly(parity):
    assert parity["pv_type"]["mismatches"] == 0
    assert parity["pv_type"]["exact_matches"] == 46948
