"""Split routing must not drift.

These assertions were carried, in the research repository, by
``.scratch/demap/paper_v13_scientific_audit/run_fixed_k_stepG_v2.py`` — a driver
script that checked them at the top of the final feature build and then exited.
They are the guard against the single most damaging silent error in this
pipeline: routing CCTG, OID ALT or CDASH to the external allowance-1.0 branch.

That mistake fails nothing. It just hands three of the six reported evaluation
sets the exact gold-metadata evidence the paper deliberately withheld, and every
lexical and reranker number on those sets goes up.
"""
from __future__ import annotations

import pytest

from demap_repro.reranker import split_routing as R

pytestmark = pytest.mark.tier1


@pytest.mark.parametrize("split", ["cctg", "oid_alt", "cdash"])
def test_canonical_externals_are_cadsr_derived(split):
    """CCTG and OID ALT derive from external_holdout_org, CDASH from
    external_holdout_refslice. Their fuzzy tables were generated at allow0.70, so
    they are exact-controlled like any caDSR split despite the legacy naming."""
    assert R.is_cadsr_derived(split)
    assert R.allowance_for_split(split) == R.CADSR_ALLOWANCE == 0.70
    assert R.fuzzy_table_filename(split).endswith("_fuzzy_a070.parquet")


@pytest.mark.parametrize("split", ["gdc_combined", "cimac_v2"])
def test_externally_curated_sets_are_not_controlled(split):
    """GDC and CIMAC queries were written by other organizations against their own
    conventions, so an exact match is legitimate signal and is never withheld."""
    assert not R.is_cadsr_derived(split)
    assert R.allowance_for_split(split) == R.EXTERNAL_ALLOWANCE == 1.0
    assert R.fuzzy_table_filename(split).endswith("_fuzzy_a10.parquet")


@pytest.mark.parametrize("split", ["val_train", "val_dev", "test"])
def test_internal_splits_remain_cadsr_derived(split):
    assert R.is_cadsr_derived(split)
    assert R.allowance_for_split(split) == 0.70


def test_every_paper_split_has_a_hydration_policy():
    """A split with no registered policy would silently take a default. The paper
    build covers all eight; adding a ninth must be a deliberate edit."""
    for split in R.PAPER_FEATURE_TABLE_SPLITS:
        assert split in R.DEFAULT_HYDRATION_POLICY
        assert R.hydration_for_split(split) in ("thinned", "hydrated")


def test_hydration_matches_provenance():
    """caDSR-derived splits stay thinned (PV_BLOCK_CDE only, preserving leakage
    protection); externally curated splits are hydrated."""
    for split in R.PAPER_FEATURE_TABLE_SPLITS:
        expected = "thinned" if R.is_cadsr_derived(split) else "hydrated"
        assert R.hydration_for_split(split) == expected, split


def test_unregistered_split_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="no hydration policy registered"):
        R.hydration_for_split("a_split_that_does_not_exist")


def test_paper_feature_table_splits_are_the_eight_built():
    assert R.PAPER_FEATURE_TABLE_SPLITS == (
        "val_train", "val_dev", "test", "cctg", "oid_alt", "cdash",
        "gdc_combined", "cimac_v2",
    )
