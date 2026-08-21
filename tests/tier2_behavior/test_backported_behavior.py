"""Behavioural halves of the backported metamodel bug fixes.

Where ``tests/tier1_invariants/test_backported_plumbing.py`` asserts that a path
resolves or an input is present, these exercise the corrected behaviour: the
evaluation-population lock, the pooling-variant setter under both
sentence-transformers generations, the guarded CIMAC import, and the
by-dataset denominators.

See ``docs/bugfix/2026-08-21-metamodel-bug-backport-audit.md``.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.tier2

ROOT = pathlib.Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Bug 1 — what the curator allowlists actually say
# --------------------------------------------------------------------------

def test_curator_allowlists_match_their_schema_and_buckets():
    """The three files are exact-only literal tables; their shape is load-bearing
    for query construction, so a silent column rename must fail here."""
    allow = pd.read_csv(ROOT / "configs/allowlists/alt_allowlist_current.csv")
    assert list(allow.columns) == ["alternate_name_type", "context_name", "family"]
    assert len(allow) == 13
    assert not allow.isna().any().any(), "exact-only matching forbids blank fields"
    # (type, context) is the lookup key and must be unique.
    assert not allow.duplicated(["alternate_name_type", "context_name"]).any()

    recipes = pd.read_csv(ROOT / "configs/allowlists/alt_query_recipes_current.csv")
    assert set(["recipe_id", "context_name", "alt_type_a", "alt_type_b",
                "joiner", "family", "max_a", "max_b", "enabled"]) == set(recipes.columns)
    assert len(recipes) == 5
    assert recipes["recipe_id"].is_unique

    refdoc = pd.read_csv(ROOT / "configs/allowlists/refdoc_allowlist_current.csv")
    assert list(refdoc.columns) == ["document_type"]
    assert set(refdoc["document_type"]) == {
        "Preferred Question Text", "Alternate Question Text",
        "Application Standard Question Text"}


def test_allowlist_matching_is_case_sensitive_and_literal():
    """The loader documents exact-only literal matching. A case-folded lookup
    would admit buckets the curator never approved."""
    from demap_repro.data.queries import _load_alt_allowlist_csv

    _, bucket_to_family = _load_alt_allowlist_csv(
        str(ROOT / "configs/allowlists/alt_allowlist_current.csv"))
    assert ("TCIA Alt Name", "CIP") in bucket_to_family
    assert ("tcia alt name", "cip") not in bucket_to_family


# --------------------------------------------------------------------------
# Bug 6 — the evaluation-population lock
# --------------------------------------------------------------------------

def test_shipped_registry_is_the_paper_evaluation_population():
    from demap_repro.evaluation.canonical_datasets import assert_paper_evaluation_population

    assert_paper_evaluation_population()          # must not raise


def test_a_v2_derived_registry_is_rejected(tmp_path):
    """``eval_canonical_v2`` is the Rule-E train-decontaminated derivative kept
    for the Table S5 leakage analysis. It is a paper artifact, but it is not the
    reporting population: loading it here would silently swap Table 4's
    denominators (test 3,959 -> 1,151 rows)."""
    from demap_repro.evaluation import canonical_datasets as cd

    registry = yaml.safe_load(pathlib.Path(cd.REGISTRY_PATH).read_text())
    for entry in registry["canonical"]:
        entry["path"] = entry["path"].replace("eval_canonical", "eval_canonical_v2")
    tampered = tmp_path / "registry.yaml"
    tampered.write_text(yaml.safe_dump(registry))

    with pytest.raises(ValueError, match="eval_canonical_v2"):
        cd.assert_paper_evaluation_population(tampered)


def test_a_registry_missing_a_paper_dataset_is_rejected(tmp_path):
    from demap_repro.evaluation import canonical_datasets as cd

    registry = yaml.safe_load(pathlib.Path(cd.REGISTRY_PATH).read_text())
    registry["canonical"] = [e for e in registry["canonical"] if e["name"] != "cimac_v2"]
    tampered = tmp_path / "registry.yaml"
    tampered.write_text(yaml.safe_dump(registry))

    with pytest.raises(ValueError, match="paper population"):
        cd.assert_paper_evaluation_population(tampered)


def test_registry_denominators_are_the_manuscript_ones():
    """Section 2.1 and Table 4: 3,959 / 1,097 / 1,766 / 324 / 72 / 131 queries."""
    from demap_repro.evaluation.canonical_datasets import (
        PAPER_EVAL_QUERY_COUNTS, canonical_count_map)

    counts = canonical_count_map()
    for name, queries in PAPER_EVAL_QUERY_COUNTS.items():
        # Rows are pair-level and queries are deduplicated, so rows >= queries.
        assert counts[name]["reachable_rows"] >= queries, name


# --------------------------------------------------------------------------
# Bug 8 — pooling variants under both sentence-transformers generations
# --------------------------------------------------------------------------

class _LegacyPooling:
    """sentence-transformers < 5: one boolean per mode."""

    def __init__(self):
        self.pooling_mode_cls_token = False
        self.pooling_mode_mean_tokens = True
        self.pooling_mode_max_tokens = False


class _ModernPooling:
    """sentence-transformers >= 5 (5.4.1 is pinned): a single mode string."""

    def __init__(self):
        self.pooling_mode = "mean"


class _FakeSentenceTransformer:
    def __init__(self, pooling):
        self._modules = {"0": object(), "1": pooling}


@pytest.mark.parametrize("pooling_cls", [_LegacyPooling, _ModernPooling],
                         ids=["sentence_transformers_4", "sentence_transformers_5"])
@pytest.mark.parametrize("variant", ["cls", "mean"])
def test_pooling_variant_is_applied_and_reads_back(pooling_cls, variant):
    """Duck-typing only the legacy booleans meant that on 5.x no Pooling module
    was ever found and ``__cls``/``__mean`` raised ``ValueError``."""
    from demap_repro.biencoder.engine.st_loader import (
        apply_pooling_variant, effective_pooling)

    model = _FakeSentenceTransformer(pooling_cls())
    apply_pooling_variant(model, variant)
    assert effective_pooling(model) == variant


def test_modern_pooling_gets_no_inert_legacy_flag():
    """Writing both APIs would leave a stale boolean that disagrees with the mode
    actually in force."""
    from demap_repro.biencoder.engine.st_loader import apply_pooling_variant

    pooling = _ModernPooling()
    apply_pooling_variant(_FakeSentenceTransformer(pooling), "cls")
    assert pooling.pooling_mode == "cls"
    assert not hasattr(pooling, "pooling_mode_cls_token")


def test_base_variant_still_leaves_the_model_alone():
    """The paper's own bi-encoder runs used ``base``, which returns early."""
    from demap_repro.biencoder.engine.st_loader import apply_pooling_variant

    pooling = _ModernPooling()
    apply_pooling_variant(_FakeSentenceTransformer(pooling), "base")
    assert pooling.pooling_mode == "mean"


def test_a_model_with_no_pooling_module_still_raises():
    from demap_repro.biencoder.engine.st_loader import apply_pooling_variant

    class NoPooling:
        _modules = {"0": object()}

    with pytest.raises(ValueError, match="Could not locate"):
        apply_pooling_variant(NoPooling(), "mean")


# --------------------------------------------------------------------------
# Bug 4 / 10 — by-dataset evaluation
# --------------------------------------------------------------------------

def test_by_dataset_imports_without_the_unmigrated_cimac_script():
    import demap_repro.evaluation.by_dataset as by_dataset

    assert callable(by_dataset.by_dataset)


def test_cimac_strata_explains_itself_when_unavailable():
    """It used to surface as a bare ``ImportError`` from inside a metrics call."""
    import demap_repro.evaluation.by_dataset as by_dataset

    with pytest.raises(RuntimeError, match="eval_hgbc_cimac131"):
        by_dataset.cimac_strata(pd.DataFrame(), "hgbc")


def test_default_query_sets_are_the_canonical_evaluation_population():
    """The default named ``splits_v3_cdisc_reachable_2026-06-18_cimacpv``, whose
    ``test`` split holds 3,968 queries against the paper's 3,959 and which has no
    ``cctg`` / ``oid_alt`` / ``cdash`` / ``gdc_combined`` parquet at all."""
    import demap_repro.evaluation.by_dataset as by_dataset
    from demap_repro.evaluation.canonical_datasets import materialized_dir

    assert by_dataset.DEFAULT_REACH_DIR.name == pathlib.Path(materialized_dir()).name
    assert "splits_v3_cdisc" not in str(by_dataset.DEFAULT_REACH_DIR)


def _scored(splits):
    rows = []
    for split in splits:
        for query in ("q1", "q2"):
            for rank, is_label in ((1, query == "q1"), (2, query == "q2")):
                rows.append({"split": split, "query_id": query,
                             "cde_id": f"{rank}v1", "hgbc_rank": rank,
                             "is_label": is_label})
    return pd.DataFrame(rows)


def _write_query_sets(directory, splits):
    directory.mkdir(parents=True, exist_ok=True)
    for split in splits:
        pd.DataFrame({"query_id": ["q1", "q2"]}).to_parquet(directory / f"{split}.parquet")


def test_a_split_with_no_query_set_is_an_error_not_a_note(tmp_path):
    """Silently skipping is how four of the six evaluation datasets could vanish
    from the output table without anyone noticing."""
    import demap_repro.evaluation.by_dataset as by_dataset

    _write_query_sets(tmp_path, ["test"])
    with pytest.raises(FileNotFoundError, match="cctg"):
        by_dataset.by_dataset(_scored(["test", "cctg"]), tmp_path, "hgbc")


def test_missing_splits_can_be_skipped_deliberately(tmp_path):
    import demap_repro.evaluation.by_dataset as by_dataset

    _write_query_sets(tmp_path, ["test"])
    out = by_dataset.by_dataset(_scored(["test", "val_dev"]), tmp_path, "hgbc",
                                allow_missing=True)
    assert list(out["split"]) == ["test"]


def test_the_denominator_is_the_query_set_not_the_rankings(tmp_path):
    """A query absent from the rankings is a miss, not an omission — which is why
    pointing at the wrong query set changes every reported number."""
    import demap_repro.evaluation.by_dataset as by_dataset

    tmp_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"query_id": ["q1", "q2", "q3"]}).to_parquet(tmp_path / "test.parquet")
    out = by_dataset.by_dataset(_scored(["test"]), tmp_path, "hgbc")
    assert int(out.iloc[0]["n"]) == 3
    # q3 is in the query set but not in the rankings: a miss, not an omission.
    assert out.iloc[0]["R@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert int(out.iloc[0]["gold_in_pool"]) == 2


# --------------------------------------------------------------------------
# The BM25 metric helper
# --------------------------------------------------------------------------

def test_bm25_metric_helper_resolves_without_the_research_repository():
    """It used to put ``<repo>/scripts`` on ``sys.path`` and import
    ``evaluate_non_exact_subset``; this repository has no ``scripts/``, so BM25
    evaluation — a Table 4 method — raised ``ModuleNotFoundError``."""
    from demap_repro.lexical.bm25.run import _metric_helper

    helper = _metric_helper()
    assert helper.__name__ == "demap_repro.lexical.non_exact_subset"
    assert callable(helper.build_gold_map)
    assert callable(helper.eval_method)


def test_gold_map_is_public_id_level_any_gold(tmp_path):
    """Version-exact matching would understate every method; the paper matches on
    the public identifier and accepts any of a query's gold CDEs."""
    from demap_repro.lexical.non_exact_subset import build_gold_map

    path = tmp_path / "eval.parquet"
    # ``cde_id`` is ``<public_id>::<version>``, as in data/processed/eval_canonical.
    pd.DataFrame({"query_id": ["q1", "q1", "q2"],
                  "cde_id": ["100::1", "200::3", "300::2"]}).to_parquet(path)
    assert build_gold_map(path) == {"q1": {"100", "200"}, "q2": {"300"}}
