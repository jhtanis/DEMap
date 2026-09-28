"""Executable contract tests for the manuscript-selected Stage-2 pipeline."""
from __future__ import annotations

import json
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from demap_repro.config.paper import (
    PaperConfigError,
    public_hgbc_feature_schema,
    resolve_paper_artifact_path,
    resolve_paper_data_path,
    resolved_paper_config,
    stage_job,
    validate_paper_config,
)
from demap_repro.ranking import assign_canonical_score_ranks


pytestmark = pytest.mark.tier2
REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_CONFIG = REPO_ROOT / "configs/paper/final_system_v1.yaml"


def test_final_system_manifest_validates_and_references_exist():
    paper = validate_paper_config(PAPER_CONFIG)
    assert paper["manifest_id"] == "final_system_v1"
    for value in paper["references"].values():
        assert (REPO_ROOT / value).is_file()


def test_ft_mpnet_final_lineage_is_exact_and_two_seed_selection_is_retained():
    paper = validate_paper_config(PAPER_CONFIG)
    model = paper["ft_mpnet"]
    assert model["base_model"] == "sentence-transformers/all-mpnet-base-v2"
    assert model["phase1"] == {
        **model["phase1"],
        "learning_rate": 1e-4,
        "temperature": 0.04,
        "epochs": 3,
        "batch_size": 128,
        "seeds": [0, 1],
        "retained_seed": 1,
    }
    assert model["phase2"]["parent_run_id"] == model["phase1"]["retained_run_id"]
    assert model["phase2"]["negative_strategy"] == "hard_top25"
    assert model["phase2"]["negative_rank_band"] == [1, 25]
    assert model["phase2"]["learning_rate"] == 5e-5
    assert model["phase2"]["temperature"] == 0.03
    assert model["phase2"]["epochs"] == 1
    assert model["phase2"]["batch_size"] == 11
    assert model["phase2"]["negatives_per_anchor"] == 10
    assert model["phase2"]["encoded_texts_per_example"] == 12
    assert model["phase2"]["encoded_texts_per_batch"] == 132
    assert model["phase2"]["seeds"] == [0, 1]
    assert model["phase2"]["selection_aggregation"] == "two_seed_mean"
    assert model["phase2"]["retained_seed"] == 1
    assert "20260705_203956" in model["phase2"]["retained_run_id"]


def test_candidate_allowance_and_evaluation_contract_are_exact():
    paper = validate_paper_config(PAPER_CONFIG)
    candidates = paper["candidate_construction"]
    assert (candidates["ft_mpnet_top_k"], candidates["cde_match_fuzzy_top_k"]) == (20, 10)
    assert candidates["operation"] == "union"
    assert candidates["deduplicate_by"] == "cde_public_id"
    assert candidates["nominal_max_candidates"] == 30
    assert candidates["inject_evaluation_gold"] is False
    assert candidates["fuzzy"]["exact_match_seed"] == 42
    assert candidates["fuzzy"]["allowance"] == {
        "train": .70, "test": .70, "cctg": .70, "oid_alt": .70,
        "cdash": .70, "gdc_combined": 1.0, "cimac_v2": 1.0,
        "val_train": .70, "val_dev": .70,
    }
    evaluation = paper["data"]["evaluation"]
    assert evaluation["reachable_gold_only"] is True
    assert evaluation["any_accepted_gold_is_hit"] is True
    assert evaluation["metrics"] == ["recall@1", "recall@5", "recall@10", "mrr@100"]


def test_ft_medcpt_contract_and_distinct_ranking_policy_are_exact():
    paper = validate_paper_config(PAPER_CONFIG)
    med = paper["ft_medcpt"]
    assert med["base_model"] == "ncbi/MedCPT-Cross-Encoder"
    assert med["query_representation"]["id"] == "Q3"
    assert med["candidate_representation"]["recipe"] == "SN_DEC_DEF_PQT_PV"
    assert (med["objective"], med["epochs"], med["learning_rate"]) == (
        "pointwise_bce", 2, 2e-5)
    assert (med["batch_size"], med["max_sequence_length"], med["seed"]) == (
        32, 512, 20260527)
    assert (med["warmup_fraction"], med["precision"], med["device"]) == (
        0.1, "fp16", "cuda")
    assert paper["ranking"]["ft_medcpt"]["policy_id"] == "archived_crossenc_rank_v1"
    assert paper["ranking"]["ft_medcpt"]["policy_id"] != paper["ranking"]["final_hgbc"]["policy_id"]


def test_exact_117_feature_contract_is_ordered_unique_and_publicly_generated():
    resolved = resolved_paper_config(PAPER_CONFIG)
    features = resolved["resolved_jobs"]["hgbc"]["features"]
    assert len(features) == 117
    assert len(set(features)) == 117
    assert features == public_hgbc_feature_schema()
    assert resolved["hgbc"]["categorical_features"] == []
    assert resolved["hgbc"]["missing_values"] == "retained_as_nan"
    assert resolved["hgbc"]["class_weight"] is None
    assert resolved["hgbc"]["sample_weight"] is None


def test_hgbc_job_is_fixed_not_grid_and_has_exact_configuration():
    job = stage_job("hgbc", PAPER_CONFIG)
    assert job["hyperparameter_selection"] == "fixed_final_configuration"
    assert job["hyperparameters"] == {
        "max_iter": 200, "max_depth": 3, "learning_rate": .05,
        "min_samples_leaf": 30, "random_seed": 42,
    }


def test_final_hgbc_numeric_tie_order_is_input_order_invariant():
    left = pd.DataFrame({
        "split": ["test", "test"], "query_id": ["q", "q"],
        "cde_publicid": ["10", "2"], "hgbc_score": [.5, .5],
    })
    right = left.iloc[::-1].reset_index(drop=True)
    ranked_left = assign_canonical_score_ranks(
        left, group_cols=["split", "query_id"], score_col="hgbc_score",
        public_id_col="cde_publicid", rank_col="rank")
    ranked_right = assign_canonical_score_ranks(
        right, group_cols=["split", "query_id"], score_col="hgbc_score",
        public_id_col="cde_publicid", rank_col="rank")
    assert ranked_left.set_index("cde_publicid")["rank"].to_dict() == {"10": 2, "2": 1}
    assert ranked_right.set_index("cde_publicid")["rank"].to_dict() == {"2": 1, "10": 2}


def test_final_hgbc_ranking_rejects_duplicate_public_ids():
    frame = pd.DataFrame({
        "query_id": ["q", "q"], "cde_publicid": ["2", "2"],
        "hgbc_score": [.5, .4],
    })
    with pytest.raises(ValueError, match="duplicate public identifier"):
        assign_canonical_score_ranks(
            frame, group_cols=["query_id"], score_col="hgbc_score",
            public_id_col="cde_publicid", rank_col="rank")


def test_resolved_jobs_bridge_public_build_to_paper_consumers():
    resolved = resolved_paper_config(PAPER_CONFIG)
    jobs = resolved["resolved_jobs"]
    build = resolved["data"]["build"]
    assert jobs["deep_retrieval"]["split_groups"]["training"]["splits"] == [
        "train", "val_train", "val_dev"]
    assert jobs["deep_retrieval"]["split_groups"]["training"]["splits_dir"] == build["training_splits_dir"]
    assert jobs["deep_retrieval"]["split_groups"]["evaluation"]["splits_dir"] == build["canonical_eval_dir"]
    assert jobs["ft_medcpt_pool"]["inject_gold"] == {"train": True, "val_dev": False}
    assert build["historical_training_populations"] == {
        "train": {"rows": 51594, "unique_queries": 47645},
        "val_train": {"rows": 4260, "unique_queries": 3946},
        "val_dev": {"rows": 4234, "unique_queries": 3934},
    }
    assert jobs["candidate_pool"]["splits"] == [
        "val_train", "val_dev", "test", "cctg", "oid_alt", "cdash",
        "gdc_combined", "cimac_v2"]
    assert jobs["candidate_pool"]["keyword_top_k_per_rule"] == 500
    assert jobs["candidate_pool"]["keyword_allow_rate_cadsr"] == 0.70
    assert jobs["candidate_pool"]["keyword_allow_seed"] == 42


def test_evaluation_registry_names_reachable_build_sources():
    import yaml
    registry = yaml.safe_load(
        (REPO_ROOT / "configs/paper/eval_datasets_v1.yaml").read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in registry["canonical"]}
    prefix = "data/processed/splits_catalog_filtered"
    assert rows["test"]["source"] == [f"{prefix}/test.parquet"]
    assert rows["cctg"]["source"] == [f"{prefix}/external_holdout_org.parquet"]
    assert rows["oid_alt"]["source"] == [f"{prefix}/external_holdout_org.parquet"]
    assert rows["cdash"]["source"] == [f"{prefix}/external_holdout_refslice.parquet"]


def test_data_and_artifact_roots_are_resolved_at_call_time(monkeypatch, tmp_path):
    data = tmp_path / "data-root"
    artifacts = tmp_path / "artifact-root"
    monkeypatch.setenv("DEMAP_DATA_ROOT", str(data))
    monkeypatch.setenv("DEMAP_ARTIFACT_ROOT", str(artifacts))
    assert resolve_paper_data_path("data/example.parquet") == data / "data/example.parquet"
    assert resolve_paper_artifact_path("artifacts/example") == artifacts / "artifacts/example"


def test_paper_dataset_builder_routes_outputs_to_configured_roots(monkeypatch, tmp_path):
    if importlib.util.find_spec("lxml") is None:
        lxml = types.ModuleType("lxml")
        lxml.etree = types.ModuleType("lxml.etree")
        monkeypatch.setitem(sys.modules, "lxml", lxml)
        monkeypatch.setitem(sys.modules, "lxml.etree", lxml.etree)
    from demap_repro.data import pipeline

    data = tmp_path / "data-root"
    artifacts = tmp_path / "artifact-root"
    query_xml = data / "query-xml"
    catalog_xml = data / "catalog-xml"
    query_xml.mkdir(parents=True)
    catalog_xml.mkdir(parents=True)
    monkeypatch.setenv("DEMAP_DATA_ROOT", str(data))
    monkeypatch.setenv("DEMAP_ARTIFACT_ROOT", str(artifacts))
    captured = {}

    def fake_catalog_filter(argv):
        captured["argv"] = list(argv)

    monkeypatch.setattr(pipeline.catalog_filter, "main", fake_catalog_filter)
    pipeline.main([
        "--paper-config", str(PAPER_CONFIG),
        "--query-xml", str(query_xml), "--catalog-xml", str(catalog_xml),
        "--start-at", "catalog-filter", "--stop-after", "catalog-filter",
    ])
    argv = captured["argv"]
    assert str(data / "data/processed/splits") in argv
    assert str(data / "data/processed/splits_catalog_filtered") in argv
    assert str(data / "data/processed/cadsr_xml_2026-06-18/cde_master_enriched_eval_production_cde_match.parquet") in argv
    assert str(artifacts / "artifacts/manifests/catalog_filter") in argv
    assert (data / "data/processed/dataset_build_manifest.json").is_file()
    assert (artifacts / "artifacts/manifests/dataset_build_manifest.json").is_file()


def test_resolved_public_config_contains_no_internal_absolute_paths():
    rendered = json.dumps(resolved_paper_config(PAPER_CONFIG), sort_keys=True)
    assert "/data/nextgen2" not in rendered
    assert "/vf/users" not in rendered


@pytest.mark.parametrize("stage", [
    "ft_mpnet_phase1", "ft_mpnet_phase2", "deep_retrieval", "cde_match_fuzzy",
    "ft_medcpt_pool", "candidate_pool", "ft_medcpt_train", "ft_medcpt_score",
    "merge_ft_medcpt_features", "hgbc", "evaluation",
])
def test_every_paper_stage_has_a_machine_readable_dry_run(stage):
    job = stage_job(stage, PAPER_CONFIG)
    assert isinstance(job, dict) and job


def test_ft_mpnet_selected_dry_run_has_no_stale_may_fallback(capsys):
    from demap_repro.biencoder import cli
    cli.main(["--paper-config", str(PAPER_CONFIG), "--selected-final",
              "--stage", "phase2", "--dry-run"])
    output = capsys.readouterr().out
    assert "20260703_142804" in output
    assert "hard_top25" in output
    assert "May" not in output


def test_ft_medcpt_dry_run_cannot_fall_back_to_minilm(capsys):
    from demap_repro.crossencoder import train
    assert train.main(["--paper-config", str(PAPER_CONFIG), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "ncbi/MedCPT-Cross-Encoder" in output
    assert "MiniLM" not in output


def test_paper_mode_rejects_conflicting_scientific_override():
    from demap_repro.crossencoder import train
    with pytest.raises(SystemExit):
        train.main(["--paper-config", str(PAPER_CONFIG), "--dry-run",
                    "--base-model", "cross-encoder/ms-marco-MiniLM-L-6-v2"])


def test_fuzzy_paper_adapter_rejects_generic_scientific_override():
    from demap_repro.lexical.cde_match import paper_candidates
    with pytest.raises(SystemExit):
        paper_candidates.main([
            "--paper-config", str(PAPER_CONFIG), "--paper-dataset", "test",
            "--dry-run", "--splits", "cimac_v2",
        ])


def test_config_drift_fails_loudly(tmp_path):
    import yaml
    altered = yaml.safe_load(PAPER_CONFIG.read_text(encoding="utf-8"))
    altered["candidate_construction"]["ft_mpnet_top_k"] = 21
    path = tmp_path / "drift.yaml"
    path.write_text(yaml.safe_dump(altered, sort_keys=False), encoding="utf-8")
    with pytest.raises(PaperConfigError, match="candidate FT-MPNet depth"):
        validate_paper_config(path)
