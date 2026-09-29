"""Strict loader for the canonical manuscript pipeline configuration.

The paper manifest is deliberately a small integration layer over the historical
protocol and winner manifests.  It does not replace research configs; it makes the
one selected manuscript path explicit and fails if a referenced contract drifts.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

from demap_repro.utils.paths import artifact_root, data_root, repo_root

DEFAULT_PAPER_CONFIG = "configs/paper/final_system_v1.yaml"
PAPER_CONFIG_SCHEMA_VERSION = 1


class PaperConfigError(ValueError):
    """The manuscript configuration is missing, inconsistent, or unsafe."""


def resolve_paper_data_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root() / path


def resolve_paper_artifact_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else artifact_root() / path


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = Path.cwd() / path
    return direct if direct.exists() else repo_root() / path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = _repo_path(path)
    if not resolved.is_file():
        raise PaperConfigError(f"required paper config does not exist: {path}")
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PaperConfigError(f"{path}: expected a mapping")
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = _repo_path(path)
    if not resolved.is_file():
        raise PaperConfigError(f"required paper config does not exist: {path}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PaperConfigError(f"{path}: expected a JSON object")
    return value


def _winner(rows: Iterable[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise PaperConfigError(
            f"expected exactly one winner with {key}={value!r}; found {len(matches)}")
    return matches[0]


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise PaperConfigError(f"{label}: expected {expected!r}, found {actual!r}")


def public_hgbc_feature_schema() -> list[str]:
    """Ordered final feature schema producible by the public feature pipeline."""
    from demap_repro.reranker.features.biencoder import biencoder_feature_columns
    from demap_repro.reranker.features.cdematch import CDEMATCH_FEATURE_COLUMNS
    from demap_repro.reranker.features.lexical_features import kw_feature_columns
    from demap_repro.reranker.features.pv_overlap import feature_columns as pv_feature_columns
    from demap_repro.reranker.features.text_features import text_feature_columns
    from demap_repro.reranker.merge_crossenc_features import CE_FEATURES

    raw = [
        "in_biencoder_topk", "in_cdematch_topk", "biencoder_rank",
        "biencoder_score", "cdematch_score", "in_keyword_topk",
    ]
    produced = (
        raw
        + list(biencoder_feature_columns())
        + list(CDEMATCH_FEATURE_COLUMNS)
        + list(pv_feature_columns())
        + list(text_feature_columns(include_tier2=True, include_tier3=True))
        + list(kw_feature_columns())
        + list(CE_FEATURES)
    )
    excluded = {
        "cdematch_rank", "log1p_cdematch_rank", "text_family",
        "text_query_source", "keyword_rank",
    }
    return [name for name in produced if name not in excluded]


def load_paper_config(path: str | Path = DEFAULT_PAPER_CONFIG) -> dict[str, Any]:
    cfg = _load_yaml(path)
    _expect(cfg.get("schema_version"), PAPER_CONFIG_SCHEMA_VERSION, "paper schema_version")
    _expect(cfg.get("manifest_id"), "final_system_v1", "paper manifest_id")
    return cfg


def validate_paper_config(path: str | Path = DEFAULT_PAPER_CONFIG) -> dict[str, Any]:
    """Validate the manifest and all referenced paper contracts, fail closed."""
    cfg = load_paper_config(path)
    refs = cfg.get("references") or {}
    required_refs = {
        "biencoder_protocol", "phase1_winners", "phase2_winners",
        "crossencoder_protocol", "feature_contract", "allowance",
        "evaluation_registry", "determinism", "table4_results", "public_build",
    }
    _expect(set(refs), required_refs, "paper reference keys")
    for ref in refs.values():
        if not _repo_path(ref).is_file():
            raise PaperConfigError(f"referenced config does not exist: {ref}")

    protocol = _load_yaml(refs["biencoder_protocol"])
    phase1 = _load_yaml(refs["phase1_winners"])
    phase2 = _load_yaml(refs["phase2_winners"])
    ce = _load_yaml(refs["crossencoder_protocol"])
    allowance = _load_yaml(refs["allowance"])
    registry = _load_yaml(refs["evaluation_registry"])
    determinism = _load_json(refs["determinism"])
    features = _load_json(refs["feature_contract"])
    pipeline = _load_yaml(refs["public_build"])

    data = cfg["data"]
    catalog = data["catalog"]
    _expect(catalog["expected_records"], 62976, "catalog record count")
    _expect(catalog["expected_unique_public_ids"], 62858, "catalog public-ID count")
    _expect(catalog["eligibility_mode"], "production_cde_match", "catalog eligibility")
    _expect(protocol["data"]["production_catalog"], catalog["logical_path"],
            "bi-encoder production catalog")
    _expect(protocol["data"]["production_catalog_rows"], catalog["expected_records"],
            "bi-encoder catalog rows")
    _expect(protocol["data"]["splits_dir"], data["build"]["training_splits_dir"],
            "bi-encoder historical training splits")
    _expect(registry["production_catalog"], catalog["logical_path"],
            "evaluation production catalog")
    _expect(registry["production_catalog_size"], catalog["expected_records"],
            "evaluation catalog rows")
    _expect(registry["materialized_dir"], data["build"]["canonical_eval_dir"],
            "canonical evaluation directory")
    _expect([row["name"] for row in registry["canonical"]],
            data["evaluation"]["datasets"], "canonical evaluation datasets")
    registry_by_name = {row["name"]: row for row in registry["canonical"]}
    expected_derived_sources = {
        "test": [f"{data['build']['reachable_splits_dir']}/test.parquet"],
        "cctg": [f"{data['build']['reachable_splits_dir']}/external_holdout_org.parquet"],
        "oid_alt": [f"{data['build']['reachable_splits_dir']}/external_holdout_org.parquet"],
        "cdash": [f"{data['build']['reachable_splits_dir']}/external_holdout_refslice.parquet"],
    }
    for name, expected in expected_derived_sources.items():
        _expect(registry_by_name[name]["source"], expected,
                f"canonical evaluation source {name}")
    _expect(pipeline["pipeline"]["catalog"]["out_parquet"], catalog["logical_path"],
            "public builder catalog output")
    _expect(pipeline["pipeline"]["emit"]["out_dir"], data["build"]["reachable_splits_dir"],
            "public builder filtered-splits output")
    _expect(pipeline["pipeline"]["emit"]["splits"],
            data["build"]["reachable_emit_splits"],
            "public builder reachable split coverage")
    _expect(data["build"]["historical_training_populations"], {
        "train": {"rows": 51594, "unique_queries": 47645},
        "val_train": {"rows": 4260, "unique_queries": 3946},
        "val_dev": {"rows": 4234, "unique_queries": 3934},
    }, "historical training populations")

    mpnet = cfg["ft_mpnet"]
    p1 = _winner(phase1["winners"], "model_id", "all-mpnet-base-v2")
    for key, value in {
        "query_variant": mpnet["query_representation"]["id"],
        "recipe": mpnet["candidate_representation"]["recipe"],
        "cde_format": mpnet["candidate_representation"]["format"],
        "loss": mpnet["loss"],
        "lr": mpnet["phase1"]["learning_rate"],
        "temperature": mpnet["phase1"]["temperature"],
        "epochs": mpnet["phase1"]["epochs"],
        "batch_size": mpnet["phase1"]["batch_size"],
        "retained_seed": mpnet["phase1"]["retained_seed"],
        "retained_run_id": mpnet["phase1"]["retained_run_id"],
    }.items():
        _expect(p1[key], value, f"FT-MPNet Phase-1 {key}")
    _expect(mpnet["base_model"], "sentence-transformers/all-mpnet-base-v2",
            "FT-MPNet public base")
    _expect(mpnet["base_revision"], "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "FT-MPNet public base revision")
    _expect(mpnet["training_pair_policy"], {
        "source_split_rows": 51594, "built_pairs": 50972,
        "dropped_empty_query": 0, "dropped_missing_target_text": 622,
        "missing_target_text_policy": "drop",
    }, "FT-MPNet training-pair policy")
    _expect(mpnet["phase1"]["seeds"], phase1["selection_rule"]["required_seeds"],
            "Phase-1 seed pair")
    _expect(mpnet["phase1"]["selection_rule"], phase1["selection_rule"]["rule_id"],
            "Phase-1 selection rule")

    p2 = _winner(phase2["winners"], "model", "all-MPNet")
    for key, value in {
        "negative_strategy": mpnet["phase2"]["negative_strategy"],
        "strategy_rank_band": mpnet["phase2"]["negative_rank_band"],
        "lr": mpnet["phase2"]["learning_rate"],
        "temperature": mpnet["phase2"]["temperature"],
        "epochs": mpnet["phase2"]["epochs"],
        "batch_size": mpnet["phase2"]["batch_size"],
        "encoded_strings_per_optimizer_step": mpnet["phase2"]["encoded_texts_per_batch"],
        "retained_seed": mpnet["phase2"]["retained_seed"],
        "retained_run_id": mpnet["phase2"]["retained_run_id"],
        "retained_checkpoint_hash": mpnet["phase2"]["historical_checkpoint_hash"],
    }.items():
        _expect(p2[key], value, f"FT-MPNet Phase-2 {key}")
    _expect(phase2["selection_rule"]["required_seeds"], mpnet["phase2"]["seeds"],
            "Phase-2 seed pair")
    _expect(phase2["selection_rule"]["configuration_ranking_metrics"],
            mpnet["phase2"]["selection_metrics"], "Phase-2 two-seed selection")
    _expect(protocol["phase2"]["parent_source"], refs["phase1_winners"],
            "Phase-2 parent source")
    for protocol_key, manifest_key in (
        ("mine_split", "mine_split"), ("top_k", "top_k"),
        ("exclude_all_accepted_golds", "exclude_all_accepted_golds"),
        ("dedup_by_cde_id", "deduplicate_by_cde_id"),
    ):
        _expect(protocol["phase2"]["hardneg"][protocol_key],
                mpnet["phase2"]["hard_negative_mining"][manifest_key],
                f"Phase-2 hard-negative {manifest_key}")
    _expect(protocol["phase2"]["hardneg"]["n_negatives"],
            mpnet["phase2"]["negatives_per_anchor"],
            "Phase-2 hard-negative count")

    candidates = cfg["candidate_construction"]
    _expect(candidates["ft_mpnet_top_k"], 20, "candidate FT-MPNet depth")
    _expect(candidates["cde_match_fuzzy_top_k"], 10, "candidate fuzzy depth")
    _expect(candidates["nominal_max_candidates"], 30, "candidate nominal maximum")
    _expect(candidates["deduplicate_by"], "cde_public_id", "candidate deduplication")
    _expect(candidates["inject_evaluation_gold"], False, "candidate gold injection")
    _expect(candidates["fuzzy"]["top_k_per_rule"], 500,
            "CDE Match-Fuzzy per-rule depth")
    _expect(candidates["fuzzy"]["query_id_mask"], "deterministic_hash",
            "CDE Match-Fuzzy query mask")
    _expect(candidates["fuzzy"]["exact_match_scope"], "gold",
            "CDE Match-Fuzzy exact-match scope")
    ce_pool = ce["candidate_pool"]
    _expect(ce_pool["se_k"], candidates["ft_mpnet_top_k"], "CE FT-MPNet depth")
    _expect(ce_pool["kw_k"], candidates["cde_match_fuzzy_top_k"], "CE fuzzy depth")
    _expect(ce_pool["use_clone"], False, "CE clone exclusion")

    expected_allowance = candidates["fuzzy"]["allowance"]
    observed_allowance = {row["name"]: float(row["allowrate"])
                          for row in allowance["datasets"]}
    for split in data["evaluation"]["datasets"]:
        _expect(observed_allowance[split], float(expected_allowance[split]),
                f"allowance {split}")
    _expect(allowance["seed"], candidates["fuzzy"]["exact_match_seed"], "allowance seed")

    med = cfg["ft_medcpt"]
    for key, expected in {
        "base_model": "ncbi/MedCPT-Cross-Encoder",
        "base_revision": "71caf65d4927987813984f54c284405a13fcca49",
        "objective": "pointwise_bce",
        "positive_weight": "negative_to_positive_ratio",
        "epochs": 2,
        "learning_rate": 2.0e-5,
        "batch_size": 32,
        "max_sequence_length": 512,
        "warmup_fraction": 0.1,
        "precision": "fp16",
        "device": "cuda",
        "seed": 20260527,
    }.items():
        _expect(med[key], expected, f"FT-MedCPT {key}")
    _expect(med["query_representation"]["id"], "Q3",
            "FT-MedCPT query representation ID")
    _expect(med["candidate_representation"]["recipe"], "SN_DEC_DEF_PQT_PV",
            "FT-MedCPT candidate representation")
    _expect(ce["winner"]["backbone"], med["base_model"], "FT-MedCPT backbone")
    _expect(ce["winner"]["objective"], med["objective"], "FT-MedCPT objective")
    _expect(ce["query_text_col"], med["query_representation"]["column"],
            "FT-MedCPT query representation")
    _expect(ce["cde_text_recipe"], med["candidate_representation"]["recipe"],
            "FT-MedCPT candidate representation")
    _expect(ce["train_split"], med["train_split"], "FT-MedCPT training split")
    _expect(ce["dev_split"], med["selection_split"], "FT-MedCPT selection split")
    _expect(ce["training"], {
        "objective": med["objective"], "epochs": med["epochs"],
        "learning_rate": med["learning_rate"], "batch_size": med["batch_size"],
        "max_sequence_length": med["max_sequence_length"],
        "warmup_fraction": med["warmup_fraction"],
        "precision": med["precision"], "device": med["device"],
        "seed": med["seed"],
        "positive_weight": med["positive_weight"],
    }, "FT-MedCPT selected training job")
    checkpoint = str(ce["winner"]["checkpoint"]).rstrip("/")
    if not checkpoint.endswith(med["selected_checkpoint_identity"]):
        raise PaperConfigError(
            "FT-MedCPT selected checkpoint identity disagrees with the winner protocol")

    _expect(features["schema_version"], 1, "feature schema version")
    ordered = features["ordered_features"]
    _expect(len(ordered), features["expected_feature_count"], "feature count")
    if len(set(ordered)) != len(ordered):
        raise PaperConfigError("feature contract contains duplicate names")
    _expect(ordered, public_hgbc_feature_schema(), "publicly generatable feature schema")
    _expect(features["categorical_features"], [], "categorical feature set")
    _expect(cfg["hgbc"]["feature_contract"], refs["feature_contract"],
            "HGBC feature contract reference")
    _expect(cfg["hgbc"]["expected_feature_count"], 117, "HGBC feature count")
    _expect(cfg["hgbc"]["hyperparameters"], {
        "max_iter": 200, "max_depth": 3, "learning_rate": 0.05,
        "min_samples_leaf": 30, "random_seed": 42,
    }, "HGBC fixed configuration")
    _expect(cfg["hgbc"]["hyperparameter_selection"], "fixed_final_configuration",
            "HGBC selection mode")

    policy = determinism["policies"]["final_score_tie_break_policy_version"]
    _expect(cfg["ranking"]["final_hgbc"]["policy_id"], policy,
            "final HGBC tie policy")
    if cfg["ranking"]["ft_medcpt"]["policy_id"] == policy:
        raise PaperConfigError("HGBC tie policy must not leak into FT-MedCPT ranking")

    # A public contract may contain artifact identities and relative paths, never
    # a machine-specific checkout dependency.
    serialized = json.dumps(cfg, sort_keys=True)
    for forbidden in ("/data/nextgen2", "/vf/users"):
        if forbidden in serialized:
            raise PaperConfigError(f"paper config contains internal path {forbidden!r}")
    return cfg


def resolved_paper_config(path: str | Path = DEFAULT_PAPER_CONFIG) -> dict[str, Any]:
    """Return the validated configuration plus executable stage job records."""
    cfg = copy.deepcopy(validate_paper_config(path))
    data = cfg["data"]
    mp = cfg["ft_mpnet"]
    pool = cfg["candidate_construction"]
    med = cfg["ft_medcpt"]
    hgbc = cfg["hgbc"]
    common_recipe = {
        "placeholder_policy": "placeholder",
        "short_name_placeholder": "<MISSING_SHORT_NAME>",
        "pv_placeholder": "<MISSING_PV_SUMMARY>",
        "v1": {"filter_numeric_only": True, "filter_versioned_id_short_name": True},
    }
    cfg["resolved_jobs"] = {
        "ft_mpnet_phase1": {
            "runner": "demap_repro.biencoder.engine.finetune_phase1",
            "finetune_phase1": {
                "cde_master_enriched": data["catalog"]["logical_path"],
                "splits_dir": data["build"]["training_splits_dir"],
                "artifacts_dir": "artifacts/paper/final_system_v1/ft_mpnet/phase1",
                "runs_dir": "auto", "base_model_id": mp["base_model"],
                "model_name": mp["base_model"],
                "model_revision": mp["base_revision"],
                "query_variant": mp["query_representation"]["id"],
                "recipe": mp["candidate_representation"]["recipe"],
                "cde_format": mp["candidate_representation"]["format"],
                "loss": mp["loss"], "recipe_configs": common_recipe,
                "train": {
                    "seeds": mp["phase1"]["seeds"], "losses": [mp["loss"]],
                    "lrs": [mp["phase1"]["learning_rate"]],
                    "batch_sizes": [mp["phase1"]["batch_size"]],
                    "temperatures": [mp["phase1"]["temperature"]],
                    "epochs": [mp["phase1"]["epochs"]],
                    "max_seq_length": mp["phase1"]["max_sequence_length"],
                    "fit_api": mp["phase1"]["fit_api"], "bf16": False, "fp16": False,
                    "weight_decay": mp["phase1"]["weight_decay"],
                    "warmup_ratio": mp["phase1"]["warmup_ratio"],
                },
                "eval": {"eval_splits": ["val_dev"], "k_values": [1, 5, 10],
                         "top_k": 100, "normalize_embeddings": True},
            },
            "selection": {"rule": mp["phase1"]["selection_rule"],
                          "retained_seed": mp["phase1"]["retained_seed"],
                          "retained_run_id": mp["phase1"]["retained_run_id"]},
        },
        "ft_mpnet_phase2": {
            "runner": "demap_repro.biencoder.engine.finetune_phase2",
            "finetune_phase2": {
                "cde_master_enriched": data["catalog"]["logical_path"],
                "splits_dir": data["build"]["training_splits_dir"],
                "artifacts_dir": "artifacts/paper/final_system_v1/ft_mpnet/phase2",
                "runs_dir": "auto", "base_model_id": mp["base_model"],
                "init_model_name_or_path": mp["phase1"]["reconstruction_checkpoint"],
                "miner_model_name_or_path": mp["phase1"]["reconstruction_checkpoint"],
                "query_variant": mp["query_representation"]["id"],
                "recipe": mp["candidate_representation"]["recipe"],
                "cde_format": mp["candidate_representation"]["format"],
                "loss": mp["loss"], "recipe_configs": common_recipe,
                "train": {
                    "seeds": mp["phase2"]["seeds"], "losses": [mp["loss"]],
                    "lrs": [mp["phase2"]["learning_rate"]],
                    "batch_sizes": [mp["phase2"]["batch_size"]],
                    "temperatures": [mp["phase2"]["temperature"]],
                    "epochs": [mp["phase2"]["epochs"]],
                    "strategies": [mp["phase2"]["negative_strategy"]],
                    "nneg": mp["phase2"]["negatives_per_anchor"],
                },
                "hard_negative_mining": mp["phase2"]["hard_negative_mining"],
            },
            "selection": {"rule": mp["phase2"]["selection_rule"],
                          "aggregation": mp["phase2"]["selection_aggregation"],
                          "retained_seed": mp["phase2"]["retained_seed"],
                          "parent_run_id": mp["phase2"]["parent_run_id"],
                          "retained_run_id": mp["phase2"]["retained_run_id"],
                          "historical_checkpoint_hash": mp["phase2"]["historical_checkpoint_hash"]},
        },
        "deep_retrieval": {
            "run_dir": mp["phase2"]["reconstruction_run_dir"],
            "cde_master": data["catalog"]["logical_path"],
            "split_groups": {
                "training": {
                    "splits": ["train", "val_train", "val_dev"],
                    "splits_dir": data["build"]["training_splits_dir"],
                    "out_dir": mp["retrieval"]["output"] + "/training",
                },
                "evaluation": {
                    "splits": data["evaluation"]["datasets"],
                    "splits_dir": data["build"]["canonical_eval_dir"],
                    "out_dir": mp["retrieval"]["output"] + "/evaluation",
                },
            },
            "top_k": mp["retrieval"]["full_depth"],
            "embeddings_cache_dir": mp["retrieval"]["embeddings_cache"],
        },
        "cde_match_fuzzy": {
            "splits": data["evaluation"]["datasets"],
            "split_dir": data["build"]["canonical_eval_dir"],
            "training_split_dir": data["build"]["training_splits_dir"],
            "cde_master": data["catalog"]["logical_path"],
            "alt_names": "data/interim/cadsr_merged/cde_alternate_names.parquet",
            "permissible_values": "data/interim/cadsr_merged/cde_permissible_values.parquet",
            "reference_documents": "data/interim/cadsr_merged/cde_reference_documents.parquet",
            "fuzzy_fallback": pool["fuzzy"]["variant"],
            "eligibility": pool["fuzzy"]["eligibility_mode"],
            "top_k_per_rule": pool["fuzzy"]["top_k_per_rule"],
            "exact_match_seed": pool["fuzzy"]["exact_match_seed"],
            "allowance": pool["fuzzy"]["allowance"],
            "out_dir_cadsr": pool["artifacts"]["fuzzy_cadsr_dir"],
            "out_dir_external": pool["artifacts"]["fuzzy_external_dir"],
        },
        "ft_medcpt_pool": {
            "semantic_rankings": mp["retrieval"]["output"] + "/training/biencoder_deep_rankings_top1000.parquet",
            "fuzzy_dir": pool["artifacts"]["fuzzy_cadsr_dir"],
            "fuzzy_candidate_files": {
                "train": "cde_match_clone_candidates_train_fuzzy_a070.parquet",
                "val_dev": "cde_match_clone_candidates_val_dev_fuzzy_a070.parquet",
            },
            "production_catalog": data["catalog"]["logical_path"],
            "full_catalog": "data/processed/cde_master_enriched.parquet",
            "splits_dir": data["build"]["training_splits_dir"],
            "output_dir": "artifacts/paper/final_system_v1/ft_medcpt/pool",
            "optional_eval_pool_summary": None,
            "semantic_k": pool["ft_mpnet_top_k"],
            "fuzzy_k": pool["cde_match_fuzzy_top_k"],
            "deduplicate_by": pool["deduplicate_by"],
            "inject_gold": {"train": True, "val_dev": False},
            "output_file": "pool_se20_kwfuzzy10.parquet",
        },
        "candidate_pool": {
            "splits": ["val_train", "val_dev"] + data["evaluation"]["datasets"],
            "training_splits_dir": data["build"]["training_splits_dir"],
            "evaluation_splits_dir": data["build"]["canonical_eval_dir"],
            "cde_master": data["catalog"]["logical_path"],
            "rankings": [
                mp["retrieval"]["output"] + "/training/biencoder_deep_rankings_top1000.parquet",
                mp["retrieval"]["output"] + "/evaluation/biencoder_deep_rankings_top1000.parquet",
            ],
            "winner_id": "phase2_allmpnet_rep_6f14e0fbed",
            "keyword_index": pool["artifacts"]["keyword_index"],
            "fuzzy_dir_cadsr": pool["artifacts"]["fuzzy_cadsr_dir"],
            "fuzzy_dir_external": pool["artifacts"]["fuzzy_external_dir"],
            "biencoder_k": pool["ft_mpnet_top_k"],
            "keyword_fuzzy_k": pool["cde_match_fuzzy_top_k"],
            "keyword_top_k_per_rule": pool["fuzzy"]["top_k_per_rule"],
            "keyword_allow_rate_cadsr": pool["fuzzy"]["allowance"]["test"],
            "keyword_allow_seed": pool["fuzzy"]["exact_match_seed"],
            "out": pool["artifacts"]["feature_table_base"],
        },
        "ft_medcpt_train": {
            "base_model": med["base_model"], "model_revision": med["base_revision"],
            "objective": med["objective"],
            "query_text_col": med["query_representation"]["column"],
            "cde_text_recipe": med["candidate_representation"]["recipe"],
            "epochs": med["epochs"], "lr": med["learning_rate"],
            "batch_size": med["batch_size"], "max_length": med["max_sequence_length"],
            "warmup_fraction": med["warmup_fraction"],
            "precision": med["precision"], "device": med["device"],
            "positive_weight": med["positive_weight"],
            "seed": med["seed"], "train_split": med["train_split"],
            "dev_split": med["selection_split"],
            "candidate_pool": "artifacts/paper/final_system_v1/ft_medcpt/pool/pool_se20_kwfuzzy10.parquet",
            "train_pairs": "artifacts/paper/final_system_v1/ft_medcpt/pairs/train_SN_DEC_DEF_PQT_PV.parquet",
            "dev_pairs": "artifacts/paper/final_system_v1/ft_medcpt/pairs/dev_SN_DEC_DEF_PQT_PV.parquet",
            "output_dir": med["reconstruction_checkpoint"],
        },
        "ft_medcpt_score": {
            "model": med["reconstruction_checkpoint"],
            "feature_table": pool["artifacts"]["feature_table_base"],
            "cde_master": data["catalog"]["logical_path"],
            "query_text_col": med["query_representation"]["column"],
            "cde_text_recipe": med["candidate_representation"]["recipe"],
            "splits": ["val_train", "val_dev"] + data["evaluation"]["datasets"],
            "out": "artifacts/paper/final_system_v1/ft_medcpt/f5_scores.parquet",
            "ranking_policy": med["ranking_policy"],
        },
        "merge_ft_medcpt_features": {
            "feature_table": pool["artifacts"]["feature_table_base"],
            "crossenc_scores": "artifacts/paper/final_system_v1/ft_medcpt/f5_scores.parquet",
            "out": hgbc["feature_table"],
            "ranking_policy": med["ranking_policy"],
        },
        "hgbc": {
            "feature_table": hgbc["feature_table"], "out": hgbc["output_dir"],
            "feature_contract": hgbc["feature_contract"],
            "features": _load_json(hgbc["feature_contract"])["ordered_features"],
            "hyperparameters": hgbc["hyperparameters"],
            "hyperparameter_selection": hgbc["hyperparameter_selection"],
            "ranking_policy": cfg["ranking"]["final_hgbc"],
        },
        "evaluation": {
            "datasets": data["evaluation"]["datasets"],
            "eval_dir": data["build"]["canonical_eval_dir"],
            "metrics": data["evaluation"]["metrics"],
            "reachable_gold_only": True, "any_accepted_gold_is_hit": True,
        },
    }
    return cfg


def stage_job(stage: str, path: str | Path = DEFAULT_PAPER_CONFIG) -> dict[str, Any]:
    jobs = resolved_paper_config(path)["resolved_jobs"]
    if stage not in jobs:
        raise PaperConfigError(f"unknown paper stage {stage!r}; expected one of {sorted(jobs)}")
    return copy.deepcopy(jobs[stage])


def reject_conflicting_flags(argv: Iterable[str], flags: Iterable[str], *, context: str) -> None:
    """Fail if paper-owned CLI flags were also explicitly supplied."""
    tokens = list(argv)
    conflicts = sorted({flag for flag in flags
                        if any(tok == flag or tok.startswith(flag + "=") for tok in tokens)})
    if conflicts:
        raise PaperConfigError(
            f"{context}: paper mode owns {conflicts}; remove the conflicting override(s)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="demap paper-config", description=__doc__)
    ap.add_argument("--config", default=DEFAULT_PAPER_CONFIG)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--show-resolved", action="store_true")
    group.add_argument("--dry-run", choices=[
        "all", "ft_mpnet_phase1", "ft_mpnet_phase2", "deep_retrieval",
        "cde_match_fuzzy", "ft_medcpt_pool", "candidate_pool", "ft_medcpt_train",
        "ft_medcpt_score", "merge_ft_medcpt_features", "hgbc", "evaluation",
    ])
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args(argv)
    try:
        resolved = resolved_paper_config(args.config)
    except PaperConfigError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    if args.validate:
        payload: Any = {"status": "PASS", "manifest_id": resolved["manifest_id"],
                        "feature_count": resolved["hgbc"]["expected_feature_count"],
                        "resolved_stages": sorted(resolved["resolved_jobs"])}
    elif args.show_resolved:
        payload = resolved
    elif args.dry_run == "all":
        payload = resolved["resolved_jobs"]
    else:
        payload = resolved["resolved_jobs"][args.dry_run]
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
