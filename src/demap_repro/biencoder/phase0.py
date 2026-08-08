"""Canonical Phase 0 grid validation (fail-closed). Does NOT rerun Phase 0."""

from __future__ import annotations

from typing import Any, Dict, List

from demap_repro.biencoder import protocol as _proto

# The required corrected recipe set (order-independent membership check).
REQUIRED_RECIPES = {
    "v3", "v2", "v1_v3", "v1_v2a_v2b_v3", "v1_v2a_v2b_v5", "v1_v2a_v3_v5",
    "v1_v2a_v2b_v3_v5", "v1_v6_v2b_v3_v5", "v1_v2a_v2b_v3_v4_v5", "v1_v2a_v6_v2b_v3_v5",
}
# A known legacy grid signature (the June-30/old July sweep) that must be rejected.
LEGACY_JUNE30_RECIPES = {"v3", "v1_v3", "v1_v2b_v3", "v1_v6_v2b_v3", "v1_v6_v2b_v3_v5"}


class Phase0Error(SystemExit):
    pass


def assert_not_legacy_grid(recipes: List[str]) -> None:
    got = set(recipes)
    if got != REQUIRED_RECIPES:
        missing = REQUIRED_RECIPES - got
        extra = got - REQUIRED_RECIPES
        raise Phase0Error(
            f"[phase0] recipe grid is not the corrected required set. missing={sorted(missing)} extra={sorted(extra)}"
        )
    if got & LEGACY_JUNE30_RECIPES == got and got != REQUIRED_RECIPES:  # pragma: no cover (covered above)
        raise Phase0Error("[phase0] refusing the legacy June-30 recipe grid")


def figure_provenance(proto: Dict[str, Any]) -> Dict[str, Any]:
    """Provenance fields a Phase 0 figure/table MUST agree with (title/caption/model/source/split/catalog)."""
    d = proto["data"]
    return {
        "split": d["tuning_split"],
        "val_dev_denominator": d["val_dev_denominator"],
        "production_catalog": d["production_catalog"],
        "production_catalog_rows": d["production_catalog_rows"],
        "cde_format": d["cde_format"],
        "max_seq_length_policy": proto["phase0"].get("max_seq_length_policy", "checkpoint_native_with_loader_floor_256"),
        "checkpoint_native_max_seq_length_by_model": proto["phase0"].get("checkpoint_native_max_seq_length_by_model", {}),
        "architectural_max_position_embeddings_by_model": proto["phase0"].get("architectural_max_position_embeddings_by_model", {}),
        "effective_max_seq_length_by_model": proto["phase0"].get("effective_max_seq_length_by_model", {}),
        "max_seq_length_note": "Phase 0 effective = max(checkpoint_native, 256) via the repo loader floor: "
                               "all-MPNet 384 (native 384), BioSimCSE 256 (native 128, RAISED), PubMedBERT 512 "
                               "(native 512). The 256 lock applies to Phase 1 fine-tuning only.",
        "off_the_shelf": proto["phase0"]["off_the_shelf"],
        "models": proto["phase0"]["models"],
        "no_legacy_val_split": True,
        "no_manual_model_name_substitution": True,
    }


def validate_phase0(proto: Dict[str, Any], *, check_rows: bool = True) -> Dict[str, Any]:
    # Structural validation already ran in load_protocol; re-assert grid identity + legacy rejection.
    recipes = [c["recipe"] for c in proto["cde_representations"]]
    assert_not_legacy_grid(recipes)

    cells = _proto.phase0_cells(proto)  # 40 ordered cells, dedup-checked
    anchor = proto["anchor"]
    anchor_in = any(qid == anchor["query_id"] and recipe == anchor["recipe"] for (_m, qid, _qn, recipe, _cn) in cells)
    if not anchor_in:
        raise Phase0Error("[phase0] anchor cell not present in the grid")

    fps = _proto.verify_fingerprints(proto, check_rows=check_rows)
    models = proto["phase0"]["models"]

    return {
        "n_models": len(models),
        "n_query_representations": len(proto["query_representations"]),
        "n_cde_representations": len(proto["cde_representations"]),
        "n_cells_per_model": len(cells),
        "n_total_cells": len(cells) * len(models),
        "query_order": [(q["id"], q["name"]) for q in proto["query_representations"]],
        "recipe_order": [(c["order"], c["name"], c["recipe"]) for c in proto["cde_representations"]],
        "anchor": anchor,
        "fingerprints": fps,
        "figure_provenance": figure_provenance(proto),
        "eval_splits": proto["phase0"]["eval_splits"],
    }
