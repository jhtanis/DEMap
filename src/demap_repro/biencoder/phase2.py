"""Canonical Phase 2 config generation with Phase 1 winner propagation (fail-closed).

Phase 2 initialises from the RETAINED checkpoint of the canonical Phase 1 winner manifest
(``configs/paper/phase1_winners_for_phase2_v1.yaml``) and derives EXCLUSIVELY from it:

    parent checkpoint · query representation · CDE recipe/fields · CDE format ·
    inherited loss · Phase 1 winning lr and temperature (which seed the Phase 2 grids) ·
    retained seed · all source hashes

There is no base config supplying competing defaults. Any attempt to override a
winner-derived field fails unless the explicit research-only ``allow_winner_override`` is
set (which marks the output non-canonical and records the deviation).

Author-approved locks (see ``configs/paper/biencoder_protocol_v1.yaml`` ``phase2:``):
  * precision fp32 — no autocast, no GradScaler, FP32 parameters;
  * gradient_accumulation_steps 1 (the historical unused value 4 is rejected);
  * physical batch 11 examples; `none` encodes 2 texts/example (<=22 strings/batch),
    `hard_top25`/`semihard_1_50` encode 12 texts/example (<=132 strings/batch);
  * val_dev is the ONLY split evaluated during tuning;
  * `hardcurr_1_50__1_25__1_15` is excluded from the paper route.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from demap_repro.biencoder import hardneg as _hn
from demap_repro.biencoder import phase1_results as _p1r
from demap_repro.biencoder import protocol as _proto
from demap_repro.biencoder import quarantine as _q

SUBMISSION_TOKEN_NAME = "SUBMISSION_TOKEN.json"
DEFAULT_OUT_ROOT = "artifacts/phase2_canonical_v1"
DEFAULT_PHASE1_MANIFEST = "configs/paper/phase1_winners_for_phase2_v1.yaml"

# Fields owned by the Phase 1 winner manifest; a base config must NOT set them.
WINNER_OWNED_FIELDS = (
    "init_model_name_or_path", "miner_model_name_or_path",
    "query_variant", "recipe", "cde_format", "loss",
)

# Only fp32 is honest on this path (see protocol phase2.allowed_precisions).
_PRECISION = {"fp32": {"fp16": False, "bf16": False, "precision": "fp32"}}


class Phase2Error(SystemExit):
    pass


def _slug(s: str) -> str:
    return str(s).replace("/", "__")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _query_col_for(proto: Dict[str, Any], query_id: str) -> str:
    for q in proto["query_representations"]:
        if q["id"] == query_id:
            return q["column"]
    raise Phase2Error(f"[phase2] query id {query_id!r} not in protocol grid")


def _strategy_spec(proto: Dict[str, Any], strategy_id: str) -> Dict[str, Any]:
    for s in proto["phase2"]["strategies"]:
        if s["id"] == strategy_id:
            return s
    raise Phase2Error(f"[phase2] strategy {strategy_id!r} is not in the approved allow-list")


def strategy_batch_size(proto: Dict[str, Any], spec: Dict[str, Any]) -> int:
    """Physical examples per optimizer step for one strategy.

    The comparison across negative strategies is held fixed on ENCODED STRINGS per
    optimizer step, not on examples: an example costs ``texts_per_example`` encoded
    strings (2 with no explicit negatives, 12 with ten). A strategy may therefore
    declare its own ``batch_size``; strategies that do not fall back to the global
    ``phase2.batch_size``. The product must equal the strategy's declared
    ``max_texts_per_batch`` — asserted here so a mis-specified protocol cannot reach
    the GPUs.
    """
    bs = int(spec.get("batch_size", proto["phase2"]["batch_size"]))
    encoded = bs * int(spec["texts_per_example"])
    declared = int(spec["max_texts_per_batch"])
    if encoded != declared:
        raise Phase2Error(
            f"[phase2] strategy {spec['id']!r}: batch_size {bs} x texts_per_example "
            f"{spec['texts_per_example']} = {encoded} encoded strings per optimizer step, "
            f"but the protocol declares max_texts_per_batch={declared}")
    return bs


def assert_encoded_text_budget_matched(proto: Dict[str, Any]) -> int:
    """Every approved strategy must encode the SAME number of strings per step.

    Returns that common budget. This is the fairness invariant of the Phase 2 grid:
    `none` (2 strings/example) needs a larger physical batch than the hard-negative
    arms (12 strings/example) to spend an equal encoder budget per optimizer step.
    """
    budgets = {}
    for spec in proto["phase2"]["strategies"]:
        bs = strategy_batch_size(proto, spec)
        budgets[spec["id"]] = bs * int(spec["texts_per_example"])
    distinct = set(budgets.values())
    if len(distinct) != 1:
        raise Phase2Error(
            f"[phase2] encoded-string budget differs across strategies: {budgets}; "
            "the strategy comparison would be confounded by encoder compute")
    return distinct.pop()


def derive_lr_grid(proto: Dict[str, Any], phase1_lr: float) -> List[float]:
    table = {float(k): [float(x) for x in v] for k, v in proto["phase2"]["lr_by_phase1_lr"].items()}
    if float(phase1_lr) not in table:
        raise Phase2Error(
            f"[phase2] no approved Phase 2 learning-rate grid for Phase 1 lr {phase1_lr!r}; "
            f"approved keys: {sorted(table)}")
    return table[float(phase1_lr)]


def derive_temperature_grid(proto: Dict[str, Any], phase1_temp: float) -> List[float]:
    table = {float(k): [float(x) for x in v]
             for k, v in proto["phase2"]["temperature_by_phase1_temperature"].items()}
    if float(phase1_temp) not in table:
        raise Phase2Error(
            f"[phase2] no approved Phase 2 temperature grid for Phase 1 temperature "
            f"{phase1_temp!r}; approved keys: {sorted(table)}")
    return table[float(phase1_temp)]


def _assert_strategies(proto: Dict[str, Any], strategies: List[str]) -> None:
    allowed = [s["id"] for s in proto["phase2"]["strategies"]]
    excluded = list(proto["phase2"].get("excluded_strategies") or [])
    for s in strategies:
        if s in excluded:
            raise Phase2Error(
                f"[phase2] strategy {s!r} is explicitly EXCLUDED from the canonical paper route")
        if s not in allowed:
            raise Phase2Error(f"[phase2] strategy {s!r} not in the approved allow-list {allowed}")


def generate_phase2(
    *,
    protocol_path: str,
    phase1_manifest_path: str,
    model_ids: List[str],
    precision: str = "fp32",
    out_root: str = DEFAULT_OUT_ROOT,
    hardneg_root: Optional[str] = None,
    allow_winner_override: bool = False,
    allow_legacy: bool = False,
    base_overrides: Optional[Dict[str, Any]] = None,
    code_commit: str = "unknown",
    dirty: bool = False,
    check_rows: bool = True,
    write: bool = True,
    require_hardneg: bool = False,
) -> Dict[str, Any]:
    """Generate the canonical Phase 2 grid. Writes nothing unless ``write``."""
    if precision not in _PRECISION:
        raise Phase2Error(
            f"[phase2] precision {precision!r} not supported; canonical Phase 2 is fp32 "
            f"(no autocast, no GradScaler)")

    proto = _proto.load_protocol(protocol_path)
    _q.assert_protocol_not_legacy(proto, allow_legacy=allow_legacy)
    p2 = proto.get("phase2")
    if not p2:
        raise Phase2Error("[phase2] protocol has no phase2: block")
    if precision not in p2["allowed_precisions"]:
        raise Phase2Error(f"[phase2] precision {precision!r} not allowed by protocol")
    if p2.get("autocast") is not False or p2.get("grad_scaler") is not False:
        raise Phase2Error("[phase2] protocol must lock autocast=false and grad_scaler=false")
    if int(p2["gradient_accumulation_steps"]) != 1:
        raise Phase2Error(
            f"[phase2] gradient_accumulation_steps must be 1, got "
            f"{p2['gradient_accumulation_steps']}")
    if list(p2["eval_splits"]) != ["val_dev"]:
        raise Phase2Error(f"[phase2] eval_splits must be exactly ['val_dev'], got {p2['eval_splits']}")

    fps = _proto.verify_fingerprints(proto, check_rows=check_rows)
    proto_sha = _proto.sha256_file(protocol_path)

    # ---- Phase 1 winner manifest is the ONLY source of parent/representation/loss.
    man = _p1r.load_manifest(phase1_manifest_path)
    man_sha = _p1r.sha256_file(phase1_manifest_path)
    if man.get("manifest_id") not in _p1r.ACCEPTED_MANIFEST_IDS:
        raise Phase2Error(
            f"[phase2] {phase1_manifest_path}: manifest_id {man.get('manifest_id')!r} is not "
            f"an approved Phase 1 parent source {sorted(_p1r.ACCEPTED_MANIFEST_IDS)!r}")
    if man.get("selection_rule_id") != _p1r.SELECTION_RULE_ID:
        raise Phase2Error(
            f"[phase2] Phase 1 manifest was not produced by the canonical seed-mean rule")

    base_overrides = dict(base_overrides or {})
    conflicting = [f for f in WINNER_OWNED_FIELDS if f in base_overrides]
    if conflicting and not allow_winner_override:
        raise Phase2Error(
            f"[phase2] base config attempts to set winner-owned fields {conflicting}; these come "
            f"ONLY from the Phase 1 winner manifest. Use allow_winner_override for research runs.")

    d = proto["data"]
    strategies = [s["id"] for s in p2["strategies"]]
    _assert_strategies(proto, strategies)
    hn_root = hardneg_root or p2["hardneg"]["artifact_root"]

    out_root_p = Path(out_root)
    summary: Dict[str, Any] = {
        "protocol_path": protocol_path, "protocol_sha256": proto_sha,
        "phase1_manifest_path": phase1_manifest_path, "phase1_manifest_sha256": man_sha,
        "phase1_selection_rule_id": man.get("selection_rule_id"),
        "precision_requested": precision,
        "autocast": False, "grad_scaler": False, "param_dtype": "torch.float32",
        "gradient_accumulation_steps": 1,
        "out_root": str(out_root_p), "hardneg_root": hn_root,
        "runs_per_model": int(p2["runs_per_model"]),
        "canonical": not (allow_winner_override or allow_legacy),
        "models": {}, "total_runs": 0,
    }
    manifest_rows: List[Dict[str, Any]] = []

    for model_id in model_ids:
        w = _winner_for(man, model_id)
        if allow_winner_override and base_overrides:
            w = dict(w)
            for f in ("query_variant", "recipe", "cde_format", "loss"):
                if f in base_overrides:
                    w[f] = base_overrides[f]

        parent = _p1r.resolve_parent_checkpoint(man, model_id)  # raises if missing on disk
        parent_hash = _hn.checkpoint_hash(parent)

        exp_col = _query_col_for(proto, w["query_variant"])
        if w["cde_format"] != d["cde_format"]:
            raise Phase2Error(
                f"[phase2] {model_id}: cde_format {w['cde_format']} != protocol {d['cde_format']}")
        from demap_repro.biencoder.phase1 import recipe_fields
        fields = recipe_fields(w["recipe"])

        lrs = derive_lr_grid(proto, float(w["lr"]))
        temps = derive_temperature_grid(proto, float(w["temperature"]))
        grid = list(itertools.product(lrs, temps, strategies, p2["epochs"], p2["seeds"]))
        if len(grid) != int(p2["runs_per_model"]):
            raise Phase2Error(
                f"[phase2] {model_id}: grid has {len(grid)} runs but protocol declares "
                f"{p2['runs_per_model']}")

        model_slug = _slug(model_id)
        model_dir = out_root_p / model_slug
        gen_dir = model_dir / "generated_configs"

        # Parent-bound hard-negative artifact identity (one per model; shared across seeds).
        hn_identity = _hn.build_identity(
            model_id=model_id, parent_checkpoint=parent, parent_checkpoint_sha256=parent_hash,
            query_variant=w["query_variant"], recipe=w["recipe"], cde_format=w["cde_format"],
            train_split_sha256=fps["train"]["sha256"], catalog_sha256=fps["catalog"]["sha256"],
            top_k=int(p2["hardneg"]["top_k"]), n_negatives=int(p2["hardneg"]["n_negatives"]),
            mine_split=str(p2["hardneg"]["mine_split"]), code_commit=code_commit,
        )
        hn_dir = str(Path(hn_root) / hn_identity["artifact_key"])
        hn_parquet = str(Path(hn_dir) / f"{p2['hardneg']['mine_split']}_topk{p2['hardneg']['top_k']}.parquet")
        hn_present = Path(hn_parquet).exists()
        if require_hardneg and not hn_present:
            raise Phase2Error(
                f"[phase2] {model_id}: required hard-negative artifact missing: {hn_parquet}\n"
                f"  mine it first with: demap paper biencoder --stage phase2 --mine-hardneg")

        pv = _PRECISION[precision]
        prov_common = {
            "phase0_protocol_path": protocol_path, "phase0_protocol_sha256": proto_sha,
            "phase1_winner_manifest_path": phase1_manifest_path,
            "phase1_winner_manifest_sha256": man_sha,
            "phase1_selection_rule_id": man.get("selection_rule_id"),
            "phase1_model_id": model_id,
            "phase1_retained_seed": w["retained_seed"],
            "phase1_retained_run_id": w.get("retained_run_id"),
            "phase1_winning_lr": float(w["lr"]),
            "phase1_winning_temperature": float(w["temperature"]),
            "phase1_recall5_mean": w.get("recall5_mean"),
            "section4_3_lineage": {
                "query_variant": w["query_variant"], "recipe": w["recipe"],
                "cde_format": w["cde_format"], "loss": w["loss"],
            },
            "parent_checkpoint": parent,
            "parent_checkpoint_sha256": parent_hash,
            "inherited_loss": w["loss"],
            "resolved_cde_fields": fields,
            "requested_precision": precision,
            "autocast": False, "grad_scaler": False, "param_dtype": "torch.float32",
            "batch_size": int(p2["batch_size"]),
            "gradient_accumulation_steps": 1,
            "effective_optimizer_batch_size": int(p2["batch_size"]),
            "grad_accum_note": "gradient accumulation is 1; it would NOT enlarge the physical "
                               "negative pool in any case",
            "no_duplicate_sampler": "NoDuplicateTextBatchSampler on anchor+positive",
            "similarity_fn": p2["similarity_fn"],
            "max_seq_length": d["max_seq_length"],
            "weight_decay": p2["weight_decay"], "optimizer": p2["optimizer"],
            "scheduler": p2["scheduler"], "warmup_ratio": p2["warmup_ratio"],
            "train_sha256": fps["train"]["sha256"],
            "val_dev_sha256": fps["val_dev"]["sha256"],
            "production_catalog": d["production_catalog"],
            "production_catalog_sha256": fps["catalog"]["sha256"],
            "production_catalog_rows": d["production_catalog_rows"],
            "val_dev_denominator": d["val_dev_denominator"],
            "eval_splits": ["val_dev"],
            "checkpoint_selection_metric": p2["checkpoint_selection_metric"],
            "hardneg_identity": hn_identity,
            "hardneg_dir": hn_dir,
            "code_commit": code_commit, "dirty_tree": bool(dirty),
            "is_canonical_paper_run": not (allow_winner_override or allow_legacy),
            "protocol_deviation": bool(allow_winner_override),
        }

        model_runs = []
        if write:
            gen_dir.mkdir(parents=True, exist_ok=True)
        for (lr, temp, strat, ep, seed) in grid:
            spec = _strategy_spec(proto, strat)
            run_bs = strategy_batch_size(proto, spec)
            run_key = f"lr{lr:g}__t{temp:g}__{strat}__ep{ep}__seed{seed}"
            uses_negs = int(spec["n_negatives"]) > 0
            cfg = {
                "finetune_phase2": {
                    "cde_master_enriched": d["production_catalog"],
                    "splits_dir": d["splits_dir"],
                    "artifacts_dir": str(model_dir),
                    "runs_dir": "auto",
                    "base_model_id": model_id,
                    "init_model_name_or_path": parent,
                    "miner_model_name_or_path": parent,
                    "mined_parquet": hn_parquet if uses_negs else None,
                    "query_variant": w["query_variant"],
                    "recipe": w["recipe"],
                    "cde_format": w["cde_format"],
                    "rerank_mode": "R0",
                    "sep": " | ",
                    "recipe_configs": {
                        "placeholder_policy": "placeholder",
                        "short_name_placeholder": "<MISSING_SHORT_NAME>",
                        "pv_placeholder": "<MISSING_PV_SUMMARY>",
                        "v1": {"filter_numeric_only": True, "filter_versioned_id_short_name": True},
                    },
                    "hardneg": {
                        "top_k": int(p2["hardneg"]["top_k"]),
                        "nneg": int(spec["n_negatives"]),
                        "strategies": [strat],
                        "hard_band": "1-25",
                        "semihard_band": "1-50",
                    },
                    "train": {
                        "device": "auto",
                        "max_seq_length": d["max_seq_length"],
                        "warmup_ratio": p2["warmup_ratio"],
                        "weight_decay": p2["weight_decay"],
                        "fp16": pv["fp16"], "bf16": pv["bf16"], "precision": pv["precision"],
                        "gradient_accumulation_steps": 1,
                        "seeds": [int(seed)],
                        "losses": [w["loss"]],
                        "lrs": [float(lr)],
                        "batch_sizes": [run_bs],
                        "temperatures": [float(temp)],
                        "epochs": [int(ep)],
                    },
                    "eval": {
                        "top_k": 100, "k_values": [5, 10, 20], "output_top_k": 20,
                        "block_size": 2048, "encode_batch_size": 64,
                        "normalize_embeddings": True,
                        "eval_splits": ["val_dev"],
                    },
                    "stage_tag": model_slug,
                    "hybrid_alpha": 0.5,
                    "paper_provenance": {
                        **prov_common, "run_key": run_key,
                        "lr": float(lr), "temperature": float(temp), "epochs": int(ep),
                        "seed": int(seed), "negative_strategy": strat,
                        "n_negatives_per_example": int(spec["n_negatives"]),
                        "texts_per_example": int(spec["texts_per_example"]),
                        "max_texts_per_physical_batch": int(spec["max_texts_per_batch"]),
                        "strategy_rank_band": spec["rank_band"],
                        "batch_size": run_bs,
                        "encoded_strings_per_optimizer_step": run_bs * int(spec["texts_per_example"]),
                        "mined_parquet": hn_parquet if uses_negs else None,
                        "uses_mined_negatives": uses_negs,
                    },
                }
            }
            cfg_bytes = yaml.safe_dump(cfg, sort_keys=False).encode()
            cfg["finetune_phase2"]["paper_provenance"]["generated_config_sha256"] = _sha256_bytes(cfg_bytes)
            cfg_path = gen_dir / f"{model_slug}__{run_key}__{pv['precision']}.yaml"
            if write:
                if cfg_path.exists():
                    existing = yaml.safe_load(open(cfg_path))
                    ex = (((existing or {}).get("finetune_phase2") or {}).get("paper_provenance") or {})
                    if ex.get("generated_config_sha256") != \
                            cfg["finetune_phase2"]["paper_provenance"]["generated_config_sha256"]:
                        raise Phase2Error(
                            f"[phase2] refuse to overwrite incompatible existing config: {cfg_path}")
                with open(cfg_path, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
            row = {
                "model_id": model_id, "config_path": str(cfg_path), "run_key": run_key,
                "lr": float(lr), "temperature": float(temp), "strategy": strat,
                "epochs": int(ep), "seed": int(seed),
                "query_id": w["query_variant"], "recipe": w["recipe"], "loss": w["loss"],
                "precision": precision, "batch_size": run_bs,
                "texts_per_example": int(spec["texts_per_example"]),
                "max_texts_per_batch": int(spec["max_texts_per_batch"]),
                "parent_checkpoint_sha256": parent_hash,
                "hardneg_artifact_key": hn_identity["artifact_key"] if uses_negs else "",
            }
            model_runs.append(row)
            manifest_rows.append(row)

        if write:
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / "resolved_parent_and_lineage.json", "w") as f:
                json.dump({"model_id": model_id, "provenance": prov_common,
                           "lr_grid": lrs, "temperature_grid": temps,
                           "strategies": strategies}, f, indent=1)
        summary["models"][model_id] = {
            "parent_checkpoint": parent, "parent_checkpoint_sha256": parent_hash,
            "inherited_loss": w["loss"], "query_variant": w["query_variant"],
            "recipe": w["recipe"], "cde_format": w["cde_format"],
            "phase1_lr": float(w["lr"]), "phase1_temperature": float(w["temperature"]),
            "lr_grid": lrs, "temperature_grid": temps, "strategies": strategies,
            "n_runs": len(model_runs), "generated_dir": str(gen_dir),
            "hardneg_dir": hn_dir, "hardneg_parquet": hn_parquet,
            "hardneg_present": hn_present,
            "hardneg_artifact_key": hn_identity["artifact_key"],
        }
        summary["total_runs"] += len(model_runs)

    if write:
        out_root_p.mkdir(parents=True, exist_ok=True)
        cols = ["model_id", "config_path", "run_key", "lr", "temperature", "strategy", "epochs",
                "seed", "query_id", "recipe", "loss", "precision", "batch_size",
                "texts_per_example", "max_texts_per_batch", "parent_checkpoint_sha256",
                "hardneg_artifact_key"]
        with open(out_root_p / "phase2_preview_manifest.tsv", "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in manifest_rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        with open(out_root_p / "phase2_preview_summary.json", "w") as f:
            json.dump(summary, f, indent=1)
    return summary


def _winner_for(man: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    for w in man.get("winners") or []:
        if w.get("model_id") == model_id:
            return w
    raise Phase2Error(
        f"[phase2] model {model_id!r} not in Phase 1 winner manifest; available: "
        f"{[w.get('model_id') for w in man.get('winners') or []]}")


# ----------------------------------------------------------- submission validation + token
def validate_generated_submission(
    *, out_root: str, protocol_path: str, phase1_manifest_path: str, model_ids: List[str],
    precision: str = "fp32", expected_total: Optional[int] = None,
    expected_per_model: Optional[int] = None, require_hardneg: bool = True,
) -> Dict[str, Any]:
    """Independently re-verify a generated Phase 2 submission (used by --submit AND sbatch).

    Fail-closed on: protocol/manifest hashes, row counts, per-model counts, seeds {0,1} per
    configuration (strategy INCLUDED in the identity), inherited loss, parent checkpoint
    existence + hash, strategy allow-list, batching semantics, fp32, msl, grad-accum 1,
    val_dev-only evaluation, and hard-negative artifact presence + parent binding.
    """
    proto = _proto.load_protocol(protocol_path)
    p2 = proto["phase2"]
    # Run counts come from the protocol and the requested models, not a hard-coded
    # two-model assumption, so a single-model submission validates as strictly as a
    # multi-model one.
    if expected_per_model is None:
        expected_per_model = int(p2["runs_per_model"])
    if expected_total is None:
        expected_total = expected_per_model * len(model_ids)
    man = _p1r.load_manifest(phase1_manifest_path)
    proto_sha = _proto.sha256_file(protocol_path)
    man_sha = _p1r.sha256_file(phase1_manifest_path)
    pv = _PRECISION[precision]

    root = Path(out_root)
    manifest = root / "phase2_preview_manifest.tsv"
    if not manifest.exists():
        raise Phase2Error(f"[submit] preview manifest missing: {manifest}")
    lines = manifest.read_text().splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, l.split("\t"))) for l in lines[1:] if l.strip()]
    if len(rows) != expected_total:
        raise Phase2Error(f"[submit] manifest has {len(rows)} rows, expected {expected_total}")

    allowed = [s["id"] for s in p2["strategies"]]
    excluded = list(p2.get("excluded_strategies") or [])
    per_model: Dict[str, int] = {}
    seeds_by_cfg: Dict[tuple, set] = {}
    hardneg_keys: Dict[str, str] = {}

    for r in rows:
        mid = r["model_id"]
        w = _winner_for(man, mid)
        per_model[mid] = per_model.get(mid, 0) + 1
        # NOTE: strategy is part of the Phase 2 configuration identity.
        seeds_by_cfg.setdefault((mid, r["lr"], r["temperature"], r["strategy"], r["epochs"]),
                                set()).add(int(r["seed"]))
        if r["strategy"] in excluded:
            raise Phase2Error(f"[submit] excluded strategy {r['strategy']!r} present in manifest")
        if r["strategy"] not in allowed:
            raise Phase2Error(f"[submit] strategy {r['strategy']!r} not in allow-list {allowed}")

        c = yaml.safe_load(open(r["config_path"]))["finetune_phase2"]
        prov = c["paper_provenance"]
        if c["query_variant"] != w["query_variant"] or c["recipe"] != w["recipe"]:
            raise Phase2Error(f"[submit] {r['config_path']}: representation != Phase 1 winner")
        if c["train"]["losses"] != [w["loss"]]:
            raise Phase2Error(
                f"[submit] {r['config_path']}: loss {c['train']['losses']} != inherited "
                f"[{w['loss']}]")
        if c["init_model_name_or_path"] != w["retained_checkpoint"]:
            raise Phase2Error(
                f"[submit] {r['config_path']}: parent != Phase 1 retained checkpoint")
        if not Path(c["init_model_name_or_path"]).is_dir():
            raise Phase2Error(
                f"[submit] {r['config_path']}: parent checkpoint missing on disk: "
                f"{c['init_model_name_or_path']}")
        got_hash = _hn.checkpoint_hash(c["init_model_name_or_path"])
        if got_hash != prov["parent_checkpoint_sha256"]:
            raise Phase2Error(
                f"[submit] {r['config_path']}: STALE PARENT — checkpoint hash {got_hash} != "
                f"recorded {prov['parent_checkpoint_sha256']}")
        if (c["train"]["fp16"], c["train"]["bf16"], c["train"]["precision"]) != \
                (pv["fp16"], pv["bf16"], pv["precision"]):
            raise Phase2Error(f"[submit] {r['config_path']}: precision != {precision}")
        if int(c["train"]["gradient_accumulation_steps"]) != 1:
            raise Phase2Error(f"[submit] {r['config_path']}: gradient_accumulation_steps != 1")
        if int(c["train"]["max_seq_length"]) != 256:
            raise Phase2Error(f"[submit] {r['config_path']}: max_seq_length != 256")
        _spec_bs = strategy_batch_size(proto, _strategy_spec(proto, r["strategy"]))
        if int(c["train"]["batch_sizes"][0]) != _spec_bs:
            raise Phase2Error(
                f"[submit] {r['config_path']}: batch_size {c['train']['batch_sizes'][0]} != "
                f"{_spec_bs} required for strategy {r['strategy']!r}")
        if list(c["eval"]["eval_splits"]) != ["val_dev"]:
            raise Phase2Error(
                f"[submit] {r['config_path']}: eval_splits {c['eval']['eval_splits']} != ['val_dev'] "
                f"(test/val_train must not be evaluated during tuning)")
        spec = _strategy_spec(proto, r["strategy"])
        if int(prov["texts_per_example"]) != int(spec["texts_per_example"]) or \
                int(prov["max_texts_per_physical_batch"]) != int(spec["max_texts_per_batch"]):
            raise Phase2Error(f"[submit] {r['config_path']}: batching semantics misrecorded")
        if prov["phase1_winner_manifest_sha256"] != man_sha:
            raise Phase2Error(f"[submit] {r['config_path']}: stale Phase 1 manifest hash")
        if prov["phase0_protocol_sha256"] != proto_sha:
            raise Phase2Error(f"[submit] {r['config_path']}: stale protocol hash")

        # hard-negative artifact must exist and be bound to THIS parent
        if int(spec["n_negatives"]) > 0:
            if require_hardneg:
                _hn.assert_artifact_matches(
                    prov["mined_parquet"], identity=prov["hardneg_identity"])
            hardneg_keys[mid] = prov["hardneg_identity"]["artifact_key"]
        elif c["mined_parquet"] is not None:
            raise Phase2Error(
                f"[submit] {r['config_path']}: strategy 'none' must not reference a mined parquet")

    for mid in model_ids:
        if per_model.get(mid) != expected_per_model:
            raise Phase2Error(
                f"[submit] model {mid} has {per_model.get(mid)} runs, expected {expected_per_model}")
    for key, seeds in seeds_by_cfg.items():
        if seeds != {0, 1}:
            raise Phase2Error(f"[submit] config {key} seeds {sorted(seeds)} != required {{0,1}}")
    if len(seeds_by_cfg) != expected_total // 2:
        raise Phase2Error(
            f"[submit] {len(seeds_by_cfg)} configurations, expected {expected_total // 2}")

    return {
        "schema": "phase2_submission_token_v1",
        "protocol_path": protocol_path, "protocol_sha256": proto_sha,
        "phase1_manifest_path": phase1_manifest_path, "phase1_manifest_sha256": man_sha,
        "out_root": str(root), "precision": precision,
        "total_runs": len(rows), "per_model": per_model,
        "n_configs": len(seeds_by_cfg), "seeds_ok": True,
        "strategies": allowed,
        "batch_size_by_strategy": {
            s: strategy_batch_size(proto, _strategy_spec(proto, s)) for s in allowed},
        "encoded_strings_per_optimizer_step": assert_encoded_text_budget_matched(proto),
        "gradient_accumulation_steps": 1,
        "hardneg_artifact_keys": hardneg_keys,
    }


def write_submission_token(out_root: str, token: Dict[str, Any], *, code_commit: str) -> str:
    tok = dict(token)
    tok["code_commit"] = code_commit
    body = json.dumps({k: v for k, v in sorted(tok.items())}, sort_keys=True).encode()
    tok["token_sha256"] = _sha256_bytes(body)
    p = Path(out_root) / SUBMISSION_TOKEN_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(tok, f, indent=1, sort_keys=True)
    return str(p)


def verify_submission_token(out_root: str, *, expected_commit: str) -> Dict[str, Any]:
    """Used by the array launcher: refuse to run without a valid, commit-bound token."""
    p = Path(out_root) / SUBMISSION_TOKEN_NAME
    if not p.exists():
        raise Phase2Error(
            f"[submit-guard] no submission token at {p}. A direct sbatch is REFUSED; generate via "
            f"`demap paper biencoder --stage phase2 --submit`.")
    tok = json.load(open(p))
    if tok.get("schema") != "phase2_submission_token_v1":
        raise Phase2Error(f"[submit-guard] {p}: wrong token schema {tok.get('schema')!r}")
    if tok.get("code_commit") != expected_commit:
        raise Phase2Error(
            f"[submit-guard] token commit {tok.get('code_commit')} != current {expected_commit}")
    stored = tok.pop("token_sha256", None)
    body = json.dumps({k: v for k, v in sorted(tok.items())}, sort_keys=True).encode()
    if _sha256_bytes(body) != stored:
        raise Phase2Error(f"[submit-guard] {p}: token hash mismatch (tampered)")
    tok["token_sha256"] = stored
    return tok
