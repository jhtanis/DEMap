"""Canonical Phase 1 config generation with winner propagation (fail-closed).

Derives query representation, query column, CDE recipe/fields, CDE format, and loss
EXCLUSIVELY from the Section 4.3 winner manifest. There is no base config supplying
competing defaults. Any attempt to override a winner-derived field fails unless the
explicit research-only ``allow_winner_override`` is set (which marks the output
non-canonical and records the deviation).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

SUBMISSION_TOKEN_NAME = "SUBMISSION_TOKEN.json"

from demap_repro.biencoder import protocol as _proto
from demap_repro.biencoder import quarantine as _q
from demap_repro.biencoder import winners as _win

_PRECISION = {
    "fp16_mixed": {"fp16": True, "bf16": False, "precision": "fp16"},
    "bf16_mixed": {"fp16": False, "bf16": True, "precision": "bf16"},
    "fp32": {"fp16": False, "bf16": False, "precision": "fp32"},
}

# Winner-derived fields the base config MUST NOT set (they come only from the manifest).
WINNER_OWNED_FIELDS = ("query_id", "query_col", "cde_recipe", "cde_representation", "cde_format", "loss")


class Phase1Error(SystemExit):
    pass


def _slug(s: str) -> str:
    return s.replace("/", "__")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def recipe_fields(recipe: str) -> List[str]:
    from demap_repro.text.recipes import RECIPE_FIELDS  # atomic map

    fields: List[str] = []
    for tok in str(recipe).split("_"):
        if tok not in RECIPE_FIELDS:
            raise Phase1Error(f"[phase1] recipe token {tok!r} (in {recipe!r}) is not a known atomic recipe")
        fields.extend(RECIPE_FIELDS[tok])
    return fields


def _query_col_for(proto: Dict[str, Any], query_id: str) -> str:
    for q in proto["query_representations"]:
        if q["id"] == query_id:
            return q["column"]
    raise Phase1Error(f"[phase1] query id {query_id!r} not in protocol grid")


def _resolved_text_audit(winner: Dict[str, Any], proto: Dict[str, Any]) -> Dict[str, Any]:
    """Small, non-sensitive audit: a few val_dev query strings + the resolved CDE fields."""
    audit: Dict[str, Any] = {
        "query_id": winner["query_id"],
        "query_name": winner["query_name"],
        "query_col": winner["query_col"],
        "cde_recipe": winner["cde_recipe"],
        "cde_representation": winner["cde_representation"],
        "cde_fields_in_order": recipe_fields(winner["cde_recipe"]),
        "cde_format": winner["cde_format"],
        "sep": " | ",
    }
    try:
        import pandas as pd

        vd = pd.read_parquet(f"{proto['data']['splits_dir']}/val_dev.parquet")
        col = winner["query_col"]
        if col in vd.columns:
            audit["query_text_samples"] = [str(x)[:160] for x in vd[col].dropna().head(3).tolist()]
        else:
            audit["query_text_samples"] = f"[column {col} not present in val_dev sample]"
    except Exception as e:
        audit["query_text_samples"] = f"[unavailable: {type(e).__name__}]"
    return audit


def generate_phase1(
    *,
    protocol_path: str,
    winners_path: str,
    model_ids: List[str],
    precision: str = "fp16_mixed",
    out_root: str = "artifacts/phase1_canonical_v1",
    allow_winner_override: bool = False,
    allow_legacy: bool = False,
    base_overrides: Optional[Dict[str, Any]] = None,
    code_commit: str = "unknown",
    dirty: bool = False,
    check_rows: bool = True,
    write: bool = True,
) -> Dict[str, Any]:
    if precision not in _PRECISION:
        raise Phase1Error(f"[phase1] precision {precision!r} not in {sorted(_PRECISION)}")

    proto = _proto.load_protocol(protocol_path)
    _q.assert_protocol_not_legacy(proto, allow_legacy=allow_legacy)
    _q.assert_canonical_trainer(proto["phase1"]["trainer"], allow_legacy=allow_legacy)
    if precision not in proto["phase1"]["allowed_precisions"]:
        raise Phase1Error(f"[phase1] precision {precision!r} not allowed by protocol")

    fps = _proto.verify_fingerprints(proto, check_rows=check_rows)
    man = _win.load_winner_manifest(winners_path)
    proto_sha = _proto.sha256_file(protocol_path)

    # Fail-closed: base config must not provide competing winner-owned fields.
    base_overrides = dict(base_overrides or {})
    conflicting = [f for f in WINNER_OWNED_FIELDS if f in base_overrides]
    if conflicting and not allow_winner_override:
        raise Phase1Error(
            f"[phase1] base config attempts to set winner-owned fields {conflicting}; "
            f"these come ONLY from the winner manifest. Use allow_winner_override for research runs."
        )

    d = proto["data"]
    ss = proto["phase1"]["search_space"]
    grid = list(itertools.product(ss["learning_rates"], ss["temperatures"], ss["epochs"], ss["seeds"]))
    runs_per_model = len(grid)
    if runs_per_model != proto["phase1"]["runs_per_model"]:
        raise Phase1Error(f"[phase1] grid has {runs_per_model} runs but protocol declares {proto['phase1']['runs_per_model']}")

    out_root_p = Path(out_root)
    summary: Dict[str, Any] = {
        "protocol_path": protocol_path,
        "protocol_sha256": proto_sha,
        "winners_path": winners_path,
        "winner_manifest_sha256": man["_manifest_sha256"],
        "precision_requested": precision,
        "trainer": "modern_trainer",
        "out_root": str(out_root_p),
        "runs_per_model": runs_per_model,
        "canonical": not (allow_winner_override or allow_legacy),
        "models": {},
        "total_runs": 0,
    }
    manifest_rows: List[Dict[str, Any]] = []

    for model_id in model_ids:
        w = _win.get_winner(man, model_id)
        if allow_winner_override and base_overrides:
            for f in WINNER_OWNED_FIELDS:
                if f in base_overrides:
                    w = dict(w)
                    w[f] = base_overrides[f]

        # Assertions: query col + recipe resolution consistent with protocol/manifest.
        exp_col = _query_col_for(proto, w["query_id"])
        if w["query_col"] != exp_col:
            raise Phase1Error(f"[phase1] {model_id}: winner query_col {w['query_col']} != protocol {exp_col} for {w['query_id']}")
        if w["cde_format"] != d["cde_format"]:
            raise Phase1Error(f"[phase1] {model_id}: cde_format {w['cde_format']} != protocol {d['cde_format']}")
        fields = recipe_fields(w["cde_recipe"])  # raises on unknown token

        model_slug = _slug(model_id)
        model_dir = out_root_p / model_slug
        gen_dir = model_dir / "generated_configs"

        pv = _PRECISION[precision]
        prov_common = {
            "phase0_protocol_path": protocol_path,
            "phase0_protocol_sha256": proto_sha,
            "winner_manifest_path": winners_path,
            "winner_manifest_sha256": man["_manifest_sha256"],
            "winner_model_id": model_id,
            "winner_row": {k: w[k] for k in WINNER_OWNED_FIELDS + ("query_name", "cde_representation", "representation_role")},
            "source_off_the_shelf_checkpoint": model_id,
            "resolved_cde_fields": fields,
            "requested_precision": precision,
            "trainer": "modern_trainer",
            "batch_size": proto["phase1"]["batch_size"],
            "gradient_accumulation_steps": proto["phase1"]["gradient_accumulation_steps"],
            "effective_optimizer_batch_size": proto["phase1"]["batch_size"] * proto["phase1"]["gradient_accumulation_steps"],
            "in_batch_negatives_note": "MNRL family uses in-batch negatives = physical batch size - 1; grad-accum does NOT enlarge the in-batch negative pool",
            "no_duplicate_sampler": "SentenceTransformer BatchSamplers.NO_DUPLICATES",
            "loss": w["loss"],
            "similarity_fn": proto["phase1"]["similarity_fn"],
            "max_seq_length": d["max_seq_length"],
            "weight_decay": proto["phase1"]["weight_decay"],
            "optimizer": proto["phase1"]["optimizer"],
            "scheduler": proto["phase1"]["scheduler"],
            "warmup_ratio": proto["phase1"]["warmup_ratio"],
            "train_sha256": fps["train"]["sha256"],
            "val_dev_sha256": fps["val_dev"]["sha256"],
            "production_catalog_sha256": fps["catalog"]["sha256"],
            "production_catalog_rows": d["production_catalog_rows"],
            "val_dev_denominator": d["val_dev_denominator"],
            "code_commit": code_commit,
            "dirty_tree": bool(dirty),
            "is_canonical_paper_run": not (allow_winner_override or allow_legacy),
            "protocol_deviation": bool(allow_winner_override),
        }

        model_runs = []
        if write:
            gen_dir.mkdir(parents=True, exist_ok=True)
        for (lr, temp, ep, seed) in grid:
            run_key = f"lr{lr:g}__t{temp:g}__ep{ep}__seed{seed}"
            cfg = {
                "finetune_phase1": {
                    "cde_master_enriched": d["production_catalog"],
                    "splits_dir": d["splits_dir"],
                    "artifacts_dir": str(model_dir),
                    "runs_dir": "auto",
                    "base_model_id": model_id,
                    "model_name": model_id,
                    "query_variant": w["query_id"],
                    "recipe": w["cde_recipe"],
                    "cde_format": w["cde_format"],
                    "rerank_mode": "R0",
                    "sep": " | ",
                    "recipe_configs": {
                        "placeholder_policy": "placeholder",
                        "short_name_placeholder": "<MISSING_SHORT_NAME>",
                        "pv_placeholder": "<MISSING_PV_SUMMARY>",
                        "v1": {"filter_numeric_only": True, "filter_versioned_id_short_name": True},
                    },
                    "train": {
                        "device": "auto",
                        "max_seq_length": d["max_seq_length"],
                        "warmup_ratio": proto["phase1"]["warmup_ratio"],
                        "weight_decay": proto["phase1"]["weight_decay"],
                        "fit_api": "modern_trainer",
                        "fp16": pv["fp16"],
                        "bf16": pv["bf16"],
                        "precision": pv["precision"],
                        "seeds": [seed],
                        "losses": [w["loss"]],
                        "lrs": [float(lr)],
                        "batch_sizes": [proto["phase1"]["batch_size"]],
                        "temperatures": [float(temp)],
                        "epochs": [int(ep)],
                    },
                    "eval": {
                        "top_k": 100, "k_values": [5, 10, 20], "output_top_k": 20,
                        "block_size": 2048, "encode_batch_size": 64, "normalize_embeddings": True,
                        "eval_splits": ["val_dev"],
                    },
                    "stage_tag": model_slug,
                    "hybrid_alpha": 0.5,
                    "paper_provenance": {**prov_common, "run_key": run_key,
                                         "lr": float(lr), "temperature": float(temp), "epochs": int(ep), "seed": int(seed)},
                }
            }
            cfg_bytes = yaml.safe_dump(cfg, sort_keys=False).encode()
            cfg["finetune_phase1"]["paper_provenance"]["generated_config_sha256"] = _sha256_bytes(cfg_bytes)
            cfg_path = gen_dir / f"{model_slug}__{run_key}__{pv['precision']}.yaml"
            if write:
                if cfg_path.exists():
                    existing = yaml.safe_load(open(cfg_path))
                    ex_hash = (((existing or {}).get("finetune_phase1") or {}).get("paper_provenance") or {}).get("generated_config_sha256")
                    if ex_hash != cfg["finetune_phase1"]["paper_provenance"]["generated_config_sha256"]:
                        raise Phase1Error(f"[phase1] refuse to overwrite incompatible existing config: {cfg_path}")
                with open(cfg_path, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
            row = {"model_id": model_id, "config_path": str(cfg_path), "run_key": run_key,
                   "lr": float(lr), "temperature": float(temp), "epochs": int(ep), "seed": int(seed),
                   "query_id": w["query_id"], "recipe": w["cde_recipe"], "loss": w["loss"], "precision": precision}
            model_runs.append(row)
            manifest_rows.append(row)

        audit = _resolved_text_audit(w, proto)
        if write:
            with open(model_dir / "resolved_text_audit.json", "w") as f:
                json.dump({"model_id": model_id, "provenance": prov_common, "audit": audit}, f, indent=1)
        summary["models"][model_id] = {
            "winner": {k: w[k] for k in WINNER_OWNED_FIELDS + ("query_name", "cde_representation")},
            "n_runs": len(model_runs),
            "generated_dir": str(gen_dir),
        }
        summary["total_runs"] += len(model_runs)

    if write:
        out_root_p.mkdir(parents=True, exist_ok=True)
        with open(out_root_p / "phase1_preview_manifest.tsv", "w") as f:
            cols = ["model_id", "config_path", "run_key", "lr", "temperature", "epochs", "seed",
                    "query_id", "recipe", "loss", "precision"]
            f.write("\t".join(cols) + "\n")
            for r in manifest_rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        with open(out_root_p / "phase1_preview_summary.json", "w") as f:
            json.dump(summary, f, indent=1)
    return summary


# ----------------------------------------------------------- submission validation + token
def validate_generated_submission(
    *, out_root: str, protocol_path: str, winners_path: str, model_ids: List[str],
    precision: str = "fp16_mixed", expected_total: Optional[int] = None,
    expected_per_model: Optional[int] = None,
) -> Dict[str, Any]:
    """Independently re-verify a generated submission (used by --submit AND the sbatch guard).

    Fail-closed on: protocol/winner hash, manifest row count, per-model count, seeds {0,1}
    per hyperparameter configuration, correct model-specific representation+loss, and that
    every generated config uses modern_trainer + the requested precision. Returns a token dict.
    """
    proto = _proto.load_protocol(protocol_path)
    _q.assert_canonical_trainer(proto["phase1"]["trainer"])
    man = _win.load_winner_manifest(winners_path)
    proto_sha = _proto.sha256_file(protocol_path)
    pv = _PRECISION[precision]

    # Derive expected counts from the protocol + model list (mirrors phase2.py);
    # the old 108/54 literals assumed a fixed two-model submission.
    if expected_per_model is None:
        expected_per_model = int(proto["phase1"].get("runs_per_model", 54))
    if expected_total is None:
        expected_total = expected_per_model * len(model_ids)

    root = Path(out_root)
    manifest = root / "phase1_preview_manifest.tsv"
    if not manifest.exists():
        raise Phase1Error(f"[submit] preview manifest missing: {manifest}")
    rows = [l.split("\t") for l in manifest.read_text().splitlines()[1:] if l.strip()]
    if len(rows) != expected_total:
        raise Phase1Error(f"[submit] manifest has {len(rows)} rows, expected {expected_total}")

    per_model: Dict[str, int] = {}
    seeds_by_cfg: Dict[tuple, set] = {}
    for r in rows:
        mid, cfg_path, run_key = r[0], r[1], r[2]
        lr, temp, ep, seed = r[3], r[4], r[5], r[6]
        per_model[mid] = per_model.get(mid, 0) + 1
        seeds_by_cfg.setdefault((mid, lr, temp, ep), set()).add(int(seed))
        c = yaml.safe_load(open(cfg_path))["finetune_phase1"]
        w = _win.get_winner(man, mid)
        if c["query_variant"] != w["query_id"] or c["recipe"] != w["cde_recipe"] or c["train"]["losses"] != [w["loss"]]:
            raise Phase1Error(f"[submit] {cfg_path}: rep/loss != winner ({w['query_id']}/{w['cde_recipe']}/{w['loss']})")
        if c["train"]["fit_api"] != "modern_trainer":
            raise Phase1Error(f"[submit] {cfg_path}: not modern_trainer")
        if (c["train"]["fp16"], c["train"]["bf16"]) != (pv["fp16"], pv["bf16"]):
            raise Phase1Error(f"[submit] {cfg_path}: precision != {precision}")
        if c["train"]["max_seq_length"] != 256:
            raise Phase1Error(f"[submit] {cfg_path}: max_seq_length != 256")
    for mid in model_ids:
        if per_model.get(mid) != expected_per_model:
            raise Phase1Error(f"[submit] model {mid} has {per_model.get(mid)} runs, expected {expected_per_model}")
    for key, seeds in seeds_by_cfg.items():
        if seeds != {0, 1}:
            raise Phase1Error(f"[submit] config {key} seeds {sorted(seeds)} != required {{0,1}}")

    return {
        "schema": "phase1_submission_token_v1",
        "protocol_path": protocol_path, "protocol_sha256": proto_sha,
        "winners_path": winners_path, "winner_manifest_sha256": man["_manifest_sha256"],
        "out_root": str(root), "precision": precision,
        "total_runs": len(rows), "per_model": per_model,
        "n_configs": len(seeds_by_cfg), "seeds_ok": True,
        "trainer": "modern_trainer",
    }


def write_submission_token(out_root: str, token: Dict[str, Any], *, code_commit: str) -> str:
    token = dict(token)
    token["code_commit"] = code_commit
    p = Path(out_root) / SUBMISSION_TOKEN_NAME
    p.write_text(json.dumps(token, indent=1, sort_keys=True))
    token["token_sha256"] = _sha256_bytes(p.read_bytes())
    p.write_text(json.dumps(token, indent=1, sort_keys=True))
    return str(p)


def verify_submission_token(out_root: str, *, expected_commit: Optional[str] = None) -> Dict[str, Any]:
    """Independently re-validate before the array runs. Refuse if token absent/tampered/stale."""
    p = Path(out_root) / SUBMISSION_TOKEN_NAME
    if not p.exists():
        raise Phase1Error(f"[submit-guard] no submission token at {p}; the canonical CLI must create it. "
                          f"Direct sbatch is refused.")
    token = json.loads(p.read_text())
    model_ids = list(token.get("per_model", {}).keys())
    fresh = validate_generated_submission(
        out_root=out_root, protocol_path=token["protocol_path"], winners_path=token["winners_path"],
        model_ids=model_ids, precision=token["precision"],
    )
    for k in ("protocol_sha256", "winner_manifest_sha256", "total_runs", "per_model"):
        if token.get(k) != fresh.get(k):
            raise Phase1Error(f"[submit-guard] token field {k} does not match current state; refusing.")
    if expected_commit and token.get("code_commit") not in (expected_commit, None):
        raise Phase1Error(f"[submit-guard] token commit {token.get('code_commit')} != {expected_commit}; refusing.")
    return token
