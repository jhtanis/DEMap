"""Phase 1 winner selection.

CANONICAL rule (author-approved): ``phase1_seed_mean_lexicographic_v1``
    1. group runs by model + full hyperparameter identity, EXCLUDING seed;
    2. require exactly seeds {0, 1} per configuration (reject missing/duplicate/unexpected);
    3. mean + sample SD (ddof=1) across the two seeds for val_dev R@5, R@10, MRR@100;
    4. rank configurations by mean R@5 -> mean R@10 -> mean MRR@100 (NO tie margin);
    5. exact ties (all three means equal) reported, broken by a deterministic stable key;
    6. retained seed = better individual seed WITHIN the winning configuration, by
       seed R@5 -> R@10 -> MRR@100.
Within-run checkpoint selection is val_dev MRR@100 (done at train time).

HISTORICAL rule (reproduction only): ``phase1_operative_per_run_v1`` -- ranks all
seed-specific runs against one another. It must NOT produce a canonical winner manifest.
"""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

CANONICAL_RULE = {
    "rule_id": "phase1_seed_mean_lexicographic_v1",
    "checkpoint_selection_metric_within_run": "val_dev_mrr@100",
    "grouping": "model + full hyperparameter identity, seed excluded",
    "required_seeds": [0, 1],
    "configuration_ranking_metrics": ["mean_val_dev_recall@5", "mean_val_dev_recall@10", "mean_val_dev_mrr@100"],
    "metric_ordering": "descending",
    "tie_margin": "none",
    "exact_tie_handling": "reported; broken by deterministic stable config key",
    "retained_seed_rule": "better seed WITHIN winning config by seed recall@5 -> recall@10 -> mrr@100",
    "seed_aggregation": "mean; SD sample (ddof=1)",
}

CANONICAL_PHASE2_RULE = {
    "rule_id": "phase2_seed_mean_lexicographic_v1",
    "checkpoint_selection_metric_within_run": "val_dev_mrr@100",
    "grouping": "model + full hyperparameter identity INCLUDING negative strategy, seed excluded",
    "required_seeds": [0, 1],
    "configuration_ranking_metrics": ["mean_val_dev_recall@5", "mean_val_dev_recall@10", "mean_val_dev_mrr@100"],
    "metric_ordering": "descending",
    "tie_margin": "none",
    "exact_tie_handling": "reported; broken by deterministic stable config key",
    "retained_seed_rule": "better seed WITHIN winning config by seed recall@5 -> recall@10 -> mrr@100",
    "seed_aggregation": "mean; SD sample (ddof=1)",
    "reporting_metrics": ["recall@1", "recall@5", "recall@10", "mrr@100"],
}

HISTORICAL_PER_RUN_RULE = {
    "rule_id": "phase1_operative_per_run_v1",
    "configuration_ranking_metrics": ["val_dev_recall@5", "val_dev_recall@10", "val_dev_mrr@100"],
    "seed_aggregation": "none (per-run, per-seed)",
    "status": "historical_reproduction_only",
}

# Hyperparameter identity (seed excluded).
CONFIG_FIELDS = ("model_id", "query_variant", "recipe", "cde_format", "loss", "lr", "temperature", "epochs", "batch_size")

# Phase 2 varies the negative strategy too; omitting it would collapse the three
# strategies at one (lr, temp, epoch) into a single group of six runs.
PHASE2_CONFIG_FIELDS = CONFIG_FIELDS + ("negative_strategy",)


class SelectionError(SystemExit):
    pass


def _read_run(run_dir: str) -> Optional[Dict[str, Any]]:
    mpath = os.path.join(run_dir, "metrics.json")
    rpath = os.path.join(run_dir, "run_config.json")
    if not (os.path.exists(mpath) and os.path.exists(rpath)):
        return None
    m = json.load(open(mpath))
    r = json.load(open(rpath))
    vd = (m.get("metrics_by_split", {}) or {}).get("val_dev")
    if not vd:
        return None
    t = r.get("train", {})
    return {
        "run_dir": run_dir,
        "run_id": os.path.basename(run_dir),
        "model_id": r.get("base_model_id") or r.get("init_model_name_or_path"),
        "seed": t.get("seed"),
        "lr": t.get("lr"),
        "temperature": t.get("temperature"),
        "epochs": t.get("epochs"),
        "batch_size": t.get("batch_size"),
        "loss": t.get("loss"),
        "precision": t.get("precision"),
        "query_variant": r.get("query_variant"),
        "recipe": r.get("recipe"),
        "cde_format": r.get("cde_format"),
        # recall@1 is reported (not ranked on) by the durable Phase 1 manifest.
        "recall1": float(vd["recall@1"]) if vd.get("recall@1") is not None else None,
        "recall5": float(vd.get("recall@5")),
        "recall10": float(vd.get("recall@10")),
        "mrr100": float(vd.get("mrr@100")),
        "checkpoint": os.path.join(run_dir, "model"),
    }


def collect_runs(root: str) -> List[Dict[str, Any]]:
    runs = []
    for rc in glob.glob(os.path.join(root, "**", "run_config.json"), recursive=True):
        r = _read_run(os.path.dirname(rc))
        if r is not None:
            runs.append(r)
    return runs


def _config_key(run: Dict[str, Any], fields: Tuple[str, ...] = CONFIG_FIELDS) -> Tuple:
    return tuple(run.get(f) for f in fields)


def _mean_sd(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


# ----------------------------------------------------------------- canonical (seed-mean)
def select_seed_mean(root: str, *, required_seeds=(0, 1),
                     config_fields: Tuple[str, ...] = CONFIG_FIELDS,
                     rule: Optional[Dict[str, Any]] = None,
                     runs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    rule = rule or CANONICAL_RULE
    if runs is None:
        runs = collect_runs(root)
    if not runs:
        raise SelectionError(f"[select] no completed runs with val_dev metrics under {root}")

    # group by full config identity (seed excluded)
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault(_config_key(r, config_fields), []).append(r)

    req = set(required_seeds)
    configs = []
    for key, rs in groups.items():
        seeds = [r["seed"] for r in rs]
        seen = set(seeds)
        if len(seeds) != len(seen):
            raise SelectionError(f"[select] duplicate seed(s) in config {dict(zip(config_fields, key))}: {sorted(seeds)}")
        if seen != req:
            raise SelectionError(
                f"[select] config {dict(zip(config_fields, key))} has seeds {sorted(seen)}, "
                f"required exactly {sorted(req)} (missing/unexpected seed)."
            )
        by_seed = {r["seed"]: r for r in rs}
        r5m, r5sd = _mean_sd([by_seed[s]["recall5"] for s in sorted(req)])
        r10m, r10sd = _mean_sd([by_seed[s]["recall10"] for s in sorted(req)])
        mrm, mrsd = _mean_sd([by_seed[s]["mrr100"] for s in sorted(req)])
        r1vals = [by_seed[s].get("recall1") for s in sorted(req)]
        r1m, r1sd = _mean_sd([float(x) for x in r1vals]) if all(x is not None for x in r1vals) else (None, None)
        configs.append({
            "recall1_mean": r1m, "recall1_sd": r1sd,
            "config_key": key,
            "fields": dict(zip(config_fields, key)),
            "by_seed": by_seed,
            "recall5_mean": r5m, "recall5_sd": r5sd,
            "recall10_mean": r10m, "recall10_sd": r10sd,
            "mrr100_mean": mrm, "mrr100_sd": mrsd,
        })

    winners: Dict[str, Any] = {}
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for c in configs:
        by_model.setdefault(c["fields"]["model_id"], []).append(c)

    for model_id, cs in by_model.items():
        # rank by mean R@5 -> R@10 -> MRR@100 (desc), then a deterministic stable key.
        cs_sorted = sorted(
            cs,
            key=lambda c: (c["recall5_mean"], c["recall10_mean"], c["mrr100_mean"], tuple(str(x) for x in c["config_key"])),
            reverse=True,
        )
        top = cs_sorted[0]
        exact_tie = False
        if len(cs_sorted) > 1:
            b = cs_sorted[1]
            exact_tie = (top["recall5_mean"], top["recall10_mean"], top["mrr100_mean"]) == \
                        (b["recall5_mean"], b["recall10_mean"], b["mrr100_mean"])
        # retained seed = better seed WITHIN the winning config
        retained = max(top["by_seed"].values(), key=lambda r: (r["recall5"], r["recall10"], r["mrr100"], -int(r["seed"])))
        winners[model_id] = {"config": top, "exact_tie_at_top": exact_tie, "retained": retained}
    return {"rule": rule, "config_fields": list(config_fields), "n_runs": len(runs),
            "n_configs": len(configs), "winners": winners}


def build_phase2_winner_manifest(
    root: str,
    *,
    section4_3_manifest_id: str = "section4_3_winners_v1",
    section4_3_manifest_sha256: str = "unknown",
    precision: str = "fp16_mixed",
    code_commit: str = "unknown",
) -> Dict[str, Any]:
    """CANONICAL Phase 2 winner manifest (seed-mean rule)."""
    sel = select_seed_mean(root)
    rows = []
    for model_id, w in sel["winners"].items():
        c = w["config"]
        s0, s1 = c["by_seed"].get(0), c["by_seed"].get(1)
        ret = w["retained"]
        rows.append({
            "model_id": model_id,
            "section4_3_manifest_id": section4_3_manifest_id,
            "section4_3_manifest_sha256": section4_3_manifest_sha256,
            "query_variant": c["fields"]["query_variant"],
            "recipe": c["fields"]["recipe"],
            "cde_format": c["fields"]["cde_format"],
            "loss": c["fields"]["loss"],
            "lr": c["fields"]["lr"],
            "temperature": c["fields"]["temperature"],
            "epochs": c["fields"]["epochs"],
            "batch_size": c["fields"]["batch_size"],
            "precision": precision,
            "seed0_run_id": s0["run_id"], "seed0_checkpoint": s0["checkpoint"],
            "seed1_run_id": s1["run_id"], "seed1_checkpoint": s1["checkpoint"],
            "recall5_mean": c["recall5_mean"], "recall5_sd": c["recall5_sd"],
            "recall10_mean": c["recall10_mean"], "recall10_sd": c["recall10_sd"],
            "mrr100_mean": c["mrr100_mean"], "mrr100_sd": c["mrr100_sd"],
            "retained_seed": ret["seed"], "retained_checkpoint": ret["checkpoint"],
            "exact_tie_at_top": w["exact_tie_at_top"],
            "selection_rule_id": CANONICAL_RULE["rule_id"],
        })
    return {
        "schema_version": 1,
        "manifest_id": "phase1_winners_for_phase2_v1",
        "selection_rule": CANONICAL_RULE,
        "source_runs_root": root,
        "n_runs_considered": sel["n_runs"],
        "n_configs_considered": sel["n_configs"],
        "code_commit": code_commit,
        "winners": rows,
    }


# ----------------------------------------------------------------- canonical Phase 2
def _read_phase2_run(run_dir: str) -> Optional[Dict[str, Any]]:
    r = _read_run(run_dir)
    if r is None:
        return None
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
    prov = cfg.get("paper_provenance") or {}
    strat = prov.get("negative_strategy")
    if strat is None:
        # historical Phase 2 runs record only the base strategy name
        strat = (cfg.get("hardneg") or {}).get("strategy")
    r["negative_strategy"] = strat
    r["parent_checkpoint"] = cfg.get("init_model_name_or_path")
    r["requested_precision"] = prov.get("requested_precision") or (cfg.get("train") or {}).get("precision")
    r["effective_precision"] = ((cfg.get("precision_verification") or {}).get("runtime") or {}).get("effective_precision")
    return r


def collect_phase2_runs(root: str) -> List[Dict[str, Any]]:
    runs = []
    for rc in glob.glob(os.path.join(root, "**", "run_config.json"), recursive=True):
        r = _read_phase2_run(os.path.dirname(rc))
        if r is not None:
            runs.append(r)
    return runs


def select_phase2_seed_mean(root: str, *, required_seeds=(0, 1)) -> Dict[str, Any]:
    """CANONICAL Phase 2 rule ``phase2_seed_mean_lexicographic_v1``.

    Identical ranking semantics to Phase 1 (mean R@5 -> mean R@10 -> mean MRR@100, no tie
    margin, retained seed chosen only WITHIN the winning configuration) but the
    configuration identity additionally includes the negative strategy.
    """
    runs = collect_phase2_runs(root)
    if not runs:
        raise SelectionError(f"[select] no completed Phase 2 runs with val_dev metrics under {root}")
    missing = [r["run_id"] for r in runs if not r.get("negative_strategy")]
    if missing:
        raise SelectionError(
            f"[select] {len(missing)} Phase 2 run(s) record no negative strategy; the Phase 2 "
            f"configuration identity requires it (first: {missing[0]})")
    return select_seed_mean(root, required_seeds=required_seeds,
                            config_fields=PHASE2_CONFIG_FIELDS,
                            rule=CANONICAL_PHASE2_RULE, runs=runs)


def build_phase2_final_manifest(
    root: str, *, phase1_manifest_path: str, phase1_manifest_sha256: str,
    code_commit: str = "unknown", precision: str = "fp32",
) -> Dict[str, Any]:
    """Final Phase 2 winner manifest (the checkpoint promoted to reranking)."""
    sel = select_phase2_seed_mean(root)
    rows = []
    for model_id, w in sorted(sel["winners"].items()):
        c, ret = w["config"], w["retained"]
        f = c["fields"]
        s0, s1 = c["by_seed"][0], c["by_seed"][1]
        rows.append({
            "model_id": model_id,
            "phase1_manifest_path": phase1_manifest_path,
            "phase1_manifest_sha256": phase1_manifest_sha256,
            "parent_checkpoint": s0.get("parent_checkpoint"),
            "query_variant": f["query_variant"], "recipe": f["recipe"],
            "cde_format": f["cde_format"], "loss": f["loss"],
            "negative_strategy": f["negative_strategy"],
            "lr": f["lr"], "temperature": f["temperature"], "epochs": f["epochs"],
            "batch_size": f["batch_size"], "precision": precision,
            "seed0_run_id": s0["run_id"], "seed1_run_id": s1["run_id"],
            "seed0_metrics": {"recall@1": s0.get("recall1"), "recall@5": s0["recall5"],
                              "recall@10": s0["recall10"], "mrr@100": s0["mrr100"]},
            "seed1_metrics": {"recall@1": s1.get("recall1"), "recall@5": s1["recall5"],
                              "recall@10": s1["recall10"], "mrr@100": s1["mrr100"]},
            "recall1_mean": c.get("recall1_mean"), "recall1_sd": c.get("recall1_sd"),
            "recall5_mean": c["recall5_mean"], "recall5_sd": c["recall5_sd"],
            "recall10_mean": c["recall10_mean"], "recall10_sd": c["recall10_sd"],
            "mrr100_mean": c["mrr100_mean"], "mrr100_sd": c["mrr100_sd"],
            "retained_seed": ret["seed"], "retained_run_id": ret["run_id"],
            "retained_checkpoint": ret["checkpoint"],
            "exact_tie_at_top": w["exact_tie_at_top"],
            "selection_rule_id": CANONICAL_PHASE2_RULE["rule_id"],
        })
    return {
        "schema_version": 1,
        "manifest_id": "phase2_winners_final_v1",
        "selection_rule": CANONICAL_PHASE2_RULE,
        "selection_rule_id": CANONICAL_PHASE2_RULE["rule_id"],
        "source_runs_root": root,
        "n_runs_considered": sel["n_runs"], "n_configs_considered": sel["n_configs"],
        "configuration_identity": list(PHASE2_CONFIG_FIELDS),
        "code_commit": code_commit,
        "winners": rows,
    }


# --------------------------------------------------- historical (per-run) reproduction only
def select_winners_per_run(root: str) -> Dict[str, Any]:
    """HISTORICAL per-run rule (ranks all seed-specific runs). Diagnostic/reproduction only."""
    runs = collect_runs(root)
    if not runs:
        raise SelectionError(f"[select] no completed runs under {root}")
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        by_model.setdefault(r["model_id"], []).append(r)
    winners = {}
    for model_id, rs in by_model.items():
        winners[model_id] = sorted(rs, key=lambda x: (x["recall5"], x["recall10"], x["mrr100"], x["run_id"]), reverse=True)[0]
    return {"rule": HISTORICAL_PER_RUN_RULE, "n_runs": len(runs), "winners": winners}


def build_phase2_winner_manifest_per_run(root: str, **kw) -> Dict[str, Any]:
    """Blocked: the historical per-run selector must not emit a canonical manifest."""
    raise SelectionError(
        "[select] the historical per-run rule (phase1_operative_per_run_v1) is reproduction-only "
        "and cannot emit a canonical Phase 2 winner manifest. Use build_phase2_winner_manifest "
        "(phase1_seed_mean_lexicographic_v1)."
    )
