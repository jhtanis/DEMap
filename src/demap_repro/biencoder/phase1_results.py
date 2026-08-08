"""Durable canonical Phase 1 results: winner manifest build + fail-closed validation.

The manifest emitted here is the ONLY approved hand-off from Phase 1 to Phase 2. It is
built exclusively from completed run artifacts under a Phase 1 output root, using the
canonical selection rule ``phase1_seed_mean_lexicographic_v1``
(see :mod:`demap_repro.biencoder.select`).

Two entry points:

``build_manifest(root, ...)``
    Re-derive the winners from artifacts and emit the manifest dict.

``validate_manifest(manifest, root, ...)``
    Re-derive independently and return a list of discrepancies. An empty list means the
    manifest is reproducible from the artifacts. Detects altered metrics, altered
    hashes, altered lineage, incomplete seed pairs, and missing checkpoints.

Nothing here selects a "best individual run": the retained seed is always chosen WITHIN
the winning seed-averaged configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from demap_repro.biencoder import select as _select

SCHEMA_VERSION = 1
MANIFEST_ID = "phase1_winners_for_phase2_v1"
# Manifests the Phase 2 generator will accept as a parent source. Every entry must be
# produced by the canonical seed-mean rule (checked separately). v2 extends v1 with
# all-MPNet, whose Phase 1 grid predates artifacts/phase1_canonical_v1 and is therefore
# registered with origin: historical_canonical_artifact; the two shared entries are
# copied verbatim from v1.
ACCEPTED_MANIFEST_IDS = frozenset({MANIFEST_ID, "phase1_winners_final_v2"})
SELECTION_RULE_ID = _select.CANONICAL_RULE["rule_id"]
REQUIRED_SEEDS = (0, 1)

# Fields whose value must be reproduced exactly by validate_manifest().
_WINNER_SCALARS = (
    "model_id", "query_variant", "recipe", "cde_format", "loss",
    "lr", "temperature", "epochs", "batch_size",
    "seed0_run_id", "seed1_run_id", "retained_seed",
    "retained_checkpoint", "seed0_checkpoint", "seed1_checkpoint",
    "exact_tie_at_top", "selection_rule_id",
)
_WINNER_METRICS = (
    "recall1_mean", "recall1_sd", "recall5_mean", "recall5_sd",
    "recall10_mean", "recall10_sd", "mrr100_mean", "mrr100_sd",
)
_SEED_METRICS = ("recall@1", "recall@5", "recall@10", "mrr@100")

_FLOAT_TOL = 1e-12


class Phase1ResultsError(SystemExit):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_fingerprint(run_dir: str) -> str:
    """Tamper-evident digest of a single run: its resolved config + its metrics."""
    h = hashlib.sha256()
    for name in ("run_config.json", "metrics.json"):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            raise Phase1ResultsError(f"[phase1-results] {run_dir}: missing {name}")
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def runs_digest(root: str) -> Dict[str, Any]:
    """Aggregate digest over every completed run under ``root`` (order-independent)."""
    fps = {}
    for rc in sorted(_iter_run_configs(root)):
        d = os.path.dirname(rc)
        fps[os.path.basename(d)] = run_fingerprint(d)
    h = hashlib.sha256()
    for rid in sorted(fps):
        h.update(f"{rid}:{fps[rid]}\n".encode())
    return {"n_runs": len(fps), "digest_sha256": h.hexdigest()}


def _iter_run_configs(root: str) -> List[str]:
    import glob

    return glob.glob(os.path.join(root, "**", "run_config.json"), recursive=True)


def _seed_metrics(run: Dict[str, Any]) -> Dict[str, float]:
    return {
        "run_id": run["run_id"],
        "recall@1": run["recall1"],
        "recall@5": run["recall5"],
        "recall@10": run["recall10"],
        "mrr@100": run["mrr100"],
    }


def _mean_sd(a: float, b: float) -> tuple:
    m = (a + b) / 2.0
    sd = (((a - m) ** 2 + (b - m) ** 2) / 1.0) ** 0.5  # sample SD, ddof=1, n=2
    return m, sd


def build_manifest(
    root: str,
    *,
    protocol_path: str,
    winners_path: str,
    code_commit: str,
    precision: str = "fp16_mixed",
    slurm_job_id: Optional[str] = None,
    created_utc: Optional[str] = None,
    section4_3_results_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the durable Phase 1 -> Phase 2 winner manifest from completed artifacts."""
    sel = _select.select_seed_mean(root, required_seeds=REQUIRED_SEEDS)

    winners: List[Dict[str, Any]] = []
    for model_id, w in sorted(sel["winners"].items()):
        c, ret = w["config"], w["retained"]
        f = c["fields"]
        s0, s1 = c["by_seed"][0], c["by_seed"][1]
        r1m, r1sd = _mean_sd(s0["recall1"], s1["recall1"])
        winners.append({
            "model_id": model_id,
            # ---- Section 4.3 lineage (representation + loss)
            "query_variant": f["query_variant"],
            "recipe": f["recipe"],
            "cde_format": f["cde_format"],
            "loss": f["loss"],
            # ---- winning hyperparameters
            "lr": f["lr"],
            "temperature": f["temperature"],
            "epochs": f["epochs"],
            "batch_size": f["batch_size"],
            "precision": precision,
            # ---- both seeds of the winning configuration
            "seed0_run_id": s0["run_id"],
            "seed1_run_id": s1["run_id"],
            "seed0_checkpoint": s0["checkpoint"],
            "seed1_checkpoint": s1["checkpoint"],
            "seed0_metrics": _seed_metrics(s0),
            "seed1_metrics": _seed_metrics(s1),
            "seed0_run_fingerprint": run_fingerprint(s0["run_dir"]),
            "seed1_run_fingerprint": run_fingerprint(s1["run_dir"]),
            # ---- configuration means and sample SDs
            "recall1_mean": r1m, "recall1_sd": r1sd,
            "recall5_mean": c["recall5_mean"], "recall5_sd": c["recall5_sd"],
            "recall10_mean": c["recall10_mean"], "recall10_sd": c["recall10_sd"],
            "mrr100_mean": c["mrr100_mean"], "mrr100_sd": c["mrr100_sd"],
            # ---- retained seed (chosen WITHIN the winning configuration)
            "retained_seed": ret["seed"],
            "retained_run_id": ret["run_id"],
            "retained_checkpoint": ret["checkpoint"],
            "exact_tie_at_top": w["exact_tie_at_top"],
            "selection_rule_id": SELECTION_RULE_ID,
        })

    src_hashes = {
        "phase0_protocol_path": protocol_path,
        "phase0_protocol_sha256": sha256_file(protocol_path),
        "section4_3_winner_manifest_path": winners_path,
        "section4_3_winner_manifest_sha256": sha256_file(winners_path),
    }
    if section4_3_results_csv:
        src_hashes["section4_3_results_csv"] = section4_3_results_csv
        src_hashes["section4_3_results_sha256"] = sha256_file(section4_3_results_csv)

    man: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": MANIFEST_ID,
        "selection_rule": _select.CANONICAL_RULE,
        "selection_rule_id": SELECTION_RULE_ID,
        "required_seeds": list(REQUIRED_SEEDS),
        "precision": precision,
        "source_runs_root": root,
        "n_runs_considered": sel["n_runs"],
        "n_configs_considered": sel["n_configs"],
        "source_runs_digest": runs_digest(root),
        "source_hashes": src_hashes,
        "code_commit": code_commit,
        "winners": winners,
    }
    if slurm_job_id:
        man["slurm_job_id"] = slurm_job_id
    if created_utc:
        man["created_utc"] = created_utc
    return man


def validate_manifest(
    manifest: Dict[str, Any],
    root: str,
    *,
    protocol_path: Optional[str] = None,
    winners_path: Optional[str] = None,
    check_checkpoints: bool = True,
) -> List[str]:
    """Re-derive from artifacts and return every discrepancy found (empty == valid)."""
    problems: List[str] = []

    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {manifest.get('schema_version')!r} != {SCHEMA_VERSION}")
    if manifest.get("manifest_id") != MANIFEST_ID:
        problems.append(f"manifest_id {manifest.get('manifest_id')!r} != {MANIFEST_ID}")
    if manifest.get("selection_rule_id") != SELECTION_RULE_ID:
        problems.append(f"selection_rule_id {manifest.get('selection_rule_id')!r} != {SELECTION_RULE_ID}")
    if list(manifest.get("required_seeds") or []) != list(REQUIRED_SEEDS):
        problems.append(f"required_seeds {manifest.get('required_seeds')!r} != {list(REQUIRED_SEEDS)}")

    # Re-run the canonical selector against the artifacts. A missing/duplicated seed
    # raises SelectionError, which is itself a validation failure.
    try:
        sel = _select.select_seed_mean(root, required_seeds=REQUIRED_SEEDS)
    except SystemExit as e:
        return problems + [f"selector rejected artifacts under {root}: {e}"]

    if manifest.get("n_runs_considered") != sel["n_runs"]:
        problems.append(f"n_runs_considered {manifest.get('n_runs_considered')} != {sel['n_runs']}")
    if manifest.get("n_configs_considered") != sel["n_configs"]:
        problems.append(f"n_configs_considered {manifest.get('n_configs_considered')} != {sel['n_configs']}")

    # digest over all runs
    got_digest = runs_digest(root)
    man_digest = manifest.get("source_runs_digest") or {}
    if man_digest.get("digest_sha256") != got_digest["digest_sha256"]:
        problems.append("source_runs_digest.digest_sha256 does not match the artifacts")
    if man_digest.get("n_runs") != got_digest["n_runs"]:
        problems.append(
            f"source_runs_digest.n_runs {man_digest.get('n_runs')} != {got_digest['n_runs']}")

    # source hashes
    src = manifest.get("source_hashes") or {}
    if protocol_path:
        exp = sha256_file(protocol_path)
        if src.get("phase0_protocol_sha256") != exp:
            problems.append("phase0_protocol_sha256 mismatch")
    if winners_path:
        exp = sha256_file(winners_path)
        if src.get("section4_3_winner_manifest_sha256") != exp:
            problems.append("section4_3_winner_manifest_sha256 mismatch")

    # per-model winners
    man_winners = {w.get("model_id"): w for w in (manifest.get("winners") or [])}
    if set(man_winners) != set(sel["winners"]):
        problems.append(
            f"winner model_ids {sorted(man_winners)} != derived {sorted(sel['winners'])}")

    for model_id, w in sel["winners"].items():
        mw = man_winners.get(model_id)
        if mw is None:
            continue
        c, ret = w["config"], w["retained"]
        f = c["fields"]
        s0, s1 = c["by_seed"][0], c["by_seed"][1]
        r1m, r1sd = _mean_sd(s0["recall1"], s1["recall1"])
        derived = {
            "model_id": model_id,
            "query_variant": f["query_variant"], "recipe": f["recipe"],
            "cde_format": f["cde_format"], "loss": f["loss"],
            "lr": f["lr"], "temperature": f["temperature"], "epochs": f["epochs"],
            "batch_size": f["batch_size"],
            "seed0_run_id": s0["run_id"], "seed1_run_id": s1["run_id"],
            "seed0_checkpoint": s0["checkpoint"], "seed1_checkpoint": s1["checkpoint"],
            "retained_seed": ret["seed"], "retained_checkpoint": ret["checkpoint"],
            "exact_tie_at_top": w["exact_tie_at_top"],
            "selection_rule_id": SELECTION_RULE_ID,
            "recall1_mean": r1m, "recall1_sd": r1sd,
            "recall5_mean": c["recall5_mean"], "recall5_sd": c["recall5_sd"],
            "recall10_mean": c["recall10_mean"], "recall10_sd": c["recall10_sd"],
            "mrr100_mean": c["mrr100_mean"], "mrr100_sd": c["mrr100_sd"],
        }
        for k in _WINNER_SCALARS:
            if mw.get(k) != derived[k]:
                problems.append(f"{model_id}: {k}={mw.get(k)!r} != derived {derived[k]!r}")
        for k in _WINNER_METRICS:
            got, exp = mw.get(k), derived[k]
            if got is None or abs(float(got) - float(exp)) > _FLOAT_TOL:
                problems.append(f"{model_id}: {k}={got!r} != derived {exp!r}")
        for seed, run in ((0, s0), (1, s1)):
            block = mw.get(f"seed{seed}_metrics") or {}
            for mk in _SEED_METRICS:
                key = {"recall@1": "recall1", "recall@5": "recall5",
                       "recall@10": "recall10", "mrr@100": "mrr100"}[mk]
                got, exp = block.get(mk), run[key]
                if got is None or abs(float(got) - float(exp)) > _FLOAT_TOL:
                    problems.append(f"{model_id}: seed{seed}_metrics[{mk}]={got!r} != {exp!r}")
            fp_key = f"seed{seed}_run_fingerprint"
            if mw.get(fp_key) != run_fingerprint(run["run_dir"]):
                problems.append(f"{model_id}: {fp_key} does not match the run artifacts")

        # the retained seed must belong to the winning configuration
        if mw.get("retained_run_id") not in (s0["run_id"], s1["run_id"]):
            problems.append(
                f"{model_id}: retained_run_id {mw.get('retained_run_id')!r} is outside the "
                "winning configuration")
        if check_checkpoints and not os.path.isdir(str(mw.get("retained_checkpoint"))):
            problems.append(
                f"{model_id}: retained_checkpoint does not exist: {mw.get('retained_checkpoint')!r}")

    return problems


def load_manifest(path: str) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        man = yaml.safe_load(f)
    if not isinstance(man, dict):
        raise Phase1ResultsError(f"[phase1-results] {path}: not a mapping")
    return man


def resolve_parent_checkpoint(manifest: Dict[str, Any], model_id: str) -> str:
    """The ONLY approved way for Phase 2 to obtain a parent checkpoint for a model."""
    for w in manifest.get("winners") or []:
        if w.get("model_id") == model_id:
            ck = w.get("retained_checkpoint")
            if not ck:
                raise Phase1ResultsError(f"[phase1-results] {model_id}: no retained_checkpoint")
            if not os.path.isdir(ck):
                raise Phase1ResultsError(
                    f"[phase1-results] {model_id}: retained_checkpoint missing on disk: {ck}")
            return ck
    raise Phase1ResultsError(
        f"[phase1-results] model {model_id!r} not in manifest "
        f"{manifest.get('manifest_id')!r}; available: "
        f"{[w.get('model_id') for w in manifest.get('winners') or []]}")


def write_manifest_yaml(manifest: Dict[str, Any], path: str, *, header: str = "") -> None:
    import yaml

    def _native(o):
        if isinstance(o, dict):
            return {_native(k): _native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_native(v) for v in o]
        if hasattr(o, "item") and not isinstance(o, (str, bytes)):
            try:
                return o.item()
            except Exception:
                return o
        return o

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        if header:
            f.write(header if header.endswith("\n") else header + "\n")
        yaml.safe_dump(_native(manifest), f, sort_keys=False, default_flow_style=False)
