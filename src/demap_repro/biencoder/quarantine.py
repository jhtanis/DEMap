"""Legacy quarantine for Phase 0 / Phase 1 entry points.

Canonical paper runs go only through `demap paper biencoder`. Historical generators,
launchers, trainers, and selection scripts are preserved for provenance/reproduction but
are blocked from canonical use unless an explicit legacy override is supplied.
"""

from __future__ import annotations

from typing import Dict


class LegacyBlocked(SystemExit):
    pass


# Machine-readable inventory + classification. status:
#   canonical                -> the new supported path
#   reusable_component       -> internal building block reused by canonical code
#   historical_reproduction  -> preserved; only via --allow-legacy
#   conflicting_unsafe       -> must never drive a canonical run
LEGACY_INVENTORY: Dict[str, Dict[str, str]] = {
    # Canonical
    "demap paper biencoder": {"kind": "entry_point", "status": "canonical",
                              "note": "the one supported route for new Phase 0/Phase 1 work"},
    "src/demap/paper/biencoder/phase1.py": {"kind": "generator", "status": "canonical"},
    "src/demap/paper/biencoder/phase0.py": {"kind": "phase0_validator", "status": "canonical"},
    "src/demap/paper/biencoder/select.py": {"kind": "selector", "status": "canonical"},
    "src/demap/experiments/modern_trainer.py": {"kind": "trainer", "status": "canonical",
                                                "note": "SentenceTransformerTrainer (honest fp16/bf16)"},
    # Reusable internal components
    "src/demap/experiments/finetune_phase1.py": {"kind": "trainer_host", "status": "reusable_component",
                                                 "note": "hosts modern_trainer branch; old_fit path is historical"},
    "src/demap/paper_phase1_screening.py": {"kind": "lib", "status": "reusable_component"},
    "src/demap/paper_selection.py": {"kind": "lib", "status": "reusable_component"},
    # Historical reproduction only (preserved; --allow-legacy)
    "old_fit": {"kind": "trainer_path", "status": "historical_reproduction",
                "note": "FitMixin.old_fit; torch.cuda.amp float16-only; cannot do true bf16"},
    "scripts/paper_generate_phase1_screening_tasks.py": {"kind": "generator", "status": "historical_reproduction",
                                                         "note": "Section 4.3 screen generator (common-anchor era)"},
    "src/demap/pipeline/paper_phase1_pipeline.py": {"kind": "orchestrator", "status": "historical_reproduction"},
    "slurm/finetune_phase1_task_manifest_array.sbatch": {"kind": "launcher", "status": "historical_reproduction"},
    "slurm/finetune_phase1_array.sbatch": {"kind": "launcher", "status": "historical_reproduction"},
    "scripts/paper_generate_phase1_narrowing_configs.py": {"kind": "generator", "status": "historical_reproduction"},
    # ---- Phase 2 (added 2026-07-26)
    "src/demap/paper/biencoder/phase2.py": {"kind": "generator", "status": "canonical",
                                            "note": "winner-manifest-driven Phase 2 generation"},
    "src/demap/paper/biencoder/hardneg.py": {"kind": "miner", "status": "canonical",
                                             "note": "parent-bound, versioned, no-overwrite mining"},
    "slurm/paper_biencoder_phase2_canonical_array.sbatch": {"kind": "launcher", "status": "canonical",
                                                            "note": "token-guarded Phase 2 array"},
    "src/demap/experiments/finetune_phase2.py": {"kind": "trainer_host", "status": "reusable_component",
                                                 "note": "hardened: explicit use_amp=False fp32 contract, "
                                                         "grad-accum 1, val_dev-only, no positive-as-negative"},
    "artifacts/phase2_hardneg_mining": {"kind": "artifacts", "status": "historical_reproduction",
                                        "note": "mined from the OLD common-anchor Phase 1 parents; "
                                                "never reusable for the corrected run; never overwrite"},
    "artifacts/phase2_finetuning": {"kind": "artifacts", "status": "historical_reproduction",
                                    "note": "216 runs on the OLD parents; all-MPNet winner remains valid"},
    "scripts/paper_generate_phase2_configs.py": {"kind": "generator", "status": "historical_reproduction",
                                                 "note": "duplicates grid derivation; no fail-closed guards"},
    "scripts/paper_generate_phase2_tasks.py": {"kind": "generator", "status": "historical_reproduction"},
    "slurm/paper_finetune_phase2_task_manifest_array.sbatch": {"kind": "launcher",
                                                               "status": "historical_reproduction",
                                                               "note": "unguarded; superseded by the "
                                                                       "token-guarded canonical launcher"},
    "slurm/paper_hardneg_task_manifest_array.sbatch": {"kind": "launcher",
                                                       "status": "historical_reproduction"},
    "scripts/paper_select_phase2_winner.py": {"kind": "selector", "status": "conflicting_unsafe",
                                              "note": "historical Phase 2 selector; not the canonical rule"},
    "scripts/paper_select_phase2_winners_by_model.py": {"kind": "selector", "status": "conflicting_unsafe",
                                                        "note": "ranks R@5 -> MRR@100 (Recall@10 MISSING) "
                                                                "with DEFAULT_TIE_MARGIN=0.005 and complexity "
                                                                "tie-breaks; the canonical rule is "
                                                                "phase2_seed_mean_lexicographic_v1 with no "
                                                                "tie margin"},
    "DEFAULT_TIE_MARGIN": {"kind": "policy", "status": "conflicting_unsafe",
                           "note": "0.005 near-tie band; NOT used by any canonical selection rule"},

    # Conflicting / unsafe for canonical use
    "common_anchor_phase1_generator": {"kind": "generator", "status": "conflicting_unsafe",
                                       "note": "fixed one anchor for all models; must not drive new canonical runs"},
    "scripts/paper_select_phase1_winner.py": {"kind": "selector", "status": "conflicting_unsafe",
                                             "note": "competing 'official' winner selector"},
    "scripts/paper_select_phase1_representation_winner.py": {"kind": "selector", "status": "conflicting_unsafe"},
    "scripts/paper_select_phase1_winners_by_model.py": {"kind": "selector", "status": "conflicting_unsafe",
                                                       "note": "seed-mean+tie-margin rule; NOT the operative rule"},
}

CANONICAL_TRAINER = "modern_trainer"
_LEGACY_TRAINERS = {"old_fit", "fit", "auto"}


def warn_legacy(name: str) -> None:
    print(
        f"\n*** HISTORICAL-ONLY PATH: {name} ***\n"
        f"    This is a legacy reproduction path and is NOT a canonical paper run.\n"
        f"    Reached only because --allow-legacy was supplied.\n",
        flush=True,
    )


def assert_canonical_trainer(trainer: str, *, allow_legacy: bool = False) -> None:
    """Canonical Phase 1 must use the modern trainer. old_fit/fit/auto are historical."""
    t = str(trainer or "").strip().lower()
    if t == CANONICAL_TRAINER:
        return
    if t in _LEGACY_TRAINERS:
        if allow_legacy:
            warn_legacy(f"trainer={t}")
            return
        raise LegacyBlocked(
            f"[quarantine] trainer '{t}' is historical-reproduction only. Canonical paper Phase 1 "
            f"requires trainer '{CANONICAL_TRAINER}'. Pass --allow-legacy to reproduce a historical run."
        )
    raise LegacyBlocked(f"[quarantine] unknown trainer '{trainer}'.")


def assert_protocol_not_legacy(proto: dict, *, allow_legacy: bool = False) -> None:
    """A canonical protocol must lock the canonical (modern) trainer."""
    trainer = (proto.get("phase1") or {}).get("trainer", CANONICAL_TRAINER)
    assert_canonical_trainer(trainer, allow_legacy=allow_legacy)


def block_legacy_entry(name: str, *, allow_legacy: bool) -> None:
    """Guard placed at historical entry points."""
    if allow_legacy:
        warn_legacy(name)
        return
    raise LegacyBlocked(
        f"[quarantine] '{name}' is a historical/legacy path and is blocked by default. "
        f"Use `demap paper biencoder` for new canonical runs, or pass --allow-legacy to reproduce history."
    )
