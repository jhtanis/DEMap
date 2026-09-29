"""Canonical CLI: `demap paper biencoder` -- the one supported route for new paper work."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import List, Optional

from demap_repro.biencoder import hardneg as _hn
from demap_repro.biencoder import phase0 as _p0
from demap_repro.biencoder import phase1 as _p1
from demap_repro.biencoder import phase1_results as _p1r
from demap_repro.biencoder import phase2 as _p2
from demap_repro.biencoder import protocol as _proto
from demap_repro.biencoder import quarantine as _q
from demap_repro.biencoder import winners as _win

DEFAULT_PROTOCOL = "configs/paper/biencoder_protocol_v1.yaml"
DEFAULT_WINNERS = "configs/paper/section4_3_winners_v1.yaml"
DEFAULT_PHASE1_MANIFEST = "configs/paper/phase1_winners_for_phase2_v1.yaml"


def _selected_paper_run(args) -> None:
    """Run only the manuscript-retained FT-MPNet seed from the canonical job."""
    from demap_repro.config.paper import (
        resolve_paper_artifact_path, resolve_paper_data_path, stage_job,
    )

    stage = "ft_mpnet_phase1" if args.stage == "phase1" else "ft_mpnet_phase2"
    job = stage_job(stage, args.paper_config)
    if args.dry_run:
        print(json.dumps(job, indent=2, sort_keys=True))
        print("[paper biencoder] selected-final dry-run: no files written, no training run.")
        return
    if args.stage == "phase1":
        from demap_repro.biencoder.engine import finetune_phase1
        block = job["finetune_phase1"]
        train = block["train"]
        finetune_phase1.main([
            "--cde-master-enriched", str(resolve_paper_data_path(block["cde_master_enriched"])),
            "--splits-dir", str(resolve_paper_data_path(block["splits_dir"])),
            "--artifacts-dir", str(resolve_paper_artifact_path(block["artifacts_dir"])),
            "--runs-dir", "auto", "--model-name", block["model_name"],
            "--model-revision", block["model_revision"],
            "--base-model-id", block["base_model_id"],
            "--query-variant", block["query_variant"], "--recipe", block["recipe"],
            "--cde-format", block["cde_format"], "--losses", block["loss"],
            "--seeds", str(job["selection"]["retained_seed"]),
            "--lrs", str(train["lrs"][0]), "--temperatures", str(train["temperatures"][0]),
            "--epochs", str(train["epochs"][0]),
            "--batch-sizes", str(train["batch_sizes"][0]),
            "--max-seq-length", str(train["max_seq_length"]),
            "--fit-api", train["fit_api"], "--no-bf16", "--no-fp16",
            "--device", "cuda", "--eval-splits", "val_dev",
        ])
        return

    from demap_repro.biencoder.engine import finetune_phase2
    block = job["finetune_phase2"]
    train = block["train"]
    parent = args.parent_checkpoint or block["init_model_name_or_path"]
    parent = str(resolve_paper_artifact_path(parent))
    if args.mine_hardneg:
        mining = block["hard_negative_mining"]
        splits_dir = resolve_paper_data_path(block["splits_dir"])
        catalog = resolve_paper_data_path(block["cde_master_enriched"])
        identity = _hn.build_identity(
            model_id="all-mpnet-base-v2",
            parent_checkpoint=parent,
            parent_checkpoint_sha256=_hn.checkpoint_hash(parent),
            query_variant=block["query_variant"], recipe=block["recipe"],
            cde_format=block["cde_format"],
            train_split_sha256=_hn.sha256_file(
                str(splits_dir / f"{mining['mine_split']}.parquet")),
            catalog_sha256=_hn.sha256_file(str(catalog)),
            top_k=mining["top_k"], n_negatives=train["nneg"],
            mine_split=mining["mine_split"], code_commit=_git_commit(),
        )
        result = _hn.mine(
            identity=identity,
            root=str(resolve_paper_artifact_path(mining["artifact_root"])),
            splits_dir=str(splits_dir), catalog_path=str(catalog),
            recipe_configs=block["recipe_configs"],
            embeddings_dir=str(resolve_paper_artifact_path(mining["embeddings_cache"])),
            overwrite=args.overwrite_hardneg, device="cuda",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if not args.mined_parquet:
        raise SystemExit(
            "[paper biencoder] selected Phase 2 requires --mined-parquet produced with "
            "hard_top25 ranks 1-25 from the retained Phase-1 checkpoint")
    finetune_phase2.main([
        "--cde-master-enriched", str(resolve_paper_data_path(block["cde_master_enriched"])),
        "--splits-dir", str(resolve_paper_data_path(block["splits_dir"])),
        "--artifacts-dir", str(resolve_paper_artifact_path(block["artifacts_dir"])),
        "--runs-dir", "auto", "--base-model-id", block["base_model_id"],
        "--init-model-name-or-path", parent,
        "--miner-model-name-or-path", parent,
        "--mined-parquet", args.mined_parquet,
        "--query-variant", block["query_variant"], "--recipe", block["recipe"],
        "--cde-format", block["cde_format"], "--losses", block["loss"],
        "--seeds", str(job["selection"]["retained_seed"]),
        "--lrs", str(train["lrs"][0]), "--temperatures", str(train["temperatures"][0]),
        "--epochs", str(train["epochs"][0]),
        "--batch-sizes", str(train["batch_sizes"][0]),
        "--strategies", train["strategies"][0], "--nneg", str(train["nneg"]),
        "--hard-band", "1-25", "--device", "cuda", "--require-gpu",
        "--eval-splits", "val_dev",
    ])


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        return True


def _phase2(args, *, check_rows: bool) -> None:
    """Canonical Phase 2: winner-manifest-driven generation, mining, validation, submission."""
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        raise SystemExit("[paper biencoder] --stage phase2 requires --models")

    dirty = _git_dirty()
    commit = _git_commit()
    canonical = not (args.allow_winner_override or args.allow_legacy)

    # ---- mine the parent-bound hard negatives (separate, explicit action)
    if args.mine_hardneg:
        summary = _p2.generate_phase2(
            protocol_path=args.protocol, phase1_manifest_path=args.phase1_manifest,
            model_ids=model_ids, precision=args.precision, out_root=args.out_root,
            hardneg_root=args.hardneg_root, allow_winner_override=args.allow_winner_override,
            allow_legacy=args.allow_legacy, code_commit=commit, dirty=dirty,
            check_rows=check_rows, write=False, require_hardneg=False)
        proto = _proto.load_protocol(args.protocol)
        p2 = proto["phase2"]
        root = args.hardneg_root or p2["hardneg"]["artifact_root"]
        for mid, mv in summary["models"].items():
            ident = json.loads(json.dumps(
                _hn.build_identity(
                    model_id=mid, parent_checkpoint=mv["parent_checkpoint"],
                    parent_checkpoint_sha256=mv["parent_checkpoint_sha256"],
                    query_variant=mv["query_variant"], recipe=mv["recipe"],
                    cde_format=mv["cde_format"],
                    train_split_sha256=_proto.sha256_file(f"{proto['data']['splits_dir']}/train.parquet"),
                    catalog_sha256=_proto.sha256_file(proto["data"]["production_catalog"]),
                    top_k=int(p2["hardneg"]["top_k"]), n_negatives=int(p2["hardneg"]["n_negatives"]),
                    mine_split=str(p2["hardneg"]["mine_split"]), code_commit=commit)))
            res = _hn.mine(
                identity=ident, root=root, splits_dir=proto["data"]["splits_dir"],
                catalog_path=proto["data"]["production_catalog"],
                recipe_configs={"placeholder_policy": "placeholder",
                                "short_name_placeholder": "<MISSING_SHORT_NAME>",
                                "pv_placeholder": "<MISSING_PV_SUMMARY>",
                                "v1": {"filter_numeric_only": True,
                                       "filter_versioned_id_short_name": True}},
                overwrite=args.overwrite_hardneg, dry_run=args.dry_run)
            print(f"[paper biencoder] hardneg {mid}: {res['status']} -> {res['parquet']}")
        return

    if args.submit and canonical and dirty and not args.allow_dirty:
        raise SystemExit(
            "[paper biencoder] refusing to submit a canonical paper run from a DIRTY working "
            "tree. Commit your changes, or pass --allow-dirty for a noncanonical dev run.")

    write = args.dry_run or args.submit
    summary = _p2.generate_phase2(
        protocol_path=args.protocol, phase1_manifest_path=args.phase1_manifest,
        model_ids=model_ids, precision=args.precision, out_root=args.out_root,
        hardneg_root=args.hardneg_root, allow_winner_override=args.allow_winner_override,
        allow_legacy=args.allow_legacy, code_commit=commit, dirty=dirty,
        check_rows=check_rows, write=write, require_hardneg=bool(args.submit))

    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=1))
    for mid, mv in summary["models"].items():
        print(f"  {mid}: {mv['n_runs']} runs  loss={mv['inherited_loss']} "
              f"rep={mv['query_variant']}/{mv['recipe']}")
        print(f"    parent   {mv['parent_checkpoint']}")
        print(f"    parent#  {mv['parent_checkpoint_sha256']}")
        print(f"    lr grid  {[f'{x:g}' for x in mv['lr_grid']]}   temp grid {mv['temperature_grid']}")
        print(f"    hardneg  {mv['hardneg_parquet']} (present={mv['hardneg_present']})")
    print(f"\n[paper biencoder] canonical={summary['canonical']} "
          f"total_runs={summary['total_runs']} precision={args.precision} "
          f"(autocast=False, GradScaler=False, grad_accum=1)")

    if args.validate_only:
        print("[paper biencoder] --validate-only: no configs written, nothing submitted.")
        return
    if args.dry_run:
        print(f"[paper biencoder] --dry-run: wrote configs + preview manifest under "
              f"{args.out_root}. Nothing submitted.")
        return

    # ---- --submit
    token = _p2.validate_generated_submission(
        out_root=args.out_root, protocol_path=args.protocol,
        phase1_manifest_path=args.phase1_manifest, model_ids=model_ids,
        precision=args.precision)
    token_path = _p2.write_submission_token(args.out_root, token, code_commit=commit)
    print(f"[paper biencoder] submission validated: total_runs={token['total_runs']} "
          f"per_model={token['per_model']} seeds_ok={token['seeds_ok']} commit={commit}")
    print(f"[paper biencoder] wrote submission token: {token_path}")

    n = token["total_runs"]
    launcher = ("slurm/paper_biencoder_phase2_canonical_array.sbatch"
                if args.launcher.endswith("phase1_canonical_array.sbatch") else args.launcher)
    repo_dir = os.getcwd()
    sbatch_cmd = (f"REPO_DIR={repo_dir} SUBMISSION_TOKEN={token_path} OUT_ROOT={args.out_root} "
                  f"MANIFEST={args.out_root}/phase2_preview_manifest.tsv "
                  f"sbatch --array=0-{n - 1}%8 {launcher}")
    if os.environ.get("DEMAP_CONFIRM_SUBMIT") == "1":
        import subprocess as _sp
        # REPO_DIR is exported explicitly: Slurm relocates the script body, so the launcher
        # cannot derive the checkout from its own path.
        env = dict(os.environ, REPO_DIR=repo_dir, SUBMISSION_TOKEN=token_path,
                   OUT_ROOT=args.out_root,
                   MANIFEST=f"{args.out_root}/phase2_preview_manifest.tsv")
        r = _sp.run(["sbatch", f"--array=0-{n - 1}%8", launcher], env=env,
                    capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
    else:
        print("[paper biencoder] --submit prepared but NOT executed "
              "(set DEMAP_CONFIRM_SUBMIT=1 to launch).")
        print(f"[paper biencoder] submission command:\n    {sbatch_cmd}")


def main(argv: Optional[List[str]] = None) -> None:
    argv_list = list(argv) if argv is not None else list(os.sys.argv[1:])
    ap = argparse.ArgumentParser(prog="demap paper biencoder")
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--winners", default=DEFAULT_WINNERS)
    ap.add_argument("--stage", required=True, choices=["phase0", "phase1", "phase2"])
    ap.add_argument("--models", default="", help="comma-separated model IDs (phase1/phase2)")
    ap.add_argument("--precision", default=None,
                    choices=["fp16_mixed", "bf16_mixed", "fp32"],
                    help="default: fp16_mixed for phase1, fp32 for phase2")
    ap.add_argument("--out-root", default=None,
                    help="default: artifacts/phase1_canonical_v1 or artifacts/phase2_canonical_v1")
    # phase2-only
    ap.add_argument("--phase1-manifest", default=DEFAULT_PHASE1_MANIFEST,
                    help="canonical Phase 1 winner manifest (phase2 parent source)")
    ap.add_argument("--hardneg-root", default=None,
                    help="override the versioned hard-negative artifact root (phase2)")
    ap.add_argument("--mine-hardneg", action="store_true",
                    help="phase2: mine the parent-bound hard-negative artifacts, then exit")
    ap.add_argument("--overwrite-hardneg", action="store_true",
                    help="phase2: permit replacing a NON-matching mined artifact (never historical)")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-legacy", action="store_true", help="explicit historical reproduction only")
    ap.add_argument("--allow-winner-override", action="store_true", help="research-only; marks output non-canonical")
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree for a NONcanonical dev run")
    ap.add_argument("--no-check-rows", action="store_true", help="skip parquet row-count checks (fast validate)")
    ap.add_argument("--verify-submission", action="store_true",
                    help="re-validate the generated submission + token (used by the array launcher guard)")
    ap.add_argument("--launcher", default="slurm/paper_biencoder_phase1_canonical_array.sbatch")
    ap.add_argument("--paper-config", default=None,
                    help="canonical final-system manifest")
    ap.add_argument("--model-revision", default=None,
                    help="generic upstream revision; paper mode owns this value")
    ap.add_argument("--selected-final", action="store_true",
                    help="paper mode: train only the retained FT-MPNet seed/configuration")
    ap.add_argument("--parent-checkpoint", default=None,
                    help="paper selected Phase 2: path to reconstructed retained Phase-1 model")
    ap.add_argument("--mined-parquet", default=None,
                    help="paper selected Phase 2: hard_top25 mined-negative parquet")
    args = ap.parse_args(argv_list)

    # Stage-specific defaults (kept out of argparse so each stage owns its own lock).
    if args.precision is None:
        args.precision = "fp32" if args.stage == "phase2" else "fp16_mixed"
    if args.out_root is None:
        args.out_root = ("artifacts/phase2_canonical_v1" if args.stage == "phase2"
                         else "artifacts/phase1_canonical_v1")

    check_rows = not args.no_check_rows

    if args.paper_config:
        from demap_repro.config.paper import PaperConfigError, reject_conflicting_flags, validate_paper_config
        try:
            reject_conflicting_flags(
                argv_list,
                ["--protocol", "--winners", "--phase1-manifest", "--models",
                 "--precision", "--allow-legacy", "--allow-winner-override",
                 "--model-revision"],
                context="biencoder --paper-config")
            paper = validate_paper_config(args.paper_config)
        except PaperConfigError as exc:
            ap.error(str(exc))
        args.protocol = paper["references"]["biencoder_protocol"]
        args.phase1_manifest = paper["references"]["phase1_winners"]
        args.models = "all-mpnet-base-v2"
        if args.stage == "phase0":
            ap.error("final-system paper mode supports Phase 1 and Phase 2, not the Phase-0 screen")
        if not args.selected_final:
            ap.error("--paper-config requires --selected-final for the bounded final-model path")
        _selected_paper_run(args)
        return

    # Independent submission-guard used by the array launcher: refuse the array unless a valid
    # token (written by the canonical --submit path) matches the current generated state.
    if args.stage == "phase1" and args.verify_submission:
        _p1.verify_submission_token(args.out_root, expected_commit=_git_commit())
        print(f"[paper biencoder] submission token VERIFIED for {args.out_root}")
        return

    if args.stage == "phase2" and args.verify_submission:
        _p2.verify_submission_token(args.out_root, expected_commit=_git_commit())
        print(f"[paper biencoder] phase2 submission token VERIFIED for {args.out_root}")
        return

    if args.stage == "phase2":
        _phase2(args, check_rows=check_rows)
        return

    if args.stage == "phase0":
        proto = _proto.load_protocol(args.protocol)
        _q.assert_protocol_not_legacy(proto, allow_legacy=args.allow_legacy)
        res = _p0.validate_phase0(proto, check_rows=check_rows)
        print(json.dumps(res, indent=1))
        print(f"\n[paper biencoder] Phase 0 protocol VALID: {res['n_models']} models x {res['n_cells_per_model']} cells "
              f"= {res['n_total_cells']} cells. Phase 0 is NOT rerun by this command.")
        return

    # phase1
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        raise SystemExit("[paper biencoder] --stage phase1 requires --models")

    # Fail-closed guards happen inside generate_phase1 (protocol/winner/quarantine/fingerprints).
    dirty = _git_dirty()
    canonical = not (args.allow_winner_override or args.allow_legacy)
    if args.submit and canonical and dirty and not args.allow_dirty:
        raise SystemExit(
            "[paper biencoder] refusing to submit a canonical paper run from a DIRTY working tree. "
            "Commit your changes, or pass --allow-dirty for a noncanonical dev run."
        )

    write = args.dry_run or args.submit  # validate-only does not write configs
    summary = _p1.generate_phase1(
        protocol_path=args.protocol,
        winners_path=args.winners,
        model_ids=model_ids,
        precision=args.precision,
        out_root=args.out_root,
        allow_winner_override=args.allow_winner_override,
        allow_legacy=args.allow_legacy,
        code_commit=_git_commit(),
        dirty=dirty,
        check_rows=check_rows,
        write=write,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=1))
    for mid, mv in summary["models"].items():
        print(f"  {mid}: {mv['n_runs']} runs  winner={mv['winner']['query_id']}/{mv['winner']['cde_representation']} loss={mv['winner']['loss']}")
    print(f"\n[paper biencoder] canonical={summary['canonical']} total_runs={summary['total_runs']} precision={args.precision}")

    if args.validate_only:
        print("[paper biencoder] --validate-only: no configs written, nothing submitted.")
        return
    if args.dry_run:
        print(f"[paper biencoder] --dry-run: wrote configs + preview manifest under {args.out_root}. Nothing submitted.")
        return

    # --submit: full canonical validation -> token -> submit the array (guarded).
    commit = _git_commit()
    token = _p1.validate_generated_submission(
        out_root=args.out_root, protocol_path=args.protocol, winners_path=args.winners,
        model_ids=model_ids, precision=args.precision,
    )
    token_path = _p1.write_submission_token(args.out_root, token, code_commit=commit)
    print(f"[paper biencoder] submission validated: total_runs={token['total_runs']} "
          f"per_model={token['per_model']} seeds_ok={token['seeds_ok']} commit={commit}")
    print(f"[paper biencoder] wrote submission token: {token_path}")

    n = token["total_runs"]
    sbatch_cmd = (f"SUBMISSION_TOKEN={token_path} OUT_ROOT={args.out_root} "
                  f"MANIFEST={args.out_root}/phase1_preview_manifest.tsv "
                  f"sbatch --array=0-{n - 1}%8 {args.launcher}")
    if os.environ.get("DEMAP_CONFIRM_SUBMIT") == "1":
        import subprocess as _sp
        env = dict(os.environ, SUBMISSION_TOKEN=token_path, OUT_ROOT=args.out_root,
                   MANIFEST=f"{args.out_root}/phase1_preview_manifest.tsv")
        r = _sp.run(["sbatch", f"--array=0-{n - 1}%8", args.launcher], env=env, capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
    else:
        print("[paper biencoder] --submit prepared but NOT executed (set DEMAP_CONFIRM_SUBMIT=1 to launch).")
        print(f"[paper biencoder] submission command:\n    {sbatch_cmd}")
