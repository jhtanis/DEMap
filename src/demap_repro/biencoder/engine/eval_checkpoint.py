#!/usr/bin/env python3
"""eval_checkpoint.py

Evaluate a saved SentenceTransformers model directory (typically a Phase 1 checkpoint)
using DEMAP's standard evaluation protocol.

This module exists to support a clean workflow:

  1) Train a run with per-epoch checkpoints enabled (Phase 1 fine-tune).
  2) Select the epoch using *validation* metrics only.
  3) Evaluate the selected checkpoint on test/holdouts in a separate step.

Example:

  demap eval-checkpoint \
    --config configs/experiments/finetune_phase1.yaml \
    --model-path artifacts/runs/<run_id>/checkpoints/checkpoint-1234 \
    --splits val,test

By default, results are written to:

  <runs_dir>/<timestamp>__evalckpt__<model_slug>/

and include the standard artifacts:

  run_config.json
  metrics.json
  rankings.parquet
  reliability.csv
  ece.json
  failures_sample.csv

Notes
-----
- This command does *not* train anything.
- Splits are evaluated according to the updated evaluation protocol already implemented
  in :mod:`demap_repro.biencoder.engine.finetune_phase1` / :mod:`demap_repro.biencoder.engine.baseline_grid`.

"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from demap_repro.utils.config import deep_merge, load_config
from demap_repro.utils.io import write_json
from demap_repro.utils.paths import ensure_dir

# Reuse the standardized evaluator / run-writer from finetune_phase1.
from demap_repro.biencoder.engine import finetune_phase1 as ft1


def main(argv: Optional[List[str]] = None) -> None:
    argv_list = list(argv) if argv is not None else None
    if argv_list is None:
        import sys

        argv_list = sys.argv[1:]

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML config file (can be repeated). Expected key: finetune_phase1.",
    )
    ap.add_argument(
        "--model-path",
        required=True,
        help="Path to a SentenceTransformers model directory (e.g., a fine-tune checkpoint dir).",
    )
    ap.add_argument(
        "--out-run-dir",
        default=None,
        help="Where to write evaluation artifacts. Default: create a new directory under runs_dir.",
    )
    ap.add_argument(
        "--splits",
        default=None,
        help=(
            "Comma-separated list of split names to evaluate (e.g., 'val,test'). "
            "Default: evaluate all splits found in splits_dir."
        ),
    )
    ap.add_argument("--rerank-mode", default=None, help="Optional override for the rerank mode (R0/R1/R2).")
    ap.add_argument("--hybrid-alpha", type=float, default=None, help="Optional override for hybrid reranking alpha (semantic weight in R2).")
    ap.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")

    args = ap.parse_args(argv_list)

    # Merge configs
    cfg: Dict[str, Any] = {}
    for cpath in args.config:
        cfg = deep_merge(cfg, load_config(cpath))
    cfg = cfg.get("finetune_phase1", cfg) if cfg else {}

    def _get(key: str, default: Any = None) -> Any:
        return cfg.get(key, default) if isinstance(cfg, dict) else default

    # Required paths from config
    cde_master_enriched = Path(_get("cde_master_enriched"))
    splits_dir = Path(_get("splits_dir"))
    artifacts_dir = Path(_get("artifacts_dir", "artifacts"))
    runs_dir = Path(_get("runs_dir", os.path.join("artifacts", "runs")))

    if not cde_master_enriched.exists():
        raise SystemExit(f"Missing cde_master_enriched: {cde_master_enriched}")
    if not splits_dir.exists():
        raise SystemExit(f"Missing splits_dir: {splits_dir}")

    # Representation settings (default to Phase 1 decisions)
    recipe = str(_get("recipe", "v1_v2_v3_v5"))
    cde_format = str(_get("cde_format", "labeled"))
    query_variant = str(_get("query_variant", "Q3"))
    rerank_mode = str(args.rerank_mode or _get("rerank_mode", "R0"))
    hybrid_alpha = float(args.hybrid_alpha) if args.hybrid_alpha is not None else float(_get("hybrid_alpha", 0.5))
    sep = str(_get("sep", " | "))

    recipe_configs = _get("recipe_configs") if isinstance(_get("recipe_configs"), dict) else None

    # Eval settings
    ev = _get("eval", {}) if isinstance(_get("eval"), dict) else {}
    top_k = int(ev.get("top_k", 200))
    k_values = ev.get("k_values", [1, 5, 10, 20])
    output_top_k = int(ev.get("output_top_k", 20))
    block_size = int(ev.get("block_size", 2048))
    encode_batch_size = int(ev.get("encode_batch_size", 64))
    normalize_embeddings = bool(ev.get("normalize_embeddings", True))

    # Device
    device = str(args.device).strip().lower()
    if device == "auto":
        device = ft1._auto_device()
    if device not in {"cpu", "cuda", "mps"}:
        raise SystemExit(f"Unknown device: {args.device}. Expected auto|cpu|cuda|mps")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Missing --model-path: {model_path}")

    # Output directory
    if args.out_run_dir:
        run_dir = Path(args.out_run_dir)
    else:
        ensure_dir(runs_dir)
        ts = time.strftime("%Y%m%d_%H%M%S")
        model_slug = ft1._safe_cache_slug(str(model_path))
        run_id = f"{ts}__evalckpt__{model_slug}"
        run_dir = runs_dir / run_id

    ensure_dir(run_dir)
    ensure_dir(artifacts_dir)

    only_splits: Optional[List[str]] = None
    if args.splits:
        only_splits = ft1._parse_csv_list(str(args.splits))

    # Record config used for this evaluation run
    run_config: Dict[str, Any] = {
        "run_type": "eval_checkpoint",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_path": str(model_path),
        "device": device,
        "only_splits": only_splits,
        "paths": {
            "cde_master_enriched": str(cde_master_enriched),
            "splits_dir": str(splits_dir),
            "artifacts_dir": str(artifacts_dir),
            "runs_dir": str(runs_dir),
        },
        "representation": {
            "query_variant": query_variant,
            "recipe": recipe,
            "cde_format": cde_format,
            "rerank_mode": rerank_mode,
            "hybrid_alpha": hybrid_alpha,
            "sep": sep,
            "recipe_configs": recipe_configs,
        },
        "eval": {
            "top_k": top_k,
            "k_values": list(k_values) if isinstance(k_values, (list, tuple)) else k_values,
            "output_top_k": output_top_k,
            "block_size": block_size,
            "encode_batch_size": encode_batch_size,
            "normalize_embeddings": normalize_embeddings,
        },
        "config_files": list(args.config),
    }
    write_json(run_config, run_dir / "run_config.json")

    # Delegate to the shared evaluator.
    ft1._evaluate_and_write_run(
        run_dir=run_dir,
        model_name_or_path=str(model_path),
        device=device,
        cde_master_enriched=cde_master_enriched,
        splits_dir=splits_dir,
        recipe=recipe,
        cde_format=cde_format,
        query_variant=query_variant,
        rerank_mode=rerank_mode,
        sep=sep,
        recipe_configs=recipe_configs,
        top_k=top_k,
        k_values=[int(x) for x in (k_values or [])],
        output_top_k=output_top_k,
        block_size=block_size,
        batch_size=encode_batch_size,
        normalize_embeddings=normalize_embeddings,
        hybrid_alpha=hybrid_alpha,
        only_splits=only_splits,
        artifacts_dir=artifacts_dir,
    )

    print(f"[eval-checkpoint] Wrote artifacts to: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
