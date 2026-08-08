#!/usr/bin/env python3
"""mine_hardneg.py

Mine hard negatives for Phase 2 contrastive fine-tuning.

This command produces a reusable Parquet artifact that stores, for each
*unique normalized query text* in the training split:

  - the set of gold CDE ids observed for that query (to avoid false negatives)
  - the top-K retrieved candidate CDE ids + scores under a miner checkpoint

The resulting artifact is consumed by :mod:`demap_repro.biencoder.engine.finetune_phase2`.

Design goals
------------
- Deterministic and reproducible.
- Conservative about false negatives: gold sets are computed at the
  **normalized query_text_q3** level (duplicates are expected).
- Fast on GPU: encode catalog once, encode unique queries once, then compute
  top-K blockwise dot products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from demap_repro.biencoder.engine import baseline_grid as bg
from demap_repro.text.normalize import normalize_query_text
from demap_repro.text.recipes import build_catalog, is_valid_recipe
from demap_repro.utils.config import load_config
from demap_repro.utils.io import write_json
from demap_repro.utils.paths import ensure_dir


def _slug(s: str) -> str:
    return str(s).replace("/", "__").replace(" ", "_").replace(":", "_")


def _safe_cache_slug(s: str, max_len: int = 80) -> str:
    """Short, filesystem-safe slug for cache directories."""
    raw = str(s)
    base = Path(raw).name if ("/" in raw or "\\" in raw) else raw
    base_slug = _slug(base)
    h = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
    out = f"{base_slug}__{h}" if base_slug else h
    if len(out) <= max_len:
        return out
    keep = max(1, max_len - (len(h) + 2))
    return f"{base_slug[:keep]}__{h}"


def _auto_device() -> str:
    try:
        import torch  # type: ignore

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_sentence_transformer(model_name_or_path: str, *, device: str):
    from demap_repro.biencoder.engine.st_loader import load_sentence_transformer

    return load_sentence_transformer(model_name_or_path, device=device)


def _encode_np(model, texts: Sequence[str], *, batch_size: int, normalize: bool) -> np.ndarray:
    emb = model.encode(
        list(texts),
        batch_size=int(batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(normalize),
    )
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    if bool(normalize):
        n = np.linalg.norm(emb, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        emb = emb / n
    return emb


def _coerce_cde_id(df: pd.DataFrame) -> pd.DataFrame:
    if "cde_id" in df.columns:
        return df
    if "cde_publicid" in df.columns and "cde_version" in df.columns:
        d = df.copy()
        d["cde_id"] = d["cde_publicid"].astype(str) + "::" + d["cde_version"].astype(str)
        return d
    raise KeyError("Split missing cde_id (and missing cde_publicid/cde_version)")


def _apply_config_overrides(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    """Fill unset CLI args from cfg['mine_hardneg'] if present."""

    block = cfg.get("mine_hardneg") if isinstance(cfg.get("mine_hardneg"), dict) else {}
    if not block:
        return

    def flag_present(name: str) -> bool:
        # Very small compatibility helper: argparse doesn't expose a clean way to see
        # whether a value came from a CLI flag. We treat None as unset.
        return getattr(args, name) is not None

    # Paths
    if args.cde_master_enriched is None:
        args.cde_master_enriched = block.get("cde_master_enriched")
    if args.splits_dir is None:
        args.splits_dir = block.get("splits_dir")
    if args.out_parquet is None:
        args.out_parquet = block.get("out_parquet")
    if getattr(args, 'out_dir', None) is None:
        args.out_dir = block.get("out_dir")
    if getattr(args, 'miner_tag', None) is None:
        args.miner_tag = block.get("miner_tag")
    if args.artifacts_dir is None:
        args.artifacts_dir = block.get("artifacts_dir", "artifacts")
    if getattr(args, 'stage_tag', None) is None:
        args.stage_tag = block.get("stage_tag")

    # Representation
    if args.model_name_or_path is None:
        args.model_name_or_path = block.get("model_name_or_path")
    if args.query_variant is None:
        args.query_variant = block.get("query_variant", "Q3")
    if args.recipe is None:
        args.recipe = block.get("recipe", "v1_v2_v3_v5")
    if args.cde_format is None:
        args.cde_format = block.get("cde_format", "labeled")
    if args.sep is None:
        args.sep = block.get("sep", " | ")

    # Mining knobs
    if args.split_name is None:
        args.split_name = block.get("split_name", "train")
    if args.top_k is None:
        args.top_k = int(block.get("top_k", 200))
    if args.block_size is None:
        args.block_size = int(block.get("block_size", 2048))
    if args.encode_batch_size is None:
        args.encode_batch_size = int(block.get("encode_batch_size", 64))
    if args.normalize_embeddings is None:
        args.normalize_embeddings = bool(block.get("normalize_embeddings", True))
    if args.device is None:
        args.device = block.get("device", "auto")

    # Recipe configs are passed through as-is.
    if getattr(args, "recipe_configs", None) is None:
        args.recipe_configs = block.get("recipe_configs")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="demap mine-hardneg",
        description="Mine top-K hard negative candidates per train query using a miner checkpoint.",
    )

    p.add_argument("--config", type=str, default=None, help="YAML config file (expects mine_hardneg: block)")

    # Paths
    p.add_argument("--cde-master-enriched", dest="cde_master_enriched", type=str, default=None)
    p.add_argument("--splits-dir", dest="splits_dir", type=str, default=None)
    p.add_argument("--out-parquet", dest="out_parquet", type=str, default=None)
    p.add_argument("--out-dir", dest="out_dir", type=str, default=None, help="Output directory for mined artifacts (used if --out-parquet is not set)")
    p.add_argument("--miner-tag", dest="miner_tag", type=str, default=None, help="Short tag used to name output directory when --out-dir is not set")
    p.add_argument("--artifacts-dir", dest="artifacts_dir", type=str, default=None)
    p.add_argument("--stage-tag", dest="stage_tag", type=str, default=None, help="Optional stage tag used for default output locations.")

    # Miner
    p.add_argument(
        "--model-name-or-path",
        dest="model_name_or_path",
        type=str,
        default=None,
        help="SentenceTransformers model name or local path (e.g., Phase 1 checkpoint dir)",
    )
    p.add_argument("--device", type=str, default=None, help="auto|cuda|mps|cpu")

    # Representation (must match downstream training)
    p.add_argument("--query-variant", type=str, default=None, help="Q1|Q2|Q3 (maps to query_text column)")
    p.add_argument("--recipe", type=str, default=None, help="CDE recipe (e.g., v1_v2_v3_v5)")
    p.add_argument("--cde-format", type=str, default=None, help="labeled|raw")
    p.add_argument("--sep", type=str, default=None, help="Separator used in recipe concatenation")

    # Mining
    p.add_argument("--split-name", type=str, default=None, help="Which split to mine from (default: train)")
    p.add_argument("--top-k", dest="top_k", type=int, default=None)
    p.add_argument("--block-size", dest="block_size", type=int, default=None)
    p.add_argument("--encode-batch-size", dest="encode_batch_size", type=int, default=None)
    p.add_argument(
        "--normalize-embeddings",
        dest="normalize_embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would be mined and exit")

    args = p.parse_args(argv)

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)
        _apply_config_overrides(args, cfg)

    # Validate required.
    if args.cde_master_enriched is None:
        raise SystemExit("Missing --cde-master-enriched (or mine_hardneg.cde_master_enriched in config)")
    if args.splits_dir is None:
        raise SystemExit("Missing --splits-dir (or mine_hardneg.splits_dir in config)")
    if args.model_name_or_path is None:
        raise SystemExit("Missing --model-name-or-path (or mine_hardneg.model_name_or_path in config)")

    if not is_valid_recipe(str(args.recipe)):
        raise SystemExit(f"Unknown recipe: {args.recipe}")

    device = str(args.device or "auto").strip().lower()
    if device == "auto":
        device = _auto_device()
    if device not in {"cpu", "cuda", "mps"}:
        raise SystemExit(f"Unknown device: {args.device}. Expected auto|cpu|cuda|mps")

    splits_dir = Path(args.splits_dir)
    splits = bg._load_splits(splits_dir)
    split_name = str(args.split_name or "train").strip()
    split_path = splits.get(split_name)
    if split_path is None:
        raise SystemExit(f"Split '{split_name}' not found in splits_dir={splits_dir}")

    # Derive output location. Prefer explicit out_parquet; otherwise use out_dir; otherwise
    # default to <artifacts_dir>/hardneg/<miner_tag>/<split>_topk<K>.parquet.
    miner_tag = str(getattr(args, 'miner_tag', None) or '').strip()
    if miner_tag:
        miner_tag = _slug(miner_tag).strip('_') or 'miner'
    else:
        miner_tag = _safe_cache_slug(str(args.model_name_or_path))

    artifacts_dir = Path(args.artifacts_dir or 'artifacts')
    stage_root = _slug(str(getattr(args, 'stage_tag', '') or '')).strip('_')
    default_root = artifacts_dir / (stage_root or 'hardneg') / miner_tag
    out_dir = Path(getattr(args, 'out_dir', None) or default_root)
    if args.out_parquet:
        out_path = Path(args.out_parquet)
    else:
        out_path = out_dir / f"{split_name}_topk{int(args.top_k or 200)}.parquet"

    # Ensure downstream code sees the resolved path (useful for logging / dry-run).
    args.out_parquet = str(out_path)

    # Load data
    query_col = bg.QUERY_VARIANT_TO_COL.get(str(args.query_variant), str(args.query_variant))
    cols_needed = [query_col, "cde_id", "cde_publicid", "cde_version"]
    try:
        df = pd.read_parquet(split_path, columns=[c for c in cols_needed if c])
    except Exception:
        df = pd.read_parquet(split_path)
    df = _coerce_cde_id(df)
    if query_col not in df.columns:
        raise SystemExit(f"Split missing query column '{query_col}'. Available: {list(df.columns)[:50]}")

    # Build unique query table: query_key -> query_text + gold ids.
    q_text = df[query_col].fillna("").astype(str)
    q_text = q_text.map(lambda s: str(s).strip())
    d2 = df.copy()
    d2["_query_text"] = q_text
    d2 = d2[d2["_query_text"] != ""]
    d2["query_key"] = d2["_query_text"].map(normalize_query_text)

    gold = (
        d2.groupby("query_key")["cde_id"]
        .apply(lambda s: sorted({str(x) for x in s.dropna().astype(str).tolist() if str(x).strip()}))
        .reset_index()
        .rename(columns={"cde_id": "gold_cde_ids"})
    )
    # Representative surface form.
    rep = (
        d2.groupby("query_key")["_query_text"]
        .first()
        .reset_index()
        .rename(columns={"_query_text": "query_text"})
    )
    uq = gold.merge(rep, on="query_key", how="left")

    n_unique = int(len(uq))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "split": split_name,
                    "n_rows": int(len(df)),
                    "n_unique_query_keys": n_unique,
                    "top_k": int(args.top_k),
                    "query_col": query_col,
                    "model_name_or_path": str(args.model_name_or_path),
                    "miner_tag": miner_tag,
                    "out_parquet": str(out_path),
                },
                indent=2,
            )
        )
        return

    # Build catalog texts
    master = pd.read_parquet(Path(args.cde_master_enriched))
    recipe_configs = getattr(args, "recipe_configs", None)
    if recipe_configs is not None and not isinstance(recipe_configs, dict):
        recipe_configs = None

    catalog_df = build_catalog(
        master,
        recipe=str(args.recipe),
        cde_format=str(args.cde_format),
        sep=str(args.sep),
        recipe_configs=recipe_configs,
    )
    cde_texts = catalog_df["cde_text"].astype(str).tolist()

    # Load miner model
    model = _load_sentence_transformer(str(args.model_name_or_path), device=device)
    model_slug = _safe_cache_slug(str(args.model_name_or_path))

    # Catalog embeddings (cached under artifacts/embeddings)
    artifacts_dir = Path(args.artifacts_dir or "artifacts")
    embeddings_dir = artifacts_dir / "embeddings"
    ensure_dir(embeddings_dir)
    cde_ids, cde_emb, _meta = bg._load_or_build_catalog_embeddings(
        model=model,
        model_slug=model_slug,
        embeddings_dir=embeddings_dir,
        master=master,
        recipe=str(args.recipe),
        cde_format=str(args.cde_format),
        sep=str(args.sep),
        recipe_configs=recipe_configs,
        batch_size=int(args.encode_batch_size),
        normalize=bool(args.normalize_embeddings),
    )

    # Encode unique queries
    q_unique_texts = uq["query_text"].fillna("").astype(str).tolist()
    q_emb = _encode_np(
        model,
        q_unique_texts,
        batch_size=int(args.encode_batch_size),
        normalize=bool(args.normalize_embeddings),
    )

    # Retrieve top-K
    idx_top, score_top = bg._topk_blockwise(
        query_emb=q_emb,
        cde_emb=cde_emb,
        top_k=int(args.top_k),
        block_size=int(args.block_size),
    )

    # Materialize candidates as lists
    cde_ids_list = cde_ids.astype(str).tolist()
    cand_ids: List[List[str]] = []
    cand_scores: List[List[float]] = []
    ranks = list(range(1, int(args.top_k) + 1))

    for row_i in range(idx_top.shape[0]):
        idxs = idx_top[row_i].tolist()
        ids = [cde_ids_list[int(j)] if int(j) >= 0 else "" for j in idxs]
        sc = [float(x) for x in score_top[row_i].tolist()]
        cand_ids.append(ids)
        cand_scores.append(sc)

    out = uq.copy()
    out["miner_model_name_or_path"] = str(args.model_name_or_path)
    out["mined_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["top_k"] = int(args.top_k)
    out["candidates_cde_id"] = cand_ids
    out["candidates_score"] = cand_scores
    out["candidates_rank"] = [ranks for _ in range(n_unique)]

    ensure_dir(out_path.parent)
    out.to_parquet(out_path, index=False)

    # Small manifest alongside the parquet.
    manifest = {
        "created_at": out["mined_at"].iloc[0] if len(out) else time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": split_name,
        "query_variant": str(args.query_variant),
        "query_col": query_col,
        "recipe": str(args.recipe),
        "cde_format": str(args.cde_format),
        "sep": str(args.sep),
        "top_k": int(args.top_k),
        "block_size": int(args.block_size),
        "encode_batch_size": int(args.encode_batch_size),
        "normalize_embeddings": bool(args.normalize_embeddings),
        "miner_model_name_or_path": str(args.model_name_or_path),
        "miner_tag": miner_tag,
        "out_dir": str(out_dir),
        "n_unique_query_keys": int(n_unique),
        "n_rows_in_split": int(len(df)),
        "parquet": str(out_path),
    }
    write_json(manifest, out_path.with_suffix(out_path.suffix + ".manifest.json"))

    print(f"[mine-hardneg] wrote: {out_path} (n_unique_queries={n_unique}, top_k={int(args.top_k)})")


if __name__ == "__main__":  # pragma: no cover
    main()
