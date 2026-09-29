#!/usr/bin/env python3
"""finetune_phase1.py

Phase 1 fine-tuning for sentence embedding retrieval.

This runner is intentionally conservative and is designed to match the
*evaluation protocol* already implemented in this repo:

  - query representation: Q3 (``query_text_q3``)
  - CDE text recipe: v1_v2_v3_v5
  - CDE format: labeled
  - rerank: none (R0)

It performs a grid of training runs (loss / LR / batch size / temperature /
epochs / seed), saves the best checkpoint per run (by validation MRR@100), and
then evaluates the best checkpoint on all splits using the standard run
artifacts:

  artifacts/runs/<run_id>/{
    run_config.json,
    metrics.json,
    rankings.parquet,
    reliability.csv,
    ece.json,
    failures_sample.csv,
    model/  (SentenceTransformers model)
  }

Notes on PV-summary "parameterization"
--------------------------------------
``query_text_q3`` is materialized during dataset construction (``demap build-queries``).
If a dataset build manifest is not available, we cannot prove which CLI flags
were used to generate Q3. In that case we log the **build-queries defaults**
from :mod:`demap_repro.data.queries` as an *assumption*, and we also log
observable PV attachment statistics from the splits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from demap_repro.biencoder.engine import baseline_grid as bg
from demap_repro.text.normalize import normalize_query_text
from demap_repro.text.recipes import build_catalog, is_valid_recipe
from demap_repro.biencoder.engine.samplers import NoDuplicateTextBatchSampler, BatchSamplerAsSampler
from demap_repro.biencoder.engine.loss_logging import LossLoggingWrapper
from demap_repro.utils.config import deep_merge, load_config
from demap_repro.utils.io import write_json
from demap_repro.utils.paths import ensure_dir


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _slug(s: str) -> str:
    return str(s).replace("/", "__").replace(" ", "_").replace(":", "_")


def _safe_cache_slug(s: str, max_len: int = 80) -> str:
    """Return a filesystem-safe slug suitable for cache directory names.

    Notes
    -----
    - On macOS, individual path components are commonly limited to 255 bytes.
      When `s` is an absolute path (e.g., a fine-tuned model output dir),
      a naive slug can easily exceed that and crash with "File name too long".
    - We keep the *basename* for readability, and append a short SHA1 suffix
      for uniqueness.
    """

    raw = str(s)
    base = Path(raw).name if ("/" in raw or "\\" in raw) else raw
    base_slug = _slug(base)
    h = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
    out = f"{base_slug}__{h}" if base_slug else h
    if len(out) <= max_len:
        return out
    # Truncate the base portion but preserve the hash suffix.
    keep = max(1, max_len - (len(h) + 2))
    return f"{base_slug[:keep]}__{h}"


def _truncate_with_hash(s: str, max_len: int = 200) -> str:
    """Truncate a run_id to avoid filesystem path-component limits."""
    s = str(s)
    if len(s) <= max_len:
        return s
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    keep = max_len - (len(h) + 1)
    if keep < 10:
        return h[:max_len]
    return f"{s[:keep]}-{h}"


def _is_existing_local_path(s: str) -> bool:
    """Return True if s resolves to an existing filesystem path."""
    try:
        return Path(str(s)).exists()
    except Exception:
        return False


def _infer_base_model_id_from_path(model_path: str) -> str:
    """
    Best-effort inference of base_model_id from a checkpoint path.
    This is only used if base_model_id is not provided.
    """
    p = Path(str(model_path))
    for part in p.parts:
        # Look for components like "sentence-transformers__all-mpnet-base-v2"
        if re.match(r"^[A-Za-z][A-Za-z0-9\-]*__.+", part):
            org, rest = part.split("__", 1)
            if org in {"artifacts", "finetune_phase1", "finetune_phase2", "runs"}:
                continue
            # Avoid matching timestamp-prefixed run_id components
            if re.match(r"^\d{8}_\d{6}$", org):
                continue
            return f"{org}/{rest}"
    return "local_model"


def _extract_initfrom_tag(model_name_or_path: str) -> Optional[str]:
    """
    If model_name_or_path is an existing checkpoint path, return a short, readable init-from tag.
    Prefer: <timestamp>__ep<k>__seed<s> extracted from the parent run_id.
    """
    if not _is_existing_local_path(model_name_or_path):
        return None

    p = Path(str(model_name_or_path))
    if p.name == "model" and p.parent is not None:
        parent = p.parent.name
    else:
        parent = p.name

    ts_m = re.match(r"^(\d{8}_\d{6})", parent)
    ep_m = re.search(r"__ep(\d+)", parent)
    seed_m = re.search(r"__seed(\d+)", parent)

    if ts_m:
        parts = [ts_m.group(1)]
        if ep_m:
            parts.append(f"ep{ep_m.group(1)}")
        if seed_m:
            parts.append(f"seed{seed_m.group(1)}")
        # Join with "__" to match existing run_id style
        return "__".join(parts)

    # Fallback: safe/truncated parent folder name
    return _safe_cache_slug(parent, max_len=80)


def _normalize_stage_tag(stage_tag: Optional[str]) -> str:
    t = str(stage_tag or "").strip()
    if not t:
        return ""
    return _slug(t).strip("_")


def _resolve_runs_dir(
    *, artifacts_dir: str, runs_dir: str, phase: str, base_model_id: str, stage_tag: Optional[str] = None
) -> Path:
    """Resolve runs_dir; supports runs_dir='auto' to write per-model runs."""
    if str(runs_dir).lower() != "auto":
        return Path(runs_dir)
    model_slug = _slug(base_model_id)
    phase_root = _normalize_stage_tag(stage_tag) or phase
    return Path(artifacts_dir) / phase_root / model_slug / "runs"


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for x in _parse_csv_list(s):
        out.append(float(x))
    return out


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for x in _parse_csv_list(s):
        out.append(int(float(x)))
    return out


def _auto_device() -> str:
    """Pick the best available device (prefers MPS on Apple, then CUDA, else CPU)."""
    try:
        import torch  # type: ignore

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _validate_mixed_precision_flags(*, bf16: bool, fp16: bool) -> None:
    if bool(bf16) and bool(fp16):
        raise SystemExit("Choose at most one mixed-precision mode: --bf16 or --fp16.")


def _get_cuda_autocast_dtype(torch_mod) -> Any:
    try:
        return torch_mod.get_autocast_dtype("cuda")
    except TypeError:
        pass
    except Exception:
        pass
    return torch_mod.get_autocast_gpu_dtype()


def _set_cuda_autocast_dtype(torch_mod, dtype: Any) -> None:
    try:
        torch_mod.set_autocast_dtype("cuda", dtype)
        return
    except TypeError:
        pass
    except Exception:
        pass
    torch_mod.set_autocast_gpu_dtype(dtype)


@contextmanager
def _temporary_cuda_bf16_autocast(*, enabled: bool, device: str) -> Iterator[None]:
    if not bool(enabled) or str(device).strip().lower() != "cuda":
        yield
        return

    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("bf16 requested, but torch is unavailable for CUDA autocast configuration") from e

    prev_dtype = _get_cuda_autocast_dtype(torch)
    _set_cuda_autocast_dtype(torch, torch.bfloat16)
    try:
        yield
    finally:
        try:
            _set_cuda_autocast_dtype(torch, prev_dtype)
        except Exception:
            pass


def _resolve_training_fit_callable(
    model: Any, *, bf16: bool, fit_api: Optional[str] = None
) -> Tuple[str, Callable[..., Any]]:
    """Return the SentenceTransformer training API to use for this run.

    ``SentenceTransformer.fit`` only exposes the legacy boolean ``use_amp`` flag.
    For explicit bf16 runs, prefer ``old_fit`` so the training loop respects the
    CUDA autocast dtype configured via :func:`_temporary_cuda_bf16_autocast`.

    ``fit_api`` is an optional, opt-in override:

    * ``None``/``"auto"`` (default): preserve historical behavior -- ``old_fit`` when
      ``bf16`` is requested, else the modern ``fit``.
    * ``"old_fit"``: force the legacy ``old_fit`` path. This is the established path
      that performs evaluator-scalar (MRR@100) best-checkpoint selection, and it
      supports genuine fp16 mixed precision (legacy ``torch.cuda.amp.autocast`` +
      ``GradScaler``) without requesting bf16.
    * ``"fit"``: force the modern ``fit`` path.

    Every pre-existing config omits ``fit_api`` and therefore resolves via ``auto``,
    keeping completed Phase 1 / Phase 2 behavior byte-identical.
    """

    mode = str(fit_api or "auto").strip().lower()
    if mode not in {"auto", "old_fit", "fit", "modern_trainer"}:
        raise SystemExit(f"Unknown fit_api={fit_api!r}; choose one of auto|old_fit|fit|modern_trainer.")

    if mode == "modern_trainer":
        # Handled by the modern-trainer branch at the call site; no legacy callable.
        return "modern_trainer", None

    want_old_fit = (mode == "old_fit") or (mode == "auto" and bool(bf16))
    if want_old_fit:
        fit_fn = getattr(model, "old_fit", None)
        if fit_fn is None:
            raise SystemExit(
                "old_fit path requested (bf16 or fit_api=old_fit), but the loaded "
                "SentenceTransformer does not expose old_fit(). Use a sentence-transformers "
                "build that still provides old_fit(), or select fit_api=fit."
            )
        return "old_fit", fit_fn

    fit_fn = getattr(model, "fit", None)
    if fit_fn is None:  # pragma: no cover
        raise RuntimeError("Loaded SentenceTransformer model does not expose fit().")
    return "fit", fit_fn


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch  # type: ignore

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        # Torch not available in docs/test environments.
        pass


def _load_sentence_transformer(
    model_name_or_path: str, *, device: str, revision: Optional[str] = None
):
    # Import lazily so `demap finetune-phase1 --help` works even if
    # sentence-transformers isn't installed in the current environment.
    from demap_repro.biencoder.engine.st_loader import load_sentence_transformer

    return load_sentence_transformer(
        model_name_or_path, device=device, revision=revision
    )


def _input_example(text_a: str, text_b: str):
    # ST moved InputExample across modules in some versions; support both.
    try:
        from sentence_transformers import InputExample  # type: ignore

        return InputExample(texts=[text_a, text_b])
    except Exception:
        from sentence_transformers.readers import InputExample  # type: ignore

        return InputExample(texts=[text_a, text_b])


def _pv_defaults_assumed() -> Dict[str, Any]:
    """Return build-queries PV defaults (used for logging when manifest is missing)."""
    # These defaults are defined in demap_repro.data.queries argparse.
    return {
        "pv_max_n": 10,
        "pv_huge_threshold": 20,
        "sde_generic_label_p": 0.30,
        "pv_hash_salt": "demap",
        "source": "assumed_defaults_from_demap_build-queries (manifest not found)",
    }


def _make_run_config(
    *,
    run_id: str,
    timestamp: str,
    device: str,
    base_model: str,
    base_model_id: Optional[str] = None,
    model_revision: Optional[str] = None,
    init_model_name_or_path: Optional[str] = None,
    initfrom_tag: Optional[str] = None,
    query_variant: str,
    query_col: str,
    recipe: str,
    cde_format: str,
    rerank_mode: str,
    sep: str,
    recipe_configs: Optional[Dict[str, Dict[str, Any]]],
    pv_observed: Dict[str, Any],
    train_pair_stats: Dict[str, Any],
    val_pair_stats: Dict[str, Any],
    validation_split: str = "val",
    n_catalog: int,
    # Training
    seed: int,
    loss: str,
    loss_cls: str,
    lr: float,
    batch_size: int,
    temperature: float,
    scale: float,
    epochs: int,
    warmup_ratio: float,
    max_seq_length: int,
    bf16: bool = False,
    fp16: bool = False,
    fit_api: Optional[str] = None,
    continue_on_run_error: bool = True,
    # Eval
    top_k: int,
    k_values: Sequence[int],
    output_top_k: int,
    block_size: int,
    encode_batch_size: int,
    normalize_embeddings: bool,
    hybrid_alpha: float = 0.5,
    eval_splits: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build the serialized run configuration for a Phase 1 run.

    This is factored out for testability (ensures we can unit-test that run
    metadata includes the required fields without invoking training).
    """

    return {
        "phase": "phase1",
        "run_id": str(run_id),
        "timestamp": str(timestamp),
        "device": str(device),
        "base_model": str(base_model),

        "base_model_id": str(base_model_id or base_model),
        "model_revision": str(model_revision) if model_revision is not None else None,
        "init_model_name_or_path": str(init_model_name_or_path) if init_model_name_or_path is not None else None,
        "initfrom_tag": str(initfrom_tag) if initfrom_tag is not None else None,

        # Backward-compatible top-level aliases for representation (used by inspection/export tools).
        "query_variant": str(query_variant),
        "query_col": str(query_col),
        "recipe": str(recipe),
        "cde_format": str(cde_format),
        "rerank_mode": str(rerank_mode),
        "hybrid_alpha": float(hybrid_alpha),
        "sep": str(sep),
        "recipe_configs": recipe_configs or {},

        "representation": {
            "query_variant": str(query_variant),
            "query_col": str(query_col),
            "recipe": str(recipe),
            "cde_format": str(cde_format),
            "rerank_mode": str(rerank_mode),
            "hybrid_alpha": float(hybrid_alpha),
            "sep": str(sep),
            "recipe_configs": recipe_configs or {},
        },
        "pv_summary": {
            "parameterization": _pv_defaults_assumed(),
            "observed": pv_observed,
        },
        "data": {
            "train": train_pair_stats,
            "val": val_pair_stats,
            "validation_split": str(validation_split),
            "n_catalog": int(n_catalog),
        },
        "train": {
            "seed": int(seed),
            "loss": str(loss),
            "loss_cls": str(loss_cls),
            "lr": float(lr),
            "batch_size": int(batch_size),
            "temperature": float(temperature),
            "scale": float(scale),
            "epochs": int(epochs),
            "warmup_ratio": float(warmup_ratio),
            "max_seq_length": int(max_seq_length),
            "bf16": bool(bf16),
            "fp16": bool(fp16),
            "precision": "bf16" if bool(bf16) else ("fp16" if bool(fp16) else "fp32"),
            "fit_api": str(fit_api) if fit_api is not None else None,
            "continue_on_run_error": bool(continue_on_run_error),
        },
        "eval": {
            "top_k": int(top_k),
            "k_values": [int(x) for x in k_values],
            "output_top_k": int(output_top_k),
            "block_size": int(block_size),
            "encode_batch_size": int(encode_batch_size),
            "normalize_embeddings": bool(normalize_embeddings),
            "eval_splits": [str(x) for x in (eval_splits or [])],
        },
    }


def _assert_unique_query_ids(df: pd.DataFrame, *, split_name: str) -> None:
    if "query_id" not in df.columns:
        raise KeyError(f"Split {split_name} missing required column: query_id")
    s = df["query_id"].astype(str)
    if s.is_unique:
        return
    dup = s[s.duplicated()].value_counts().head(10)
    msg = (
        f"Split integrity violation: split='{split_name}' contains duplicate query_id values. "
        f"n={len(df)} n_unique={int(s.nunique())}. Top duplicates: {dup.to_dict()}"
    )
    raise ValueError(msg)



def _warn_duplicate_query_ids(df: pd.DataFrame, *, split_name: str) -> None:
    """Warn (do not fail) if a split contains duplicate `query_id` values.

    Notes
    -----
    In this repo, `query_id` is defined as a SHA1 hash of the *surface form* of
    the query (ALT/REF + raw text). It is therefore expected that the same
    `query_id` may appear multiple times in a split (e.g., the same short token
    reused across datasets/contexts), and it may even be associated with
    multiple CDE IDs.

    This check exists to surface the fact (for transparency) without blocking
    training.
    """
    if "query_id" not in df.columns:
        return
    s = df["query_id"].astype(str)
    if s.is_unique:
        return
    dup = s[s.duplicated()].value_counts().head(10)
    print(
        f"[warn] split='{split_name}' has duplicate query_id values: "
        f"n={len(df)} n_unique={int(s.nunique())}. Top duplicates: {dup.to_dict()}"
    )


def _coerce_cde_id(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a split has a canonical `cde_id` column."""
    if "cde_id" in df.columns:
        return df
    if "cde_publicid" in df.columns and "cde_version" in df.columns:
        d = df.copy()
        d["cde_id"] = d["cde_publicid"].astype(str) + "::" + d["cde_version"].astype(str)
        return d
    raise KeyError("Split missing cde_id (and missing cde_publicid/cde_version)")


def _prepare_pairs(
    df: pd.DataFrame,
    *,
    query_col: str,
    cde_id_to_text: Dict[str, str],
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """Return (pairs, stats) where pairs are (query_text, cde_text)."""
    d = _coerce_cde_id(df)
    if query_col not in d.columns:
        raise KeyError(f"Split missing query column: {query_col}")

    q = d[query_col].fillna("").astype(str).tolist()
    ids = d["cde_id"].astype(str).tolist()

    pairs: List[Tuple[str, str]] = []
    n_missing_target = 0
    n_empty_query = 0
    for qt, cid in zip(q, ids):
        qt2 = str(qt).strip()
        if qt2 == "":
            n_empty_query += 1
            continue
        ct = cde_id_to_text.get(str(cid))
        if ct is None or str(ct).strip() == "":
            n_missing_target += 1
            continue
        pairs.append((qt2, str(ct)))

    stats = {
        "n_rows_in_split": int(len(df)),
        "n_pairs_built": int(len(pairs)),
        "n_dropped_empty_query": int(n_empty_query),
        "n_dropped_missing_target_text": int(n_missing_target),
    }
    return pairs, stats


def _batch_collision_keys(pairs: Sequence[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    q_keys = [normalize_query_text(q) for (q, _c) in pairs]
    c_keys = [normalize_query_text(c) for (_q, c) in pairs]
    return q_keys, c_keys


def _loss_factory(loss_name: str):
    """Resolve a loss constructor from sentence-transformers.losses."""
    from sentence_transformers import losses  # type: ignore

    name = (loss_name or "").strip().lower()
    if name in {"mnrl", "mnr", "multiple_negatives", "multiple_negatives_ranking"}:
        return "MultipleNegativesRankingLoss", getattr(losses, "MultipleNegativesRankingLoss")
    if name in {"symmetric_mnrl", "sym_mnrl", "symmetric", "multiple_negatives_symmetric"}:
        cls = getattr(losses, "MultipleNegativesSymmetricRankingLoss", None)
        if cls is None:
            raise ValueError(
                "Your installed sentence-transformers does not expose MultipleNegativesSymmetricRankingLoss. "
                "Either upgrade sentence-transformers or remove this loss from the grid."
            )
        return "MultipleNegativesSymmetricRankingLoss", cls
    if name in {"cached_mnrl", "cached_multiple_negatives"}:
        cls = getattr(losses, "CachedMultipleNegativesRankingLoss", None)
        if cls is None:
            raise ValueError(
                "Your installed sentence-transformers does not expose CachedMultipleNegativesRankingLoss. "
                "Either upgrade sentence-transformers or remove this loss from the grid."
            )
        return "CachedMultipleNegativesRankingLoss", cls
    raise ValueError(f"Unknown loss: {loss_name}. Expected mnrl|symmetric_mnrl|cached_mnrl")


def _encode_np(model, texts: Sequence[str], *, batch_size: int, normalize: bool) -> np.ndarray:
    emb = model.encode(
        list(texts),
        batch_size=int(batch_size),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(normalize),
    )
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    if bool(normalize):
        # defensive: ensure unit norm
        n = np.linalg.norm(emb, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        emb = emb / n
    return emb


def _compute_val_metrics(
    *,
    model,
    val_df: pd.DataFrame,
    query_col: str,
    cde_ids: np.ndarray,
    cde_texts: Sequence[str],
    top_k: int,
    block_size: int,
    encode_batch_size: int,
    normalize_embeddings: bool,
) -> Dict[str, float]:
    """Compute headline val metrics used for checkpoint selection."""

    d = _coerce_cde_id(val_df)
    query_texts = d[query_col].fillna("").astype(str).tolist()
    true_ids = d["cde_id"].astype(str).tolist()

    cde_emb = _encode_np(model, cde_texts, batch_size=encode_batch_size, normalize=normalize_embeddings)
    q_emb = _encode_np(model, query_texts, batch_size=encode_batch_size, normalize=normalize_embeddings)

    idx_top, _score_top = bg._topk_blockwise(query_emb=q_emb, cde_emb=cde_emb, top_k=int(top_k), block_size=int(block_size))

    cde_id_to_idx = {cid: i for i, cid in enumerate(cde_ids.tolist())}
    true_idx = np.array([cde_id_to_idx.get(t, -1) for t in true_ids], dtype=np.int32)

    ranks = np.full((len(true_idx),), fill_value=np.inf, dtype=np.float32)
    for i, ti in enumerate(true_idx):
        if ti < 0:
            continue
        hits = np.where(idx_top[i] == ti)[0]
        if len(hits) > 0:
            ranks[i] = float(hits[0] + 1)

    rec5 = float((np.isfinite(ranks) & (ranks <= 5)).mean())
    rec10 = float((np.isfinite(ranks) & (ranks <= 10)).mean())
    top1 = float((np.isfinite(ranks) & (ranks <= 1)).mean())
    rr100 = np.where(np.isfinite(ranks) & (ranks <= 100), 1.0 / ranks, 0.0)
    mrr100 = float(rr100.mean())

    return {
        "recall@5": rec5,
        "recall@10": rec10,
        "top1_accuracy": top1,
        "mrr@100": mrr100,
        "n": float(len(val_df)),
        "n_missing_target_in_catalog": float(int((true_idx < 0).sum())),
    }


def _evaluate_and_write_run(
    *,
    run_dir: Path,
    model_name_or_path: str,
    device: str,
    cde_master_enriched: Path,
    splits_dir: Path,
    recipe: str,
    cde_format: str,
    query_variant: str,
    rerank_mode: str,
    sep: str,
    recipe_configs: Optional[Dict[str, Dict[str, Any]]],
    top_k: int,
    k_values: Sequence[int],
    output_top_k: int,
    block_size: int,
    batch_size: int,
    normalize_embeddings: bool,
    hybrid_alpha: float = 0.5,
    only_splits: Optional[Sequence[str]] = None,
    artifacts_dir: Path,
) -> None:
    """Run the standard evaluation protocol and write run artifacts."""

    ensure_dir(run_dir)

    master = pd.read_parquet(cde_master_enriched)
    splits = bg._load_splits(splits_dir)
    if only_splits:
        want = [str(s).strip() for s in only_splits if str(s).strip()]
        want_set = set(want)
        missing = [s for s in want if s not in splits]
        if missing:
            print(f"[eval] Warning: requested splits not found in splits_dir: {missing}")
        splits = {k: v for k, v in splits.items() if k in want_set}
        if not splits:
            raise ValueError(f"No splits to evaluate after filtering. only_splits={want}")

    # Build catalog texts (for later reporting). We will re-order them to match
    # the cde_ids returned from the embeddings cache builder.
    catalog_df = build_catalog(master, recipe=recipe, cde_format=cde_format, sep=sep, recipe_configs=recipe_configs)
    catalog_text_map = dict(
        zip(
            catalog_df["cde_id"].astype(str).tolist(),
            catalog_df["cde_text"].astype(str).tolist(),
        )
    )

    # Load model (respect requested device; baseline helper doesn't set device).
    # NOTE: the model *path* can be very long (e.g., under pytest tmp dirs).
    # Use a short, stable cache slug to avoid "File name too long" on macOS.
    # Fail-fast GPU contract (same as Phase 2): under DEMAP_REQUIRE_GPU=1 a
    # resolved non-CUDA device is an error, never a silent CPU fallback.
    if os.environ.get("DEMAP_REQUIRE_GPU") == "1" and not str(device).startswith("cuda"):
        raise RuntimeError(
            f"DEMAP_REQUIRE_GPU=1 but Phase 1 training resolved device={device!r}; "
            f"refusing to train on CPU")
    model_slug = _safe_cache_slug(model_name_or_path)
    model = _load_sentence_transformer(model_name_or_path, device=device)

    embeddings_dir = artifacts_dir / "embeddings"
    tfidf_cache_dir = artifacts_dir / "cache" / "tfidf"
    ensure_dir(embeddings_dir)
    ensure_dir(tfidf_cache_dir)

    cde_ids, cde_emb, catalog_meta = bg._load_or_build_catalog_embeddings(
        model=model,
        model_slug=model_slug,
        embeddings_dir=embeddings_dir,
        master=master,
        recipe=recipe,
        cde_format=cde_format,
        sep=sep,
        recipe_configs=recipe_configs,
        batch_size=int(batch_size),
        normalize=bool(normalize_embeddings),
    )

    # Ensure we have cde_texts aligned with cde_ids (defensive against ordering drift).
    cde_texts = [catalog_text_map.get(str(cid), "") for cid in cde_ids.tolist()]

    tfidf_cache_key = f"{recipe}__{cde_format}__{catalog_meta.get('recipe_key','')}"

    query_col = bg.QUERY_VARIANT_TO_COL.get(query_variant, query_variant)
    rerank_mode_resolved = bg.RERANK_MODE_ALIASES.get(rerank_mode, rerank_mode)

    split_metrics: Dict[str, Dict[str, float]] = {}
    rankings_all: List[pd.DataFrame] = []
    failures_all: List[pd.DataFrame] = []

    for split_name, p in splits.items():
        df_split = pd.read_parquet(p)
        df_split = _coerce_cde_id(df_split)

        query_texts = df_split[query_col].fillna("").astype(str).tolist()
        q_emb = bg._load_or_build_query_embeddings(
            model=model,
            model_slug=model_slug,
            embeddings_dir=embeddings_dir,
            split_name=split_name,
            query_variant=query_variant,
            query_texts=query_texts,
            batch_size=int(batch_size),
            normalize=bool(normalize_embeddings),
        )

        m, rankings, failures = bg._evaluate_split(
            split_name=split_name,
            df_split=df_split,
            query_col=query_col,
            cde_ids=cde_ids,
            cde_texts=cde_texts,
            cde_emb=cde_emb,
            query_emb=q_emb,
            top_k=int(top_k),
            k_values=k_values,
            block_size=int(block_size),
            rerank_mode=rerank_mode_resolved,
            alpha=float(hybrid_alpha),
            tfidf_cache_dir=tfidf_cache_dir,
            tfidf_cache_key=tfidf_cache_key,
            output_top_k=int(output_top_k),
        )
        split_metrics[split_name] = m
        rankings_all.append(rankings)
        if len(failures) > 0:
            failures_all.append(failures)

    rankings_df = pd.concat(rankings_all, ignore_index=True) if rankings_all else pd.DataFrame()

    # Ambiguity diagnostic:
    # Compare performance on "unambiguous" vs "ambiguous" normalized query texts,
    # where ambiguity is measured as nunique(cde_id) per normalized query text *in TRAIN*.
    ambiguity_train_summary: Dict[str, Any] = {"available": False}
    ambiguity_metrics_by_split: Dict[str, Any] = {}
    card_map: Dict[str, int] = {}

    train_path = splits.get("train")
    if train_path is not None:
        try:
            cols = [query_col, "cde_id", "cde_publicid", "cde_version"]
            df_train = pd.read_parquet(train_path, columns=[c for c in cols if c])
            df_train = _coerce_cde_id(df_train)
            if query_col not in df_train.columns:
                raise KeyError(f"TRAIN split missing query column: {query_col}")
            qnorm = df_train[query_col].fillna("").astype(str).map(normalize_query_text)
            tmp = df_train[["cde_id"]].copy()
            tmp["_qnorm"] = qnorm
            card = tmp.groupby("_qnorm")["cde_id"].nunique()

            card_map = card.to_dict()

            # Summaries at the *unique normalized query* level and at the *row* level.
            ambiguity_train_summary = {
                "available": True,
                "definition": f"nunique(cde_id | normalize({query_col})) computed on TRAIN",
                "n_rows": int(len(df_train)),
                "n_unique_norm_queries": int(len(card)),
                "n_unambiguous_norm_queries": int((card == 1).sum()),
                "n_ambiguous_norm_queries": int((card > 1).sum()),
                "ambiguous_rate_norm_queries": float((card > 1).mean()) if len(card) else 0.0,
                "max_labels_per_norm_query": int(card.max()) if len(card) else 0,
            }

            # Row-weighted ambiguous rate.
            row_card = qnorm.map(card_map).fillna(0).astype(int)
            ambiguity_train_summary["ambiguous_rate_rows"] = float((row_card > 1).mean()) if len(row_card) else 0.0
        except Exception as e:
            ambiguity_train_summary = {
                "available": False,
                "error": f"failed_to_compute_train_ambiguity: {type(e).__name__}: {e}",
            }

    def _subset_metrics(d: pd.DataFrame) -> Dict[str, Any]:
        if d is None or len(d) == 0:
            return {"n": 0, "top1_accuracy": float("nan"), "recall@5": float("nan"), "recall@10": float("nan"), "mrr@100": float("nan")}
        if "true_rank" not in d.columns:
            return {"n": int(len(d)), "error": "missing_true_rank"}
        r = pd.to_numeric(d["true_rank"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(r)
        top1 = float((finite & (r <= 1)).mean())
        r5 = float((finite & (r <= 5)).mean())
        r10 = float((finite & (r <= 10)).mean())
        rr100 = float(np.where(finite & (r <= 100), 1.0 / r, 0.0).mean())
        return {"n": int(len(d)), "top1_accuracy": top1, "recall@5": r5, "recall@10": r10, "mrr@100": rr100}

    if len(rankings_df) > 0 and card_map:
        rankings_df = rankings_df.copy()
        rankings_df["_qnorm"] = rankings_df["query_text"].map(normalize_query_text)
        rankings_df["train_label_cardinality"] = rankings_df["_qnorm"].map(card_map).fillna(0).astype(int)

        def _bucket(n: int) -> str:
            if n <= 0:
                return "unseen_in_train"
            if n == 1:
                return "unambiguous"
            return "ambiguous"

        rankings_df["ambiguity_bucket"] = rankings_df["train_label_cardinality"].map(_bucket)

        # Metrics by split × ambiguity bucket.
        ambiguity_metrics_by_split = {}
        for split_name in sorted(rankings_df["split"].dropna().unique().tolist()):
            ds = rankings_df[rankings_df["split"] == split_name]
            ambiguity_metrics_by_split[split_name] = {
                "unambiguous": _subset_metrics(ds[ds["ambiguity_bucket"] == "unambiguous"]),
                "ambiguous": _subset_metrics(ds[ds["ambiguity_bucket"] == "ambiguous"]),
                "unseen_in_train": _subset_metrics(ds[ds["ambiguity_bucket"] == "unseen_in_train"]),
            }

        # Persist a small standalone JSON for quick inspection.
        write_json(
            {
                "train_summary": ambiguity_train_summary,
                "metrics_by_split": ambiguity_metrics_by_split,
            },
            run_dir / "ambiguity_diagnostics.json",
        )

    # Calibration removed 2026-09-01: it imported `demap.evaluation.confidence`
    # from the RESEARCH package, inside a bare except, so it silently produced
    # empty artifacts here. No calibration or ECE result is in the manuscript.
    # Write core artifacts
    rankings_df.to_parquet(run_dir / "rankings.parquet", index=False)

    if failures_all:
        pd.concat(failures_all, ignore_index=True).to_csv(run_dir / "failures_sample.csv", index=False)

    metrics_out = {
        "run_id": run_dir.name,
        "spec": {
            "recipe": recipe,
            "cde_format": cde_format,
            "query_variant": query_variant,
            "rerank_mode": rerank_mode,
            "hybrid_alpha": float(hybrid_alpha),
        },
        "metrics_by_split": split_metrics,
    }

    # Add ambiguity diagnostic summary into metrics.json (Phase 1 debugging aid).
    metrics_out["ambiguity_diagnostics"] = {
        "train_summary": ambiguity_train_summary,
        "metrics_by_split": ambiguity_metrics_by_split,
    }

    write_json(metrics_out, run_dir / "metrics.json")


@dataclass(frozen=True)
class RunRow:
    run_id: str
    base_model_id: str
    query_variant: str
    recipe: str
    cde_format: str
    rerank_mode: str
    seed: int
    loss: str
    lr: float
    batch_size: int
    temperature: float
    epochs: int
    val_mrr100: float
    val_recall5: float


@dataclass(frozen=True)
class RunFailure:
    run_id: str
    base_model_id: str
    query_variant: str
    recipe: str
    cde_format: str
    rerank_mode: str
    seed: int
    loss: str
    lr: float
    batch_size: int
    temperature: float
    epochs: int
    error_type: str
    error_message: str


RUN_ROW_COLUMNS = [
    "run_id",
    "base_model_id",
    "query_variant",
    "recipe",
    "cde_format",
    "rerank_mode",
    "seed",
    "loss",
    "lr",
    "batch_size",
    "temperature",
    "epochs",
    "val_mrr100",
    "val_recall5",
]

RUN_FAILURE_COLUMNS = [
    "run_id",
    "base_model_id",
    "query_variant",
    "recipe",
    "cde_format",
    "rerank_mode",
    "seed",
    "loss",
    "lr",
    "batch_size",
    "temperature",
    "epochs",
    "error_type",
    "error_message",
]


def _read_existing_csv(path: Path, *, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=list(columns))
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df.loc[:, list(columns)]


def _merge_existing_rows(
    rows: Sequence[Dict[str, Any]],
    out_csv: Path,
    *,
    columns: Sequence[str],
    dedupe_col: str = "run_id",
) -> pd.DataFrame:
    existing = _read_existing_csv(out_csv, columns=columns)
    current = pd.DataFrame(list(rows), columns=list(columns)) if rows else pd.DataFrame(columns=list(columns))
    frames = [df for df in (existing, current) if not df.empty]
    if not frames:
        return pd.DataFrame(columns=list(columns))
    merged = frames[0].copy() if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    if dedupe_col in merged.columns:
        merged = merged.drop_duplicates(subset=[dedupe_col], keep="last")
    return merged.loc[:, list(columns)]


def _write_leaderboard(rows: List[RunRow], out_csv: Path) -> None:
    if not rows and not out_csv.exists():
        return
    ensure_dir(out_csv.parent)
    df = _merge_existing_rows([r.__dict__ for r in rows], out_csv, columns=RUN_ROW_COLUMNS)
    if df.empty:
        return
    df = df.sort_values(["val_recall5", "val_mrr100", "run_id"], ascending=[False, False, True]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)


def _write_group_summary(rows: List[RunRow], out_csv: Path, *, leaderboard_csv: Optional[Path] = None) -> None:
    if leaderboard_csv is not None and leaderboard_csv.exists():
        df = _read_existing_csv(leaderboard_csv, columns=RUN_ROW_COLUMNS)
    else:
        if not rows:
            return
        df = pd.DataFrame([r.__dict__ for r in rows], columns=RUN_ROW_COLUMNS)
    if df.empty:
        return
    ensure_dir(out_csv.parent)
    group_cols = ["base_model_id", "query_variant", "recipe", "cde_format", "rerank_mode", "loss", "lr", "batch_size", "temperature", "epochs"]
    g = df.groupby(group_cols, dropna=False)
    summ = g.agg(
        n_runs=("run_id", "count"),
        val_mrr100_mean=("val_mrr100", "mean"),
        val_mrr100_std=("val_mrr100", "std"),
        val_recall5_mean=("val_recall5", "mean"),
        val_recall5_std=("val_recall5", "std"),
    ).reset_index()
    summ = summ.sort_values(["val_recall5_mean", "val_mrr100_mean"], ascending=[False, False]).reset_index(drop=True)
    summ.to_csv(out_csv, index=False)


def _write_failed_runs(rows: List[RunFailure], out_csv: Path) -> None:
    if not rows and not out_csv.exists():
        return
    ensure_dir(out_csv.parent)
    df = _merge_existing_rows([r.__dict__ for r in rows], out_csv, columns=RUN_FAILURE_COLUMNS)
    if df.empty:
        return
    df = df.sort_values(["base_model_id", "query_variant", "recipe", "loss", "lr", "batch_size", "temperature", "epochs", "seed", "run_id"], ascending=True).reset_index(drop=True)
    df.to_csv(out_csv, index=False)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


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

    # Paths
    ap.add_argument("--cde-master-enriched", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--runs-dir", default=os.path.join("artifacts", "runs"))
    ap.add_argument("--stage-tag", default=None, help="Optional stage tag used in auto roots and run IDs.")

    # Fixed representation choices (Phase 1)
    ap.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--model-revision", default=None,
                    help="Optional immutable Hugging Face revision for --model-name.")
    ap.add_argument("--base-model-id", default=None, help="Canonical hub id for the model family (recommended when model_name is a local path).")
    ap.add_argument("--query-variant", default="Q3")
    ap.add_argument("--recipe", default="v1_v2_v3_v5")
    ap.add_argument("--cde-format", default="labeled")
    ap.add_argument("--rerank-mode", default="R0")
    ap.add_argument("--hybrid-alpha", type=float, default=0.5)
    ap.add_argument("--sep", default=" | ")

    # Training grid
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--losses", default="mnrl,symmetric_mnrl")
    ap.add_argument("--lrs", default="2e-5")
    ap.add_argument("--batch-sizes", default="64")
    ap.add_argument("--temperatures", default="0.05")
    ap.add_argument("--epochs", default="1")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay (default 0.01 matches historical old_fit).")
    ap.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help=(
            "Diagnostic/smoke cap on optimizer steps per epoch (bounds a run to a few "
            "batches). Default None = full epochs. Does not change precision, loss, or "
            "selection logic; intended for GPU smoke tests only."
        ),
    )
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--fit-api",
        choices=["auto", "old_fit", "fit", "modern_trainer"],
        default="auto",
        help=(
            "Which SentenceTransformer training API to use. 'auto' (default) preserves "
            "historical behavior (old_fit iff bf16). 'old_fit' forces the established "
            "evaluator-scalar (MRR@100) best-checkpoint path and supports genuine fp16. "
            "'modern_trainer' routes through SentenceTransformerTrainer for honest "
            "fp16/bf16 mixed precision (opt-in; used by the Section 4.3 precision validation)."
        ),
    )
    ap.add_argument(
        "--continue-on-run-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Continue the remaining grid when an individual run fails. "
            "Failed runs are recorded under reports/finetune_phase1/failed_runs.csv."
        ),
    )

    # Checkpointing (optional)
    ap.add_argument(
        "--save-epoch-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, save per-epoch checkpoints under <run_dir>/checkpoints so you can "
            "evaluate earlier epochs posthoc (e.g., demap eval-checkpoint)."
        ),
    )
    ap.add_argument(
        "--checkpoint-save-total-limit",
        type=int,
        default=3,
        help="Max checkpoints to keep when --save-epoch-checkpoints is enabled.",
    )

    # Evaluation settings
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--k-values", default="1,5,10,20")
    ap.add_argument("--output-top-k", type=int, default=20)
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--encode-batch-size", type=int, default=64)
    ap.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval-splits", default=None, help="Comma-separated splits to evaluate after training (default: all splits in splits_dir).")

    ap.add_argument("--dry-run", action="store_true", help="Print planned runs, but do not train.")

    args = ap.parse_args(argv_list)

    # Merge configs
    cfg: Dict[str, Any] = {}
    for cpath in args.config:
        cfg = deep_merge(cfg, load_config(cpath))
    cfg = cfg.get("finetune_phase1", cfg) if cfg else {}
    # Canonical paper lineage: carry the generator's provenance block into each run_config
    # (winner-manifest hash, protocol hash, resolved rep/loss, precision, fingerprints, ...).
    _paper_provenance = cfg.get("paper_provenance") if isinstance(cfg, dict) else None

    def flag_present(*flags: str) -> bool:
        return any(f in argv_list for f in flags)

    # Apply config defaults only if flag not provided.
    if cfg:
        if (not flag_present("--cde-master-enriched")) and args.cde_master_enriched is None:
            args.cde_master_enriched = cfg.get("cde_master_enriched")
        if (not flag_present("--splits-dir")) and args.splits_dir is None:
            args.splits_dir = cfg.get("splits_dir")
        if not flag_present("--artifacts-dir"):
            args.artifacts_dir = cfg.get("artifacts_dir", args.artifacts_dir)
        if not flag_present("--runs-dir"):
            args.runs_dir = cfg.get("runs_dir", args.runs_dir)
        if not flag_present("--stage-tag") and "stage_tag" in cfg:
            args.stage_tag = cfg.get("stage_tag")

        for key, flag in [
            ("base_model_id", "--base-model-id"),
            ("model_name", "--model-name"),
            ("model_revision", "--model-revision"),
            ("query_variant", "--query-variant"),
            ("recipe", "--recipe"),
            ("cde_format", "--cde-format"),
            ("rerank_mode", "--rerank-mode"),
            ("sep", "--sep"),
        ]:
            if not flag_present(flag) and key in cfg:
                setattr(args, flag.lstrip("--").replace("-", "_"), cfg.get(key))

        if not flag_present("--hybrid-alpha") and "hybrid_alpha" in cfg:
            args.hybrid_alpha = float(cfg.get("hybrid_alpha"))

        tr = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
        if tr:
            if not flag_present("--seeds") and "seeds" in tr:
                args.seeds = ",".join(str(x) for x in (tr.get("seeds") or []))
            if not flag_present("--losses") and "losses" in tr:
                args.losses = ",".join(str(x) for x in (tr.get("losses") or []))
            if not flag_present("--lrs") and "lrs" in tr:
                args.lrs = ",".join(str(x) for x in (tr.get("lrs") or []))
            if not flag_present("--batch-sizes") and "batch_sizes" in tr:
                args.batch_sizes = ",".join(str(x) for x in (tr.get("batch_sizes") or []))
            if not flag_present("--temperatures") and "temperatures" in tr:
                args.temperatures = ",".join(str(x) for x in (tr.get("temperatures") or []))
            if not flag_present("--epochs") and "epochs" in tr:
                args.epochs = ",".join(str(x) for x in (tr.get("epochs") or []))
            if not flag_present("--warmup-ratio") and "warmup_ratio" in tr:
                args.warmup_ratio = float(tr.get("warmup_ratio"))
            if not flag_present("--max-seq-length") and "max_seq_length" in tr:
                args.max_seq_length = int(tr.get("max_seq_length"))
            if not flag_present("--device") and "device" in tr:
                args.device = str(tr.get("device"))
            if not flag_present("--bf16", "--no-bf16") and "bf16" in tr:
                args.bf16 = bool(tr.get("bf16"))
            if not flag_present("--fp16", "--no-fp16") and "fp16" in tr:
                args.fp16 = bool(tr.get("fp16"))
            if not flag_present("--fit-api") and "fit_api" in tr:
                args.fit_api = str(tr.get("fit_api"))
            if not flag_present("--max-train-steps") and "max_train_steps" in tr:
                args.max_train_steps = int(tr.get("max_train_steps"))
            if not flag_present("--weight-decay") and "weight_decay" in tr:
                args.weight_decay = float(tr.get("weight_decay"))
            if not flag_present("--continue-on-run-error", "--no-continue-on-run-error") and "continue_on_run_error" in tr:
                args.continue_on_run_error = bool(tr.get("continue_on_run_error"))
            if not flag_present("--save-epoch-checkpoints") and "save_epoch_checkpoints" in tr:
                args.save_epoch_checkpoints = bool(tr.get("save_epoch_checkpoints"))
            if not flag_present("--checkpoint-save-total-limit") and "checkpoint_save_total_limit" in tr:
                args.checkpoint_save_total_limit = int(tr.get("checkpoint_save_total_limit"))

        ev = cfg.get("eval", {}) if isinstance(cfg.get("eval"), dict) else {}
        if ev:
            if not flag_present("--top-k") and "top_k" in ev:
                args.top_k = int(ev.get("top_k"))
            if not flag_present("--k-values") and "k_values" in ev:
                args.k_values = ",".join(str(x) for x in (ev.get("k_values") or []))
            if not flag_present("--output-top-k") and "output_top_k" in ev:
                args.output_top_k = int(ev.get("output_top_k"))
            if not flag_present("--block-size") and "block_size" in ev:
                args.block_size = int(ev.get("block_size"))
            if not flag_present("--encode-batch-size") and "encode_batch_size" in ev:
                args.encode_batch_size = int(ev.get("encode_batch_size"))
            if not flag_present("--normalize-embeddings") and "normalize_embeddings" in ev:
                args.normalize_embeddings = bool(ev.get("normalize_embeddings"))
            if not flag_present("--eval-splits") and "eval_splits" in ev:
                args.eval_splits = ",".join(str(x) for x in (ev.get("eval_splits") or []))

    _validate_mixed_precision_flags(bf16=bool(args.bf16), fp16=bool(args.fp16))

    if args.cde_master_enriched is None:
        raise SystemExit("Missing --cde-master-enriched (or finetune_phase1.cde_master_enriched in config)")
    if args.splits_dir is None:
        raise SystemExit("Missing --splits-dir (or finetune_phase1.splits_dir in config)")

    if not is_valid_recipe(str(args.recipe)):
        raise SystemExit(f"Unknown recipe: {args.recipe}")

    # Device
    device = str(args.device).strip().lower()
    if device == "auto":
        device = _auto_device()
    if device not in {"cpu", "cuda", "mps"}:
        raise SystemExit(f"Unknown device: {args.device}. Expected auto|cpu|cuda|mps")

    # Grid expansion
    seeds = _parse_csv_ints(args.seeds)
    losses = _parse_csv_list(args.losses)
    lrs = _parse_csv_floats(args.lrs)
    batch_sizes = _parse_csv_ints(args.batch_sizes)
    temps = _parse_csv_floats(args.temperatures)
    epochs_list = _parse_csv_ints(args.epochs)

    only_splits = _parse_csv_list(args.eval_splits) if args.eval_splits is not None else None

    # Recipe knobs: rely on config if present, else keep repo defaults.
    recipe_configs = cfg.get("recipe_configs") if isinstance(cfg.get("recipe_configs"), dict) else None

    # Load core data once (catalog + splits)
    cde_master_enriched = Path(args.cde_master_enriched)
    splits_dir = Path(args.splits_dir)
    artifacts_dir = Path(args.artifacts_dir)
    ensure_dir(artifacts_dir)
    # NOTE: runs_dir is resolved later (supports runs_dir='auto' for per-model run layout).


    master = pd.read_parquet(cde_master_enriched)
    catalog_df = build_catalog(
        master,
        recipe=str(args.recipe),
        cde_format=str(args.cde_format),
        sep=str(args.sep),
        recipe_configs=recipe_configs,
    )
    cde_ids = catalog_df["cde_id"].astype(str).values
    cde_texts = catalog_df["cde_text"].astype(str).tolist()
    cde_id_to_text = dict(zip(catalog_df["cde_id"].astype(str).tolist(), cde_texts))

    splits = bg._load_splits(splits_dir)

    # Train on train.parquet; validate (in-training, per-epoch best-model selection) on the
    # validation split. Prefer legacy `val`; else the canonical v3-CDISC `val_dev`. The old bare
    # `val` scheme is legacy (see data/processed/splits/SPLIT_MANIFEST.json).
    train_path = splits.get("train")
    val_split_name = "val" if splits.get("val") is not None else (
        "val_dev" if splits.get("val_dev") is not None else None)
    val_path = splits.get(val_split_name) if val_split_name else None
    test_path = splits.get("test")
    if train_path is None or val_path is None:
        raise SystemExit("splits_dir must contain train.parquet and (val.parquet or val_dev.parquet)")
    print(f"[finetune-phase1] train={os.path.basename(str(train_path))}  "
          f"validation={val_split_name}({os.path.basename(str(val_path))})")

    # Load only columns we need.
    query_col = bg.QUERY_VARIANT_TO_COL.get(str(args.query_variant), str(args.query_variant))
    cols_needed = ["query_id", "cde_id", "cde_publicid", "cde_version", query_col, "pv_attached", "PV_N", "PV_TYPE"]

    def _read_split(p: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(p, columns=[c for c in cols_needed if c])
        except Exception:
            return pd.read_parquet(p)

    train_df = _read_split(train_path)
    val_df = _read_split(val_path)
    if test_path is not None:
        test_df = _read_split(test_path)
    else:
        test_df = None

    # Phase 1A: split integrity diagnostics.
    # NOTE: `query_id` is a hash of the query surface form, so duplicates are expected.
    _warn_duplicate_query_ids(train_df, split_name="train")
    _warn_duplicate_query_ids(val_df, split_name="val")
    if test_df is not None:
        _warn_duplicate_query_ids(test_df, split_name="test")

    # PV attachment observable stats (for logging).
    def _pv_obs(df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if "pv_attached" in df.columns:
            out["pv_attached_rate"] = float(df["pv_attached"].fillna(False).astype(bool).mean())
        if "PV_N" in df.columns:
            out["pv_n_mean"] = float(pd.to_numeric(df["PV_N"], errors="coerce").fillna(0).mean())
        if "PV_TYPE" in df.columns:
            out["pv_type_top"] = (
                df["PV_TYPE"].fillna("").astype(str).value_counts().head(5).to_dict()
            )
        return out

    pv_obs = {
        "train": _pv_obs(train_df),
        "val": _pv_obs(val_df),
        "test": _pv_obs(test_df) if test_df is not None else {},
    }

    # Pre-build training pairs once.
    train_pairs, train_pair_stats = _prepare_pairs(train_df, query_col=query_col, cde_id_to_text=cde_id_to_text)
    val_pairs, val_pair_stats = _prepare_pairs(val_df, query_col=query_col, cde_id_to_text=cde_id_to_text)

    if len(train_pairs) == 0:
        raise SystemExit("No training pairs were built (check query_col and catalog mapping).")


    # ---------------------------------------------------------------------
    # Run directory + run_id naming helpers
    # ---------------------------------------------------------------------
    initfrom_tag = _extract_initfrom_tag(str(args.model_name))

    # base_model_id: required for readable per-model run dirs when model_name is a local checkpoint path.
    # If not provided, fall back to model_name when it looks like a hub id; otherwise best-effort infer.
    base_model_id = (
        str(args.base_model_id)
        if getattr(args, "base_model_id", None)
        else (
            str(args.model_name)
            if not _is_existing_local_path(str(args.model_name))
            else _infer_base_model_id_from_path(str(args.model_name))
        )
    )

    # Resolve runs_dir (supports runs_dir='auto' for per-model run layout)
    stage_tag = _normalize_stage_tag(getattr(args, "stage_tag", None)) or None
    runs_dir_path = _resolve_runs_dir(
        artifacts_dir=str(args.artifacts_dir),
        runs_dir=str(args.runs_dir),
        phase="finetune_phase1",
        base_model_id=base_model_id,
        stage_tag=stage_tag,
    )
    args.runs_dir = str(runs_dir_path)
    # Use the resolved runs_dir everywhere (avoid stale local variables when runs_dir='auto').
    runs_dir = Path(args.runs_dir)
    ensure_dir(runs_dir)


    planned: List[Tuple[str, int, str, float, int, float, int]] = []
    for loss in losses:
        for lr in lrs:
            for bs in batch_sizes:
                for temp in temps:
                    for ep in epochs_list:
                        for seed in seeds:
                            run_id = (
                                f"{time.strftime('%Y%m%d_%H%M%S')}"
                                + (f"__{stage_tag}" if stage_tag else "")
                                + f"__ft__{_slug(base_model_id)}"
                                + (f"__initfrom__{_slug(initfrom_tag)}" if initfrom_tag else "")
                                + f"__{args.query_variant}__{args.recipe}__{args.cde_format}__{args.rerank_mode}"
                                + f"__{loss}__lr{lr:g}__bs{bs}__t{temp:g}__ep{ep}__seed{seed}"
                            )
                            run_id = _truncate_with_hash(run_id, max_len=200)
                            planned.append((run_id, seed, loss, float(lr), int(bs), float(temp), int(ep)))

    if args.dry_run:
        print(f"Planned runs: {len(planned)}")
        for r in planned[:50]:
            print(r[0])
        if len(planned) > 50:
            print("...")
        return

    # Run grid.
    leaderboard_rows: List[RunRow] = []
    failed_rows: List[RunFailure] = []
    reports_dir = artifacts_dir / "reports" / "finetune_phase1"
    leaderboard_csv = reports_dir / "leaderboard.csv"
    summary_csv = reports_dir / "summary_by_config.csv"
    failed_runs_csv = reports_dir / "failed_runs.csv"
    ensure_dir(reports_dir)

    for run_id, seed, loss_name, lr, bs, temp, ep in planned:
        run_dir = runs_dir / run_id
        model_out_dir = run_dir / "model"
        ensure_dir(model_out_dir)

        try:
            # Deterministic RNG
            _set_seed(int(seed))

            # Build InputExamples
            train_samples = [_input_example(q, c) for (q, c) in train_pairs]

            # Batch collision sampler
            q_keys, c_keys = _batch_collision_keys(train_pairs)
            batch_sampler = NoDuplicateTextBatchSampler(
                q_keys,
                c_keys,
                batch_size=int(bs),
                seed=int(seed),
                drop_last=False,
                shuffle=True,
            )
            sampler = BatchSamplerAsSampler(batch_sampler)

            # Build model
            model = _load_sentence_transformer(
                str(args.model_name), device=device,
                revision=getattr(args, "model_revision", None),
            )
            try:
                model.max_seq_length = int(args.max_seq_length)
            except Exception:
                pass

            fit_api_name, fit_callable = _resolve_training_fit_callable(
                model, bf16=bool(args.bf16), fit_api=str(getattr(args, "fit_api", "auto"))
            )

            # DataLoader
            try:
                from torch.utils.data import DataLoader  # type: ignore

                train_dataloader = DataLoader(
                    train_samples,
                    sampler=sampler,
                    batch_size=int(bs),
                    drop_last=False,
                    collate_fn=model.smart_batching_collate,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to construct training DataLoader: {type(e).__name__}: {e}")

            # Loss
            loss_cls_name, LossCls = _loss_factory(loss_name)
            scale = float(1.0 / float(temp)) if float(temp) > 0 else 20.0
            try:
                train_loss = LossCls(model=model, scale=scale)
            except TypeError:
                # Some loss variants may not expose `scale` in older ST versions.
                train_loss = LossCls(model=model)

            # Minimal on-the-fly val evaluator for checkpoint selection.
            # We keep this lightweight: compute val MRR@100 + Recall@5.
            #
            # IMPORTANT: SentenceTransformers has multiple trainer backends depending on installed deps
            # (legacy FitMixin vs HF Trainer-backed SentenceTransformerTrainer). Some versions
            # will wrap `evaluator` into a SequentialEvaluator and expect an *iterable* of evaluators.
            # To be robust across versions, we always pass a SequentialEvaluator([val_evaluator]).
            #
            # Also, we keep the history on the inner evaluator for later diagnostics.
            try:
                from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator  # type: ignore
            except Exception:  # pragma: no cover
                SentenceEvaluator = object  # type: ignore
                SequentialEvaluator = None  # type: ignore

            class _ValEvaluator(SentenceEvaluator):  # type: ignore
                def __init__(self):
                    # Some ST versions' SentenceEvaluator has no __init__.
                    try:
                        super().__init__()  # type: ignore[misc]
                    except Exception:
                        pass
                    self.history: List[Dict[str, Any]] = []
                    self._epoch_ckpt_dir: Optional[Path] = None
                    self._epoch_ckpt_keep_last: int = 0

                def enable_epoch_model_saving(self, ckpt_dir: Path, keep_last: int = 3) -> None:
                    """Enable per-epoch *model snapshot* saving.

                    This is a fallback used when the active SentenceTransformers trainer backend
                    does not support built-in ``checkpoint_*`` args on ``model.fit``.
                    """
                    self._epoch_ckpt_dir = Path(ckpt_dir)
                    self._epoch_ckpt_keep_last = int(max(0, keep_last))

                def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
                    m = _compute_val_metrics(
                        model=model,
                        val_df=val_df,
                        query_col=query_col,
                        cde_ids=cde_ids,
                        cde_texts=cde_texts,
                        top_k=max(int(args.top_k), 100),
                        block_size=int(args.block_size),
                        encode_batch_size=int(args.encode_batch_size),
                        normalize_embeddings=bool(args.normalize_embeddings),
                    )
                    rec = {
                        "epoch": int(epoch),
                        "steps": int(steps),
                        **{k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating))},
                    }
                    self.history.append(rec)

                    # Optional fallback checkpointing: save a full model snapshot per epoch.
                    if self._epoch_ckpt_dir is not None and int(epoch) >= 0:
                        out_dir = Path(self._epoch_ckpt_dir) / f"epoch_{int(epoch):02d}"
                        if not out_dir.exists():
                            ensure_dir(out_dir)
                            model.save(str(out_dir))
                        # Keep only the most recent N snapshots.
                        if int(self._epoch_ckpt_keep_last) > 0:
                            try:
                                dirs = [p for p in Path(self._epoch_ckpt_dir).iterdir() if p.is_dir() and p.name.startswith("epoch_")]
                                dirs.sort(key=lambda p: p.name)
                                while len(dirs) > int(self._epoch_ckpt_keep_last):
                                    old = dirs.pop(0)
                                    try:
                                        import shutil
                                        shutil.rmtree(old)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                    # Higher is better.
                    return float(m.get("mrr@100", float("nan")))

            val_evaluator = _ValEvaluator()
            if str(fit_api_name) == "old_fit":
                # Legacy FitMixin expects a scalar score from the evaluator. Newer
                # SequentialEvaluator variants return a metrics dict, which breaks
                # old_fit() checkpoint selection (score > best_score).
                evaluator = val_evaluator
            elif SequentialEvaluator is None:  # pragma: no cover
                evaluator = val_evaluator
            else:
                evaluator = SequentialEvaluator([val_evaluator])

            # Training config logging (write before training so partial runs are inspectable).
            run_config = _make_run_config(
                run_id=str(run_id),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                device=str(device),
                base_model=str(base_model_id),
                base_model_id=str(base_model_id),
                model_revision=getattr(args, "model_revision", None),
                init_model_name_or_path=str(args.model_name),
                initfrom_tag=str(initfrom_tag) if initfrom_tag is not None else None,
                query_variant=str(args.query_variant),
                query_col=str(query_col),
                recipe=str(args.recipe),
                cde_format=str(args.cde_format),
                rerank_mode=str(args.rerank_mode),
                sep=str(args.sep),
                recipe_configs=recipe_configs,
                pv_observed=pv_obs,
                train_pair_stats=train_pair_stats,
                val_pair_stats=val_pair_stats,
                validation_split=str(val_split_name),
                n_catalog=int(len(cde_ids)),
                seed=int(seed),
                loss=str(loss_name),
                loss_cls=str(loss_cls_name),
                lr=float(lr),
                batch_size=int(bs),
                temperature=float(temp),
                scale=float(scale),
                epochs=int(ep),
                warmup_ratio=float(args.warmup_ratio),
                max_seq_length=int(args.max_seq_length),
                bf16=bool(args.bf16),
                fp16=bool(args.fp16),
                fit_api=str(fit_api_name),
                continue_on_run_error=bool(args.continue_on_run_error),
                top_k=int(args.top_k),
                k_values=[int(x) for x in _parse_csv_ints(args.k_values)],
                output_top_k=int(args.output_top_k),
                block_size=int(args.block_size),
                encode_batch_size=int(args.encode_batch_size),
                normalize_embeddings=bool(args.normalize_embeddings),
                hybrid_alpha=float(args.hybrid_alpha),
                eval_splits=only_splits,
            )
            run_config["checkpointing"] = {
                "save_epoch_checkpoints": bool(getattr(args, "save_epoch_checkpoints", False)),
                "checkpoint_save_total_limit": int(getattr(args, "checkpoint_save_total_limit", 3)),
                "checkpoints_dir": str(run_dir / "checkpoints") if bool(getattr(args, "save_epoch_checkpoints", False)) else None,
            }
            steps_per_epoch = max(int(len(train_dataloader)), 1)
            _max_train_steps = getattr(args, "max_train_steps", None)
            if _max_train_steps is not None and int(_max_train_steps) > 0:
                steps_per_epoch = min(steps_per_epoch, int(_max_train_steps))
            loss_history_jsonl = run_dir / "train_loss_history.jsonl"
            loss_history_csv = run_dir / "train_loss_history.csv"
            run_config["training_loss_history"] = {
                "jsonl": str(loss_history_jsonl),
                "csv": str(loss_history_csv),
                "steps_per_epoch": int(steps_per_epoch),
            }
            if _paper_provenance is not None:
                run_config["paper_provenance"] = _paper_provenance

            # Precision verification, layer 1 (config-state preflight, fail-closed):
            # confirm the configured autocast dtype matches the requested precision. The
            # authoritative execution evidence is captured during the first training batch
            # by TrainingPrecisionVerifier (installed just before fit, below).
            from demap.experiments.precision_probe import (
                preflight_autocast_config,
                TrainingPrecisionVerifier,
            )

            _use_modern_trainer = str(fit_api_name) == "modern_trainer"
            if _use_modern_trainer:
                _req_label = (
                    "bf16_mixed" if bool(args.bf16) else ("fp16_mixed" if bool(args.fp16) else "fp32")
                )
                run_config["precision_verification"] = {
                    "stage": "preflight_modern_trainer",
                    "requested_precision": _req_label,
                    "trainer_class": "SentenceTransformerTrainer",
                    "device": str(device),
                }
            else:
                run_config["precision_verification"] = preflight_autocast_config(
                    requested_bf16=bool(args.bf16),
                    requested_fp16=bool(args.fp16),
                    device=device,
                    bf16_autocast_ctx=_temporary_cuda_bf16_autocast,
                    get_autocast_dtype=_get_cuda_autocast_dtype,
                    fit_api_name=str(fit_api_name),
                    fit_callable=fit_callable,
                )
            write_json(run_config, run_dir / "run_config.json")

            # Train
            warmup_steps = int(len(train_dataloader) * int(ep) * float(args.warmup_ratio))
            # NOTE: We use a DataLoader with a custom `batch_sampler` to prevent in-batch text collisions.
            # When a DataLoader is constructed with `batch_sampler`, `dataloader.batch_size` is None.
            # SentenceTransformers' FitMixin (as of the versions we support) may attempt to infer
            # `steps_per_epoch` from `len(dataset) // batch_size` which fails when `batch_size` is None.
            # We therefore pass an explicit `steps_per_epoch` based on the dataloader length.
            logged_train_loss = LossLoggingWrapper(
                train_loss,
                run_dir=run_dir,
                configured_lr=float(lr),
                steps_per_epoch=steps_per_epoch,
            )
            _prec_verifier = None
            try:
                fit_kwargs: Dict[str, Any] = dict(
                    train_objectives=[(train_dataloader, logged_train_loss)],
                    evaluator=evaluator,
                    epochs=int(ep),
                    steps_per_epoch=steps_per_epoch,
                    evaluation_steps=int(steps_per_epoch),
                    warmup_steps=max(int(warmup_steps), 0),
                    output_path=str(model_out_dir),
                    save_best_model=True,
                    optimizer_params={"lr": float(lr)},
                    show_progress_bar=True,
                )

                if bool(args.bf16) or bool(args.fp16):
                    fit_kwargs["use_amp"] = True

                # Optional: save per-epoch checkpoints for posthoc evaluation.
                # We *prefer* SentenceTransformers' built-in checkpointing args when available,
                # falling back to saving full model snapshots inside the evaluator if needed.
                if bool(getattr(args, "save_epoch_checkpoints", False)):
                    ckpt_dir = run_dir / "checkpoints"
                    ensure_dir(ckpt_dir)

                    try:
                        sig = inspect.signature(fit_callable)
                        has = lambda name: name in sig.parameters
                    except Exception:  # pragma: no cover
                        has = lambda name: False  # type: ignore[assignment]

                    if has("checkpoint_path"):
                        fit_kwargs["checkpoint_path"] = str(ckpt_dir)
                        if has("checkpoint_save_steps"):
                            # Save at the end of each epoch (steps are counted globally).
                            fit_kwargs["checkpoint_save_steps"] = int(steps_per_epoch)
                        if has("checkpoint_save_total_limit"):
                            fit_kwargs["checkpoint_save_total_limit"] = int(getattr(args, "checkpoint_save_total_limit", 3))
                    else:
                        # Fallback (older / alternative trainer backends): save full model snapshots per epoch.
                        try:
                            val_evaluator.enable_epoch_model_saving(
                                ckpt_dir=ckpt_dir,
                                keep_last=int(getattr(args, "checkpoint_save_total_limit", 3)),
                            )
                        except Exception:
                            print(
                                "[train] Warning: --save-epoch-checkpoints requested, but checkpointing is not supported "
                                "by this SentenceTransformers backend and the fallback saver failed."
                            )

                # Precision verification, layer 2 (runtime execution evidence, fail-closed):
                # one-shot self-removing hooks that observe the dtypes produced by the
                # actual training forward's first batch. No extra forward, no RNG use, no
                # model-mode change, no gradient/optimizer effect.
                _prec_verifier = TrainingPrecisionVerifier(
                    model,
                    requested_bf16=bool(args.bf16),
                    requested_fp16=bool(args.fp16),
                    device_type=("cuda" if str(device).startswith("cuda") else "cpu"),
                ).install()

                if _use_modern_trainer:
                    # Honest fp16/bf16 mixed precision via SentenceTransformerTrainer. Same
                    # data, loss, no-duplicate batching, optimizer, scheduler, warmup, and
                    # per-epoch MRR@100 selection; the _prec_verifier hooks observe the real
                    # training-forward dtype during trainer.train().
                    from demap_repro.biencoder.engine.modern_trainer import train_with_modern_trainer

                    run_config["modern_trainer"] = train_with_modern_trainer(
                        model=model,
                        train_pairs=train_pairs,
                        train_loss=train_loss,
                        compute_val_metrics=_compute_val_metrics,
                        val_df=val_df,
                        query_col=query_col,
                        cde_ids=cde_ids,
                        cde_texts=cde_texts,
                        lr=float(lr),
                        batch_size=int(bs),
                        epochs=int(ep),
                        seed=int(seed),
                        temperature=float(temp),
                        warmup_ratio=float(args.warmup_ratio),
                        weight_decay=float(getattr(args, "weight_decay", 0.01)),
                        device=device,
                        model_out_dir=model_out_dir,
                        run_dir=run_dir,
                        requested_bf16=bool(args.bf16),
                        requested_fp16=bool(args.fp16),
                        top_k=int(args.top_k),
                        block_size=int(args.block_size),
                        encode_batch_size=int(args.encode_batch_size),
                        normalize_embeddings=bool(args.normalize_embeddings),
                        max_train_steps=getattr(args, "max_train_steps", None),
                        prec_verifier=_prec_verifier,
                    )
                else:
                    with _temporary_cuda_bf16_autocast(enabled=bool(args.bf16), device=device):
                        fit_callable(**fit_kwargs)
            except Exception as e:
                # Write a small error marker so failed runs are discoverable.
                write_json(
                    {
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                        }
                    },
                    run_dir / "_train_error.json",
                )
                raise
            finally:
                logged_train_loss.close()
                # Persist runtime precision evidence (whether the run succeeded, was
                # aborted by the fail-closed hook, or raised for another reason).
                if _prec_verifier is not None:
                    try:
                        _prec_verifier.remove_all()
                        run_config["precision_verification"]["runtime"] = _prec_verifier.evidence
                        write_json(run_config, run_dir / "run_config.json")
                        write_json(_prec_verifier.evidence, run_dir / "precision_verification.json")
                    except Exception:
                        pass

            # Fail-closed on the runtime execution evidence (raises SystemExit on failure).
            # Enforced only when real CUDA training occurred; CPU/mocked runs (no GPU) cannot
            # produce a mislabeled bf16/fp16 artifact, so we record evidence without hard-failing.
            if _prec_verifier is not None:
                try:
                    import torch as _torch_fc

                    _cuda_avail = bool(_torch_fc.cuda.is_available())
                except Exception:
                    _cuda_avail = False
                if _cuda_avail:
                    _prec_verifier.finalize_and_verify()
                elif _prec_verifier.evidence.get("effective_precision") is None:
                    _prec_verifier.evidence["effective_precision"] = "unenforced_no_cuda"
                # Re-persist so the recorded evidence includes the resolved effective_precision
                # (the finally-block persist above ran before finalize set this field).
                try:
                    run_config["precision_verification"]["runtime"] = _prec_verifier.evidence
                    write_json(run_config, run_dir / "run_config.json")
                    write_json(_prec_verifier.evidence, run_dir / "precision_verification.json")
                except Exception:
                    pass

            # Persist batch collision stats and per-epoch val scores.
            write_json(
                {
                    "batch_collision_history": [s.__dict__ for s in sampler.history],
                    "val_history": val_evaluator.history,
                    "loss_history": {
                        "jsonl": str(loss_history_jsonl),
                        "csv": str(loss_history_csv),
                        "n_steps_logged": int(getattr(logged_train_loss, "global_step", 0)),
                    },
                },
                run_dir / "training_diagnostics.json",
            )

            # If checkpointing is enabled, write a small manifest of discovered checkpoints.
            if bool(getattr(args, "save_epoch_checkpoints", False)):
                ckpt_dir = run_dir / "checkpoints"
                if ckpt_dir.exists():
                    ckpts: List[Dict[str, Any]] = []
                    for child in sorted([p for p in ckpt_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                        name = child.name
                        step: Optional[int] = None
                        epoch_n: Optional[int] = None
                        if name.startswith("checkpoint-"):
                            tail = name.split("checkpoint-", 1)[1]
                            if tail.isdigit():
                                step = int(tail)
                                if int(steps_per_epoch) > 0 and step % int(steps_per_epoch) == 0:
                                    epoch_n = int(step // int(steps_per_epoch))
                        elif name.startswith("epoch_"):
                            tail = name.split("epoch_", 1)[1]
                            if tail.isdigit():
                                epoch_n = int(tail)
                        ckpts.append({"name": name, "path": str(child), "step": step, "epoch": epoch_n})
                    write_json(
                        {"steps_per_epoch": int(steps_per_epoch), "checkpoints": ckpts},
                        run_dir / "checkpoints_manifest.json",
                    )

            # Evaluate the saved best checkpoint using the standard protocol.
            _evaluate_and_write_run(
                run_dir=run_dir,
                model_name_or_path=str(model_out_dir),
                device=device,
                cde_master_enriched=cde_master_enriched,
                splits_dir=splits_dir,
                recipe=str(args.recipe),
                cde_format=str(args.cde_format),
                query_variant=str(args.query_variant),
                rerank_mode=str(args.rerank_mode),
                sep=str(args.sep),
                recipe_configs=recipe_configs,
                top_k=int(args.top_k),
                k_values=[int(x) for x in _parse_csv_ints(args.k_values)],
                output_top_k=int(args.output_top_k),
                block_size=int(args.block_size),
                batch_size=int(args.encode_batch_size),
                normalize_embeddings=bool(args.normalize_embeddings),
                hybrid_alpha=float(args.hybrid_alpha),
                only_splits=only_splits,
                artifacts_dir=artifacts_dir,
            )

            # Collect leaderboard metrics from metrics.json
            try:
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                valm = (metrics.get("metrics_by_split") or {}).get("val") or {}
                val_mrr100 = float(valm.get("mrr@100", float("nan")))
                val_rec5 = float(valm.get("recall@5", float("nan")))
            except Exception:
                val_mrr100 = float("nan")
                val_rec5 = float("nan")

            leaderboard_rows.append(
                RunRow(
                    run_id=str(run_id),
                    base_model_id=str(base_model_id),
                    query_variant=str(args.query_variant),
                    recipe=str(args.recipe),
                    cde_format=str(args.cde_format),
                    rerank_mode=str(args.rerank_mode),
                    seed=int(seed),
                    loss=str(loss_name),
                    lr=float(lr),
                    batch_size=int(bs),
                    temperature=float(temp),
                    epochs=int(ep),
                    val_mrr100=float(val_mrr100),
                    val_recall5=float(val_rec5),
                )
            )

            # Update reports incrementally.
            _write_leaderboard(leaderboard_rows, leaderboard_csv)
            _write_group_summary(leaderboard_rows, summary_csv, leaderboard_csv=leaderboard_csv)
        except Exception as e:
            write_json(
                {
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                },
                run_dir / "_run_error.json",
            )
            failed_rows.append(
                RunFailure(
                    run_id=str(run_id),
                    base_model_id=str(base_model_id),
                    query_variant=str(args.query_variant),
                    recipe=str(args.recipe),
                    cde_format=str(args.cde_format),
                    rerank_mode=str(args.rerank_mode),
                    seed=int(seed),
                    loss=str(loss_name),
                    lr=float(lr),
                    batch_size=int(bs),
                    temperature=float(temp),
                    epochs=int(ep),
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )
            _write_failed_runs(failed_rows, failed_runs_csv)
            if bool(args.continue_on_run_error):
                print(f"[run-error] {run_id}: {type(e).__name__}: {e}")
                continue
            raise


if __name__ == "__main__":  # pragma: no cover
    main()
