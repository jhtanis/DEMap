#!/usr/bin/env python3
"""finetune_phase2.py

Phase 2 fine-tuning for sentence embedding retrieval using *mined hard negatives*.

This runner mirrors Phase 1's conservative evaluation protocol, but replaces
"pure" in-batch negatives with a hybrid:

  - in-batch negatives from other positives in the batch
  - explicit per-example hard negatives sampled from a mined top-K list

Mining artifacts
----------------
Phase 2 expects a mined Parquet artifact produced by:

  demap mine-hardneg --config configs/experiments/mine_hardneg.yaml

The artifact must include (at minimum):
  - query_key (normalized query text)
  - query_text
  - gold_cde_ids (list)
  - candidates_cde_id (list)
  - candidates_score (list)

Mining strategies
-----------------
We support the following strategies:

  - none:       no mined negatives (in-batch negatives only)
  - hard:       sample from ranks 1–25
  - semihard:   sample from ranks 25–100
  - semihard_tight: sample from a tighter semi-hard band (default 10–50)
  - curr:       curriculum by epoch
                 epoch 1: 50–200
                 epoch 2: 25–100
                 epoch 3+: 1–25
  - hardcurr:   harder curriculum by epoch
                 epoch 1: 10–25
                 epoch 2: 5–15
                 epoch 3+: 1–10

Implementation notes
--------------------
We implement a small custom loss that is compatible with SentenceTransformers'
FitMixin API and accepts *N>=2* text columns per training example:

  texts=[anchor, positive, neg1, ..., negN]

The loss computes cross-entropy against a candidate pool containing:

  [positives_in_batch] + [hard_negatives_in_batch]

so each anchor must rank its paired positive at the correct index among a
large set of negatives.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib

import numpy as np
import pandas as pd

from demap_repro.biencoder.engine import baseline_grid as bg
from demap_repro.biencoder.engine import finetune_phase1 as ft1
from demap_repro.text.normalize import normalize_query_text
from demap_repro.text.recipes import build_catalog, is_valid_recipe
from demap_repro.biencoder.engine.samplers import NoDuplicateTextBatchSampler, BatchSamplerAsSampler
from demap_repro.biencoder import hardneg as _canon_hn
from demap_repro.biencoder.engine.loss_logging import LossLoggingWrapper
from demap_repro.utils.config import load_config
from demap_repro.utils.io import write_json
from demap_repro.utils.paths import ensure_dir


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------



def _listify_parquet_cell(v: Any) -> List[Any]:
    """Convert a parquet cell that should be list-like into a Python list.

    Pandas/pyarrow can materialize list columns as python lists, numpy arrays,
    or pyarrow scalars. Using `v or []` crashes for numpy arrays due to
    ambiguous truth value.

    This helper is intentionally standalone (not nested inside main) so that
    regressions like accidentally indenting unrelated logic under a return path
    are easier to catch with unit tests.
    """

    if v is None:
        return []

    # Handle pyarrow scalars (e.g., ListScalar) first.
    if hasattr(v, "as_py"):
        try:
            v = v.as_py()  # type: ignore[assignment]
        except Exception:
            pass
        if v is None:
            return []

    # Common fast paths
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)

    # numpy.ndarray / pandas array-like
    if hasattr(v, "tolist") and not isinstance(v, str):
        try:
            out = v.tolist()  # type: ignore[attr-defined]
            return out if isinstance(out, list) else [out]
        except Exception:
            pass

    # Strings are iterable but not list-cells.
    if isinstance(v, str):
        return [v]

    # Fallback: try to iterate.
    try:
        return list(v)  # type: ignore[arg-type]
    except Exception:
        return [v]


def _build_mined_query_maps(mined: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build (query_key -> candidates) and (query_key -> gold) maps from a mined DF."""

    mined_query_to_candidates: Dict[str, List[str]] = {}
    mined_query_to_gold: Dict[str, List[str]] = {}

    if mined is None or len(mined) == 0:
        return mined_query_to_candidates, mined_query_to_gold

    for _, row in mined.iterrows():
        qk = str(row.get("query_key", ""))
        qk = normalize_query_text(qk)
        if not qk:
            continue

        cand = row.get("candidates_cde_id")
        gold = row.get("gold_cde_ids")

        cand_list = [str(x) for x in _listify_parquet_cell(cand) if x is not None and str(x).strip() != ""]
        gold_list = [str(x) for x in _listify_parquet_cell(gold) if x is not None and str(x).strip() != ""]

        mined_query_to_candidates[qk] = cand_list
        mined_query_to_gold[qk] = gold_list

    return mined_query_to_candidates, mined_query_to_gold

def _slug(s: str) -> str:
    return str(s).replace("/", "__").replace(" ", "_").replace(":", "_")


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x) for x in _parse_csv_list(s)]


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for x in _parse_csv_list(s):
        out.append(int(float(x)))
    return out


def _short_model_tag(model_name_or_path: str, max_len: int = 60) -> str:
    """
    Create a short, filesystem-safe tag for a model path/name.
    Avoid embedding long absolute paths into run directory names.
    """
    s = str(model_name_or_path)

    # Use a readable basename; if path ends with ".../model", use parent folder name too.
    p = Path(s)
    base = p.name
    if base == "model" and p.parent is not None:
        base = p.parent.name  # e.g., run_id folder
    if not base:
        base = "model"

    base_slug = _slug(base)[:max_len].strip("_") or "model"
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
    return f"{base_slug}-{h}"


def _normalize_stage_tag(stage_tag: Optional[str]) -> str:
    t = str(stage_tag or "").strip()
    if not t:
        return ""
    return _slug(t).strip("_")


def _truncate_with_hash(s: str, max_len: int = 200) -> str:
    """
    Ensure s is not too long for a single filesystem path component.
    If it exceeds max_len, truncate and append a short hash for uniqueness.
    """
    s = str(s)
    if len(s) <= max_len:
        return s
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    keep = max_len - (len(h) + 1)  # 1 for "-"
    if keep < 10:
        # Fallback: just return the hash if max_len is very small
        return h[:max_len]
    return f"{s[:keep]}-{h}"


class GPURequiredError(RuntimeError):
    """A run that must train on GPU resolved to CPU."""


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


def assert_gpu_training(device: str, *, require: bool) -> None:
    """Fail fast rather than silently training a canonical run on CPU.

    Enabled by ``--require-gpu`` or ``DEMAP_REQUIRE_GPU=1`` (set by the Slurm
    launcher). Semantic-model training under the paper protocol must run on a GPU;
    a CPU fallback would be both unusably slow and scientifically mislabeled, so
    every failure mode below is an error, never a downgrade.
    """
    if not require:
        return
    try:
        import torch  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise GPURequiredError(f"--require-gpu: torch is not importable ({e})") from None
    if not torch.cuda.is_available():
        raise GPURequiredError(
            "--require-gpu: torch.cuda.is_available() is False; refusing to train on CPU. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")
    if str(device).split(":")[0] != "cuda":
        raise GPURequiredError(
            f"--require-gpu: resolved device {device!r} is not CUDA; refusing to train on CPU")
    idx = int(str(device).split(":")[1]) if ":" in str(device) else torch.cuda.current_device()
    if idx >= torch.cuda.device_count():
        raise GPURequiredError(
            f"--require-gpu: CUDA device index {idx} >= visible device count "
            f"{torch.cuda.device_count()}")
    print(f"[gpu-guard] training on CUDA device {idx} "
          f"({torch.cuda.get_device_name(idx)}), CUDA {torch.version.cuda}", flush=True)


def assert_model_on_cuda(model, *, require: bool) -> None:
    """After the model is built, verify its parameters actually live on CUDA."""
    if not require:
        return
    import torch  # type: ignore
    devs = {p.device.type for p in model.parameters()}
    if devs != {"cuda"}:
        raise GPURequiredError(
            f"--require-gpu: model parameters are on {sorted(devs)}, expected only 'cuda'")
    print("[gpu-guard] verified: all model parameters are on CUDA", flush=True)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch  # type: ignore

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _load_sentence_transformer(model_name_or_path: str, *, device: str):
    from demap_repro.biencoder.engine.st_loader import load_sentence_transformer

    return load_sentence_transformer(model_name_or_path, device=device)


def _input_example(texts: Sequence[str]):
    """Create a SentenceTransformers InputExample for N text fields."""
    try:
        from sentence_transformers import InputExample  # type: ignore

        return InputExample(texts=list(texts))
    except Exception:
        from sentence_transformers.readers import InputExample  # type: ignore

        return InputExample(texts=list(texts))


def _coerce_cde_id(df: pd.DataFrame) -> pd.DataFrame:
    if "cde_id" in df.columns:
        return df
    if "cde_publicid" in df.columns and "cde_version" in df.columns:
        d = df.copy()
        d["cde_id"] = d["cde_publicid"].astype(str) + "::" + d["cde_version"].astype(str)
        return d
    raise KeyError("Split missing cde_id (and missing cde_publicid/cde_version)")


# -----------------------------------------------------------------------------
# Hard-negative dataset + loss
# -----------------------------------------------------------------------------


Band = Tuple[int, int]  # inclusive ranks, 1-indexed

_STRATEGY_LABEL_TO_BASE: Dict[str, str] = {
    "none": "none",
    "hard": "hard",
    "hard_top25": "hard",
    "hard_1_25": "hard",
    "semihard": "semihard",
    "semihard_1_50": "semihard",
    "semihard_25_100": "semihard",
    "semihard_tight": "semihard_tight",
    "curr": "curr",
    "hardcurr": "hardcurr",
    "hardcurr_1_50__1_25__1_15": "hardcurr",
}


def _normalize_strategies(raw: List[str]) -> List[str]:
    """Map strategy label names to base names used by HardNegTupleDataset.

    Raises SystemExit on unknown strategy names instead of silently dropping.
    Deduplicates while preserving order.
    """
    out: List[str] = []
    seen: set = set()
    for s in raw:
        base = _STRATEGY_LABEL_TO_BASE.get(s)
        if base is None:
            raise SystemExit(
                f"Unknown mining strategy: {s!r}. "
                f"Supported values: {', '.join(sorted(_STRATEGY_LABEL_TO_BASE))}"
            )
        if base not in seen:
            out.append(base)
            seen.add(base)
    return out


def _parse_band(s: str) -> Band:
    parts = [p.strip() for p in (s or "").split("-") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid band '{s}'. Expected like '25-100'.")
    lo = int(parts[0])
    hi = int(parts[1])
    if lo <= 0 or hi <= 0 or hi < lo:
        raise ValueError(f"Invalid band '{s}'. Expected positive ranks with hi>=lo.")
    return (lo, hi)


class HardNegTupleDataset:
    """Dynamic dataset that samples hard negatives per example.

    Each item returns an InputExample with:
      texts = [anchor, positive, neg1, ..., negN]
    """

    def __init__(
        self,
        *,
        query_texts: Sequence[str],
        query_keys: Sequence[str],
        pos_cde_ids: Sequence[str],
        pos_texts: Sequence[str],
        cde_id_to_text: Dict[str, str],
        all_cde_ids: Sequence[str],
        mined_query_to_candidates: Dict[str, List[str]],
        mined_query_to_gold: Dict[str, List[str]],
        strategy: str,
        nneg: int,
        top_k: int,
        hard_band: Band,
        semihard_band: Band,
        curriculum_epoch_bands: Sequence[Band],
        seed: int,
    ) -> None:
        if not (len(query_texts) == len(query_keys) == len(pos_cde_ids) == len(pos_texts)):
            raise ValueError("Mismatched dataset column lengths")
        self.query_texts = list(query_texts)
        self.query_keys = list(query_keys)
        self.pos_cde_ids = list(pos_cde_ids)
        self.pos_texts = list(pos_texts)
        self.cde_id_to_text = dict(cde_id_to_text)
        self.all_cde_ids = list(all_cde_ids)

        self.mined_query_to_candidates = mined_query_to_candidates
        self.mined_query_to_gold = mined_query_to_gold

        self.strategy = str(strategy).strip().lower()
        # NOTE: strategy='none' must behave as "0 mined negatives" regardless of
        # configured nneg (we log effective_nneg separately in run_config.json).
        self.nneg = 0 if self.strategy == "none" else int(nneg)
        self.top_k = int(top_k)
        self.hard_band = hard_band
        self.semihard_band = semihard_band
        self.curriculum_epoch_bands = list(curriculum_epoch_bands)
        self.seed = int(seed)
        self._epoch = 0

        if self.strategy not in {"none", "hard", "semihard", "semihard_tight", "curr", "hardcurr"}:
            raise ValueError(f"Unknown mining strategy: {self.strategy}")
        if self.strategy != "none" and self.nneg <= 0:
            raise ValueError("nneg must be > 0 for mining strategies")

        # Auditable fallback accounting (never silent).
        self.fallback_stats = {"n_examples_with_fallback": 0, "n_from_topk": 0,
                               "n_from_catalog_fill": 0}
        # Small, deterministic global RNG used only for fallback cases.
        self._global_rng = random.Random(self.seed)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(max(0, epoch))

    def __len__(self) -> int:
        return len(self.query_texts)

    def _band_for_epoch(self) -> Band:
        if self.strategy == "hard":
            return self.hard_band
        if self.strategy in {"semihard", "semihard_tight"}:
            return self.semihard_band
        # curriculum (curr or hardcurr)
        if not self.curriculum_epoch_bands:
            return self.semihard_band
        # epoch is 0-indexed internally; curriculum bands are specified in epoch order.
        i = min(int(self._epoch), len(self.curriculum_epoch_bands) - 1)
        return self.curriculum_epoch_bands[i]

    def _slice_rank_band(self, candidates: Sequence[str], band: Band) -> List[str]:
        lo, hi = band
        lo_i = max(int(lo) - 1, 0)
        hi_i = min(int(hi), len(candidates))
        return list(candidates[lo_i:hi_i])

    def _sample_neg_texts(self, *, qkey: str, idx: int) -> List[str]:
        """Delegate to the canonical selector.

        Guarantees (enforced in demap_repro.biencoder.hardneg.select_negatives):
        the example's positive and every accepted gold are excluded, ids are de-duplicated,
        a positive is NEVER used as a negative, and an exhausted band falls back
        deterministically to valid non-gold catalog entries (recorded, never silently padded).
        """
        rng = _canon_hn.example_rng(seed=self.seed, epoch=self._epoch, index=idx)
        cand = list((self.mined_query_to_candidates.get(qkey) or [])[: self.top_k])
        gold = list(self.mined_query_to_gold.get(qkey) or [])
        pos_id = str(self.pos_cde_ids[idx]) if idx < len(self.pos_cde_ids) else ""

        band = self._band_for_epoch()
        strategy_key = self._canonical_strategy_key(band)
        res = _canon_hn.select_negatives(
            candidates=cand, gold_ids=gold, positive_id=pos_id,
            strategy=strategy_key, n_negatives=self.nneg, rng=rng,
            fill_pool=self.all_cde_ids,
            # Uniqueness is enforced on the RENDERED TEXT, not just the id: distinct ids can
            # render identically, and the trainer encodes texts.
            text_of=lambda cid: self.cde_id_to_text.get(str(cid), ""),
            positive_text=str(self.pos_texts[idx]) if idx < len(self.pos_texts) else None,
        )
        if res["fallback_used"]:
            self.fallback_stats["n_examples_with_fallback"] += 1
            self.fallback_stats["n_from_topk"] += res["n_from_topk"]
            self.fallback_stats["n_from_catalog_fill"] += res["n_from_fill"]
        return res["negative_texts"]

    def _canonical_strategy_key(self, band) -> str:
        """Map the resolved rank band onto a canonical strategy id for the selector."""
        if self.strategy == "none" or self.nneg <= 0:
            return "none"
        lo, hi = band
        # Register the effective band so any (curriculum) band is honoured exactly.
        _canon_hn.STRATEGY_BANDS[f"_effective_{lo}_{hi}"] = (int(lo), int(hi))
        if (int(lo), int(hi)) == (1, 25):
            return "hard_top25"
        if (int(lo), int(hi)) == (1, 50):
            return "semihard_1_50"
        return f"_effective_{lo}_{hi}"

    def __getitem__(self, idx: int):
        q = self.query_texts[idx]
        pos = self.pos_texts[idx]
        qkey = self.query_keys[idx]

        # strategy='none' means: train on (anchor, positive) only. This uses
        # in-batch negatives only (no mined negatives), and does not touch any
        # mined artifacts.
        if self.strategy == "none" or self.nneg <= 0:
            return _input_example([q, pos])

        neg_texts = self._sample_neg_texts(qkey=qkey, idx=idx)
        if len(neg_texts) != self.nneg:
            raise ValueError(
                f"[phase2] example {idx}: resolved {len(neg_texts)} negative texts, required "
                f"{self.nneg}. Refusing to pad with a duplicate or with the positive."
            )

        return _input_example([q, pos, *neg_texts])


def _hardneg_loss_factory(loss_name: str):
    """Create a loss object compatible with SentenceTransformers FitMixin.

    The returned loss supports N>=2 text columns:
      [anchor, positive, neg1, ..., negN]
    """

    name = (loss_name or "").strip().lower()
    if name not in {"mnrl", "mnr", "multiple_negatives", "multiple_negatives_ranking", "symmetric_mnrl", "sym_mnrl", "symmetric"}:
        raise ValueError(f"Unknown/unsupported loss for phase2: {loss_name}. Expected mnrl|symmetric_mnrl")

    import torch  # type: ignore
    from torch import nn  # type: ignore
    import torch.nn.functional as F  # type: ignore

    class _HardNegMN(nn.Module):
        def __init__(self, model, scale: float = 20.0, symmetric: bool = False):
            super().__init__()
            self.model = model
            self.scale = float(scale)
            self.symmetric = bool(symmetric)

        def _embed(self, sentence_features):
            # SentenceTransformers returns a dict with 'sentence_embedding'.
            return self.model(sentence_features)["sentence_embedding"]

        def forward(self, sentence_features, labels=None):  # noqa: ARG002
            if not isinstance(sentence_features, (list, tuple)) or len(sentence_features) < 2:
                raise ValueError("HardNegMN expects >=2 sentence feature columns")

            reps = [self._embed(sf) for sf in sentence_features]
            a = F.normalize(reps[0], p=2, dim=1)
            p = F.normalize(reps[1], p=2, dim=1)
            negs = [F.normalize(x, p=2, dim=1) for x in reps[2:]]

            # Candidate pool: all positives in batch + all hard negatives in batch.
            cand = [p]
            if negs:
                cand.extend(negs)
            cand_mat = torch.cat(cand, dim=0)  # (B*(1+K), d)

            scores = torch.matmul(a, cand_mat.t()) * self.scale  # (B, B*(1+K))
            labels_fwd = torch.arange(scores.size(0), device=scores.device)
            loss_fwd = F.cross_entropy(scores, labels_fwd)

            if not self.symmetric:
                return loss_fwd

            # Reverse direction (classic symmetric MN): positives query against anchors.
            # NOTE: hard negatives are CDE-side items, so they are not included in
            # the reverse candidate pool.
            scores_rev = torch.matmul(p, a.t()) * self.scale  # (B, B)
            labels_rev = torch.arange(scores_rev.size(0), device=scores_rev.device)
            loss_rev = F.cross_entropy(scores_rev, labels_rev)
            return 0.5 * (loss_fwd + loss_rev)

    symmetric = name in {"symmetric_mnrl", "sym_mnrl", "symmetric"}
    return _HardNegMN, symmetric


# -----------------------------------------------------------------------------
# Config handling
# -----------------------------------------------------------------------------



def _verify_fp32_runtime(model, *, requested_precision: str = "fp32") -> Dict[str, Any]:
    """Prove at runtime that Phase 2 really runs FP32: no autocast, no GradScaler, fp32 params.

    Returns a record whose ``effective_precision`` is derived from OBSERVED state, never from
    the requested value -- a mismatch is therefore detectable rather than assumed.
    """
    import torch

    param_dtype = None
    for prm in model.parameters():
        param_dtype = str(prm.dtype)
        break
    autocast_enabled = bool(torch.is_autocast_enabled())
    try:
        autocast_dtype = str(torch.get_autocast_dtype("cuda")) if torch.cuda.is_available() else None
    except Exception:
        autocast_dtype = None

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lin_out = None
    try:
        lin = torch.nn.Linear(8, 8).to(dev)
        lin_out = str(lin(torch.zeros(2, 8, device=dev)).dtype)
    except Exception:
        pass

    effective = "fp32" if (param_dtype == "torch.float32" and not autocast_enabled
                           and lin_out in (None, "torch.float32")) else "UNKNOWN_NOT_FP32"
    if effective != requested_precision:
        raise SystemExit(
            f"[phase2] PRECISION MISMATCH: requested {requested_precision!r} but observed "
            f"{effective!r} (param_dtype={param_dtype}, autocast={autocast_enabled}, "
            f"linear_out={lin_out})")

    rec = {
        "stage": "runtime_training_forward",
        "requested_precision": requested_precision,
        "effective_precision": effective,
        "autocast_enabled": autocast_enabled,
        "autocast_dtype": autocast_dtype,
        "grad_scaler_present": False,
        "grad_scaler_enabled": False,
        "param_dtype": param_dtype,
        "linear_out_dtype": lin_out,
        "device_type": dev,
        "trainer_class": "SentenceTransformer.fit (FitMixin, use_amp=False)",
        "use_amp": False,
    }
    try:
        import sentence_transformers, transformers
        rec["versions"] = {"torch": torch.__version__,
                           "sentence_transformers": sentence_transformers.__version__,
                           "transformers": transformers.__version__}
        if torch.cuda.is_available():
            rec["gpu"] = {"cuda_available": True, "gpu_name": torch.cuda.get_device_name(0),
                          "cuda_version": torch.version.cuda}
        else:
            rec["gpu"] = {"cuda_available": False}
    except Exception:
        pass
    return rec


def _apply_config_overrides(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    block = cfg.get("finetune_phase2") if isinstance(cfg.get("finetune_phase2"), dict) else {}
    if not block:
        return

    # Paths
    for k in ["cde_master_enriched", "splits_dir", "artifacts_dir", "runs_dir", "stage_tag", "mined_parquet", "init_model_name_or_path", "miner_model_name_or_path", "base_model_id"]:
        if getattr(args, k, None) is None and k in block:
            setattr(args, k, block.get(k))

    # Representation
    for k, default in [
        ("query_variant", "Q3"),
        ("recipe", "v1_v2_v3_v5"),
        ("cde_format", "labeled"),
        ("rerank_mode", "R0"),
        ("sep", " | "),
    ]:
        if getattr(args, k, None) is None:
            setattr(args, k, block.get(k, default))

    if getattr(args, "hybrid_alpha", None) is None and block.get("hybrid_alpha") is not None:
        args.hybrid_alpha = float(block.get("hybrid_alpha"))

    # Canonical paper provenance block (written verbatim into run_config.json).
    if isinstance(block.get("paper_provenance"), dict):
        args.paper_provenance = dict(block["paper_provenance"])

    # gradient_accumulation_steps was historically present in configs but NEVER consumed by
    # model.fit(). Reject any value other than 1 rather than silently ignoring it.
    tr = block.get("train") if isinstance(block.get("train"), dict) else {}
    if "gradient_accumulation_steps" in tr and int(tr["gradient_accumulation_steps"]) != 1:
        raise SystemExit(
            f"[phase2] config sets gradient_accumulation_steps="
            f"{tr['gradient_accumulation_steps']}, but Phase 2 runs with 1 and this key was "
            f"never consumed by the trainer. Remove it or set it to 1.")

    # Recipe configs (dict)
    if getattr(args, "recipe_configs", None) is None and isinstance(block.get("recipe_configs"), dict):
        args.recipe_configs = block.get("recipe_configs")

    # Training/eval sections
    tr = block.get("train") if isinstance(block.get("train"), dict) else {}
    ev = block.get("eval") if isinstance(block.get("eval"), dict) else {}
    hn = block.get("hardneg") if isinstance(block.get("hardneg"), dict) else {}

    if getattr(args, "device", None) is None:
        args.device = tr.get("device", "auto")
    if getattr(args, "max_seq_length", None) is None:
        args.max_seq_length = int(tr.get("max_seq_length", 256))
    if getattr(args, "warmup_ratio", None) is None:
        args.warmup_ratio = float(tr.get("warmup_ratio", 0.1))

    if getattr(args, "seeds", None) is None and tr.get("seeds") is not None:
        args.seeds = ",".join(str(x) for x in (tr.get("seeds") or []))
    if getattr(args, "losses", None) is None and tr.get("losses") is not None:
        args.losses = ",".join(str(x) for x in (tr.get("losses") or []))
    if getattr(args, "lrs", None) is None and tr.get("lrs") is not None:
        args.lrs = ",".join(str(x) for x in (tr.get("lrs") or []))
    if getattr(args, "batch_sizes", None) is None and tr.get("batch_sizes") is not None:
        args.batch_sizes = ",".join(str(x) for x in (tr.get("batch_sizes") or []))
    if getattr(args, "temperatures", None) is None and tr.get("temperatures") is not None:
        args.temperatures = ",".join(str(x) for x in (tr.get("temperatures") or []))
    if getattr(args, "epochs", None) is None and tr.get("epochs") is not None:
        args.epochs = ",".join(str(x) for x in (tr.get("epochs") or []))

    if getattr(args, "top_k", None) is None:
        args.top_k = int(ev.get("top_k", 100))
    if getattr(args, "k_values", None) is None:
        args.k_values = ",".join(str(x) for x in (ev.get("k_values") or [5, 10, 20]))
    if getattr(args, "output_top_k", None) is None:
        args.output_top_k = int(ev.get("output_top_k", 20))
    if getattr(args, "block_size", None) is None:
        args.block_size = int(ev.get("block_size", 2048))
    if getattr(args, "encode_batch_size", None) is None:
        args.encode_batch_size = int(ev.get("encode_batch_size", 64))
    if getattr(args, "normalize_embeddings", None) is None:
        args.normalize_embeddings = bool(ev.get("normalize_embeddings", True))
    if getattr(args, "eval_splits", None) is None and ev.get("eval_splits") is not None:
        args.eval_splits = ",".join(str(x) for x in (ev.get("eval_splits") or []))

    if getattr(args, "write_epoch_rankings", None) is None:
        args.write_epoch_rankings = bool(ev.get("write_epoch_rankings", False))
    if getattr(args, "epoch_rankings_splits", None) is None and ev.get("epoch_rankings_splits") is not None:
        args.epoch_rankings_splits = ",".join(str(x) for x in (ev.get("epoch_rankings_splits") or []))

    # Hardneg knobs
    if getattr(args, "strategies", None) is None and hn.get("strategies") is not None:
        args.strategies = ",".join(str(x) for x in (hn.get("strategies") or []))
    if getattr(args, "nneg", None) is None:
        args.nneg = int(hn.get("nneg", 10))
    if getattr(args, "mined_top_k", None) is None:
        args.mined_top_k = int(hn.get("top_k", 200))

    if getattr(args, "hard_band", None) is None and hn.get("hard_band") is not None:
        args.hard_band = str(hn.get("hard_band"))
    if getattr(args, "semihard_band", None) is None and hn.get("semihard_band") is not None:
        args.semihard_band = str(hn.get("semihard_band"))
    if getattr(args, "curriculum_bands", None) is None and hn.get("curriculum_bands") is not None:
        args.curriculum_bands = ",".join(str(x) for x in (hn.get("curriculum_bands") or []))

    # Optional additional strategy knobs
    if getattr(args, "hardcurr_bands", None) is None and hn.get("hardcurr_bands") is not None:
        args.hardcurr_bands = ",".join(str(x) for x in (hn.get("hardcurr_bands") or []))
    if getattr(args, "semihard_tight_band", None) is None and hn.get("semihard_tight_band") is not None:
        args.semihard_tight_band = str(hn.get("semihard_tight_band"))


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
    strategy: str
    nneg: int
    val_mrr100: float
    val_recall5: float


def _write_leaderboard(rows: Sequence[RunRow], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df.sort_values(["val_recall5", "val_mrr100"], ascending=False)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="demap finetune-phase2",
        description="Phase 2 fine-tune with mined hard negatives.",
    )

    p.add_argument("--config", type=str, default=None, help="YAML config file (expects finetune_phase2: block)")

    # Paths
    p.add_argument("--cde-master-enriched", dest="cde_master_enriched", type=str, default=None)
    p.add_argument("--splits-dir", dest="splits_dir", type=str, default=None)
    p.add_argument("--artifacts-dir", dest="artifacts_dir", type=str, default=None)
    p.add_argument("--runs-dir", dest="runs_dir", type=str, default=None)
    p.add_argument("--stage-tag", dest="stage_tag", type=str, default=None, help="Optional stage tag used in auto roots and run IDs.")

    p.add_argument("--mined-parquet", dest="mined_parquet", type=str, default=None, help="Mined hardneg parquet")

    # Models
    p.add_argument(
        "--init-model-name-or-path",
        dest="init_model_name_or_path",
        type=str,
        default=None,
        help="Model to initialize Phase 2 training from (typically a Phase 1 checkpoint dir)",
    )
    p.add_argument(
        "--miner-model-name-or-path",
        dest="miner_model_name_or_path",
        type=str,
        default=None,
        help="Model used to mine negatives (for logging only; defaults to init model)",
    )

    p.add_argument(
        "--base-model-id",
        dest="base_model_id",
        type=str,
        default=None,
        help="Canonical hub id for the model family (recommended when runs_dir=auto and init_model_name_or_path is a local checkpoint path).",
    )

    # Representation
    p.add_argument("--query-variant", dest="query_variant", type=str, default=None)
    p.add_argument("--recipe", dest="recipe", type=str, default=None)
    p.add_argument("--cde-format", dest="cde_format", type=str, default=None)
    p.add_argument("--rerank-mode", dest="rerank_mode", type=str, default=None)
    p.add_argument("--hybrid-alpha", dest="hybrid_alpha", type=float, default=None)
    p.add_argument("--sep", dest="sep", type=str, default=None)

    # Training grid
    p.add_argument("--device", type=str, default=None, help="auto|cuda|mps|cpu")
    p.add_argument("--require-gpu", action="store_true",
                   help="fail fast unless training actually runs on CUDA (no CPU fallback); "
                        "also enabled by DEMAP_REQUIRE_GPU=1")
    p.add_argument("--max-seq-length", dest="max_seq_length", type=int, default=None)
    p.add_argument("--warmup-ratio", dest="warmup_ratio", type=float, default=None)

    p.add_argument("--seeds", type=str, default=None, help="CSV of seeds")
    p.add_argument("--losses", type=str, default=None, help="CSV of losses (mnrl,symmetric_mnrl)")
    p.add_argument("--lrs", type=str, default=None, help="CSV of learning rates")
    p.add_argument("--batch-sizes", dest="batch_sizes", type=str, default=None, help="CSV of batch sizes")
    p.add_argument("--temperatures", type=str, default=None, help="CSV of temperatures")
    p.add_argument("--epochs", type=str, default=None, help="CSV of epochs")

    # Hardneg
    p.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="CSV of strategies: none,hard,semihard,semihard_tight,curr,hardcurr",
    )
    p.add_argument("--nneg", type=int, default=None, help="# explicit negatives per example")
    p.add_argument("--max-train-rows", dest="max_train_rows", type=int, default=None,
                   help="SMOKE TEST ONLY: cap training rows (deterministic head). Marks the run "
                        "non-canonical in provenance; never use for a paper run.")
    p.add_argument("--mined-top-k", dest="mined_top_k", type=int, default=None, help="top_k used in mined parquet")
    p.add_argument("--hard-band", type=str, default=None, help="Rank band like 1-25")
    p.add_argument("--semihard-band", type=str, default=None, help="Rank band like 25-100")
    p.add_argument(
        "--curriculum-bands",
        type=str,
        default=None,
        help="CSV of bands by epoch starting at epoch1, e.g. '50-200,25-100,1-25'",
    )
    p.add_argument(
        "--hardcurr-bands",
        dest="hardcurr_bands",
        type=str,
        default=None,
        help="CSV of bands by epoch for hardcurr, e.g. '10-25,5-15,1-10'",
    )
    p.add_argument(
        "--semihard-tight-band",
        dest="semihard_tight_band",
        type=str,
        default=None,
        help="Rank band for semihard_tight, e.g. 10-50",
    )

    # Eval
    p.add_argument("--top-k", dest="top_k", type=int, default=None)
    p.add_argument("--k-values", dest="k_values", type=str, default=None)
    p.add_argument("--output-top-k", dest="output_top_k", type=int, default=None)
    p.add_argument("--block-size", dest="block_size", type=int, default=None)
    p.add_argument("--encode-batch-size", dest="encode_batch_size", type=int, default=None)
    p.add_argument(
        "--normalize-embeddings",
        dest="normalize_embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--eval-splits", dest="eval_splits", type=str, default=None, help="Comma-separated splits to evaluate after training (default: all splits in splits_dir).")

    # Optional: write a rankings parquet per epoch (useful for diagnostics notebooks)
    p.add_argument(
        "--write-epoch-rankings",
        dest="write_epoch_rankings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If enabled, write rankings_epoch_XX.parquet at each epoch (default split: val).",
    )
    p.add_argument(
        "--epoch-rankings-splits",
        dest="epoch_rankings_splits",
        type=str,
        default=None,
        help="CSV of splits to evaluate for per-epoch rankings (e.g. 'val' or 'val,test').",
    )

    p.add_argument("--dry-run", action="store_true", help="Print planned runs and exit")

    args = p.parse_args(argv)

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)
        _apply_config_overrides(args, cfg)

    # Fill safe defaults (so `--help` works even if config is omitted).
    if args.query_variant is None:
        args.query_variant = "Q3"
    if args.recipe is None:
        args.recipe = "v1_v2_v3_v5"
    if args.cde_format is None:
        args.cde_format = "labeled"
    if args.rerank_mode is None:
        args.rerank_mode = "R0"
    if args.sep is None:
        args.sep = " | "
    if args.device is None:
        args.device = "auto"
    if args.max_seq_length is None:
        args.max_seq_length = 256
    if args.warmup_ratio is None:
        args.warmup_ratio = 0.1
    if args.top_k is None:
        args.top_k = 100
    if args.k_values is None:
        args.k_values = "5,10,20"
    if args.output_top_k is None:
        args.output_top_k = 20
    if args.block_size is None:
        args.block_size = 2048
    if args.encode_batch_size is None:
        args.encode_batch_size = 64
    if args.normalize_embeddings is None:
        args.normalize_embeddings = True
    if args.hybrid_alpha is None:
        args.hybrid_alpha = 0.5

    if getattr(args, "write_epoch_rankings", None) is None:
        args.write_epoch_rankings = False
    if getattr(args, "epoch_rankings_splits", None) is None:
        args.epoch_rankings_splits = "val"

    # Required args.
    if args.cde_master_enriched is None:
        raise SystemExit("Missing --cde-master-enriched (or finetune_phase2.cde_master_enriched in config)")
    if args.splits_dir is None:
        raise SystemExit("Missing --splits-dir (or finetune_phase2.splits_dir in config)")
    if args.init_model_name_or_path is None:
        raise SystemExit(
            "Missing --init-model-name-or-path (or finetune_phase2.init_model_name_or_path in config)"
        )
    if not is_valid_recipe(str(args.recipe)):
        raise SystemExit(f"Unknown recipe: {args.recipe}")

    device = str(args.device or "auto").strip().lower()
    if device == "auto":
        device = _auto_device()
    if device not in {"cpu", "cuda", "mps"}:
        raise SystemExit(f"Unknown device: {args.device}. Expected auto|cpu|cuda|mps")
    require_gpu = bool(getattr(args, "require_gpu", False)) or \
        os.environ.get("DEMAP_REQUIRE_GPU") == "1"
    assert_gpu_training(device, require=require_gpu)

    miner_model = str(args.miner_model_name_or_path or args.init_model_name_or_path)

    # Expand grid.
    seeds = _parse_csv_ints(args.seeds or "0")
    losses = _parse_csv_list(args.losses or "symmetric_mnrl")
    lrs = _parse_csv_floats(args.lrs or "2e-5")
    batch_sizes = _parse_csv_ints(args.batch_sizes or "64")
    temps = _parse_csv_floats(args.temperatures or "0.05")
    epochs_list = _parse_csv_ints(args.epochs or "3")
    only_splits = _parse_csv_list(args.eval_splits) if args.eval_splits is not None else None
    # Canonical Phase 2 tuning NEVER touches test (or val_train as a competing metric).
    if getattr(args, "paper_provenance", None):
        if only_splits is None or [str(x).strip().lower() for x in only_splits] != ["val_dev"]:
            raise SystemExit(
                f"[phase2] canonical Phase 2 must evaluate exactly ['val_dev'] during tuning; "
                f"got {only_splits!r}. All-dataset evaluation is a separate read-only stage.")

    raw_strategies = [s.strip().lower() for s in _parse_csv_list(args.strategies or "none,semihard,hard,curr")]
    strategies = _normalize_strategies(raw_strategies)
    if not strategies:
        raise SystemExit(
            "No strategies specified. "
            f"Supported values: {', '.join(sorted(_STRATEGY_LABEL_TO_BASE))}"
        )

    # Configured nneg (may be ignored for strategy='none')
    nneg = int(args.nneg or 10)
    mined_top_k = int(args.mined_top_k or 200)
    hard_band = _parse_band(args.hard_band or "1-25")
    semihard_band = _parse_band(args.semihard_band or "25-100")
    curriculum_bands = [_parse_band(x) for x in _parse_csv_list(args.curriculum_bands or "50-200,25-100,1-25")]

    # Additional strategy-specific bands
    semihard_tight_band = _parse_band(getattr(args, "semihard_tight_band", None) or "10-50")
    hardcurr_bands = [_parse_band(x) for x in _parse_csv_list(getattr(args, "hardcurr_bands", None) or "10-25,5-15,1-10")]

    # Mined parquet is required iff any strategy uses mined negatives.
    requires_mined = any(s != "none" for s in strategies)
    mined_path: Optional[Path] = None
    mined_query_to_candidates: Dict[str, List[str]] = {}
    mined_query_to_gold: Dict[str, List[str]] = {}
    if requires_mined:
        if args.mined_parquet is None:
            raise SystemExit("Missing --mined-parquet (required when using mined-negative strategies)")
        mined_path = Path(args.mined_parquet)
        if not mined_path.exists():
            raise SystemExit(
                f"Mined parquet not found: {mined_path}. Run `demap mine-hardneg ...` to create it."
            )
    else:
        # If provided, we intentionally ignore mined_parquet for strategy='none'.
        if args.mined_parquet is not None:
            mined_path = Path(args.mined_parquet)

    # Paths
    artifacts_dir = Path(args.artifacts_dir or "artifacts")
    if args.runs_dir is None:
        runs_dir = artifacts_dir / "runs"
    elif str(args.runs_dir).lower() == "auto":
        base_model_id = getattr(args, "base_model_id", None)
        if not base_model_id:
            # If init is a hub id (not a local path), we can use it as base_model_id.
            init_s = str(args.init_model_name_or_path or "")
            if init_s and not Path(init_s).exists():
                base_model_id = init_s
        if not base_model_id:
            raise ValueError("runs_dir=auto requires base_model_id when init_model_name_or_path is a local checkpoint path")
        phase_root = _normalize_stage_tag(getattr(args, "stage_tag", None)) or "finetune_phase2"
        runs_dir = artifacts_dir / phase_root / _slug(str(base_model_id)) / "runs"
    else:
        runs_dir = Path(args.runs_dir)
    ensure_dir(artifacts_dir)
    ensure_dir(runs_dir)

    # Load data (catalog + splits)
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
    cde_ids = catalog_df["cde_id"].astype(str).values
    cde_texts = catalog_df["cde_text"].astype(str).tolist()
    cde_id_to_text = dict(zip(catalog_df["cde_id"].astype(str).tolist(), cde_texts))
    all_cde_ids = catalog_df["cde_id"].astype(str).tolist()

    splits = bg._load_splits(Path(args.splits_dir))
    train_path = splits.get("train")
    # Validation split resolution: prefer legacy `val`; else the canonical v3-CDISC `val_dev`.
    # Mirrors the finetune_phase1 fallback (commit 5fa9df4) so canonical splits that have no bare
    # `val.parquet` still train (the old hard requirement crashed at data-load on those dirs).
    val_split_name = "val" if splits.get("val") is not None else (
        "val_dev" if splits.get("val_dev") is not None else None)
    val_path = splits.get(val_split_name) if val_split_name else None
    test_path = splits.get("test")
    if train_path is None or val_path is None:
        raise SystemExit("splits_dir must contain train.parquet and (val.parquet or val_dev.parquet)")
    print(f"[finetune-phase2] train={os.path.basename(str(train_path))}  "
          f"validation={val_split_name}({os.path.basename(str(val_path))})")

    query_col = bg.QUERY_VARIANT_TO_COL.get(str(args.query_variant), str(args.query_variant))
    cols_needed = ["query_id", "cde_id", "cde_publicid", "cde_version", query_col, "pv_attached", "PV_N", "PV_TYPE"]

    def _read_split(p: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(p, columns=[c for c in cols_needed if c])
        except Exception:
            return pd.read_parquet(p)

    train_df = _read_split(train_path)
    val_df = _read_split(val_path)
    test_df = _read_split(test_path) if test_path is not None else None

    train_df = _coerce_cde_id(train_df)
    if getattr(args, "max_train_rows", None):
        n_cap = int(args.max_train_rows)
        train_df = train_df.head(n_cap).reset_index(drop=True)
        print(f"[phase2] SMOKE TEST: training capped to {len(train_df)} rows (non-canonical)")
    val_df = _coerce_cde_id(val_df)
    if test_df is not None:
        test_df = _coerce_cde_id(test_df)

    if query_col not in train_df.columns:
        raise SystemExit(f"Train split missing query column: {query_col}")

    # Build training arrays
    q_raw = train_df[query_col].fillna("").astype(str).tolist()
    c_ids = train_df["cde_id"].astype(str).tolist()
    query_texts: List[str] = []
    query_keys: List[str] = []
    pos_cde_ids: List[str] = []
    pos_texts: List[str] = []
    n_empty_query = 0
    n_missing_target = 0
    for qt, cid in zip(q_raw, c_ids):
        qt2 = str(qt).strip()
        if qt2 == "":
            n_empty_query += 1
            continue
        ct = cde_id_to_text.get(str(cid))
        if ct is None or str(ct).strip() == "":
            n_missing_target += 1
            continue
        query_texts.append(qt2)
        query_keys.append(normalize_query_text(qt2))
        pos_cde_ids.append(str(cid))
        pos_texts.append(str(ct))

    train_pair_stats = {
        "n_rows_in_split": int(len(train_df)),
        "n_pairs_built": int(len(query_texts)),
        "n_dropped_empty_query": int(n_empty_query),
        "n_dropped_missing_target_text": int(n_missing_target),
    }

    if len(query_texts) == 0:
        raise SystemExit("No training pairs were built (check query_col and catalog mapping).")

    # Mined artifact mapping (only needed for mining strategies)
    if requires_mined:
        assert mined_path is not None
        mined = pd.read_parquet(mined_path)
        need_cols = {"query_key", "gold_cde_ids", "candidates_cde_id"}
        missing_cols = [c for c in need_cols if c not in mined.columns]
        if missing_cols:
            raise SystemExit(f"Mined parquet missing required columns: {missing_cols}")

        mined_query_to_candidates, mined_query_to_gold = _build_mined_query_maps(mined)

        # Sanity: if mined parquet has *zero* overlap with training queries, then
        # Phase 2 will silently degrade into mostly-random negatives. Fail fast.
        unique_train_qkeys = set(query_keys)
        n_cov = sum(1 for qk in unique_train_qkeys if mined_query_to_candidates.get(qk))
        if n_cov == 0:
            raise SystemExit(
                "Mined parquet produced 0 candidate lists for the training queries. "
                "This usually indicates a bug (mined mapping not loaded) or a config mismatch "
                "(wrong mined parquet, split, query variant, or query_key normalization)."
            )
        coverage = n_cov / max(len(unique_train_qkeys), 1)
        if coverage < 0.5:
            print(
                f"[finetune-phase2] WARNING: mined coverage is low ({coverage:.1%}). "
                "Many queries will fall back to random negatives."
            )
        else:
            print(f"[finetune-phase2] mined coverage: {coverage:.1%} ({n_cov}/{len(unique_train_qkeys)})")

    # Planned runs.
    planned: List[Tuple[str, int, str, float, int, float, int, str]] = []
    for strat in strategies:
        effective_nneg = 0 if str(strat).lower() == "none" else int(nneg)
        for loss in losses:
            for lr in lrs:
                for bs in batch_sizes:
                    for temp in temps:
                        for ep in epochs_list:
                            for seed in seeds:
                                init_model_tag = _short_model_tag(args.init_model_name_or_path)
                                stage_tag = _normalize_stage_tag(getattr(args, "stage_tag", None)) or None
                                run_id = (
                                    f"{time.strftime('%Y%m%d_%H%M%S')}"
                                    + (f"__{stage_tag}" if stage_tag else "")
                                    + f"__ft2__{init_model_tag}"
                                    + f"__{args.query_variant}__{args.recipe}__{args.cde_format}__{args.rerank_mode}"
                                    + f"__{loss}__lr{lr:g}__bs{bs}__t{temp:g}__ep{ep}__seed{seed}"
                                    + f"__topk{mined_top_k}__nneg{effective_nneg}__mine{strat}"
                                )
                                run_id = _truncate_with_hash(run_id, max_len=200)
                                planned.append(
                                    (run_id, int(seed), str(loss), float(lr), int(bs), float(temp), int(ep), str(strat))
                                )

    if args.dry_run:
        print(f"Planned runs: {len(planned)}")
        for r in planned[:50]:
            print(r[0])
        if len(planned) > 50:
            print("...")
        return

    leaderboard_rows: List[RunRow] = []
    reports_dir = artifacts_dir / "reports" / "finetune_phase2"
    ensure_dir(reports_dir)

    for run_id, seed, loss_name, lr, bs, temp, ep, strat in planned:
        run_dir = runs_dir / run_id
        model_out_dir = run_dir / "model"
        ensure_dir(model_out_dir)

        _set_seed(seed)

        strat_l = str(strat).lower()
        effective_nneg = 0 if strat_l == "none" else int(nneg)
        semihard_band_for_run = semihard_tight_band if strat_l == "semihard_tight" else semihard_band
        curriculum_bands_for_run = hardcurr_bands if strat_l == "hardcurr" else curriculum_bands

        # Dataset (strategy='none' yields (anchor, positive) only)
        train_dataset = HardNegTupleDataset(
            query_texts=query_texts,
            query_keys=query_keys,
            pos_cde_ids=pos_cde_ids,
            pos_texts=pos_texts,
            cde_id_to_text=cde_id_to_text,
            all_cde_ids=all_cde_ids,
            mined_query_to_candidates=mined_query_to_candidates,
            mined_query_to_gold=mined_query_to_gold,
            strategy=strat,
            nneg=effective_nneg,
            top_k=mined_top_k,
            hard_band=hard_band,
            semihard_band=semihard_band_for_run,
            curriculum_epoch_bands=curriculum_bands_for_run,
            seed=seed,
        )
        train_dataset.set_epoch(0)

        # Batch collision sampler (on anchor+positive only, as in Phase 1)
        q_keys = [normalize_query_text(q) for q in query_texts]
        c_keys = [normalize_query_text(c) for c in pos_texts]
        batch_sampler = NoDuplicateTextBatchSampler(
            q_keys,
            c_keys,
            batch_size=int(bs),
            seed=int(seed),
            drop_last=False,
            shuffle=True,
        )
        sampler = BatchSamplerAsSampler(batch_sampler)

        # Model
        model = _load_sentence_transformer(str(args.init_model_name_or_path), device=device)
        assert_model_on_cuda(model, require=require_gpu)
        try:
            model.max_seq_length = int(args.max_seq_length)
        except Exception:
            pass

        # DataLoader
        try:
            from torch.utils.data import DataLoader  # type: ignore

            train_dataloader = DataLoader(
                train_dataset,
                sampler=sampler,
                batch_size=int(bs),
                drop_last=False,
                collate_fn=model.smart_batching_collate,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to construct training DataLoader: {type(e).__name__}: {e}")

        # Loss (custom, supports extra negative columns)
        scale = float(1.0 / float(temp)) if float(temp) > 0 else 20.0
        LossCls, symmetric_flag = _hardneg_loss_factory(loss_name)
        train_loss = LossCls(model=model, scale=scale, symmetric=symmetric_flag)

        # Val evaluator for checkpoint selection + curriculum epoch updates.
        try:
            from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator  # type: ignore
        except Exception:  # pragma: no cover
            SentenceEvaluator = object  # type: ignore
            SequentialEvaluator = None  # type: ignore

        # Per-epoch rankings writing (optional)
        epoch_rankings_splits = [
            s.strip().lower() for s in _parse_csv_list(str(getattr(args, "epoch_rankings_splits", "val") or "val"))
        ]
        epoch_rankings_splits = [s for s in epoch_rankings_splits if s]
        if not epoch_rankings_splits:
            epoch_rankings_splits = ["val"]

        split_dfs: Dict[str, Optional[pd.DataFrame]] = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

        rerank_mode_resolved = bg.RERANK_MODE_ALIASES.get(str(args.rerank_mode), str(args.rerank_mode))
        tfidf_cache_dir = artifacts_dir / "cache" / "tfidf"
        try:
            recipe_cfg_key = json.dumps(recipe_configs or {}, sort_keys=True)
        except Exception:
            recipe_cfg_key = ""
        tfidf_cache_key = _slug(f"phase2__{args.recipe}__{args.cde_format}__{recipe_cfg_key}")

        class _ValEvaluator(SentenceEvaluator):  # type: ignore
            def __init__(
                self,
                dataset: HardNegTupleDataset,
                total_epochs: int,
                *,
                run_dir: Path,
                write_epoch_rankings: bool,
                epoch_rankings_splits: Sequence[str],
            ):
                try:
                    super().__init__()  # type: ignore[misc]
                except Exception:
                    pass
                self.history: List[Dict[str, Any]] = []
                self.dataset = dataset
                self.total_epochs = int(total_epochs)
                self.run_dir = Path(run_dir)
                self.write_epoch_rankings = bool(write_epoch_rankings)
                self.epoch_rankings_splits = list(epoch_rankings_splits)

            def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
                # Default path: use the lighter Phase-1-style val evaluator.
                # When write_epoch_rankings is enabled, we compute rankings anyway
                # (heavier) and reuse the same pass to obtain val metrics.
                cde_emb = None
                q_emb_val = None
                rankings_val = None
                if not self.write_epoch_rankings:
                    m = ft1._compute_val_metrics(
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
                else:
                    # Encode catalog once per evaluation call.
                    cde_emb = ft1._encode_np(
                        model=model,
                        texts=cde_texts,
                        batch_size=int(args.encode_batch_size),
                        normalize=bool(args.normalize_embeddings),
                    )

                    # Standard val metrics used for checkpoint selection.
                    q_val = val_df[query_col].fillna("").astype(str).tolist()
                    q_emb_val = ft1._encode_np(
                        model=model,
                        texts=q_val,
                        batch_size=int(args.encode_batch_size),
                        normalize=bool(args.normalize_embeddings),
                    )
                    m, rankings_val, _fail_val = bg._evaluate_split(
                        split_name="val",
                        df_split=val_df,
                        query_col=query_col,
                        cde_ids=cde_ids,
                        cde_texts=cde_texts,
                        cde_emb=cde_emb,
                        query_emb=q_emb_val,
                        top_k=max(int(args.top_k), 100),
                        k_values=_parse_csv_ints(args.k_values or "5,10,20"),
                        block_size=int(args.block_size),
                        rerank_mode=rerank_mode_resolved,
                        alpha=float(args.hybrid_alpha),
                        tfidf_cache_dir=tfidf_cache_dir,
                        tfidf_cache_key=tfidf_cache_key,
                        output_top_k=int(args.output_top_k),
                    )
                rec = {
                    "epoch": int(epoch),
                    "steps": int(steps),
                    **{k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating))},
                }
                self.history.append(rec)

                # Optional: write per-epoch rankings parquet(s) for diagnostics.
                # Guard on epoch>=0 to avoid the initial pre-train evaluation.
                if self.write_epoch_rankings and int(epoch) >= 0:
                    try:
                        import pyarrow  # noqa: F401

                        out_pq = self.run_dir / f"rankings_epoch_{int(epoch):02d}.parquet"
                        if not out_pq.exists():
                            rankings_all: List[pd.DataFrame] = []

                            # Reuse val rankings from the checkpoint-selection pass.
                            if rankings_val is not None and "val" in self.epoch_rankings_splits:
                                r0 = rankings_val.copy()
                                r0.insert(0, "epoch", int(epoch))
                                rankings_all.append(r0)

                            for split_name in self.epoch_rankings_splits:
                                if str(split_name).lower() == "val" and rankings_val is not None:
                                    continue
                                df_split = split_dfs.get(split_name)
                                if df_split is None or not isinstance(df_split, pd.DataFrame):
                                    continue
                                if query_col not in df_split.columns:
                                    continue

                                # Defensive: cde_emb is required for rankings, but should exist here.
                                if cde_emb is None:
                                    break

                                q_txt = df_split[query_col].fillna("").astype(str).tolist()
                                q_emb = ft1._encode_np(
                                    model=model,
                                    texts=q_txt,
                                    batch_size=int(args.encode_batch_size),
                                    normalize=bool(args.normalize_embeddings),
                                )
                                _m_s, r_s, _f_s = bg._evaluate_split(
                                    split_name=str(split_name),
                                    df_split=df_split,
                                    query_col=query_col,
                                    cde_ids=cde_ids,
                                    cde_texts=cde_texts,
                                    cde_emb=cde_emb,
                                    query_emb=q_emb,
                                    top_k=max(int(args.top_k), 100),
                                    k_values=_parse_csv_ints(args.k_values or "5,10,20"),
                                    block_size=int(args.block_size),
                                    rerank_mode=rerank_mode_resolved,
                                    alpha=float(args.hybrid_alpha),
                                    tfidf_cache_dir=tfidf_cache_dir,
                                    tfidf_cache_key=tfidf_cache_key,
                                    output_top_k=int(args.output_top_k),
                                )
                                r_s = r_s.copy()
                                r_s.insert(0, "epoch", int(epoch))
                                rankings_all.append(r_s)

                            if rankings_all:
                                rankings_epoch = pd.concat(rankings_all, ignore_index=True)
                                rankings_epoch.to_parquet(out_pq, index=False)
                    except Exception:
                        # Best-effort: per-epoch rankings should not break training.
                        pass

                # Curriculum update: set dataset epoch for *next* epoch.
                if str(strat).lower() in {"curr", "hardcurr"} and int(epoch) >= 0:
                    self.dataset.set_epoch(int(epoch) + 1)

                return float(m.get("mrr@100", float("nan")))

        val_evaluator = _ValEvaluator(
            train_dataset,
            total_epochs=int(ep),
            run_dir=run_dir,
            write_epoch_rankings=bool(getattr(args, "write_epoch_rankings", False)),
            epoch_rankings_splits=epoch_rankings_splits,
        )
        evaluator = SequentialEvaluator([val_evaluator]) if SequentialEvaluator is not None else val_evaluator

        # Log run config (minimal but explicit)
        run_config = {
            "run_id": str(run_id),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": str(device),

            "base_model_id": str(getattr(args, "base_model_id", "")),
            # Backward-compatible top-level aliases for representation (used by inspection/export tools).
            "query_variant": str(args.query_variant),
            "query_col": str(query_col),
            "recipe": str(args.recipe),
            "cde_format": str(args.cde_format),
            "rerank_mode": str(args.rerank_mode),
            "sep": str(args.sep),
            "recipe_configs": (recipe_configs or {}),

            "init_model_name_or_path": str(args.init_model_name_or_path),
            "miner_model_name_or_path": str(miner_model),
            "mined_parquet": str(mined_path) if mined_path is not None else None,
            "representation": {
                "query_variant": str(args.query_variant),
                "query_col": str(query_col),
                "recipe": str(args.recipe),
                "cde_format": str(args.cde_format),
                "rerank_mode": str(args.rerank_mode),
                "sep": str(args.sep),
                "recipe_configs": (recipe_configs or {}),
            },
            "hardneg": {
                "strategy": str(strat),
                "nneg": int(nneg),
                "effective_nneg": int(effective_nneg),
                "top_k": int(mined_top_k),
                "hard_band": list(hard_band),
                "semihard_band": list(semihard_band),
                "semihard_tight_band": list(semihard_tight_band),
                "curriculum_bands": [list(b) for b in curriculum_bands],
                "hardcurr_bands": [list(b) for b in hardcurr_bands],
                "effective_semihard_band": list(semihard_band_for_run),
                "effective_curriculum_bands": [list(b) for b in curriculum_bands_for_run]
                if strat_l in {"curr", "hardcurr"}
                else None,
            },
            "train": {
                "seed": int(seed),
                "loss": str(loss_name),
                "lr": float(lr),
                "batch_size": int(bs),
                "temperature": float(temp),
                "scale": float(scale),
                "epochs": int(ep),
                "warmup_ratio": float(args.warmup_ratio),
                "max_seq_length": int(args.max_seq_length),
            },
            "train_pair_stats": train_pair_stats,
            "data": {
                "splits_dir": str(args.splits_dir),
                "validation_split": str(val_split_name),
            },
            "eval": {
                "validation_split": str(val_split_name),
                "top_k": int(args.top_k),
                "k_values": [int(x) for x in _parse_csv_ints(args.k_values or "5,10,20")],
                "output_top_k": int(args.output_top_k),
                "block_size": int(args.block_size),
                "encode_batch_size": int(args.encode_batch_size),
                "normalize_embeddings": bool(args.normalize_embeddings),
                "write_epoch_rankings": bool(getattr(args, "write_epoch_rankings", False)),
                "epoch_rankings_splits": [
                    s.strip().lower()
                    for s in _parse_csv_list(str(getattr(args, "epoch_rankings_splits", "val") or "val"))
                    if s.strip()
                ],
            },
        }
        steps_per_epoch = max(int(len(train_dataloader)), 1)
        loss_history_jsonl = run_dir / "train_loss_history.jsonl"
        loss_history_csv = run_dir / "train_loss_history.csv"
        run_config["training_loss_history"] = {
            "jsonl": str(loss_history_jsonl),
            "csv": str(loss_history_csv),
            "steps_per_epoch": int(steps_per_epoch),
        }
        write_json(run_config, run_dir / "run_config.json")

        # ---- Canonical Phase 2 precision contract: honest FP32.
        # FitMixin.fit defaults use_amp=False; we pass it EXPLICITLY so the intent is in the
        # call, not in a default, and we verify the realised dtypes at runtime.
        prov_cfg = dict(getattr(args, "paper_provenance", None) or {})
        requested_precision = str(prov_cfg.get("requested_precision") or "fp32")
        if requested_precision != "fp32":
            raise SystemExit(
                f"[phase2] canonical Phase 2 precision is fp32; got {requested_precision!r}")
        grad_accum = int(prov_cfg.get("gradient_accumulation_steps", 1))
        if grad_accum != 1:
            raise SystemExit(
                f"[phase2] gradient_accumulation_steps must be 1 (got {grad_accum}); the "
                f"historical value 4 was never consumed and is rejected")
        precision_record = _verify_fp32_runtime(model, requested_precision=requested_precision)
        write_json(precision_record, run_dir / "precision_verification.json")
        run_config["precision_verification"] = {
            "requested_precision": requested_precision,
            "stage": "preflight_fp32",
            "runtime": precision_record,
        }
        if prov_cfg:
            _capped = int(getattr(args, "max_train_rows", 0) or 0)
            run_config["paper_provenance"] = {
                **prov_cfg,
                "smoke_test_max_train_rows": _capped or None,
                "is_canonical_paper_run": bool(prov_cfg.get("is_canonical_paper_run")) and not _capped,
                "effective_precision": precision_record["effective_precision"],
                "autocast_enabled": precision_record["autocast_enabled"],
                "grad_scaler_present": precision_record["grad_scaler_present"],
                "grad_scaler_enabled": precision_record["grad_scaler_enabled"],
                "param_dtype": precision_record["param_dtype"],
                "gradient_accumulation_steps": grad_accum,
                "effective_optimizer_batch_size": int(bs) * grad_accum,
                "negative_strategy": prov_cfg.get("negative_strategy", str(strat)),
                "texts_per_example": int(prov_cfg.get("texts_per_example", 2 if effective_nneg == 0 else effective_nneg + 2)),
                "max_texts_per_physical_batch": int(bs) * int(prov_cfg.get("texts_per_example", 2 if effective_nneg == 0 else effective_nneg + 2)),
                "n_negatives_per_example": int(effective_nneg),
            }
            write_json(run_config, run_dir / "run_config.json")

        # Train
        warmup_steps = int(len(train_dataloader) * int(ep) * float(args.warmup_ratio))
        logged_train_loss = LossLoggingWrapper(
            train_loss,
            run_dir=run_dir,
            configured_lr=float(lr),
            steps_per_epoch=steps_per_epoch,
        )
        try:
            fit_kwargs: Dict[str, Any] = dict(
                train_objectives=[(train_dataloader, logged_train_loss)],
                evaluator=evaluator,
                epochs=int(ep),
                steps_per_epoch=steps_per_epoch,
                evaluation_steps=max(int(len(train_dataloader)), 1),
                warmup_steps=max(int(warmup_steps), 0),
                output_path=str(model_out_dir),
                save_best_model=True,
                optimizer_params={"lr": float(lr)},
                show_progress_bar=True,
                use_amp=False,          # EXPLICIT: canonical Phase 2 is fp32, no autocast
            )

            model.fit(**fit_kwargs)
        except Exception as e:
            write_json({"error": {"type": type(e).__name__, "message": str(e)}}, run_dir / "_train_error.json")
            raise
        finally:
            logged_train_loss.close()

        # Diagnostics
        write_json(
            {
                "batch_collision_history": [s.__dict__ for s in sampler.history],
                "negative_fallback_stats": dict(getattr(train_dataset, "fallback_stats", {})),
                "batching": {
                    "physical_batch_size_examples": int(bs),
                    "texts_per_example": 2 if int(effective_nneg) == 0 else int(effective_nneg) + 2,
                    "max_texts_per_physical_batch": int(bs) * (2 if int(effective_nneg) == 0 else int(effective_nneg) + 2),
                    "gradient_accumulation_steps": 1,
                    "note": "strategy 'none' encodes 2 texts/example (<=22/batch); hard-negative "
                            "strategies encode 12 texts/example (<=132/batch)",
                },
                "val_history": val_evaluator.history,
                "loss_history": {
                    "jsonl": str(loss_history_jsonl),
                    "csv": str(loss_history_csv),
                    "n_steps_logged": int(getattr(logged_train_loss, "global_step", 0)),
                },
            },
            run_dir / "training_diagnostics.json",
        )

        # Evaluate best checkpoint
        ft1._evaluate_and_write_run(
            run_dir=run_dir,
            model_name_or_path=str(model_out_dir),
            device=device,
            cde_master_enriched=Path(args.cde_master_enriched),
            splits_dir=Path(args.splits_dir),
            recipe=str(args.recipe),
            cde_format=str(args.cde_format),
            query_variant=str(args.query_variant),
            rerank_mode=str(args.rerank_mode),
            sep=str(args.sep),
            recipe_configs=recipe_configs,
            top_k=int(args.top_k),
            k_values=[int(x) for x in _parse_csv_ints(args.k_values or "5,10,20")],
            output_top_k=int(args.output_top_k),
            block_size=int(args.block_size),
            batch_size=int(args.encode_batch_size),
            normalize_embeddings=bool(args.normalize_embeddings),
            hybrid_alpha=float(args.hybrid_alpha),
            only_splits=only_splits,
            artifacts_dir=artifacts_dir,
        )

        # Collect leaderboard metrics.
        try:
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            _mbs = metrics.get("metrics_by_split") or {}
            # Prefer the resolved validation split (canonical `val_dev`), fall back to legacy `val`.
            valm = _mbs.get(val_split_name) or _mbs.get("val_dev") or _mbs.get("val") or {}
            val_mrr100 = float(valm.get("mrr@100", float("nan")))
            val_rec5 = float(valm.get("recall@5", float("nan")))
        except Exception:
            val_mrr100 = float("nan")
            val_rec5 = float("nan")

        leaderboard_rows.append(
            RunRow(
                run_id=str(run_id),
                base_model_id=str(getattr(args, "base_model_id", None) or ""),
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
                strategy=str(strat),
                nneg=int(effective_nneg),
                val_mrr100=float(val_mrr100),
                val_recall5=float(val_rec5),
            )
        )
        _write_leaderboard(leaderboard_rows, reports_dir / "leaderboard.csv")


if __name__ == "__main__":  # pragma: no cover
    main()
