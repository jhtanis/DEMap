"""Canonical cache + compute-fallback for **catalog** (CDE) bi-encoder embeddings.

The deep bi-encoder retrieval stage (``scripts/retrieve_biencoder_deep.py``) must encode
the whole CDE catalog (~63k CDEs) before it can retrieve top-K candidates. On CPU that
encode takes ~2 hours; on a GPU it is ~1-2 minutes. Those catalog vectors depend only on
(model checkpoint, catalog text, normalization) — **not** on the query set — so they can be
computed once and reused across every eval split and every re-run.

This module provides:

* a **documented canonical cache location** for those vectors, keyed by model + catalog
  text so different models / catalogs never collide;
* a **load-if-present** path that validates and returns cached vectors, skipping encode;
* a **compute-fallback ladder** used only on a cache miss, in this order:

    1. a small **Slurm GPU** job for the catalog-encode stage (preferred),
    2. a **local GPU** on the current node,
    3. a **local CPU** encode (slow ~2h) — emitted with a prominent warning.

* a small **manifest** written next to the vectors recording model path, catalog snapshot,
  text recipe, normalization, timestamp, and whether they were loaded or computed.

Correctness: the saved vectors are the exact float32, L2-normalized embeddings the encode
would produce, so retrieval output is identical whether loaded from cache or recomputed.
The cache-key + text-sampling helpers are imported from ``baseline_grid`` so the recipe
key matches the repo's established embedding-cache convention.

Import-time safety: this module does **not** import ``torch`` / ``sentence_transformers``
at import time (only lazily, inside the compute path), so it is cheap to unit-test.
"""
from __future__ import annotations

import json
import os
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse the *exact* canonical helpers so recipe keys + text sampling match the rest of the
# repo's embedding cache (baseline_grid). These are numpy/pandas only — no torch.
from demap_repro.biencoder.engine.baseline_grid import (  # noqa: E402
    _catalog_cache_key,
    _sha1_sample,
    _slug,
)

# Repo root = .../demap (this file is src/demap/retrieval/catalog_embedding_cache.py)
REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Canonical cache location (documented; repo-relative; NEVER /tmp, NEVER artifacts_v3_cdisc)
# ---------------------------------------------------------------------------
DEFAULT_CACHE_DIR = REPO_ROOT / "artifacts/final_reranker/biencoder_deep/catalog_embeddings"

# Paths we refuse to use for the cache (Biowulf shared-/tmp rule + frozen legacy tree).
_FORBIDDEN_SUBSTRINGS = ("/tmp/", "/tmp\\", "artifacts_v3_cdisc")
_FORBIDDEN_EXACT = ("/tmp",)

CPU_FALLBACK_WARNING = (
    "\n"
    "################################################################################\n"
    "# SLOW FALLBACK: encoding the CDE catalog on CPU.                               #\n"
    "# No cached embeddings were found and no GPU (Slurm or local) is available, so  #\n"
    "# the ~63k-CDE catalog is being encoded on CPU. THIS TAKES AROUND 2 HOURS.      #\n"
    "# It will be cached afterwards so subsequent runs are instant. To avoid this,   #\n"
    "# run the catalog-encode on a GPU (Slurm or local) or drop a verified embedding #\n"
    "# into the canonical cache directory.                                           #\n"
    "################################################################################\n"
)


def assert_safe_cache_dir(path: os.PathLike | str) -> None:
    """Raise ``ValueError`` if ``path`` is under /tmp or the frozen artifacts_v3_cdisc tree."""
    p = str(Path(path))
    norm = p.replace("\\", "/")
    if p in _FORBIDDEN_EXACT or norm.rstrip("/") == "/tmp":
        raise ValueError(f"cache dir must not be /tmp (Biowulf shared-scratch rule): {p}")
    if norm.startswith("/tmp/"):
        raise ValueError(f"cache dir must not live under /tmp (Biowulf shared-scratch rule): {p}")
    if "artifacts_v3_cdisc" in norm.split("/"):
        raise ValueError(f"cache dir must not touch artifacts_v3_cdisc (frozen legacy tree): {p}")


# ---------------------------------------------------------------------------
# Cache-key / path helpers
# ---------------------------------------------------------------------------
def model_slug_for(model_dir: os.PathLike | str) -> str:
    """Stable, filesystem-safe slug that uniquely namespaces a model checkpoint.

    Namespacing the cache by model means two different checkpoints can never read each
    other's vectors (the same guarantee ``baseline_grid`` relies on). Based on the resolved
    model directory path so it is stable across runs.
    """
    import hashlib

    resolved = str(Path(model_dir).resolve())
    return "model__" + hashlib.sha1(resolved.encode("utf-8", errors="ignore")).hexdigest()[:12]


def compute_recipe_key(
    recipe: str,
    cde_format: str,
    sep: str,
    recipe_configs: Optional[Dict[str, Dict]],
) -> str:
    """Recipe key identical to the repo's established catalog-embedding cache scheme.

    Mirrors the settings extraction in ``baseline_grid._load_or_build_catalog_embeddings``
    so the ``recipe_key`` (and therefore the cache filename) is consistent.
    """
    v1_filter_numeric_only = True
    v1_filter_versioned_id_short_name = False
    placeholder_policy = "omit"
    short_name_placeholder = "<MISSING_SHORT_NAME>"
    pv_placeholder = "<MISSING_PV_SUMMARY>"

    if recipe_configs and isinstance(recipe_configs.get("v1"), dict):
        v1 = recipe_configs["v1"]
        v1_filter_numeric_only = bool(v1.get("filter_numeric_only", True))
        v1_filter_versioned_id_short_name = bool(v1.get("filter_versioned_id_short_name", False))
    if recipe_configs and isinstance(recipe_configs, dict):
        if "placeholder_policy" in recipe_configs:
            placeholder_policy = str(recipe_configs.get("placeholder_policy") or "omit")
        if "short_name_placeholder" in recipe_configs:
            short_name_placeholder = str(recipe_configs.get("short_name_placeholder") or short_name_placeholder)
        if "pv_placeholder" in recipe_configs:
            pv_placeholder = str(recipe_configs.get("pv_placeholder") or pv_placeholder)

    return _catalog_cache_key(
        recipe=recipe,
        cde_format=cde_format,
        sep=sep,
        v1_filter_numeric_only=v1_filter_numeric_only,
        v1_filter_versioned_id_short_name=v1_filter_versioned_id_short_name,
        placeholder_policy=placeholder_policy,
        short_name_placeholder=short_name_placeholder,
        pv_placeholder=pv_placeholder,
    )


@dataclass(frozen=True)
class CachePaths:
    emb: Path
    ids: Path
    meta: Path

    def all_exist(self) -> bool:
        return self.emb.exists() and self.ids.exists() and self.meta.exists()


def cache_paths(
    cache_dir: os.PathLike | str,
    model_slug: str,
    recipe: str,
    cde_format: str,
    recipe_key: str,
) -> CachePaths:
    """Canonical triple of (emb.npy, ids.npy, meta.json) paths for this model+catalog."""
    base = Path(cache_dir) / model_slug / "catalog"
    stem = f"cde_catalog__{recipe}__{cde_format}__{recipe_key}"
    return CachePaths(
        emb=base / f"{stem}__emb.npy",
        ids=base / f"{stem}__ids.npy",
        meta=base / f"{stem}__meta.json",
    )


# ---------------------------------------------------------------------------
# Load-if-present
# ---------------------------------------------------------------------------
def load_if_present(
    paths: CachePaths,
    *,
    expected_sample_sha1: str,
    n_texts: int,
    model_max_seq_length: Optional[int],
    normalize: bool,
    model_dir: os.PathLike | str,
    log: Callable[[str], None] = print,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, object]]]:
    """Return ``(cde_ids, cde_emb, meta)`` if a *valid* cache is present, else ``None``.

    Validation guards against silently reusing stale/wrong vectors: the cached catalog text
    sample SHA1, CDE count, model max-seq-length, normalization flag, and model path must all
    match the current request.
    """
    if not paths.all_exist():
        return None
    try:
        meta = json.loads(paths.meta.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover - corrupt meta is a miss
        log(f"[cache] present but unreadable meta ({e}); will recompute")
        return None

    checks = {
        "catalog_text_sample_sha1": (meta.get("catalog_text_sample_sha1"), expected_sample_sha1),
        "n_cdes": (meta.get("n_cdes"), int(n_texts)),
        "model_max_seq_length": (meta.get("model_max_seq_length"), model_max_seq_length),
        "normalize": (meta.get("normalize"), bool(normalize)),
        "model_dir": (str(meta.get("model_dir", "")), str(Path(model_dir).resolve())),
    }
    mismatches = [k for k, (got, want) in checks.items() if got != want]
    if mismatches:
        log(f"[cache] present but STALE (mismatch: {mismatches}); will recompute")
        return None

    ids = np.load(paths.ids, allow_pickle=False)
    emb = np.load(paths.emb, allow_pickle=False)
    if emb.shape[0] != ids.shape[0] or emb.shape[0] != int(n_texts):
        log(f"[cache] present but shape mismatch emb={emb.shape} ids={ids.shape} n={n_texts}; recompute")
        return None
    log(f"[cache] HIT — loaded catalog embeddings from cache, skipping encode:")
    log(f"[cache]   emb : {paths.emb}")
    log(f"[cache]   ids : {paths.ids}")
    log(f"[cache]   meta: {paths.meta}")
    log(f"[cache]   n_cdes={emb.shape[0]} dim={emb.shape[1]} normalize={normalize}")
    return ids, emb, meta


# ---------------------------------------------------------------------------
# Manifest / save
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(
    *,
    model_dir: os.PathLike | str,
    run_dir: os.PathLike | str,
    cde_master: os.PathLike | str,
    recipe: str,
    cde_format: str,
    sep: str,
    recipe_key: str,
    normalize: bool,
    n_cdes: int,
    catalog_text_sample_sha1: str,
    model_max_seq_length: Optional[int],
    source: str,
    compute_backend: Optional[str],
) -> Dict[str, object]:
    """Assemble the traceability manifest saved next to the vectors."""
    return {
        "model_dir": str(Path(model_dir).resolve()),
        "run_dir": str(Path(run_dir).resolve()),
        "cde_master": str(Path(cde_master).resolve()),
        "recipe": recipe,
        "cde_format": cde_format,
        "sep": sep,
        "recipe_key": recipe_key,
        "normalize": bool(normalize),
        "n_cdes": int(n_cdes),
        "catalog_text_sample_sha1": catalog_text_sample_sha1,
        "model_max_seq_length": model_max_seq_length,
        "source": source,                 # "computed" (writer) — readers log "loaded"
        "compute_backend": compute_backend,  # "slurm_gpu" | "local_gpu" | "local_cpu" | None
        "timestamp": _now_iso(),
    }


def save_embeddings(
    paths: CachePaths,
    cde_ids: np.ndarray,
    cde_emb: np.ndarray,
    meta: Dict[str, object],
    *,
    log: Callable[[str], None] = print,
) -> None:
    paths.emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(paths.emb, cde_emb)
    np.save(paths.ids, np.asarray(cde_ids))
    paths.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"[cache] wrote catalog embeddings -> {paths.emb}")
    log(f"[cache]   ids : {paths.ids}")
    log(f"[cache]   meta: {paths.meta}")


# ---------------------------------------------------------------------------
# Compute-backend selection (pure; unit-tested)
# ---------------------------------------------------------------------------
def slurm_available() -> bool:
    """True when an sbatch client is on PATH (i.e. we can submit a Slurm job)."""
    return shutil.which("sbatch") is not None


def local_gpu_available() -> bool:
    """True when a CUDA GPU is visible on the current node (lazy torch import)."""
    try:
        import torch  # noqa: WPS433 (lazy import by design)
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_encode_backend(
    *,
    allow_slurm: bool,
    slurm_ok: bool,
    gpu_ok: bool,
) -> str:
    """Pure fallback-order policy → ``"slurm_gpu"`` | ``"local_gpu"`` | ``"local_cpu"``.

    Order: (1) Slurm GPU when allowed & available, (2) local GPU, (3) local CPU.
    """
    if allow_slurm and slurm_ok:
        return "slurm_gpu"
    if gpu_ok:
        return "local_gpu"
    return "local_cpu"


@dataclass
class EncodeOutcome:
    ids: np.ndarray
    emb: np.ndarray
    source: str           # "loaded" | "computed"
    backend: Optional[str]  # None when loaded; else the compute backend
    paths: CachePaths


def ensure_catalog_embeddings(
    *,
    cde_ids: Sequence[str],
    cde_texts: Sequence[str],
    model_dir: os.PathLike | str,
    run_dir: os.PathLike | str,
    cde_master: os.PathLike | str,
    recipe: str,
    cde_format: str,
    sep: str,
    recipe_configs: Optional[Dict[str, Dict]],
    normalize: bool,
    cache_dir: os.PathLike | str = DEFAULT_CACHE_DIR,
    allow_slurm: bool = True,
    encode_fn: Optional[Callable[[], Tuple[np.ndarray, Optional[int], str]]] = None,
    slurm_submit_fn: Optional[Callable[[CachePaths], Tuple[np.ndarray, Optional[int]]]] = None,
    slurm_ok_fn: Callable[[], bool] = slurm_available,
    gpu_ok_fn: Callable[[], bool] = local_gpu_available,
    warn_fn: Callable[[str], None] = lambda m: warnings.warn(m, stacklevel=2),
    log: Callable[[str], None] = print,
) -> EncodeOutcome:
    """Load catalog embeddings from the canonical cache, or compute them via the fallback ladder.

    ``encode_fn`` (injected for testability) performs a *local* encode and returns
    ``(emb, model_max_seq_length, backend)`` where backend is ``"local_gpu"`` or
    ``"local_cpu"``. ``slurm_submit_fn`` submits + waits for a Slurm GPU encode and returns
    ``(emb, model_max_seq_length)``; it is responsible only for producing the array (this
    function writes the cache + manifest centrally, so all paths stay identical).
    """
    assert_safe_cache_dir(cache_dir)
    cde_ids = np.asarray([str(x) for x in cde_ids])
    cde_texts = [str(t) for t in cde_texts]
    n = len(cde_texts)
    sample_sha1 = _sha1_sample(cde_texts)
    recipe_key = compute_recipe_key(recipe, cde_format, sep, recipe_configs)
    slug = model_slug_for(model_dir)
    paths = cache_paths(cache_dir, slug, recipe, cde_format, recipe_key)

    # NOTE model_max_seq_length is only known after a real model load; for the cache-hit
    # validation we accept whatever the cached meta recorded as long as the *text* matches.
    # We therefore validate on text/count/normalize/model_dir and trust the recorded seq len.
    hit = load_if_present(
        paths,
        expected_sample_sha1=sample_sha1,
        n_texts=n,
        model_max_seq_length=None if not paths.meta.exists() else _peek_seq_len(paths.meta),
        normalize=normalize,
        model_dir=model_dir,
        log=log,
    )
    if hit is not None:
        ids, emb, _meta = hit
        return EncodeOutcome(ids=ids, emb=emb, source="loaded", backend=None, paths=paths)

    backend = resolve_encode_backend(
        allow_slurm=allow_slurm, slurm_ok=slurm_ok_fn(), gpu_ok=gpu_ok_fn()
    )
    log(f"[cache] MISS — computing catalog embeddings via backend='{backend}' "
        f"(cache dir: {cache_dir})")

    model_max_seq_length: Optional[int]
    if backend == "slurm_gpu":
        if slurm_submit_fn is None:
            raise RuntimeError("slurm backend selected but no slurm_submit_fn provided")
        emb, model_max_seq_length = slurm_submit_fn(paths)
        # A Slurm encode job typically writes the cache itself; if so, just load it back.
        if paths.all_exist():
            loaded = load_if_present(
                paths, expected_sample_sha1=sample_sha1, n_texts=n,
                model_max_seq_length=model_max_seq_length, normalize=normalize,
                model_dir=model_dir, log=log,
            )
            if loaded is not None:
                ids, emb2, _m = loaded
                return EncodeOutcome(ids=ids, emb=emb2, source="loaded", backend="slurm_gpu", paths=paths)
    elif backend == "local_gpu":
        if encode_fn is None:
            raise RuntimeError("local_gpu backend selected but no encode_fn provided")
        emb, model_max_seq_length, _b = encode_fn()
    else:  # local_cpu
        warn_fn(CPU_FALLBACK_WARNING)
        log(CPU_FALLBACK_WARNING)
        if encode_fn is None:
            raise RuntimeError("local_cpu backend selected but no encode_fn provided")
        emb, model_max_seq_length, _b = encode_fn()

    emb = np.asarray(emb, dtype=np.float32)
    meta = build_manifest(
        model_dir=model_dir, run_dir=run_dir, cde_master=cde_master, recipe=recipe,
        cde_format=cde_format, sep=sep, recipe_key=recipe_key, normalize=normalize,
        n_cdes=n, catalog_text_sample_sha1=sample_sha1,
        model_max_seq_length=model_max_seq_length, source="computed", compute_backend=backend,
    )
    save_embeddings(paths, cde_ids, emb, meta, log=log)
    return EncodeOutcome(ids=cde_ids, emb=emb, source="computed", backend=backend, paths=paths)


def _peek_seq_len(meta_path: Path) -> Optional[int]:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")).get("model_max_seq_length")
    except Exception:
        return None
