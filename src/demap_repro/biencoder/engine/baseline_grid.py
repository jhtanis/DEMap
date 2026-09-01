#!/usr/bin/env python3
"""baseline_grid.py

Run a factorial grid of baseline retrieval experiments for *semantic matching*
of "in-the-wild" query strings to caDSR CDEs.

Experiment axes
---------------
Target-side CDE text recipes (v1-v6)
  v1  SHORT_NAME (PREFERREDNAME in the XML)    [optionally filtered to drop numeric-only names]
  v2  LONG_NAME | DEFINITION
  v3  PREFERRED_QUESTION_TEXT
  v4  VALUE_DOMAIN_TYPE | VALUE_DOMAIN_DATATYPE
  v5  PV_SUMMARY (<= N PV meaning long names, sorted by display order)
  v6  DEC_LONG_NAME

Target formatting (`cde_format`)
  labeled : include field labels (e.g., "LONG_NAME: ...")
  raw     : values only

Query variants
  Q1  query_text_raw
  Q2  query_text

Re-ranking modes
  R0  none (embedding-only)
  R1  pure TF-IDF rerank (applied within top-K embedding candidates)
  R2  hybrid rerank = alpha * embedding + (1-alpha) * TF-IDF (within top-K; alpha defaults to 0.5)

Outputs
-------
Each run writes to a run directory:
  <runs_dir>/<run_id>/{

By default, runs_dir is:
  artifacts/runs

If you set runs_dir="auto", runs are written under:
  - artifacts/off_the_shelf/<model_slug>/runs               (default)
  - artifacts/<stage_tag>/<model_slug>/runs                 (when stage_tag is set)

    run_config.json,
    metrics.json,
    rankings.parquet,
    failures_sample.csv,
    reliability.csv,
    ece.json,
    by_family.csv,
  }

Embedding caches are stored under:
  artifacts/embeddings/<model_slug>/...

Notes
-----
- We score queries in blocks to avoid materializing the full (n_queries × n_cdes)
  similarity matrix.
- TF-IDF rerank is always restricted to the top-K embedding candidates.
"""

import argparse
import hashlib
import json
import os
import random
import pickle
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from demap_repro.text.recipes import RECIPE_FIELDS, build_catalog, is_valid_recipe, required_fields_for_recipe
from demap_repro.utils.config import deep_merge, load_config


# -----------------------------
# Small helpers
# -----------------------------

def _slug(s: str) -> str:
    return str(s).replace("/", "__").replace(" ", "_").replace(":", "_")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _sha1_sample(texts: Sequence[str], n: int = 1000) -> str:
    take = list(texts[: min(len(texts), n)])
    h = hashlib.sha1()
    for t in take:
        h.update(str(t).encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _nonnull_rate(s: pd.Series) -> float:
    if s is None:
        return float("nan")
    return float(s.notna().mean())


def _empty_rate(s: pd.Series) -> float:
    if s is None:
        return float("nan")
    ss = s.fillna("").astype(str).str.strip()
    return float((ss == "").mean())


def _recipe_fields(recipe: str) -> List[str]:
    """Return the list of master-table fields used by a recipe (atomic or composite)."""
    try:
        return required_fields_for_recipe(recipe)
    except Exception:
        return []


# -----------------------------
# Run specification
# -----------------------------

QUERY_VARIANT_TO_COL = {
    # Query text variants emitted by `demap build-queries`.
    #
    # Q1: `query_text_raw` (no prefixes, no PV)
    # Q2: `query_text` (prefix-labeled; may include PV + provenance labels)
    # Q3: `query_text_q3` (label-free; may include PV without provenance labels)
    # Q4: `query_text_q4` (label-free; PV slot always present via <MISSING_PV_SUMMARY> placeholder)
    "Q1": "query_text_raw",
    "Q2": "query_text",
    "Q3": "query_text_q3",
    "Q4": "query_text_q4",
    # backwards-compatible aliases
    "query_text_raw": "query_text_raw",
    "query_text": "query_text",
    "query_text_q3": "query_text_q3",
    "query_text_q4": "query_text_q4",
}

RERANK_MODE_ALIASES = {
    "R0": "none",
    "none": "none",
    "R1": "tfidf",
    "tfidf": "tfidf",
    "R2": "hybrid",
    "hybrid": "hybrid",
}

CDE_FORMAT_ALIASES = {
    "labeled": "labeled",
    "raw": "raw",
    # common synonyms
    "cde_text": "labeled",
    "raw_cde_text": "raw",
}


@dataclass(frozen=True)
class RunSpec:
    recipe: str
    cde_format: str
    query_variant: str
    rerank_mode: str
    alpha: float = 0.5

    @property
    def id(self) -> str:
        return f"{self.recipe}__{self.cde_format}__{self.query_variant}__{self.rerank_mode}"


# Deterministic rerank tags for run directory naming (paper workflow).
RERANK_MODE_TO_TAG = {
    "none": "R0",
    "tfidf": "R1",
    "hybrid": "R2",
}


def rerank_tag(mode: str) -> str:
    """Return the paper-style tag (R0/R1/R2) for a rerank mode or alias."""
    raw = str(mode).strip()
    m = raw.lower()
    canonical = RERANK_MODE_ALIASES.get(m, m)
    return RERANK_MODE_TO_TAG.get(canonical, raw)


def normalize_stage_tag(stage_tag: Optional[str]) -> str:
    """Return a filesystem-safe stage tag, or an empty string when unset."""
    t = str(stage_tag or "").strip()
    if not t:
        return ""
    return _slug(t).strip("_")


def resolve_runs_root(
    *,
    artifacts_dir: Path,
    runs_dir_cfg: str | Path,
    model_name: str,
    stage_tag: Optional[str] = None,
) -> Path:
    """Resolve the per-model root that contains run directories."""
    if str(runs_dir_cfg).lower() != "auto":
        return Path(runs_dir_cfg)
    stage_root = normalize_stage_tag(stage_tag) or "off_the_shelf"
    return Path(artifacts_dir) / stage_root / _slug(str(model_name)) / "runs"


def make_run_id(
    *,
    model_name: str,
    recipe: str,
    cde_format: str,
    query_variant: str,
    rerank_mode: str,
    seed: int,
    stage_tag: Optional[str] = None,
) -> str:
    """Deterministic run_id used for paper-grade checks + aggregation.

    Format
    ------
      __ots__<stage_tag>__<model_slug>__<recipe>__<format>__<query>__<rerank>__seed{N}

    If stage_tag is unset, the legacy format is preserved without the extra
    stage component.

    Notes
    -----
    - Includes the __ots__ marker to distinguish off-the-shelf runs.
    - Includes stage_tag when provided so paper stages cannot collide even when
      they reuse the same model/query/recipe/seed grid.
    - Intentionally *does not* include a timestamp so downstream scripts can
      compute expected run directories deterministically.
    """

    model_slug = _slug(model_name)
    rr = rerank_tag(rerank_mode)
    stage_slug = normalize_stage_tag(stage_tag)
    if stage_slug:
        return f"__ots__{stage_slug}__{model_slug}__{recipe}__{cde_format}__{query_variant}__{rr}__seed{int(seed)}"
    return f"__ots__{model_slug}__{recipe}__{cde_format}__{query_variant}__{rr}__seed{int(seed)}"


def _load_splits(splits_dir: Path) -> Dict[str, Path]:
    """Return a mapping {split_name -> parquet_path}.

    Historically, demap evaluated a fixed set of split filenames. For the paper
    workflow (and for externally curated test sets like the GDC batches), it is
    useful to allow additional split parquets to live alongside the defaults.

    We therefore:
      1) load a preferred, stable list of known split filenames (if present)
      2) then include any other *.parquet files found in splits_dir
    """

    preferred = [
        "train.parquet",
        "val.parquet",
        "test.parquet",
        "external_holdout_org.parquet",
        "external_holdout_standard.parquet",
        "external_holdout_refslice.parquet",
        # Optional externally curated splits
        "external_holdout_gdc_altnames.parquet",
        "external_holdout_gdc_questiontext.parquet",
    ]

    out: Dict[str, Path] = {}
    seen = set()

    for fname in preferred:
        p = splits_dir / fname
        if p.exists():
            out[p.stem] = p
            seen.add(p.name)

    # Include any additional split parquets (sorted for determinism).
    for p in sorted([x for x in splits_dir.glob("*.parquet") if x.is_file() and x.name not in seen]):
        out[p.stem] = p

    if not out:
        raise FileNotFoundError(f"No split parquet files found in: {splits_dir}")
    return out


# -----------------------------
# Embeddings + cache
# -----------------------------

def _load_sentence_transformer(model_name: str):
    from demap_repro.biencoder.engine.st_loader import load_sentence_transformer

    return load_sentence_transformer(model_name)


def _encode_texts(model, texts: Sequence[str], batch_size: int, normalize: bool) -> np.ndarray:
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    if normalize:
        # Some versions don't normalize perfectly
        emb = _l2_normalize(emb)
    return emb


def _load_or_build_query_embeddings(
    model,
    model_slug: str,
    embeddings_dir: Path,
    split_name: str,
    query_variant: str,
    query_texts: Sequence[str],
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    cache_dir = embeddings_dir / model_slug / "queries"
    _ensure_dir(cache_dir)
    emb_path = cache_dir / f"{split_name}__{query_variant}__emb.npy"
    meta_path = cache_dir / f"{split_name}__{query_variant}__meta.json"

    texts = [str(t) for t in query_texts]
    sample_sha1 = _sha1_sample(texts)
    model_max_seq_length = getattr(model, "max_seq_length", None)

    if emb_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("query_text_sample_sha1") == sample_sha1
                and meta.get("n_queries") == int(len(texts))
                and meta.get("model_max_seq_length") == model_max_seq_length
                and meta.get("normalize") == bool(normalize)
            ):
                return np.load(emb_path, allow_pickle=False)
        except Exception:
            pass

    emb = _encode_texts(model, texts, batch_size=batch_size, normalize=normalize)
    np.save(emb_path, emb)
    meta_path.write_text(
        json.dumps(
            {
                "n_queries": int(len(texts)),
                "query_text_sample_sha1": sample_sha1,
                "query_variant": query_variant,
                "model_max_seq_length": model_max_seq_length,
                "normalize": bool(normalize),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return emb


def _catalog_cache_key(
    recipe: str,
    cde_format: str,
    sep: str,
    v1_filter_numeric_only: bool,
    v1_filter_versioned_id_short_name: bool,
    placeholder_policy: str,
    short_name_placeholder: str,
    pv_placeholder: str,
) -> str:
    """Stable cache key for catalog embeddings.

    IMPORTANT: This key must include *all* settings that affect the rendered
    catalog text, otherwise we can accidentally reuse stale embeddings across
    semantically different catalogs.

    In particular, the paper workflow toggles CDE-side placeholder insertion for
    SHORT_NAME and PV_SUMMARY, so those settings must be included.
    """

    h = hashlib.sha1()

    def _u(x: object) -> None:
        h.update(str(x).encode("utf-8", errors="ignore"))
        h.update(b"|")

    _u(recipe)
    _u(cde_format)
    _u(sep)
    _u(bool(v1_filter_numeric_only))
    _u(bool(v1_filter_versioned_id_short_name))
    _u(str(placeholder_policy).strip().lower())
    _u(str(short_name_placeholder))
    _u(str(pv_placeholder))

    return h.hexdigest()[:12]


def _load_or_build_catalog_embeddings(
    model,
    model_slug: str,
    embeddings_dir: Path,
    master: pd.DataFrame,
    recipe: str,
    cde_format: str,
    sep: str,
    recipe_configs: Optional[Dict[str, Dict]] ,
    batch_size: int,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Returns (cde_ids, cde_emb, meta)."""    # Recipe configs that affect catalog text (must be part of the cache key).
    v1_filter_numeric_only = True
    v1_filter_versioned_id_short_name = False

    # Global placeholder policy (applies to SHORT_NAME + PV_SUMMARY).
    placeholder_policy = "omit"
    short_name_placeholder = "<MISSING_SHORT_NAME>"
    pv_placeholder = "<MISSING_PV_SUMMARY>"

    if recipe_configs and isinstance(recipe_configs.get("v1"), dict):
        v1 = recipe_configs["v1"]
        v1_filter_numeric_only = bool(v1.get("filter_numeric_only", True))
        v1_filter_versioned_id_short_name = bool(v1.get("filter_versioned_id_short_name", False))

    # Keep this permissive: ignore unknown keys.
    if recipe_configs and isinstance(recipe_configs, dict):
        if "placeholder_policy" in recipe_configs:
            placeholder_policy = str(recipe_configs.get("placeholder_policy") or "omit")
        if "short_name_placeholder" in recipe_configs:
            short_name_placeholder = str(recipe_configs.get("short_name_placeholder") or short_name_placeholder)
        if "pv_placeholder" in recipe_configs:
            pv_placeholder = str(recipe_configs.get("pv_placeholder") or pv_placeholder)

    recipe_key = _catalog_cache_key(
        recipe=recipe,
        cde_format=cde_format,
        sep=sep,
        v1_filter_numeric_only=v1_filter_numeric_only,
        v1_filter_versioned_id_short_name=v1_filter_versioned_id_short_name,
        placeholder_policy=placeholder_policy,
        short_name_placeholder=short_name_placeholder,
        pv_placeholder=pv_placeholder,
    )

    cache_dir = embeddings_dir / model_slug / "catalog"
    _ensure_dir(cache_dir)

    emb_path = cache_dir / f"cde_catalog__{recipe}__{cde_format}__{recipe_key}__emb.npy"
    ids_path = cache_dir / f"cde_catalog__{recipe}__{cde_format}__{recipe_key}__ids.npy"
    meta_path = cache_dir / f"cde_catalog__{recipe}__{cde_format}__{recipe_key}__meta.json"

    # Build catalog text (this also drops empty rows)
    catalog = build_catalog(
        master,
        recipe=recipe,
        cde_format=cde_format,
        sep=sep,
        recipe_configs=recipe_configs,
    )
    cde_ids = catalog["cde_id"].astype(str).values
    cde_texts = catalog["cde_text"].astype(str).tolist()

    sample_sha1 = _sha1_sample(cde_texts)
    model_max_seq_length = getattr(model, "max_seq_length", None)

    if emb_path.exists() and ids_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("catalog_text_sample_sha1") == sample_sha1
                and meta.get("n_cdes") == int(len(cde_texts))
                and meta.get("model_max_seq_length") == model_max_seq_length
                and meta.get("normalize") == bool(normalize)
            ):
                return (
                    np.load(ids_path, allow_pickle=False),
                    np.load(emb_path, allow_pickle=False),
                    meta,
                )
        except Exception:
            pass

    cde_emb = _encode_texts(model, cde_texts, batch_size=batch_size, normalize=normalize)

    np.save(emb_path, cde_emb)
    np.save(ids_path, cde_ids)

    meta = {
        "recipe": recipe,
        "cde_format": cde_format,
        "sep": sep,
        "v1_filter_numeric_only": v1_filter_numeric_only,
        "v1_filter_versioned_id_short_name": v1_filter_versioned_id_short_name,
        "placeholder_policy": str(placeholder_policy),
        "short_name_placeholder": str(short_name_placeholder),
        "pv_placeholder": str(pv_placeholder),
        "n_cdes": int(len(cde_texts)),
        "catalog_text_sample_sha1": sample_sha1,
        "recipe_key": recipe_key,
        "model_max_seq_length": model_max_seq_length,
        "normalize": bool(normalize),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return cde_ids, cde_emb, meta


# -----------------------------
# Retrieval + rerank
# -----------------------------

def _topk_blockwise(
    query_emb: np.ndarray,
    cde_emb: np.ndarray,
    top_k: int,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top-k indices + scores for each query (cosine/dot-product).

    Assumes embeddings are already L2-normalized if cosine similarity is desired.
    """

    n_q = query_emb.shape[0]
    n_c = cde_emb.shape[0]

    # For each query, keep current best top_k scores/indices
    best_scores = np.full((n_q, top_k), -np.inf, dtype=np.float32)
    best_idx = np.full((n_q, top_k), -1, dtype=np.int32)

    for start in range(0, n_c, block_size):
        end = min(start + block_size, n_c)
        block = cde_emb[start:end]
        scores = query_emb @ block.T  # (n_q, block)

        # Merge current best with this block
        merged_scores = np.concatenate([best_scores, scores], axis=1)
        merged_idx = np.concatenate(
            [best_idx, np.tile(np.arange(start, end, dtype=np.int32), (n_q, 1))], axis=1
        )

        # Select top_k per row
        part = np.argpartition(-merged_scores, kth=top_k - 1, axis=1)[:, :top_k]
        row = np.arange(n_q)[:, None]
        best_scores = merged_scores[row, part]
        best_idx = merged_idx[row, part]

        # Sort descending within top_k
        order = np.argsort(-best_scores, axis=1)
        best_scores = np.take_along_axis(best_scores, order, axis=1)
        best_idx = np.take_along_axis(best_idx, order, axis=1)

    return best_idx, best_scores


def _load_or_build_tfidf(
    cache_dir: Path,
    cache_key: str,
    cde_texts: Sequence[str],
):
    """Return (vectorizer, cde_tfidf_matrix).

    Cached by cache_key.
    """

    _ensure_dir(cache_dir)
    vec_path = cache_dir / f"tfidf__{cache_key}__vectorizer.pkl"
    mat_path = cache_dir / f"tfidf__{cache_key}__cde_matrix.pkl"

    if vec_path.exists() and mat_path.exists():
        try:
            with vec_path.open("rb") as f:
                vectorizer = pickle.load(f)
            with mat_path.open("rb") as f:
                cde_mat = pickle.load(f)
            return vectorizer, cde_mat
        except Exception:
            pass

    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

    vectorizer = TfidfVectorizer(norm="l2")
    cde_mat = vectorizer.fit_transform(list(cde_texts))

    with vec_path.open("wb") as f:
        pickle.dump(vectorizer, f)
    with mat_path.open("wb") as f:
        pickle.dump(cde_mat, f)

    return vectorizer, cde_mat


def _rerank_with_tfidf(
    query_texts: Sequence[str],
    cde_texts: Sequence[str],
    idx_top: np.ndarray,
    base_scores: np.ndarray,
    mode: str,
    alpha: float,
    tfidf_cache_dir: Path,
    tfidf_cache_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rerank within the candidate list using TF-IDF or hybrid TF-IDF + embedding.

    Returns (idx_reranked, scores_reranked).
    """

    if mode not in {"tfidf", "hybrid"}:
        return idx_top, base_scores

    vectorizer, cde_mat = _load_or_build_tfidf(
        cache_dir=tfidf_cache_dir,
        cache_key=tfidf_cache_key,
        cde_texts=cde_texts,
    )

    # Transform queries
    q_mat = vectorizer.transform(list(query_texts))

    n_q, k = idx_top.shape
    tfidf_scores = np.zeros((n_q, k), dtype=np.float32)

    # Compute TF-IDF cosine similarity within candidates
    for i in range(n_q):
        cand_idx = idx_top[i]
        # (1 x vocab) @ (vocab x k) => (1 x k)
        sim = (q_mat[i] @ cde_mat[cand_idx].T).toarray().ravel().astype(np.float32)
        tfidf_scores[i] = sim

    # Per-query normalize TF-IDF to [0,1] by max (avoid scale mismatch)
    tfidf_max = tfidf_scores.max(axis=1, keepdims=True)
    tfidf_norm = np.where(tfidf_max > 0, tfidf_scores / tfidf_max, tfidf_scores)

    if mode == "tfidf":
        new_scores = tfidf_norm
    else:
        # Embedding scores are cosine if normalized; map to [0,1] as well
        emb01 = np.clip((base_scores + 1.0) / 2.0, 0.0, 1.0)
        new_scores = alpha * emb01 + (1.0 - alpha) * tfidf_norm

    order = np.argsort(-new_scores, axis=1)
    idx_new = np.take_along_axis(idx_top, order, axis=1)
    scores_new = np.take_along_axis(new_scores, order, axis=1)

    return idx_new, scores_new


def _evaluate_split(
    split_name: str,
    df_split: pd.DataFrame,
    query_col: str,
    cde_ids: np.ndarray,
    cde_texts: Sequence[str],
    cde_emb: np.ndarray,
    query_emb: np.ndarray,
    top_k: int,
    k_values: Sequence[int],
    block_size: int,
    rerank_mode: str,
    alpha: float,
    tfidf_cache_dir: Path,
    tfidf_cache_key: str,
    output_top_k: int,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Return (metrics, rankings_df, failures_sample_df)."""

    # Map true ids to indices in catalog
    cde_id_to_idx = {cid: i for i, cid in enumerate(cde_ids.tolist())}
    true_ids = df_split["cde_id"].astype(str).tolist()
    true_idx = np.array([cde_id_to_idx.get(t, -1) for t in true_ids], dtype=np.int32)

    query_texts = df_split[query_col].fillna("").astype(str).tolist()

    idx_top, score_top = _topk_blockwise(query_emb=query_emb, cde_emb=cde_emb, top_k=top_k, block_size=block_size)

    # True-pair similarity in embedding space (cosine if embeddings are normalized).
    # This is useful for margin / geometric diagnostics even when the true CDE is
    # not present in the top-k predictions.
    true_score = np.full((len(true_idx),), np.nan, dtype=np.float32)
    m_true = true_idx >= 0
    if bool(m_true.any()):
        # Gather the true target embeddings (aligned to each query row).
        # NOTE: true_idx is int32; safe for indexing.
        c_true = cde_emb[true_idx[m_true]]
        q_true = query_emb[m_true]
        true_score[m_true] = np.sum(q_true * c_true, axis=1).astype(np.float32)

    if rerank_mode in {"tfidf", "hybrid"}:
        idx_top, score_top = _rerank_with_tfidf(
            query_texts=query_texts,
            cde_texts=cde_texts,
            idx_top=idx_top,
            base_scores=score_top,
            mode=rerank_mode,
            alpha=alpha,
            tfidf_cache_dir=tfidf_cache_dir,
            tfidf_cache_key=tfidf_cache_key,
        )

    # Compute rank of true target
    ranks = np.full((len(true_idx),), fill_value=np.inf, dtype=np.float32)
    for i, ti in enumerate(true_idx):
        if ti < 0:
            continue
        hits = np.where(idx_top[i] == ti)[0]
        if len(hits) > 0:
            ranks[i] = float(hits[0] + 1)  # 1-indexed

    # Metrics
    #
    # Evaluation plan (see docs/Evaluation Plan.docx):
    #   Primary: Recall@5
    #   Secondary: Top-1 accuracy, Recall@10, MRR@100
    metrics: Dict[str, float] = {
        "n": float(len(df_split)),
        "n_missing_target_in_catalog": float(int((true_idx < 0).sum())),
    }

    # Always compute headline retrieval metrics regardless of config.
    wanted_ks = {1, 5, 10, 100}
    wanted_ks.update(int(k) for k in (k_values or []))

    for k in sorted(wanted_ks):
        metrics[f"recall@{int(k)}"] = float((np.isfinite(ranks) & (ranks <= int(k))).mean())

    metrics["top1_accuracy"] = float((np.isfinite(ranks) & (ranks <= 1)).mean())

    rr100 = np.where(np.isfinite(ranks) & (ranks <= 100), 1.0 / ranks, 0.0)
    metrics["mrr@100"] = float(rr100.mean())

    # Keep a couple of diagnostics that are useful during iteration.
    rr_uncapped = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
    metrics["mrr"] = float(rr_uncapped.mean())
    metrics["median_rank"] = float(np.median(ranks[np.isfinite(ranks)]) if np.isfinite(ranks).any() else np.inf)

    # Rankings output
    out_k = int(min(output_top_k, top_k))
    top_ids = [[str(cde_ids[j]) for j in row[:out_k]] for row in idx_top]
    top_scores = [row[:out_k].astype(float).tolist() for row in score_top]

    top1_id = [ids[0] if ids else "" for ids in top_ids]
    top1_score = [sc[0] if sc else float("nan") for sc in top_scores]

    # Score diagnostics for confidence reporting.
    # We compute these from the *post-rerank* score list.
    s1 = [float(row[0]) if row is not None and len(row) > 0 else float("nan") for row in score_top]
    s2 = [float(row[1]) if row is not None and len(row) > 1 else float("nan") for row in score_top]
    s10 = [float(row[9]) if row is not None and len(row) > 9 else float("nan") for row in score_top]
    margin_1_2 = [a - b if np.isfinite(a) and np.isfinite(b) else float("nan") for a, b in zip(s1, s2)]
    margin_1_10 = [a - b if np.isfinite(a) and np.isfinite(b) else float("nan") for a, b in zip(s1, s10)]

    is_correct_top1 = [bool(np.isfinite(r) and r == 1.0) for r in ranks]

    rankings = pd.DataFrame(
        {
            "split": split_name,
            "pair_id": df_split.get("pair_id", pd.Series([None] * len(df_split))).astype(str),
            "query_id": df_split.get("query_id", pd.Series([None] * len(df_split))).astype(str),
            "query_source": df_split.get("query_source", pd.Series([None] * len(df_split))).astype(str),
            "family": df_split.get("family", pd.Series([None] * len(df_split))).astype(str),
            "query_text": query_texts,
            "true_cde_id": true_ids,
            "true_rank": ranks,
            "true_score": true_score,
            "top1_cde_id": top1_id,
            "top1_score": top1_score,
            "s1": s1,
            "s2": s2,
            "s10": s10,
            "margin_1_2": margin_1_2,
            "margin_1_10": margin_1_10,
            # Default raw confidence feature used for calibration.
            "confidence_raw": margin_1_2,
            "is_correct_top1": is_correct_top1,
            "topk_cde_ids": top_ids,
            "topk_scores": top_scores,
        }
    )

    # Failures sample
    bad = rankings[(~np.isfinite(rankings["true_rank"])) | (rankings["true_rank"] > 10)].copy()
    failures_sample = bad.head(200)

    return metrics, rankings, failures_sample


# -----------------------------
# Main grid runner
# -----------------------------

def run_grid(
    *,
    cde_master_enriched: Path,
    splits_dir: Path,
    model_name: str,
    recipes: Sequence[str],
    cde_formats: Sequence[str],
    query_variants: Sequence[str],
    rerank_modes: Sequence[str],
    alpha: float,
    top_k: int,
    k_values: Sequence[int],
    output_top_k: int,
    block_size: int,
    batch_size: int,
    normalize_embeddings: bool,
    sep: str,
    recipe_configs: Optional[Dict[str, Dict]],
    runs_dir: Path,
    artifacts_dir: Path,
    stage_tag: Optional[str] = None,
    seed: int = 1,
    eval_splits: Optional[Sequence[str]] = None,
    run_specs: Optional[Sequence[Dict[str, object]]] = None,
) -> None:
    """Run the full experiment grid."""

    t0 = time.time()

    # Seed: baseline-grid is mostly deterministic, but we still set RNG seeds
    # so any downstream sampling or library-level randomness is reproducible.
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    model_slug = _slug(model_name)
    embeddings_dir = artifacts_dir / "embeddings"
    tfidf_cache_dir = artifacts_dir / "cache" / "tfidf"

    _ensure_dir(runs_dir)
    _ensure_dir(embeddings_dir)
    _ensure_dir(tfidf_cache_dir)

    # Load inputs
    master = pd.read_parquet(cde_master_enriched)
    splits = _load_splits(splits_dir)

    # Optionally restrict evaluation to a subset of splits.
    if eval_splits:
        wanted = [str(s).strip() for s in eval_splits if str(s).strip()]
        missing = [s for s in wanted if s not in splits]
        if missing:
            available = sorted(list(splits.keys()))
            if len(missing) == 1:
                raise ValueError(
                    f"Requested split '{missing[0]}' not found in splits_dir; available: {available}"
                )
            raise ValueError(
                f"Requested splits {missing} not found in splits_dir; available: {available}"
            )
        # Preserve the user-provided order.
        splits = {s: splits[s] for s in wanted}

    # Load model once
    model = _load_sentence_transformer(model_name)

    # Pre-validate axis values
    for r in recipes:
        if not is_valid_recipe(r):
            raise ValueError(
                f"Unknown recipe: {r}. Expected an atomic recipe from {sorted(RECIPE_FIELDS)} or a composite like v1_v2_v3 (underscore-joined)."
            )

    resolved_cde_formats = [CDE_FORMAT_ALIASES.get(x, x) for x in cde_formats]
    for cf in resolved_cde_formats:
        if cf not in {"labeled", "raw"}:
            raise ValueError(f"Unknown cde_format: {cf}. Expected labeled/raw")

    resolved_query_variants = []
    for qv in query_variants:
        if qv not in QUERY_VARIANT_TO_COL:
            raise ValueError(f"Unknown query_variant: {qv}. Expected one of: {sorted(QUERY_VARIANT_TO_COL)}")
        resolved_query_variants.append(qv)

    resolved_rerank_modes = [RERANK_MODE_ALIASES.get(x, x) for x in rerank_modes]
    for rm in resolved_rerank_modes:
        if rm not in {"none", "tfidf", "hybrid"}:
            raise ValueError(f"Unknown rerank_mode: {rm}. Expected none/tfidf/hybrid or R0/R1/R2")

    # Optional explicit run_specs selection (avoids running the full cartesian product).
    allowed_specs = None
    if run_specs:
        if not isinstance(run_specs, (list, tuple)):
            raise TypeError("run_specs must be a list of dicts")

        allowed_specs = set()
        ordered_recipes: List[str] = []
        ordered_formats: List[str] = []
        ordered_qvars: List[str] = []
        ordered_reranks: List[str] = []

        def _uniq_in_order(xs: List[str]) -> List[str]:
            seen = set()
            out: List[str] = []
            for x in xs:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return out

        for item in run_specs:
            if not isinstance(item, dict):
                raise TypeError(f"Each run_specs item must be a dict, got: {type(item)}")

            recipe_i = str(item.get("recipe") or "").strip()
            qv_i = str(item.get("query_variant") or "").strip()
            if not recipe_i or not qv_i:
                raise ValueError(f"Each run_specs item must include recipe and query_variant. Got: {item!r}")

            if not is_valid_recipe(recipe_i):
                raise ValueError(f"Unknown recipe in run_specs: {recipe_i}")

            # cde_format defaults to the first resolved format (usually labeled)
            cf_raw = str(item.get("cde_format") or (resolved_cde_formats[0] if resolved_cde_formats else "labeled")).strip()
            cde_format_i = CDE_FORMAT_ALIASES.get(cf_raw, cf_raw)
            if cde_format_i not in {"labeled", "raw"}:
                raise ValueError(f"Unknown cde_format in run_specs: {cde_format_i}")

            if qv_i not in QUERY_VARIANT_TO_COL:
                raise ValueError(f"Unknown query_variant in run_specs: {qv_i}. Expected one of: {sorted(QUERY_VARIANT_TO_COL)}")

            rm_raw = str(item.get("rerank_mode") or item.get("rerank") or (resolved_rerank_modes[0] if resolved_rerank_modes else "none")).strip()
            rerank_i = RERANK_MODE_ALIASES.get(rm_raw, rm_raw)
            if rerank_i not in {"none", "tfidf", "hybrid"}:
                raise ValueError(f"Unknown rerank_mode in run_specs: {rerank_i}")

            tup = (recipe_i, cde_format_i, qv_i, rerank_i)
            if tup not in allowed_specs:
                allowed_specs.add(tup)
                ordered_recipes.append(recipe_i)
                ordered_formats.append(cde_format_i)
                ordered_qvars.append(qv_i)
                ordered_reranks.append(rerank_i)

        # Reduce axes to the exact union needed by run_specs (preserving list order).
        recipes = _uniq_in_order(ordered_recipes)
        resolved_cde_formats = _uniq_in_order(ordered_formats)
        resolved_query_variants = _uniq_in_order(ordered_qvars)
        resolved_rerank_modes = _uniq_in_order(ordered_reranks)

    # Run grid
    for recipe in recipes:
        for cde_format in resolved_cde_formats:
            # Build catalog once per recipe+format (embeddings cached)
            catalog_df = build_catalog(master, recipe=recipe, cde_format=cde_format, sep=sep, recipe_configs=recipe_configs)
            cde_texts = catalog_df["cde_text"].astype(str).tolist()

            cde_ids, cde_emb, catalog_meta = _load_or_build_catalog_embeddings(
                model=model,
                model_slug=model_slug,
                embeddings_dir=embeddings_dir,
                master=master,
                recipe=recipe,
                cde_format=cde_format,
                sep=sep,
                recipe_configs=recipe_configs,
                batch_size=batch_size,
                normalize=normalize_embeddings,
            )

            tfidf_cache_key = f"{recipe}__{cde_format}__{catalog_meta.get('recipe_key','')}"

            # Stats about fields used
            used_fields = _recipe_fields(recipe)
            field_stats = {
                f: {
                    "nonnull_rate": _nonnull_rate(master.get(f)),
                    "empty_rate": _empty_rate(master.get(f)),
                }
                for f in used_fields
                if f in master.columns
            }

            for query_variant in resolved_query_variants:
                query_col = QUERY_VARIANT_TO_COL[query_variant]

                for rerank_mode in resolved_rerank_modes:
                    if allowed_specs is not None and (recipe, cde_format, query_variant, rerank_mode) not in allowed_specs:
                        continue

                    spec = RunSpec(recipe=recipe, cde_format=cde_format, query_variant=query_variant, rerank_mode=rerank_mode, alpha=alpha)

                    run_id = make_run_id(
                        model_name=model_name,
                        recipe=recipe,
                        cde_format=cde_format,
                        query_variant=query_variant,
                        rerank_mode=rerank_mode,
                        seed=seed,
                        stage_tag=stage_tag,
                    )
                    run_dir = runs_dir / run_id
                    _ensure_dir(run_dir)

                    # Save run config early
                    run_config = {
                        "run_id": run_id,
                        "stage_tag": normalize_stage_tag(stage_tag) or None,
                        "seed": int(seed),
                        "eval_splits": list(splits.keys()),
                        "model_name": model_name,
                        "model_slug": model_slug,
                        "recipe": recipe,
                        "cde_format": cde_format,
                        "query_variant": query_variant,
                        "query_column": query_col,
                        "rerank_mode": rerank_mode,
                        "hybrid_alpha": float(alpha),
                        "top_k": int(top_k),
                        "k_values": list(map(int, k_values)),
                        "output_top_k": int(output_top_k),
                        "block_size": int(block_size),
                        "batch_size": int(batch_size),
                        "normalize_embeddings": bool(normalize_embeddings),
                        "sep": sep,
                        "recipe_configs": recipe_configs or {},
                        "inputs": {
                            "cde_master_enriched": str(cde_master_enriched),
                            "splits_dir": str(splits_dir),
                            "runs_root": str(runs_dir),
                        },
                        "catalog": {
                            "n_cdes": int(len(cde_ids)),
                            "catalog_embedding_cache": str((embeddings_dir / model_slug / 'catalog').resolve()),
                            "catalog_meta": catalog_meta,
                            "used_fields": used_fields,
                            "field_stats": field_stats,
                        },
                    }
                    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

                    # Evaluate each split
                    split_metrics: Dict[str, Dict[str, float]] = {}
                    rankings_all: List[pd.DataFrame] = []
                    failures_all: List[pd.DataFrame] = []

                    for split_name, split_path in splits.items():
                        df_split = pd.read_parquet(split_path)
                        if query_col not in df_split.columns:
                            raise KeyError(f"Split {split_name} missing required column: {query_col}")
                        if "cde_id" not in df_split.columns:
                            # Backwards compatibility: allow cde_publicid/cde_version
                            if {"cde_publicid", "cde_version"}.issubset(df_split.columns):
                                df_split = df_split.copy()
                                df_split["cde_id"] = df_split["cde_publicid"].astype(str) + "::" + df_split["cde_version"].astype(str)
                            else:
                                raise KeyError(f"Split {split_name} missing cde_id (and missing cde_publicid/cde_version)")

                        query_texts = df_split[query_col].fillna("").astype(str).tolist()
                        q_emb = _load_or_build_query_embeddings(
                            model=model,
                            model_slug=model_slug,
                            embeddings_dir=embeddings_dir,
                            split_name=split_name,
                            query_variant=query_variant,
                            query_texts=query_texts,
                            batch_size=batch_size,
                            normalize=normalize_embeddings,
                        )

                        m, rankings, failures = _evaluate_split(
                            split_name=split_name,
                            df_split=df_split,
                            query_col=query_col,
                            cde_ids=cde_ids,
                            cde_texts=cde_texts,
                            cde_emb=cde_emb,
                            query_emb=q_emb,
                            top_k=top_k,
                            k_values=k_values,
                            block_size=block_size,
                            rerank_mode=rerank_mode,
                            alpha=alpha,
                            tfidf_cache_dir=tfidf_cache_dir,
                            tfidf_cache_key=tfidf_cache_key,
                            output_top_k=output_top_k,
                        )
                        split_metrics[split_name] = m
                        rankings_all.append(rankings)
                        if len(failures) > 0:
                            failures_all.append(failures)

                    # Concatenate per-split rankings, then calibrate confidence using the validation split.
                    rankings_df = pd.concat(rankings_all, ignore_index=True) if rankings_all else pd.DataFrame()

                    # Calibration removed 2026-09-01. This block fitted an isotonic
                    # confidence calibrator and emitted reliability.csv / ece.json, via
                    # `from demap.evaluation import confidence` - the RESEARCH package,
                    # not this one. On a clean clone that import always failed; it was
                    # inside a bare `except Exception`, so every run silently wrote empty
                    # artifacts and nobody noticed. No calibration, reliability or ECE
                    # result appears anywhere in the manuscript, which reports Recall@K
                    # and MRR@100 only, so the block was out of scope rather than broken.
                    rankings_df.to_parquet(run_dir / "rankings.parquet", index=False)

                    # Write failures sample (unchanged)
                    if failures_all:
                        pd.concat(failures_all, ignore_index=True).to_csv(run_dir / "failures_sample.csv", index=False)

                    metrics_out = {
                        "run_id": run_id,
                        "spec": {
                            "recipe": recipe,
                            "cde_format": cde_format,
                            "query_variant": query_variant,
                            "rerank_mode": rerank_mode,
                            "hybrid_alpha": float(alpha),
                        },
                        "metrics_by_split": split_metrics,
                    }
                    (run_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

                    # By-family summary (test split only, if present)
                    if "test" in splits:
                        test_rankings = rankings_df[rankings_df["split"] == "test"].copy()
                        if len(test_rankings) > 0 and "family" in test_rankings.columns:
                            # Compute recall@k and MRR by family (legacy artifact).
                            # The repo's recommended protocol uses summarize_eval.py,
                            # but keeping this file is helpful during iteration.
                            rows = []
                            for (src, fam), g in test_rankings.groupby(["query_source", "family"], dropna=False):
                                ranks = g["true_rank"].values
                                out = {
                                    "query_source": src,
                                    "family": fam,
                                    "n": int(len(g)),
                                    "top1_accuracy": float((np.isfinite(ranks) & (ranks <= 1)).mean()),
                                    "mrr@100": float(np.where(np.isfinite(ranks) & (ranks <= 100), 1.0 / ranks, 0.0).mean()),
                                }
                                # Always include headline Ks for the paper/report.
                                wanted_ks = {5, 10}
                                wanted_ks.update(int(k) for k in (k_values or []))
                                for k in sorted(wanted_ks):
                                    out[f"recall@{int(k)}"] = float((np.isfinite(ranks) & (ranks <= int(k))).mean())
                                rows.append(out)
                            pd.DataFrame(rows).sort_values(["query_source", "family"]).to_csv(run_dir / "by_family.csv", index=False)

                    print(f"Wrote run -> {run_dir}")

    dt = time.time() - t0
    print(f"Completed baseline grid in {dt:.1f}s")


def _parse_csv_list(s: str) -> List[str]:
    if s is None:
        return []
    out = []
    for tok in str(s).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    ap = argparse.ArgumentParser(description="Run the baseline experiment grid.")

    ap.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML config file(s). If provided, values under the `baseline_grid` key are used as defaults.",
    )

    ap.add_argument(
        "--cde-master-enriched",
        default=None,
        help="Path to data/processed/cde_master_enriched.parquet",
    )
    ap.add_argument(
        "--splits-dir",
        default=None,
        help="Directory containing split parquet files (e.g., data/processed/splits)",
    )

    ap.add_argument("--model-name", default=None)

    ap.add_argument("--seed", type=int, default=None, help="Random seed (embedded in run directory name).")
    ap.add_argument(
        "--eval-splits",
        default=None,
        help="Comma-separated split names to evaluate (e.g., test,val). If omitted, evaluate all available splits.",
    )

    ap.add_argument("--recipes", default=None)
    ap.add_argument("--cde-formats", default=None)
    ap.add_argument("--query-variants", default=None)
    ap.add_argument("--rerank-modes", default=None)

    ap.add_argument("--hybrid-alpha", type=float, default=None)

    ap.add_argument("--top-k", type=int, default=None, help="Candidate list size before rerank")
    ap.add_argument("--k-values", default=None)
    ap.add_argument("--output-top-k", type=int, default=None, help="How many candidates to store per query in rankings.parquet")

    ap.add_argument("--block-size", type=int, default=None, help="CDE embedding block size for scoring")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--normalize-embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="L2-normalize embeddings before cosine similarity",
    )

    ap.add_argument("--sep", default=None, help="Separator used when joining multiple fields")

    ap.add_argument(
        "--v1-filter-numeric-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Drop v1 SHORT_NAME values that contain no alphabet characters (digits/._- only)",
    )

    ap.add_argument("--runs-dir", default=None, help="Directory for run outputs. Use \"auto\" for artifacts/<stage_tag>/<model_slug>/runs (or artifacts/off_the_shelf/... when stage_tag is unset).")
    ap.add_argument("--artifacts-dir", default=None, help="Artifacts root (embedding caches, etc.). Default: artifacts")
    ap.add_argument("--stage-tag", default=None, help="Optional stage tag used for collision-free auto roots and deterministic run IDs.")

    args = ap.parse_args(argv_list)

    # Load config (if any) and apply as defaults.
    cfg = {}
    for cpath in args.config or []:
        cfg = deep_merge(cfg, load_config(cpath))
    cfg = cfg.get("baseline_grid", cfg) if cfg else {}

    def flag_present(*flags: str) -> bool:
        return any(f in argv_list for f in flags)

    # Required-ish paths
    cde_master_enriched = args.cde_master_enriched if flag_present("--cde-master-enriched") else (args.cde_master_enriched or cfg.get("cde_master_enriched"))
    splits_dir = args.splits_dir if flag_present("--splits-dir") else (args.splits_dir or cfg.get("splits_dir"))
    if not cde_master_enriched:
        cde_master_enriched = os.path.join("data", "processed", "cde_master_enriched.parquet")
    if not splits_dir:
        splits_dir = os.path.join("data", "processed", "splits")

    model_name = args.model_name if flag_present("--model-name") else (args.model_name or cfg.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2")

    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 1))

    if args.eval_splits is not None:
        eval_splits = _parse_csv_list(args.eval_splits)
    elif cfg.get("eval_splits") is not None:
        # Accept YAML list or comma-separated string
        es = cfg.get("eval_splits")
        if isinstance(es, str):
            eval_splits = _parse_csv_list(es)
        else:
            eval_splits = list(es or [])
    else:
        eval_splits = None


    # Grid axes
    recipes = _parse_csv_list(args.recipes) if args.recipes is not None else []
    if (not flag_present("--recipes")) and cfg.get("recipes") is not None:
        recipes = list(cfg.get("recipes") or [])
    if not recipes:
        recipes = ["v1", "v2", "v3", "v4", "v5", "v6"]

    cde_formats = _parse_csv_list(args.cde_formats) if args.cde_formats is not None else []
    if (not flag_present("--cde-formats")) and cfg.get("cde_formats") is not None:
        cde_formats = list(cfg.get("cde_formats") or [])
    if not cde_formats:
        cde_formats = ["labeled"]

    query_variants = _parse_csv_list(args.query_variants) if args.query_variants is not None else []
    if (not flag_present("--query-variants")) and cfg.get("query_variants") is not None:
        query_variants = list(cfg.get("query_variants") or [])
    if not query_variants:
        query_variants = ["Q1", "Q2"]

    rerank_modes = _parse_csv_list(args.rerank_modes) if args.rerank_modes is not None else []
    if (not flag_present("--rerank-modes")) and cfg.get("rerank_modes") is not None:
        rerank_modes = list(cfg.get("rerank_modes") or [])
    if not rerank_modes:
        rerank_modes = ["R0", "R1", "R2"]

    alpha = float(args.hybrid_alpha) if args.hybrid_alpha is not None else float(cfg.get("hybrid_alpha", 0.5))

    top_k = int(args.top_k) if args.top_k is not None else int(cfg.get("top_k", 200))
    if args.k_values is not None:
        k_values = [int(x) for x in _parse_csv_list(args.k_values)]
    else:
        k_values = list(cfg.get("k_values", [1, 5, 10, 20]))

    output_top_k = int(args.output_top_k) if args.output_top_k is not None else int(cfg.get("output_top_k", 20))
    block_size = int(args.block_size) if args.block_size is not None else int(cfg.get("block_size", 50000))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.get("batch_size", 64))
    normalize_embeddings = bool(args.normalize_embeddings) if args.normalize_embeddings is not None else bool(cfg.get("normalize_embeddings", True))
    sep = str(args.sep) if args.sep is not None else str(cfg.get("sep", " | "))

    # Recipe configs (currently only v1.filter_numeric_only is exposed as a flag)
    recipe_configs = cfg.get("recipe_configs", {}) or {}
    v1_cfg = dict(recipe_configs.get("v1", {}) or {})
    if args.v1_filter_numeric_only is not None:
        v1_cfg["filter_numeric_only"] = bool(args.v1_filter_numeric_only)
    elif "filter_numeric_only" not in v1_cfg:
        v1_cfg["filter_numeric_only"] = True
    recipe_configs = dict(recipe_configs)
    recipe_configs["v1"] = v1_cfg


    # Optional explicit run_specs (list of dicts) to run a curated set of cells.
    run_specs = cfg.get("run_specs")

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir is not None else Path(cfg.get("artifacts_dir", "artifacts"))
    stage_tag = args.stage_tag if args.stage_tag is not None else cfg.get("stage_tag")

    # Resolve runs_dir. Supports runs_dir='auto' to write under:
    #   artifacts/off_the_shelf/<model_slug>/runs/<run_id>/              (default)
    #   artifacts/<stage_tag>/<model_slug>/runs/<run_id>/                (when stage_tag is set)
    # Backwards-compatible: if runs_dir is not provided, fall back to cfg.output_root, else artifacts/runs.
    if args.runs_dir is not None:
        runs_dir_cfg = args.runs_dir
    elif cfg.get("runs_dir") is not None:
        runs_dir_cfg = cfg.get("runs_dir")
    else:
        runs_dir_cfg = cfg.get("output_root", os.path.join("artifacts", "runs"))

    runs_dir = resolve_runs_root(
        artifacts_dir=artifacts_dir,
        runs_dir_cfg=runs_dir_cfg,
        model_name=model_name,
        stage_tag=stage_tag,
    )

    run_grid(
        cde_master_enriched=Path(cde_master_enriched),
        splits_dir=Path(splits_dir),
        model_name=model_name,
        recipes=recipes,
        cde_formats=cde_formats,
        query_variants=query_variants,
        rerank_modes=rerank_modes,
        alpha=alpha,
        top_k=top_k,
        k_values=k_values,
        output_top_k=output_top_k,
        block_size=block_size,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        sep=sep,
        recipe_configs=recipe_configs,
        runs_dir=runs_dir,
        artifacts_dir=artifacts_dir,
        stage_tag=stage_tag,
        seed=seed,
        eval_splits=eval_splits,
        run_specs=run_specs,
    )


if __name__ == "__main__":
    main()
