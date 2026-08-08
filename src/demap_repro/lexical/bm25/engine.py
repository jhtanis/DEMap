#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from demap_repro.lexical.bm25.quarantine import require_catalog
from demap_repro.text.normalize import normalize_query_text
from demap_repro.text.recipes import RECIPE_FIELDS, build_catalog, is_valid_recipe
from demap_repro.utils.config import deep_merge, load_config


QUERY_VARIANT_TO_COL = {
    "Q1": "query_text_raw",
    "Q2": "query_text",
    "Q3": "query_text_q3",
    "Q4": "query_text_q4",
    "query_text_raw": "query_text_raw",
    "query_text": "query_text",
    "query_text_q3": "query_text_q3",
    "query_text_q4": "query_text_q4",
}

CDE_FORMAT_ALIASES = {
    "labeled": "labeled",
    "raw": "raw",
    "cde_text": "labeled",
    "raw_cde_text": "raw",
}

_BASE_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")


def _slug(s: str) -> str:
    return str(s).replace("/", "__").replace(" ", "_").replace(":", "_")


def _float_tag(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_stage_tag(stage_tag: Optional[str]) -> str:
    t = str(stage_tag or "").strip()
    if not t:
        return ""
    return _slug(t).strip("_")


def resolve_runs_root(*, artifacts_dir: Path, runs_dir_cfg: str | Path, stage_tag: Optional[str] = None) -> Path:
    if str(runs_dir_cfg).lower() != "auto":
        return Path(runs_dir_cfg)
    stage_root = normalize_stage_tag(stage_tag) or "bm25_baseline"
    return Path(artifacts_dir) / stage_root / "bm25" / "runs"


def make_run_id(
    *,
    recipe: str,
    cde_format: str,
    query_variant: str,
    tokenizer_name: str,
    k1: float,
    b: float,
    query_binary: bool,
    stage_tag: Optional[str] = None,
) -> str:
    parts = ["__bm25__"]
    stage_slug = normalize_stage_tag(stage_tag)
    if stage_slug:
        parts.append(f"{stage_slug}__")
    parts.append(
        f"{_slug(tokenizer_name)}__{recipe}__{cde_format}__{query_variant}"
        f"__k1{_float_tag(k1)}__b{_float_tag(b)}__qb{int(bool(query_binary))}"
    )
    return "".join(parts)


def _parse_csv_list(s: str | None) -> List[str]:
    if s is None:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _load_splits(splits_dir: Path) -> Dict[str, Path]:
    preferred = [
        "train.parquet",
        "val.parquet",
        "test.parquet",
        "external_holdout_org.parquet",
        "external_holdout_standard.parquet",
        "external_holdout_refslice.parquet",
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
    for p in sorted([x for x in splits_dir.glob("*.parquet") if x.is_file() and x.name not in seen]):
        out[p.stem] = p
    if not out:
        raise FileNotFoundError(f"No split parquet files found in: {splits_dir}")
    return out


def tokenize_baseline(x: Any, *, min_token_len: int = 1) -> List[str]:
    s = normalize_query_text(x).casefold()
    if not s:
        return []
    toks = _BASE_WORD_RE.findall(s)
    if min_token_len > 1:
        toks = [t for t in toks if len(t) >= min_token_len]
    return toks


def tokenize_altaware(x: Any, *, min_token_len: int = 1) -> List[str]:
    s0 = normalize_query_text(x)
    if not s0:
        return []
    s0 = _CAMEL_RE.sub(r"\1 \2", s0)
    s = s0.casefold()
    s = _NON_ALNUM_RE.sub(" ", s)
    toks = [t for t in s.split() if t]
    if min_token_len > 1:
        toks = [t for t in toks if len(t) >= min_token_len]
    return toks


def _get_tokenizer(name: str):
    n = str(name).strip().lower()
    if n == "baseline":
        return tokenize_baseline
    if n == "altaware":
        return tokenize_altaware
    raise ValueError(f"Unknown tokenizer: {name}. Expected baseline or altaware")


def _base_metrics_from_true_rank(ranks: np.ndarray) -> Dict[str, float]:
    r = ranks.astype(float)
    fin = np.isfinite(r)
    out: Dict[str, float] = {
        "n": float(len(r)),
        "top1_accuracy": float((fin & (r <= 1)).mean()),
        "mrr@100": float(np.where(fin & (r <= 100), 1.0 / r, 0.0).mean()),
        "mrr": float(np.where(fin, 1.0 / r, 0.0).mean()),
        "median_rank": float(np.median(r[fin]) if fin.any() else np.inf),
    }
    for k in [1, 5, 10, 20, 100]:
        out[f"recall@{k}"] = float((fin & (r <= k)).mean())
    return out


def build_bm25_model(
    *,
    catalog_ids: List[str],
    catalog_texts: List[str],
    tokenizer_name: str,
    min_token_len: int,
    k1: float,
    b: float,
    query_binary: bool,
) -> Dict[str, Any]:
    tokenizer = _get_tokenizer(tokenizer_name)
    vectorizer = CountVectorizer(
        analyzer=lambda s: tokenizer(s, min_token_len=min_token_len),
        token_pattern=None,
        lowercase=False,
        min_df=1,
    )
    x_tf = vectorizer.fit_transform(catalog_texts).tocsr()
    n_docs, vocab_size = x_tf.shape
    df = np.bincount(x_tf.indices, minlength=vocab_size).astype(np.float32)
    N = float(n_docs)
    idf = np.log(((N - df + 0.5) / (df + 0.5)) + 1.0).astype(np.float32)

    dl = np.asarray(x_tf.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(dl.mean()) if dl.size else 0.0
    K = (float(k1) * (1.0 - float(b) + float(b) * (dl / (avgdl + 1e-9)))).astype(np.float32)

    x_bm25 = x_tf.astype(np.float32).copy()
    for i in range(n_docs):
        start, end = x_bm25.indptr[i], x_bm25.indptr[i + 1]
        if start == end:
            continue
        tf = x_bm25.data[start:end]
        idx = x_bm25.indices[start:end]
        denom = tf + K[i]
        x_bm25.data[start:end] = idf[idx] * (tf * (float(k1) + 1.0)) / denom
    x_bm25.sort_indices()

    return {
        "vectorizer": vectorizer,
        "x_bm25": x_bm25,
        "catalog_ids": np.array(catalog_ids, dtype=object),
        "cde_id_to_doc_idx": {str(cid): i for i, cid in enumerate(catalog_ids)},
        "query_binary": bool(query_binary),
        "tokenizer_name": str(tokenizer_name),
        "min_token_len": int(min_token_len),
        "k1": float(k1),
        "b": float(b),
        "avgdl": avgdl,
        "vocab_size": int(vocab_size),
        "n_docs": int(n_docs),
    }


def _rank_order(indices: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.zeros((0,), dtype=np.int32)
    order = np.lexsort((indices.astype(np.int64), -scores.astype(np.float64)))
    return order.astype(np.int32)


def _evaluate_split(
    *,
    split_name: str,
    df_split: pd.DataFrame,
    query_col: str,
    model: Dict[str, Any],
    top_k: int,
    output_top_k: int,
    batch_size: int,
    k_values: Sequence[int],
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    vectorizer = model["vectorizer"]
    x_bm25 = model["x_bm25"]
    catalog_ids = model["catalog_ids"]
    cde_id_to_doc_idx = model["cde_id_to_doc_idx"]
    query_binary = bool(model["query_binary"])

    true_ids = df_split["cde_id"].astype(str).tolist()
    true_idx = np.array([cde_id_to_doc_idx.get(t, -1) for t in true_ids], dtype=np.int32)
    query_texts = df_split[query_col].fillna("").astype(str).tolist()

    n = len(df_split)
    ranks = np.full((n,), fill_value=np.inf, dtype=np.float32)
    true_scores = np.zeros((n,), dtype=np.float32)
    top_ids: List[List[str]] = []
    top_scores: List[List[float]] = []
    top1_ids: List[str] = []
    top1_scores: List[float] = []
    s1s: List[float] = []
    s2s: List[float] = []
    s10s: List[float] = []
    margins12: List[float] = []
    margins110: List[float] = []
    is_correct_top1: List[bool] = []

    for start in range(0, n, int(batch_size)):
        stop = min(n, start + int(batch_size))
        q_batch = vectorizer.transform(query_texts[start:stop])
        if query_binary and q_batch.nnz:
            q_batch.data[:] = 1.0
        scores_mat = (q_batch @ x_bm25.T).tocsr()
        scores_mat.sort_indices()

        for row_local in range(stop - start):
            row_idx = start + row_local
            rs, re = scores_mat.indptr[row_local], scores_mat.indptr[row_local + 1]
            idx = scores_mat.indices[rs:re].astype(np.int32, copy=False)
            data = scores_mat.data[rs:re].astype(np.float32, copy=False)
            order = _rank_order(idx, data)
            idx_sorted = idx[order]
            data_sorted = data[order]

            gold_idx = int(true_idx[row_idx])
            if gold_idx >= 0:
                hits = np.where(idx_sorted == gold_idx)[0]
                if hits.size > 0:
                    true_scores[row_idx] = float(data_sorted[hits[0]])
                    ranks[row_idx] = float(hits[0] + 1)
                else:
                    true_scores[row_idx] = 0.0
                    ranks[row_idx] = np.inf

            out_k = int(min(output_top_k, top_k, data_sorted.size))
            ids = [str(catalog_ids[j]) for j in idx_sorted[:out_k]]
            scs = data_sorted[:out_k].astype(float).tolist()
            top_ids.append(ids)
            top_scores.append(scs)
            top1_ids.append(ids[0] if ids else "")
            top1_scores.append(float(scs[0]) if scs else float("nan"))
            s1 = float(scs[0]) if len(scs) > 0 else float("nan")
            s2 = float(scs[1]) if len(scs) > 1 else float("nan")
            s10 = float(scs[9]) if len(scs) > 9 else float("nan")
            s1s.append(s1)
            s2s.append(s2)
            s10s.append(s10)
            margins12.append(s1 - s2 if np.isfinite(s1) and np.isfinite(s2) else float("nan"))
            margins110.append(s1 - s10 if np.isfinite(s1) and np.isfinite(s10) else float("nan"))
            is_correct_top1.append(bool(np.isfinite(ranks[row_idx]) and ranks[row_idx] == 1.0))

    metrics = _base_metrics_from_true_rank(ranks)
    for k in sorted({int(k) for k in (k_values or [])}):
        metrics[f"recall@{int(k)}"] = float((np.isfinite(ranks) & (ranks <= int(k))).mean())
    metrics["n_missing_target_in_catalog"] = float(int((true_idx < 0).sum()))

    rankings = pd.DataFrame(
        {
            "split": split_name,
            "pair_id": df_split.get("pair_id", pd.Series([None] * n)).astype(str),
            "query_id": df_split.get("query_id", pd.Series([None] * n)).astype(str),
            "query_source": df_split.get("query_source", pd.Series([None] * n)).astype(str),
            "family": df_split.get("family", pd.Series([None] * n)).astype(str),
            "query_text": query_texts,
            "true_cde_id": true_ids,
            "true_rank": ranks,
            "true_score": true_scores,
            "top1_cde_id": top1_ids,
            "top1_score": top1_scores,
            "s1": s1s,
            "s2": s2s,
            "s10": s10s,
            "margin_1_2": margins12,
            "margin_1_10": margins110,
            "confidence_raw": margins12,
            "is_correct_top1": is_correct_top1,
            "p_correct": [float("nan")] * n,
            "topk_cde_ids": top_ids,
            "topk_scores": top_scores,
        }
    )

    bad = rankings[(~np.isfinite(rankings["true_rank"])) | (rankings["true_rank"] > 10)].copy()
    failures_sample = bad.head(200)
    return metrics, rankings, failures_sample


def run_grid(
    *,
    cde_master_enriched: Path,
    splits_dir: Path,
    recipes: Sequence[str],
    cde_formats: Sequence[str],
    query_variants: Sequence[str],
    top_k: int,
    k_values: Sequence[int],
    output_top_k: int,
    sep: str,
    recipe_configs: Optional[Dict[str, Dict]],
    eval_splits: Optional[Sequence[str]],
    runs_dir: Path,
    stage_tag: Optional[str],
    tokenizer_name: str,
    min_token_len: int,
    bm25_k1: float,
    bm25_b: float,
    query_binary: bool,
    batch_size: int,
    run_specs: Optional[Sequence[Dict[str, object]]] = None,
) -> None:
    t0 = time.time()
    _ensure_dir(runs_dir)

    master = pd.read_parquet(cde_master_enriched)
    if "cde_id" not in master.columns:
        if {"cde_publicid", "cde_version"}.issubset(master.columns):
            master = master.copy()
            master["cde_id"] = master["cde_publicid"].astype(str) + "::" + master["cde_version"].astype(str)
        else:
            raise ValueError("cde_master_enriched must contain cde_id or cde_publicid/cde_version")

    splits = _load_splits(splits_dir)
    if eval_splits:
        wanted = list(eval_splits)
        missing = [s for s in wanted if s not in splits]
        if missing:
            raise ValueError(f"Requested splits {missing} not found in splits_dir; available: {sorted(splits)}")
        splits = {s: splits[s] for s in wanted}

    for r in recipes:
        if not is_valid_recipe(r):
            raise ValueError(
                f"Unknown recipe: {r}. Expected an atomic recipe from {sorted(RECIPE_FIELDS)} or a composite like v1_v2_v3."
            )

    resolved_cde_formats = [CDE_FORMAT_ALIASES.get(x, x) for x in cde_formats]
    for cf in resolved_cde_formats:
        if cf not in {"labeled", "raw"}:
            raise ValueError(f"Unknown cde_format: {cf}. Expected labeled/raw")

    resolved_query_variants: List[str] = []
    for qv in query_variants:
        if qv not in QUERY_VARIANT_TO_COL:
            raise ValueError(f"Unknown query_variant: {qv}. Expected one of: {sorted(QUERY_VARIANT_TO_COL)}")
        resolved_query_variants.append(qv)

    allowed_specs = None
    if run_specs:
        if not isinstance(run_specs, (list, tuple)):
            raise TypeError("run_specs must be a list of dicts")
        allowed_specs = set()
        ordered_recipes: List[str] = []
        ordered_formats: List[str] = []
        ordered_qvars: List[str] = []

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
            cf_raw = str(item.get("cde_format") or (resolved_cde_formats[0] if resolved_cde_formats else "labeled")).strip()
            cde_format_i = CDE_FORMAT_ALIASES.get(cf_raw, cf_raw)
            if cde_format_i not in {"labeled", "raw"}:
                raise ValueError(f"Unknown cde_format in run_specs: {cde_format_i}")
            if qv_i not in QUERY_VARIANT_TO_COL:
                raise ValueError(f"Unknown query_variant in run_specs: {qv_i}")
            tup = (recipe_i, cde_format_i, qv_i)
            if tup not in allowed_specs:
                allowed_specs.add(tup)
                ordered_recipes.append(recipe_i)
                ordered_formats.append(cde_format_i)
                ordered_qvars.append(qv_i)

        recipes = _uniq_in_order(ordered_recipes)
        resolved_cde_formats = _uniq_in_order(ordered_formats)
        resolved_query_variants = _uniq_in_order(ordered_qvars)

    for recipe in recipes:
        for cde_format in resolved_cde_formats:
            catalog_df = build_catalog(master, recipe=recipe, cde_format=cde_format, sep=sep, recipe_configs=recipe_configs)
            cde_texts = catalog_df["cde_text"].fillna("").astype(str).tolist()
            cde_ids = catalog_df["cde_id"].astype(str).tolist()
            model = build_bm25_model(
                catalog_ids=cde_ids,
                catalog_texts=cde_texts,
                tokenizer_name=tokenizer_name,
                min_token_len=min_token_len,
                k1=bm25_k1,
                b=bm25_b,
                query_binary=query_binary,
            )

            for query_variant in resolved_query_variants:
                if allowed_specs is not None and (recipe, cde_format, query_variant) not in allowed_specs:
                    continue
                query_col = QUERY_VARIANT_TO_COL[query_variant]
                run_id = make_run_id(
                    recipe=recipe,
                    cde_format=cde_format,
                    query_variant=query_variant,
                    tokenizer_name=tokenizer_name,
                    k1=bm25_k1,
                    b=bm25_b,
                    query_binary=query_binary,
                    stage_tag=stage_tag,
                )
                run_dir = runs_dir / run_id
                _ensure_dir(run_dir)

                run_config = {
                    "run_id": run_id,
                    "stage_tag": normalize_stage_tag(stage_tag) or None,
                    "scorer_name": "bm25",
                    "eval_splits": list(splits.keys()),
                    "query_variant": query_variant,
                    "query_column": query_col,
                    "recipe": recipe,
                    "cde_format": cde_format,
                    "top_k": int(top_k),
                    "k_values": [int(x) for x in k_values],
                    "output_top_k": int(output_top_k),
                    "batch_size": int(batch_size),
                    "sep": sep,
                    "recipe_configs": recipe_configs or {},
                    "bm25": {
                        "tokenizer": tokenizer_name,
                        "min_token_len": int(min_token_len),
                        "k1": float(bm25_k1),
                        "b": float(bm25_b),
                        "query_binary": bool(query_binary),
                    },
                    "inputs": {
                        "cde_master_enriched": str(cde_master_enriched),
                        "splits_dir": str(splits_dir),
                        "runs_root": str(runs_dir),
                    },
                    "catalog": {
                        "n_cdes": int(len(cde_ids)),
                        "vocab_size": int(model["vocab_size"]),
                        "avgdl": float(model["avgdl"]),
                    },
                }
                (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

                split_metrics: Dict[str, Dict[str, float]] = {}
                rankings_all: List[pd.DataFrame] = []
                failures_all: List[pd.DataFrame] = []

                for split_name, split_path in splits.items():
                    df_split = pd.read_parquet(split_path)
                    if query_col not in df_split.columns:
                        raise KeyError(f"Split {split_name} missing required column: {query_col}")
                    if "cde_id" not in df_split.columns:
                        if {"cde_publicid", "cde_version"}.issubset(df_split.columns):
                            df_split = df_split.copy()
                            df_split["cde_id"] = df_split["cde_publicid"].astype(str) + "::" + df_split["cde_version"].astype(str)
                        else:
                            raise KeyError(f"Split {split_name} missing cde_id (and missing cde_publicid/cde_version)")

                    m, rankings, failures = _evaluate_split(
                        split_name=split_name,
                        df_split=df_split,
                        query_col=query_col,
                        model=model,
                        top_k=top_k,
                        output_top_k=output_top_k,
                        batch_size=batch_size,
                        k_values=k_values,
                    )
                    split_metrics[split_name] = m
                    rankings_all.append(rankings)
                    if len(failures) > 0:
                        failures_all.append(failures)

                rankings_df = pd.concat(rankings_all, ignore_index=True) if rankings_all else pd.DataFrame()
                rankings_df.to_parquet(run_dir / "rankings.parquet", index=False)

                pd.DataFrame(
                    {
                        "split": [],
                        "bin_idx": [],
                        "bin_lo": [],
                        "bin_hi": [],
                        "n": [],
                        "avg_confidence": [],
                        "accuracy": [],
                    }
                ).to_csv(run_dir / "reliability.csv", index=False)
                ece_json = {
                    "_error": {
                        "available": False,
                        "reason": "bm25_baseline_has_no_probability_calibration",
                    }
                }
                (run_dir / "ece.json").write_text(json.dumps(ece_json, indent=2), encoding="utf-8")

                if failures_all:
                    pd.concat(failures_all, ignore_index=True).to_csv(run_dir / "failures_sample.csv", index=False)

                metrics_out = {
                    "run_id": run_id,
                    "spec": {
                        "recipe": recipe,
                        "cde_format": cde_format,
                        "query_variant": query_variant,
                        "scorer_name": "bm25",
                    },
                    "bm25": run_config["bm25"],
                    "metrics_by_split": split_metrics,
                    "confidence": {
                        "calibration": {
                            "available": False,
                            "method": None,
                            "reason": "bm25_baseline_has_no_probability_calibration",
                        },
                        "confidence_threshold": 0.9,
                        "n_bins": 0,
                    },
                }
                (run_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

                if "test" in splits and len(rankings_df) > 0 and "family" in rankings_df.columns:
                    test_rankings = rankings_df[rankings_df["split"] == "test"].copy()
                    if len(test_rankings) > 0:
                        rows = []
                        for (src, fam), g in test_rankings.groupby(["query_source", "family"], dropna=False):
                            ranks_arr = g["true_rank"].astype(float).to_numpy()
                            met = _base_metrics_from_true_rank(ranks_arr)
                            rows.append(
                                {
                                    "query_source": src,
                                    "family": fam,
                                    "n": int(len(g)),
                                    "top1_accuracy": float(met["top1_accuracy"]),
                                    "mrr@100": float(met["mrr@100"]),
                                    "recall@5": float(met["recall@5"]),
                                    "recall@10": float(met["recall@10"]),
                                }
                            )
                        pd.DataFrame(rows).sort_values(["query_source", "family"]).to_csv(run_dir / "by_family.csv", index=False)

                print(f"Wrote BM25 run -> {run_dir}")

    print(f"Completed BM25 grid in {time.time() - t0:.1f}s")


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Run a BM25 baseline retrieval grid.")
    ap.add_argument("--config", action="append", default=[], help="YAML config file(s). Expected key: bm25_baseline.")
    ap.add_argument("--cde-master-enriched", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--eval-splits", default=None, help="Comma-separated split names to evaluate.")
    ap.add_argument("--recipes", default=None)
    ap.add_argument("--cde-formats", default=None)
    ap.add_argument("--query-variants", default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--k-values", default=None)
    ap.add_argument("--output-top-k", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="Query batch size for sparse scoring.")
    ap.add_argument("--sep", default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--stage-tag", default=None)
    ap.add_argument("--tokenizer", default=None, help="baseline or altaware")
    ap.add_argument("--min-token-len", type=int, default=None)
    ap.add_argument("--k1", type=float, default=None)
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--query-binary", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--allow-legacy", action="store_true",
        help="Authorize a historical-reproduction run (e.g. the January 79,479-row catalog). "
             "Output is NOT valid for the paper; use scripts/paper_bm25_canonical.py instead.")
    args = ap.parse_args(argv_list)

    cfg: Dict[str, Any] = {}
    for cpath in args.config or []:
        cfg = deep_merge(cfg, load_config(cpath))
    cfg = cfg.get("bm25_baseline", cfg) if cfg else {}

    def flag_present(*flags: str) -> bool:
        return any(f in argv_list for f in flags)

    cde_master_enriched = args.cde_master_enriched if flag_present("--cde-master-enriched") else (args.cde_master_enriched or cfg.get("cde_master_enriched"))
    splits_dir = args.splits_dir if flag_present("--splits-dir") else (args.splits_dir or cfg.get("splits_dir"))
    # Fail closed. The historical code silently defaulted to the January 79,479-row
    # benchmark-construction catalog here; that default is removed. See bm25_quarantine.
    cde_master_enriched = require_catalog(
        cde_master_enriched, allow_legacy=bool(args.allow_legacy), context="demap bm25-baseline")
    if not splits_dir:
        splits_dir = os.path.join("data", "processed", "splits")

    if args.eval_splits is not None:
        eval_splits = _parse_csv_list(args.eval_splits)
    elif cfg.get("eval_splits") is not None:
        es = cfg.get("eval_splits")
        eval_splits = _parse_csv_list(es) if isinstance(es, str) else list(es or [])
    else:
        eval_splits = None

    def cfg_list(arg_name: str, cfg_name: str, default_csv: str) -> List[str]:
        cli_val = getattr(args, arg_name)
        if cli_val is not None:
            return _parse_csv_list(cli_val)
        val = cfg.get(cfg_name)
        if val is None:
            return _parse_csv_list(default_csv)
        if isinstance(val, str):
            return _parse_csv_list(val)
        return list(val or [])

    recipes = cfg_list("recipes", "recipes", "v3")
    cde_formats = cfg_list("cde_formats", "cde_formats", "labeled")
    query_variants = cfg_list("query_variants", "query_variants", "Q1")
    k_values = [int(x) for x in cfg_list("k_values", "k_values", "1,5,10,20")]

    top_k = int(args.top_k) if args.top_k is not None else int(cfg.get("top_k", 200))
    output_top_k = int(args.output_top_k) if args.output_top_k is not None else int(cfg.get("output_top_k", min(20, top_k)))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.get("batch_size", 256))
    sep = args.sep if args.sep is not None else str(cfg.get("sep", " | "))
    runs_dir_cfg = args.runs_dir if args.runs_dir is not None else str(cfg.get("runs_dir", "auto"))
    artifacts_dir = Path(args.artifacts_dir if args.artifacts_dir is not None else cfg.get("artifacts_dir", "artifacts"))
    stage_tag = args.stage_tag if args.stage_tag is not None else cfg.get("stage_tag")
    tokenizer_name = args.tokenizer if args.tokenizer is not None else str(cfg.get("tokenizer", "baseline"))
    min_token_len = int(args.min_token_len) if args.min_token_len is not None else int(cfg.get("min_token_len", 1))
    bm25_k1 = float(args.k1) if args.k1 is not None else float(cfg.get("k1", 1.2))
    bm25_b = float(args.b) if args.b is not None else float(cfg.get("b", 0.75))
    query_binary = bool(args.query_binary) if args.query_binary is not None else bool(cfg.get("query_binary", True))
    recipe_configs = cfg.get("recipe_configs") or {}
    run_specs = cfg.get("run_specs")

    runs_root = resolve_runs_root(artifacts_dir=artifacts_dir, runs_dir_cfg=runs_dir_cfg, stage_tag=stage_tag)

    run_grid(
        cde_master_enriched=Path(cde_master_enriched),
        splits_dir=Path(splits_dir),
        recipes=recipes,
        cde_formats=cde_formats,
        query_variants=query_variants,
        top_k=top_k,
        k_values=k_values,
        output_top_k=output_top_k,
        sep=sep,
        recipe_configs=recipe_configs,
        eval_splits=eval_splits,
        runs_dir=runs_root,
        stage_tag=stage_tag,
        tokenizer_name=tokenizer_name,
        min_token_len=min_token_len,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        query_binary=query_binary,
        batch_size=batch_size,
        run_specs=run_specs,
    )


if __name__ == "__main__":
    main()
