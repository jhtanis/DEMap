"""Canonical, parent-bound hard-negative mining for Phase 2 (fail-closed).

Every mined artifact is identified by the full lineage that determines its contents:

    model ID · parent checkpoint hash · query representation · CDE recipe · CDE format ·
    training-split hash · catalog hash · mining strategy family · mining parameters ·
    code commit

The identity is hashed into the artifact directory name, so a mined file produced from a
different parent checkpoint can never be silently reused, and the historical root
(``artifacts/phase2_hardneg_mining/``) can never be overwritten — this module writes only
under the new versioned root declared by ``phase2.hardneg.artifact_root``.

Guarantees enforced here:
  * mining reads ONLY an allow-listed split (canonical: ``train``); val_dev/test refused;
  * the production catalog hash must match the protocol;
  * every accepted gold CDE for a normalised query key is excluded from that key's
    negatives, and the example's own positive is excluded unconditionally;
  * candidates are de-duplicated by CDE id;
  * exactly ``n_negatives`` valid negatives are produced per (example, epoch);
  * a positive is NEVER used as a negative -- if the strategy band is exhausted the fill
    is deterministic from valid non-gold catalog entries and is RECORDED per example;
  * re-mining refuses to overwrite unless the existing artifact matches the identity, in
    which case it is a no-op (safe resume).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ARTIFACT_SCHEMA = "phase2_hardneg_artifact_v2"
ALLOWED_MINE_SPLITS = ("train",)

# Strategy -> inclusive 1-based rank band over the mined top-K candidate list.
STRATEGY_BANDS: Dict[str, Optional[Tuple[int, int]]] = {
    "none": None,
    "hard_top25": (1, 25),
    "semihard_1_50": (1, 50),
}
EXCLUDED_STRATEGIES = ("hardcurr_1_50__1_25__1_15",)


class HardNegError(SystemExit):
    pass


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_hash(checkpoint_dir: str) -> str:
    """Stable content hash of a SentenceTransformer checkpoint directory.

    Hashes the weight file plus the structural configs, in a fixed order. Cheap enough to
    call per generation and strong enough to detect a swapped parent.
    """
    p = Path(checkpoint_dir)
    if not p.is_dir():
        raise HardNegError(f"[hardneg] checkpoint directory not found: {checkpoint_dir}")
    names = ["model.safetensors", "pytorch_model.bin", "config.json",
             "config_sentence_transformers.json", "modules.json", "sentence_bert_config.json",
             "tokenizer_config.json"]
    h = hashlib.sha256()
    found_weights = False
    for n in names:
        f = p / n
        if not f.exists():
            continue
        if n in ("model.safetensors", "pytorch_model.bin"):
            found_weights = True
        h.update(n.encode())
        h.update(sha256_file(str(f)).encode())
    if not found_weights:
        raise HardNegError(f"[hardneg] no weight file in checkpoint: {checkpoint_dir}")
    return h.hexdigest()


def build_identity(
    *, model_id: str, parent_checkpoint: str, parent_checkpoint_sha256: str,
    query_variant: str, recipe: str, cde_format: str,
    train_split_sha256: str, catalog_sha256: str,
    top_k: int, n_negatives: int, mine_split: str, code_commit: str,
    similarity: str = "cosine_on_l2_normalized_embeddings",
) -> Dict[str, Any]:
    """Full mined-artifact identity + a short deterministic directory key."""
    if mine_split not in ALLOWED_MINE_SPLITS:
        raise HardNegError(
            f"[hardneg] split {mine_split!r} is not allow-listed for mining; "
            f"allowed: {list(ALLOWED_MINE_SPLITS)}")
    ident = {
        "schema": ARTIFACT_SCHEMA,
        "model_id": model_id,
        "parent_checkpoint": parent_checkpoint,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "query_variant": query_variant,
        "recipe": recipe,
        "cde_format": cde_format,
        "train_split_sha256": train_split_sha256,
        "catalog_sha256": catalog_sha256,
        "mine_split": mine_split,
        "top_k": int(top_k),
        "n_negatives": int(n_negatives),
        "similarity": similarity,
        "strategy_bands": {k: list(v) if v else None for k, v in STRATEGY_BANDS.items()},
        "code_commit": code_commit,
    }
    # The key deliberately EXCLUDES code_commit so a no-op recommit does not orphan a
    # valid artifact; the commit is still recorded and reported.
    keyed = {k: v for k, v in ident.items() if k != "code_commit"}
    digest = _sha256_bytes(json.dumps(keyed, sort_keys=True).encode())
    ident["identity_sha256"] = digest
    slug = model_id.replace("/", "__")
    ident["artifact_key"] = f"{slug}__{query_variant}__{recipe}__{digest[:16]}"
    return ident


def artifact_paths(root: str, identity: Dict[str, Any]) -> Dict[str, str]:
    d = Path(root) / identity["artifact_key"]
    pq = d / f"{identity['mine_split']}_topk{identity['top_k']}.parquet"
    return {"dir": str(d), "parquet": str(pq),
            "identity_json": str(d / "IDENTITY.json"),
            "manifest_json": str(pq) + ".manifest.json"}


def assert_artifact_matches(parquet_path: Optional[str], *, identity: Dict[str, Any]) -> Dict[str, Any]:
    """Fail-closed check that a mined artifact exists and belongs to THIS identity."""
    if not parquet_path:
        raise HardNegError("[hardneg] no mined parquet recorded for a strategy that needs negatives")
    pq = Path(parquet_path)
    if not pq.exists():
        raise HardNegError(f"[hardneg] mined artifact missing: {parquet_path}")
    ident_path = pq.parent / "IDENTITY.json"
    if not ident_path.exists():
        raise HardNegError(
            f"[hardneg] mined artifact has no IDENTITY.json (historical/unversioned artifact "
            f"cannot be reused): {parquet_path}")
    on_disk = json.load(open(ident_path))
    if on_disk.get("schema") != ARTIFACT_SCHEMA:
        raise HardNegError(f"[hardneg] {ident_path}: wrong schema {on_disk.get('schema')!r}")
    for k in ("identity_sha256", "parent_checkpoint_sha256", "model_id", "query_variant",
              "recipe", "train_split_sha256", "catalog_sha256", "mine_split", "top_k"):
        if on_disk.get(k) != identity.get(k):
            raise HardNegError(
                f"[hardneg] STALE NEGATIVES: {parquet_path} {k}={on_disk.get(k)!r} != required "
                f"{identity.get(k)!r}. Re-mine for the current parent checkpoint.")
    return on_disk


# --------------------------------------------------------------------- negative sampling
def select_negatives(
    *,
    candidates: Sequence[str],
    gold_ids: Sequence[str],
    positive_id: str,
    strategy: str,
    n_negatives: int,
    rng: random.Random,
    fill_pool: Optional[Sequence[str]] = None,
    text_of: Optional[Any] = None,
    positive_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Pick exactly ``n_negatives`` usable negatives for one training example.

    Never returns the positive or any accepted gold. Order of preference:
      1. ids inside the strategy's rank band,
      2. ids elsewhere in the mined top-K,
      3. deterministic fill from ``fill_pool`` (valid non-gold catalog entries).
    The number taken from each source is reported so the fallback is auditable.

    If ``text_of`` is supplied, uniqueness is enforced on the RENDERED TEXT as well as the
    id: two distinct CDE ids that render identically count once, and any candidate whose
    rendering is empty or equals ``positive_text`` is rejected. This matters because the
    trainer encodes texts, not ids -- selecting 10 distinct ids that collapse to 9 distinct
    texts would otherwise under-fill the example.
    """
    if strategy in EXCLUDED_STRATEGIES:
        raise HardNegError(f"[hardneg] strategy {strategy!r} is excluded from the paper route")
    if strategy not in STRATEGY_BANDS:
        raise HardNegError(f"[hardneg] unknown strategy {strategy!r}")
    band = STRATEGY_BANDS[strategy]
    if band is None or n_negatives <= 0:
        return {"negatives": [], "n_from_band": 0, "n_from_topk": 0, "n_from_fill": 0,
                "fallback_used": False}

    blocked = {str(x) for x in gold_ids if str(x).strip()}
    if str(positive_id).strip():
        blocked.add(str(positive_id))  # unconditional: a positive is never a negative

    pos_text = str(positive_text).strip() if positive_text is not None else None
    picked: List[str] = []
    picked_texts: List[str] = []
    seen_ids: set = set()
    seen_texts: set = set()

    def _accept(cid) -> bool:
        """Try to accept one candidate id; enforces id AND rendered-text uniqueness."""
        cid = str(cid)
        if not cid or cid in blocked or cid in seen_ids:
            return False
        txt = None
        if text_of is not None:
            txt = str(text_of(cid) or "").strip()
            # A blank rendering is unusable, and a rendering equal to the positive would
            # silently turn the positive into its own negative.
            if not txt or (pos_text is not None and txt == pos_text) or txt in seen_texts:
                return False
            seen_texts.add(txt)
            picked_texts.append(txt)
        seen_ids.add(cid)
        picked.append(cid)
        return True

    def _clean_ids(seq):
        out, seen = [], set()
        for c in seq:
            c = str(c)
            if not c or c in blocked or c in seen:
                continue
            seen.add(c)
            out.append(c)
        return out

    lo, hi = band
    in_band = _clean_ids(list(candidates)[max(lo - 1, 0):hi])
    rng.shuffle(in_band)
    for c in in_band:
        if len(picked) >= n_negatives:
            break
        _accept(c)
    n_band = len(picked)

    if len(picked) < n_negatives:
        rest = [c for c in _clean_ids(candidates) if c not in seen_ids]
        rng.shuffle(rest)
        for c in rest:
            if len(picked) >= n_negatives:
                break
            _accept(c)
    n_topk = len(picked) - n_band

    if len(picked) < n_negatives:
        pool = [c for c in _clean_ids(fill_pool or []) if c not in seen_ids]
        pool.sort()                      # deterministic base order...
        rng.shuffle(pool)                # ...then a per-example deterministic shuffle
        for c in pool:
            if len(picked) >= n_negatives:
                break
            _accept(c)
    n_fill = len(picked) - n_band - n_topk

    if len(picked) < n_negatives:
        raise HardNegError(
            f"[hardneg] could not assemble {n_negatives} usable negatives for positive "
            f"{positive_id!r} (got {len(picked)}); refusing to pad with a duplicate or a positive")
    return {"negatives": picked[:n_negatives],
            "negative_texts": picked_texts[:n_negatives] if text_of is not None else None,
            "n_from_band": n_band, "n_from_topk": n_topk,
            "n_from_fill": n_fill, "fallback_used": (n_topk + n_fill) > 0}


def example_rng(*, seed: int, epoch: int, index: int) -> random.Random:
    """Deterministic per-example RNG: identical for identical (seed, epoch, index)."""
    return random.Random(int(seed) + 1000003 * int(epoch) + 9176 * int(index))


# --------------------------------------------------------------------- mining driver
def mine(
    *,
    identity: Dict[str, Any],
    root: str,
    splits_dir: str,
    catalog_path: str,
    recipe_configs: Optional[Dict[str, Any]] = None,
    sep: str = " | ",
    block_size: int = 2048,
    encode_batch_size: int = 64,
    device: str = "auto",
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Mine the top-K candidate lists for one (model, parent, representation) identity.

    Safe resume: if a matching artifact already exists it is returned unchanged. Refuses to
    overwrite a non-matching artifact unless ``overwrite`` is set.
    """
    import numpy as np
    import pandas as pd

    from demap_repro.biencoder.engine import baseline_grid as bg
    from demap_repro.text.normalize import normalize_query_text

    split = identity["mine_split"]
    if split not in ALLOWED_MINE_SPLITS:
        raise HardNegError(
            f"[hardneg] split {split!r} not allow-listed; allowed {list(ALLOWED_MINE_SPLITS)}")

    paths = artifact_paths(root, identity)
    if os.path.exists(paths["parquet"]):
        try:
            assert_artifact_matches(paths["parquet"], identity=identity)
            return {"status": "exists_matching", **paths, "identity": identity}
        except SystemExit:
            if not overwrite:
                raise HardNegError(
                    f"[hardneg] refusing to overwrite a non-matching artifact at "
                    f"{paths['parquet']}. Pass overwrite=True only if you intend to replace it.")

    split_path = Path(splits_dir) / f"{split}.parquet"
    if not split_path.exists():
        raise HardNegError(f"[hardneg] split file not found: {split_path}")
    got = sha256_file(str(split_path))
    if got != identity["train_split_sha256"]:
        raise HardNegError(
            f"[hardneg] {split} split hash {got} != identity {identity['train_split_sha256']}")
    cat_hash = sha256_file(catalog_path)
    if cat_hash != identity["catalog_sha256"]:
        raise HardNegError(
            f"[hardneg] catalog hash {cat_hash} != identity {identity['catalog_sha256']}")

    query_col = bg.QUERY_VARIANT_TO_COL.get(identity["query_variant"], identity["query_variant"])
    df = pd.read_parquet(split_path)
    if query_col not in df.columns:
        raise HardNegError(f"[hardneg] split missing query column {query_col!r}")
    df = df.copy()
    df["cde_id"] = df["cde_id"].astype(str)
    df["_qt"] = df[query_col].fillna("").astype(str).map(str.strip)
    df = df[df["_qt"] != ""]
    df["query_key"] = df["_qt"].map(normalize_query_text)

    gold = (df.groupby("query_key")["cde_id"]
              .apply(lambda s: sorted({str(x) for x in s if str(x).strip()}))
              .reset_index().rename(columns={"cde_id": "gold_cde_ids"}))
    rep = (df.groupby("query_key")["_qt"].first().reset_index().rename(columns={"_qt": "query_text"}))
    uq = gold.merge(rep, on="query_key", how="left")

    if dry_run:
        return {"status": "dry_run", **paths, "identity": identity,
                "n_unique_query_keys": int(len(uq)), "n_rows_in_split": int(len(df))}

    master = pd.read_parquet(catalog_path)
    model = _load_model(identity["parent_checkpoint"], device=device)
    got_ck = checkpoint_hash(identity["parent_checkpoint"])
    if got_ck != identity["parent_checkpoint_sha256"]:
        raise HardNegError(
            f"[hardneg] parent checkpoint hash {got_ck} != identity "
            f"{identity['parent_checkpoint_sha256']}")

    from demap_repro.utils.io import ensure_dir
    emb_dir = Path("artifacts") / "embeddings"
    ensure_dir(emb_dir)
    cde_ids, cde_emb, _meta = bg._load_or_build_catalog_embeddings(
        model=model, model_slug=f"phase2hn__{identity['identity_sha256'][:16]}",
        embeddings_dir=emb_dir, master=master, recipe=identity["recipe"],
        cde_format=identity["cde_format"], sep=sep, recipe_configs=recipe_configs,
        batch_size=encode_batch_size, normalize=True)

    q_emb = model.encode(uq["query_text"].tolist(), batch_size=encode_batch_size,
                         convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
    idx_top, score_top = bg._topk_blockwise(query_emb=np.asarray(q_emb), cde_emb=cde_emb,
                                            top_k=int(identity["top_k"]), block_size=block_size)
    ids_list = np.asarray(cde_ids).astype(str).tolist()
    out = uq.copy()
    out["candidates_cde_id"] = [[ids_list[int(j)] if int(j) >= 0 else "" for j in row]
                               for row in idx_top.tolist()]
    out["candidates_score"] = [[float(x) for x in row] for row in score_top.tolist()]
    out["candidates_rank"] = [list(range(1, int(identity["top_k"]) + 1)) for _ in range(len(out))]
    out["miner_model_name_or_path"] = identity["parent_checkpoint"]
    out["parent_checkpoint_sha256"] = identity["parent_checkpoint_sha256"]

    ensure_dir(Path(paths["dir"]))
    out.to_parquet(paths["parquet"], index=False)
    with open(paths["identity_json"], "w") as f:
        json.dump(identity, f, indent=1, sort_keys=True)
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "identity": identity,
        "n_unique_query_keys": int(len(uq)),
        "n_rows_in_split": int(len(df)),
        "parquet": paths["parquet"],
        "parquet_sha256": sha256_file(paths["parquet"]),
        "query_col": query_col,
        "block_size": block_size, "encode_batch_size": encode_batch_size,
    }
    with open(paths["manifest_json"], "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return {"status": "mined", **paths, "identity": identity, "manifest": manifest}


def _load_model(path: str, *, device: str = "auto"):
    from demap_repro.biencoder.engine.st_loader import load_sentence_transformer

    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    # Honor the same fail-fast contract as Phase 2 training: mining embeds the
    # full catalog with a neural model, so a silent CPU fallback is an error
    # when the launcher exported DEMAP_REQUIRE_GPU=1 (the GPU sbatch does).
    if os.environ.get("DEMAP_REQUIRE_GPU") == "1" and not str(device).startswith("cuda"):
        raise RuntimeError(
            f"DEMAP_REQUIRE_GPU=1 but hard-negative mining resolved device={device!r}; "
            f"refusing to run the parent-model catalog encode on CPU")
    return load_sentence_transformer(path, device=device)
