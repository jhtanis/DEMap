#!/usr/bin/env python3
"""Deep bi-encoder retrieval: top-1000 CDE candidates per eval-split query.

Reuses the EXISTING building blocks so retrieval matches final_eval exactly:
  - the selected Phase-2 natural winner model (a full SentenceTransformer dir),
  - its ``run_config.json`` catalog settings (recipe / cde_format / sep /
    recipe_configs / query_variant / normalize) — so the CDE catalog is built
    identically to final-eval,
  - ``demap_repro.text.recipes.build_catalog`` + ``demap_repro.biencoder.engine.st_loader`` +
    ``sentence_transformers.util.semantic_search`` for encode + cosine top-K.

Emits a long ranking table (split, query_id, cde_id, biencoder_rank,
biencoder_score, query_text, is_label) plus a recall summary.

Catalog embeddings (the expensive part — ~63k CDEs, ~2h on CPU / ~2min on GPU) are
**cached** and reused. On each run the script:
  1. looks for the vectors in a documented canonical cache dir (``--embeddings-cache-dir``,
     default ``artifacts/final_reranker/biencoder_deep/catalog_embeddings``; never /tmp,
     never artifacts_v3_cdisc) and loads them if present (skipping encode);
  2. on a miss, computes them via the fallback ladder Slurm-GPU → local-GPU → local-CPU
     (the CPU path prints a prominent ~2h warning), saving them + a manifest to the cache.
Retrieval output is identical whether the vectors were loaded or recomputed. See
``demap_repro.biencoder.catalog_embedding_cache``.

Usage:
    # normal retrieval (loads cached catalog vectors if present, else computes+caches)
    PYTHONPATH=src .venv/bin/python scripts/retrieve_biencoder_deep.py \
        --run-dir <phase2 winner run dir> --top-k 1000
    # isolated catalog-encode-and-cache only (what the Slurm fallback submits)
    PYTHONPATH=src .venv/bin/python scripts/retrieve_biencoder_deep.py \
        --run-dir <...> --encode-only --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demap_repro.text.recipes import build_catalog  # noqa: E402
from demap_repro.biencoder.engine.st_loader import load_sentence_transformer  # noqa: E402
from demap_repro.lexical.bm25.engine import QUERY_VARIANT_TO_COL  # noqa: E402
from demap_repro.biencoder.engine.baseline_grid import _encode_texts  # noqa: E402
from demap_repro.biencoder.catalog_embedding_cache import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    ensure_catalog_embeddings,
    local_gpu_available,
    slurm_available,
)

DEFAULT_RUN_DIR = (REPO_ROOT / "artifacts_v3_cdisc/phase2/natural/sentence-transformers__all-mpnet-base-v2/runs/"
                   "20260525_105723__v3_phase2_natural__ft2__20260523_173356__v3_phase1_natural__ft__"
                   "sentence-transformer-c7df217f__Q3__v1_v2_v3_v5__labeled__R0__symmetric_mnrl__"
                   "lr7e-05__bs11__t0.06__ep1__seed-fd7dc27022")
SPLITS_DIR = REPO_ROOT / "data/processed/splits_v3_cdisc"
CDE_MASTER = REPO_ROOT / "data/processed/cde_master_enriched.parquet"
OUT_DIR = REPO_ROOT / "artifacts_v3_cdisc/biencoder_deep_top1000"
DEFAULT_SPLITS = ["val_train", "val_dev", "test", "external_holdout_org",
                  "external_holdout_gdc_altnames", "external_holdout_gdc_questiontext",
                  "cimac_appendix_a_eval"]
RECALL_KS = [20, 100, 200, 500, 1000]


def set_safe_tempdir():
    # Biowulf rule: never /tmp (shared). Prefer per-job node-local /lscratch when allocated;
    # otherwise a repo-local .scratch dir. NEVER artifacts_v3_cdisc (frozen legacy tree).
    job = os.environ.get("SLURM_JOB_ID")
    if job and Path(f"/lscratch/{job}").is_dir():
        cand = Path(f"/lscratch/{job}")
    else:
        cand = REPO_ROOT / ".scratch/demap/retrieve_biencoder_deep/tmp"
    Path(cand).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(cand)
    os.environ["TMPDIR"] = str(cand)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(cand)


def _slurm_encode_and_wait(args, paths, t0):
    """Submit an isolated Slurm GPU job running this script with --encode-only, wait for it.

    Keeps the catalog-encode isolated so later stages (and this same run, on the poll below)
    just load the saved embeddings. Writes the sbatch under the cache dir (never /tmp).
    Returns (emb, model_max_seq_length) once the cache file appears; raises on timeout/failure.
    """
    import subprocess
    import numpy as _np

    job_dir = Path(paths.meta).parent
    job_dir.mkdir(parents=True, exist_ok=True)
    sbatch_path = job_dir / "encode_catalog.sbatch"
    log_path = job_dir / "encode_catalog.%j.log"
    py = sys.executable
    script = str(Path(__file__).resolve())
    # Re-invoke this script in --encode-only mode on the GPU node. --no-slurm so the job
    # itself uses its local GPU (no recursive submission).
    cmd = (
        f'PYTHONPATH="{REPO_ROOT}/src" "{py}" "{script}" --encode-only --no-slurm '
        f'--run-dir "{args.run_dir}" --cde-master "{args.cde_master}" '
        f'--embeddings-cache-dir "{args.embeddings_cache_dir}" '
        f'--batch-size {args.batch_size} --device cuda'
    )
    sbatch_path.write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH --job-name=cde_catalog_encode\n"
        f"#SBATCH --partition={args.slurm_encode_partition}\n"
        f"#SBATCH --gres={args.slurm_encode_gres}\n"
        f"#SBATCH --time={args.slurm_encode_time}\n"
        f"#SBATCH --cpus-per-task=8\n"
        f"#SBATCH --mem=48g\n"
        f"#SBATCH --output={log_path}\n\n"
        f"set -euo pipefail\ncd \"{REPO_ROOT}\"\n{cmd}\n"
    )
    print(f"[slurm] submitting catalog-encode job: {sbatch_path}")
    try:
        jid = subprocess.check_output(["sbatch", "--parsable", str(sbatch_path)], text=True).strip()
    except Exception as e:
        raise RuntimeError(f"sbatch submission failed: {e}")
    print(f"[slurm] submitted job {jid}; polling up to {args.slurm_poll_secs}s for {paths.emb}")

    waited = 0
    poll = 15
    while waited < args.slurm_poll_secs:
        if paths.all_exist():
            emb = _np.load(paths.emb, allow_pickle=False)
            try:
                msl = json.loads(paths.meta.read_text()).get("model_max_seq_length")
            except Exception:
                msl = None
            print(f"[slurm] job {jid} produced embeddings after ~{waited}s")
            return emb, msl
        # if the job has left the queue without producing output, stop waiting
        try:
            q = subprocess.run(["squeue", "-j", jid, "-h"], capture_output=True, text=True)
            if q.returncode == 0 and not q.stdout.strip() and not paths.all_exist():
                # give the filesystem a moment, then check once more
                time.sleep(poll)
                if paths.all_exist():
                    emb = _np.load(paths.emb, allow_pickle=False)
                    return emb, None
                raise RuntimeError(f"slurm job {jid} finished but no embeddings at {paths.emb}")
        except FileNotFoundError:
            pass  # squeue not present; keep polling on the file
        time.sleep(poll)
        waited += poll
    raise RuntimeError(f"timed out after {args.slurm_poll_secs}s waiting for slurm job {jid}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Phase-2 natural winner run dir")
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    ap.add_argument("--splits-dir", default=str(SPLITS_DIR))
    ap.add_argument("--cde-master", default=str(CDE_MASTER))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--query-chunk", type=int, default=256)
    ap.add_argument("--device", default="cuda",
                    help="preferred device for LIVE encode (query encode + local catalog "
                         "encode). 'cuda' falls back to CPU automatically if no GPU is visible.")
    ap.add_argument("--embeddings-cache-dir", default=str(DEFAULT_CACHE_DIR),
                    help="canonical cache dir for catalog embeddings (never /tmp, never "
                         "artifacts_v3_cdisc). Loaded if present; else computed + saved here.")
    ap.add_argument("--no-slurm", action="store_true",
                    help="disable the Slurm-GPU catalog-encode fallback (use local GPU/CPU only).")
    ap.add_argument("--slurm-encode-time", default="00:30:00",
                    help="wall-time for the Slurm catalog-encode job (fallback path).")
    ap.add_argument("--slurm-encode-partition", default="gpu",
                    help="Slurm partition for the catalog-encode job.")
    ap.add_argument("--slurm-encode-gres", default="gpu:a100:1",
                    help="Slurm --gres for the catalog-encode job.")
    ap.add_argument("--slurm-poll-secs", type=int, default=1800,
                    help="max seconds to wait for the Slurm catalog-encode job to finish.")
    ap.add_argument("--encode-only", action="store_true",
                    help="build + cache the catalog embeddings, then exit (no retrieval). "
                         "This is the isolated stage the Slurm fallback submits.")
    args = ap.parse_args(argv)
    set_safe_tempdir()
    t0 = time.time()

    run_dir = Path(args.run_dir)
    model_dir = run_dir / "model"
    cfg = json.loads((run_dir / "run_config.json").read_text())
    qv = str(cfg.get("query_variant", "Q3"))
    recipe = str(cfg.get("recipe", "v1_v2_v3_v5"))
    cde_format = str(cfg.get("cde_format", "labeled"))
    sep = str(cfg.get("sep", " | "))
    recipe_configs = cfg.get("recipe_configs") if isinstance(cfg.get("recipe_configs"), dict) else None
    normalize = bool(cfg.get("normalize_embeddings", True))
    query_col = QUERY_VARIANT_TO_COL.get(qv, "query_text_q3")
    print(f"[cfg] model={model_dir}\n[cfg] qv={qv} col={query_col} recipe={recipe} "
          f"fmt={cde_format} normalize={normalize}")

    # --- build catalog identically to final-eval ---
    master = pd.read_parquet(args.cde_master)
    catalog = build_catalog(master, recipe=recipe, cde_format=cde_format, sep=sep,
                            recipe_configs=recipe_configs)
    cat_ids = catalog["cde_id"].astype(str).to_numpy()
    cat_texts = catalog["cde_text"].astype(str).tolist()
    print(f"[catalog] {len(cat_ids)} CDEs")

    from sentence_transformers import util
    import torch

    # Live device for query-encode (and local catalog-encode): honor --device but degrade
    # 'cuda' -> 'cpu' when no GPU is present, so the script never crashes on a CPU node.
    gpu_here = local_gpu_available()
    live_device = "cuda" if (gpu_here and str(args.device) != "cpu") else "cpu"
    model = load_sentence_transformer(str(model_dir), device=live_device)
    print(f"[model] loaded in {time.time()-t0:.0f}s (device={live_device}, gpu_here={gpu_here})")

    def _local_encode_catalog():
        """Encode the catalog with the already-loaded model on the live device."""
        which = "local_gpu" if live_device == "cuda" else "local_cpu"
        print(f"[catalog] encoding {len(cat_texts)} CDEs on {live_device} ...")
        emb = _encode_texts(model, cat_texts, batch_size=args.batch_size, normalize=normalize)
        return emb, getattr(model, "max_seq_length", None), which

    def _submit_slurm_encode(paths):
        """Submit an isolated Slurm GPU job that runs THIS script with --encode-only,
        then poll for the cache file. Returns (emb, model_max_seq_length)."""
        return _slurm_encode_and_wait(args, paths, t0)

    outcome = ensure_catalog_embeddings(
        cde_ids=cat_ids,
        cde_texts=cat_texts,
        model_dir=str(model_dir),
        run_dir=str(run_dir),
        cde_master=str(args.cde_master),
        recipe=recipe,
        cde_format=cde_format,
        sep=sep,
        recipe_configs=recipe_configs,
        normalize=normalize,
        cache_dir=args.embeddings_cache_dir,
        allow_slurm=not args.no_slurm,
        encode_fn=_local_encode_catalog,
        slurm_submit_fn=_submit_slurm_encode,
    )
    print(f"[catalog] embeddings {outcome.source}"
          + (f" (backend={outcome.backend})" if outcome.backend else "")
          + f"; path={outcome.paths.emb}  ({time.time()-t0:.0f}s)")
    cat_ids = np.asarray([str(x) for x in outcome.ids])
    cat_emb = torch.as_tensor(np.asarray(outcome.emb, dtype=np.float32), device=live_device)

    if args.encode_only:
        print(f"[encode-only] catalog embeddings ready at {outcome.paths.emb}; exiting "
              f"without retrieval ({time.time()-t0:.0f}s)")
        return 0

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    parts, summary = [], []
    for split in splits:
        ts = time.time()
        p = Path(args.splits_dir) / f"{split}.parquet"
        if not p.exists():
            print(f"  WARN: split missing, skipped: {p}"); continue
        s = pd.read_parquet(p, columns=["query_id", query_col, "cde_id"])
        s["query_id"] = s["query_id"].astype(str); s["cde_id"] = s["cde_id"].astype(str)
        gmap = s.groupby("query_id")["cde_id"].apply(set).to_dict()
        q = s.drop_duplicates("query_id")[["query_id", query_col]].reset_index(drop=True)
        qtexts = q[query_col].fillna("").astype(str).tolist()
        q_emb = torch.as_tensor(
            _encode_texts(model, qtexts, batch_size=args.batch_size, normalize=normalize),
            device=live_device,
        )
        hits = util.semantic_search(q_emb, cat_emb, top_k=args.top_k,
                                    query_chunk_size=args.query_chunk)
        rows = []
        for i, hlist in enumerate(hits):
            qid = q["query_id"].iloc[i]; qtx = qtexts[i]; gold = gmap.get(qid, set())
            for rank, h in enumerate(hlist, 1):
                cid = str(cat_ids[h["corpus_id"]])
                rows.append((split, qid, cid, rank, float(h["score"]), qtx, cid in gold))
        df = pd.DataFrame(rows, columns=["split", "query_id", "cde_id", "biencoder_rank",
                                         "biencoder_score", "query_text", "is_label"])
        parts.append(df)
        # recall summary (deployment-safe: total queries = nunique in split)
        nq = s["query_id"].nunique()
        gr = (df[df["is_label"]].groupby("query_id")["biencoder_rank"].min())
        rec = {"split": split, "n_queries": nq, "n_rows": len(df),
               "gold_in_top{}".format(args.top_k): int((gr <= args.top_k).sum())}
        for k in RECALL_KS:
            rec[f"recall@{k}"] = float((gr <= k).sum()) / nq
        rec["secs"] = round(time.time() - ts, 1)
        summary.append(rec)
        print(f"  [{split}] q={nq} rows={len(df)} R@20={rec['recall@20']:.3f} "
              f"R@100={rec['recall@100']:.3f} R@1000={rec[f'recall@1000']:.3f} ({time.time()-ts:.0f}s)")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    allrk = pd.concat(parts, ignore_index=True)
    assert not allrk.duplicated(["split", "query_id", "cde_id"]).any(), "duplicate (split,query_id,cde_id)"
    allrk.to_parquet(out_dir / "biencoder_deep_rankings_top1000.parquet", index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / "recall_summary.csv", index=False)

    lines = ["# Deep bi-encoder retrieval (Phase-2 natural winner), top-1000", "",
             f"Model run dir: `{run_dir.name}`  |  catalog: {len(cat_ids)} CDEs  |  "
             f"recipe={recipe}/{cde_format}, query={query_col}, normalize={normalize}", "",
             "| split | n_queries | n_rows | " + " | ".join(f"R@{k}" for k in RECALL_KS) +
             f" | gold_in_top{args.top_k} |",
             "|" + "---|" * (len(RECALL_KS) + 4)]
    for _, r in sdf.iterrows():
        lines.append(f"| {r['split']} | {int(r['n_queries'])} | {int(r['n_rows'])} | " +
                     " | ".join(f"{r[f'recall@{k}']:.4f}" for k in RECALL_KS) +
                     f" | {int(r[f'gold_in_top{args.top_k}'])} |")
    lines += ["", f"Total runtime {time.time()-t0:.0f}s. "
              "Sanity check vs existing top-20 rankings recommended (top-20 prefix should match)."]
    (out_dir / "summary.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_dir}/biencoder_deep_rankings_top1000.parquet, recall_summary.csv, summary.md "
          f"in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
