#!/usr/bin/env python3
"""Fine-tune a sentence_transformers.CrossEncoder for CDE candidate reranking.

Pointwise binary objective (BinaryCrossEntropyLoss) on (query, candidate) pairs
from ``build_ce_training_pairs.py``, using the native ST 5.x ``CrossEncoderTrainer``.
After each epoch the model is evaluated on the val_dev deployable union with the
EXISTING deployment-safe metric logic (``train_hgbc_reranker._deployment_metrics``);
the best checkpoint by val_dev Recall@5 is saved.

Offline-safe (HF cache only) unless ``--allow-download``. CPU dry-run:
    PYTHONPATH=src .venv/bin/python scripts/finetune_crossencoder.py \
        --train-pairs artifacts_v3_cdisc/crossencoder/finetune/data/train_SN_DEC_DEF_PQT_PV.parquet \
        --dev-pairs   artifacts_v3_cdisc/crossencoder/finetune/data/dev_SN_DEC_DEF_PQT_PV.parquet \
        --device cpu --limit-queries 200 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from demap_repro.utils.paths import data_root

REPO_ROOT = data_root()

from demap_repro.crossencoder import score as sc
from demap_repro.reranker import train as thr

DEFAULT_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_OUT_ROOT = REPO_ROOT / "artifacts_v3_cdisc/crossencoder/finetune"
_METRIC_KEYS = ["recall@1", "recall@5", "recall@10", "mrr@100", "n_queries"]


def _limit(df: pd.DataFrame, n: int) -> pd.DataFrame:
    qs = sorted(df["query_id"].astype(str).unique())[:n]
    return df[df["query_id"].astype(str).isin(qs)].reset_index(drop=True)


def _dev_metrics(model, dev_df: pd.DataFrame, dev_pairs, batch_size: int) -> dict:
    scores = model.predict(dev_pairs, batch_size=batch_size,
                           convert_to_numpy=True, show_progress_bar=False)
    d = dev_df[["query_id", "is_label", "is_injected_gold"]].copy()
    d["_s"] = np.asarray(scores, dtype=float).reshape(-1)
    return thr._deployment_metrics(d, "_s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--train-pairs", required=True)
    ap.add_argument("--dev-pairs", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="BCE positive weight; default = n_neg/n_pos on (limited) train")
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="restrict train AND dev to the first N query_ids (CPU dry-run)")
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260527)
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args(argv)

    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sc._set_safe_tempdir()

    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    from datasets import Dataset
    from transformers import TrainerCallback
    from sentence_transformers import (
        CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    train = pd.read_parquet(args.train_pairs)
    dev = pd.read_parquet(args.dev_pairs)
    if args.limit_queries:
        train = _limit(train, args.limit_queries)
        dev = _limit(dev, args.limit_queries)
    train["is_label"] = train["label"] > 0.5
    dev["is_label"] = dev["label"] > 0.5
    dev["is_injected_gold"] = dev["is_injected_gold"].astype(bool)

    n_pos = int((train["label"] == 1.0).sum())
    n_neg = int((train["label"] == 0.0).sum())
    pos_weight = args.pos_weight if args.pos_weight is not None else (n_neg / max(1, n_pos))
    print(f"base={args.base_model} device={args.device} epochs={args.epochs} "
          f"lr={args.lr} bs={args.batch_size} max_len={args.max_length}")
    print(f"train: {len(train)} rows ({n_pos} pos / {n_neg} neg)  pos_weight={pos_weight:.2f}")
    print(f"dev:   {len(dev)} rows  {dev['query_id'].nunique()} queries")

    out_dir = Path(args.output_dir) if args.output_dir else \
        DEFAULT_OUT_ROOT / args.base_model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = CrossEncoder(args.base_model, num_labels=1,
                             max_length=args.max_length, device=args.device)
    except Exception as e:
        print(f"\nERROR loading base model (offline={'no' if args.allow_download else 'yes'}): "
              f"{type(e).__name__}: {e}")
        print("Pre-cache it on a login node:\n  " + sc._precache_cmd(args.base_model))
        return 2

    train_ds = Dataset.from_dict({
        "query": train["query_text"].astype(str).tolist(),
        "candidate": train["candidate_text"].astype(str).tolist(),
        "label": train["label"].astype(float).tolist(),
    })
    loss = BinaryCrossEntropyLoss(
        model, pos_weight=torch.tensor(pos_weight, dtype=torch.float))

    dev_pairs = list(zip(dev["query_text"].astype(str), dev["candidate_text"].astype(str)))

    history = []
    m0 = _dev_metrics(model, dev, dev_pairs, args.batch_size)
    print(f"epoch 0 (pre-FT) dev: R@1={m0['recall@1']:.4f} R@5={m0['recall@5']:.4f} "
          f"R@10={m0['recall@10']:.4f} MRR={m0['mrr@100']:.4f}")
    history.append({"epoch": 0, "train_loss": None, **{k: m0[k] for k in _METRIC_KEYS}})

    state = {"best_r5": -1.0, "best_epoch": -1, "epoch": 0, "last_loss": None}

    class DevEval(TrainerCallback):
        def on_log(self, a, st, ctrl, logs=None, **kw):
            if logs and "loss" in logs:
                state["last_loss"] = float(logs["loss"])

        def on_epoch_end(self, a, st, ctrl, **kw):
            state["epoch"] += 1
            m = _dev_metrics(model, dev, dev_pairs, args.batch_size)
            print(f"epoch {state['epoch']} train_loss="
                  f"{state['last_loss'] if state['last_loss'] is not None else float('nan'):.4f} "
                  f"dev: R@1={m['recall@1']:.4f} R@5={m['recall@5']:.4f} "
                  f"R@10={m['recall@10']:.4f} MRR={m['mrr@100']:.4f}")
            history.append({"epoch": state["epoch"], "train_loss": state["last_loss"],
                            **{k: m[k] for k in _METRIC_KEYS}})
            if m["recall@5"] > state["best_r5"]:
                state["best_r5"], state["best_epoch"] = m["recall@5"], state["epoch"]
                model.save(str(out_dir))
                print(f"  -> new best val_dev R@5={state['best_r5']:.4f}; saved -> {out_dir}")

    hf_out = os.path.join(tempfile.gettempdir(), "ce_hf_trainer")
    targs = CrossEncoderTrainingArguments(
        output_dir=hf_out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_frac,
        fp16=args.device.startswith("cuda"),
        logging_steps=20,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=args.seed,
        disable_tqdm=True,
        dataloader_num_workers=0,
    )
    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=train_ds,
                                  loss=loss, callbacks=[DevEval()])
    trainer.train()

    summary = {
        "base_model": args.base_model, "train_pairs": args.train_pairs,
        "dev_pairs": args.dev_pairs, "output_dir": str(out_dir),
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "max_length": args.max_length, "pos_weight": pos_weight,
        "limit_queries": args.limit_queries, "seed": args.seed,
        "n_train_rows": len(train), "n_train_pos": n_pos, "n_train_neg": n_neg,
        "n_dev_rows": len(dev), "n_dev_queries": int(dev["query_id"].nunique()),
        "best_epoch": state["best_epoch"], "best_val_dev_recall@5": state["best_r5"],
        "history": history,
    }
    if state["best_epoch"] < 0:  # never improved over pre-FT: still save final
        model.save(str(out_dir))
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nbest epoch={state['best_epoch']} val_dev R@5={state['best_r5']:.4f}")
    print(f"Wrote checkpoint + training_summary.json under: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
