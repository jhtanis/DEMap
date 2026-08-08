"""Opt-in modern SentenceTransformerTrainer path for the Section 4.3 precision validation.

This is a narrow, additive alternative to the legacy ``old_fit`` training call. It runs the
*same* data, loss, no-duplicate batching, optimizer (AdamW), linear scheduler, warmup,
per-epoch MRR@100 checkpoint selection, and final-metric computation, but through
``SentenceTransformerTrainer`` so that ``fp16``/``bf16`` mixed precision is honestly honored
by the framework (unlike ``old_fit``, whose ``torch.cuda.amp.autocast()`` is float16-only).

Precision is set via ``SentenceTransformerTrainingArguments(fp16=..., bf16=...)``. GradScaler
status is read back from the accelerator and asserted (fp16 -> enabled scaler; bf16 -> none).
The caller's ``TrainingPrecisionVerifier`` hooks observe the real training-forward dtype.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def build_modern_training_args(
    *,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    seed: int,
    requested_bf16: bool,
    requested_fp16: bool,
    weight_decay: float = 0.01,
    max_train_steps: Optional[int] = None,
):
    """Build SentenceTransformerTrainingArguments. Precision is the ONLY field that differs
    between the fp16 and bf16 conditions; everything else is fixed by the inputs.

    Uses an explicit integer ``warmup_steps`` (not the transformers-v5-deprecated
    ``warmup_ratio``) so warmup is applied unambiguously and identically to both conditions.
    ``weight_decay`` defaults to 0.01 to match the historical old_fit optimizer.
    """
    from sentence_transformers import SentenceTransformerTrainingArguments  # type: ignore
    from sentence_transformers.training_args import BatchSamplers  # type: ignore

    return SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(epochs),
        per_device_train_batch_size=int(batch_size),
        learning_rate=float(lr),
        warmup_steps=int(warmup_steps),
        weight_decay=float(weight_decay),
        fp16=bool(requested_fp16),
        bf16=bool(requested_bf16),
        eval_strategy="no",
        save_strategy="no",  # full ST models are snapshotted per epoch via a callback
        logging_steps=50,
        report_to=[],
        seed=int(seed),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_drop_last=False,
        max_steps=(int(max_train_steps) if (max_train_steps and int(max_train_steps) > 0) else -1),
    )


def assert_scaler_matches_precision(
    *,
    requested_bf16: bool,
    requested_fp16: bool,
    present: Optional[bool],
    enabled: Optional[bool],
    mixed_precision: Optional[str] = None,
) -> None:
    """Fail-closed: fp16 requires an enabled GradScaler; bf16 must not have one enabled."""
    if requested_fp16 and not (bool(present) and bool(enabled)):
        raise SystemExit(
            f"[precision-verify] fp16_mixed requires an enabled GradScaler, but "
            f"present={present} enabled={enabled} (mixed_precision={mixed_precision})."
        )
    if requested_bf16 and bool(present) and bool(enabled):
        raise SystemExit(
            f"[precision-verify] bf16_mixed must not use an enabled GradScaler, but "
            f"present={present} enabled={enabled} (mixed_precision={mixed_precision})."
        )


def _scaler_status(trainer) -> Tuple[Optional[bool], Optional[bool], Optional[str]]:
    """Return (scaler_present, scaler_enabled, mixed_precision_label) from the accelerator."""
    present: Optional[bool] = None
    enabled: Optional[bool] = None
    mp: Optional[str] = None
    try:
        acc = getattr(trainer, "accelerator", None)
        if acc is not None:
            mp = getattr(acc, "mixed_precision", None)
            scaler = getattr(acc, "scaler", None)
            present = scaler is not None
            if scaler is not None:
                try:
                    enabled = bool(scaler.is_enabled())
                except Exception:
                    enabled = None
            else:
                enabled = False
    except Exception:
        pass
    return present, enabled, mp


def train_with_modern_trainer(
    *,
    model: Any,
    train_pairs: Sequence[Tuple[str, str]],
    train_loss: Any,
    compute_val_metrics: Callable[..., Dict[str, float]],
    val_df: Any,
    query_col: str,
    cde_ids: Any,
    cde_texts: Sequence[str],
    lr: float,
    batch_size: int,
    epochs: int,
    seed: int,
    temperature: float,
    warmup_ratio: float,
    weight_decay: float,
    device: Any,
    model_out_dir: Path,
    run_dir: Path,
    requested_bf16: bool,
    requested_fp16: bool,
    top_k: int,
    block_size: int,
    encode_batch_size: int,
    normalize_embeddings: bool,
    max_train_steps: Optional[int] = None,
    prec_verifier: Any = None,
) -> Dict[str, Any]:
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from sentence_transformers import (  # type: ignore
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.training_args import BatchSamplers  # type: ignore
    from transformers import TrainerCallback  # type: ignore

    requested = "bf16_mixed" if requested_bf16 else ("fp16_mixed" if requested_fp16 else "fp32")

    # Dataset: column order (anchor, positive) drives MultipleNegatives* losses.
    ds = Dataset.from_dict(
        {"query": [q for (q, _c) in train_pairs], "cde": [c for (_q, c) in train_pairs]}
    )

    snap_root = Path(run_dir) / "epoch_snapshots"
    snap_root.mkdir(parents=True, exist_ok=True)
    saved_epochs: List[Tuple[int, str]] = []

    class _EpochSnapshot(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
            n = int(round(float(getattr(state, "epoch", 0) or 0)))
            if n <= 0:
                n = len(saved_epochs) + 1
            out = snap_root / f"epoch_{n}"
            try:
                (model).save(str(out))
                saved_epochs.append((n, str(out)))
            except Exception:
                pass

    import math

    steps_per_epoch = max(1, math.ceil(len(train_pairs) / max(1, int(batch_size))))
    warmup_steps = int(steps_per_epoch * int(epochs) * float(warmup_ratio))

    ta = build_modern_training_args(
        output_dir=str(Path(run_dir) / "trainer"),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        warmup_steps=warmup_steps,
        seed=int(seed),
        requested_bf16=bool(requested_bf16),
        requested_fp16=bool(requested_fp16),
        weight_decay=float(weight_decay),
        max_train_steps=max_train_steps,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=ta,
        train_dataset=ds,
        loss=train_loss,
        callbacks=[_EpochSnapshot()],
    )

    train_output = trainer.train()

    # --- GradScaler status (fail-closed on precision/scaler disagreement) ---------------
    scaler_present, scaler_enabled, mixed_precision = _scaler_status(trainer)
    if prec_verifier is not None:
        try:
            prec_verifier.evidence["grad_scaler_present"] = scaler_present
            prec_verifier.evidence["grad_scaler_enabled"] = scaler_enabled
            prec_verifier.evidence["accelerator_mixed_precision"] = mixed_precision
            prec_verifier.evidence["trainer_class"] = "SentenceTransformerTrainer"
        except Exception:
            pass
    assert_scaler_matches_precision(
        requested_bf16=bool(requested_bf16),
        requested_fp16=bool(requested_fp16),
        present=scaler_present,
        enabled=scaler_enabled,
        mixed_precision=mixed_precision,
    )

    # --- Training-loss behaviour (NaN check) --------------------------------------------
    log_hist = list(getattr(trainer.state, "log_history", []) or [])
    losses = [float(r["loss"]) for r in log_hist if isinstance(r, dict) and "loss" in r]
    any_nan = any((l != l) or (l in (float("inf"), float("-inf"))) for l in losses)  # noqa: E741

    # --- Per-epoch MRR@100 checkpoint selection (mirrors old_fit save_best_model) --------
    per_epoch: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for (n, d) in sorted(saved_epochs, key=lambda x: x[0]):
        m = SentenceTransformer(d, device=str(device))
        metr = compute_val_metrics(
            model=m,
            val_df=val_df,
            query_col=query_col,
            cde_ids=cde_ids,
            cde_texts=cde_texts,
            top_k=max(int(top_k), 100),
            block_size=int(block_size),
            encode_batch_size=int(encode_batch_size),
            normalize_embeddings=bool(normalize_embeddings),
        )
        row = {"epoch": int(n), **{k: float(v) for k, v in metr.items() if isinstance(v, (int, float))}}
        per_epoch.append(row)
        if best is None or row.get("mrr@100", float("-inf")) > best.get("mrr@100", float("-inf")):
            best = {"epoch": int(n), "dir": d, **row}
        del m

    if best is None:
        raise SystemExit("[modern-trainer] no epoch snapshots were saved; cannot select a checkpoint.")

    # Save the selected checkpoint as the run's model (downstream final eval reloads this).
    model_out_dir = Path(model_out_dir)
    model_out_dir.mkdir(parents=True, exist_ok=True)
    SentenceTransformer(best["dir"], device=str(device)).save(str(model_out_dir))

    # Reclaim disk: snapshots are large; the selected model now lives in model_out_dir.
    try:
        shutil.rmtree(snap_root)
    except Exception:
        pass

    return {
        "trainer_class": "SentenceTransformerTrainer",
        "requested_precision": requested,
        "grad_scaler_present": scaler_present,
        "grad_scaler_enabled": scaler_enabled,
        "accelerator_mixed_precision": mixed_precision,
        "optimizer": "AdamW (SentenceTransformerTrainer default)",
        "scheduler": "linear-with-warmup (default)",
        "warmup_ratio": float(warmup_ratio),
        "warmup_steps": int(warmup_steps),
        "weight_decay": float(weight_decay),
        "steps_per_epoch_est": int(steps_per_epoch),
        "batch_sampler": "NO_DUPLICATES",
        "n_train_pairs": int(len(train_pairs)),
        "num_epochs": int(epochs),
        "selected_epoch": int(best["epoch"]),
        "selected_checkpoint_source": best["dir"],
        "selection_metric": "val_dev mrr@100",
        "per_epoch_val": per_epoch,
        "train_loss_final": (losses[-1] if losses else None),
        "train_loss_n_logged": int(len(losses)),
        "train_loss_has_nan_or_inf": bool(any_nan),
        "global_step": int(getattr(train_output, "global_step", 0) or 0),
    }
