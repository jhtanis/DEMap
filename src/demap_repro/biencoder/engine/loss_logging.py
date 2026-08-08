from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


class LossLoggingWrapper(nn.Module):
    """Wrap a training loss module and persist per-step loss history.

    The wrapper is backend-agnostic: it simply records one row each time the
    wrapped loss is called. For the SentenceTransformers training APIs used in
    this repo, that corresponds closely to one optimizer/training step.
    """

    def __init__(
        self,
        inner_loss: nn.Module,
        *,
        run_dir: str | Path,
        configured_lr: Optional[float] = None,
        steps_per_epoch: Optional[int] = None,
        jsonl_name: str = "train_loss_history.jsonl",
        csv_name: str = "train_loss_history.csv",
    ) -> None:
        super().__init__()
        self.inner_loss = inner_loss
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.configured_lr = None if configured_lr is None else float(configured_lr)
        self.steps_per_epoch = None if steps_per_epoch is None else max(int(steps_per_epoch), 1)
        self.global_step = 0
        self.jsonl_path = self.run_dir / jsonl_name
        self.csv_path = self.run_dir / csv_name
        self._jsonl_fh = self.jsonl_path.open("a", encoding="utf-8")
        self._csv_fh = self.csv_path.open("a", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_fh,
            fieldnames=["global_step", "epoch", "step_in_epoch", "loss", "lr", "timestamp"],
        )
        if self.csv_path.stat().st_size == 0:
            self._csv_writer.writeheader()
            self._csv_fh.flush()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner_loss, name)

    def _loss_scalar(self, loss_obj: Any) -> float:
        if isinstance(loss_obj, torch.Tensor):
            try:
                return float(loss_obj.detach().cpu().item())
            except Exception:
                return float(loss_obj.detach().cpu().reshape(-1)[0].item())
        return float(loss_obj)

    def _record(self, loss_value: float) -> None:
        self.global_step += 1
        epoch = None
        step_in_epoch = None
        if self.steps_per_epoch is not None and self.steps_per_epoch > 0:
            epoch = int((self.global_step - 1) // self.steps_per_epoch) + 1
            step_in_epoch = int((self.global_step - 1) % self.steps_per_epoch) + 1
        row: Dict[str, Any] = {
            "global_step": int(self.global_step),
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "loss": float(loss_value),
            "lr": self.configured_lr,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._jsonl_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._jsonl_fh.flush()
        self._csv_writer.writerow(row)
        self._csv_fh.flush()

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        loss_obj = self.inner_loss(*args, **kwargs)
        try:
            self._record(self._loss_scalar(loss_obj))
        except Exception:
            pass
        return loss_obj

    def close(self) -> None:
        for fh in [self._jsonl_fh, self._csv_fh]:
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass
