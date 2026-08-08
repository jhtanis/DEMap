"""samplers.py

Batch sampling utilities for contrastive retrieval fine-tuning.

Phase 1 uses Multiple Negatives Ranking Loss (MNRL) variants. These losses
assume that other items in a batch act as negatives. If a batch contains
duplicate (or near-duplicate) texts, that assumption can be violated, silently
creating **false negatives**.

This module provides a conservative batch sampler that prevents *within-batch*
collisions based on a precomputed normalization key.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Sequence


@dataclass
class BatchCollisionStats:
    """Per-epoch stats emitted by :class:`NoDuplicateTextBatchSampler`."""

    epoch: int
    n_examples: int
    batch_size: int
    n_batches: int
    n_yielded: int
    n_skipped_query_dupe: int
    n_skipped_cde_dupe: int

    @property
    def n_skipped_total(self) -> int:
        return int(self.n_skipped_query_dupe + self.n_skipped_cde_dupe)

    @property
    def skip_rate(self) -> float:
        denom = float(max(self.n_examples, 1))
        return float(self.n_skipped_total) / denom


class NoDuplicateTextBatchSampler:
    """Yield batches of indices with no within-batch text collisions.

    Parameters
    ----------
    query_keys
        Per-example normalization key for the *query* text.
    cde_keys
        Per-example normalization key for the *target/CDE* text.
    batch_size
        Desired batch size.
    seed
        Base RNG seed. The effective seed is ``seed + epoch``.
    drop_last
        If True, drop the final incomplete batch.
    shuffle
        If True, shuffle indices each epoch.

    Notes
    -----
    - This sampler enforces uniqueness **within each batch** only.
    - It may skip some examples during an epoch if many collisions exist.
      Skipped counts are recorded in :attr:`history`.
    """

    def __init__(
        self,
        query_keys: Sequence[str],
        cde_keys: Sequence[str],
        *,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        if len(query_keys) != len(cde_keys):
            raise ValueError("query_keys and cde_keys must have the same length")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0")

        self.query_keys = list(query_keys)
        self.cde_keys = list(cde_keys)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)

        self._epoch = 0
        self.history: List[BatchCollisionStats] = []

    def __len__(self) -> int:
        # Conservative estimate (ignores collisions).
        n = len(self.query_keys)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        n = len(self.query_keys)
        epoch = self._epoch
        self._epoch += 1

        idxs = list(range(n))
        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(idxs)

        n_batches = 0
        n_yielded = 0
        n_skip_q = 0
        n_skip_c = 0

        batch: List[int] = []
        seen_q = set()
        seen_c = set()

        def _flush() -> Iterator[List[int]]:
            nonlocal batch, seen_q, seen_c, n_batches, n_yielded
            if not batch:
                return iter(())
            if len(batch) < self.batch_size and self.drop_last:
                # Drop incomplete batch.
                batch = []
                seen_q = set()
                seen_c = set()
                return iter(())
            out = batch
            n_batches += 1
            n_yielded += len(out)
            batch = []
            seen_q = set()
            seen_c = set()
            return iter((out,))

        # We implement as a generator so we can finalize stats after consumption.
        def _gen() -> Iterator[List[int]]:
            nonlocal n_skip_q, n_skip_c, batch, seen_q, seen_c
            for i in idxs:
                qk = self.query_keys[i]
                ck = self.cde_keys[i]

                if qk in seen_q:
                    n_skip_q += 1
                    continue
                if ck in seen_c:
                    n_skip_c += 1
                    continue

                batch.append(i)
                seen_q.add(qk)
                seen_c.add(ck)

                if len(batch) >= self.batch_size:
                    for b in _flush():
                        yield b

            # final batch
            for b in _flush():
                yield b

            # record stats at end of epoch
            self.history.append(
                BatchCollisionStats(
                    epoch=int(epoch),
                    n_examples=int(n),
                    batch_size=int(self.batch_size),
                    n_batches=int(n_batches),
                    n_yielded=int(n_yielded),
                    n_skipped_query_dupe=int(n_skip_q),
                    n_skipped_cde_dupe=int(n_skip_c),
                )
            )

        return _gen()

class BatchSamplerAsSampler:
    """Wrap a *batch sampler* as an index sampler.

    Some training frameworks (including newer SentenceTransformers + HF Trainer
    integrations) infer the training batch size from ``DataLoader.batch_size``.
    When a PyTorch ``DataLoader`` is constructed with ``batch_sampler=...``,
    ``DataLoader.batch_size`` becomes ``None`` by design, which can break those
    frameworks.

    This wrapper allows us to keep collision-free batching logic in a batch
    sampler while still constructing a DataLoader with an explicit
    ``batch_size=...`` (and therefore a non-None batch_size attribute).
    """

    def __init__(self, batch_sampler: "NoDuplicateTextBatchSampler") -> None:
        self.batch_sampler = batch_sampler

    @property
    def history(self):
        # Proxy history for diagnostics.
        return getattr(self.batch_sampler, "history", [])

    def __len__(self) -> int:
        # Conservative estimate: total examples (ignores collision skips).
        return len(self.batch_sampler.query_keys)

    def __iter__(self) -> Iterator[int]:
        for batch in self.batch_sampler:
            for idx in batch:
                yield int(idx)

