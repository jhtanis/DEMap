# Running on Slurm

**Optional.** Nothing in this repository requires Slurm, and the core path in
[`running_experiments.md`](running_experiments.md) is platform-neutral. This
page is for the parts that were run as cluster jobs, and for anyone who wants
the exact resource envelopes we used.

The drivers under `workflows/` are the ones that produced the published
artifacts. They were written for NIH Biowulf, so they carry site-specific
directives (`--partition=norm`, `--gres=lscratch:N`). Treat them as records of
what ran and as templates, not as portable scripts.

---

## What is here

```
workflows/
  crossencoder/
    pool_pairs_fulltrain.sbatch          Stage E: candidate pool + training pairs
  sensitivity/
    J1_regenerate_fuzzy_per_rate.sbatch  regenerate Fuzzy candidates per allowance
    J2_rebuild_pools_features_ce.sbatch  rebuild pools, features and CE scores
    J3_retrain_hgbc_per_rate.sbatch      the retrained arm: one HGBC per rate
    J3b_retrain_hgbc_archived_features.sbatch  same, from archived features
    K1_score_fixed_070.sbatch            the fixed arm: score the 0.70 model at every rate
```

The J and K chains together produce the allowance sweep behind Figures S7–S8
and Table S7. J is the **retrained** arm (a new HGBC at each rate); K is the
**fixed** arm (the shipped 0.70 model scored at every rate). Their aggregation
is `demap allowance-sensitivity`.

---

## Paths and logs

Every driver resolves its tree from `DEMAP_DATA_ROOT`. The original hardcoded
research-repository root was parameterized on migration, so:

```bash
export DEMAP_DATA_ROOT=/path/to/data
```

`#SBATCH --output` and `--error` are static directives and **cannot** expand
shell variables, so pass them on the command line:

```bash
sbatch --output=/path/to/logs/%A_%a.out \
       --error=/path/to/logs/%A_%a.err \
       workflows/sensitivity/J1_regenerate_fuzzy_per_rate.sbatch
```

---

## Resource envelopes as run

| driver | partition | cpus | mem | time | array |
|---|---|---|---|---|---|
| `pool_pairs_fulltrain` | norm | 8 | 64G | 02:00 | — |
| `J1_regenerate_fuzzy_per_rate` | norm | 8 | 64G | 06:00 | 0–5 (six rates) |
| `J2_rebuild_pools_features_ce` | see file | | | | 0–5 |
| `J3_retrain_hgbc_per_rate` | see file | | | | 0–5 |
| `K1_score_fixed_070` | see file | | | | 0–5 |

The six-element arrays are the six allowance rates: 0.0, 0.5, 0.6, 0.7, 0.8, 1.0.

Node-local scratch (`--gres=lscratch:N`) is a Biowulf facility. On another
cluster, drop it and point the job's temporary directory wherever local scratch
lives.

---

## Bi-encoder and cross-encoder training

The GPU grids — 54 Phase 1 runs per model, plus Phase 2 and the three
cross-encoder backbones — were run as job arrays over the configurations that
`demap biencoder` and `demap ce-train` enumerate. Those drivers are not
included, because they encode a scheduler layout rather than any part of the
method: the protocol is fully specified in
[`running_experiments.md`](running_experiments.md) and in
`configs/paper/biencoder_protocol_v2.yaml`, and one configuration per job is the
only structure needed.

---

## Running without a scheduler

Every stage is an ordinary Python entry point. Anything in `workflows/` can be
run directly by reading the command out of the driver and substituting the array
index:

```bash
# what J1 array task 3 does, run in the foreground
demap cdematch-candidates --splits val_train val_dev \
    --fuzzy-fallback keyword_fuzzy_all \
    --eligibility production_cde_match \
    --exact-query-match-allow-rate 0.7 --exact-match-seed 42 \
    --out-dir <out>
```

The scheduler buys parallelism across the six rates, nothing else.

---

## A note on cost

Only the neural stages need a GPU, and only the allowance sweep needs many CPU
hours. Everything else — the keyword arm, the candidate pool, the HGBC, all
reporting, the coverage analysis and the family ablation — runs in minutes on a
single machine.

Reproducing the paper's headline result from frozen artifacts needs neither: it
is about four seconds on one CPU core. See [`reproducing.md`](reproducing.md).
