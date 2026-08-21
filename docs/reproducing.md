# Reproducing the reported results

Three levels of reproduction, in increasing cost and decreasing exactness. Start
at level 1 — it needs nothing but a clone.

---

## Level 1 — offline, no data

```bash
pip install -e '.[dev]'
pytest
```

537 tests. No data, no weights, no network. They check:

- every headline number against a committed fixture (Table 4, Table S5, Table S6,
  Figures 4/5/S5/S6/S7, the K selection, the cross-encoder bake-off, the final HGBC
  configuration and hyperparameter grid);
- the pipeline invariants the paper depends on (split routing, allowance masking,
  the 172-query cross-encoder exclusion, leakage-filter counts);
- that migrated implementations still match the ones that produced the published
  results, by abstract-syntax-tree hash.

This is the level that catches a regression. It cannot catch a case where the
original artifact was itself wrong — for that, see level 2.

---

## Level 2 — regenerate from frozen artifacts

Requires the experiment artifact tree.

```bash
export DEMAP_ARTIFACT_ROOT=/path/to/artifacts
export DEMAP_DATA_ROOT=/path/to/data
pytest -m needs_artifacts
```

These tests recompute results from saved per-method rankings and metric files and
compare against the published values. Tolerance **0.0005** — exact to the
manuscript's rounding.

Stages that can be re-run at this level:

```bash
demap leakage-filter --dry-run     # 51 removed queries, 1089/1759/310/68/113
demap select-k --grid <k_selection_grid.csv>
demap ce-select --ce-root <crossencoder_fulltrain_v2_eligible>
demap table4 --out-dir <out>
demap figure4-inputs --out-dir <out>
demap figure5-inputs --out-dir <out>
demap leakage-sensitivity
```

---

## Level 3 — retrain

The bi-encoder and cross-encoder protocols are fully specified and start from
public HuggingFace checkpoints. Retraining will **not** reproduce weights
bit-for-bit — different GPUs, kernel versions and non-deterministic reductions all
move the last digits.

The acceptance criterion is therefore stated in
[`manifests/expected_results.json`](../manifests/expected_results.json):

- within **0.01 Recall@5** of the published value, **and**
- **method ordering preserved exactly**.

An ordering flip is a real failure even inside the tolerance; a small magnitude
difference is not.

```bash
demap biencoder --help      # screening, Phase 1, Phase 2
demap ce-train --help       # one backbone under the frozen protocol
demap train-hgbc --help     # the 16-config grid, selected on Validation Dev
```

Fixed settings that must not drift when retraining:

| stage | setting |
|---|---|
| Phase 1 | lr {7e-5, 1e-4, 1.5e-4} × temperature {0.04, 0.07, 0.10} × epochs {1,2,3}, 2 seeds |
| Phase 2 | 1 epoch, 2 seeds, strategies `none` / `hard_top25` / `semihard_1_50` |
| cross-encoder | BCE pointwise, 2 epochs, lr 2e-5, batch 32, max length 512, seed 20260527 |
| HGBC | 16 configs, selected on Validation Dev Recall@5; final 200 / 3 / 0.05 / 30, seed 42 |
| candidate pool | FT-MPNet top 20 ∪ CDE Match-Fuzzy top 10, dedup by public id, K = 30 |
| allowance | 0.70 on the caDSR-derived sets, 1.0 on GDC and CIMAC, mask seed 42 |

---

## What cannot be reproduced, and why

**The official NCI CDE Match column of Table 4.** It is saved output of a live
production service. It is not reproducible by us or by anyone else, at any level,
and is reported as a frozen external number.

**The exact Table S3 row counts as printed before 2026-08-08.** The published
diagnostics ran over a `pairs.parquet` build that was later overwritten. Table S3
has since been corrected to the canonical 69,102-pair computation, which is what
this repository reproduces; the superseded artifact is retained in
`tests/fixtures/table_s3_pv_overlap_SUPERSEDED.csv` purely to explain the
discrepancy. The PV generation code itself never drifted — it regenerates the
frozen blocks with 100% row-level parity.

**Byte-identical caDSR catalogs.** The exports are nightly snapshots, not
versioned releases. Their digests are pinned so you can tell whether you hold the
same file.

---

## Environment

Python 3.10.8, with the versions the paper ran pinned in `pyproject.toml`. Install
extras as needed:

```bash
pip install -e '.'            # lexical baselines, reporting, all offline tests
pip install -e '.[neural]'    # + torch / transformers / sentence-transformers
pip install -e '.[figures]'   # + matplotlib
pip install -e '.[all]'       # everything
```
