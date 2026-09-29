# Reproducing the reported results

The release separates a quick referee check from the much larger experiment
workflow. These answer different questions and must not be described as the
same reproduction.

Which command produces which table or figure:
[`manuscript_map.md`](manuscript_map.md).

---

## Level A — quick referee verification

Level A recomputes every published Table 4 Recall@5 cell from a compact,
checksummed bundle of frozen query-level outputs:

```bash
demap reproduce-paper-table4 \
  --bundle data/frozen/table4_v1 \
  --out-dir ../demap-table4-verification \
  --offline
```

The command validates the stable `configs/paper/table4_results_v1.yaml`
scientific contract, both Parquet checksums and byte sizes, all 7,349 query
identities, the six exact denominators, and all 44,094 query/method rows. It
rejects missing or extra queries, duplicate candidates, invalid depths and
scores, ranking-policy violations, and any integer R@5 hit-count discrepancy.
It writes `table4.csv`, `metrics.json`, and `verification.json` only to the
requested external output directory.

This is **metric recomputation from frozen query-level outputs**. It does not
retrain the bi-encoder, cross-encoder, or HGBC; it does not rerun neural
inference; and it does not reconstruct every upstream experiment.

The repository includes the verified bundle at `data/frozen/table4_v1/`; no
download or network access is required. Missing bundle data is a hard failure,
never a skipped check.

## Level B — full / expensive reproduction

The remaining workflows range from inexpensive regression checks to full model
retraining. They remain separate from Level A and retain their existing cost and
artifact requirements.

### B1 — offline regression, no data

```bash
pip install -e '.[dev]'
pytest -q
```

No data, no weights, no network, under a minute. Checks:

- every headline number against a committed fixture — Table 4, Table S5,
  Table S6, Table S7, Figures 4/5/S5/S6/S7/S8, the K selection, the
  cross-encoder bake-off, the final HGBC configuration and its grid;
- the invariants the paper depends on — split routing, the allowance mask, the
  172-query cross-encoder exclusion, leakage-filter counts, the 117-feature
  contract;
- that every migrated implementation still matches the one that produced the
  published result, by abstract-syntax-tree hash;
- that the documentation names only commands that exist, and that every
  declared config is present.

This is the level that catches a regression. It cannot catch a case where the
original artifact was itself wrong — for that, use Level A or B2.

The ~25 skips are explicit: stages needing a data or artifact tree you have not
pointed at, plus two optional-extra guards.

---

### B2 — recompute models from frozen artifacts

Needs the experiment artifact tree.

```bash
export DEMAP_DATA_ROOT=/path/to/data
export DEMAP_ARTIFACT_ROOT=/path/to/artifacts
pytest -q                      # the skips become real checks
```

Recomputes results from saved rankings and metric files and compares to the
published values. Tolerance **0.0005** — exact to the manuscript's rounding.

**The paper's central result recomputes here in about four seconds on one CPU
core.** It reloads the frozen HGBC, rebuilds the matrix from the frozen
423,395-row feature table, and recomputes with the trainer's own metrics:

```bash
pytest tests/tier3_regression/test_core_result_contract.py -v
```

| dataset | n | R@1 | R@5 | R@10 | MRR@100 |
|---|---|---|---|---|---|
| Test | 3,959 | 0.8957 | **0.9715** | 0.9803 | 0.9289 |
| CCTG | 1,097 | 0.7976 | **0.9088** | 0.9344 | 0.8471 |
| OID ALT | 1,766 | 0.7644 | **0.8324** | 0.8533 | 0.7963 |
| CDASH | 324 | 0.8086 | **0.9198** | 0.9444 | 0.8580 |
| GDC | 72 | 0.9306 | **0.9722** | 0.9861 | 0.9537 |
| CIMAC | 131 | 0.7328 | **0.8015** | 0.8397 | 0.7669 |

Agreement with the frozen evaluation table is ~10⁻¹⁶. Anything looser than
~10⁻¹² should be investigated rather than accepted.

Stages re-runnable at this level:

```bash
demap leakage-filter --dry-run     # 51 removed: 1089/1759/310/68/113
demap select-k --grid <k_selection_grid.csv>
demap coverage
demap ce-select --ce-root <crossencoder_fulltrain_v2_eligible>
demap table4 --out-dir <out>
demap figure4-inputs --out-dir <out>
demap figure5-inputs --out-dir <out>
demap leakage-sensitivity
demap figureS6
```

---

### B3 — rebuild the data

Needs only the two public caDSR downloads. See
[`data_sources.md`](data_sources.md) and
[`building_datasets.md`](building_datasets.md).

Everything structural is reproducible exactly: the 69,102 constructed pairs, the
splits, the 62,976-record catalog, the six evaluation sets and their
denominators, the leakage filter. These are asserted as **exact integers**, not
tolerances.

The two external evaluation sets are not rebuilt at this level — they ship in
`data/frozen/` instead. CIMAC's source workbook is a public download whose
queries and gold reproduce from it; GDC's gold is expert curation with no public
upstream. The four caDSR-derived sets are unaffected.

---

### B4 — retrain

The manuscript-selected route is defined by the validated, machine-readable
`configs/paper/final_system_v1.yaml`. It references the exact winner manifests,
allowance and evaluation registries, and the ordered 117-feature contract. Start
by inspecting the resolved jobs; this performs no training:

```bash
demap paper-config --validate
demap paper-config --show-resolved --out ../demap-paper-config.json
demap paper-config --dry-run all

demap biencoder --paper-config configs/paper/final_system_v1.yaml \
  --selected-final --stage phase1 --dry-run
demap biencoder --paper-config configs/paper/final_system_v1.yaml \
  --selected-final --stage phase2 --dry-run
demap ce-train --paper-config configs/paper/final_system_v1.yaml --dry-run
demap train-hgbc --paper-config configs/paper/final_system_v1.yaml --dry-run
```

The selected bi-encoder and cross-encoder start from public HuggingFace
checkpoints at immutable historical revisions:

- `sentence-transformers/all-mpnet-base-v2@e8c3b32edf5434bc2275fc9bab85f82640a19130`
- `ncbi/MedCPT-Cross-Encoder@71caf65d4927987813984f54c284405a13fcca49`

Paper mode passes those revisions explicitly. The fine-tuned checkpoints are
not distributed, so Level B reconstructs them by training. Retraining will **not** reproduce weights
bit-for-bit: different GPUs, kernel versions and non-deterministic reductions
all move the last digits.

The acceptance criterion is therefore stated in
[`manifests/expected_results.json`](../manifests/expected_results.json):

- within **0.01 Recall@5** of the published value, **and**
- **method ordering preserved exactly**.

An ordering flip is a real failure even inside the tolerance; a small magnitude
difference is not.

The complete ordered paper-mode commands, including the public data build,
candidate pools, scoring, and final evaluation, are in
[`running_experiments.md`](running_experiments.md). Generic commands without
`--paper-config` retain historical defaults for backward compatibility and are
not authoritative for reconstructing the final manuscript system.

Settings that must not drift when retraining:

| stage | setting |
|---|---|
| Phase 1 | lr {7e-5, 1e-4, 1.5e-4} × temperature {0.04, 0.07, 0.10} × epochs {1,2,3}, 2 seeds |
| Phase 2 | 1 epoch, 2 seeds, strategies `none` / `hard_top25` / `semihard_1_50` |
| cross-encoder | BCE pointwise, 2 epochs, lr 2e-5, batch 32, max length 512, warmup 0.1, FP16 on CUDA, seed 20260527 |
| HGBC | fixed final 200 / 3 / 0.05 / 30, seed 42, exact ordered 117-feature contract |
| candidate pool | FT-MPNet top 20 ∪ CDE Match-Fuzzy top 10, dedup by public id, K = 30 |
| allowance | 0.70 on the caDSR-derived sets, 1.0 on GDC and CIMAC, mask seed 42 |

**The HGBC is CPU work and takes minutes.** Only the bi-encoder and
cross-encoder stages need a GPU, so a reader who accepts the published
checkpoints can rebuild the entire reranking layer cheaply.

---

## Reproducibility boundaries

<a id="frozen-inputs"></a>

### Regenerable from public raw inputs

The benchmark, the splits, the catalog, the four caDSR-derived evaluation sets,
BM25, both keyword methods, the candidate pool, the HGBC, and every reporting
and sensitivity analysis downstream of a saved checkpoint.

### Reproduced from frozen study artifacts

**Table S3.** The manuscript's Table S3 values are reproduced from the frozen
study artifact used for that analysis. The surviving canonical benchmark
produces a separately documented 38,964-row recomputation, so the historical
Table S3 input cannot be regenerated exactly from the current benchmark.

Both are committed, the difference moves a single displayed rate cell, and it
changes no conclusion of the permissible-value overlap analysis. `demap
pv-diagnostics` runs the same grouping code and reproduces the recomputation,
which demonstrates the code did not drift — only the input it was originally run
over is gone.

**The GDC evaluation set.** Manually curated: a curator assigned, for each GDC
property, the caDSR CDE it maps to, and that assignment is the benchmark label.
The GDC Data Dictionary publishes properties and permissible values but no
property→CDE linkage, so the gold mappings are not recoverable from a public
GDC source. The 72-query evaluation set is shipped as
`data/frozen/gdc_combined.parquet`; the raw curation submissions are not, being
workflow metadata with no reproduction value.

**The CIMAC evaluation set.** Its queries and gold CDE mappings come from NCI's
public CIMAC-CIDC clinical-data-element template — template and data-element
metadata, not patient-level study data — and the Appendix A workbook NCI serves
today is byte-identical to the one this study used. Permissible-value metadata
came from a second public workbook that has since changed upstream, with no
dated archive, so the exact PV-enriched 131-query evaluation set is distributed
here as `data/frozen/cimac_v2.parquet`. CIMAC is a public input; only its
PV-enriched representation is frozen rather than regenerated.

**The neural checkpoints**, if you choose not to retrain. B2 exists for
exactly this.

### Dependent on external systems or undistributed inputs

**The official NCI CDE Match narrative comparison.** The reported values are
frozen outputs from the external NCI CDE Match service and are not reproducible
locally from this repository, at any level. They are reported for GDC and CIMAC
only and are not a column in the manuscript's six-method Table 4. The manuscript
reports it for those two sets alone because the four caDSR-derived sets were
built from the same alternate-name and question-text fields the service queries,
which would make the comparison circular, and because the live service cannot be
subjected to the query-level exact-match masking the controlled experiments use.

**Fine-tuned weights.** Ours, but not distributed. Reproducible by retraining at
level 4.

**Byte-identical caDSR catalogs.** No longer a boundary — both snapshots are in
the public dated archive with digests in [`data_sources.md`](data_sources.md).
The January archive file was verified byte-identical to the copy this study ran
on.

### What is *not* a boundary

The Python approximation to NCI CDE Match and the CDE Match-Fuzzy retriever ship
in full and run. Only NCI's own supplied PL/SQL and its logic PDF are withheld,
and nothing depends on them.

---

## Environment

Python 3.10.8, with the versions the paper ran pinned in `pyproject.toml`. See
[`environment.md`](environment.md).
