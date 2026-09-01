# Reproducing the reported results

Four levels, in increasing cost and decreasing exactness. Start at level 1 — it
needs nothing but a clone.

Which command produces which table or figure:
[`manuscript_map.md`](manuscript_map.md).

---

## Level 1 — offline, no data

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
original artifact was itself wrong — for that, see level 2.

The ~25 skips are explicit: stages needing a data or artifact tree you have not
pointed at, plus two optional-extra guards.

---

## Level 2 — recompute from frozen artifacts

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

## Level 3 — rebuild the data

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

## Level 4 — retrain

The bi-encoder and cross-encoder protocols are fully specified and start from
public HuggingFace checkpoints. Retraining will **not** reproduce weights
bit-for-bit: different GPUs, kernel versions and non-deterministic reductions
all move the last digits.

The acceptance criterion is therefore stated in
[`manifests/expected_results.json`](../manifests/expected_results.json):

- within **0.01 Recall@5** of the published value, **and**
- **method ordering preserved exactly**.

An ordering flip is a real failure even inside the tolerance; a small magnitude
difference is not.

```bash
demap biencoder --help      # screening, Phase 1, Phase 2
demap ce-train --help       # one backbone under the frozen protocol
demap train-hgbc --help     # the 16-config grid, selected on Validation
```

Settings that must not drift when retraining:

| stage | setting |
|---|---|
| Phase 1 | lr {7e-5, 1e-4, 1.5e-4} × temperature {0.04, 0.07, 0.10} × epochs {1,2,3}, 2 seeds |
| Phase 2 | 1 epoch, 2 seeds, strategies `none` / `hard_top25` / `semihard_1_50` |
| cross-encoder | BCE pointwise, 2 epochs, lr 2e-5, batch 32, max length 512, seed 20260527 |
| HGBC | 16 configs on Validation Recall@5; final 200 / 3 / 0.05 / 30, seed 42 |
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

**The neural checkpoints**, if you choose not to retrain. Level 2 exists for
exactly this.

### Dependent on external systems or undistributed inputs

**The official NCI CDE Match column of Table 4.** The reported values are
frozen outputs from the external NCI CDE Match service and are not reproducible
locally from this repository, at any level. They are reported for GDC and CIMAC
only. The manuscript
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
