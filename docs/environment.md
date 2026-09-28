# Environment and configuration

Nothing here assumes a particular cluster. If you are running on Biowulf, read
this page first and [`slurm.md`](slurm.md) second.

---

## Install

Python **3.10.8** is what the paper ran. The dependency pins in
`pyproject.toml` are the versions from the paper environment freeze, not
guesses.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e '.[dev]'        # lexical baselines, reporting, the full test suite
```

Extras, by what you intend to run:

| extra | adds | needed for |
|---|---|---|
| *(base)* | pandas, numpy, scipy, scikit-learn, pyarrow, regex, joblib | BM25, the keyword arm, the HGBC, all reporting, every offline test |
| `neural` | torch, transformers, sentence-transformers, accelerate, datasets | bi-encoder and cross-encoder training and inference |
| `figures` | matplotlib | regenerating any figure |
| `extract` | lxml | parsing the caDSR XML |
| `excel` | openpyxl | reading the CIMAC Appendix A workbook |
| `documents` | python-pptx, python-docx | reading the schematic sources and one validator |
| `dev` | pytest | the test suite |
| `all` | everything above | |

```bash
pip install -e '.[all]'         # everything
pip install -e '.[neural,extract]'   # build datasets and train models
```

The base install is deliberately light: **the whole HGBC reranker, both lexical
methods, all reporting and 600+ tests run without torch.**

### Checking the install

```bash
pytest -q          # expect: all pass, ~25 skipped, under a minute, no network
demap --list       # the pipeline, in order
```

The skips are explicit and expected — they are the stages that need a data or
artifact tree you have not pointed at yet, plus the two optional-extra guards.

---

## The two environment variables

```bash
export DEMAP_DATA_ROOT=/path/to/data        # the prepared data tree
export DEMAP_ARTIFACT_ROOT=/path/to/artifacts   # frozen experiment outputs
```

| | what it points at | who reads it |
|---|---|---|
| `DEMAP_DATA_ROOT` | `data/raw/`, `data/processed/`, the splits and evaluation sets | every dataset and training stage |
| `DEMAP_ARTIFACT_ROOT` | `artifacts/` — saved rankings, candidate tables, feature tables, trained models | the reporting, coverage and ablation stages |

`DEMAP_DATA_ROOT` defaults to the current directory, so running from inside a
prepared tree needs no configuration. `DEMAP_ARTIFACT_ROOT` falls back to
`DEMAP_DATA_ROOT`; in practice the two are often the same directory, and they are
separate knobs because you may hold one without the other.

The explicit paper-mode path resolves configured `data/` inputs under the data
root and generated `artifacts/` outputs under the artifact root. Repository
contracts under `configs/` remain repository-relative. Tests assert that the
resolved public paper config contains no developer path and that root overrides
are applied before paper jobs are assembled.

---

## Expected directory layout

```
$DEMAP_DATA_ROOT/
  data/
    raw/
      cadsr_xml/                       # the 2026-01-12 export (or its zip)
      gdc/                             # supplied: the two curation tables
      cimac/                           # public download: the Appendix A workbook
    interim/                           # extraction and merge scratch
    processed/
      cde_master_enriched.parquet      # January construction catalog
      cadsr_xml_2026-06-18/
        cde_master_enriched_eval_production_cde_match.parquet
                                       # June retrieval catalog, 62,976 records
      queries.parquet
      pairs.parquet                    # 69,102 pairs
      splits/
      splits_catalog_filtered/         # reachable-gold training/evaluation sources
      eval_canonical/                  # the six evaluation sets
                                       #   written by `demap materialize-eval`

$DEMAP_ARTIFACT_ROOT/
  artifacts/
    paper/final_system_v1/             # canonical reconstruction outputs
    final_reranker/                    # historical artifact layout
    bm25_canonical_v1/
    evaluation/
```

The names come from `configs/pipeline.yaml`, which is the file to edit if you
want a different layout.

---

## Hardware

| stage | needs |
|---|---|
| dataset construction | CPU. The XML parse is memory-streamed; allow ~8 GB and some patience |
| BM25, keyword arm, candidate pool | CPU |
| bi-encoder screening, Phase 1, Phase 2 | **GPU.** The full published grid is 54 runs per model across two seeds |
| deep retrieval | GPU for encoding, then CPU |
| cross-encoder training | **GPU**, one run per backbone |
| cross-encoder scoring | GPU strongly preferred |
| HGBC training and evaluation | CPU. Minutes, not hours |
| every reporting and sensitivity stage | CPU |
| the full offline test suite | CPU, under a minute |

The expensive parts are the bi-encoder and cross-encoder training. Everything
downstream of a saved checkpoint is CPU work, which is why reproducing the
headline result from frozen artifacts costs seconds rather than GPU-days — see
[`reproducing.md`](reproducing.md).

---

## Model checkpoints

All six are public, and are downloaded from HuggingFace on first use:

```
sentence-transformers/all-mpnet-base-v2        bi-encoder (selected: FT-MPNet)
kamalkraj/BioSimCSE-BioLinkBERT-BASE           bi-encoder
NeuML/pubmedbert-base-embeddings               bi-encoder
ncbi/MedCPT-Cross-Encoder                      cross-encoder (selected: FT-MedCPT)
BAAI/bge-reranker-base                         cross-encoder   (base, not large)
cross-encoder/ms-marco-MiniLM-L-6-v2           cross-encoder
```

Our **fine-tuned** weights are not distributed. That does not block
reproduction: the training protocols are fully specified and start from these
public checkpoints. See [`reproducing.md`](reproducing.md) for the difference
between retraining and reproducing from frozen weights, and the acceptance
criterion for each.

---

## Determinism

For the final HGBC ranking, equal scores are broken by ascending numeric CDE
public identifier (with a deterministic lexical fallback for nonnumeric IDs).
The standalone FT-MedCPT result preserves the authoritative archived ordering;
the HGBC tie policy is not imposed on it. Seeds are fixed and recorded:

| | |
|---|---|
| split construction | 1234 |
| bi-encoder Phase 1 / Phase 2 | 0 and 1 (two seeds, averaged) |
| cross-encoder | 20260527 |
| HGBC | 42 |
| exact-match allowance mask | 42 |

`configs/paper/determinism_policy_v1.json` records the policy. Retraining will
not reproduce neural weights bit-for-bit — different GPUs and kernel versions
move the last digits — which is why the retraining acceptance criterion is a
tolerance plus unchanged method ordering, not equality.
