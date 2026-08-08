# DEMap

DEMap provides the code and reproducibility workflows used for the analyses
reported in the accompanying manuscript on mapping free-text clinical data
descriptors to caDSR Common Data Elements.

The system retrieves candidate CDEs with a fine-tuned bi-encoder and a lexical
keyword arm, scores each candidate with a fine-tuned cross-encoder, and ranks the
pooled candidates with a gradient-boosted classifier over 117 lexical and semantic
features. It reaches Recall@5 of 0.971 on the internal test set and 0.802–0.972
across five distribution-shifted external evaluation sets.

This repository contains **only** the code behind results reported in the
manuscript, plus what those results depend on. Abandoned approaches, superseded
runs and debugging variants are deliberately absent. The manuscript is the scope
boundary, and every included component maps to an entry in
[`manifests/paper_scope.yaml`](manifests/paper_scope.yaml).

---

## Quick start

```bash
pip install -e '.[all]'      # or: pip install -e .  for the lexical/reporting subset
pytest                       # 484 tests, no data or network required
demap --list                 # the pipeline, in order
```

The test suite runs on a clean clone. It verifies the published numbers against
small committed fixtures and checks that the migrated implementations still match
the ones that produced them.

Point `DEMAP_DATA_ROOT` at a prepared data tree to run the stages themselves, and
`DEMAP_ARTIFACT_ROOT` at the frozen experiment artifacts to enable the additional
tests that check the fixtures against the full-size originals.

---

## What is here

| path | |
|---|---|
| `src/demap_repro/data/` | benchmark construction: caDSR extraction, query and pair building, splits, reachability, leakage filtering, PV summaries |
| `src/demap_repro/biencoder/` | representation screening, Phase 1 and Phase 2 fine-tuning, selection, deep retrieval |
| `src/demap_repro/lexical/` | BM25 baseline, the exact-match allowance mask, the non-exact subset evaluation |
| `src/demap_repro/pool/` | candidate-pool union and the selection of K |
| `src/demap_repro/crossencoder/` | pool and pair construction, training, scoring, selection |
| `src/demap_repro/reranker/` | fixed-K feature assembly, split routing, HGBC training and ranking |
| `src/demap_repro/sensitivity/` | leakage and exact-match-allowance sensitivity analyses |
| `src/demap_repro/reporting/` | tables, figure inputs, figure generators |
| `manifests/` | paper scope, source migration ledger, expected results |
| `tests/` | invariants, behaviour, and regression against the published results |

---

## Pipeline

```
caDSR export ──► benchmark pairs ──► splits ──► reachability filter
                                                        │
                        ┌───────────────────────────────┴───────────────┐
                        ▼                                               ▼
              FT-MPNet bi-encoder                            CDE Match-Fuzzy
                  top 20                                          top 10
                        └───────────────┬───────────────────────────────┘
                                        ▼
                    candidate pool, deduplicated by public id
                            nominal K = 30, realized ≈ 27.6
                                        ▼
                          FT-MedCPT cross-encoder scores
                                        ▼
                      117-feature HGBC ──► final ranking
```

Ties are broken by ascending CDE public identifier, so the output is fully
deterministic.

---

## Reproducing the reported numbers

Three levels, in increasing cost:

**1. Offline (no data).** `pytest` checks every headline result against committed
fixtures: Table 4, the K selection, the cross-encoder bake-off, the allowance
sensitivity, the leakage filter, the final HGBC configuration.

**2. From frozen artifacts.** With `DEMAP_ARTIFACT_ROOT` set, the reporting and
sensitivity stages regenerate their outputs from saved experiment results and are
compared to the published values. Tolerance: 0.0005.

**3. By retraining.** The bi-encoder and cross-encoder protocols are fully
specified and start from public base checkpoints. Retraining will not reproduce
weights bit-for-bit; the acceptance criterion is 0.01 Recall@5 **and** unchanged
method ordering.

---

## Data

**Every dataset this project uses is public.** caDSR is a public NCI registry;
CIMAC-CIDC, GDC and the CDISC-derived evaluation sets are public resources. Nothing
here is withheld for privacy.

Two practical caveats. The caDSR exports are nightly snapshots rather than
versioned releases, so byte-exact re-acquisition of the January and June catalogs
is not possible after the fact — their SHA256 digests are pinned in
`configs/paper/` so a reader can tell whether they hold the same file. And the data
tree is far too large to version, so it is not included in the repository.

---

## Components with unresolved redistribution status

Some code and weights are not included, for reasons that are about redistribution
rights rather than about the data:

| not included | why |
|---|---|
| the Python approximation to NCI CDE Match, and CDE Match-Fuzzy built on it | derivative of Oracle PL/SQL supplied to us by NCI; redistribution unresolved |
| fine-tuned bi-encoder and cross-encoder weights | our own weights, but redistribution not yet determined |
| saved output of the live NCI CDE Match service | a third-party service result, never reproducible by anyone |

These are recorded in `manifests/source_migration.yaml` with the gate that blocks
each one, and none has ever entered this repository or its git history.

**This does not block reproduction.** Stages that would call the gated keyword arm
resolve it through a documented adapter
(`src/demap_repro/lexical/cde_match_interface.py`) and accept a precomputed
candidate artifact instead. BM25 is a lexical baseline with no such dependency and
is included in full. And an unresolved *weights* gate does not prevent retraining:
the base checkpoints are public and the protocols are specified.

---

## Provenance

This repository was migrated from a separate research repository under a
copy-first rule: copy the bytes that actually ran, prove parity, then refactor.
[`docs/provenance.md`](docs/provenance.md) states the rules;
`manifests/source_migration.yaml` records, for every file, where it came from, its
hash at the time of copying, and any divergence from a competing implementation.

Two consequences worth knowing when reading the code:

- The implementation that produced a published number was not always the committed
  one. Where a working-tree or exploratory version was the one that ran, that
  version is what was migrated, and the ledger says so.
- `tests/fixtures/source_parity.json` pins the syntax tree of each migrated
  function against the hash it had in the research repository, so later cleanup
  cannot silently change behaviour.

---

## License

To be determined. No license has been chosen for this repository, and none is
declared in `pyproject.toml`.

## Citation

See the manuscript. If you use this code, please also cite caDSR and the
evaluation datasets it builds on.
