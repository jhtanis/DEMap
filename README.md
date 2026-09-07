# DEMap

Code and reproducibility workflows for the analyses reported in the accompanying
manuscript on mapping free-text clinical data descriptors to caDSR Common Data
Elements.

The system retrieves candidate CDEs with a fine-tuned bi-encoder and a lexical
keyword arm, scores each candidate with a fine-tuned cross-encoder, and ranks the
pooled candidates with a gradient-boosted classifier over 117 lexical and
semantic features. It reaches Recall@5 of 0.971 on the internal test set and
0.802–0.972 across five distribution-shifted external evaluation sets.

This repository contains **only** the code behind results reported in the
manuscript, plus what those results depend on. Every included component maps to an 
entry in [`manifests/paper_scope.yaml`](manifests/paper_scope.yaml).

---

## Quick start

```bash
pip install -e '.[dev]'      # or: '.[all]' to include the neural stages
pytest -q                    # ~660 tests, no data or network required
demap --list                 # the pipeline, in order
```

The test suite runs on a clean clone. It verifies the published numbers against
small committed fixtures and checks that the migrated implementations still match
the ones that produced them.

---

## Documentation

Start wherever your question is.

| | |
|---|---|
| **[Obtaining the data](docs/data_sources.md)** | exact download URLs and digests for the two caDSR snapshots, and what must be supplied instead |
| **[Building the datasets](docs/building_datasets.md)** | raw XML → benchmark → splits → catalog → the six evaluation sets |
| **[Environment](docs/environment.md)** | install, extras, the two environment variables, directory layout, hardware |
| **[Running the experiments](docs/running_experiments.md)** | the final workflow as 15 ordered stages |
| **[Manuscript map](docs/manuscript_map.md)** | one row per table and figure: command, inputs, output, verifying fixture |
| **[Reproducing](docs/reproducing.md)** | four levels, and the reproducibility boundaries |
| **[Provenance](docs/provenance.md)** | the copy-first rule, the migration ledger, what is excluded and why |
| **[Slurm](docs/slurm.md)** | optional; the cluster drivers and their resource envelopes |

---

## The pipeline

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

## What is here

| path | |
|---|---|
| `src/demap_repro/data/` | benchmark construction: caDSR extraction, query and pair building, splits, reachability, leakage filtering, PV summaries |
| `src/demap_repro/biencoder/` | representation screening, Phase 1 and Phase 2 fine-tuning, selection, deep retrieval |
| `src/demap_repro/lexical/` | BM25 baseline, the exact-match allowance mask, the non-exact subset evaluation |
| `src/demap_repro/lexical/cde_match/` | the Python approximation to NCI CDE Match, the CDE Match-Fuzzy retriever, and the candidate builder |
| `src/demap_repro/pool/` | candidate-pool union, the selection of K, and coverage by arm |
| `src/demap_repro/crossencoder/` | pool and pair construction, training, scoring, selection |
| `src/demap_repro/reranker/` | fixed-K feature assembly, split routing, HGBC training, the evidence-family ablation |
| `src/demap_repro/sensitivity/` | leakage and exact-match-allowance sensitivity analyses |
| `src/demap_repro/reporting/` | tables, figure inputs, figure generators |
| `configs/` | the dataset pipeline contract, curator allowlists, and the frozen protocols |
| `data/frozen/` | the two frozen evaluation sets distributed here: CIMAC and GDC |
| `manifests/` | paper scope, source migration ledger, expected results |
| `workflows/` | optional Slurm drivers |
| `tests/` | invariants, behaviour, and regression against the published results |

---

## Reproducing the reported numbers

Four levels, in increasing cost — full detail in
[`reproducing.md`](docs/reproducing.md):

1. **Offline.** `pytest` checks every headline result against committed fixtures.
2. **From frozen artifacts.** The paper's central result recomputes from the
   trained model in about four seconds on one CPU core.
3. **Rebuild the data.** Both caDSR snapshots are publicly downloadable with
   verified digests; the structural counts reproduce exactly. The two external
   evaluation sets ship in `data/frozen/`.
4. **Retrain.** Protocols are fully specified and start from public checkpoints.
   Acceptance is 0.01 Recall@5 **and** unchanged method ordering.

---

## Data

**Every dataset this project uses is public.** caDSR is a public NCI registry,
and both snapshots the paper used are downloadable today from caDSR's dated
archive — URLs and sha256 digests in [`data_sources.md`](docs/data_sources.md).
The January archive file was verified byte-identical to the copy this study ran
on. Nothing is withheld here for privacy.

CIMAC's source is NCI's public CIMAC-CIDC clinical-data-element template —
template and data-element metadata, not patient-level study data — and the
workbook NCI serves today is byte-identical to the one this study used. Its
exact PV-enriched evaluation set ships in
[`data/frozen/`](data/frozen/README.md), because the separate permissible-value
workbook it drew on has since changed upstream.

GDC's evaluation set ships in `data/frozen/` too, for the opposite reason: its
gold CDE assignments are a curator's expert judgement and no public GDC endpoint
carries them, so distributing the derived set is the only thing that makes the
reported GDC evaluation reproducible. The raw curation submissions are not
included — they are workflow metadata with no reproduction value.

**Nothing here requires private access.**

The data tree itself is far too large to version and is not included.

---

## Not included

| | why |
|---|---|
| NCI-supplied Oracle PL/SQL and the CDE Match logic PDF | NCI's source material; redistribution unresolved |
| saved output of the live NCI CDE Match service | a frozen output of an external service; not reproducible locally from this repository |
| fine-tuned bi-encoder and cross-encoder weights | ours, but redistribution not yet determined |
| large derived artifact trees | regenerable, and gigabytes; excluded by size, not by rights |

**The keyword arm itself is included.** Both the Python approximation to NCI CDE
Match and the CDE Match-Fuzzy retriever ship here and run; only NCI's own
supplied material is withheld. An unresolved *weights* gate does not prevent
retraining, and the live-service column is a single reported comparison rather
than an input to the system. What each exclusion costs is set out in
[`reproducing.md`](docs/reproducing.md#reproducibility-boundaries).

---

## License

To be determined. No license has been chosen for this repository, and none is
declared in `pyproject.toml`.

## Citation

See the manuscript. If you use this code, please also cite caDSR and the
evaluation datasets it builds on.
