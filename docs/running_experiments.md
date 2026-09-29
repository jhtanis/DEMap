# Running the experiments

The final paper workflow, in order. Every command below is a registered stage —
`demap --list` prints the same sequence, and each accepts `--help`.

Prerequisites: [`environment.md`](environment.md), then
[`building_datasets.md`](building_datasets.md) for stages 1–2.

```bash
export DEMAP_DATA_ROOT=/path/to/data
export DEMAP_ARTIFACT_ROOT=/path/to/artifacts
```

The final manuscript path is owned by
`configs/paper/final_system_v1.yaml`; generic defaults are retained only for
historical compatibility. Before running anything expensive:

```bash
demap paper-config --validate
demap paper-config --show-resolved --out ../demap-paper-config.json
demap paper-config --dry-run all
```

Every paper-mode command below rejects a conflicting manuscript-facing CLI
override. Relative `data/` inputs resolve under `DEMAP_DATA_ROOT`; relative
`artifacts/` outputs resolve under `DEMAP_ARTIFACT_ROOT`.

Stages marked **GPU** are the only expensive ones. Everything else is CPU work
measured in minutes.

---

## 1. Benchmark and datasets

```bash
demap make-dataset --paper-config configs/paper/final_system_v1.yaml \
  --query-xml /path/to/cadsr_2026-01-12 \
  --catalog-xml /path/to/cadsr_2026-06-18 \
  --stop-after catalog-filter
demap materialize-eval --paper-config configs/paper/final_system_v1.yaml
demap leakage-filter
demap characterize                   # Tables 1, S1, S2
```

The manuscript's canonical GDC and CIMAC inputs are already included as
`data/frozen/gdc_combined.parquet` and `data/frozen/cimac_v2.parquet`.
`materialize-eval` verifies and installs those exact files; no private GDC
curation tables or superseded CIMAC workbook are required. `demap import-gdc`
and `demap import-cimac` remain optional historical/raw-source regeneration
utilities for readers who independently hold those source formats, and are not
steps in the paper reconstruction path.

Full detail, including the counts to expect:
[`building_datasets.md`](building_datasets.md).

---

## 2. Text-representation screening — **GPU**

The 4 × 10 query × CDE representation grid, off the shelf, for each of the three
bi-encoders. Produces Figure 2 (all-MPNet) and Figures S1–S2 (the biomedical
models).

```bash
demap biencoder --help               # the screen, Phase 1 and Phase 2 live here
```

Protocol: `configs/paper/biencoder_protocol_v2.yaml`. Screening runs at each
model's native maximum sequence length (all-MPNet 384, BioSimCSE 256,
PubMedBERT 512); fine-tuning later standardizes to 256.

Two representation pairs carry forward per model: the prespecified anchor
(raw query + PV summary × SN + DEC + DEF + PQT + PV) and that model's best
non-anchor cell.

---

## 3. Phase 1 fine-tuning — **GPU**

The historical search was learning rate {7e-5, 1e-4, 1.5e-4} × temperature
{0.04, 0.07, 0.10} × epochs {1, 2, 3}, two seeds each. The selected FT-MPNet
parent is pinned in the canonical manifest: lr 1e-4, temperature 0.04, 3 epochs,
batch 128, max length 256, symmetric MNRL, retained seed 1. Its effective run
precision was FP32; the manifest separately records the historical BF16 request
so those two facts cannot be conflated.

Selection is on Validation Recall@5, then Recall@10, then MRR@100 — ties only.
Winners: `configs/paper/phase1_winners_final_v2.yaml`.

```bash
demap biencoder --paper-config configs/paper/final_system_v1.yaml \
  --selected-final --stage phase1 --dry-run
# Remove --dry-run to train the retained Phase-1 reconstruction on a GPU.
```

---

## 4. Phase 2 hard-negative refinement — **GPU**

One epoch, two seeds, FP32, 132 encoded sequences per optimizer step. Three
strategies compared: no mined negatives, hard negatives from ranks 1–25,
semi-hard from ranks 1–50. Learning-rate and temperature grids are defined
relative to each model's Phase 1 winner (§S4.4).

Winners: `configs/paper/phase2_winners_final_v3.yaml`. **FT-MPNet is selected**
(Validation Recall@5 0.9014 ± 0.0008). A single seed from the winning
configuration is retained downstream.

Produces Figure 3, Figure S4, Table S4.

```bash
demap biencoder --paper-config configs/paper/final_system_v1.yaml \
  --selected-final --stage phase2 --dry-run
# After Phase 1, remove --dry-run and add --mine-hardneg plus the reconstructed
# parent checkpoint. Then train from the emitted hash-addressed parquet:
demap biencoder --paper-config configs/paper/final_system_v1.yaml \
  --selected-final --stage phase2 --mine-hardneg \
  --parent-checkpoint artifacts/paper/final_system_v1/ft_mpnet/phase1/retained/model
demap biencoder --paper-config configs/paper/final_system_v1.yaml \
  --selected-final --stage phase2 \
  --parent-checkpoint artifacts/paper/final_system_v1/ft_mpnet/phase1/retained/model \
  --mined-parquet /path/printed/by/the/mining/command.parquet
```

---

## 5. Deep FT-MPNet retrieval — **GPU** to encode, then CPU

```bash
demap deep-retrieval --paper-config configs/paper/final_system_v1.yaml \
  --paper-split-group training --dry-run
demap deep-retrieval --paper-config configs/paper/final_system_v1.yaml \
  --paper-split-group evaluation --dry-run
# Remove --dry-run for the two retrieval runs.
```

Ranks the full 62,976-record catalog for every query and writes the top-1000
rankings that feed the candidate pool.

---

## 6. Keyword retrieval

Both lexical methods, from the implementations in
`src/demap_repro/lexical/cde_match/`.

```bash
# CDE Match-Fuzzy — the arm that contributes candidates. Repeat for train,
# val_train, val_dev, test, cctg, oid_alt, cdash, gdc_combined, and cimac_v2.
demap cdematch-candidates --paper-config configs/paper/final_system_v1.yaml \
  --paper-dataset test --dry-run

# The Python approximation to NCI CDE Match — a Table 4 comparison method,
# and the source of seven HGBC features. It contributes no candidates.
demap cdematch-candidates --splits test --eligibility production_cde_match \
    --exact-query-match-allow-rate 0.70 --exact-match-seed 42
```

Allowance is **0.70 on the four caDSR-derived sets** and **1.00 on GDC and
CIMAC** (`configs/paper/non_exact_eval_defaults.yaml`). Per-rule depth 500,
truncated to the top 10 merged candidates.

```bash
demap bm25                # the lexical baseline; deterministic, run once
demap non-exact-eval      # Figure 5C, the no-exact-evidence subset
demap figure5-inputs      # Figure 5 input tables
```

---

## 7. Stage-1 candidate pool

```bash
demap select-k --grid <k_selection_grid.csv>    # the K = 20/30/40/60 ceiling
demap coverage                                   # Table S5
demap fixed-k-features --paper-config configs/paper/final_system_v1.yaml --dry-run
```

The pool is **FT-MPNet top-20 ∪ CDE Match-Fuzzy top-10**, deduplicated by CDE
public identifier: nominal K = 30, realized mean 27.6. K was selected on
Reranker Training; Validation is descriptive confirmation only.

`coverage` reports gold coverage per arm and for the union — Table S5, and the
evidence that the union beats either arm on every dataset.

---

## 8. Cross-encoder — **GPU**

```bash
demap ce-pool-train --paper-config configs/paper/final_system_v1.yaml --dry-run
demap ce-pairs --paper-config configs/paper/final_system_v1.yaml --dry-run
demap ce-train --paper-config configs/paper/final_system_v1.yaml --dry-run
demap ce-score --paper-config configs/paper/final_system_v1.yaml --dry-run
demap ce-select          # aggregate the bake-off, write CE_WINNER.json
demap ce-evaluate
demap figure4-inputs     # Figure 4
```

Remove `--dry-run` to execute each selected paper stage. `ce-pool-train` is the
full `train` + `val_dev` construction used for FT-MedCPT training; the separate
fixed-K pool in step 7 is the eight-split, no-gold-injection pool used by the
final scorer and HGBC.

The generic `demap ce-pool-eval` stage remains available for the historical
three-backbone bake-off; the selected final paper path uses the fixed-K pool.

One frozen protocol for all three backbones: BCE pointwise, 2 epochs, lr 2e-5,
batch 32, max length 512, warmup fraction 0.1, FP16 on CUDA,
**seed 20260527**, positive class weighted by the negative-to-positive ratio.
Compared on Validation Recall@5.

**FT-MedCPT selected at 0.912**, ahead of FT-BGE (0.904) and FT-MiniLM (0.874).

---

## 9. The 117-feature HGBC reranker

```bash
demap fixed-k-features --paper-config configs/paper/final_system_v1.yaml --dry-run
demap merge-ce-features --paper-config configs/paper/final_system_v1.yaml --dry-run
demap train-hgbc --paper-config configs/paper/final_system_v1.yaml --dry-run
```

Trained on **Reranker Training** (`val_train`): 3,946 queries, 109,141
query–candidate pairs, 4,109 positive (~3.7%). No negative sampling, no class
weighting.

Historically, a grid of 16 combinations of `max_iter {200,500}` × `max_depth {3,5}` ×
`learning_rate {0.05,0.1}` × `min_samples_leaf {10,30}`, selected on Validation
Recall@5. Paper mode does not rerun that grid: it fits the selected
**200 / 3 / 0.05 / 30, seed 42** configuration against the exact ordered
contract in `configs/paper/features_117_final_v1.json`, with no categorical
features and no class or sample weighting.

To train one fixed configuration instead of searching — which is what the
ablation needs:

```bash
demap train-hgbc --fixed-config '{"max_iter":200,"max_depth":3,"learning_rate":0.05,"min_samples_leaf":30}'
```

---

## 10. Final six-dataset evaluation

```bash
demap eval-by-dataset --paper-config configs/paper/final_system_v1.yaml --dry-run
demap table4                  # Table 4, the six-method comparison
```

`table4` fails loudly if any per-method artifact is missing, cross-checks every
full-precision value against its legacy 4-dp artifact, and gates the 3-dp
rendering against the manuscript in all 36 cells.

---

## 11. Candidate-pool coverage — Table S5

```bash
demap coverage --out-dir <out>
```

Reported in step 7; listed separately because it is a manuscript result in its
own right (§S5.6).

---

## 12. Broad evidence-family ablation — Figure S6

```bash
demap family-ablation --print-commands        # see what will run
demap family-ablation --out-dir <conditions>  # write the four drop-lists
```

Then train one HGBC per condition with the hyperparameters **frozen**, so the
feature set is the only design change:

```bash
demap train-hgbc --feature-table <feature_table.parquet> \
    --exclude-features-file <conditions>/minus_lexical_keyword.json \
    --fixed-config '{"max_iter":200,"max_depth":3,"learning_rate":0.05,"min_samples_leaf":30}' \
    --out <conditions>/minus_lexical_keyword
```

Four conditions: `full`, `minus_lexical_keyword`, `minus_ft_mpnet`,
`minus_ft_medcpt`. The `full` condition must reproduce the shipped model — that
is the control. Render with:

```bash
demap figureS6
```

CPU only, and cheap relative to anything neural.

---

## 13. Training–overlap sensitivity — Table S6

```bash
demap leakage-sensitivity
```

Rescores the same saved rankings against the leakage-filtered query lists. No
retraining and no re-retrieval: it is a denominator change.

---

## 14. Exact-match allowance sensitivity — Figures S7–S8, Table S7

The sweep regenerates CDE Match-Fuzzy candidates, merged pools and
allowance-dependent features at each of six rates (0.0, 0.5, 0.6, 0.7, 0.8,
1.0), holding FT-MPNet, FT-MedCPT and — for the fixed arm — the HGBC weights
constant.

```bash
demap allowance-sensitivity      # aggregate the sweep into the four-method grid
```

This is the most expensive non-neural analysis: six rates × two arms. The Slurm
drivers that ran it are in `workflows/sensitivity/` — see
[`slurm.md`](slurm.md). The aggregated result is committed as
`tests/fixtures/allowance_sensitivity_four_methods.csv`, so the published values
can be checked without rerunning it.

---

## 15. Figures

```bash
demap figure4-inputs && demap figure5-inputs && demap figureS6
```

Figures 2, 3, S1–S5 are rendered by the generators in
`src/demap_repro/reporting/figures/`. Figures 1 and 6 are manually maintained
schematics; their editable sources are in
`src/demap_repro/reporting/schematics/` and they are validated, not regenerated.

Which command produces which manuscript item:
[`manuscript_map.md`](manuscript_map.md).

---

## If you only want the headline result

You do not need any of the above. With frozen artifacts, the paper's central
result recomputes from the trained model in about four seconds on one CPU core:

```bash
pytest tests/tier3_regression/test_core_result_contract.py -v
```

See [`reproducing.md`](reproducing.md).
