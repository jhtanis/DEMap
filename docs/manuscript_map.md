# Manuscript result map

One row per reported table and figure: the command that produces it, what it
needs, where it lands, and the fixture that verifies it.

**Reproducibility tier** is the column that matters most:

| tier | meaning |
|---|---|
| **raw** | regenerable from the public caDSR downloads, end to end |
| **frozen** | reproduced from a committed fixture or a frozen study artifact; the original input is not regenerable here |
| **external** | depends on a system or input we cannot distribute |

Tiers are explained in [`reproducing.md`](reproducing.md). Every fixture path is
relative to `tests/fixtures/`; every command is a stage from `demap --list`.

---

## Main text

| item | § | command | inputs | verified against | tier |
|---|---|---|---|---|---|
| **Figure 1** | 2 | *manual schematic* | — | reference source in `reporting/schematics/_reference_figure1.py` | raw |
| **Table 1** | 3.1 | `demap characterize` | evaluation sets, catalog | `expected_results.json` → `A_dataset_counts` | raw |
| **Figure 2** | 3.2 | `demap biencoder` → `make_figure2_S1_S2_heatmaps.py` | representation screen | `B1_representation_screen` | raw |
| **Table 2** | 3.3 | `demap biencoder` → `inputs/build_phase12.py` | rep × loss screen | `B2_representation_x_loss` | raw |
| **Table 3** | 3.3 | same as Table 2 | rep × loss screen | `B2_representation_x_loss` | raw |
| **Figure 3** | 3.4 | `demap biencoder` → `make_figure3_phase1_phase2.py` | Phase 1 + Phase 2 runs | `B3_B4_phase1_phase2` | raw |
| **Figure 4** | 3.5 | `demap figure4-inputs` → `make_figure4_crossencoder.py` | CE bake-off | `ce_bakeoff/`, `E_crossencoder` | raw |
| **Figure 5** | 3.6 | `demap figure5-inputs` → `make_figure5_keyword.py` | keyword runs, non-exact subset | `figure_inputs/*.csv`, `C_keyword_selection` | raw |
| **Figure 6** | 3.7 | *manual schematic* | — | reference source in `reporting/schematics/_reference_figure6.py` | raw |
| **Table 4** | 3.8 | `demap table4` | all six per-method artifacts | `final_table4.csv`, `G_table4_final_comparison` | raw + **external** |

**Table 4 carries one external column.** The official NCI CDE Match row is saved
output of a live production service — not reproducible by us or anyone, at any
tier. It is reported as a frozen external number for GDC and CIMAC only.

---

## Supplement

| item | § | command | inputs | verified against | tier |
|---|---|---|---|---|---|
| **Table S1** | S2.1 | `demap characterize` | evaluation sets | `A_dataset_characterization` | raw |
| **Table S2** | S2.2 | `demap characterize` | evaluation sets | `A_dataset_characterization` | raw |
| **Table S3** | S3.5 | *frozen* — see below | frozen 2026-04-23 diagnostics | `table_s3_pv_overlap_manuscript.csv`, `A_pv_overlap` | **frozen** |
| **Figure S1** | S3.6 | `make_figure2_S1_S2_heatmaps.py` | BioSimCSE screen | `B1_representation_screen` | raw |
| **Figure S2** | S3.6 | `make_figure2_S1_S2_heatmaps.py` | PubMedBERT screen | `B1_representation_screen` | raw |
| **Figure S3** | S4.2 | `make_figureS3_repxloss.py` | rep × loss screen | `B2_representation_x_loss` | raw |
| **Figure S4** | S4.5 | `make_figureS4_stage_comparison.py` | all three stages | `B5a_retained_biencoder_eval` | raw |
| **Table S4** | S4.5 | `demap biencoder` → `inputs/build_stage_comparison.py` | Phase 2 winners on six sets | `B5a_retained_biencoder_eval` | raw |
| **Figure S5** | S5.3 | `demap select-k` → `make_figureS5_k_ceiling.py` | K ceiling grid | `k_selection_grid.csv`, `D_candidate_pool_k` | raw |
| **Table S5** | S5.6 | `demap coverage` | fixed-K feature table | `candidate_coverage_by_dataset.csv`, `D_candidate_pool_coverage` | raw |
| **Figure S6** | S5.7 | `demap family-ablation` + `demap train-hgbc --fixed-config` → `demap figureS6` | fixed-K feature table | `broad_family_recall{5,1}_delta.csv`, `F_broad_family_ablation` | raw |
| **Table S6** | S6.1 | `demap leakage-sensitivity` | saved rankings, filtered lists | `table_s5_leakage.csv`, `H1_leakage_sensitivity` | raw |
| **Figure S7** | S6.2 | `demap allowance-sensitivity` → `make_figureS7_S8_allowance.py` | allowance sweep | `allowance_sensitivity_four_methods.csv` | raw |
| **Figure S8** | S6.2 | same as Figure S7 | allowance sweep | `allowance_sensitivity_four_methods.csv` | raw |
| **Table S7** | S6.2 | `demap allowance-sensitivity` | allowance sweep | `allowance_sensitivity_four_methods.csv`, `H2_allowance_sensitivity` | raw |

### Table S3 is the one frozen result

The manuscript's Table S3 values are reproduced from the frozen study artifact
used for that analysis. The surviving canonical benchmark produces a separately
documented 38,964-row recomputation, so the historical Table S3 input cannot be
regenerated exactly from the current benchmark.

Both are committed — `table_s3_pv_overlap_manuscript.csv` and
`table_s3_pv_overlap_69102_recomputed.csv` — and the difference moves a single
displayed rate cell and changes no conclusion. `demap pv-diagnostics` runs the
same grouping code and reproduces the recomputation, which is what demonstrates
the code itself did not drift.

---

## Prose claims with their own checks

Not every reported number is in a table. These are asserted directly:

| claim | § | verified against |
|---|---|---|
| benchmark counts: 69,102 / 68,142 / 62,847 pairs and queries | S1.3 | `A_dataset_counts`, as exact integers |
| catalog arithmetic: 79,827 → 62,976 | S1.4 | rebuilt from the raw export in `test_core_result_contract.py` |
| training–evaluation overlap: 8 / 7 / 14 / 4 / 18 | S1.5 | `test_leakage_filter.py` |
| bi-encoder margins 0.0048 and 0.0073 | 3.9 | `B3_B4_phase1_phase2` |
| keyword selection: Fuzzy 0.782 vs approximation 0.728 | 3.6 | `C_keyword_selection` |
| K = 30, realized mean 27.6 | S5.3 | `D_candidate_pool_k` |
| cross-encoder: FT-MedCPT 0.912 / BGE 0.904 / MiniLM 0.874 | S5.4 | `E_crossencoder` |
| HGBC: 3,946 queries, 109,141 pairs, 4,109 positives | S5.5 | `F_hgbc_final_model` |
| the 117-feature no-provenance contract | S5.5 | `hgbc_feature_set.json` |
| GDC 70/72 and 71/72 | 3.8 | asserted as integer counts |
| CIMAC 0.802 → 0.779, approximation 0.679 → 0.637 | 3.8 | `H1_leakage_sensitivity` |
| allowance robustness magnitudes | S6.2 | `H2_allowance_sensitivity` |

---

## The fastest way to check all of it

```bash
pytest -q
```

No data, no weights, no network. Every headline number above is checked against
its committed fixture. It will not catch a case where the original artifact was
itself wrong — for that, point at a data tree and run the artifact-backed tiers
described in [`reproducing.md`](reproducing.md).

---

## What is gated, and what that costs you

Three inputs are not distributed. None of them blocks the tier they sit in:

| gate | affects | consequence |
|---|---|---|
| GDC curation tables | Table 1, Table 4, Table S1/S2/S4/S5/S6, Figure S6, §S1.5 | one of six evaluation columns; the other five reproduce |
| fine-tuned weights | Figure 3, Figure 4, Table S4 | retrain from the specified protocols, or use frozen artifacts |
| official NCI CDE Match output | Table 4, §3.8 | not reproducible by anyone; reported as a frozen external number |
| NCI-supplied PL/SQL and logic PDF | — | **nothing.** Our Python implementations ship and run |

The keyword arm — the Python approximation to NCI CDE Match and CDE Match-Fuzzy
— is included in full. Only NCI's own source material is withheld.

CIMAC is **not** on this list. Its source workbook is a public NCI download,
byte-identical to the study copy, and its exact evaluation input ships as
`data/frozen/cimac_v2.parquet`.
