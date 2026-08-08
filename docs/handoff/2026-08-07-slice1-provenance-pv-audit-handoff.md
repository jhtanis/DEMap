# Self-handoff — 2026-08-07 — DEMAP public-reproducibility repository

Self-contained. A fresh session can resume from this file alone; no chat history needed.

---

## 1. Current state

| | |
|---|---|
| **Authoritative manuscript** | `/data/nextgen2/james/tasks/cde_project/demap/manuscript/cde_paper_v21.docx` — sha256 `2a374c5d67c8913f7654b0b7adb4ede0ff7e03df849e9d2b3a6c568d7cad1197` |
| **SRC (immutable)** | `/data/nextgen2/james/tasks/cde_project/demap` — HEAD `cf8743f535f4f4fb8eb7bcc71362986d615ec58f`, branch `phase0-representation-sweep`, porcelain 97 lines (10 modified tracked + 87 untracked) |
| **DST (only writable location)** | `/data/nextgen2/james/tasks/cde_project/demap_repro` |
| **Scratch** | `demap_repro/.scratch/` only. **Never `/tmp`.** |

`/vf/users/nextgen2/...` and `/data/nextgen2/...` are the **same tree** (identical device:inode). Either path works.

**SRC is permanently read-only for this entire effort.** Do not modify, stage, commit, rename, delete, touch, reformat or tag anything under it — not tracked files, not untracked files, not `.scratch/`, not docs, configs, tests, manifests, or git state. It remains the research/provenance archive and may still be needed for manuscript revisions. Corrections discovered during migration are applied **only to copies inside DST** and recorded as `divergence_note` entries.

### Completed

- **Slice 1** — manuscript scope (`paper_scope.yaml`), source-migration ledger, expected results, result provenance, provenance policy. 42 scope items derived directly from v21.
- **Slice 1.5** — Table S3 producer chain resolved; `eval_canonical_v2` producer resolved; `_build_data2.py` classified `historical_reference_stale`; release/runnability metadata refined into 8 independent dimensions; Figure S7 confirmed present in v21 and retained; HGBC metadata verified.
- **Slice 1.6** — PV-summary generator regression audit (§4 below). **No regression.**

### Not done (deliberately)

- **No scientific implementation code has been copied to DST.**
- **No frozen experiment artifacts have been copied to DST.** No `artifacts_frozen/`.
- **DST git has not been initialized** — `demap_repro/.git` does not exist.
- Nothing committed. No remote.

DST currently contains exactly five permanent files plus `.scratch/` and this handoff.

---

## 2. Authoritative metadata files

All paths relative to `demap_repro/`.

| path | sha256 |
|---|---|
| `manifests/paper_scope.yaml` | `378c6daa76276fbf5ce52b0ea7a957a260485db7f11d9ad8c28e0c766ccbd8ab` |
| `manifests/source_migration.yaml` | `a62f4137fb83249908abe232d53b92695df6c1091e386a72827eedd6e962c326` |
| `manifests/expected_results.json` | `0e172f6f0570dc7b5feeab1601dabf96d46b5382ed27aac3799ecb3e17c55b6c` |
| `configs/paper/results_provenance_v1.yaml` | `a4f57aadd63ed559ca15299ff8364045791a443469370a799462513cee84e257` |
| `docs/provenance.md` | `c923d88caa4d0a41ca6acef41f6180c1e38af54f100f98316f457bc79d1c178f` |

**Counts:** 42 paper_scope entries · 145 migration entries (80 tracked, 4 tracked-modified, 27 untracked, 34 scratch) · 20 gated entries (recorded, never copied) · 17 expected-results groups · 20 provenance rows.

Builders and the validator live under `.scratch/slice1/` (`scope_data.py`, `migration_data.py`, `build_manifests.py`, `build_expected.py`, `build_provenance.py`, `validate.py`). Rebuild with:

```bash
cd /data/nextgen2/james/tasks/cde_project/demap        # SRC, read-only; venv lives here
P=/data/nextgen2/james/tasks/cde_project/demap_repro/.scratch/slice1
.venv/bin/python $P/build_manifests.py && .venv/bin/python $P/build_expected.py \
  && .venv/bin/python $P/build_provenance.py && .venv/bin/python $P/validate.py
```

Last run: **30/30 checks PASS**. Note the hashes shift on every rebuild because each manifest stamps `generated_utc`.

---

## 3. Scientific decisions already locked

Public scope is defined by **v21 only**. An experiment not represented in v21, and not required to reproduce something in v21, does not belong in `demap_repro`.

- **Production catalog** = 62,976 eligible CDE records (`data/processed/cadsr_xml_2026-06-18/cde_master_enriched_eval_production_cde_match.parquet`, sha256 `2e360368…`, 62,858 unique public ids, 0 retired).
- **Retained bi-encoders** = all-MPNet, BioSimCSE, PubMedBERT.
- **Phase 2 strategies** = `none`, `hard_top25`, `semihard_1_50` only. **`hardcurr` excluded** (protocol v2 `excluded_strategies`).
- **Exact-match pinning excluded** — removed from every manuscript-facing artifact as of v16. Distinct from the *allowance-rate* sensitivity of S6.2, which IS in the paper.
- **Final candidate pool** = FT-MPNet top-20 ∪ CDE Match-Fuzzy top-10, deduplicated by CDE public identifier.
- **Selected nominal K = 30** (selected on `val_train`; `val_dev` descriptive only).
- **Selected cross-encoder = FT-MedCPT** (val_dev R@5 0.9123 > BGE 0.9037 > MiniLM 0.8742).
- **Final HGBC = 117-feature no-provenance model** — `artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov/`; excluded exactly 5 features (`keyword_rank`, `cdematch_rank`, `log1p_cdematch_rank`, `text_family`, `text_query_source`); hyperparameters `max_iter 200 / max_depth 3 / lr 0.05 / min_samples_leaf 30 / random_state 42`, selected from the frozen 16-config grid on val_dev Recall@5; ties broken by ascending CDE public identifier.
- **HGBC provenance / A-B selection experiment** is historical provenance only. Not a public runnable experiment, no `AB_SELECTION` replay, no expected-results group. Its single manuscript trace is the S5.5 sentence *"A prespecified follow-up comparison selected the feature set without query-provenance variables."*
- **Figures 1 and 6** are manually maintained schematics — validate-only. DST code checks the claims they encode (117 features, top-20/top-10 pool, ≤30 candidates, public-id tie-break, catalog counts); it does not regenerate them from experiment output.
- **Figure S7 is present in v21** (`word/media/image13.png` ↔ `figureS7_allowance_fixed_vs_retrained_v19.png`) and remains in public scope, along with its J/K allowance implementation paths.
- **The final six-method evaluation (Table 4)** remains the primary end-to-end comparison: final reranker, Python approximation to NCI CDE Match, BM25, FT-MPNet, CDE Match-Fuzzy, FT-MedCPT top-30 pool reranking.

**Manuscript structure is intentional:** `1 Introduction · 2 Methods · 3 Results · 4 Discussion and Conclusion`. There is intentionally **no standalone Related Work section** — Related Work was merged into the Introduction in v20. Do not flag this as a regression or alter scope because of it.

---

## 4. PV-summary audit (Slice 1.6) — IMPORTANT

### NO PV REGRESSION DETECTED.

The current implementation

- `src/demap/text/pv_summary.py` — sha256 `7186aed5e43eb49c46101ce3ed0fcc1448bfa3af3d0ea2f11d9998a8a6d25945`
- `src/demap/text/pv_summary_v2.py` — sha256 `74ad3443c8617e5c939ab44327ec197ad49dfa40096b958df7e348b079c37d20`

**is the same implementation that produced the paper-era build.** Evidence: exactly one commit each (`13821ac`, 2026-05-12 13:32); worktree clean vs HEAD; `pairs.parquet` written 2026-05-13 14:43 (sha256 `1664cc53…`); the only later commit touching the chain (`c651f44`, 2026-07-05) adds the `catalog-filter` stage and contains zero PV-related lines.

Call chain: `inwild_queries.py:2065` → `pv_summary.build_pv_summary_table` → `pv_summary_v2.build_pv_summary_tables`. `build_pairs.py` only carries the columns forward. Parameters are recorded in `data/processed/dataset_build_manifest.json` (`pv_max_n_query 8`, `pv_max_n_cde 10`, `sde_generic_label_p 0.30`, salt default `demap`, `pv_center_fraction 0.5`, `pv_min_n 2`, `pv_size_jitter 1`, `pv_generic_boost 0.10`, `pv_generic_cap 0.70`, `pv_small_n_generic_threshold 0.5`, `pv_max_resample_attempts 3`). That manifest also records `dirty: true, dirty_paths: 2` — the build ran from a tree with 2 unrecorded uncommitted files; the regeneration test below empirically rules them out of the PV chain.

### Safeguards — all verified present and firing

Exercised live on the paper CDE set (14,590 groups) with the manifest's exact parameters:

| safeguard | evidence |
|---|---|
| Query-side cap **lower** than CDE-side cap | `pv_max_n_query 8` vs `pv_max_n_cde 10`, enforced in `compute_side_sample_size`; observed max `k_query` = 8, max `k_cde` = 10, never exceeded |
| Generic-label handling | `_query_render_generic_as_label`, deterministic SHA-1 bucket hash at p = 0.30 |
| Source/code-oriented query rendering | `render_query_side` prefers `valid_value` (the code); `render_cde_side` always uses `preferred_meaning_label` |
| Small-PV-set policy | `apply_small_n_policy` fired on 2,709 / 14,590 groups (18.57%): `n1` 191, `n2_non_generic` 305 omitted; `n2_all_generic` 2,265, `n3_generic_dominant` 2,103, `n3_non_generic_dominant` 530 retained-with-rule |
| Deterministic anti-identity retry → omission | `apply_anti_leakage`, up to 3 reseeded resamples then omission; fired on 2,749 groups; identity 18.84% pre → 15.17% residual (those are emptied) |
| Subset/overlap safeguard | `_items_strict_subset` guard for n≤3 non-generic-dominant with query length 1–2, plus an unconditional strict-subset guard under `gdc_strict`. **Fires 0 times under the `cadsr` profile** used for the paper — matching the paper-era diagnostic's own `subset_trigger_rate: 0.0`. Present but inert for caDSR; active for GDC |
| Deterministic seeding + parameters | `_seed_base` = salt∷publicid∷version∷side (SHA-1); jitter ±1; generic boost 0.10 / cap 0.70; `use_placeholder_for_query_omission=False`, placeholder rate 0.0 |

### Row-level parity — full dataset, no sampling

Regenerated from `data/interim/cadsr_merged/cde_permissible_values.parquet` (719,304 rows → 402,080 over 14,590 groups after restriction to the benchmark CDE set) and compared against the frozen `pairs.parquet` (69,102 rows):

- **query-side (`PV_BLOCK_SDE`): 46,948 / 46,948 exact (100.000000%)**
- **CDE-side (`PV_BLOCK_CDE`): 46,948 / 46,948 exact (100.000000%)**
- changed rendered strings: **0** · changed item sets: **0**
- missing→present: **0** · present→missing: **0**
- `PV_TYPE`: 46,948 / 46,948 exact · `PV_N`: 46,948 / 46,948 numerically identical

The other 22,154 rows are CDEs with zero upstream PV rows; in the frozen file all have `PV_N = 0`, both blocks empty, `pv_attached = False`. Verified — not a coverage gap.

Additional control: the diagnostics computed from **frozen** vs **regenerated** blocks are identical to full float precision across all six PV-type groups.

### Consequence

**The current PV-generation implementation is the future `demap_repro` source of truth.** No paper-producing variant needs separate preservation — they are the same bytes. Registered in `source_migration.yaml` with `provenance_status: verified_equivalent_slice2_pre_audit`.

### Regression tests still missing (to be added in DST, not yet implemented)

- row-level parity against a frozen PV fixture;
- Table S3 aggregate parity;
- direct tests of each safeguard: query vs CDE display caps, deterministic generation, small-PV-set handling, identical-block avoidance, source-like/code-oriented rendering, subset/overlap safeguard, ENUM behaviour, BINARY_WITH_UNKNOWN behaviour, BINARY_WITH_NA behaviour.

Existing SRC coverage: `tests/test_pv_summary.py` (4 tests — side-specific caps, codes vs labels, identity suppression, `gdc_strict` omission, adapter schema), plus `test_pv_parser.py`, `test_pv_frozen_diagnostics.py`, `test_pv_attachment_build_queries.py`, `test_pv_overlap.py`. **No** row-level or aggregate parity test exists anywhere.

---

## 5. Table S3 issue — PENDING USER DECISION

**This is the main unresolved manuscript/data-lineage issue. Do not act on it without explicit approval.**

Table S3 was generated on **2026-04-23 15:48** from a `pairs.parquet` build of **69,844 rows** (both-present 39,391). That file was **overwritten** on 2026-05-13 14:43 by the paper-era build of **69,102 rows**, which is what is on disk now and what every downstream stage consumed.

Producer chain (confirmed, unchanged): `pairs.parquet` → `src/demap/analysis/pv_frozen_diagnostics.py` (CLI `demap pv-frozen-diagnostics`) → `artifacts/summaries/pv_frozen_diagnostics/query_cde_pv_pair_records.csv` → `notebooks/07_dataset_statistics_tables_and_figure.ipynb` cell 13 → `artifacts/tables/table04_pv_overlap_query_vs_cde.csv` (sha256 `1b0a5522…`). `PV_DIAG_DIR` resolves to `pv_frozen_diagnostics`, **not** the `_with_gdc` variant.

| group | v21 as printed | recomputed from the canonical 69,102-pair dataset |
|---|---|---|
| overall | 39,391 · 0.4% · 30.1% · 0.168 | **38,964** · 0.4% · 30.1% · 0.168 |
| ENUM | 24,543 · 0.4% · 45.2% · 0.153 | **24,158** · 0.4% · **45.3%** · 0.153 |
| BINARY_WITH_UNKNOWN | 3,074 · 0.1% · 0.0% · 0.333 | **3,040** · 0.1% · 0.0% · 0.333 |
| BINARY_WITH_NA | 8,816 · 0.0% · 0.0% · 0.123 | **8,815** · 0.0% · 0.0% · 0.123 |

11 of 12 rate cells still round to the published values; all four row counts differ.

**This is NOT PV-code drift.** It is an input/artifact-lineage issue confined to descriptive Table S3 reporting. **No downstream model result is affected** — every model stage consumed the frozen `pairs.parquet` PV blocks, which current code reproduces exactly.

Note the internal tension it creates: v21 §3.1 and S1.3 report **69,102** constructed pairs (the current file), while Table S3's denominators come from the 69,844-row build.

**Status: PENDING USER DECISION BEFORE FINAL PUBLIC-REPO FREEZE.**

The likely correction, **if approved**, is to update Table S3 to the values recomputed from the canonical 69,102-pair dataset and update the accompanying S3.5 prose denominator from 39,391 to 38,964. **Do not change the manuscript without explicit user approval.**

Supporting outputs (read-only, under DST): `.scratch/pv_regression_audit/` — `regen.py`, `recompute_s3.py`, `control_s3.py`, `row_level_parity.json`, `table_s3_from_current_code.csv`, `table_s3_control_frozen_pv.csv`, `regen_diagnostics.parquet`, `control_verdict.json`.

---

## 6. Other important provenance findings

**`eval_canonical_v2` producer — RESOLVED.**
`.scratch/demap/paper_v11_validation/build_eval_canonical_v2.py` (sha256 `1c2c004a03576720520312ba4b8c90577bb83dfdc269bb77c0dda64e868bc6b3`), Rule E:

```
exclude external query q iff EXISTS train pair t with
      norm(q.query_text_raw) == norm(t.query_text_raw)
  AND publicid(q.cde_publicid) == publicid(t.cde_publicid)
   OR q.query_id in train.query_id
norm(x) = collapse_ws(re.sub(r'[^a-z0-9]+', ' ', lower(x)));  publicid(x) = x.split('::')[0]
```
Whole-query removal; `test` passes through unchanged; refuses to overwrite outputs.

Reproduces exactly: **CCTG 1,089 · OID ALT 1,759 · CDASH 310 · GDC 68 · CIMAC 113 · Test 3,959 (passthrough)**. **All 51 removals fired on the normalized-query-text + gold-CDE recurrence clause**; the `query_id_in_train` clause never triggered, so the effective criterion is exactly the one stated in v21 S1.5. `artifacts/manifests/eval_canonical_v2/removed_queries.csv` (51 rows) is the DST regression fixture.

**`_build_data2.py` — `historical_reference_stale`, `copy_verbatim: false`.**
`.scratch/demap/paper_v10_revision/data2/_build_data2.py` **must not become production reproduction code.** Its sibling `provenance.json` (git head `fab9719`) proves it reads the pre-correction roots `artifacts/final_reranker/{crossencoder, crossencoder_fulltrain}/`. The five CSVs it once wrote were overwritten by the v13 correction (`*.pre_v13.bak` are the pre-correction versions); the builder was never updated. **Re-running it would restore pre-v13 Figure 4/5 values** (MedCPT 0.9128 / BGE 0.8940 / MiniLM 0.8699 and the pre-correction keyword values).

Future Figure 4/5 reporting code must be **newly written in DST** from these corrected roots, then parity-tested against the current v21 figure-input CSVs:

| output | corrected source |
|---|---|
| `crossencoder_finetuned.csv` | `crossencoder_fulltrain_v2_eligible/comparison/BAKEOFF_ce_selection_val_dev.csv` (chain F3, Slurm 26337570–26337572) + val_train slice-bakeoff rows from `crossencoder_v2_eligible/comparison/` |
| `crossencoder_offtheshelf.csv` | `crossencoder_v2_eligible/offtheshelf_epoch0/comparison/crossenc_eval_by_split_SN_DEC_DEF_PQT_PV_SN_DEC_DEF_PQT_PV_<backbone>__epoch0.csv` (Slurm 26469876) |
| `keyword_selection.csv` | `paper_v13_scientific_audit/keyword_selection_val_dev_v13.json` + `keyword_fuzzy_v2_eligible/…summary_val_dev__allow0.70.json` + `cde_match_clone/…summary_val_dev__allow0.70.json` |
| `keyword_fullset_recall5.csv` | `paper_v13_scientific_audit/kwfuzzy_corrected_metrics.csv` + `keyword_fuzzy_v2_eligible/…summary_<dataset>__allow<rate>.json` |
| `nonexact_subset_recall5.csv` | `non_exact_subset_eval_v2_eligible/method_metrics_non_exact.csv` |

---

## 7. Important nontrivial migration sources

Class: **T** tracked · **M** tracked-modified · **U** untracked · **S** scratch · **G** gated · **H** stale/historical. All SRC-relative; all `migration_state: not_copied`.

| component | source | class |
|---|---|---|
| **Canonical CIMAC chain (4 steps only)** | `scripts/build_cimac_appendix_a_v2_split.py` → `scripts/build_reachable_eval_splits.py` → `scripts/build_corrected_cimac_pv_split.py` → `scripts/materialize_canonical_eval.py`. Input `data/raw/cimac/CIMAC-CIDC_Master_AppendixA_v2.xlsx` is **U + G** (the git-tracked `…AppendixA.xlsx` is the superseded v1, NOT the paper input) | T + G |
| **PV-summary generation** | `src/demap/text/pv_summary.py`, `pv_summary_v2.py` — verified equivalent (§4) | T |
| **Deep FT-MPNet retrieval (B5b)** | `scripts/retrieve_biencoder_deep.py` (worktree ≠ HEAD: adds catalog-embedding cache + GPU→CPU ladder) | **M** |
| **Catalog embedding cache** | `src/demap/retrieval/catalog_embedding_cache.py` + `__init__.py` — **never committed**, yet imported by the modified tracked script | **U** |
| **Phase 1 / Phase 2 code** | `src/demap/paper/biencoder/` (11 modules) + `src/demap/experiments/{finetune_phase1,finetune_phase2,mine_hardneg,modern_trainer,baseline_grid,eval_checkpoint}.py` | T |
| **Phase 1 / Phase 2 configs** | `configs/paper/{biencoder_protocol_v2,phase1_winners_final_v2,phase2_winners_final_v3}.yaml` — contain absolute `/data/nextgen2` checkpoint paths, must be rewritten relative in DST | T |
| **K selection** | `.scratch/demap/final_reranker_steps_cd/k_selection_recheck.py` — **executed producer** of `k_selection_grid.csv`, no canonical equivalent. Two near-duplicates retained for reconciliation | S + G |
| **CE pool construction** | `.scratch/demap/paper_v13_scientific_audit/{build_ce_pool_train_v2,build_ce_pool_se20_kwfuzzy10_v2}.py` | S + G |
| **172-query CE exclusion** | `.scratch/demap/paper_v13_scientific_audit/chain_F2_pool_pairs_fulltrain_v2.sbatch` — the removal exists **only inside this chain job**; must become an explicit testable step in DST | S + G |
| **CE selection** | `.scratch/demap/paper_v13_scientific_audit/chain_F4_ce_select_fulltrain_v2.py` → `CE_WINNER.json` | S + G |
| **Fixed-K feature table** | `scripts/build_fixed_k_feature_table.py` (worktree ≠ HEAD: adds `--hydration-policy`) | **M** + G |
| **Split-routing correction** | `.scratch/demap/paper_v13_scientific_audit/run_fixed_k_stepG_v2.py` — **governs final-paper parity**; extends `_CADSR_DERIVED_SPLITS` so `cctg/oid_alt/cdash` route to the a0.70 fuzzy tables and receive the a0.70 keyword exact-control; `gdc_combined/cimac_v2` stay external at a1.0. The tracked builder alone would produce different numbers. Merge into the copied builder in DST; **never fix SRC** | S + G |
| **HGBC training / evaluation** | `scripts/train_hgbc_reranker.py`, `scripts/eval_hgbc_reachable_by_dataset.py`, `configs/hgbc_feature_sets/fixedk30_ablation_C_drop_final_rank_noprov.json`, `configs/reranker/determinism_policy_v1.json`, `tests/test_reranker_determinism.py` | T |
| **Table 4 builder** | `.scratch/demap/paper_v13_scientific_audit/build_final_table4_v13.py` — only implementation; emits `old_*` rows that are NOT in the paper and must be dropped in DST | S + G |
| **Leakage sensitivity** | `.scratch/demap/paper_v13_scientific_audit/build_table_s7_v13.py` (manuscript Table S5) | S + G |
| **Allowance sensitivity (Fig S6/S7, Table S6)** | `.scratch/v17_claude/J_allowance/scripts/` (12 files, retrained-per-rate arm) + `manuscript/v17_claude_reports/K_fixed070/scripts/` (8 files, fixed-070 arm; `K2_aggregate.py` builds the single CSV backing all three items) | S + **U** + G |
| **Exact-match mask** | `src/demap/features/exact_match_mask.py` — NOT a CDE Match derivative; releasable independently | T |
| **Figure/report input builders** | `.scratch/demap/paper_v10_revision/data/_build_{phase0,2_3_4,5_6}.py`; figure generators `notebooks/paper_figures/*.ipynb` + `paper_figure_style.py` + `manuscript/figures/make_figure{3_v19,4,5,S3_v21,S6_S7}.py` — all **U**; `_build_data2.py` is **H** (§6) | S + U + H |
| **Table S3 chain** | `src/demap/analysis/pv_frozen_diagnostics.py` (T) + `notebooks/07_dataset_statistics_tables_and_figure.ipynb` (U) | T + U |

---

## 8. Release / redistribution gates — all UNRESOLVED

Eight gates, each naming the single dimension it constrains. Nothing gated has been copied; `copied_to_dst: false` on all 20 gated entries.

| gate | dimension | blocks retraining? |
|---|---|---|
| `nci_cde_match_source` — 7 NCI-supplied PL/SQL + logic PDF under `scripts/cde_match/` | code_redistributable | no |
| `cde_match_derivative_unresolved` — `cde_match_clone.py`, `keyword_retriever.py`, `cdematch.py`, `build_cde_match_candidates.py` | code_redistributable | no |
| `cde_match_derived_artifacts` — clone / CDE Match-Fuzzy candidate parquets | data_redistributable | no |
| `official_nci_cde_match_frozen` — saved live-service output (GDC + CIMAC) | externally_frozen_result | **yes** |
| `cimac_unresolved` — Appendix A workbook and derived n=131 set | data_redistributable | no |
| `cadsr_snapshot_unresolved` — 2026-01-12 and 2026-06-18 nightly exports | data_redistributable | no |
| `finetuned_weights_unresolved` — FT-MPNet / FT-BioSimCSE / FT-PubMedBERT | weights_redistributable | no |
| `medcpt_derivative_weights_unresolved` — FT-MedCPT (+ FT-BGE / FT-MiniLM) | weights_redistributable | no |

**Preserve these eight independent dimensions** on every `paper_scope` item (`yes` / `no` / `unknown` / `not_applicable`, plus `partial` where an item mixes recomputed and frozen columns): `code_redistributable`, `data_redistributable`, `weights_redistributable`, `runnable_from_public_inputs`, `runnable_with_user_supplied_or_private_inputs`, `reproducible_by_retraining`, `requires_precomputed_input`, `externally_frozen_result`.

The distinction that matters most: **`weights_redistributable: unknown` does not imply `reproducible_by_retraining: no`.** Base checkpoints are public HF models and the protocols are documented, so bi-encoder and cross-encoder results remain re-derivable by retraining. A validator check enforces this. Conversely, a saved third-party service result is `externally_frozen_result: yes` and is never runnable — a property of the result, not a gate on our code.

**Do not resolve any uncertain legal/licensing question by assumption.** These need written determinations.

---

## 9. Explicit exclusions

These must remain out of the public scientific workflow unless the user explicitly changes scope. Enforced by the `no_abandoned_experiments_in_scope` validator check (structural scan of scope items, expected-results and provenance; the `policy` declaration string and compliant negative assertions such as `provenance_features_present: false` are exempt).

- HGBC provenance / A-B experiment (`select_hgbc_variant_v2.py`, `compare_hgbc_variants_ab.py`, `AB_SELECTION` replay, `with_ce_stableprov_a*` arms)
- provenance dropout (`src/demap/features/provenance_dropout.py`, `__FALLBACK__`, forced-fallback)
- exact-match pinning (`scripts/rank_fixed_k_exact_pinned.py`, `.scratch/demap/exact_match_precision/`)
- Phase 2 `hardcurr` curriculum arm
- historical wrong-band semi-hard `(1,25)` all-MPNet runs
- SapBERT experiments not in v21
- XGBoost reranker ablation
- non-paper feature ablations (`kwtier3`, `seqkw`, ablations B/D/G)
- stale pre-correction result paths (`hgbc_reranker_medcpt`, `hgbc_features_with_medcpt`, `_superseded_bakeoff_ce_variant`, `with_ce`, `no_ce`, `artifacts/paper_bm25_*`, `artifacts_v3_cdisc/`, `artifacts_v2_balanced/`)
- superseded CIMAC construction / diagnostic / adjudication variants, and the v1 `AppendixA.xlsx`
- **old deployment code inconsistent with the final model**: `src/demap/deploy/final_reranker.py` and `bundle_validation.py` still default to the superseded 119-feature `hgbc_reranker_medcpt/with_ce/`. DST reimplements against `with_ce_noprov` (117). **Do not "fix" these in SRC.**

---

## 10. Copy-first / refactor-second policy

```
identify the exact final-paper implementation
→ copy verbatim into DST                    (SRC untouched)
→ record source path, class, worktree SHA256, HEAD blob, scope ids, parity artifact
→ establish parity against the authoritative artifact  BEFORE any cleanup
→ refactor only inside DST
→ re-run parity
→ accept the refactor only if parity holds within the stated tolerance
```

`migration_state` moves monotonically: `not_copied` → `copied_verbatim` → `parity_established` → `refactored` → `parity_reverified`.

**Never propagate changes back into SRC.** **Actual worktree code — including tracked-modified, untracked and `.scratch` code — may be scientifically authoritative.** When tracked code and a `.scratch` implementation disagree, the implementation that actually generated the authoritative artifact governs parity; the other is recorded as a divergence, optionally retained under a `_reference_` name, and never used to define correctness.

Tolerances (in `expected_results.json`): Level-1 regeneration from frozen artifacts 0.0005; Level-2 retraining 0.01 Recall@5 **and** method ordering preserved exactly; counts exact integers; the 70% allowance anchor must match exactly.

---

## 11. Recommended next session

**Do not perform any of this now.**

1. **Ask the user for a decision on Table S3** (§5): correct it to the canonical 69,102-pair values, or keep the printed values and footnote the earlier build.
2. **If approved:** make only the explicitly approved manuscript / Table S3 correction; update `expected_results.json` and `results_provenance_v1.yaml` accordingly; re-validate manuscript internal consistency (§3.1 / S1.3's 69,102 vs Table S3's denominator).
3. **Then begin Slice 2** — selective copy-first migration of the first bounded scientific component.

**Recommended first Slice-2 component: the BM25 baseline (`SM-S51-01`).** It is the right shape to establish the pattern safely:

- fully paper-scoped (S5.1, and one Table 4 column);
- **not legally gated** — the only lexical method with no CDE Match derivation, `code_redistributable: yes`, no weights, no CIMAC dependency;
- small and self-contained — `src/demap/paper/bm25/` (4 modules) + `src/demap/experiments/{bm25_baseline,bm25_quarantine}.py` + `configs/paper/bm25_protocol_v1.yaml`, all **tracked and clean**;
- has a SHA256-pinned catalog and a fail-closed quarantine already built in;
- immediate parity targets exist: `artifacts/bm25_canonical_v1/{bm25_val_dev_selection.csv, bm25_canonical_metrics.csv}` and the six Table 4 BM25 cells (Test 0.361 · CCTG 0.392 · OID ALT 0.175 · CDASH 0.710 · GDC 0.542 · CIMAC 0.160);
- exercises the whole loop — copy verbatim → hash → parity → refactor → re-parity — without touching any gate.

Second candidate if BM25 lands cleanly: the **PV-summary generation chain** (§4), which is now the best-verified component in the repository.

---

## 12. Integrity / stop state

Verified at the end of this handoff task:

- **SRC HEAD:** `cf8743f535f4f4fb8eb7bcc71362986d615ec58f` — unchanged.
- **SRC porcelain status:** 97 lines (10 modified tracked + 87 untracked) — byte-identical to the session start.
- **SRC files modified:** 0. **SRC files created:** 0. Only read-only git queries were run.
- **DST `.git`:** does not exist.
- **Scientific code copied during this task:** none.

Writes made by this task: this handoff file only.
