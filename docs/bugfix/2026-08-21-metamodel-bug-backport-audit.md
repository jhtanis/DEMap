# Backporting the metamodel line's bug inventory to the paper repository

**Date** 2026-08-21 · **Starting HEAD** `55e8765` · **Branch** `main`

The later metamodel work ran this codebase down paths the migration to
`jhtanis/DEMap` never exercised, and recorded what broke in
`DEMap_metamodel_research/docs/metamodel_bug_inventory.md`. That inventory is
explicit that it authorises nothing:

> nothing is backported to `jhtanis/DEMap` in this session, and no entry here
> authorises one

This document is that separate, reviewed pass. It decides, entry by entry,
whether a defect found while building a **different system** applies to the
system this repository reproduces:

    FT-MPNet top 20  +  CDE Match-Fuzzy top 10
        -> deduplicated K≈30 pool
        -> FT-MedCPT features
        -> HGBC over 117 features

The manuscript used as the claim contract is the non-metamodel August 6 paper,
*Hybrid Semantic–Lexical Retrieval for Source-to-CDE Mapping in the NCI caDSR*.
It was verified to be the non-metamodel version before use: it reports 117 HGBC
features, the 62,976-record June catalog, and no SapBERT arm. It is **not** the
manuscript the manifests pin as authoritative, and it is one author-approved
correction older than that file; the difference is a single supplementary
denominator and is set out in §5.2.

Nothing metamodel-specific is imported. Not the SapBERT candidate arm, not the
FT-MPNet top-15 + SapBERT top-5 pool, not the 127-feature contract, not the
metamodel-trained HGBC, not the deployment metadata policy (inventory entry S1)
or its model-selection conclusions.

## The headline finding

**No entry in the inventory changes a published number, and none did here.**

Every one of the thirteen software entries is plumbing: a path that resolves
into the installed package, an input that was never shipped, an import of a
script that was never migrated, an API that moved in a dependency. They are the
difference between code that runs from a clean clone and code that only ever ran
on one machine.

That claim is not taken on trust from the inventory. Section 4 below records the
independent recomputation: the production catalog's eligibility filter rebuilt
from the raw 79,827-record export, and the final reranker re-scored from the
frozen HGBC over the frozen 117-feature table on all six evaluation sets. Every
Table 4 value reproduces to floating point (max |Δ| = 1.1 × 10⁻¹⁶), and Table 4
regenerated end-to-end from the per-method artifacts is bit-identical to the
committed fixture across all 54 cells.

---

## 1. Applicability matrix

`Paper?` = does a published number change. `Fixed?` = state in this repository
before this session.

| # | Entry | Applicable here | Fixed? | Layer touched | Paper? | Disposition |
|---|---|---|---|---|---|---|
| 1 | Curator allowlists missing | **yes** | no | dataset build input | no | **fixed** |
| 2 | Registry default path does not exist | **yes** | no | evaluation plumbing | no | **fixed** |
| 3 | Four modules resolve data root inside the package | **yes** | no | paths | no | **fixed** |
| 4 | `by_dataset.py` imports an unmigrated script | **yes** | no | evaluation plumbing | no | **fixed** (guard) |
| 5 | Preferred Question Text filler | partial | n/a | CDE representation | no (category C = 0) | **not backported** — see §3 |
| 6 | `eval_canonical` vs `_v2` ambiguity | partial | partial | evaluation population | no | **fixed** (lock, without metamodel machinery) |
| 7 | Stale June-16 reference restored | **no** | n/a | pruning of that checkout | no | not applicable |
| 8 | `st_loader` pooling on sentence-transformers ≥ 5 | **yes** | no | bi-encoder loading | no | **fixed** |
| 9 | `max_seq_length` floor vs cap | **no** | n/a | — | no | resolved upstream; no change needed |
| 10 | Reachable-denominator mismatch | **yes** | no | evaluation plumbing | **no — adjudicated, §2** | **fixed** |
| 11 | Fifteen modules hardcode the research repo root | **yes** (17 here) | no | paths | no | **fixed** |
| 12 | Undeclared `pptx` / `docx` | **yes** | no | packaging | no | **fixed** |
| 13 | Stale `SPLITS_DIR` default | **yes** | no | paths | no | **fixed** (with 3) |
| S1 | Query allowlist used as deployment metadata filter | **no** | n/a | metamodel index policy | no | not applicable — research design |
| S1b | Empty-ALT boolean mask | **no** | n/a | metamodel `rows.py` | no | not applicable — code absent here |

**14 entries reviewed → 10 applicable and fixed · 1 partially applicable and
fixed (6) · 1 applicable but deliberately not backported (5) · 3 not
applicable (7, 9, S1/S1b, with 9 needing no change).**

One defect was found **here**, by tracing entries 4 and 11 into this repository
rather than by reading the inventory; it is recorded as **N1** below.

---

## A. Applicable and required a change

### 1. The curator allowlists were never migrated

`data/queries.py` resolves `DEFAULT_ALT_ALLOWLIST`,
`DEFAULT_ALT_RECIPE_ALLOWLIST` and `DEFAULT_REFDOC_ALLOWLIST` to
`configs/allowlists/*.csv`, and `manifests/dataset_build_manifest_paper_era.json`
records two of them on the executed build's command line. The directory did not
exist. `demap build-queries` could not run from a clean clone, and the curator's
inclusion rules — which decide the benchmark's composition — could not be
applied at all.

Unlike the metamodel repository, the *path resolution* here was already correct
(`parents[3]`, not `parents[1]`). Only the files were missing.

**Fix.** Copied the three `*_current.csv` files, byte-identical from the research
repository and verified identical to the metamodel repository's copies. Archived
and superseded versions deliberately not copied. Digests pinned in
`demap_repro.data.queries.ALLOWLIST_SHA256`, because an edited allowlist is a
different dataset, not a different run.

**Tests.** `test_backported_plumbing.py::test_curator_allowlists_are_tracked_inputs`,
`::test_allowlist_defaults_resolve_to_the_shipped_files`;
`test_backported_behavior.py::test_curator_allowlists_match_their_schema_and_buckets`
(13 / 5 / 3 rows, schema, uniqueness of the lookup key),
`::test_allowlist_matching_is_case_sensitive_and_literal`.

### 2. The canonical registry default named a path that does not exist

`REGISTRY_PATH = Path('configs/evaluation/canonical_eval_datasets.yaml')`. No
such file; the shipped registry is `configs/paper/eval_datasets_v1.yaml`. It was
also CWD-relative, so it could not work from outside the repository root. Every
no-argument call raised `FileNotFoundError`.

**Fix.** `repo_root() / 'configs' / 'paper' / 'eval_datasets_v1.yaml'`.

**Test.** `::test_canonical_registry_loads_with_no_arguments`.

### 3 and 13. Data roots resolved inside the installed package

`Path(__file__).resolve().parents[1]` meant "repository root" while these files
lived in `scripts/`. After packaging it resolves to `src/demap_repro` or
`src/demap_repro/data`, so every default path underneath pointed into the source
tree. Nothing raised — the paths simply did not exist, and a caller relying on a
default got a `FileNotFoundError` far from the cause.

All four modules from the inventory are affected here:
`biencoder/deep_retrieval.py:46`, `reporting/biencoder_tables.py:45`,
`data/cimac/corrected_pv_split.py:31`, `data/cimac/appendix_a_v2.py:27`.
(`evaluation/by_dataset.py` had already been converted to `data_root()`.)

`deep_retrieval.py` additionally did `sys.path.insert(0, REPO_ROOT / "src")`,
i.e. `src/demap_repro/src` — and, being at position 0, an insert like this can
shadow the installed `demap_repro` with another checkout.

**Fix.** All four use `demap_repro.utils.paths.data_root()`, the documented
`DEMAP_DATA_ROOT` seam. The `sys.path` inserts are gone. Per entry 13,
`SPLITS_DIR` now names `splits_v3_cdisc_reachable_2026-06-18_cimacpv` — the
corrected canonical tree carrying the same split names — instead of the
superseded `splits_v3_cdisc` scheme (test 3,986 against the paper's 3,959),
which had never resolved anywhere real.

**Tests.** `::test_no_module_resolves_its_data_root_inside_the_package`
(parametrised over all four), `::test_deep_retrieval_defaults_land_outside_the_source_tree`,
`::test_splits_dir_names_the_corrected_canonical_tree`.

### 4. `by_dataset.py` imported an unmigrated script

`cimac_strata()` does `from eval_hgbc_cimac131 import ...`, a research-repository
script that was not migrated, surfacing as a bare `ImportError` from inside a
metrics call.

**Fix.** Guarded; raises a `RuntimeError` that names the missing script and says
`by_dataset()` — the function the module exists for — does not need it. Following
the inventory, `eval_hgbc_cimac131` is the one unmigrated script deliberately
left as an import rather than repointed: it has no equivalent here.

**Tests.** `::test_by_dataset_imports_without_the_unmigrated_cimac_script`,
`::test_cimac_strata_explains_itself_when_unavailable`.

### 6. The `eval_canonical` / `eval_canonical_v2` ambiguity — partial

Two directories can plausibly be called "the canonical evaluation set", and
nothing prevented an experiment from picking the wrong one. `eval_canonical_v2`
is the Rule-E train-decontaminated derivative; it is a paper artifact —
`paper_scope.yaml` names it as authoritative for the S6.1 leakage analysis
(Table S5) — but it is **not** the reporting population. Reading it through the
loader would silently swap Table 4's denominators.

**Applies only partially.** This repository already pins all six datasets by
path, `n_queries` and SHA256 in `configs/paper/bm25_protocol_v1.yaml`, and
records the denominators per Table 4 result in
`configs/paper/results_provenance_v1.yaml`. What was missing was *enforcement*.
The metamodel's `eval_registry` is metamodel machinery and was not copied.

**Fix.** `canonical_datasets.assert_paper_evaluation_population()` checks the six
names, the manuscript denominators, and that no entry resolves into
`eval_canonical_v2`, `eval_paper_v1` or `splits_v3_cdisc`.

**Tests.** `::test_shipped_registry_is_the_paper_evaluation_population`,
`::test_a_v2_derived_registry_is_rejected`,
`::test_a_registry_missing_a_paper_dataset_is_rejected`,
`::test_registry_denominators_are_the_manuscript_ones`.

### 8. `st_loader` could not set pooling on sentence-transformers ≥ 5

`apply_pooling_variant` locates the Pooling module by duck-typing
`pooling_mode_cls_token` / `pooling_mode_mean_tokens`. sentence-transformers 5.x
— **5.4.1 is what `pyproject.toml` pins** — replaced the per-mode booleans with a
single `pooling_mode` string, so `__cls` and `__mean` raised
`ValueError: Could not locate a SentenceTransformers Pooling module`.

The failure is loud, not silent, and the paper's own bi-encoder runs used the
`base` variant, which returns early — so no paper result is affected. But any
`__cls` / `__mean` ablation is broken here under the pinned install.

**Fix.** `_is_pooling_module` accepts either generation; the setter writes the
legacy booleans when they exist and the `pooling_mode` string otherwise, never
both, so an inert legacy flag cannot disagree with the mode in force.
`effective_pooling` reads the value back through whichever API is installed.

**Tests.** `::test_pooling_variant_is_applied_and_reads_back` (4 cases: both
generations × both variants), `::test_modern_pooling_gets_no_inert_legacy_flag`,
`::test_base_variant_still_leaves_the_model_alone`,
`::test_a_model_with_no_pooling_module_still_raises`.

### 10. Reachable denominators — the inventory's one OPEN scientific question

The inventory left this OPEN because it "needs someone who knows which artifact
backed `hgbc_eval_by_split.csv` to adjudicate". **It is adjudicated here.**
See §2 for the evidence; the short form is that this module did not produce any
published number, and its default was wrong in two ways at once.

**Fix.** `DEFAULT_REACH_DIR` is now `data/processed/eval_canonical`, the
population declared in `configs/paper/eval_datasets_v1.yaml`. A split present in
the rankings with no query-set parquet is now a `FileNotFoundError` naming the
split, not a printed note; `--allow-missing-splits` opts back in.

**Tests.** `::test_default_query_sets_are_the_canonical_evaluation_population`,
`::test_a_split_with_no_query_set_is_an_error_not_a_note`,
`::test_missing_splits_can_be_skipped_deliberately`,
`::test_the_denominator_is_the_query_set_not_the_rankings`.

### 11. Absolute research-repository paths

`grep` returns **17** files here (the inventory counted 15 in the metamodel
repository): reporting figures, `table4.py`, `sensitivity/allowance/*`,
`sensitivity/leakage.py`, `lexical/non_exact_subset.py`,
`data/characterization.py`, `biencoder/cimac_q2_shim.py`,
`reporting/inputs/build_phase0.py`, `build_phase12.py`. Commit `aa4bd47`
("Remove the last runtime dependencies on the research repository") removed the
runtime *imports*; these absolute *data* paths survived. In a repository intended
for release they are worse than inconvenient: they re-couple it to one tree on
one machine.

**Fix.** All 17 use `data_root()`. This is mechanical and behaviour-preserving:
with `DEMAP_DATA_ROOT` pointed at the research tree every path resolves exactly
as before, which is how the Level A and Level B recomputations in §4 were run.

**Tests.** `::test_no_module_hardcodes_an_absolute_machine_path`,
`::test_no_module_manipulates_sys_path` — invariants over the whole tree, so the
defect cannot return one file at a time.

### 12. `python-pptx` / `python-docx` were undeclared dependencies

`reporting/schematics/_reference_figure6.py` imports `pptx` and
`sensitivity/allowance/validate.py` imports `docx`, both at module level, and
neither distribution was declared anywhere in `pyproject.toml`. **These were the
two failing tests in this repository's baseline** (2 failed, 475 passed, 7
skipped at `55e8765`): `test_import_safety` failed rather than skipped, because
its optional-dependency escape hatch matched a hard-coded substring list that
omitted both.

"Pre-existing" is the wrong disposition. A module whose dependency nobody
declared is a packaging defect, and a loose substring match is what let it pass
unnoticed.

**Fix.** A `documents` extra (`python-pptx`, `python-docx`), included in `all`.
Bounds rather than exact pins: unlike every other pin in that file these were
never installed in the paper environment, so no version can honestly be claimed
to have run. The test now skips only when `exc.name` is in an explicit
`OPTIONAL_MODULE_EXTRA` map, and the skip message names the extra to install.

**Tests.** `test_import_safety.py::test_optional_modules_are_declared_in_pyproject`
(every forgiven module must be pinned under a declared extra) and
`::test_all_extra_includes_every_optional_extra`. Both were verified to fail
against a `pyproject.toml` with the extra removed.

### N1. Three modules still imported unmigrated research scripts — found here

Not in the inventory. Found by tracing entries 4 and 11 into this repository.

Three modules put a research-repository directory on `sys.path` and imported a
script from it by bare name. `bm25/run.py` is the serious one: it resolves
`repo_root() / "scripts"`, which **does not exist in this repository at all**, so
`import evaluate_non_exact_subset` raised `ModuleNotFoundError` and BM25
evaluation — a Table 4 method — could not run from a clean clone.

| module | imported | equivalent here |
|---|---|---|
| `lexical/bm25/run.py` | `evaluate_non_exact_subset` | `lexical.non_exact_subset` |
| `reporting/figures/make_figure5_keyword.py`, `make_figureS3_repxloss.py`, `make_figureS6_S7_allowance.py` | `paper_figure_style` | `reporting.figures.style` |
| `sensitivity/allowance/score_fixed_070.py` | `train_hgbc_reranker` | `reranker.train` |

**Equivalence was verified before repointing**, by AST comparison of every symbol
actually used:

* `prepare_features`, `apply_style`, `panel_letter` — **AST-identical**;
* `eval_method`, `build_gold_map` — differ only in type annotations, dict
  formatting and one defensive `.copy()`; no numeric semantics differ;
* `save_figure` — differs only in how it formats a relative path inside the
  provenance JSON;
* `PALETTE` — identical.

**Fix.** Each repointed at its migrated equivalent. `non_exact_subset` gained a
path-taking `build_gold_map` (the research helper's signature) with `_gold_map`
as a thin by-name wrapper, so the migrated code keeps one implementation.

**Tests.** `::test_no_module_imports_an_unmigrated_research_script` (tree-wide,
with `eval_hgbc_cimac131` explicitly exempt because it is guarded instead),
`::test_bm25_metric_helper_resolves_without_the_research_repository`,
`::test_gold_map_is_public_id_level_any_gold`.

---

## B. Applicable but already fixed here

* **Entry 3, `evaluation/by_dataset.py`.** Already used `data_root()`; only the
  other four modules needed the change.
* **Entry 1, path resolution.** `queries.py` already used `parents[3]`
  correctly. Only the files themselves were missing.
* **Entry 6, the pins.** `bm25_protocol_v1.yaml` already pinned all six datasets
  by path, `n_queries` and SHA256, and `results_provenance_v1.yaml` already
  recorded the per-result denominators. Only enforcement was added. **All six
  pins were verified against the data tree and hold**, along with the catalog
  digest.
* **Allowance routing.** `reranker/split_routing.py` already centralises the
  0.70 / 1.00 decision with the reasoning written down, and
  `tests/tier1_invariants/test_split_routing.py` already guards it. No inventory
  entry touches it, and Level A confirms it is correct.

---

## C. Not applicable to the non-metamodel paper

### 5. Preferred Question Text filler — deliberately not backported

1,783 of the 62,976 production CDEs (2.83%) carry
`"Data Element <LONG_NAME> does not have Preferred Question Text"` as their
`PREFERRED_QUESTION_TEXT`, and that string is in the paper's v3/PQT CDE
representation.

The inventory measured the thing that would make this a correctness defect and
found it is not one. Replicating the paper's selection logic reproduces the
shipped catalog for **100.00%** of rows, and of the 1,783 affected CDEs,
**1,783 have no non-filler PQT anywhere** and **0 had a real PQT available that
was discarded**. The placeholder exists precisely because no real question text
does. There is no suboptimal-selection category.

The inventory's own disposition is *"only as a deliberate representation change,
never as a silent fix"*, and it is a **DIAGNOSED** entry, not a FIXED one — the
metamodel drops filler at its own index-build time and leaves `enrich_master`
untouched so the historical representation stays reproducible. Backporting it
here would change the FT-MPNet CDE representation and invalidate every frozen
bi-encoder artifact, to fix nothing. **Not backported.** It is a candidate for a
future representation, alongside the related observation that
`drop_duplicates(keep='first')` discards 1,352 additional real PQT rows.

### 7. Stale June-16 reference restored

An artefact of pruning that checkout: the metamodel repository deleted the
June-16 tree and had to restore one file. This repository does not prune data.
`appendix_a_v2.py` still resolves CIMAC gold versions against the June-16
catalog, which is **correct and deliberate** — June-16 and June-18 resolve
different current versions, so repointing it would silently change CIMAC gold
versioning. Not applicable.

### 9. `max_seq_length` floor versus cap

Resolved upstream, and resolved *in favour of the paper era*: the research
repository carries the identical floor logic and the same 256 default, and 136 of
136 SapBERT run configs record the resolved value as 512. The floor is the
paper-era behaviour. No change needed.

### S1. Query-construction allowlist reused as the deployment metadata filter

Marked **METHOD**, not a software defect: the row builder faithfully applied the
filter it was given, and the filter was the wrong one *for a deployment
question*. This is exactly the class of later scientific change that must not be
backported. It concerns the metamodel's candidate-side index policy
(`configs/metamodel/deployment_human_metadata.yaml`), a file and a concept that
do not exist in this repository. The inventory records **"Effect on the frozen
paper: None."** Its own backport line reads *"Not applicable — nothing to
backport."*

Worth noting for a future paper as a methods lesson — benchmark-construction
filters leaking into deployment-side design can *understate* a method — but that
is a writing decision, not a code change.

### S1b. Empty-ALT boolean mask

A defect in `demap_repro/metamodel/rows.py`, which does not exist here.

---

## 2. Adjudicating entry 10

The inventory could not close this one. The question is whether
`by_dataset.py`'s `DEFAULT_REACH_DIR` — which yields test 3,968 and refslice 325
against the paper's 3,959 and 324 — represents a corrupted published denominator
or a stale default in a module nothing published used.

**It is a stale default.** Three independent lines of evidence:

1. **`hgbc_eval_by_split.csv` has a different producer.**
   `demap_repro/reranker/train.py:636` writes it, from the feature table, and
   `manifests/expected_results.json` names that artifact as the authoritative
   one. `by_dataset.py` is not in that chain.
2. **The committed fixture already carries the paper's denominators** — test
   3,959, cctg 1,097, oid_alt 1,766, cdash 324, gdc 72, cimac 131 — which are
   the `eval_canonical` counts, not the reachable-tree counts.
3. **The stale directory cannot even name four of the six datasets.** It holds
   `external_holdout_org` (2,882 queries) and `external_holdout_refslice` (325),
   the legacy splits `cctg` / `oid_alt` / `cdash` were later carved from, and
   splits GDC across two files. Under the canonical split names, `by_dataset()`
   would have found no parquet for `cctg`, `oid_alt`, `cdash` or `gdc_combined`
   and **silently skipped all four**, while reporting 3,968 for `test`.

So the default was wrong in two ways at once, and the silent skip is what would
have hidden it. Both are now fixed, and the skip is an error by default.

**No paper number changes.** Level B (§4) confirms it directly: recomputed
against `eval_canonical`, every denominator is the manuscript's.

---

## 3. What was checked and found already correct

The task named several shared bug classes to watch for. Each was traced into the
inventory and into this repository. **No inventory entry reports a defect in any
of them**, and Level A confirms them independently:

| class | status |
|---|---|
| production CDE eligibility / catalog filtering | correct — rebuilt from the raw export, §4 |
| June 18, 2026 production snapshot | correct — pinned by digest, verified |
| 79,827-row export vs 62,976-CDE catalog | correct — arithmetic reproduces exactly |
| exact-match allowance masking | correct — deterministic, nested, seed 42 |
| routing Test/CCTG/OID ALT/CDASH → 0.70, GDC/CIMAC → 1.00 | correct — centralised in `split_routing.py` |
| CDE Match-Fuzzy eligibility, retrieval and fallback depth | no entry; per-rule depth 500 / fuzzy 1000 / merged top 10 as documented |
| canonical reachability filtering | correct — zero unreachable gold CDEs across all six sets |
| query / public-ID deduplication | correct — public-id-level any-gold, verified |
| GDC construction / deduplication | no entry; 72 queries, digest holds |
| CIMAC construction / PV handling | entries 7 and 13 touch its provenance; no defect. 131 queries, digest holds |
| deterministic candidate / rank ordering | no entry; guarded by `test_determinism.py` |
| feature construction from eligible candidates | no entry; 117-feature contract holds |
| research-side counting artifacts | entries 5 and 10, both dispositioned above |

---

## 4. Independent recomputation

A regression test against a frozen artifact preserves whatever produced it. So
the claims above are backed by recomputation from upstream, not by reading the
final CSVs. All of it is committed as
`tests/tier3_regression/test_core_result_contract.py` (18 tests, **4 seconds,
CPU only**) and skips without `DEMAP_DATA_ROOT` / `DEMAP_ARTIFACT_ROOT`.

**Level A — rebuilt from the raw export and the evaluation sets.**

| check | result |
|---|---|
| Query counts, six sets | 3,959 / 1,097 / 1,766 / 324 / 72 / 131 — exact |
| Catalog identity | 62,976 records, 62,858 unique public identifiers — exact |
| Eligibility filter from the 79,827-row export | 16,846 retired · 21 TEST/Training · 16 both · **62,976 kept** — exact |
| Gold reachability | 0 unreachable gold CDEs in all six sets |
| Allowance routing | 0.70 × 4, 1.00 × 2 — exact |
| Mask | nested across 0/50/60/70/80/100, process-stable, realized 0.6964 at 0.70 |
| Pool and feature contract | K = 30, keyword depth 10, `kwfuzzy`, selected on `val_train`; 117 features |
| Registry digests and counts | all six pins hold; catalog digest holds |

**Level B — final reranker re-scored from the frozen HGBC.**

The frozen model is reloaded, `prepare_features` rebuilds the matrix from the
frozen 423,395-row feature table, the 117 included features are selected, and
metrics are recomputed with the trainer's own `_deployment_metrics`.

| dataset | n | R@1 | R@5 | R@10 | MRR@100 | Δ vs frozen R@5 |
|---|---|---|---|---|---|---|
| Test | 3,959 | 0.8957 | **0.9715** | 0.9803 | 0.9289 | 0 |
| CCTG | 1,097 | 0.7976 | **0.9088** | 0.9344 | 0.8471 | 0 |
| OID ALT | 1,766 | 0.7644 | **0.8324** | 0.8533 | 0.7963 | 0 |
| CDASH | 324 | 0.8086 | **0.9198** | 0.9444 | 0.8580 | −1.1e−16 |
| GDC | 72 | 0.9306 | **0.9722** | 0.9861 | 0.9537 | 0 |
| CIMAC | 131 | 0.7328 | **0.8015** | 0.8397 | 0.7669 | 0 |

* Test Recall@5 → **0.971** ✓
* External range → **0.802–0.972** ✓
* GDC → **70 / 72** ✓ (asserted as an integer count, not a rounded rate)

**Table 4 regenerated**, not asserted: `demap_repro.reporting.table4` run
end-to-end over the per-method artifacts is **bit-identical** to
`tests/fixtures/final_table4.csv` across all 54 cells (max |Δ| = 0.000e+00), and
"highest Recall@5 on five of six datasets" recomputes from the regenerated
values, with GDC going to the Python approximation exactly as reported.

---

## 5. Effect on the manuscript

**None.** No manuscript value is known or suspected to change as a result of the
fixes in this session, and no manuscript was edited.

### 5.1 The manuscript pin is correct — do not change it

`manifests/paper_scope.yaml:11` and `manifests/expected_results.json:5` both pin
the authoritative manuscript as:

    manuscript/cde_paper_v21.docx
    sha256 2df7895cd8657a7928be220c958aaa38a85c5b5199526f03979c4967b159eecd
    size_bytes 6159078

**That pin is valid and resolvable.** The file exists at
`manuscript/cde_paper_v21.docx` inside the historical research checkout, which
`paper_scope.yaml:16` declares as `source_repository … status:
immutable_read_only`, and its digest and byte size match the pin exactly. The pin
was never meant to resolve inside this repository — the 2026-08-08 handoff
records the path as being "in the research checkout" — so its pointing outside
the tree is by design, not decay.

### 5.2 The supplied August-6 manuscript is an earlier state, not an equivalent copy

The manuscript supplied for this session, `cde_paper_20260806.docx`
(`sha256 6295986f…`), is **an earlier manuscript state**. It is not a rename, a
re-save, or an equivalent copy of the pinned file.

Compared paragraph by paragraph, the two documents agree on 1,176 of 1,177
paragraphs and carry byte-identical embedded figures (13 media parts, matching
CRCs). They differ in **exactly one checked scientific value**, the Section S3.5
permissible-value overlap denominator:

| manuscript | S3.5 PV-overlap denominator |
|---|---|
| `cde_paper_v21.docx` — pinned, correct | **38,964** |
| `cde_paper_20260806.docx` — supplied, August 6 | **39,391** |

**39,391 is the superseded value.** Per
`docs/handoff/2026-08-07-slice1-provenance-pv-audit-handoff.md`, Table S3 was
generated on 2026-04-23 from a `pairs.parquet` build of 69,844 rows
(both-present 39,391). That file was overwritten on 2026-05-13 by the paper-era
69,102-row build that every downstream stage consumed. The recomputation to
38,964 was **author-approved on 2026-08-08** and is recorded in
`expected_results.json`, `paper_scope.yaml:686` and
`configs/paper/results_provenance_v1.yaml:332`.

**38,964 is this repository's regression target.**
`tests/tier3_regression/test_table_s3_pv_overlap.py:38` asserts `rows: 38964`,
`tests/fixtures/table_s3_corrected.csv` carries it, and
`tests/fixtures/table_s3_pv_overlap_SUPERSEDED.csv` holds the 39,391 values
explicitly marked as historical provenance that "must not be used as a parity
target".

### 5.3 The pin must not be repointed at the August-6 manuscript

Changing `paper_scope.yaml` or `expected_results.json` to name
`cde_paper_20260806.docx` would make the authoritative manuscript carry a value
this repository's own regression test asserts against, and would re-elevate an
artifact the manifests label superseded. **Leave both pins as they are.**

### 5.4 This discrepancy does not touch any headline result

The difference is confined to the Section S3.5 prose denominator in a
supplementary PV-overlap diagnostic. **It does not affect Table 4, the abstract,
the final-system conclusions, or any number recomputed in §4** — Table 4
regenerates bit-identically across all 54 cells, Test Recall@5 is 0.9715, the
external range is 0.802–0.972, GDC is 70/72, and the final reranker is best on
five of six datasets. Reconciling the manuscript file itself is a matter for the
manuscript task; nothing in this repository needs to change for it.

## 6. Frozen artifacts

No frozen artifact was regenerated or overwritten. `expected_results.json`,
`final_table4.csv`, `hgbc_eval_by_split.csv` and every other manuscript-aligned
fixture are unchanged, and all pre-existing tier-3 regression tests still pass —
including, with the research tree as `DEMAP_ARTIFACT_ROOT`, the seven that check
the fixtures against the full-size originals.
