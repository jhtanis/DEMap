# Release readiness

Status as of 2026-08-08. The repository is **private and unlicensed**. This
document records four audits that a reviewer should be able to check
independently: what was deliberately not migrated, what the accidental writes to
the research repository touched, why one previously-gated file was reclassified,
and what each remaining gate actually rests on.

---

## 1. License

**License: to be determined.** `pyproject.toml` previously declared MIT; that was
never a decision anyone made — it carried over from the research repository's
packaging and has been removed. There is no `LICENSE` file, and none should be
added until the team decides.

---

## 2. Non-migrated ledger entries

150 candidates. 131 migrated, 19 deliberately declined. Every declined entry
carries a written reason, and a test (`test_no_migration_entry_is_left_undecided`)
fails if any entry is left merely uninspected.

### The audit found a real gap, which has been fixed

Five paper items had **no code in this repository that produced them**, despite
being marked "superseded":

| item | what was wrong | fix |
|---|---|---|
| Figures 2, S1, S2 | reason claimed a `manuscript/figures` generator superseded the notebook; no such generator exists — the notebook was the only producer | migrated to `reporting/figures/make_figure2_S1_S2_heatmaps.py` |
| Figure S4 | reason confused the *input table* builder with the figure | migrated to `reporting/figures/make_figureS4_stage_comparison.py` |
| Figure S5 | K selection was migrated; the *figure* was not | migrated to `reporting/figures/make_figureS5_k_ceiling.py` |
| Table S3 | reason claimed tests covered it; no DST code performed the grouping | migrated to `reporting/dataset_tables_pv_overlap.py` |

All four were re-migrated under the standard loop and reach
`parity_reverified`:

- the three figure generators reproduce the **published provenance records
  exactly** — same input hash, same geometry, same fonts, same recorded anchor and
  best cells;
- the Table S3 chain (`pairs.parquet` → `pv_frozen_diagnostics` → grouping) now
  reproduces, end to end, the table the surviving 69,102-row benchmark implies:
  38,964 / 24,158 / 3,040 / 8,815 with ENUM query-shorter 45.3%. The manuscript
  prints the frozen 2026-04-23 run instead (39,391 / 24,543 / 3,074 / 8,816); see
  `A_pv_overlap` in `manifests/expected_results.json` for why both are kept.

`tests/tier1_invariants/test_paper_scope_coverage.py` now asserts that every
reported figure and table maps to a module that exists, so this class of gap
cannot reopen silently.

### The 19 that remain declined

| classification | n | entries |
|---|---|---|
| `superseded_by_hardened_code` | 10 | `_build_data2.py`; two near-duplicate K selectors; `build_final_report.py`; `build_allowance_sensitivity_v13.py`; J5/J6/`compare_fuzzy_070`; K3/K4/K5 |
| `historical/stale` | 1 | `_build_data2.py` is also the canonical stale case: it reads pre-correction roots and would restore superseded Figure 4/5 values |
| `provenance_only` | 4 | `evaluate_non_exact_subset.py`, `build_ce_training_pairs_fulltrain.py` (neither is the executed implementation), `K6_make_tableS6_v19.py` (reimplemented and parity-tested), `figure_repxloss_v10.ipynb` (v21 embeds the `_v21.py` generator, which *is* migrated) |
| `manual/non-executable` | 2 | `validate_v16.py` (DOCX machinery; its scientific assertions are ported into tests), and the two package `__init__.py` shims |
| `gated_external_material` | 1 | `cde_permissible_values.parquet` — gated upstream caDSR input |
| `unused_by_v21` | 1 | `build_allowance_sensitivity_v13.py`, the two-method lexical-only grid v21 replaced |

**Runtime dependency on any of them: none.** **Paper results that would become
unreproducible if the source file vanished: none** — every one is either
superseded by migrated code, a non-executed variant, or a gated input that was
never copyable in the first place.

Two are worth naming for future readers:

- `_build_data2.py` must never be run. Its own provenance record shows it reads
  the pre-correction crossencoder roots; re-running it would restore Figure 4 to
  MedCPT 0.9128 / BGE 0.8940 / MiniLM 0.8699. A test asserts those values do not
  reappear in the migrated inputs.
- `K6_make_tableS6_v19.py` did produce the Table S7 (then numbered S6) that reached v21. Its
  behaviour is reimplemented as `sensitivity/allowance/report.table_s6()` and
  parity-tested against the shipped 120-row aggregate.

---

## 3. Accidental writes to the research repository

While migrating, importing modules whose bodies executed at import time caused
**17 files under the research repository to be rewritten** across two passes on
2026-08-08. No cleanup has been performed; this is an inventory so the team can
decide whether any is wanted.

**Nothing scientific changed.** All 55 manifest-pinned artifacts verify unchanged.
Git HEAD is `cf8743f535f4f4fb8eb7bcc71362986d615ec58f` and porcelain status is 97
lines — byte-identical to the session start. Every affected file is **untracked**,
so git state was unaffected.

| group | n | tracked | pinned | verified |
|---|---|---|---|---|
| `.scratch/…/paper_v13_scientific_audit/*.csv` (Table 4, Tables S1/S2) | 3 | no (ignored) | **yes** | **byte-identical** — SHA256 matches the recorded value exactly |
| `.scratch/v17_claude/J_allowance/rates/a0.70/PARITY_FEATURES.json` | 1 | no (ignored) | no | regenerated; reported 0 differences on all six CE columns |
| `manuscript/figures/figure{5,S3,S6,S7}*.{png,pdf,json}` | 12 | no | no | rebuilt from inputs whose hashes match the pinned artifacts; plotted values printed and matched the published ones |
| `manuscript/v17_claude_reports/K_fixed070/VALIDATION_sensitivity_v19.md` | 1 | no | no | regenerated; reported ALL CHECKS PASS |

All 17 existed before and were **overwritten, not created**. The three pinned CSVs
are byte-identical, so for those the rewrite is a no-op beyond mtime. The figure
renders are PNG/PDF, which are not byte-reproducible (rendering embeds
timestamps), so their bytes almost certainly differ; their provenance sidecars now
read `generated: 2026-08-08` and record input hashes identical to the pinned
artifacts.

**The manuscript is unaffected.** v21 embeds its own copies of every figure; the
on-disk renders are sources, not the document.

**Cleanup assessment: unnecessary.** The pinned files are byte-identical; the rest
are regenerable build output that reproduces the same content. Restoring the
original bytes is not possible for the raster/vector renders and would not change
any reported result. Full record: `.scratch/src_write_inventory.json`.

**Recurrence is now prevented.** `tests/tier1_invariants/test_import_safety.py`
fails if any module can read, write or fit at import time, and every module in
this repository passes.

---

## 4. `cdematch.py` — reclassified as independent

**Classification: A — independent generic downstream feature code.**

Previously gated under `cde_match_derivative_unresolved` on the strength of its
name and directory. A content review does not support that.

**What it does.** Reads four columns — `query_id`, `cdematch_rank`,
`cdematch_score`, `in_cdematch_topk` — and emits seven features: score divided by
a constant, `log1p(rank − 1)`, the query's top-1 score, margin to top-1, margin
between ranks 1 and 2, a within-query z-score, and a candidate count. That is
standard ranked-list arithmetic.

**Evidence it is not a derivative:**

1. **It never sees CDE Match's inputs or logic.** No CDE text, no catalog field,
   no rule set, no string comparison, no field weighting. It consumes an
   already-produced ranked list.
2. **No trace of the supplied material.** Zero occurrences of SQL, Oracle, NCI,
   `matchFlow`, `longestWord`, `pvvm`, or any supplied artefact.
3. **Its history is a feature-engineering change,** not a port: one commit,
   `84d51c9` "Add CDE Match derived features (PR-C)", 2026-05-19.
4. **Decisive: it was already reused on a different retriever.**
   `build_fixed_k_feature_table.py` applies the same function verbatim to the
   *keyword* retriever's output and renames the results to `seqkw_*`. A function
   containing CDE Match logic could not do that.

The one CDE Match-specific constant is `CDEMATCH_SCORE_MAX = 100.0`, an observed
property of the upstream score range rather than algorithm internals.

**Consequence.** It is now `src/demap_repro/reranker/features/cdematch.py` and the
gated adapter no longer brokers it. Two of the four previously-gated symbols are
gone; `compute_all_features` and `feature_summary` now match their source AST
exactly rather than through a declared alias.

The genuinely derivative components — `cde_match_clone.py`,
`keyword_retriever.py`, and the NCI PL/SQL — **remain gated and uncopied.** This
review deliberately did not reassess them.

---

## 5. Remaining release gates

All project **data are public**. caDSR is a public NCI registry; CIMAC-CIDC, GDC
and the CDISC-derived sets are public resources. No gate below exists because
data is private.

| gate | material | concerns | why still open | in DST git history? | repo functions without it? |
|---|---|---|---|---|---|
| `nci_cde_match_source` | 6 NCI-supplied PL/SQL files + the CDE Match logic PDF | source-code redistribution | material was supplied to us for internal use; we hold no redistribution right and have not asked | **no** | yes — never referenced by any DST module |
| `cde_match_derivative_unresolved` | `cde_match_clone.py`, `keyword_retriever.py`, `build_cde_match_candidates.py` | derivative-code redistribution | they reconstruct the supplied logic in Python; whether we may publish a derivative is undetermined | **no** | yes — reached through a documented adapter; stages accept a precomputed candidate artifact |
| `cde_match_derived_artifacts` | clone and CDE Match-Fuzzy candidate parquets | artifact redistribution | outputs of the above; releasable status may differ from the code's, and neither has been determined | **no** | yes — downstream stages consume them as inputs the user supplies |
| `official_nci_cde_match_frozen` | saved output of the live NCI CDE Match service (GDC, CIMAC) | artifact redistribution + attribution/terms | a third-party service result; publishing it is a question for NCI, not us | **no** | yes — reported as a frozen external number, never runnable |
| `cimac_unresolved` | CIMAC-CIDC Appendix A workbook and the derived n=131 set | artifact redistribution | the workbook is public, but our redistribution of it and of the derived evaluation set has not been confirmed with CIMAC-CIDC | **no** | yes — a user holding the workbook runs the CIMAC stages locally |
| `cadsr_snapshot_unresolved` | 2026-01-12 and 2026-06-18 caDSR nightly exports | artifact redistribution | caDSR is public, but these are nightly snapshots rather than versioned releases; republishing a snapshot as if it were a release is the open question | **no** | yes — digests are pinned so a user can confirm they hold the same file |
| `finetuned_weights_unresolved` | FT-MPNet, FT-BioSimCSE, FT-PubMedBERT | model-weight redistribution | ours to publish in principle; no determination has been made | **no** | yes — retraining reproduces the result from public base checkpoints |
| `medcpt_derivative_weights_unresolved` | FT-MedCPT, FT-BGE, FT-MiniLM | model-weight redistribution | fine-tuned from third-party checkpoints whose licenses may constrain redistribution of derivatives | **no** | yes — same; retraining is documented |

**Verified: 0 gated blobs anywhere in DST git history**, checked by hashing every
object reachable from every ref against the gated file set.

The distinction that matters most: an unresolved **weights** gate does not imply
the result is irreproducible. Base checkpoints are public and protocols are
specified, so every learned component remains re-derivable by retraining.

---

## 6. Runtime independence

- No module imports from the research package.
- No runtime path executes code under `SRC/.scratch/`.
- No runtime path executes code in a manuscript revision directory.
- Absolute research-repository paths appear only in provenance and documentation
  strings, never as a runtime input; every stage resolves data through
  `DEMAP_DATA_ROOT` or an explicit `--` argument.

---

## 7. Before team review

1. **Licensing decision** — the only item that blocks anything.
2. **Redistribution determinations** for the eight gates above, in writing.
3. Optional: decide whether the 17 rewritten research-repository files warrant any
   action. The recommendation is none.
