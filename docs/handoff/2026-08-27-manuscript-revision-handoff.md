# Self-handoff — 2026-08-27 — manuscript revision (coauthor comments)

Self-contained. A fresh session can resume from this file alone. Written read-only:
no code, data, result, manuscript or Git state was modified, and nothing was
committed. This file itself is the only new artifact and is left **uncommitted**.

Conventions used below: **[VERIFIED]** = checked against a repository artifact this
session, with the path given. **[INTERPRETATION]** = my reading, not established fact.
**[OPEN]** = needs verification or a human decision.

---

## 1. Repository state

| | |
|---|---|
| repo | `/vf/users/nextgen2/james/tasks/cde_project/DEMap` |
| branch | `main` |
| HEAD | `813b6c4d6bd49c8479b7285c5fccafa6cba4dfd2` — "Correct manuscript provenance in bugfix handoff", 2026-08-21 18:56 -0400 |
| upstream | `origin/main`, **0 ahead / 0 behind** |
| working tree | **clean** (`git status --short` empty) |
| ignored only | `.venv/`, `.pytest_cache/`, `__pycache__/`, `src/DEMap.egg-info/`, `.scratch/` |

**No uncommitted or unpushed work exists.** [VERIFIED]

Research checkout (read-only, referenced throughout):
`/data/nextgen2/james/tasks/cde_project/demap`, HEAD `cf8743f5`.

Test suite state, run this session [VERIFIED]:

```bash
cd /vf/users/nextgen2/james/tasks/cde_project/DEMap
.venv/bin/python -m pytest -q                       # 512 passed, 25 skipped (~20 s, offline)

export DEMAP_DATA_ROOT=/data/nextgen2/james/tasks/cde_project/demap
export DEMAP_ARTIFACT_ROOT=$DEMAP_DATA_ROOT
.venv/bin/python -m pytest -q                       # 535 passed, 2 skipped (~67 s)
```

### Scratch note

`.scratch/audit/` in this repo holds a plain-text extraction of the manuscript body
and its 37 comments, produced this session with stdlib `zipfile` + `ElementTree`
(`.scratch/audit/extract.py`). `.scratch/` is gitignored. Retained deliberately —
it saves the next session re-extracting the DOCX. Delete when the revision closes.
No shared `/tmp` was used.

---

## 2. Current objective

Respond to coauthor comments on the manuscript describing hybrid semantic + lexical
source-data-element-to-CDE mapping.

**Manuscript under revision:**
`/data/nextgen2/james/tasks/cde_project/demap/manuscript/cde_paper_20260806_comments.docx`
· sha256 `c76b728dfe9415e0696c94155c87ae510433f74ce765ebc4cd1a73784a4fe1ae` · 6,174,422 bytes
· 37 comments from Chen, Warzel, Khanna and Yan. [VERIFIED]

**Provenance caveat, already recorded and still true.** `manifests/paper_scope.yaml:11`
and `manifests/expected_results.json:5` pin the *authoritative* manuscript as
`manuscript/cde_paper_v21.docx` (sha256 `2df7895c…`), which lives in the research
checkout. The August-6 file is an **earlier manuscript state**, not an equivalent
copy — see §10 and `docs/handoff/2026-08-21-paper-repo-bug-backport-and-core-revalidation.md`.
Do not repoint either pin at the August-6 file.

---

## 3. Confirmed sharing status — supersedes earlier repository documentation

**Both CDE Match implementations may be shared in the public GitHub repository:**

* the **Python approximation to NCI CDE Match** (`src/demap/features/cde_match_clone.py` in the research checkout);
* **CDE Match-Fuzzy** (`src/demap/features/keyword_retriever.py` plus the entry point `scripts/build_cde_match_candidates.py`).

**Neither is restricted. Do not treat either as gated.**

This decision **supersedes** the following, which still say otherwise and are now stale:

| location | stale claim |
|---|---|
| `docs/release_readiness.md` §5 | gates `cde_match_derivative_unresolved`, `cde_match_derived_artifacts` |
| `docs/handoff/2026-08-08-frozen-demap-paper-repo-handoff.md` §F | "remain gated and uncopied" |
| `manifests/source_migration.yaml` `gated_material` | 3 code entries + 2 artifact-tree entries marked `not_copied` |
| `manifests/paper_scope.yaml` | `release_gate: cde_match_derivative_unresolved` on ~12 items |
| `src/demap_repro/lexical/cde_match_interface.py` | the whole adapter rationale |
| `src/demap_repro/cli.py` | `gated=True` on `non-exact-eval`, `ce-pool-train`, `ce-pool-eval`, `ce-pairs`, `fixed-k-features` |
| `README.md` "Components with unresolved redistribution status" | first table row |

Updating these is real work and is **not** part of the manuscript revision. Do it as a
separate, deliberate change; do not fold it into a manuscript commit.

**Still excluded, unchanged:** saved output of the **live/official NCI CDE Match
service** (a third-party service result), and the six NCI-supplied PL/SQL files plus
the CDE Match logic PDF. Those gates were not part of this decision. [VERIFIED against
the user's instruction, which named only the two implementations.]

### Why this matters technically

CDE Match-Fuzzy is **not** a standalone program: it is `cde_match_clone.CdeMatchClone`
run with `--fuzzy-fallback`, which appends `keyword_retriever` candidates after the
clone's exact-rule hits in a three-tier rank; `FuzzyFallback` is a dataclass defined
inside `cde_match_clone.py`, and the shared allowance gate is the clone's
`ExactMatchControl`. [VERIFIED — `scripts/build_cde_match_candidates.py:36,146-160,219-259`
and `src/demap/features/cde_match_clone.py:629-700` in the research checkout.]

**Consequence:** because *both* may now be shared, this coupling stops being a blocker.
Previously it meant that excluding the approximation also excluded Fuzzy, the candidate
pool, the HGBC feature matrix, Table 4, Figure 5, Figure S5, Tables S5/S6 and Figures
S6–S7. That obstacle is gone.

Related fact worth knowing: **7 of the final HGBC's 117 features are derived from the
Python approximation's ranked list** (`cdematch_score`, `cdematch_score_normalized`,
`cdematch_top1_score`, `cdematch_margin_to_top1`, `cdematch_margin_1_2`,
`cdematch_score_z_within_query`, `cdematch_candidate_count`). The approximation
contributes **no candidates** to the pool (`configs/paper/crossencoder_protocol_v1.yaml`:
`use_clone: false`, `use_clone_or_fuzzy: false`) but it *is* a feature source.
[VERIFIED — `tests/fixtures/hgbc_feature_set.json`.]

---

## 4. The ~69K vs ~63K distinction — exact frozen-study counts

Two coauthor comments turn on this. Comment [3] on the abstract asks for "63,000 (to be
consistent with text below)"; comment [13] asks "isn't the number closer to 63K? that is
how many we are using"; comment [26] notes "on Production the number of nonreleased CDEs
is around 62K". These are about **two different quantities**, and both manuscript numbers
are correct as used.

### A. Constructed source-query → CDE mappings (the ~69K benchmark)

| count | what it is |
|---|---|
| **69,102** | constructed query–CDE pairs from the 2026-01-12 caDSR export |
| **68,659** | after duplicate-pair removal |
| **68,142** | analysis-ready pairs, after canonical dedup + reachable-gold filtering |
| **62,847** | distinct queries in the analysis-ready set |
| 64,346 / 59,484 | in-distribution pairs / distinct queries |
| 3,593 | caDSR-derived holdout pairs (CCTG + OID ALT + CDASH) |
| 203 | independently assembled external pairs (GDC + CIMAC) |
| 51,594 → 50,972 | train pairs, after removing 622 lacking target CDE text |

[VERIFIED — `manifests/expected_results.json` → `results.A_dataset_counts.manuscript_stated`
and `.lineage_recomputed`; asserted as exact integers by
`tests/tier3_regression/test_core_result_contract.py`.]

### B. Production CDE retrieval catalog (the ~63K)

| count | what it is |
|---|---|
| **62,976** | records in the frozen June production catalog — the retrieval universe for **every** method |
| **62,858** | unique CDE public identifiers in it |
| 79,827 | raw record count of the 2026-06-18 caDSR export it was filtered from |
| 16,846 / 21 / 16 | removed: retired · TEST or Training admin context · met both criteria |

Source file: `data/processed/cadsr_xml_2026-06-18/cde_master_enriched_eval_production_cde_match.parquet`,
sha256 `2e360368307c7589325a43eebcbd6f6ec22090530ccf3521862cc1b9cb62a200`, 15,147,443 bytes.
Eligibility mode `production_cde_match`. [VERIFIED — `configs/paper/biencoder_protocol_v2.yaml:29-31`,
`configs/paper/bm25_protocol_v1.yaml:15-21`, `manifests/source_migration.yaml` gated ledger;
the filter arithmetic is rebuilt from the raw export and asserted in
`test_core_result_contract.py`.]

### C. The benchmark-construction catalog (a third number, easy to confuse)

**79,479** records / **78,533** unique public ids, including 15,852 retired — the
2026-01-12 export. It supplied source metadata and gold links; it was **never** a
retrieval universe. [VERIFIED — same sources; manuscript §S1.4.]

**Wording that is supported:** ~69K describes *constructed source-to-CDE mappings*;
~63K describes the *production CDE candidate catalog against which queries are ranked*.
The manuscript already draws this distinction in §1 and §S1.4. [INTERPRETATION] The
comments most likely reflect the two numbers sitting close together without the contrast
being made explicit at first mention, rather than an error.

---

## 5. Outstanding factual questions

### 5.1 Why SapBERT is not one of the three reported bi-encoders

Comment [29], Warzel: *"you are not including SapBert? when you say we, to my knowledge
the nci team did not evaluate BioSimCSE."*

**[VERIFIED] SapBERT was screened and fine-tuned, in an earlier experimental generation,
and was not carried into the manuscript's canonical runs.** Evidence:

* `src/demap_repro/biencoder/constants.py:6-16` — `PAPER_CORE_MODEL_IDS` lists six models
  including `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, which is also
  `PAPER_HEATMAP_MODEL_ID`.
* `src/demap_repro/reporting/biencoder_tables.py:6-7,54` — "All **four** bi-encoders are
  preserved (all-mpnet, sapbert, biosimcse, pubmedbert); the pipeline down-selects to one
  production embedder but the publication comparison keeps all four."
* Historical artifacts exist: `artifacts/phase1_finetuning/sapbert-pubmedbert-fulltext/`
  and `artifacts/phase2_finetuning/sapbert-pubmedbert-fulltext/`.
* A pooling ablation existed: `artifacts/paper_step9_sapbert_pooling/`,
  `configs/experiments/paper_sapbert_pooling_ablation.yaml`, SapBERT [CLS] vs [mean]
  (`src/demap_repro/reporting/representation_tables.py:92-106`).
* **Not present** in the manuscript's canonical roots:
  `artifacts/phase0_representation_sweep_manuscript_v5/` holds only `all-mpnet-base-v2`,
  `biosimcse-biolinkbert-base`, `pubmedbert-base-embeddings`; `artifacts/phase1_canonical_v1/`
  and `artifacts/phase2_canonical_v1/` hold only BioSimCSE and PubMedBERT (all-MPNet's
  canonical grid is the historical `phase1_finetuning/all-mpnet-base-v2` root).
* Explicitly out of scope in four places: `manifests/expected_results.json:10` ("no SapBERT"),
  `docs/handoff/2026-08-08-…:157`, `docs/handoff/2026-08-07-…:260`,
  `docs/bugfix/2026-08-21-…:30-31`.

Historical SapBERT numbers, from
`artifacts/publication_artifact_package_20260713/tables/bi_encoder/biencoder_phase2_best_by_model.csv`
[VERIFIED]:

| dataset | SapBERT Ph2 R@5 | all-MPNet Ph2 R@5 |
|---|---|---|
| test | 0.8833 | 0.8901 |
| cctg | 0.6859 | 0.6859 |
| oid_alt | 0.4329 | 0.4235 |
| cdash | 0.6711 | 0.6274 |
| gdc_combined | 0.8889 | 0.8333 |
| cimac_v2 | 0.5038 | 0.5649 |

> **Critical caveat.** Those rows use the **superseded split denominators**
> (test n=4,258 · cctg 1,159 · oid_alt 1,908 · cdash 526) and the earlier
> pre-reachability-corrected evaluation sets. They are **not** comparable to Table S4,
> which uses 3,959 / 1,097 / 1,766 / 324. Do not quote them side by side.

**[OPEN] What still needs a decision:** whether the answer to Warzel is "SapBERT was
screened in an earlier generation and not carried forward into the final three-model
design" (supported by the artifacts above) or "excluded by design at the outset"
(not supported — the constants and pooling ablation show it was in scope early). The
repository does **not** record a written rationale for the down-selection to three
models. Whoever answers should either find that rationale outside the repository or
state the decision plainly in the manuscript. Also note the second half of comment [29]
— who "we" refers to for BioSimCSE — is an authorship/attribution question, not a
repository question.

### 5.2 Precise wording for official NCI CDE Match output

Comment [30], Warzel: *"we have lots of saved output from CDE Match, do you mean 'for
this experiment we only used the GDC and CIMAC output'?"*

**[VERIFIED] Warzel is correct, and the current manuscript sentence is not accurate.**

Manuscript §2.3 currently reads: *"The official NCI CDE Match service is the production
system described in the Introduction; **its saved output was available only for GDC and
CIMAC**."*

Saved official-service output exists for far more than GDC and CIMAC, at **two** snapshot
dates. Directory `artifacts_v3_cdisc/cde_match_official/` in the research checkout:

```
2026-06-16/raw/  and  2026-06-18/raw/     (.xls, one per split)
  test_<date>.xls                  external_holdout_org_<date>.xls      (CCTG + OID ALT)
  val_<date>.xls                   external_holdout_refslice_<date>.xls (CDASH)
  external_holdout_standard_<date>.xls
  external_holdout_gdc_alt_<date>.xls     external_holdout_gdc_questiontext_<date>.xls
  cimac_version2_<date>.xls        theradex_<date>.xls
2026-06-18/normalized/  cdematch_2026-06-18_wide.csv, _cimac_v2_wide.csv, _theradex_wide.csv
2026-06-18/comparison_vs_2026-06-16/
```

The **reason** for reporting only GDC and CIMAC is already stated correctly elsewhere in
the manuscript (§3.8): on the four caDSR-derived sets the service directly queries the
alternate-name and question-text fields those sets were built from, so evaluation would
be circular; and the live service cannot be subjected to the query-level exact-match
masking used in the controlled experiments.

**[INTERPRETATION] Supported replacement wording** — availability is not the constraint,
validity is:

> The official NCI CDE Match service is the production system described in the
> Introduction. Saved output from the service is available for additional evaluation
> sets, but we report it **only for GDC and CIMAC**: on the four caDSR-derived sets the
> service queries the same alternate-name and question-text fields from which those sets
> were constructed, so the comparison would be circular, and the live service cannot be
> subjected to the query-level exact-match allowance applied to the other lexical methods.

Note the manifest already describes this gate as `official_nci_cde_match_frozen` with
role "official NCI CDE Match service output (**GDC + CIMAC only**)"
(`manifests/source_migration.yaml`) — i.e. only the GDC/CIMAC slice was ever brought into
the study's artifact scope, which is consistent with the replacement wording.

### 5.3 Frozen June 2026 catalog vs any current/live caDSR count

**[VERIFIED]** The frozen study catalog is **62,976 records / 62,858 unique public
identifiers**, derived from the **2026-06-18** nightly caDSR XML export
(79,827 raw records), eligibility mode `production_cde_match`. Digest in §4B.

**[VERIFIED]** The caDSR exports are **nightly snapshots, not versioned releases**, so
byte-exact re-acquisition after the fact is not possible; this is stated in `README.md`,
`docs/reproducing.md` and the `cadsr_snapshot_unresolved` gate.

**[VERIFIED, correction needed]** `README.md` states the snapshot digests "are pinned in
`configs/paper/`". They are not. What `configs/paper/` pins is the **derived** production
catalog parquet (`biencoder_protocol_v2.yaml:31`). The January raw export carries a
directory-tree digest in `manifests/source_migration.yaml` (`4acdcb26…`); the **June raw
export carries no digest anywhere**.

**[OPEN]** Warzel's comment [26] ("on Production the number of nonreleased CDEs is around
62K") suggests the live production count differs from 62,976 today. Nothing in the
repository can establish a *current* caDSR count — the repository only knows the frozen
snapshot. **[INTERPRETATION] Supported framing:** describe 62,976 as "a frozen 2026-06-18
production-oriented snapshot of the caDSR catalog", never as "the caDSR production
catalog", and note that the live registry changes nightly so a reader's count will
differ. Someone with caDSR access should supply the current figure if the manuscript
wants to cite one; do not infer it here.

**[OPEN, minor]** A **third** caDSR snapshot appears in the CIMAC construction path:
`src/demap_repro/data/cimac/appendix_a_v2.py:34` resolves CDE versions against
`data/processed/cadsr_xml_2026-06-16/cde_master_enriched_eval.parquet`. The manuscript
names only 2026-01-12 and 2026-06-18. Decide whether this needs a sentence in §S1.

---

## 6. Next likely analysis — Khanna's candidate-pool coverage question

Comment [41], Khanna: *"the Final reranker and the FT-MedCPT are the only two that are
reranking from the Fuzzy+MPNet Union (~30). I am curious the coverage of each of these
components (FT-MPNet @ 20, and Fuzzy @ 10) and the coverage of the union across all of
these datasets."*

### The results already exist — do not recompute

**[VERIFIED]** All three coverage arms are in a **committed fixture**:
`tests/fixtures/ce_bakeoff/comparison/crossenc_eval_by_split_SN_DEC_DEF_PQT_PV_ncbi_MedCPT-Cross-Encoder.csv`
(source: `artifacts/final_reranker/crossencoder_fulltrain_v2_eligible/comparison/<same name>`).
Column `n_queries_with_gold_in_candidates`, rows keyed by `method`:
`crossencoder` = the pooled set (union), `biencoder` = FT-MPNet top-20 alone,
`cdematch` = **CDE Match-Fuzzy** top-10 alone.

| dataset | n | (a) FT-MPNet @20 | (b) Fuzzy @10 | (c) union |
|---|---|---|---|---|
| Test | 3,959 | 3,791 (0.9576) | 3,149 (0.7954) | 3,903 (0.9859) |
| CCTG | 1,097 | 923 (0.8414) | 885 (0.8067) | 1,052 (0.9590) |
| OID ALT | 1,766 | 981 (0.5555) | 1,253 (0.7095) | 1,528 (0.8652) |
| CDASH | 324 | 299 (0.9228) | 279 (0.8611) | 314 (0.9691) |
| GDC | 72 | 66 (0.9167) | 70 (0.9722) | 71 (0.9861) |
| CIMAC | 131 | 88 (0.6718) | 106 (0.8092) | 114 (0.8702) |
| *(Validation Dev)* | 3,934 | 3,798 (0.9654) | 3,180 (0.8083) | 3,890 (0.9888) |

Rates are mine (`covered / n_queries`); the counts are as recorded. The `method` labels
are confirmed by their Recall@5 values, which match Table 4 exactly — `biencoder`
Recall@5 = 0.9116 / 0.7046 / 0.4281 / 0.8395 / 0.8333 / 0.5649 (Table 4 FT-MPNet), and
`cdematch` Recall@5 = 0.7734 / 0.7803 / 0.7022 / 0.8302 / 0.9722 / 0.7863 (Table 4
CDE Match-Fuzzy). [VERIFIED against `tests/fixtures/final_table4.csv`.]

**[INTERPRETATION]** These numbers answer Khanna's question directly and support the
complementarity argument he and Chen both ask to see strengthened (comments [39], [43]):
on OID ALT the lexical arm alone covers more gold (1,253) than the bi-encoder alone
(981), while on Test the ordering reverses (3,149 vs 3,791) — and the union beats both
on every dataset.

### A second, independent source for arms (a) and (c)

**[VERIFIED]** `tests/fixtures/k_selection_grid.csv` (source
`artifacts/final_reranker/k_selection/k_selection_grid.csv`) carries `branch=none`
(bi-encoder alone) and `branch=kwfuzzy` (union) at `se_k ∈ {10,20,30,50}`, `kw_k=10`,
for all eight splits. At the deployed `se_k=20, kw_k=10` it gives union
3,903 / 1,052 / 1,527 / 314 / 72 / 113.

> **[OPEN] Reconcile before publishing any of these.** The two sources disagree on three
> cells: OID ALT union 1,528 vs 1,527; GDC union 71 vs 72; CIMAC union 114 vs 113. Most
> likely an allowance-routing or candidate-table-version difference (GDC and CIMAC run at
> allowance 1.0; the grid may predate the `_v2_eligible` correction). **Do not pick one
> silently** — determine which grid the deployed pool corresponds to.

### If arm (b) must be recomputed reproducibly

`src/demap_repro/pool/select_k.py::compute_k_grid` already supports it — pass
`se_depths=(0,)` and `kw_depths=(10,)` and the bi-encoder side contributes nothing:

```python
from demap_repro.pool.select_k import compute_k_grid
compute_k_grid(gold, biencoder_ranks, keyword_ranks,
               dataset="test", se_depths=(0,), kw_depths=(10,))
```

Required inputs, all under `$DEMAP_DATA_ROOT`:

| input | path |
|---|---|
| FT-MPNet deep rankings | `artifacts/final_reranker/biencoder_deep/eval_canonical/biencoder_deep_rankings_top1000.parquet` |
| CDE Match-Fuzzy candidates | `artifacts/final_reranker/keyword_fuzzy_v2_eligible/cde_match_clone_candidates_<split>__allow{0.70,1.0}.parquet` |
| per-split settings | `artifacts/final_reranker/keyword_fuzzy_v2_eligible/cde_match_clone_summary_<split>__allow*.json` |
| gold / denominators | `data/processed/eval_canonical/<split>.parquet` |
| existing grid + notes | `artifacts/final_reranker/k_selection/{k_selection_grid.csv,K_SELECTION_SUMMARY.md}` |

Third route, if a from-the-deployed-pool number is wanted: the frozen fixed-K feature
table `artifacts/final_reranker/hgbc_features_v2_eligible/feature_table_fixedk30_crossenc_v2.parquet`
(423,395 rows × 137 cols) carries `in_biencoder_topk`, `in_keyword_topk`, `in_cdematch_topk`
and `is_label` per row, so all three arms are a groupby on `(split, query_id)`. Per-split
row/query/positive counts are already summarized in `tests/fixtures/fixed_k_feature_table.json`.

**Do not run any of this as part of the handoff.** [VERIFIED] The answer already exists;
the work is reconciliation and presentation, not computation.

---

## 7. Optional — Khanna's HGBC feature-group ablation

Comment [40], Khanna: *"It might be out of scope but I am curious which of these feature
groups within the final model result in the improvement. i.e. what would dropping the
keyword rules, or the PV overlap alone from the final reranker do to the OID ALT
performance?"*

**Status: OPTIONAL. Do not start automatically. §6 is the higher-priority analysis.**

**[VERIFIED] The 117 features and their group structure.** Manuscript §S5.5 names seven
groups; the committed feature list (`tests/fixtures/hgbc_feature_set.json`) partitions
cleanly by prefix and sums to exactly 117:

| manuscript group | prefix | n | examples |
|---|---|---|---|
| retrieval-source indicators | `in_*` (+ `n_*`, `best_*`, `rrf_*`) | 3 (+5) | `in_biencoder_topk`, `in_cdematch_topk`, `in_keyword_topk`, `n_rule_hits`, `n_distinct_rules`, `best_rule_code`, `best_field_code`, `rrf_score` |
| FT-MPNet scores and ranks | `bienc_*`, `biencoder_*` | 12 | `biencoder_rank`, `bienc_score_z`, `bienc_margin_to_top1` |
| keyword exact/synonym-tier scores **and** keyword-rule evidence | `kw_*` | 35 | `kw_any_exact`, `kw_charngram_LONG_NAME`, `kw_wordngram_*`, `kw_token_max`, `kw_pv_score` |
| permissible-value overlap | `pv_*` | 31 | `pv_n_sde`, `pv_both_have_pvs`, `pv_count_ratio_min_over_max` |
| query and CDE text characteristics | `text_*` | 18 | `text_query_is_code_like`, `text_query_underscore_count` |
| FT-MedCPT scores and ranks | `crossenc_*` | 6 | `crossenc_score`, `crossenc_rank`, `crossenc_missing` |
| *(CDE Match derived — see §3)* | `cdematch_*` | 7 | `cdematch_score_z_within_query`, `cdematch_margin_1_2` |

Note the manuscript's seven groups map onto **eight** prefix families: `kw_*` covers two
of the seven (exact/synonym tiers and rule evidence), and `cdematch_*` is not separately
named in §S5.5. **[OPEN]** Agree the exact group→feature mapping with the manuscript text
before running any ablation, or the result will not be describable.

Five features are excluded from the 122-column superset:
`keyword_rank`, `cdematch_rank`, `log1p_cdematch_rank`, `text_family`, `text_query_source`.

**Artifacts and code an ablation would use:**

| purpose | path |
|---|---|
| feature matrix | `artifacts/final_reranker/hgbc_features_v2_eligible/feature_table_fixedk30_crossenc_v2.parquet` (sha256 `62eeae7a…`) |
| trainer | `src/demap_repro/reranker/train.py` (`demap train-hgbc`) |
| feature-set config format | `configs/paper/features_117_noprov.json` (a **drop list**, not a 117-name list) |
| shipped model + schema | `artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov/{hgbc_model.joblib,feature_set.json,categorical_vocab.json}` |
| frozen protocol | train on `val_train`, select on `val_dev` Recall@5, 16-config grid (`max_iter {200,500}` × `max_depth {3,5}` × `lr {0.05,0.1}` × `min_samples_leaf {10,30}`), `random_state=42` |
| per-split evaluation | `src/demap_repro/evaluation/by_dataset.py` (`demap eval-by-dataset`) |
| reference metrics | `tests/fixtures/hgbc_eval_by_split.csv`, `manifests/expected_results.json` → `F_hgbc_final_model` |

**[INTERPRETATION]** The natural design is one drop-group-and-retrain run per group under
the identical frozen protocol, reported as ΔRecall@5 per evaluation dataset. It is CPU-only
and cheap relative to anything neural. But it is a **new experiment not in the manuscript**,
so it needs an explicit scope decision first — `manifests/expected_results.json:10` currently
declares "no non-paper feature ablation" as a scope boundary, and adding one would widen the
frozen paper repository past the manuscript.

---

## 8. Artifact and code index for §6 and §7

Paths relative to `$DEMAP_DATA_ROOT = /data/nextgen2/james/tasks/cde_project/demap`
unless prefixed with the DEMap repo.

**Candidate generation and pools**
* `artifacts/final_reranker/biencoder_deep/eval_canonical/{biencoder_deep_rankings_top1000.parquet,recall_summary.csv,summary.md}`
* `artifacts/final_reranker/keyword_fuzzy_v2_eligible/` — 8 candidate parquets + 8 summary JSONs + `keyword_index/`
* `artifacts/final_reranker/k_selection/{k_selection_grid.csv,K_SELECTION_SUMMARY.md}`
* `artifacts/final_reranker/hgbc_features_v2_eligible/{feature_table_fixedk30_crossenc_v2.parquet,fixedk30_base/,scores/,ce_alone_fixedk30_eval/}`
* DEMap: `src/demap_repro/pool/{select_k.py,candidate_union.py}`, `src/demap_repro/reranker/{fixed_k_features.py,split_routing.py,merge_crossenc_features.py}`

**Saved rankings and per-method metrics**
* `artifacts/final_reranker/crossencoder_fulltrain_v2_eligible/comparison/` — the per-backbone by-split CSVs (the §6 source), plus `BAKEOFF_ce_selection_val_dev.csv` and `CE_WINNER.json`
* `artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov/hgbc_eval_by_split.csv`
* `artifacts/bm25_canonical_v1/{bm25_val_dev_selection.csv,bm25_canonical_metrics.csv}`
* `artifacts/final_reranker/cde_match_clone/clone_canonical_metrics_combined.csv`
* `.scratch/demap/paper_v13_scientific_audit/kwfuzzy_corrected_metrics.csv`

**Committed fixtures (no data root needed)**
* `tests/fixtures/ce_bakeoff/comparison/*.csv` — the §6 coverage table
* `tests/fixtures/k_selection_grid.csv`, `tests/fixtures/final_table4.csv`
* `tests/fixtures/{hgbc_feature_set.json,fixed_k_feature_table.json,hgbc_selected_config.json,hgbc_grid_results.csv,hgbc_eval_by_split.csv}`
* `manifests/expected_results.json` — 17 result families with `manuscript_stated` blocks

**SapBERT (§5.1)**
* `artifacts/publication_artifact_package_20260713/tables/bi_encoder/biencoder_{offtheshelf,phase1,phase2}_best_by_model.csv`
* `artifacts/{phase1,phase2}_finetuning/sapbert-pubmedbert-fulltext/`
* `artifacts/paper_step9_sapbert_pooling/`, `configs/experiments/paper_sapbert_pooling_ablation.yaml`
* DEMap: `src/demap_repro/biencoder/constants.py`, `src/demap_repro/reporting/{biencoder_tables.py,representation_tables.py,publication_figures.py}`

**Official CDE Match (§5.2)**
* `artifacts_v3_cdisc/cde_match_official/{2026-06-16,2026-06-18}/{raw,normalized}/`

---

## 9. Editorial work handled separately with ChatGPT — no Claude experimentation

These comments need writing, not repository work. Listed so the next session does **not**
start them.

* Terminology: define SDE, expand caDSR at first use, keep CDE/CDEs consistent
  (comments [4], [5], [7], [8], [24]).
* Expand acronyms at first use: CCTG, OID ALT, GDC, CIMAC, CDASH ([20]–[23]).
* Simplify the Figure 1 narrative — describe the workflow conceptually before naming
  models ([14]); Figure 1 has minor visual overlaps, suggest SVG export or drawio ([18], [19]).
* Define cross-encoder ([12], [36] — name the specific cross-encoder checkpoints), HGBC at
  first mention ([15]), and "anchor" ([33]).
* Explain "query-side" vs "CDE-side" PV summaries in plain terms ([28]).
* Emphasize that all three fine-tuned bi-encoders perform similarly, as a strength ([35]).
* Add 1–2 sentences of interpretation immediately after Table 4 ([39]).
* Strengthen the semantic-vs-lexical complementarity discussion and add a take-home
  sentence ([43], [44]).
* Highlight key dataset differences in the results text, not only Table 1 ([31]).
* Rename "Validation Training" to something clearer, e.g. "Reranker Training" ([32]) —
  **note:** this is a manuscript-only rename; the split is `val_train` throughout the code
  and artifacts and must not be renamed there.
* Replace em dashes with regular punctuation ([11], [38]).
* Standardize reported precision to three digits throughout ([34]).

**[INTERPRETATION]** Two of these have a factual dependency on §6: [39] and [43] both ask
for concrete numbers, and the coverage table in §6 is the natural evidence.

---

## 10. Known inconsistencies visible in repository artifacts — do not edit

### 10.1 PV-summary denominator — CONFIRMED

**[VERIFIED]** Within `cde_paper_20260806_comments.docx` itself:

| location | value |
|---|---|
| Table S3, "overall" row | **38,964** |
| §S3.5 prose, "over the … rows in which both sides carry a permissible-value summary" | **39,391** |

Khanna's comment **[49]** — *"Should this match the value in the table?"* — is anchored to
exactly that prose sentence. [VERIFIED by mapping `w:commentRangeStart` ids to paragraphs.]

**38,964 is correct.** It is the canonical computation over the paper-era 69,102-row
`pairs.parquet`; 39,391 came from a superseded 69,844-row build overwritten on 2026-05-13.
The recomputation was author-approved 2026-08-08. The repository asserts 38,964
(`tests/tier3_regression/test_table_s3_pv_overlap.py:38`) and retains 39,391 only as
`tests/fixtures/table_s3_pv_overlap_SUPERSEDED.csv`, explicitly marked never a parity
target. The pinned `cde_paper_v21.docx` carries 38,964 in **both** places.

**Classification: stale manuscript text in the August-6 file.** Confined to one
supplementary prose sentence; Table 4, the abstract and every final-system result are
unaffected. Fix it in the manuscript during the revision; change nothing in the repository.

### 10.2 Cross-encoder protocol config carries superseded values

**[VERIFIED]** `configs/paper/crossencoder_protocol_v1.yaml` records
`winner.selection_val_dev_recall_at_5: 0.9128` and points `checkpoint` / `scores` /
`comparison` at `artifacts/final_reranker/crossencoder_fulltrain/`. The corrected values
are **0.9123** (manuscript 0.912) from `crossencoder_fulltrain_v2_eligible/`, which is what
`manifests/expected_results.json`, `docs/reproducing.md` and `src/demap_repro/reporting/table4.py`
all use. 0.9128 is named as superseded in `docs/release_readiness.md`.

**Classification: stale configuration.** The manuscript is correct. No test guards this
file's `winner:` block. Repository fix, not a manuscript fix — out of scope for the revision.

### 10.3 Coverage-count disagreement between two artifacts

See §6: OID ALT union 1,528 vs 1,527, GDC 71 vs 72, CIMAC 114 vs 113. **[OPEN]** Unresolved;
must be reconciled before either source is quoted.

### 10.4 Table 4 FT-MPNet vs Table S4 all-MPNet

**[VERIFIED, not an error.]** Table 4's FT-MPNet row is the single retained downstream
seed (Test 0.912); Table S4's all-MPNet row is a two-seed mean (Test 0.9089 ± 0.0038).
Legitimately different quantities, but nothing in the manuscript or repository says so.
**[INTERPRETATION]** Worth one clarifying clause, and it interacts with comment [34] on
precision consistency.

### 10.5 README digest claim

**[VERIFIED]** `README.md` says the caDSR snapshot digests are pinned in `configs/paper/`;
only the derived June catalog parquet is. Repository documentation fix, not manuscript.

---

## 11. Recommended next Claude task

**Safest first action: reconcile the candidate-pool coverage numbers in §6, read-only.**

Concretely, in one session:

1. Confirm the `method` labels in
   `tests/fixtures/ce_bakeoff/comparison/crossenc_eval_by_split_SN_DEC_DEF_PQT_PV_ncbi_MedCPT-Cross-Encoder.csv`
   against the producer in
   `artifacts/final_reranker/crossencoder_fulltrain_v2_eligible/comparison/` and its
   generating code, so `cdematch` is definitively established as CDE Match-Fuzzy top-10
   and not the Python approximation. (Recall@5 agreement with Table 4 already indicates
   this, but the column name invites exactly the confusion the manuscript warns against.)
2. Resolve the three-cell disagreement with `tests/fixtures/k_selection_grid.csv`
   (§10.3) by identifying which candidate tables and allowance routing each was computed
   from.
3. Produce a single small coverage table — six datasets × three arms, counts and rates —
   suitable for a reply to Khanna and, if the authors want it, for a supplementary table.

Why this first: it is read-only, needs no retraining and no GPU, the underlying numbers
already exist, it directly answers the highest-priority open coauthor comment, and it
supplies the evidence that comments [39] and [43] ask for. It also surfaces the §10.3
inconsistency before anything is written into the manuscript.

**Do not, in that session:** start the §7 ablation; update the CDE Match release gates
from §3 (a separate, deliberate change); edit the manuscript; or commit anything without
being asked.
