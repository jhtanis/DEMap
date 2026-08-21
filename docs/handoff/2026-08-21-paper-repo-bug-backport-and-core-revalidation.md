# Handoff: metamodel bug backport and core-result revalidation

**Date** 2026-08-21 · **Branch** `main` · **Starting HEAD** `55e8765`
("Add frozen paper repository handoff")

## The question this session answered

> After applying the later shared bug fixes, what is the minimum computation
> needed to establish that the central conclusions of the original
> non-metamodel paper still hold?

**About five seconds of CPU, and no GPU at all.** The answer is that small
because every applicable bug turned out to be plumbing — paths, packaging,
missing inputs, a moved dependency API — and because the frozen HGBC model and
the frozen 117-feature table are both retained, so the paper's headline result
can be recomputed from the model rather than read from a CSV.

That recomputation is committed as
`tests/tier3_regression/test_core_result_contract.py` and has been run: **every
Table 4 value reproduces to floating point**, and Table 4 regenerated end-to-end
from the per-method artifacts is bit-identical to the committed fixture across
all 54 cells.

**No paper number changes. No manuscript statement is suspect. No frozen
artifact was regenerated or overwritten.**

## Baseline and outcome

| | tests | passed | failed | skipped |
|---|---|---|---|---|
| `55e8765`, clean clone | 484 | 475 | **2** | 7 |
| after, clean clone | 537 | 512 | 0 | 25 |
| after, with `DEMAP_DATA_ROOT` + `DEMAP_ARTIFACT_ROOT` | 537 | **535** | 0 | 2 |

The two baseline failures were inventory entry 12 (undeclared `pptx` / `docx`),
now declared and correctly skipped. The 25 clean-clone skips are all explicit:
2 for the `documents` extra, 23 for the data and artifact trees that are not
distributed. With both roots pointed at the research repository, **only the 2
optional-extra skips remain and every parity test passes** — including the seven
that check the committed fixtures against the full-size originals.

## Bug inventory reviewed

`DEMap_metamodel_research/docs/metamodel_bug_inventory.md` — **14 entries**
(13 software, S1 method, S1b). Full applicability matrix, per-entry reasoning and
evidence: **`docs/bugfix/2026-08-21-metamodel-bug-backport-audit.md`**.

* 10 applicable, fixed: **1, 2, 3, 4, 8, 10, 11, 12, 13** and new finding **N1**
* 1 partially applicable, fixed: **6** (the lock, without metamodel machinery)
* 1 applicable, deliberately not backported: **5** (PQT filler)
* 3 not applicable: **7, 9, S1/S1b** (9 needs no change)

**Entry 10 was the inventory's one OPEN scientific question** and is adjudicated:
`by_dataset.py`'s stale default did not produce any published number, and would
have silently dropped four of six evaluation datasets. Evidence in §2 of the
audit.

**Entry 5 was deliberately not backported.** The inventory measured it: of 1,783
CDEs carrying filler PQT, **0** had a real PQT that was discarded. It is a
representation-quality opportunity, not a correctness defect, and backporting it
would invalidate every frozen bi-encoder artifact to fix nothing.

**N1 was found here**, by tracing entries 4 and 11 into this repository: three
modules still imported unmigrated research scripts. `bm25/run.py` resolved
`repo_root()/"scripts"`, which does not exist here, so BM25 — a Table 4 method —
could not be evaluated from a clean clone. Equivalence of each replacement was
verified by AST comparison before repointing.

Nothing metamodel-specific was imported: no SapBERT arm, no top-15 + top-5 pool,
no 127-feature contract, no metamodel HGBC, no deployment metadata policy.

## Files changed

**Source (14)** — `data/queries.py` (digest pin) · `evaluation/canonical_datasets.py`
(default + lock) · `evaluation/by_dataset.py` (default, guard, loud skip) ·
`biencoder/engine/st_loader.py` (pooling) · `biencoder/deep_retrieval.py` ·
`reporting/biencoder_tables.py` · `data/cimac/{appendix_a_v2,corrected_pv_split}.py` ·
`lexical/{non_exact_subset.py,bm25/run.py}` · `reporting/figures/{make_figure5_keyword,
make_figureS3_repxloss,make_figureS6_S7_allowance}.py` ·
`sensitivity/allowance/score_fixed_070.py`, plus the `data_root()` conversion
across the 17 files that hardcoded the research tree.

**Config / packaging (4)** — `pyproject.toml` (`documents` extra) ·
`configs/allowlists/{alt_allowlist,alt_query_recipes,refdoc_allowlist}_current.csv`
(new, byte-identical to the research repository).

**Tests (5)** — `tests/conftest.py` (`data_root` fixture) ·
`tier1_invariants/test_import_safety.py` (explicit extras map + 2 guards) ·
`tier1_invariants/test_backported_plumbing.py` (new, 12) ·
`tier2_behavior/test_backported_behavior.py` (new, 21) ·
`tier3_regression/test_core_result_contract.py` (new, 18).

**Docs (3)** — the audit, this handoff, and the stale test count in `README.md`
and `docs/reproducing.md`.

## Tests added

**53 new tests.** Every one that asserts a bug fix was run against a `55e8765`
worktree first: **29 of 33** backport tests fail there. The four that pass are
the ones documenting behaviour that was already correct (the `base` pooling
early return, the no-Pooling-module error, that `by_dataset` imports, and that
the denominator is the query set). The two packaging guards were separately
verified to fail against a `pyproject.toml` with the extra removed and against
one where `all` omits it.

No frozen fixture was edited to make anything green.

## Affected historical artifacts

**None.** `manifests/expected_results.json`, `tests/fixtures/final_table4.csv`,
`hgbc_eval_by_split.csv` and every manuscript-aligned fixture are untouched, and
all pre-existing tier-3 regressions pass unchanged.

### Manuscript provenance — the pin is correct, the supplied file is older

`manifests/paper_scope.yaml:11` and `manifests/expected_results.json:5` pin the
authoritative manuscript as `manuscript/cde_paper_v21.docx`
(`sha256 2df7895c…`, 6,159,078 bytes). **That pin is valid.** The file exists at
that path in the historical research checkout — which `paper_scope.yaml:16`
declares `immutable_read_only` — and its digest and size match exactly. It was
never meant to resolve inside this repository.

The manuscript supplied for this session, `cde_paper_20260806.docx`
(`sha256 6295986f…`), is **an earlier manuscript state, not an equivalent copy**.
The two agree on 1,176 of 1,177 paragraphs with byte-identical figures, and
differ in exactly one checked scientific value — the Section S3.5 PV-overlap
denominator:

| manuscript | S3.5 denominator |
|---|---|
| `cde_paper_v21.docx` — pinned, correct | **38,964** |
| `cde_paper_20260806.docx` — supplied, August 6 | **39,391** |

**39,391 is superseded** (generated 2026-04-23 from a 69,844-row `pairs.parquet`
that was overwritten on 2026-05-13 by the paper-era 69,102-row build). The
recomputation to 38,964 was author-approved on 2026-08-08, and **38,964 is this
repository's regression target**: `test_table_s3_pv_overlap.py:38` asserts it and
`tests/fixtures/table_s3_pv_overlap_SUPERSEDED.csv` marks the 39,391 values as
provenance that must not be a parity target.

**Do not repoint either pin at the August-6 manuscript** — that would make the
authoritative manuscript carry a value this repository's own test asserts
against. **This discrepancy does not affect Table 4, the abstract, or any
final-system result**; it is confined to one supplementary prose denominator.
Full detail in §5 of the backport audit.

## The revalidation plan, and what has already been run

### Level A — structural, offline (**run, all green**)

`test_core_result_contract.py::test_A*`, needs `DEMAP_DATA_ROOT` only. **~1 s.**

Query counts (3,959 / 1,097 / 1,766 / 324 / 72 / 131, exact) · catalog identity
(62,976 records, 62,858 public identifiers) · the eligibility filter **rebuilt
from the raw 79,827-row export** (16,846 retired · 21 TEST/Training · 16 both ·
62,976 kept, exact) · zero unreachable gold CDEs · allowance routing 0.70 × 4 /
1.00 × 2 · mask nested, deterministic, realized 0.6964 at 0.70 · K = 30 with
keyword depth 10 on the `kwfuzzy` branch · the 117-feature contract · all six
registry digests and the catalog digest.

### Level B — core result recomputed from frozen artifacts (**run, all green**)

`test_core_result_contract.py::test_B*`, needs `DEMAP_ARTIFACT_ROOT`.
**~4 s, CPU only.** Reloads the frozen HGBC, rebuilds the matrix from the frozen
423,395-row feature table with `prepare_features`, and recomputes with the
trainer's own `_deployment_metrics`.

| dataset | n | R@1 | R@5 | R@10 | MRR@100 |
|---|---|---|---|---|---|
| Test | 3,959 | 0.8957 | **0.9715** | 0.9803 | 0.9289 |
| CCTG | 1,097 | 0.7976 | **0.9088** | 0.9344 | 0.8471 |
| OID ALT | 1,766 | 0.7644 | **0.8324** | 0.8533 | 0.7963 |
| CDASH | 324 | 0.8086 | **0.9198** | 0.9444 | 0.8580 |
| GDC | 72 | 0.9306 | **0.9722** | 0.9861 | 0.9537 |
| CIMAC | 131 | 0.7328 | **0.8015** | 0.8397 | 0.7669 |

Maximum |Δ| against the frozen evaluation table: **1.1 × 10⁻¹⁶**.

Test 0.971 ✓ · external 0.802–0.972 ✓ · GDC 70/72 ✓ (asserted as an integer
count) · Table 4 regenerated end-to-end, bit-identical over 54 cells ✓ · best on
five of six recomputed from the regenerated values, with GDC going to the Python
approximation exactly as reported ✓.

### Level C — expensive work required

**None.** No bug touches training data, representations, candidate generation,
feature computation or evaluation denominators-as-used. Nothing needs a GPU.

Applying the decision rule from the task: the fixes change evaluation *plumbing*
only, so no retraining, no Fuzzy rerun, no pool regeneration, no feature
regeneration. The frozen trained weights answer the paper's inference question,
and Level B demonstrates that they still answer it identically.

### Secondary results

None require retesting. No applicable bug touches Table S5 leakage filtering,
Table S6 / Figures S6–S7 allowance sensitivity, Table S3 PV diagnostics,
K-selection, cross-encoder selection, BM25 representation selection or the
fine-tuning grids. Their existing regressions all pass, and the artifact-gated
ones pass against the full-size originals.

## Acceptance criteria

**Deterministic evaluation from frozen artifacts** — exact integers for query
counts, catalog counts, eligibility arithmetic, gold reachability, feature count
and the GDC hit count; `abs=5e-4` (the level-1 tolerance in
`expected_results.json`) for recomputed recall and MRR, though observed agreement
is ~10⁻¹⁶ and anything looser than ~10⁻¹² should be investigated rather than
accepted; method ordering preserved exactly.

**Neural recomputation** — not required, so the repository's existing level-2
tolerance is untouched. No tolerance was loosened anywhere.

**Qualitative** — the paper's final-system conclusions are unchanged, and no
changed conclusion was identified.

## Estimated cost of the minimal plan

| | wall clock | hardware |
|---|---|---|
| Offline suite (clean clone) | ~7 s | any CPU |
| Level A + Level B | ~4 s | 1 CPU core, ~2 GB RAM |
| Full suite with both roots | ~30 s | 1 CPU core |
| Level C | — | not required |

## Exact next commands

```bash
cd /data/nextgen2/james/tasks/cde_project/DEMap
export DEMAP_DATA_ROOT=/data/nextgen2/james/tasks/cde_project/demap
export DEMAP_ARTIFACT_ROOT=$DEMAP_DATA_ROOT

# offline suite as a reader would run it
pytest -q                                                    # 512 passed, 25 skipped

# Level A + Level B, the core-result revalidation
pytest -q tests/tier3_regression/test_core_result_contract.py -v   # 18 passed

# everything, including the full-size parity checks
pytest -q                                                    # 535 passed, 2 skipped

# regenerate Table 4 from the per-method artifacts and diff against the fixture
python -m demap_repro.reporting.table4 \
    --root "$DEMAP_DATA_ROOT" --out-dir .scratch/table4_regen
```

To clear the two remaining skips: `pip install -e '.[documents]'`.

## Open items, deliberately not done here

1. **The manuscript file discrepancy.** The pins in `paper_scope.yaml` and
   `expected_results.json` are correct and need no change. What is open is
   reconciling the supplied August-6 `cde_paper_20260806.docx`, which carries the
   superseded S3.5 denominator 39,391 where the pinned `cde_paper_v21.docx`
   carries the author-approved 38,964. That belongs to the manuscript task, and
   nothing in this repository changes for it.
2. **Entry 5 (PQT filler)** and the 1,352 real PQT rows discarded by
   `drop_duplicates(keep='first')` are representation-improvement candidates for
   a future DEMap. Both would invalidate frozen bi-encoder artifacts and neither
   is a correctness defect.
3. **The manuscript was not edited**, as instructed. Nothing in it needs editing
   on the evidence of this session.
