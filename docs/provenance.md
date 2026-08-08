# Provenance and migration policy

DEMap is a **selective public-reproduction repository** built from a
separate, **immutable** research repository. This document states the rules that govern every
file that enters it.

---

## 1. Two repositories, one direction of flow

| | path | role |
|---|---|---|
| **SRC** | `/data/nextgen2/james/tasks/cde_project/demap` | The research and provenance archive. Holds the complete experimental history, including abandoned approaches, debugging variants, superseded runs and the manuscript revision chain. |
| **DST** | `/data/nextgen2/james/tasks/cde_project/demap_repro` | This repository. Holds only what is needed to reproduce the results reported in the paper. |

**SRC is read-only for the entire public-repository effort.** Nothing under SRC is modified,
moved, renamed, deleted, reformatted, staged, committed, tagged or otherwise altered — not
tracked files, not untracked files, not `.scratch/`, not documentation, not configs, not tests,
not manifests, not git state. SRC may still be needed for manuscript revisions, and it is the
provenance of record for every number in the paper.

Corrections discovered during migration — stale defaults, superseded documentation, missing
split routing, divergent duplicate implementations — are applied **only to the copies inside
DST**, and are recorded as `divergence_note` entries in `manifests/source_migration.yaml`.
They are never propagated back into SRC.

Information flows **SRC → DST only**.

---

## 2. Scope is defined by the paper, not by the source repository

The authoritative manuscript is recorded in `manifests/paper_scope.yaml` together with its
SHA256. Every experiment, table, figure and numerical claim in that manuscript has a
`paper_scope` entry.

A source file may be migrated only if:

- **(A)** it directly produces a `paper_scope` item, or
- **(B)** it is a necessary dependency of code that produces one.

Nothing else is migrated. "This was explored during development" is not a reason to include
something. Historical experiments, abandoned approaches, debugging variants, superseded runs
and analyses that did not reach the paper stay in SRC and are not exposed as part of the public
scientific workflow.

`manifests/paper_scope.yaml` is the **first gate**. `manifests/source_migration.yaml`,
`manifests/expected_results.json` and `configs/paper/results_provenance_v1.yaml` are all
validated against it: every entry in each of them must map to at least one valid
`paper_scope` id.

---

## 3. The actual worktree file may be authoritative — not HEAD

The implementation that produced a published result is not always the committed one. In this
project it may be:

- **tracked** and unchanged since the last commit;
- **tracked but modified** in the working tree (the worktree version is the one that ran);
- **untracked** and never committed at all;
- under **`.scratch/`**, written as exploratory work that nevertheless became the executed
  final-paper implementation.

Therefore `manifests/source_migration.yaml` records, for every candidate file:

- the SRC-relative path and the proposed DST path;
- `source_class`: `tracked` / `tracked_modified` / `untracked` / `scratch`;
- `worktree_sha256` — the hash of the file **as it exists on disk**, which is the hash that
  matters;
- `head_blob` — the committed blob hash where one exists, recorded so that a divergence between
  HEAD and the worktree is visible rather than assumed away;
- the `paper_scope` ids it supports;
- the authoritative parity artifact and its SHA256;
- a `divergence_note` where two implementations of the same operation disagree;
- the release gate, if the material is not yet cleared for publication;
- the `migration_state`.

**When tracked code and a `.scratch` implementation disagree, the implementation that actually
generated the authoritative manuscript artifact governs scientific parity.** The other is
recorded, and may be retained in DST under a `_reference_` name for reconciliation, but it is
not used to define correctness.

---

## 4. Copy-first, refactor-second

Every migrated component follows the same loop, without exception:

```
1. identify   the exact implementation that generated the final-paper result
2. copy       it verbatim into this repository (SRC untouched)
3. record     source path, source class, worktree SHA256, HEAD blob, scope ids,
              parity artifact + SHA256, divergence notes
4. parity     establish a regression test against the authoritative paper artifact
              BEFORE any cleanup
5. refactor   only inside this repository — rename, consolidate, restructure, fix
6. re-parity  re-run the regression; accept the refactor only if parity holds
              within the scientifically appropriate tolerance
```

A copied implementation is **not** "improved" before parity is established. Cleaning code whose
behavior has not yet been pinned down is how a published number silently changes.

`migration_state` moves monotonically through:
`not_copied` → `copied_verbatim` → `parity_established` → `refactored` → `parity_reverified`.

Tolerances are declared in `manifests/expected_results.json`:

- regeneration from frozen artifacts: exact to within rounding (0.0005);
- retraining on different hardware: 0.01 Recall@5 **and** method ordering preserved exactly;
- counts and feature cardinalities: exact integers.

---

## 4a. Not every historical implementation is a migration candidate

A file that once produced a published number is not automatically the file that should reproduce
it. Each `source_migration.yaml` entry therefore carries a `classification` and a
`copy_verbatim` flag:

- `migration_candidate` / `copy_verbatim: true` — copy verbatim, then follow the loop above.
- `historical_reference_stale` / `copy_verbatim: false` — **never copied as reproduction code.**
  The entry exists only as provenance: it records what the file once produced, why re-running it
  is unsafe, and — in `supersedes_current_sources` — the corrected artifacts that the newly
  written DST implementation must be built from and parity-tested against.

The first such entry is `.scratch/demap/paper_v10_revision/data2/_build_data2.py`. Its own
sibling `provenance.json` proves it reads pre-correction experiment roots; the CSVs it once wrote
were later overwritten by the corrected chain, but the builder was not updated. Re-running it
would silently restore superseded Figure 4 and Figure 5 values. The DST Figure 4/5 input builder
is written fresh against the corrected roots.

The general rule: **an implementation whose recorded inputs are superseded is evidence, not
code.**

## 4b. A frozen artifact proves its own contents, not the code that made it

Establishing that a published number is recoverable from a frozen artifact is not the same as
establishing that the current code would recreate that artifact. Those are separate claims and
this repository verifies them separately.

For each generation stage whose output was frozen before the paper, the audit asks two questions:

1. **Is the current implementation the one that produced the frozen artifact?** Answered from
   commit history, worktree cleanliness, artifact mtimes, and the build manifest — not from
   filenames or docstrings.
2. **Does the current implementation still reproduce it?** Answered by regenerating the fields
   from the same upstream inputs and comparing row by row, over the whole dataset rather than a
   sample.

The PV-summary audit (2026-08-07) is the worked example. Both questions came back clean —
100% row-level parity on 46,948 comparable rows for the query-side and CDE-side blocks — but the
audit also surfaced something neither question would have caught alone: the **input** to the
frozen Table S3 diagnostics was a `pairs.parquet` build that has since been overwritten. The code
is fine; the artifact's provenance is not.

Two lessons are encoded in this repository's process:

- **Record which build of an input an artifact consumed, not just its path.** A path that is
  rewritten in place destroys the link. Every migrated stage records the row count and SHA256 of
  the input it was verified against.
- **Distinguish "the code drifted" from "the input changed."** Run the aggregate twice — once on
  the frozen fields and once on regenerated fields — so the two causes cannot be confused. If
  those two agree and both differ from the published value, the code is exonerated and the input
  is the suspect.

## 5. Legally and data-gated material is not copied

Some material required to reproduce parts of the paper has unresolved redistribution status.
Until each gate is cleared in writing, that material is **recorded but not copied**:
`manifests/source_migration.yaml` holds its SRC path, SHA256, role, supported `paper_scope`
ids and gate; `copied_to_dst` is `false`.

Parity work may **read** such material directly from immutable SRC. It must never be copied
into DST, and must never enter this repository's git history — including its history of
deleted files. The open gates and what they affect are enumerated under `release_gates:` in
`manifests/source_migration.yaml`.

The architecture accommodates both outcomes for each gate: a gated component is either
**included later** if release is approved, or **replaced by precomputed candidate/result
artifacts** if it is not. Downstream stages are designed to run identically in either mode, so
they can be hardened while the gates remain open.

### 5a. A single "gated" flag hides more than it says

"Gated" conflates questions that have different answers and different consequences. Every
`paper_scope` item therefore carries a `release` block valued `yes` / `no` / `unknown` /
`not_applicable` (`partial` where a single item mixes recomputed and frozen columns):

| field | question |
|---|---|
| `code_redistributable` | may we publish the implementation |
| `data_redistributable` | may we publish the input data |
| `weights_redistributable` | may we publish weights we trained |
| `runnable_from_public_inputs` | can a public user run it from what we ship |
| `runnable_with_user_supplied_or_private_inputs` | can they run it holding their own copy |
| `reproducible_by_retraining` | can the result be re-derived by training |
| `requires_precomputed_input` | does it need a released intermediate artifact |
| `externally_frozen_result` | is it a saved third-party result, never runnable |

Each `release_gates:` entry in `source_migration.yaml` likewise names the single `dimension` it
constrains and states `blocks_reproducible_by_retraining` explicitly.

The distinction that matters most: **`weights_redistributable: unknown` does not imply
`reproducible_by_retraining: no`.** Our fine-tuned bi-encoder and cross-encoder checkpoints have
unresolved redistribution status, but their base checkpoints are public and the training protocol
is documented, so the results remain re-derivable by retraining. Conversely, a saved result of a
live third-party service is `externally_frozen_result: yes` and is never runnable under any
resolution — that is a property of the result, not a gate on our code. A validator check enforces
that no item pairs an unknown weights gate with `reproducible_by_retraining: no`.

---

## 6. Fresh git history

This repository will be initialized with a **new, empty git history**. The SRC repository's
history is never cloned, imported, grafted or filtered into it.

That is a requirement, not a preference: SRC's history contains material whose redistribution
status is unresolved, and rewriting published history cannot retract objects that already exist
on remotes. A fresh history is the only construction in which DST can be proven clean.

Commits are small and reviewable, made as migration phases pass parity. No AI-attribution
trailers are added to commits.

---

## 7. What is retained for every migrated implementation

For each file that reaches `copied_verbatim` or beyond:

- SRC-relative source path;
- source class (tracked / tracked-modified / untracked / scratch);
- worktree SHA256 at the time of copying, and the HEAD blob hash where applicable;
- the `paper_scope` ids it supports;
- the authoritative parity artifact and its SHA256;
- any divergence from a competing implementation, stated explicitly;
- the release gate, if any;
- the migration state.

This is what makes the public repository auditable: a reader can trace any published number to
a named artifact, and any file in this repository back to the exact bytes it was copied from.
