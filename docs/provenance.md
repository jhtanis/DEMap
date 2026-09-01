# Provenance

This repository was assembled from a separate research repository under a
**copy-first rule**: copy the bytes that actually ran, prove they behave
identically, and only then refactor. This page explains what that guarantees a
reader, and how to check it.

---

## The scope rule

The manuscript is the boundary. A file is here only if it produces a reported
result or is a dependency of code that does. Abandoned approaches, superseded
runs and debugging variants are deliberately absent.

`manifests/paper_scope.yaml` is the gate: one entry per reported table, figure
and prose claim, naming the code and configuration that produce it. Nothing is
added to this repository because an experiment exists — it is added because a
`paper_scope` entry needs it.

`tests/tier1_invariants/test_paper_scope_coverage.py` enforces the other
direction: every reported figure and table must map to a module that exists. That
test was written because the audit found four figures recorded as "superseded"
when the only thing that had produced them was an unmigrated notebook.

---

## The migration ledger

`manifests/source_migration.yaml` records, for every candidate file:

| field | meaning |
|---|---|
| `src` / `dst` | where it came from and where it lives now |
| `worktree_sha256` | its digest at the moment it was copied |
| `migration_state` | `copied_verbatim`, `parity_established`, `parity_reverified`, or `not_migrated_by_decision` |
| `divergence_note` | exactly what changed on the way in, if anything |
| `exclusion_reason` | why a declined file was declined |

Two consequences worth knowing when reading the code:

**The implementation that produced a published number was not always the
committed one.** Where a working-tree or scratch version was the one that ran,
that version is what was migrated, and the ledger says so.

**Declined files carry a written reason.** A test fails if any ledger entry is
left merely uninspected, so "not migrated" never means "nobody looked".

---

## How parity is checked without the data

Re-running a stage to prove parity is often impossible for a reader: it needs the
full artifact tree, GPUs, and in some cases inputs we cannot distribute. So
parity is pinned structurally instead.

`tests/fixtures/source_parity.json` records a hash of each migrated function's
**abstract syntax tree**, stripped of docstrings and of names that renaming
legitimately changes. The hash is insensitive to comments, formatting and the
function's own name; it is sensitive to any change in what the code does. The
suite recomputes it from this repository and fails on any difference — offline,
with no data and no network.

The keyword arm is held to the same standard by
`tests/tier2_behavior/test_cde_match_migration.py`, which asserts each migrated
module is AST-identical to its source once imports and docstrings are stripped.

This is a guard against silent drift, not a proof of correctness: two functions
with the same hash do the same thing, but agreeing hashes say nothing about
whether the original was right. Numerical parity against the published artifacts
is asserted separately in `tests/tier3_regression/`.

---

## When a pin legitimately changes

Re-pinning is allowed, but it has to be justified in the ledger rather than
edited quietly. The one example currently in the repository is instructive: while
the keyword retriever's redistribution was unresolved it was reached through a
runtime adapter, which hoisted a module-level indirection where the source had a
deferred in-function import. When the retriever shipped, restoring the source's
own structure made the function's AST hash **equal the research source's
exactly** — so the re-pin removed a migration divergence rather than accepting a
new one. `source_parity.json` records that reasoning inline.

---

## What is not here, and why

| not included | reason |
|---|---|
| NCI-supplied Oracle PL/SQL and the CDE Match logic PDF | NCI's source material; redistribution unresolved |
| saved output of the live NCI CDE Match service | a frozen output of an external service; not reproducible locally from this repository |
| fine-tuned bi-encoder and cross-encoder weights | ours, but redistribution not yet determined; reproducible by retraining |
| raw GDC curation submissions | workflow metadata with no reproduction value; the derived evaluation set ships instead |
| the superseded CIMAC permissible-value workbook | public but changed upstream; the PV-enriched evaluation set ships instead |
| large derived artifact trees (candidate tables, feature tables, per-condition runs) | regenerable, and gigabytes; excluded by size, not by rights |
| the caDSR XML exports and the derived catalog | ~1.3 GB; publicly re-acquirable with pinned digests |

None of the first three has ever entered this repository or its git history.

**Our own implementations of the keyword arm are included in full** — the Python
approximation to NCI CDE Match and the CDE Match-Fuzzy retriever both ship in
`src/demap_repro/lexical/cde_match/`. Only NCI's own supplied material is
withheld, and nothing depends on it.

**Both external evaluation sets ship**, in `data/frozen/`: CIMAC's 131-query
PV-enriched set and GDC's 72-query curated set. See
[`../data/frozen/README.md`](../data/frozen/README.md) for why each is frozen
rather than rebuilt.

What each exclusion costs a reader:
[`reproducing.md`](reproducing.md#reproducibility-boundaries).

---

## Data

**Every dataset this project uses is public**, and the two caDSR snapshots are
downloadable with verified digests — see [`data_sources.md`](data_sources.md).
Nothing is withheld here for privacy. The exclusions above are about
redistribution rights and file size.
