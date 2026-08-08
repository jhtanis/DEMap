# Self-handoff — 2026-08-08 — DEMap frozen paper repository

Self-contained. A fresh session can resume from this file alone; no chat history
needed.

---

## A. Purpose

**`jhtanis/DEMap` is the frozen paper repository.** It holds only the code and
workflows behind results reported in the authoritative manuscript, plus the
dependencies those results need. Abandoned approaches, superseded runs and
debugging variants are deliberately absent.

**Future exploratory research does not belong here.** It belongs in
`tanisjh/demap-repro` (configured as the `research` remote) or another research
branch. Do not widen this repository past the manuscript's scope.

---

## B. Authoritative manuscript

| | |
|---|---|
| path | `manuscript/cde_paper_v21.docx` in the research checkout |
| sha256 | `2df7895cd8657a7928be220c958aaa38a85c5b5199526f03979c4967b159eecd` |

**Table S3 was intentionally corrected** to the canonical 69,102-pair computation
(author-approved, 2026-08-08). The previously published counts came from a
`pairs.parquet` build of 69,844 rows that had since been overwritten by the
paper-era 69,102-row build — the one every model stage consumed and the one §3.1
and S1.3 quote. The table was internally inconsistent with the rest of the
manuscript.

| group | rows | block match | query shorter | mean Jaccard |
|---|---|---|---|---|
| overall | 38,964 | 0.4% | 30.1% | 0.168 |
| ENUM | 24,158 | 0.4% | 45.3% | 0.153 |
| BINARY_WITH_UNKNOWN | 3,040 | 0.1% | 0.0% | 0.333 |
| BINARY_WITH_NA | 8,815 | 0.0% | 0.0% | 0.123 |

The accompanying S3.5 prose denominator reads 38,964. Eleven of the twelve rate
cells were unaffected; only ENUM query-shorter moved (45.2% → 45.3%).

**This was never PV-code drift.** The PV implementation regenerates the frozen
benchmark's blocks with **100% row-level parity** (46,948 / 46,948 exact on both
the query and CDE sides, 0 missing→present, 0 present→missing). **No PV
regression.** Only the *input* to the frozen diagnostic had been replaced.

The corrected table now regenerates end to end here:
`pairs.parquet` → `data/pv_frozen_diagnostics` → `reporting/dataset_tables_pv_overlap`.

---

## C. Repository state

| | |
|---|---|
| branch | `main` |
| HEAD | `a77f717dc3422ba3b3348d4d3a5d2816a570eb82` (before this handoff commit) |
| origin | `git@github.com-jhtanis:jhtanis/DEMap.git` — frozen paper repo, private |
| research | `https://github.com/tanisjh/demap-repro.git` — retained research repo, private |
| commits | 20 (before this handoff commit) |
| working tree | clean |
| tests | **484 passed** with artifacts; **477 passed / 7 skipped** offline on a clean clone |

- `.scratch/` is gitignored and nothing under it is tracked.
- **No runtime dependency on the research checkout.** Every stage resolves inputs
  through `DEMAP_DATA_ROOT` or an explicit `--` argument. Absolute research paths
  survive only inside frozen artifacts and provenance fields.
- **No runtime code executes from a historical `.scratch` or manuscript revision
  directory.** References to those locations are provenance strings only.
- `tests/tier1_invariants/test_import_safety.py` fails if any module reads,
  writes or fits a model at import time — added after migrating research scripts
  twice caused imports to regenerate figures in the research checkout.

### Network note (machine-specific, not a secret)

Compute nodes here have no DNS for `github.com`; only an HTTP proxy. SSH reaches
GitHub over port 443 through it. If a push fails to resolve the host:

```bash
GIT_SSH_COMMAND='ssh -o "ProxyCommand=nc --proxy <proxy-host>:3128 --proxy-type http %h %p" \
  -o HostName=ssh.github.com -o Port=443' git push origin main
```

`$http_proxy` names the correct proxy host for the current node. Nothing
proxy-specific is persisted in git config or `~/.ssh/config`.

---

## D. Migration / hardening status

All **15 paper workflow stages** are represented (A dataset · B1–B5b bi-encoder ·
C lexical · D pool · E cross-encoder · F reranker · G final evaluation · H1/H2
sensitivity · I reporting).

Paper-essential behaviour that lived in scratch, untracked files, pinned
worktrees or Slurm chains is now first-class:

- **CE training pairs** came from the **pinned worktree `5f06f10`**, which is what
  chain F2 executed — not `scripts/build_ce_training_pairs_fulltrain.py`, which an
  earlier ledger named and which differs byte-wise.
- **172-query CE exclusion** is a named, tested step:
  `crossencoder/eligibility.drop_train_queries_without_obtainable_positive`
  (47,645 → 47,473; train-only; every retained query has an obtainable positive).
- **Fixed-K split routing** is first-class in `reranker/split_routing.py` and
  tested from its *effect* on the published feature table: not one
  allowance-blocked query on a caDSR-derived split retains an exact gold
  candidate, while GDC and CIMAC keep theirs.
- **Leakage filtering** is first-class in `data/leakage_filter.py` and reproduces
  all 51 removals exactly (CCTG 8 · OID ALT 7 · CDASH 14 · GDC 4 · CIMAC 18).
- **K selection** is one implementation (`pool/select_k.py`), replacing three
  near-duplicates; reproduces K = 30, ceiling 0.9888, realized pool 27.6.
- **Allowance sensitivity** is one package with the arm as a required parameter:
  `fixed_070` (shipped model, inference-time mismatch — the robustness evidence)
  versus `retrained_per_rate` (refit at each rate; cannot suffer that mismatch, so
  it is not robustness evidence). Do not conflate them.
- **Figure 4/5 reporting builders** were written fresh against the corrected
  `*_v2_eligible` roots. The research builder reads pre-correction roots and would
  restore superseded values; a test asserts those values never reappear.
- **Table S3** regenerates end to end (see §B).
- **Scope coverage** has a regression test: every reported figure and table must
  map to a module that exists (`test_paper_scope_coverage.py`). This was added
  after an audit found five paper outputs with no producer here.

### Migration ledger

| state | n |
|---|---|
| `copied_verbatim` | 114 |
| `parity_reverified` | 10 |
| `parity_established` | 7 |
| `not_migrated_by_decision` | 19 |
| **total** | **150** |

Every declined entry carries a written reason; a test fails if any is left merely
uninspected. Declined classifications: superseded by hardened code (10),
provenance only (4), manual/non-executable (2), historical/stale (1), gated
external material (1), unused by v21 (1).

Gated material: **19 entries, 0 copied.**

---

## E. Scientific scope boundary — excluded, and staying excluded

Do not reintroduce any of these into the frozen paper repository:

- HGBC provenance / A-B selection experiment
- provenance dropout (removed from the migrated HGBC trainer; the shipped model
  trained at rate 0)
- exact-match pinning
- Phase 2 `hardcurr` curriculum arm
- historical wrong-band semi-hard runs
- XGBoost reranker ablation
- SapBERT experiments not in the manuscript
- non-paper feature ablations (`kwtier3`, `seqkw`, ablations B/D/G)
- stale pre-correction result paths
- superseded CIMAC construction paths

An experiment not represented in the manuscript, and not required to reproduce
something in it, does not belong here.

---

## F. Redistribution and license status

- The repository is **private**.
- **License: to be determined.** No `LICENSE` file. No license declaration in
  `pyproject.toml`. Do not add one without explicit team approval, and do not
  characterize the current status beyond "to be determined".
- **Eight redistribution determinations remain open** (see
  `docs/release_readiness.md` for the per-gate basis): NCI-supplied source,
  CDE Match derivative code, derived candidate artifacts, the frozen official
  service output, CIMAC material, caDSR snapshots, our fine-tuned bi-encoder
  weights, and MedCPT-derivative cross-encoder weights.
- **No gated blobs anywhere in this repository's git history**, verified by
  hashing every object reachable from every ref against the gated file set.
- **All project datasets are public.** caDSR is a public NCI registry;
  CIMAC-CIDC, GDC and the CDISC-derived sets are public resources. The open gates
  concern particular source code, derivative code and artifacts, model weights, or
  attribution/terms — **not data privacy.** Do not describe this project's data as
  private.

### `reranker/features/cdematch.py` — reclassified as independent

Previously gated on the strength of its name. A content review found it reads only
`query_id`, `cdematch_rank`, `cdematch_score` and `in_cdematch_topk` and emits
normalization, log rank, top-1 and pairwise margins, a within-query z-score and a
candidate count. No CDE text, no catalog field, no rule set, no string comparison,
and no reference to SQL, Oracle or any supplied artefact. Decisively, the fixed-K
builder already applied the same function verbatim to the *keyword* retriever's
output, renaming results to `seqkw_*` — which only works because it is
retriever-agnostic. It is therefore generic ranked-list arithmetic and ships here.

**This is distinct from, and does not weaken, the still-gated material:**
`cde_match_clone.py`, `keyword_retriever.py` and the NCI-supplied PL/SQL remain
gated and uncopied. That review deliberately did not reassess them.

---

## G. Git repository roles

| remote | repository | role |
|---|---|---|
| `origin` | `jhtanis/DEMap` | **frozen paper repository** |
| `research` | `tanisjh/demap-repro` | retained repository for future research and development |

Do not merge future exploratory work from `research` into `origin` unless
explicitly requested for a manuscript correction. Do not conflate the two.

Note that the two remotes authenticate as **different GitHub accounts**: `origin`
over SSH via the `github.com-jhtanis` host alias, `research` over HTTPS. The alias
selects a key; it is not evidence of account identity on its own.

---

## H. Operating rules

- **Never use `/tmp`.** Use `./.scratch/` inside this repository.
- **No AI attribution trailers** in commits — no `Co-authored-by: Claude`, no
  `Generated-by`, no `AI-assisted`, nothing similar. The history is currently
  clean of all of these; keep it that way.
- **Never force-push the frozen paper repository.** Never rewrite its history.
- **Do not select or add a license** without explicit team approval.
- **Do not introduce gated material** into the git history — it cannot be retracted
  from a remote once pushed.
- The research checkout at `/data/nextgen2/james/tasks/cde_project/demap` is
  **read-only**. Its HEAD is `cf8743f535f4f4fb8eb7bcc71362986d615ec58f` with 97
  porcelain lines; the only authorized modification to date was the Table S3
  manuscript correction.
- **The `demap_repro` Python package name is intentional.** The public project is
  DEMap; the import package is not renamed, because doing so would break every
  module path and falsify the provenance records that name them. A public project
  name and an import package need not match. Do not "fix" this.

---

## I. What remains

Organizational and release-oriented, not scientific:

1. **Team licensing decision** — the only item that blocks anything.
2. **Written determinations** on the eight open redistribution gates.
3. **Decision on making the repository public**, once 1 and 2 are settled.
4. **Final public-release review** before any visibility change.

No further scientific cleanup is needed or recommended. Only act on a
reproducibility defect if one is actually discovered — do not refactor for
aesthetics, and do not expand scope past the manuscript.
