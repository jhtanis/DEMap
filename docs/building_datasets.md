# Building the datasets

From the two caDSR archives to the benchmark, the splits, the six evaluation
sets and the 62,976-record retrieval catalog.

Get the inputs first: [`data_sources.md`](data_sources.md). Set up the
environment first: [`environment.md`](environment.md).

Everything below resolves paths under `$DEMAP_DATA_ROOT`. Nothing resolves into
the authors' workspace.

```bash
export DEMAP_DATA_ROOT=/path/to/your/data/tree
```

---

## What gets built

```
caDSR 2026-01-12 XML
    │
    ├─ extract ─► merge ─► enrich ──────────────► cde_master_enriched.parquet
    │                                              (79,479 records)
    ├─ build-queries ──► queries.parquet          ALT and REF source-like strings
    └─ build-pairs   ──► pairs.parquet            69,102 query–CDE pairs
                              │
                              └─ split ──────────► train / val_train / val_dev / test
                                                   + CCTG, OID ALT, CDASH holdouts

caDSR 2026-06-18 XML
    └─ catalog-filter ─────────────────────────► production catalog
                                                   79,827 → 62,976 records

GDC + CIMAC (supplied)  ──► imported to the same schema

           ▼
    reachable-splits ──► materialize-eval ──► the six canonical evaluation sets
                                              3,959 / 1,097 / 1,766 / 324 / 72 / 131
           ▼
    leakage-filter ────► the filtered lists behind Table S6
```

`val_train` is the split the manuscript calls **Reranker Training**, and
`val_dev` is **Validation**. The code never uses the manuscript names; this is
the only place the two vocabularies meet.

---

## 1. Build the benchmark

`configs/pipeline.yaml` is the contract: it names every input, output and
parameter for stages 1–6. Read it before running anything — it is short, and it
is the file to edit rather than passing a dozen flags.

```bash
demap make-dataset --config configs/pipeline.yaml
```

Runs, in order: `extract` → `merge` → `enrich` → `build-queries` → `build-pairs`
→ `split`. Resumable and idempotent: each stage skips if its outputs exist.

```bash
demap make-dataset --start-at build-queries        # resume partway
demap make-dataset --force                         # rebuild, overwriting
```

**Point it at your download.** `pipeline.yaml`'s `extract.input` defaults to
`data/raw/cadsr_xml`, and accepts either the 14 unzipped XML files **or** a
directory containing the zip. To use a different snapshot without editing the
config:

```bash
demap make-dataset --query-xml /path/to/cadsr_2026-01-12
```

| output | what it is |
|---|---|
| `data/processed/cde_master_enriched.parquet` | 79,479 records, the January construction catalog |
| `data/processed/queries.parquet` | normalized ALT and REF source-like query strings |
| `data/processed/pairs.parquet` | **69,102** constructed query–CDE pairs |
| `data/processed/splits/` | `train`, `val_train`, `val_dev`, `test`, and the three caDSR-derived holdouts |

Which alternate-name and reference-document types become queries is *not*
hardcoded — it is the three curator allowlists in `configs/allowlists/`, which
ship here and are referenced from `pipeline.yaml`.

Splitting is at **query level**, stratified by query source and provenance
family, seed 1234. A query and all its pairs stay together, so no query appears
in two splits.

---

## 2. Build the retrieval catalog

This is the **June** snapshot, and the step where the two-snapshot distinction
becomes concrete.

```bash
demap catalog-filter \
    --catalog-xml /path/to/cadsr_2026-06-18 \
    --catalog-eligibility production_cde_match \
    --catalog-out data/processed/cde_catalog_enriched.parquet
```

Eligibility mode `production_cde_match` reproduces the production filter:

| | records |
|---|---|
| raw June export | 79,827 |
| − retired (`WORKFLOWSTATUS` contains RETIRED) | 16,846 |
| − TEST / Training administrative contexts | 21 |
| (met both criteria) | 16 |
| **kept** | **62,976** (62,858 unique public identifiers) |

The filter is on `WORKFLOWSTATUS`, not `REGISTRATIONSTATUS` — see
[`data_sources.md`](data_sources.md) for why that distinction matters when you
check the archive file yourself.

This filtering is administrative and has nothing to do with the
machine-learning splits.

---

## 3. Add the two external evaluation sets

Neither is built from the caDSR export. GDC must be supplied; CIMAC's source
workbook is a public NCI download and its exact evaluation set ships here (see
[`data_sources.md`](data_sources.md)).

```bash
demap import-gdc --help          # the two curated GDC tables -> split parquet schema
demap import-cimac --help        # the CIMAC workbook -> split parquet schema
```

GDC arrives as two curation tables (ALT-name and question-text batches) that
repeat one row per permissible value; the importer collapses them to one row per
query and resolves the CDE version from the catalog.

**CIMAC needs one extra step.** `demap import-cimac` reads the public Appendix A
workbook and reproduces the CIMAC query identities and gold CDEs — but not the
permissible-value enrichment, whose upstream workbook has since changed. Use the
frozen evaluation set instead:

```bash
cp data/frozen/cimac_v2.parquet \
   "$DEMAP_DATA_ROOT/data/processed/eval_canonical/cimac_v2.parquet"
```

That file is the exact PV-enriched input the manuscript used; see
[`../data/frozen/README.md`](../data/frozen/README.md).

If you cannot supply GDC, everything else still runs — it is 1 of the 6
evaluation sets, and the others are unaffected.

---

## 4. Filter to reachable gold, then materialize

A query whose gold CDE is absent from the retrieval catalog cannot be retrieved
by anyone. Those queries are **dropped**, never counted as misses — this is what
makes the denominators honest.

```bash
demap reachable-splits \
    --splits-dir data/processed/splits \
    --eligible   data/processed/cde_catalog_enriched.parquet \
    --out-dir    data/processed/splits_catalog_filtered

demap materialize-eval            # writes the six canonical sets
demap materialize-eval --dry-run  # counts only, writes nothing
```

The registry that defines the six sets is `configs/paper/eval_datasets_v1.yaml`.

### The counts you should get

| dataset | queries |
|---|---|
| Test | 3,959 |
| CCTG | 1,097 |
| OID ALT | 1,766 |
| CDASH | 324 |
| GDC | 72 |
| CIMAC | 131 |

These are **query-level** counts after canonical deduplication and reachable-gold
filtering, and they are the denominators in Table 1 and Table 4.

> **A trap worth knowing.** `eval_datasets_v1.yaml` also records
> `original_rows` / `reachable_rows` (Test 4,258, CCTG 1,159, OID ALT 1,908,
> CDASH 527). Those are **pair-level, pre-deduplication** counts, not the paper's
> denominators. Bi-encoder metrics use the pair-level Validation denominator
> 4,234; every other family uses query-level. The query-level counts above are
> locked in `demap_repro.evaluation.canonical_datasets.PAPER_EVAL_QUERY_COUNTS`
> and asserted by the test suite, so a build that drifts fails loudly.

---

## 5. Training–evaluation overlap

Section S1.5 and Table S6. Queries whose normalized text *and* gold CDE both
recur in training are identified and removed, and the same saved rankings are
rescored against the filtered lists.

```bash
demap leakage-filter --dry-run     # report only
demap leakage-filter               # write the filtered lists
```

Expected: 51 queries removed overall, leaving 1,089 / 1,759 / 310 / 68 / 113 for
CCTG / OID ALT / CDASH / GDC / CIMAC. Test is in-distribution by design and is
not filtered.

---

## 6. Descriptive statistics

```bash
demap characterize        # Tables 1, S1, S2
demap pv-diagnostics      # the permissible-value overlap input to Table S3
```

Note that Table S3 as printed comes from a frozen study artifact and is **not**
regenerated by this step — see [`reproducing.md`](reproducing.md#frozen-inputs).

---

## Verifying the build

Every count above is asserted by the test suite. Point it at your tree and the
structural checks run against what you actually built:

```bash
export DEMAP_DATA_ROOT=/path/to/your/data/tree
pytest tests/tier3_regression/test_core_result_contract.py -v
```

That checks the query counts, the catalog identity (62,976 records / 62,858
public ids), the eligibility arithmetic rebuilt from the raw export, that no
gold CDE is unreachable, the allowance routing, and K = 30 — as exact integers.
It takes a few seconds and needs no GPU.

---

## Where the inputs come from, summarized

| input | source |
|---|---|
| caDSR January + June XML | public download, digests in [`data_sources.md`](data_sources.md) |
| curator allowlists | **this repository**, `configs/allowlists/` |
| pipeline contract | **this repository**, `configs/pipeline.yaml` |
| evaluation registry | **this repository**, `configs/paper/eval_datasets_v1.yaml` |
| GDC tables | **must be supplied** |
| CIMAC workbook | **must be supplied** |

Nothing on this page requires an artifact tree, a GPU, or network access beyond
the two caDSR downloads.
