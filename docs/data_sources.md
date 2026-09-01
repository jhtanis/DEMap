# Obtaining the source data

Everything this project ranks over comes from **two caDSR snapshots**, and
everything it is evaluated on comes from those snapshots plus **two externally
curated evaluation sets**. This page tells you exactly what to get, from where,
and how to tell whether you have the same bytes we did.

Read the two-snapshot distinction first — it is the single most common source of
confusion about the counts in the paper.

---

## The two caDSR snapshots do different jobs

| | **benchmark construction** | **retrieval catalog** |
|---|---|---|
| snapshot | **2026-01-12** | **2026-06-18** |
| what it supplies | source-like query strings (alternate names, reference documents) and the gold CDE links | the candidate CDEs every method ranks over |
| used by | the benchmark, the in-distribution splits, and the CCTG / OID ALT / CDASH holdouts | training pairs, in-batch negatives, candidate generation, and **all** evaluation |
| records | 79,479 (78,533 unique public ids) | 79,827 raw → **62,976** after eligibility filtering (62,858 unique public ids) |
| in the manuscript | §S1.4, §S1.2 | §S1.4, §2.1 |

The two counts in the paper's abstract track these two roles: **~69,000** is the
number of constructed *query–CDE mappings*, and **~63,000** is the size of the
*production catalog they are ranked against*. They are different quantities and
do not reconcile to each other.

The January catalog was never a retrieval universe. It is not filtered for
production eligibility and includes retired records; it exists to supply source
metadata and gold links.

---

## caDSR: where to download it

caDSR publishes a nightly XML export, and — importantly for reproduction —
**keeps a dated archive of past nightlies**.

```
current nightly   https://cadsr.nci.nih.gov/ftp/caDSR_Downloads/CDE/XML/
dated archive     https://cadsr.nci.nih.gov/ftp/caDSR_Downloads/CDE/XML_Archive/
```

The archive held **811 dated snapshots spanning 2024-02-22 to 2026-07-01** when
this was written, one per file named `releasedCDEsXML-OD.zip.YYYY-MM-DD`.

**Both snapshots this paper used are in that archive and can be downloaded
today.**

### The exact files

```bash
# Benchmark construction — 2026-01-12
curl -L -o cadsr_2026-01-12.zip \
  https://cadsr.nci.nih.gov/ftp/caDSR_Downloads/CDE/XML_Archive/releasedCDEsXML-OD.zip.2026-01-12

# Retrieval catalog — 2026-06-18
curl -L -o cadsr_2026-06-18.zip \
  https://cadsr.nci.nih.gov/ftp/caDSR_Downloads/CDE/XML_Archive/releasedCDEsXML-OD.zip.2026-06-18
```

| snapshot | sha256 | bytes | members |
|---|---|---|---|
| 2026-01-12 | `0fa759d39f0c966aedf2623c17386601b639b5502fb99ee730e66d8cfdd6a457` | 119,919,708 | 14 |
| 2026-06-18 | `e37f90a790a987747fdb87d7cbe9937bc342cd5d8ec9a80420273135681b6494` | 122,032,967 | 14 |

```bash
sha256sum cadsr_2026-01-12.zip cadsr_2026-06-18.zip
```

Each zip contains 14 XML files named `cde_xml_<timestamp>_<n>.xml`. The January
archive file was verified member-by-member (CRC32) against the copy this study
ran on: **14 of 14 byte-identical.** So a reader downloading it today holds the
same bytes, not merely a similar export.

### Expected layout

```
$DEMAP_DATA_ROOT/
  data/raw/cadsr_xml/                 # unzip the 2026-01-12 archive here
    cde_xml_20260111204212_1.xml
    ...
    cde_xml_20260112012022_14.xml
```

`demap make-dataset` accepts either a directory of the 14 XML files **or** a
directory containing the zip, so unzipping is optional.

---

## Two things that will otherwise look like errors

### 1. The archive README's "RETIRED" line refers to a different field

The archive `README` states the export contains CDEs where
`registration_status not like '%RETIRED%'`. Read literally alongside this
project, that suggests the file cannot contain retired records — yet the
manuscript reports removing 16,846 retired records from the June export.

**These are two different fields.** Each `<DataElement>` carries both:

```xml
<WORKFLOWSTATUS>RELEASED</WORKFLOWSTATUS>
<REGISTRATIONSTATUS>Qualified</REGISTRATIONSTATUS>
```

The archive's exclusion is on `REGISTRATIONSTATUS`. This project's eligibility
filter is on `WORKFLOWSTATUS` (`ADMIN_STUS_NM_DN` in the source SQL), which is a
separate axis — see `src/demap_repro/data/eligibility.py`, which states
explicitly that `registration_status` is *not* an eligibility field.

Concretely: the 2026-06-18 archive file contains **16,785 records whose
`WORKFLOWSTATUS` contains `RETIRED`** (nearly all `RETIRED ARCHIVED`). That is
expected, and it is what the production filter removes. You have the right file.

### 2. Do not verify record counts by counting XML tags

The manuscript's 79,479 and 79,827 are **rows produced by this repository's
extractor**, after its parsing and deduplication rules. They are not counts of
`<DataElement>` elements in the raw XML, which are higher (about 80,900 and
81,200 respectively).

To check a count, run the extractor and count the parquet rows — see
[`building_datasets.md`](building_datasets.md). A `grep -c` over the XML will
not agree, and that disagreement is not a defect.

---

## The two external evaluation sets

Neither was derived from the caDSR export, and neither is distributed with this
repository. Both are small.

### GDC — must be supplied

Two curated tables mapping GDC properties to their expert-assigned CDEs:

```
data/raw/gdc/gdc_alt_names.csv        60 queries
data/raw/gdc/gdc_question_text.csv    60 queries
```

They are caDSR **curation batch submissions**, not exports of a public GDC
service. The `Entity` and `Perm Val` columns correspond to GDC Data Dictionary
properties and values, which are public — but the `Preferred CDE ID` on every
row is the curator's expert assignment, and *that mapping is the benchmark
label*. No public GDC endpoint carries it, so **the GDC evaluation set cannot be
regenerated from a public source.** It has to be supplied.

`src/demap_repro/data/gdc.py` documents the schema it expects, so an
equivalently-shaped table from your own curation will run through the same path.

### CIMAC — must be supplied

```
data/raw/cimac/CIMAC-CIDC_Master_AppendixA_v2.xlsx   →   131 queries
```

Redistribution of the CIMAC material is unresolved, so neither the workbook nor
the evaluation set derived from it is included here. See
[`reproducing.md`](reproducing.md) for what this does and does not block: CIMAC
is one of six evaluation sets, and the other five are unaffected.

---

## Everything the pipeline needs, at a glance

| input | what it is | how you get it |
|---|---|---|
| caDSR 2026-01-12 XML | benchmark construction | **public download**, digest above |
| caDSR 2026-06-18 XML | retrieval catalog | **public download**, digest above |
| curator allowlists | which alternate-name and reference-document types become queries | **in this repository**, `configs/allowlists/` |
| pipeline config | the dataset build contract | **in this repository**, `configs/pipeline.yaml` |
| protocol configs | bi-encoder, cross-encoder, BM25, HGBC settings | **in this repository**, `configs/paper/` |
| GDC evaluation tables | 120 curated query→CDE rows | **must be supplied** (not public) |
| CIMAC workbook | 131 curated query→CDE rows | **must be supplied** (redistribution unresolved) |
| base model checkpoints | all-MPNet, BioSimCSE, PubMedBERT, MedCPT, BGE, MiniLM | **public**, from HuggingFace; see [`environment.md`](environment.md) |
| fine-tuned weights | our trained bi-encoder and cross-encoder | **not distributed**; retrain from the specified protocols |
| official NCI CDE Match output | the live service's saved results | **not reproducible by anyone**; reported as a frozen external number |

Only the last four rows are not obtainable by downloading. Of those, the weights
are reproducible by retraining, and the official-service column is a single
reported comparison, not an input to the system.

---

## A note on nightly snapshots

caDSR's export is a nightly, not a versioned release. Today's
`releasedCDEsXML-OD.zip` will differ from both snapshots above, and the live
registry changes daily — so a count you take from a current export will not match
the manuscript's. That is why the archive matters: it is what makes the exact
historical inputs recoverable, and why the digests above are worth checking.
