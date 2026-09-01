# Frozen study inputs

Two files live here — the two evaluation sets that cannot be rebuilt from a
public source in the form the manuscript used. Everything else this pipeline
consumes is either downloadable (see
[`../../docs/data_sources.md`](../../docs/data_sources.md)) or built by the
repository from those downloads.

| file | rows | bytes | sha256 |
|---|---|---|---|
| `cimac_v2.parquet` | 131 | 26,963 | `1d0797330864cb8a73d277c1885fa5bba41a63d3a011cfaaa148e59c6dce4fe0` |
| `gdc_combined.parquet` | 72 | 38,833 | `289c4f5fc966748c54fc3d4c8c0edda28937b55910796a65db7834034bf86f21` |

Both are byte-identical to the inputs the study evaluated on, and both digests
are asserted by the test suite. Put them where the evaluation registry expects
them with:

```bash
export DEMAP_DATA_ROOT=/path/to/your/data/tree
demap materialize-eval --frozen-only
```

That verifies each pinned digest before and after copying, creates the
destination, and is idempotent. A destination that already matches is skipped; a
destination that differs is refused rather than overwritten. `--frozen-only`
needs no splits tree; plain `demap materialize-eval` does the same thing as part
of materializing all six canonical datasets.

**This directory is read-only.** No stage writes into it.

---

## `cimac_v2.parquet` — 131 queries

The CIMAC evaluation set exactly as the manuscript used it. Feeds Table 1,
Table 4, Tables S1, S2, S4, S5, S6, Figure S6 and §S1.5.

### Schema

| column | |
|---|---|
| `query_id` | stable identifier, `cimac_v2::NNN` |
| `query_source`, `query_field`, `family` | provenance labels used by the split router |
| `query_text_raw` | the source-side data-element name |
| `query_text_q3` | **the representation evaluation consumes**: `query_text_raw` plus the permissible-value block |
| `cde_publicid`, `cde_version`, `cde_id` | the accepted gold CDE |
| `gold_versioned` | whether the gold resolved to a specific version |
| `PV_N`, `PV_TYPE`, `PV_BLOCK_SDE`, `pv_attached`, `pv_block` | permissible-value metadata; non-empty for 59 of 131 |

### Where it comes from

CIMAC's queries and gold CDE mappings originate from the publicly available NCI
CIMAC-CIDC clinical-data-element template — **template and data-element
metadata, not patient-level study data**. The Appendix A workbook NCI serves
today is byte-identical to the one this study used (sha256 `6d657dfd3f72…c71a3`,
43,312 bytes), so that half of the provenance is independently verifiable:

```
public Appendix A workbook  ──►  131 query identities and their gold CDEs
```

Verified directly: all 131 canonical `query_id` → `query_text_raw` pairs
reproduce from the public workbook, and every canonical gold public identifier
is present in it.

### Why it is shipped rather than regenerated

Permissible-value metadata came from a **second** public workbook,
`CIMAC-CIDC_Permissible_Values.xlsx`. That file is a live upstream document: the
version NCI serves today is **not** byte-identical to the one used during study
preparation, and no dated archive of it exists. Regenerating `query_text_q3`
from today's copy would silently produce a different evaluation input and
different CIMAC numbers, so the exact representation is preserved here instead.

---

## `gdc_combined.parquet` — 72 queries

The GDC evaluation set exactly as the manuscript used it. Feeds Table 1,
Table 4, Tables S1, S2, S4, S5, S6, Figure S6 and §S1.5, and the §3.8 claim that
GDC is near ceiling at 70/72.

Two query styles are combined, which is what "combined" names:

| `query_source` | `query_field` | queries |
|---|---|---|
| `GDC_QTXT` | `preferred_question_text` | 39 |
| `GDC_ALT` | `alternate_name` | 33 |

### Schema

| column | |
|---|---|
| `query_id`, `pair_id` | stable identifiers (content hashes) |
| `query_source`, `query_field`, `family` | which GDC style the query came from |
| `query_text_raw` | the source-side GDC property text |
| `query_text_q3` | **the representation evaluation consumes** |
| `query_text`, `query_text_q4` | the Q2 and Q4 representation variants |
| `cde_publicid`, `cde_version`, `cde_id` | the accepted gold CDE, populated for all 72 |
| `PV_N`, `PV_TYPE`, `PV_BLOCK_SDE`, `PV_BLOCK_CDE`, `pv_attached`, `pv_block` | permissible-value metadata |
| `split_source`, `batch_name`, `seq_id` | provenance of the curation batch; not read by evaluation |

### Where it comes from, and why it is shipped

The GDC mappings were **manually curated**. They were assembled as caDSR
curation submissions in which a curator assigned, for each GDC property, the CDE
that property should map to. That assignment — the `cde_id` column here — **is
the benchmark label.**

**These gold mappings cannot be recovered from a public GDC source.** The GDC
Data Dictionary publishes the properties and their permissible values, so the
*query* side has a public analogue; it does not publish, and has no endpoint
for, which caDSR CDE each property maps to. That linkage exists only because a
curator made it. No public GDC API or data-dictionary export will regenerate it.

The raw curation submissions are deliberately **not** distributed. They carry
workflow metadata — submitting user, comments, per-cell tips, "do not use"
flags — and repeat one row per permissible value, 1,822 rows for what is 120
curated queries. None of that is needed to reproduce anything. What the
manuscript workflow actually consumes is this compact derived set, so this is
what ships:

```
manual expert curation  ──►  frozen gdc_combined.parquet  ──►  the manuscript's GDC results
```

This is the reason GDC is a *frozen* input rather than a *regenerable* one, and
the reason it is shipped rather than left to be supplied: without it the GDC
column of every reported table is unreproducible.
