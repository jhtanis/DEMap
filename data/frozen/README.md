# Frozen study inputs

One file lives here. Everything else this pipeline consumes is either
downloadable (see [`../../docs/data_sources.md`](../../docs/data_sources.md)) or
built by the repository from those downloads.

---

## `cimac_v2.parquet`

The CIMAC evaluation set exactly as the manuscript used it.

| | |
|---|---|
| rows | **131** — one per evaluation query, `cimac_v2::001` … `cimac_v2::154` |
| sha256 | `1d0797330864cb8a73d277c1885fa5bba41a63d3a011cfaaa148e59c6dce4fe0` |
| size | 26,963 bytes |
| used by | Table 1, Table 4, Table S1, Table S2, Table S4, Table S5, Table S6, Figure S6, §S1.5 |

### Schema

| column | |
|---|---|
| `query_id` | stable identifier, `cimac_v2::NNN` |
| `query_source`, `query_field`, `family` | provenance labels used by the split router |
| `query_text_raw` | the source-side data-element name |
| `query_text_q3` | **the representation evaluation actually consumes**: `query_text_raw` plus the permissible-value block |
| `cde_publicid`, `cde_version`, `cde_id` | the accepted gold CDE |
| `gold_versioned` | whether the gold resolved to a specific version |
| `PV_N`, `PV_TYPE`, `PV_BLOCK_SDE`, `pv_attached`, `pv_block` | permissible-value metadata; non-empty for 59 of the 131 queries |

### Where it comes from

CIMAC's queries and gold CDE mappings originate from the publicly available NCI
CIMAC-CIDC clinical-data-element template — **template and data-element
metadata, not patient-level study data**. The Appendix A workbook NCI serves
today is byte-identical to the one this study used
(sha256 `6d657dfd3f72…c71a3`, 43,312 bytes), so that half of the provenance is
independently verifiable by anyone:

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
from today's copy would therefore silently produce a different evaluation input
and different CIMAC numbers.

So the exact PV-enriched representation is preserved here instead:

```
public Appendix A workbook   ──►  reproducible query/gold provenance
this frozen artifact         ──►  the exact PV-enriched evaluation input
                             ──►  the manuscript's CIMAC results
```

27 KB of derived query text and public CDE identifiers is a smaller and more
honest artifact than shipping a superseded copy of an upstream workbook, and it
is what makes every reported CIMAC result reproducible exactly.

### Using it

Copy it into your data tree where the evaluation registry expects it:

```bash
cp data/frozen/cimac_v2.parquet \
   "$DEMAP_DATA_ROOT/data/processed/eval_canonical/cimac_v2.parquet"
```

`configs/paper/eval_datasets_v1.yaml` declares that path. Its digest is
asserted by `tests/tier3_regression/test_cimac_frozen_input.py`, so a
substitution cannot go unnoticed.
