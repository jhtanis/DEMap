#!/usr/bin/env python3
"""extract_cadsr_xml_to_parquet.py

Phase 1-2 extractor for caDSR CDE XML.

This script implements the unified extraction pipeline Phases 1-2:
  - Canonical CDE core fields
  - ISO-11179 components (DEC, Object Class, Property, Value Domain)
  - Permissible values
  - Classifications
  - Reference documents (question text, etc.)
  - Alternate names (aliases)

Design:
  - Per-file extraction: each input XML file produces its own set of Parquet tables
    under: <out_root>/<xml_stem>/
  - Memory-safe streaming parse using lxml.etree.iterparse.
  - Incremental Parquet writes using pyarrow.parquet.ParquetWriter.

Usage examples:
  # Extract one file
  python extract_cadsr_xml_to_parquet.py \
    --input /path/to/cde_xml_1.xml \
    --out-root out \
    --chunk-size 50000

  # Extract all .xml files in a directory (non-recursive)
  python extract_cadsr_xml_to_parquet.py \
    --input /path/to/xml_dir \
    --out-root out

Notes:
  - All columns are written as strings for robustness (missing values, mixed formats).
  - cde_xml_row_id is taken from <DataElement> attribute "num" when present.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from lxml import etree

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except ImportError:  # pragma: no cover
    # Keep import-time failure non-fatal so the rest of the package/tests can
    # still import. We raise a clear error at runtime when parquet I/O is used.
    pa = None  # type: ignore
    pq = None  # type: ignore


def _localname(tag: str) -> str:
    return tag.split('}')[-1] if '}' in tag else tag


def _clean_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = s.strip()
    return s2 if s2 != '' else None


def _child_text(elem: etree._Element, tags: Iterable[str]) -> Optional[str]:
    """Return the first non-empty child text among candidate tag names."""
    for t in tags:
        child = elem.find(t)
        if child is not None:
            val = _clean_text(child.text)
            if val is not None:
                return val
    return None


def _collect_concept_details_json(parent: Optional[etree._Element]) -> Optional[str]:
    """Extract ConceptDetails/ConceptDetails_ITEM blocks into a JSON string."""
    if parent is None:
        return None
    cd = parent.find('ConceptDetails')
    if cd is None:
        return None
    items = []
    for it in cd.findall('ConceptDetails_ITEM'):
        d = {
            'preferred_name': _child_text(it, ['PREFERRED_NAME']),
            'long_name': _child_text(it, ['LONG_NAME']),
            'con_id': _child_text(it, ['CON_ID']),
            'definition_source': _child_text(it, ['DEFINITION_SOURCE']),
            'origin': _child_text(it, ['ORIGIN']),
            'evs_source': _child_text(it, ['EVS_SOURCE']),
            'primary_flag_ind': _child_text(it, ['PRIMARY_FLAG_IND']),
            'display_order': _child_text(it, ['DISPLAY_ORDER']),
        }
        # Drop fully-empty dicts
        if any(v is not None for v in d.values()):
            items.append(d)
    if not items:
        return None
    return json.dumps(items, ensure_ascii=False)


class _BufferedParquetWriter:
    """Incremental Parquet writer with an in-memory record buffer."""

    def __init__(
        self,
        out_path: Path,
        schema: pa.Schema,
        chunk_size: int = 50_000,
        compression: str = 'snappy',
        overwrite: bool = False,
    ) -> None:
        if pa is None or pq is None:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency 'pyarrow'. This extractor writes Parquet and requires it.\n"
                "Install with: pip install pyarrow\n"
            )
        self.out_path = out_path
        self.schema = schema
        self.chunk_size = int(chunk_size)
        self.compression = compression
        self._buffer: List[Dict[str, Any]] = []

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and out_path.exists():
            out_path.unlink()

        self._writer = pq.ParquetWriter(str(out_path), schema=schema, compression=compression)
        self._cols = schema.names

    def append(self, record: Dict[str, Any]) -> None:
        # Ensure all columns exist
        full = {k: None for k in self._cols}
        for k, v in record.items():
            if k in full:
                full[k] = v
        self._buffer.append(full)
        if len(self._buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


def _schemas() -> Dict[str, pa.Schema]:
    """Define schemas for all output tables (strings only)."""

    if pa is None:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'pyarrow'. This extractor writes Parquet and requires it.\n"
            "Install with: pip install pyarrow\n"
        )

    def S(name: str) -> pa.Field:
        return pa.field(name, pa.string())

    common = [S('cde_publicid'), S('cde_version'), S('source_file'), S('cde_xml_row_id')]

    return {
        'cde_master': pa.schema(
            common
            + [
                S('preferred_name'),
                S('long_name'),
                S('preferred_definition'),
                S('context_name'),
                S('context_version'),
                S('workflow_status'),
                S('registration_status'),
                S('origin'),
                S('rai'),
            ]
        ),
        'cde_iso11179_dec': pa.schema(
            common
            + [
                S('dec_publicid'),
                S('dec_version'),
                S('dec_preferred_name'),
                S('dec_long_name'),
                S('dec_preferred_definition'),
                S('dec_context_name'),
                S('dec_context_version'),
                S('dec_workflow_status'),
                S('dec_origin'),
                # Conceptual domain (DEC-level)
                S('dec_conceptual_domain_publicid'),
                S('dec_conceptual_domain_version'),
                S('dec_conceptual_domain_preferred_name'),
                S('dec_conceptual_domain_long_name'),
                S('dec_conceptual_domain_context_name'),
                S('dec_conceptual_domain_context_version'),
                # Object class
                S('object_class_publicid'),
                S('object_class_version'),
                S('object_class_preferred_name'),
                S('object_class_long_name'),
                S('object_class_context_name'),
                S('object_class_context_version'),
                S('object_class_concept_details_json'),
                # Property
                S('property_publicid'),
                S('property_version'),
                S('property_preferred_name'),
                S('property_long_name'),
                S('property_context_name'),
                S('property_context_version'),
                S('property_concept_details_json'),
            ]
        ),
        'cde_iso11179_value_domain': pa.schema(
            common
            + [
                S('vd_publicid'),
                S('vd_version'),
                S('vd_preferred_name'),
                S('vd_long_name'),
                S('vd_preferred_definition'),
                S('vd_workflow_status'),
                S('vd_context_name'),
                S('vd_context_version'),
                S('vd_origin'),
                # Conceptual domain (VD-level)
                S('vd_conceptual_domain_publicid'),
                S('vd_conceptual_domain_version'),
                S('vd_conceptual_domain_preferred_name'),
                S('vd_conceptual_domain_long_name'),
                S('vd_conceptual_domain_context_name'),
                S('vd_conceptual_domain_context_version'),
                # VD typing
                S('value_domain_type'),
                S('datatype'),
                S('maximum_length'),
                S('minimum_length'),
                S('decimal_place'),
                S('maximum_value'),
                S('minimum_value'),
                # Representation
                S('representation_publicid'),
                S('representation_version'),
                S('representation_preferred_name'),
                S('representation_long_name'),
                S('representation_context_name'),
                S('representation_context_version'),
                S('representation_concept_details_json'),
            ]
        ),
        'cde_permissible_values': pa.schema(
            common
            + [
                S('valid_value'),
                S('value_meaning'),
                S('meaning_description'),
                S('meaning_concepts'),
                S('meaning_concept_origin'),
                S('meaning_concept_display_order'),
                S('pv_begin_date'),
                S('pv_end_date'),
                S('vm_publicid'),
                S('vm_version'),
            ]
        ),
        'cde_classifications': pa.schema(
            common
            + [
                # scheme
                S('classification_scheme_publicid'),
                S('classification_scheme_version'),
                S('classification_scheme_preferred_name'),
                S('classification_scheme_context_name'),
                S('classification_scheme_context_version'),
                # item
                S('classification_scheme_item_name'),
                S('classification_scheme_item_type'),
                S('csi_publicid'),
                S('csi_version'),
            ]
        ),
        'cde_reference_documents': pa.schema(
            common
            + [
                S('name'),
                S('document_type'),
                S('document_text'),
                S('url'),
                S('language'),
                S('display_order'),
            ]
        ),
        'cde_alternate_names': pa.schema(
            common
            + [
                S('context_name'),
                S('context_version'),
                S('alternate_name'),
                S('alternate_name_type'),
                S('language'),
            ]
        ),
    }


def _extract_one_file(xml_path: Path, out_root: Path, chunk_size: int, overwrite: bool) -> None:
    xml_path = xml_path.resolve()
    file_stem = xml_path.stem
    out_dir = out_root / file_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    schemas = _schemas()

    writers = {
        name: _BufferedParquetWriter(
            out_path=out_dir / f'{name}.parquet',
            schema=schema,
            chunk_size=chunk_size,
            overwrite=overwrite,
        )
        for name, schema in schemas.items()
    }

    # Stream parse
    # NOTE: caDSR XML has no namespaces in the samples; we still check localname.
    # NOTE: restricting to tag='DataElement' massively speeds up parsing for large files.
    # caDSR XML exports typically have no namespaces.
    context = etree.iterparse(
        str(xml_path),
        events=('end',),
        tag='DataElement',
        recover=True,
        huge_tree=True,
    )

    processed = 0
    for event, elem in context:
        # Defensive: in case of unexpected namespaces.
        if _localname(elem.tag) != 'DataElement':
            continue

        cde_xml_row_id = elem.get('num')
        source_file = xml_path.name

        cde_publicid = _child_text(elem, ['PUBLICID', 'PublicId'])
        cde_version = _child_text(elem, ['VERSION', 'Version'])

        # If a DataElement is malformed and missing IDs, skip it.
        if cde_publicid is None or cde_version is None:
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            continue

        # -------------------- cde_master --------------------
        writers['cde_master'].append(
            {
                'cde_publicid': cde_publicid,
                'cde_version': cde_version,
                'source_file': source_file,
                'cde_xml_row_id': cde_xml_row_id,
                'preferred_name': _child_text(elem, ['PREFERREDNAME', 'PreferredName']),
                'long_name': _child_text(elem, ['LONGNAME', 'LongName']),
                'preferred_definition': _child_text(elem, ['PREFERREDDEFINITION', 'PreferredDefinition']),
                'context_name': _child_text(elem, ['CONTEXTNAME', 'ContextName']),
                'context_version': _child_text(elem, ['CONTEXTVERSION', 'ContextVersion']),
                'workflow_status': _child_text(elem, ['WORKFLOWSTATUS', 'WorkflowStatus']),
                'registration_status': _child_text(elem, ['REGISTRATIONSTATUS', 'RegistrationStatus']),
                'origin': _child_text(elem, ['ORIGIN', 'Origin']),
                'rai': _child_text(elem, ['RAI']),
            }
        )

        # -------------------- DEC / ISO-11179 --------------------
        dec = elem.find('DATAELEMENTCONCEPT')
        if dec is not None:
            cd = dec.find('ConceptualDomain')
            oc = dec.find('ObjectClass')
            prop = dec.find('Property')

            writers['cde_iso11179_dec'].append(
                {
                    'cde_publicid': cde_publicid,
                    'cde_version': cde_version,
                    'source_file': source_file,
                    'cde_xml_row_id': cde_xml_row_id,
                    'dec_publicid': _child_text(dec, ['PublicId', 'PUBLICID']),
                    'dec_version': _child_text(dec, ['Version', 'VERSION']),
                    'dec_preferred_name': _child_text(dec, ['PreferredName', 'PREFERREDNAME']),
                    'dec_long_name': _child_text(dec, ['LongName', 'LONGNAME']),
                    'dec_preferred_definition': _child_text(dec, ['PreferredDefinition', 'PREFERREDDEFINITION']),
                    'dec_context_name': _child_text(dec, ['ContextName', 'CONTEXTNAME']),
                    'dec_context_version': _child_text(dec, ['ContextVersion', 'CONTEXTVERSION']),
                    'dec_workflow_status': _child_text(dec, ['WorkflowStatus', 'WORKFLOWSTATUS']),
                    'dec_origin': _child_text(dec, ['Origin', 'ORIGIN']),
                    # DEC conceptual domain
                    'dec_conceptual_domain_publicid': _child_text(cd, ['PublicId']) if cd is not None else None,
                    'dec_conceptual_domain_version': _child_text(cd, ['Version']) if cd is not None else None,
                    'dec_conceptual_domain_preferred_name': _child_text(cd, ['PreferredName']) if cd is not None else None,
                    'dec_conceptual_domain_long_name': _child_text(cd, ['LongName']) if cd is not None else None,
                    'dec_conceptual_domain_context_name': _child_text(cd, ['ContextName']) if cd is not None else None,
                    'dec_conceptual_domain_context_version': _child_text(cd, ['ContextVersion']) if cd is not None else None,
                    # Object class
                    'object_class_publicid': _child_text(oc, ['PublicId']) if oc is not None else None,
                    'object_class_version': _child_text(oc, ['Version']) if oc is not None else None,
                    'object_class_preferred_name': _child_text(oc, ['PreferredName']) if oc is not None else None,
                    'object_class_long_name': _child_text(oc, ['LongName']) if oc is not None else None,
                    'object_class_context_name': _child_text(oc, ['ContextName']) if oc is not None else None,
                    'object_class_context_version': _child_text(oc, ['ContextVersion']) if oc is not None else None,
                    'object_class_concept_details_json': _collect_concept_details_json(oc),
                    # Property
                    'property_publicid': _child_text(prop, ['PublicId']) if prop is not None else None,
                    'property_version': _child_text(prop, ['Version']) if prop is not None else None,
                    'property_preferred_name': _child_text(prop, ['PreferredName']) if prop is not None else None,
                    'property_long_name': _child_text(prop, ['LongName']) if prop is not None else None,
                    'property_context_name': _child_text(prop, ['ContextName']) if prop is not None else None,
                    'property_context_version': _child_text(prop, ['ContextVersion']) if prop is not None else None,
                    'property_concept_details_json': _collect_concept_details_json(prop),
                }
            )

        # -------------------- Value Domain / ISO-11179 --------------------
        vd = elem.find('VALUEDOMAIN')
        if vd is not None:
            vd_cd = vd.find('ConceptualDomain')
            rep = vd.find('Representation')

            writers['cde_iso11179_value_domain'].append(
                {
                    'cde_publicid': cde_publicid,
                    'cde_version': cde_version,
                    'source_file': source_file,
                    'cde_xml_row_id': cde_xml_row_id,
                    'vd_publicid': _child_text(vd, ['PublicId', 'PUBLICID']),
                    'vd_version': _child_text(vd, ['Version', 'VERSION']),
                    'vd_preferred_name': _child_text(vd, ['PreferredName', 'PREFERREDNAME']),
                    'vd_long_name': _child_text(vd, ['LongName', 'LONGNAME']),
                    'vd_preferred_definition': _child_text(vd, ['PreferredDefinition', 'PREFERREDDEFINITION']),
                    'vd_workflow_status': _child_text(vd, ['WorkflowStatus', 'WORKFLOWSTATUS']),
                    'vd_context_name': _child_text(vd, ['ContextName', 'CONTEXTNAME']),
                    'vd_context_version': _child_text(vd, ['ContextVersion', 'CONTEXTVERSION']),
                    'vd_origin': _child_text(vd, ['Origin', 'ORIGIN']),
                    # VD conceptual domain
                    'vd_conceptual_domain_publicid': _child_text(vd_cd, ['PublicId']) if vd_cd is not None else None,
                    'vd_conceptual_domain_version': _child_text(vd_cd, ['Version']) if vd_cd is not None else None,
                    'vd_conceptual_domain_preferred_name': _child_text(vd_cd, ['PreferredName']) if vd_cd is not None else None,
                    'vd_conceptual_domain_long_name': _child_text(vd_cd, ['LongName']) if vd_cd is not None else None,
                    'vd_conceptual_domain_context_name': _child_text(vd_cd, ['ContextName']) if vd_cd is not None else None,
                    'vd_conceptual_domain_context_version': _child_text(vd_cd, ['ContextVersion']) if vd_cd is not None else None,
                    # VD typing
                    'value_domain_type': _child_text(vd, ['ValueDomainType']),
                    'datatype': _child_text(vd, ['Datatype']),
                    'maximum_length': _child_text(vd, ['MaximumLength']),
                    'minimum_length': _child_text(vd, ['MinimumLength']),
                    'decimal_place': _child_text(vd, ['DecimalPlace']),
                    'maximum_value': _child_text(vd, ['MaximumValue']),
                    'minimum_value': _child_text(vd, ['MinimumValue']),
                    # Representation
                    'representation_publicid': _child_text(rep, ['PublicId']) if rep is not None else None,
                    'representation_version': _child_text(rep, ['Version']) if rep is not None else None,
                    'representation_preferred_name': _child_text(rep, ['PreferredName']) if rep is not None else None,
                    'representation_long_name': _child_text(rep, ['LongName']) if rep is not None else None,
                    'representation_context_name': _child_text(rep, ['ContextName']) if rep is not None else None,
                    'representation_context_version': _child_text(rep, ['ContextVersion']) if rep is not None else None,
                    'representation_concept_details_json': _collect_concept_details_json(rep),
                }
            )

            # Permissible values
            pv_container = vd.find('PermissibleValues')
            if pv_container is not None:
                for pv in pv_container.findall('PermissibleValues_ITEM'):
                    writers['cde_permissible_values'].append(
                        {
                            'cde_publicid': cde_publicid,
                            'cde_version': cde_version,
                            'source_file': source_file,
                            'cde_xml_row_id': cde_xml_row_id,
                            'valid_value': _child_text(pv, ['VALIDVALUE']),
                            'value_meaning': _child_text(pv, ['VALUEMEANING']),
                            'meaning_description': _child_text(pv, ['MEANINGDESCRIPTION']),
                            'meaning_concepts': _child_text(pv, ['MEANINGCONCEPTS']),
                            'meaning_concept_origin': _child_text(pv, ['MEANINGCONCEPTORIGIN']),
                            'meaning_concept_display_order': _child_text(pv, ['MEANINGCONCEPTDISPLAYORDER']),
                            'pv_begin_date': _child_text(pv, ['PVBEGINDATE']),
                            'pv_end_date': _child_text(pv, ['PVENDDATE']),
                            'vm_publicid': _child_text(pv, ['VMPUBLICID']),
                            'vm_version': _child_text(pv, ['VMVERSION']),
                        }
                    )

        # -------------------- Classifications --------------------
        clist = elem.find('CLASSIFICATIONSLIST')
        if clist is not None:
            for ci in clist.findall('CLASSIFICATIONSLIST_ITEM'):
                scheme = ci.find('ClassificationScheme')
                writers['cde_classifications'].append(
                    {
                        'cde_publicid': cde_publicid,
                        'cde_version': cde_version,
                        'source_file': source_file,
                        'cde_xml_row_id': cde_xml_row_id,
                        'classification_scheme_publicid': _child_text(scheme, ['PublicId']) if scheme is not None else None,
                        'classification_scheme_version': _child_text(scheme, ['Version']) if scheme is not None else None,
                        'classification_scheme_preferred_name': _child_text(scheme, ['PreferredName']) if scheme is not None else None,
                        'classification_scheme_context_name': _child_text(scheme, ['ContextName']) if scheme is not None else None,
                        'classification_scheme_context_version': _child_text(scheme, ['ContextVersion']) if scheme is not None else None,
                        'classification_scheme_item_name': _child_text(ci, ['ClassificationSchemeItemName']),
                        'classification_scheme_item_type': _child_text(ci, ['ClassificationSchemeItemType']),
                        'csi_publicid': _child_text(ci, ['CsiPublicId']),
                        'csi_version': _child_text(ci, ['CsiVersion']),
                    }
                )

        # -------------------- Reference Documents --------------------
        rlist = elem.find('REFERENCEDOCUMENTSLIST')
        if rlist is not None:
            for rd in rlist.findall('REFERENCEDOCUMENTSLIST_ITEM'):
                writers['cde_reference_documents'].append(
                    {
                        'cde_publicid': cde_publicid,
                        'cde_version': cde_version,
                        'source_file': source_file,
                        'cde_xml_row_id': cde_xml_row_id,
                        'name': _child_text(rd, ['Name']),
                        'document_type': _child_text(rd, ['DocumentType']),
                        'document_text': _child_text(rd, ['DocumentText']),
                        'url': _child_text(rd, ['URL']),
                        'language': _child_text(rd, ['Language']),
                        'display_order': _child_text(rd, ['DisplayOrder']),
                    }
                )

        # -------------------- Alternate Names --------------------
        alist = elem.find('ALTERNATENAMELIST')
        if alist is not None:
            for an in alist.findall('ALTERNATENAMELIST_ITEM'):
                writers['cde_alternate_names'].append(
                    {
                        'cde_publicid': cde_publicid,
                        'cde_version': cde_version,
                        'source_file': source_file,
                        'cde_xml_row_id': cde_xml_row_id,
                        'context_name': _child_text(an, ['ContextName']),
                        'context_version': _child_text(an, ['ContextVersion']),
                        'alternate_name': _child_text(an, ['AlternateName']),
                        'alternate_name_type': _child_text(an, ['AlternateNameType']),
                        'language': _child_text(an, ['Language']),
                    }
                )

        processed += 1
        if processed % 1000 == 0:
            print(f'[{xml_path.name}] Processed {processed:,} DataElements...')

        # Free memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    # Close writers
    for w in writers.values():
        w.close()

    print(f'[{xml_path.name}] Done. DataElements processed: {processed:,}. Output: {out_dir}')



def _iter_xml_files_in_dir(dir_path: Path) -> List[Path]:
    # List .xml files directly under dir_path (non-recursive).
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == '.xml'])


def _choose_zip_in_dir(dir_path: Path) -> Optional[Path]:
    # Prefer the canonical release filename, otherwise use a single zip if unambiguous.
    preferred = dir_path / 'releasedCDEsXML-OD.zip'
    if preferred.exists() and preferred.is_file():
        return preferred
    zips = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == '.zip'])
    if len(zips) == 1:
        return zips[0]
    return None


def _maybe_extract_zip(zip_path: Path, out_root: Path) -> Path:
    # Extract into a stable directory under out_root so runs are reproducible/auditable.
    dest = (out_root / '_unzipped_xml' / zip_path.stem).resolve()
    if dest.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)
    return dest


def _resolve_xml_inputs(input_path: Path, out_root: Path) -> List[Path]:
    input_path = input_path.resolve()

    if input_path.is_file():
        if input_path.suffix.lower() == '.zip':
            extracted = _maybe_extract_zip(input_path, out_root=out_root)
            xmls = sorted([p for p in extracted.rglob('*.xml') if p.is_file()])
            return xmls
        return [input_path]

    if input_path.is_dir():
        xmls = _iter_xml_files_in_dir(input_path)
        if xmls:
            return xmls

        z = _choose_zip_in_dir(input_path)
        if z is None:
            return []
        extracted = _maybe_extract_zip(z, out_root=out_root)
        return sorted([p for p in extracted.rglob('*.xml') if p.is_file()])

    raise FileNotFoundError(f'Input path not found: {input_path}')


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--input',
        required=True,
        help=(
            'Path to one XML file, a directory of XML files, a .zip file, '
            'or a directory containing a .zip (e.g., releasedCDEsXML-OD.zip).'
        ),
    )
    ap.add_argument('--out-root', required=True, help='Output root directory (per-file subdirs will be created).')
    ap.add_argument('--chunk-size', type=int, default=50_000, help='Records buffered per table before flushing to Parquet.')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing per-file parquet outputs.')

    args = ap.parse_args(argv)

    # Defer the dependency check until after argparse has a chance to handle
    # `--help`. This allows `demap extract-cadsr-xml --help` to work even in
    # environments without pyarrow.
    if pa is None or pq is None:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'pyarrow'. This script writes Parquet and requires it.\n"
            "Install with: pip install pyarrow\n"
        )

    input_path = Path(args.input)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    xml_files = _resolve_xml_inputs(input_path, out_root=out_root)
    if not xml_files:
        raise SystemExit(
            f"No .xml files found at: {input_path}\n"
            "Provide either:\n"
            "  - the 14 caDSR XML files directly, or\n"
            "  - a zip file containing them (e.g., releasedCDEsXML-OD.zip)."
        )

    print(f'Found {len(xml_files)} XML file(s). Writing per-file outputs under: {out_root}')
    for pth in xml_files:
        _extract_one_file(pth, out_root=out_root, chunk_size=args.chunk_size, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
