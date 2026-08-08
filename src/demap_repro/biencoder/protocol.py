"""Canonical paper bi-encoder protocol: load, validate, and derive the Phase 0 grid.

Fail-closed: any structural, ordering, fingerprint, split, or catalog deviation raises
``ProtocolError``. This is the single source of truth the canonical generator validates
against; it never launches anything itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


class ProtocolError(SystemExit):
    """Raised (fail-closed) when the protocol or on-disk data violate the locks."""


def load_protocol(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ProtocolError(f"[protocol] not found: {path}")
    with open(p) as f:
        proto = yaml.safe_load(f)
    if not isinstance(proto, dict) or proto.get("schema_version") != 1:
        raise ProtocolError(f"[protocol] {path}: expected a dict with schema_version: 1")
    validate_protocol(proto)
    return proto


def validate_protocol(proto: Dict[str, Any]) -> None:
    q = proto.get("query_representations") or []
    c = proto.get("cde_representations") or []
    if len(q) != 4:
        raise ProtocolError(f"[protocol] expected exactly 4 query representations, got {len(q)}")
    if len(c) != 10:
        raise ProtocolError(f"[protocol] expected exactly 10 CDE representations, got {len(c)}")

    # Exact displayed order for queries and recipes.
    exp_q = [("Q1", "RAW"), ("Q3", "RAW_PV"), ("Q4", "RAW_PV_PH"), ("Q2", "PREF_PV")]
    got_q = [(x["id"], x["name"]) for x in q]
    if got_q != exp_q:
        raise ProtocolError(f"[protocol] query order/identity mismatch: {got_q} != {exp_q}")

    exp_c = [
        (1, "PQT", "v3"), (2, "LN_DEF", "v2"), (3, "SN_PQT", "v1_v3"),
        (4, "SN_LN_DEF_PQT", "v1_v2a_v2b_v3"), (5, "SN_LN_DEF_PV", "v1_v2a_v2b_v5"),
        (6, "SN_LN_PQT_PV", "v1_v2a_v3_v5"), (7, "SN_LN_DEF_PQT_PV", "v1_v2a_v2b_v3_v5"),
        (8, "SN_DEC_DEF_PQT_PV", "v1_v6_v2b_v3_v5"), (9, "SN_LN_DEF_PQT_VD_PV", "v1_v2a_v2b_v3_v4_v5"),
        (10, "SN_LN_DEC_DEF_PQT_PV", "v1_v2a_v6_v2b_v3_v5"),
    ]
    got_c = [(x["order"], x["name"], x["recipe"]) for x in c]
    if got_c != exp_c:
        raise ProtocolError(f"[protocol] CDE recipe order/identity mismatch:\n got {got_c}\n exp {exp_c}")

    # Anchor must be present in the grid.
    a = proto.get("anchor") or {}
    if a.get("query_id") != "Q3" or a.get("recipe") != "v1_v6_v2b_v3_v5":
        raise ProtocolError(f"[protocol] anchor must be Q3 x v1_v6_v2b_v3_v5, got {a}")
    if a.get("recipe") not in {x["recipe"] for x in c} or a.get("query_id") not in {x["id"] for x in q}:
        raise ProtocolError("[protocol] anchor query/recipe not found in the grid")

    d = proto.get("data") or {}
    if d.get("val_dev_denominator") != 4234:
        raise ProtocolError(f"[protocol] val_dev_denominator must be 4234, got {d.get('val_dev_denominator')}")
    if d.get("production_catalog_rows") != 62976:
        raise ProtocolError(f"[protocol] production_catalog_rows must be 62976, got {d.get('production_catalog_rows')}")
    if d.get("max_seq_length") != 256:
        raise ProtocolError(f"[protocol] max_seq_length must be 256, got {d.get('max_seq_length')}")
    if d.get("cde_format") != "labeled":
        raise ProtocolError(f"[protocol] cde_format must be labeled, got {d.get('cde_format')}")
    if d.get("tuning_split") != "val_dev" or "test" in (proto.get("phase1", {}).get("eval_splits") or []):
        raise ProtocolError("[protocol] tuning split must be val_dev only; test must be excluded")

    p0 = proto.get("phase0") or {}
    if (p0.get("n_query_representations"), p0.get("n_cde_representations"), p0.get("n_cells_per_model")) != (4, 10, 40):
        raise ProtocolError("[protocol] phase0 must declare 4 x 10 = 40 cells per model")


def phase0_cells(proto: Dict[str, Any]) -> List[Tuple[str, str, str, str, str]]:
    """Return the ordered 40-cell grid PER MODEL as (model, query_id, query_name, recipe, cde_name).

    Ordering is query-major then recipe-order, matching the displayed grid.
    """
    q = proto["query_representations"]
    c = proto["cde_representations"]
    cells = []
    for qi in q:
        for ci in sorted(c, key=lambda x: x["order"]):
            cells.append((None, qi["id"], qi["name"], ci["recipe"], ci["name"]))
    if len(cells) != 40:
        raise ProtocolError(f"[protocol] phase0 grid built {len(cells)} cells, expected 40")
    keys = [(qid, recipe) for (_m, qid, _qn, recipe, _cn) in cells]
    if len(set(keys)) != 40:
        raise ProtocolError("[protocol] phase0 grid has duplicate (query, recipe) cells")
    return cells


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_fingerprints(proto: Dict[str, Any], *, check_rows: bool = True) -> Dict[str, Any]:
    """Verify on-disk split + catalog SHA256 and row counts match the protocol. Fail-closed."""
    d = proto["data"]
    fp = d["fingerprints"]
    checked = {}
    targets = {
        "train": (f"{d['splits_dir']}/train.parquet", fp["train_sha256"], None),
        "val_dev": (f"{d['splits_dir']}/val_dev.parquet", fp["val_dev_sha256"], d["val_dev_denominator"]),
        "catalog": (d["production_catalog"], fp["production_catalog_sha256"], d["production_catalog_rows"]),
    }
    for name, (path, exp_sha, exp_rows) in targets.items():
        if not Path(path).exists():
            raise ProtocolError(f"[protocol] fingerprint target missing: {path}")
        got = sha256_file(path)
        if got != exp_sha:
            raise ProtocolError(f"[protocol] {name} sha256 mismatch: {got} != {exp_sha} ({path})")
        entry = {"path": path, "sha256": got}
        if check_rows and exp_rows is not None:
            import pandas as pd  # local import; heavy

            n = len(pd.read_parquet(path))
            if int(n) != int(exp_rows):
                raise ProtocolError(f"[protocol] {name} row count mismatch: {n} != {exp_rows} ({path})")
            entry["rows"] = int(n)
        checked[name] = entry
    return checked
