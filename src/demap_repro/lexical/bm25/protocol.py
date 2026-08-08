"""Canonical BM25 protocol: load, validate, and enforce fail-closed.

Every deviation raises ``Bm25ProtocolError``. This module never runs BM25 itself; it is the
single gate the canonical CLI validates against.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

DEFAULT_PROTOCOL = "configs/paper/bm25_protocol_v1.yaml"

EXPECTED_QUERIES: List[Tuple[str, str]] = [
    ("Q1", "RAW"), ("Q3", "RAW_PV"), ("Q4", "RAW_PV_PH"), ("Q2", "PREF_PV")]
EXPECTED_RECIPES: List[Tuple[int, str, str]] = [
    (1, "PQT", "v3"), (2, "LN_DEF", "v2"), (3, "SN_PQT", "v1_v3"),
    (4, "SN_LN_DEF_PQT", "v1_v2a_v2b_v3"), (5, "SN_LN_DEF_PV", "v1_v2a_v2b_v5"),
    (6, "SN_LN_PQT_PV", "v1_v2a_v3_v5"), (7, "SN_LN_DEF_PQT_PV", "v1_v2a_v2b_v3_v5"),
    (8, "SN_DEC_DEF_PQT_PV", "v1_v6_v2b_v3_v5"),
    (9, "SN_LN_DEF_PQT_VD_PV", "v1_v2a_v2b_v3_v4_v5"),
    (10, "SN_LN_DEC_DEF_PQT_PV", "v1_v2a_v6_v2b_v3_v5"),
]


class Bm25ProtocolError(SystemExit):
    """Fail-closed protocol violation."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_protocol(path: str = DEFAULT_PROTOCOL) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    if not p.exists():
        raise Bm25ProtocolError(f"[bm25-protocol] not found: {path}")
    proto = yaml.safe_load(p.read_text())
    if not isinstance(proto, dict) or proto.get("schema_version") != 1:
        raise Bm25ProtocolError(f"[bm25-protocol] {path}: expected a dict with schema_version: 1")
    validate_protocol(proto)
    return proto


def validate_protocol(proto: Dict[str, Any]) -> None:
    """Structural locks: grid identity/order, selection policy, metric policy, parameters."""
    q = proto.get("query_representations") or []
    c = proto.get("cde_representations") or []
    if [(x["id"], x["name"]) for x in q] != EXPECTED_QUERIES:
        raise Bm25ProtocolError(f"[bm25-protocol] query grid mismatch: {[(x['id'], x['name']) for x in q]}")
    if [(x["order"], x["name"], x["recipe"]) for x in c] != EXPECTED_RECIPES:
        raise Bm25ProtocolError("[bm25-protocol] CDE recipe grid order/identity mismatch")
    n_declared = int(proto.get("n_cells", 0))
    if n_declared != 40 or len(q) * len(c) != 40:
        raise Bm25ProtocolError(
            f"[bm25-protocol] grid must be 4 x 10 = 40 cells; declared n_cells={n_declared}, "
            f"actual {len(q)} x {len(c)} = {len(q) * len(c)}")

    sel = proto.get("selection") or {}
    if sel.get("split") != "val_dev":
        raise Bm25ProtocolError(
            f"[bm25-protocol] selection split must be val_dev, got {sel.get('split')!r}")
    forbidden = set(sel.get("forbidden_selection_splits") or [])
    if "test" not in forbidden:
        raise Bm25ProtocolError("[bm25-protocol] 'test' must be a forbidden selection split")
    if sel.get("tie_margin") != "none":
        raise Bm25ProtocolError("[bm25-protocol] tie_margin must be 'none'")
    if set(sel.get("freezable_query_variants") or []) - {"Q1", "Q2", "Q3", "Q4"}:
        raise Bm25ProtocolError("[bm25-protocol] unknown freezable query variant")

    bm = proto.get("bm25") or {}
    if float(bm.get("k1", -1)) != 1.2 or float(bm.get("b", -1)) != 0.75:
        raise Bm25ProtocolError(f"[bm25-protocol] k1/b locked at 1.2/0.75, got {bm.get('k1')}/{bm.get('b')}")
    if bm.get("parameters_tuned") is not False:
        raise Bm25ProtocolError("[bm25-protocol] parameters_tuned must be false (no new tuning)")

    ev = proto.get("evaluation") or {}
    if ev.get("denominator") != "query_level":
        raise Bm25ProtocolError("[bm25-protocol] evaluation denominator must be query_level")
    if ev.get("gold_matching") != "public_id_any_gold":
        raise Bm25ProtocolError("[bm25-protocol] gold matching must be public_id_any_gold")
    if not ev.get("forbid_pair_level") or not ev.get("forbid_version_exact"):
        raise Bm25ProtocolError("[bm25-protocol] pair-level / version-exact evaluation must be forbidden")
    if ev.get("dedup") != "public_id_best_score":
        raise Bm25ProtocolError("[bm25-protocol] dedup policy must be public_id_best_score")
    if int(ev.get("ranking_depth", 0)) != 200:
        raise Bm25ProtocolError("[bm25-protocol] ranking_depth must be 200")
    if not ev.get("deterministic"):
        raise Bm25ProtocolError("[bm25-protocol] evaluation must be declared deterministic")
    names = [d["name"] for d in ev.get("datasets") or []]
    if names != ["test", "cctg", "oid_alt", "cdash", "gdc_combined", "cimac_v2"]:
        raise Bm25ProtocolError(f"[bm25-protocol] canonical dataset set/order mismatch: {names}")
    banned = {d["name"] for d in ev.get("forbidden_datasets") or []}
    if "external_holdout_standard" not in banned:
        raise Bm25ProtocolError(
            "[bm25-protocol] the leakage-excluded external_holdout_standard must be forbidden")
    for d in ev.get("datasets") or []:
        if d["name"] in banned:
            raise Bm25ProtocolError(f"[bm25-protocol] dataset {d['name']} is both canonical and forbidden")


def assert_selection_split_allowed(split: str, proto: Dict[str, Any]) -> None:
    """Reject any attempt to select a representation on Test or another reporting set."""
    sel = proto["selection"]
    if split in set(sel.get("forbidden_selection_splits") or []):
        raise Bm25ProtocolError(
            f"[bm25-protocol] refusing to select on {split!r}. Representation selection is "
            f"val_dev-only; Test and the holdout sets may never influence selection.")
    if split != sel["split"]:
        raise Bm25ProtocolError(
            f"[bm25-protocol] selection split must be {sel['split']!r}, got {split!r}")


def verify_catalog(proto: Dict[str, Any], *, check_rows: bool = True) -> Dict[str, Any]:
    """Catalog identity: path, SHA256, row/unique-id counts, retired, admin contexts."""
    import pandas as pd

    cat = proto["catalog"]
    _norm = lambda s: str(s).replace("\\", "/").lstrip("./")  # noqa: E731
    for bad in proto.get("forbidden_catalogs") or []:
        if _norm(cat["path"]).endswith(_norm(bad["path"])):
            raise Bm25ProtocolError(
                f"[bm25-protocol] catalog {cat['path']} is explicitly forbidden: {bad['reason']}")
    p = repo_root() / cat["path"]
    if not p.exists():
        raise Bm25ProtocolError(f"[bm25-protocol] catalog not found: {cat['path']}")
    got = sha256_file(p)
    if got != cat["sha256"]:
        raise Bm25ProtocolError(
            f"[bm25-protocol] catalog sha256 mismatch\n  got      {got}\n  expected {cat['sha256']}")
    out: Dict[str, Any] = {"path": cat["path"], "sha256": got}
    if check_rows:
        m = pd.read_parquet(p, columns=["cde_publicid", "workflow_status", "context_name"])
        if len(m) != int(cat["rows"]):
            raise Bm25ProtocolError(f"[bm25-protocol] catalog rows {len(m)} != {cat['rows']}")
        n_pub = int(m["cde_publicid"].nunique())
        if n_pub != int(cat["unique_public_ids"]):
            raise Bm25ProtocolError(
                f"[bm25-protocol] unique public ids {n_pub} != {cat['unique_public_ids']}")
        n_ret = int(m["workflow_status"].astype(str).str.upper().str.contains("RETIRED").sum())
        if n_ret != int(cat["retired_cdes"]):
            raise Bm25ProtocolError(f"[bm25-protocol] retired CDEs present: {n_ret}")
        n_adm = int(m["context_name"].astype(str).isin(cat["excluded_admin_contexts"]).sum())
        if n_adm != int(cat["excluded_admin_context_rows"]):
            raise Bm25ProtocolError(f"[bm25-protocol] administrative-context rows present: {n_adm}")
        out.update(rows=len(m), unique_public_ids=n_pub, retired=n_ret, admin_context_rows=n_adm)
    return out


def verify_splits(proto: Dict[str, Any], *, check_rows: bool = True) -> Dict[str, Any]:
    """Selection split and all six canonical datasets: hash + query-level denominator."""
    import pandas as pd

    out: Dict[str, Any] = {}
    sel = proto["selection"]
    got = sha256_file(sel["split_path"])
    if got != sel["split_sha256"]:
        raise Bm25ProtocolError(
            f"[bm25-protocol] {sel['split']} sha256 mismatch\n  got      {got}\n"
            f"  expected {sel['split_sha256']}")
    out[sel["split"]] = {"sha256": got}
    if check_rows:
        n = int(pd.read_parquet(repo_root() / sel["split_path"],
                                columns=["query_id"])["query_id"].nunique())
        if n != int(sel["n_queries"]):
            raise Bm25ProtocolError(
                f"[bm25-protocol] {sel['split']} query-level n {n} != {sel['n_queries']}")
        out[sel["split"]]["n_queries"] = n

    for d in proto["evaluation"]["datasets"]:
        got = sha256_file(d["path"])
        if got != d["sha256"]:
            raise Bm25ProtocolError(
                f"[bm25-protocol] {d['name']} sha256 mismatch\n  got      {got}\n"
                f"  expected {d['sha256']}")
        rec = {"sha256": got}
        if check_rows:
            n = int(pd.read_parquet(repo_root() / d["path"],
                                    columns=["query_id"])["query_id"].nunique())
            if n != int(d["n_queries"]):
                raise Bm25ProtocolError(
                    f"[bm25-protocol] {d['name']} query-level n {n} != {d['n_queries']}")
            rec["n_queries"] = n
        out[d["name"]] = rec
    return out


def grid_cells(proto: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    """The exact 40 (query_id, query_column, recipe, cde_name) cells, in displayed order."""
    cells = [(q["id"], q["column"], c["recipe"], c["name"])
             for c in proto["cde_representations"] for q in proto["query_representations"]]
    if len(cells) != 40 or len(set((a, c) for a, _, c, _ in cells)) != 40:
        raise Bm25ProtocolError(f"[bm25-protocol] grid must contain 40 unique cells, got {len(cells)}")
    return cells
