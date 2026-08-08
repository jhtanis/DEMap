"""Canonical BM25 execution and validation.

Protocol-driven: every lock in configs/paper/bm25_protocol_v1.yaml is enforced before any
scoring happens, and the saved outputs are re-validated afterwards.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from demap_repro.lexical.bm25.engine import _rank_order, build_bm25_model
from demap_repro.lexical.bm25.quarantine import assert_canonical_catalog
from demap_repro.lexical.bm25.protocol import (
    Bm25ProtocolError,
    assert_selection_split_allowed,
    grid_cells,
    repo_root,
    sha256_file,
    verify_catalog,
    verify_splits,
)
from demap_repro.text.recipes import build_catalog

REPO = repo_root()


def _metric_helper():
    """The project's canonical metric helper — identical definitions to the neural methods."""
    sys.path.insert(0, str(REPO / "scripts"))
    import evaluate_non_exact_subset as N  # noqa: E402

    return N


def _display_path(p: Path) -> str:
    """Repo-relative when inside the checkout, absolute otherwise (e.g. a scratch out-root)."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def load_queries(path: Path, query_col: str) -> Tuple[List[str], List[str]]:
    df = pd.read_parquet(path, columns=["query_id", query_col])
    df["query_id"] = df["query_id"].astype(str)
    df = df.drop_duplicates("query_id")
    return df["query_id"].tolist(), df[query_col].fillna("").astype(str).tolist()


def build_index(master: pd.DataFrame, proto: Dict[str, Any], recipe: str):
    bm = proto["bm25"]
    cat = build_catalog(master, recipe=recipe, cde_format=bm["cde_format"], sep=bm["sep"],
                        recipe_configs=bm["recipe_configs"])
    model = build_bm25_model(
        catalog_ids=cat["cde_id"].astype(str).tolist(),
        catalog_texts=cat["cde_text"].astype(str).tolist(),
        tokenizer_name=bm["tokenizer"], min_token_len=int(bm["min_token_len"]),
        k1=float(bm["k1"]), b=float(bm["b"]), query_binary=bool(bm["query_binary"]))
    model["_pub"] = np.array([str(c).split("::")[0] for c in model["catalog_ids"]], dtype=object)
    return model


def rank(model, query_ids: List[str], query_texts: List[str], *, depth: int,
         batch_size: int = 256) -> pd.DataFrame:
    """Score, then deduplicate by CDE public id (best score wins) with compacted ranks."""
    vec, x, pub = model["vectorizer"], model["x_bm25"], model["_pub"]
    ids = model["catalog_ids"]
    rows = []
    for start in range(0, len(query_ids), batch_size):
        stop = min(len(query_ids), start + batch_size)
        q = vec.transform(query_texts[start:stop])
        if model["query_binary"] and q.nnz:
            q.data[:] = 1.0
        sm = (q @ x.T).tocsr()
        sm.sort_indices()
        for loc in range(stop - start):
            qid = query_ids[start + loc]
            a, b = sm.indptr[loc], sm.indptr[loc + 1]
            idx = sm.indices[a:b].astype(np.int32, copy=False)
            dat = sm.data[a:b].astype(np.float32, copy=False)
            if idx.size == 0:
                continue
            # _rank_order returns POSITIONS into idx/dat, not catalog document indices;
            # map through idx/dat before indexing the catalog arrays.
            order = _rank_order(idx, dat)
            idx_sorted, dat_sorted = idx[order], dat[order]
            seen, r = set(), 0
            for j, sc in zip(idx_sorted, dat_sorted):
                p = str(pub[j])
                if p in seen:
                    continue
                seen.add(p)
                r += 1
                rows.append((qid, str(ids[j]), p, r, float(sc)))
                if r >= depth:
                    break
    return pd.DataFrame(rows, columns=["query_id", "cde_id", "pub", "rank", "score"])


def evaluate(ranked: pd.DataFrame, eval_path: Path) -> Dict[str, float]:
    N = _metric_helper()
    gm = N.build_gold_map(eval_path)
    return N.eval_method(ranked[["query_id", "pub", "rank"]], set(gm), gm)


# ------------------------------------------------------------------ validation of an output root
def validate_outputs(proto: Dict[str, Any], out_root: Path, *, recompute: bool = True) -> Dict[str, Any]:
    """Validate an existing canonical BM25 artifact root against the protocol."""
    if not out_root.exists():
        raise Bm25ProtocolError(f"[bm25] output root not found: {out_root}")
    prov_p, met_p = out_root / "PROVENANCE.json", out_root / "bm25_canonical_metrics.csv"
    for p in (prov_p, met_p, out_root / "bm25_val_dev_selection.csv"):
        if not p.exists():
            raise Bm25ProtocolError(f"[bm25] missing expected artifact: {p}")
    prov = json.loads(prov_p.read_text())
    cat, sel, ev = proto["catalog"], proto["selection"], proto["evaluation"]

    if prov["catalog"]["sha256"] != cat["sha256"] or int(prov["catalog"]["rows"]) != int(cat["rows"]):
        raise Bm25ProtocolError("[bm25] provenance catalog identity does not match the protocol")
    if prov["selection"]["split"] != sel["split"]:
        raise Bm25ProtocolError("[bm25] provenance selection split does not match the protocol")
    if prov["selection"].get("test_used_for_selection") is not False:
        raise Bm25ProtocolError("[bm25] provenance does not assert test_used_for_selection: false")
    exp = sel["expected_winner"]
    if (prov["selection"]["selected_query_variant"] != exp["query_variant"]
            or prov["selection"]["selected_recipe"] != exp["recipe"]):
        raise Bm25ProtocolError(
            f"[bm25] frozen winner {prov['selection']['selected_query_variant']} x "
            f"{prov['selection']['selected_recipe']} != protocol "
            f"{exp['query_variant']} x {exp['recipe']}")
    if float(prov["bm25"]["k1"]) != float(proto["bm25"]["k1"]) or \
       float(prov["bm25"]["b"]) != float(proto["bm25"]["b"]):
        raise Bm25ProtocolError("[bm25] provenance k1/b do not match the protocol")

    sel_df = pd.read_csv(out_root / "bm25_val_dev_selection.csv")
    if len(sel_df) != int(proto["n_cells"]):
        raise Bm25ProtocolError(
            f"[bm25] selection sweep has {len(sel_df)} cells, expected {proto['n_cells']}")
    have = {(r.query_variant, r.recipe) for r in sel_df.itertuples()}
    want = {(qid, recipe) for qid, _, recipe, _ in grid_cells(proto)}
    if have != want:
        raise Bm25ProtocolError(f"[bm25] selection grid mismatch; missing={want - have}")

    met = pd.read_csv(met_p).set_index("dataset")
    report: Dict[str, Any] = {"datasets": {}}
    N = _metric_helper()
    for d in ev["datasets"]:
        name = d["name"]
        if name not in met.index:
            raise Bm25ProtocolError(f"[bm25] metrics missing dataset {name}")
        if int(met.loc[name, "n_queries"]) != int(d["n_queries"]):
            raise Bm25ProtocolError(
                f"[bm25] {name} denominator {met.loc[name, 'n_queries']} != {d['n_queries']}")
        rk = out_root / "rankings" / f"bm25_rankings_{name}.parquet"
        if not rk.exists():
            raise Bm25ProtocolError(f"[bm25] missing rankings for {name}")
        r = pd.read_parquet(rk)
        if r.duplicated(["query_id", "pub"]).any():
            raise Bm25ProtocolError(f"[bm25] {name}: duplicate public id inside a query ranking")
        if int(r["rank"].max()) > int(ev["ranking_depth"]):
            raise Bm25ProtocolError(f"[bm25] {name}: ranking deeper than {ev['ranking_depth']}")
        rec: Dict[str, Any] = {"n_queries": int(d["n_queries"])}
        if recompute:
            gm = N.build_gold_map(REPO / d["path"])
            m = N.eval_method(r[["query_id", "pub", "rank"]], set(gm), gm)
            for k in ["recall@1", "recall@5", "recall@10", "mrr@100"]:
                if abs(round(m[k], 4) - float(met.loc[name, k])) > 1e-9:
                    raise Bm25ProtocolError(
                        f"[bm25] {name} {k} does not recompute: {round(m[k], 4)} != {met.loc[name, k]}")
                rec[k] = round(m[k], 6)
        report["datasets"][name] = rec
    report["frozen"] = {"query_variant": prov["selection"]["selected_query_variant"],
                        "recipe": prov["selection"]["selected_recipe"]}
    return report


# ------------------------------------------------------------------ full canonical execution
def execute(proto: Dict[str, Any], out_root: Path, *, dry_run: bool = False,
            check_rows: bool = True) -> Dict[str, Any]:
    cat_info = verify_catalog(proto, check_rows=check_rows)
    assert_canonical_catalog(proto["catalog"]["path"], verify_sha256=False)  # path-level guard
    split_info = verify_splits(proto, check_rows=check_rows)
    assert_selection_split_allowed(proto["selection"]["split"], proto)
    cells = grid_cells(proto)

    plan = {
        "protocol_id": proto["protocol_id"], "code_commit": _git_commit(),
        "catalog": cat_info, "splits": split_info,
        "n_cells": len(cells),
        "selection_split": proto["selection"]["split"],
        "freezable_query_variants": proto["selection"]["freezable_query_variants"],
        "datasets": [d["name"] for d in proto["evaluation"]["datasets"]],
        "out_root": _display_path(out_root),
        "bm25": {k: proto["bm25"][k] for k in ("variant", "k1", "b", "tokenizer", "query_binary")},
        "ranking_depth": proto["evaluation"]["ranking_depth"],
    }
    if dry_run:
        plan["would_index_recipes"] = sorted({c[2] for c in cells})
        plan["would_score_cells"] = len(cells)
        return plan

    if out_root.exists() and any(out_root.iterdir()) and proto["output"].get("no_overwrite", True):
        raise Bm25ProtocolError(
            f"[bm25] refusing to overwrite existing output root {out_root}.\n"
            f"Validate it instead:  demap paper bm25 --validate-only\n"
            f"or direct a fresh run elsewhere with --out-root <new path>.")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "rankings").mkdir(exist_ok=True)
    master = pd.read_parquet(REPO / proto["catalog"]["path"])
    depth = int(proto["evaluation"]["ranking_depth"])
    sel = proto["selection"]
    val_path = REPO / sel["split_path"]

    # ---- 1. selection sweep: val_dev ONLY -------------------------------------------------
    recs = []
    by_recipe: Dict[str, Any] = {}
    for recipe in [c["recipe"] for c in proto["cde_representations"]]:
        model = build_index(master, proto, recipe)
        by_recipe[recipe] = model
        for qrep in proto["query_representations"]:
            qids, qtexts = load_queries(val_path, qrep["column"])
            r = rank(model, qids, qtexts, depth=100)
            m = evaluate(r, val_path)
            recs.append({"query_variant": qrep["id"], "recipe": recipe,
                         "reportable": qrep["id"] in sel["freezable_query_variants"],
                         "recall@1": m["recall@1"], "recall@5": m["recall@5"],
                         "recall@10": m["recall@10"], "mrr@100": m["mrr@100"],
                         "n_queries": len(set(qids)), "n_index_docs": model["n_docs"]})
        del by_recipe[recipe]  # free memory; re-built for the frozen recipe below

    def order(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(list(sel["criterion"]),
                              ascending=[d == "asc" for d in sel["criterion_directions"]]
                              ).reset_index(drop=True)

    sel_df = order(pd.DataFrame(recs))
    sel_df.to_csv(out_root / "bm25_val_dev_selection.csv", index=False)
    best = order(sel_df[sel_df["reportable"]]).iloc[0]
    exp = sel["expected_winner"]
    if best["query_variant"] != exp["query_variant"] or best["recipe"] != exp["recipe"]:
        raise Bm25ProtocolError(
            f"[bm25] selection produced {best['query_variant']} x {best['recipe']}, protocol "
            f"expects {exp['query_variant']} x {exp['recipe']}")

    # ---- 2. frozen evaluation on the six canonical datasets --------------------------------
    qcol = {q["id"]: q["column"] for q in proto["query_representations"]}[str(best["query_variant"])]
    model = build_index(master, proto, str(best["recipe"]))
    cat_pubs = set(str(c).split("::")[0] for c in model["catalog_ids"])
    rows = []
    for d in proto["evaluation"]["datasets"]:
        p = REPO / d["path"]
        qids, qtexts = load_queries(p, qcol)
        r = rank(model, qids, qtexts, depth=depth)
        if set(r["pub"]) - cat_pubs:
            raise Bm25ProtocolError(f"[bm25] {d['name']}: ranked ids outside the catalog")
        if r.duplicated(["query_id", "pub"]).any():
            raise Bm25ProtocolError(f"[bm25] {d['name']}: duplicate public id within a ranking")
        r.to_parquet(out_root / "rankings" / f"bm25_rankings_{d['name']}.parquet", index=False)
        m = evaluate(r, p)
        rows.append({"dataset": d["name"], "n_queries": int(d["n_queries"]),
                     **{k: round(m[k], 4) for k in
                        ["recall@1", "recall@5", "recall@10", "recall@100", "mrr@100"]},
                     "coverage": round(m["coverage"], 4)})
    pd.DataFrame(rows).to_csv(out_root / "bm25_canonical_metrics.csv", index=False)

    prov = {
        "analysis": "canonical BM25 baseline (paper final comparison)",
        "protocol_id": proto["protocol_id"],
        "protocol_sha256": sha256_file("configs/paper/bm25_protocol_v1.yaml"),
        "code_commit": _git_commit(),
        "catalog": {"path": proto["catalog"]["path"], "sha256": proto["catalog"]["sha256"],
                    "rows": proto["catalog"]["rows"],
                    "unique_public_ids": proto["catalog"]["unique_public_ids"],
                    "retired_excluded": True,
                    "admin_contexts_excluded": proto["catalog"]["excluded_admin_contexts"]},
        "selection": {"split": sel["split"], "split_sha256": sel["split_sha256"],
                      "n_queries": int(sel["n_queries"]),
                      "grid": f"{len(proto['query_representations'])} query representations x "
                              f"{len(proto['cde_representations'])} CDE recipes",
                      "criterion": " -> ".join(sel["criterion"]),
                      "selected_query_variant": str(best["query_variant"]),
                      "selected_query_column": qcol,
                      "selected_recipe": str(best["recipe"]),
                      "val_dev_recall@5": float(best["recall@5"]),
                      "test_used_for_selection": False},
        "bm25": dict(proto["bm25"]),
        "evaluation": {"datasets": [d["name"] for d in proto["evaluation"]["datasets"]],
                       "dataset_sha256": {d["name"]: d["sha256"] for d in proto["evaluation"]["datasets"]},
                       "metric_helper": proto["evaluation"]["metric_helper"],
                       "denominator": proto["evaluation"]["denominator"],
                       "gold_matching": proto["evaluation"]["gold_matching"],
                       "dedup": proto["evaluation"]["dedup"],
                       "save_depth": depth, "deterministic": True, "seeds": None},
        "not_rerun": "no neural artifact read or written; no Slurm job submitted",
    }
    (out_root / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))
    plan["frozen"] = {"query_variant": str(best["query_variant"]), "recipe": str(best["recipe"])}
    plan["metrics"] = rows
    return plan
