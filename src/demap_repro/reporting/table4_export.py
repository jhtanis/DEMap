"""Release-preparation exporter for the text-free Table 4 referee bundle.

This utility is intentionally not part of the referee command.  A release
owner supplies an explicit JSON export specification naming and hashing every
authoritative source artifact.  No development path or private default is
embedded in the distributed code.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from demap_repro.reporting.table4_bundle import (
    MANIFEST_FILE,
    QUERY_FILE,
    RANKING_FILE,
    _assert_external_output,
    _require_safe_token,
    canonical_hgbc_order,
    compute_metrics,
    contract_datasets,
    contract_methods,
    dataset_map,
    default_contract_path,
    load_contract,
    method_map,
    query_set_identity_sha256,
    sha256_file,
    sort_public_ids,
    validate_ft_medcpt_order,
    write_query_parquet,
    write_ranking_parquet,
)

EXPORT_SPEC_VERSION = 1


class Table4ExportError(ValueError):
    """An export source or derived bundle invariant failed."""


def _source(
    entry: dict[str, Any], role: str, registry: dict[str, dict[str, Any]]
) -> Path:
    try:
        path = Path(entry["path"]).expanduser().resolve()
        artifact_id = _require_safe_token(entry["artifact_id"], "source artifact ID")
        expected_hash = str(entry["sha256"])
    except (KeyError, TypeError) as exc:
        raise Table4ExportError(f"invalid source specification for {role}") from exc
    if not path.is_file():
        raise Table4ExportError(f"source artifact does not exist for {role}: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise Table4ExportError(
            f"source SHA-256 mismatch for {role}/{artifact_id}: "
            f"{actual_hash} != {expected_hash}")
    record = {
        "artifact_id": artifact_id,
        "sha256": actual_hash,
        "bytes": path.stat().st_size,
        "role": role,
    }
    prior = registry.get(artifact_id)
    if prior is not None and prior != record:
        raise Table4ExportError(f"source artifact ID is reused inconsistently: {artifact_id}")
    registry[artifact_id] = record
    return path


def _public_id(value: Any) -> str:
    text = str(value)
    if "::" in text:
        text = text.split("::", 1)[0]
    return _require_safe_token(text, "CDE public identifier")


def _source_split_map(spec: dict[str, Any], datasets: list[str]) -> dict[str, str]:
    raw = spec.get("source_dataset_names", {})
    result = {dataset: str(raw.get(dataset, dataset)) for dataset in datasets}
    if set(result) != set(datasets):
        raise Table4ExportError("source_dataset_names does not cover contract datasets")
    return result


def _build_queries(
    spec: dict[str, Any],
    contract: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    source_specs = spec.get("sources", {}).get("queries", {})
    dmap = dataset_map(contract)
    rows: list[dict[str, Any]] = []
    query_ids: dict[str, set[str]] = {}
    for dataset in contract_datasets(contract):
        if dataset not in source_specs:
            raise Table4ExportError(f"missing query source for {dataset}")
        path = _source(source_specs[dataset], f"queries:{dataset}", registry)
        frame = pq.read_table(path, columns=["query_id", "cde_publicid"]).to_pandas()
        if frame[["query_id", "cde_publicid"]].isna().any().any():
            raise Table4ExportError(f"null query/gold identity in {dataset} query source")
        grouped: list[dict[str, Any]] = []
        for query_id, group in frame.groupby("query_id", sort=False, dropna=False):
            opaque_id = _require_safe_token(query_id, "query_id")
            gold = sort_public_ids(_public_id(value) for value in group["cde_publicid"])
            if not gold:
                raise Table4ExportError(f"query has no accepted gold IDs: {(dataset, opaque_id)}")
            grouped.append({"dataset": dataset, "query_id": opaque_id,
                            "gold_public_ids": gold})
        grouped.sort(key=lambda item: item["query_id"])
        expected = int(dmap[dataset]["denominator"])
        if len(grouped) != expected:
            raise Table4ExportError(
                f"query denominator mismatch for {dataset}: {len(grouped)} != {expected}")
        rows.extend(grouped)
        query_ids[dataset] = {item["query_id"] for item in grouped}
    if len(rows) != int(contract["total_queries"]):
        raise Table4ExportError(f"query total mismatch: {len(rows)}")
    return rows, query_ids


def _check_source_queries(
    frame: pd.DataFrame, dataset: str, expected: set[str], query_col: str
) -> None:
    observed = set(frame[query_col].astype(str))
    unexpected = observed - expected
    if unexpected:
        raise Table4ExportError(
            f"source contains unknown {dataset} queries: {sorted(unexpected)[:3]}")


def _archived_rankings(
    frame: pd.DataFrame,
    *,
    dataset: str,
    query_ids: set[str],
    query_col: str,
    id_col: str,
    rank_col: str,
    score_col: str | None,
    maximum_depth: int,
) -> dict[str, tuple[list[str], list[int], list[float] | None]]:
    if frame.empty:
        return {}
    _check_source_queries(frame, dataset, query_ids, query_col)
    ordered = frame.sort_values([query_col, rank_col], kind="stable")
    result: dict[str, tuple[list[str], list[int], list[float] | None]] = {}
    for raw_query_id, group in ordered.groupby(query_col, sort=False):
        query_id = str(raw_query_id)
        ids: list[str] = []
        source_ranks: list[int] = []
        scores: list[float] | None = [] if score_col is not None else None
        seen: set[str] = set()
        for row in group.itertuples(index=False):
            public_id = _public_id(getattr(row, id_col))
            if public_id in seen:
                continue
            seen.add(public_id)
            ids.append(public_id)
            source_rank = int(getattr(row, rank_col))
            if source_rank < 1 or source_rank > maximum_depth:
                raise Table4ExportError(
                    f"invalid source rank in source for {(dataset, query_id)}")
            source_ranks.append(source_rank)
            if scores is not None:
                score = float(getattr(row, score_col))
                if not math.isfinite(score):
                    raise Table4ExportError(
                        f"non-finite score in source for {(dataset, query_id)}")
                scores.append(score)
            if len(ids) == maximum_depth:
                break
        if source_ranks != sorted(set(source_ranks)):
            raise Table4ExportError(
                f"non-increasing source ranks after public-ID dedup for {(dataset, query_id)}")
        result[query_id] = (ids, source_ranks, scores)
    return result


def _read_partitioned_method(
    source_specs: dict[str, Any],
    *,
    role: str,
    contract: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    query_ids: dict[str, set[str]],
    query_col: str,
    id_col: str,
    rank_col: str,
    score_col: str | None,
    maximum_depth: int,
) -> tuple[dict[tuple[str, str], tuple[list[str], list[int], list[float] | None]], dict[str, str]]:
    rankings: dict[tuple[str, str], tuple[list[str], list[int], list[float] | None]] = {}
    artifact_ids: dict[str, str] = {}
    columns = [query_col, id_col, rank_col] + ([score_col] if score_col else [])
    for dataset in contract_datasets(contract):
        if dataset not in source_specs:
            raise Table4ExportError(f"missing {role} source for {dataset}")
        entry = source_specs[dataset]
        path = _source(entry, f"{role}:{dataset}", registry)
        table = pq.read_table(path, columns=columns,
                              filters=[(rank_col, "<=", maximum_depth)])
        frame = table.to_pandas()
        by_query = _archived_rankings(
            frame,
            dataset=dataset,
            query_ids=query_ids[dataset],
            query_col=query_col,
            id_col=id_col,
            rank_col=rank_col,
            score_col=score_col,
            maximum_depth=maximum_depth,
        )
        for query_id, value in by_query.items():
            rankings[(dataset, query_id)] = value
        artifact_ids[dataset] = entry["artifact_id"]
    return rankings, artifact_ids


def _read_ft_mpnet(
    entry: dict[str, Any], spec: dict[str, Any], contract: dict[str, Any],
    registry: dict[str, dict[str, Any]], query_ids: dict[str, set[str]],
) -> tuple[dict[tuple[str, str], tuple[list[str], list[int], list[float] | None]], str]:
    path = _source(entry, "rankings:ft_mpnet", registry)
    table = pq.read_table(
        path,
        columns=["split", "query_id", "cde_id", "biencoder_rank"],
        filters=[("biencoder_rank", "<=", 100)],
    )
    frame = table.to_pandas()
    split_map = _source_split_map(spec, contract_datasets(contract))
    result: dict[tuple[str, str], tuple[list[str], list[int], list[float] | None]] = {}
    for dataset, source_split in split_map.items():
        subset = frame[frame["split"].astype(str) == source_split]
        by_query = _archived_rankings(
            subset, dataset=dataset, query_ids=query_ids[dataset], query_col="query_id",
            id_col="cde_id", rank_col="biencoder_rank", score_col=None,
            maximum_depth=100,
        )
        for query_id, value in by_query.items():
            result[(dataset, query_id)] = value
    unknown_splits = set(frame["split"].astype(str)) - set(split_map.values())
    if unknown_splits:
        raise Table4ExportError(f"FT-MPNet source has unexpected splits: {unknown_splits}")
    return result, entry["artifact_id"]


def _read_final(
    entry: dict[str, Any], spec: dict[str, Any], contract: dict[str, Any],
    registry: dict[str, dict[str, Any]], query_ids: dict[str, set[str]],
) -> tuple[dict[tuple[str, str], tuple[list[str], list[int], list[float]]], str]:
    path = _source(entry, "rankings:final_reranker", registry)
    columns = ["split", "query_id", "cde_publicid", "hgbc_score", "is_injected_gold"]
    frame = pq.read_table(path, columns=columns).to_pandas()
    split_map = _source_split_map(spec, contract_datasets(contract))
    frame = frame[frame["split"].astype(str).isin(split_map.values())]
    frame = frame[~frame["is_injected_gold"].astype(bool)]
    result: dict[tuple[str, str], tuple[list[str], list[int], list[float]]] = {}
    for dataset, source_split in split_map.items():
        subset = frame[frame["split"].astype(str) == source_split]
        _check_source_queries(subset, dataset, query_ids[dataset], "query_id")
        for raw_query_id, group in subset.groupby("query_id", sort=False):
            ids = [_public_id(value) for value in group["cde_publicid"]]
            scores = [float(value) for value in group["hgbc_score"]]
            canonical_ids, canonical_scores = canonical_hgbc_order(ids, scores)
            if len(canonical_ids) > 30:
                raise Table4ExportError(
                    f"final candidate pool exceeds 30 for {(dataset, raw_query_id)}")
            result[(dataset, str(raw_query_id))] = (
                canonical_ids, list(range(1, len(canonical_ids) + 1)), canonical_scores)
    return result, entry["artifact_id"]


def _read_ft_medcpt(
    ranking_entry: dict[str, Any], score_entry: dict[str, Any], spec: dict[str, Any],
    contract: dict[str, Any], registry: dict[str, dict[str, Any]],
    query_ids: dict[str, set[str]],
) -> tuple[dict[tuple[str, str], tuple[list[str], list[int], list[float]]], str]:
    ranking_path = _source(ranking_entry, "rankings:ft_medcpt_archived_order", registry)
    score_path = _source(score_entry, "scores:ft_medcpt_corrected", registry)
    columns = ["split", "query_id", "cde_id", "cde_publicid", "crossenc_score",
               "crossenc_rank", "is_injected_gold"]
    frame = pq.read_table(ranking_path, columns=columns).to_pandas()
    split_map = _source_split_map(spec, contract_datasets(contract))
    frame = frame[frame["split"].astype(str).isin(split_map.values())]
    frame = frame[~frame["is_injected_gold"].astype(bool)].copy()

    score_frame = pq.read_table(
        score_path, columns=["split", "query_id", "cde_id", "crossenc_score"]
    ).to_pandas()
    score_frame = score_frame[score_frame["split"].astype(str).isin(split_map.values())]
    # The final feature table is the authoritative archived ranking used for
    # the manuscript evaluation.  The separately archived corrected-score file
    # is an identity cross-check only: some catalog-version cde_id strings and
    # a small number of scores differ, so it must not replace or reorder the
    # feature table's explicit crossenc_rank.
    score_frame["cde_publicid"] = score_frame["cde_id"].map(_public_id)
    left = frame[["split", "query_id", "cde_publicid"]].copy()
    left["cde_publicid"] = left["cde_publicid"].map(_public_id)
    merged = left.merge(
        score_frame[["split", "query_id", "cde_publicid"]],
        on=["split", "query_id", "cde_publicid"], how="outer",
        indicator=True, validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise Table4ExportError(
            "FT-MedCPT archived-order and corrected-score public-ID identities differ")

    result: dict[tuple[str, str], tuple[list[str], list[int], list[float]]] = {}
    for dataset, source_split in split_map.items():
        subset = frame[frame["split"].astype(str) == source_split]
        _check_source_queries(subset, dataset, query_ids[dataset], "query_id")
        for raw_query_id, group in subset.groupby("query_id", sort=False):
            ordered = group.sort_values("crossenc_rank", kind="stable")
            ranks = ordered["crossenc_rank"].astype(int).tolist()
            if ranks != list(range(1, len(ranks) + 1)):
                raise Table4ExportError(
                    f"non-contiguous FT-MedCPT archived ranks for {(dataset, raw_query_id)}")
            ids = [_public_id(value) for value in ordered["cde_publicid"]]
            if len(ids) != len(set(ids)):
                raise Table4ExportError(
                    f"duplicate FT-MedCPT candidate for {(dataset, raw_query_id)}")
            scores = [float(value) for value in ordered["crossenc_score"]]
            validate_ft_medcpt_order(ids, scores)
            if len(ids) > 30:
                raise Table4ExportError(
                    f"FT-MedCPT candidate pool exceeds 30 for {(dataset, raw_query_id)}")
            result[(dataset, str(raw_query_id))] = (ids, ranks, scores)
    return result, ranking_entry["artifact_id"]


def _bundle_rows(
    spec: dict[str, Any], contract: dict[str, Any], registry: dict[str, dict[str, Any]],
    query_rows: list[dict[str, Any]], query_ids: dict[str, set[str]],
) -> list[dict[str, Any]]:
    sources = spec.get("sources", {})
    methods = method_map(contract)
    rankings: dict[
        str, dict[tuple[str, str], tuple[list[str], list[int], list[float] | None]]
    ] = {}
    source_ids: dict[str, str | dict[str, str]] = {}

    rankings["final_reranker"], source_ids["final_reranker"] = _read_final(
        sources["final_reranker"], spec, contract, registry, query_ids)
    rankings["ft_medcpt"], source_ids["ft_medcpt"] = _read_ft_medcpt(
        sources["ft_medcpt_rankings"], sources["ft_medcpt_scores"], spec, contract,
        registry, query_ids)
    rankings["ft_mpnet"], source_ids["ft_mpnet"] = _read_ft_mpnet(
        sources["ft_mpnet"], spec, contract, registry, query_ids)

    definitions = {
        "cde_match_fuzzy": ("cde_match_fuzzy", "cde_id", "cdematch_rank", 100),
        "python_cde_match_approx": ("python_cde_match_approx", "cde_id", "cdematch_rank", 10),
        "bm25": ("bm25", "pub", "rank", 100),
    }
    for method, (source_key, id_col, rank_col, maximum_depth) in definitions.items():
        rankings[method], source_ids[method] = _read_partitioned_method(
            sources[source_key], role=f"rankings:{method}", contract=contract,
            registry=registry, query_ids=query_ids, query_col="query_id",
            id_col=id_col, rank_col=rank_col, score_col=None,
            maximum_depth=maximum_depth,
        )

    rows: list[dict[str, Any]] = []
    for query in query_rows:
        dataset, query_id = query["dataset"], query["query_id"]
        for method in contract_methods(contract):
            ids, source_ranks, scores = rankings[method].get(
                (dataset, query_id),
                ([], [], [] if method in {"final_reranker", "ft_medcpt"} else None),
            )
            source = source_ids[method]
            artifact_id = source[dataset] if isinstance(source, dict) else source
            rows.append({
                "dataset": dataset,
                "method": method,
                "query_id": query_id,
                "ranked_public_ids": ids,
                "source_ranks": source_ranks,
                "scores": scores,
                "returned_depth": len(ids),
                "ranking_policy": methods[method]["ranking_policy"],
                "source_artifact_id": artifact_id,
            })
    if len(rows) != 44094:
        raise Table4ExportError(f"ranking row total mismatch: {len(rows)}")
    return rows


def _preflight_results(
    contract: dict[str, Any], query_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
) -> None:
    gold = {(row["dataset"], row["query_id"]): frozenset(row["gold_public_ids"])
            for row in query_rows}
    rankings = {(row["dataset"], row["method"], row["query_id"]):
                (row["ranked_public_ids"], row["source_ranks"])
                for row in ranking_rows}
    metrics = compute_metrics(contract, gold, rankings)
    failures: list[str] = []
    for dataset in contract["datasets"]:
        for method, expected in dataset["results"].items():
            observed = metrics[dataset["id"]][method]
            if observed["hits_at_5"] != int(expected["hits_at_5"]):
                failures.append(
                    f"{dataset['id']}/{method}: {observed['hits_at_5']} != "
                    f"{expected['hits_at_5']}")
    if failures:
        raise Table4ExportError(
            "derived rankings do not reproduce the Table 4 contract:\n  " +
            "\n  ".join(failures))


def build_bundle(
    spec_path: Path,
    out_dir: Path,
    *,
    contract_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a text-free bundle from explicitly specified, hash-pinned sources."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != EXPORT_SPEC_VERSION:
        raise Table4ExportError("export specification schema_version must be 1")
    bundle_id = _require_safe_token(spec.get("bundle_id"), "bundle_id")
    contract_file = Path(contract_path or default_contract_path()).resolve()
    contract = load_contract(contract_file)
    output = _assert_external_output(Path(out_dir))
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / QUERY_FILE, output / RANKING_FILE, output / MANIFEST_FILE]
    if not overwrite and any(path.exists() for path in targets):
        raise Table4ExportError(
            "bundle output already exists; choose an empty directory or pass --overwrite")

    registry: dict[str, dict[str, Any]] = {}
    query_rows, query_ids = _build_queries(spec, contract, registry)
    ranking_rows = _bundle_rows(spec, contract, registry, query_rows, query_ids)
    _preflight_results(contract, query_rows, ranking_rows)

    query_path, ranking_path = output / QUERY_FILE, output / RANKING_FILE
    write_query_parquet(query_rows, query_path)
    write_ranking_parquet(ranking_rows, ranking_path)
    query_frame = pq.read_table(query_path).to_pandas()
    ranking_frame = pq.read_table(ranking_path).to_pandas()

    dcounts = {item["id"]: int(item["denominator"]) for item in contract["datasets"]}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "scientific_scope": (
            "Recompute published Table 4 metrics from frozen query-level outputs; "
            "no training or neural inference."),
        "results_contract": {
            "contract_id": contract["contract_id"],
            "sha256": sha256_file(contract_file),
        },
        "query_set_identity_sha256": query_set_identity_sha256(query_frame),
        "query_rows": len(query_frame),
        "ranking_rows": len(ranking_frame),
        "dataset_query_counts": dcounts,
        "method_row_counts": {method: int((ranking_frame["method"] == method).sum())
                              for method in contract_methods(contract)},
        "method_maximum_depths": {item["id"]: int(item["maximum_depth"])
                                  for item in contract["methods"]},
        "ranking_policies": {item["id"]: item["ranking_policy"]
                             for item in contract["methods"]},
        "files": {
            QUERY_FILE: {"sha256": sha256_file(query_path), "bytes": query_path.stat().st_size,
                         "rows": len(query_frame)},
            RANKING_FILE: {"sha256": sha256_file(ranking_path),
                           "bytes": ranking_path.stat().st_size,
                           "rows": len(ranking_frame)},
        },
        "source_artifacts": sorted(registry.values(), key=lambda item: item["artifact_id"]),
        "expected_results": {item["id"]: item["results"] for item in contract["datasets"]},
        "generation": {
            "software": "demap_table4_bundle_export",
            "software_schema_version": 1,
            "demap_version": version("DEMap"),
            "python_version": platform.python_version(),
            "pyarrow_version": pa.__version__,
            "pandas_version": pd.__version__,
        },
        "safety_scan": {
            "forbidden_text_columns_absent": True,
            "query_ids_checked_opaque": len(query_frame),
            "unsafe_query_ids": 0,
            "filesystem_paths_in_data_columns": 0,
            "email_like_values_in_data_columns": 0,
        },
        "provenance": spec.get("provenance", {"binding": False}),
    }
    if manifest["provenance"].get("binding") is not False:
        raise Table4ExportError("bundle provenance must be explicitly non-binding")
    (output / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path,
                        help="explicit JSON source specification with paths and SHA-256 values")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="external output directory for the candidate bundle")
    parser.add_argument("--contract", type=Path, default=None,
                        help="override the packaged Table 4 results contract")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace only the three named bundle outputs if they already exist")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_bundle(
            args.spec, args.out_dir, contract_path=args.contract, overwrite=args.overwrite)
    except (Table4ExportError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {manifest['query_rows']} query rows")
    print(f"wrote {manifest['ranking_rows']} ranking rows")
    for name, details in manifest["files"].items():
        print(f"{name}: {details['bytes']} bytes sha256={details['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
