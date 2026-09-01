"""``demap`` — one entry point per reproduction stage.

Every stage is also importable and runnable as a module; this dispatcher exists so
the pipeline reads as an ordered list rather than a directory of scripts::

    demap --list
    demap select-k --grid artifacts/final_reranker/k_selection/k_selection_grid.csv
    demap leakage-filter --dry-run

No stage is currently gated: every algorithm the paper workflow runs ships here.
The mechanism is retained because some inputs remain undistributed - see
``docs/provenance.md`` and ``manifests/source_migration.yaml``.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Dict, List, NamedTuple


class Stage(NamedTuple):
    name: str
    module: str
    group: str
    summary: str
    gated: bool = False


# Ordered as the pipeline runs.
STAGES: List[Stage] = [
    # A — dataset construction
    Stage("make-dataset", "demap_repro.data.pipeline", "A. Dataset",
          "build the benchmark: catalog extraction, queries, pairs, splits"),
    Stage("catalog-filter", "demap_repro.data.catalog_filter", "A. Dataset",
          "restrict the catalog and drop orphaned queries"),
    Stage("reachable-splits", "demap_repro.data.reachability", "A. Dataset",
          "filter evaluation splits to queries whose gold is in the catalog"),
    Stage("materialize-eval", "demap_repro.data.cli.materialize_eval", "A. Dataset",
          "materialize the six canonical evaluation datasets"),
    Stage("import-gdc", "demap_repro.data.cli.import_gdc", "A. Dataset",
          "import the two supplied GDC curation tables into the split schema"),
    Stage("import-cimac", "demap_repro.data.cimac.appendix_a_v2", "A. Dataset",
          "import the supplied CIMAC workbook into the split schema"),
    Stage("leakage-filter", "demap_repro.data.leakage_filter", "A. Dataset",
          "remove evaluation queries that recur in training with the same gold"),
    Stage("pv-diagnostics", "demap_repro.data.pv_frozen_diagnostics", "A. Dataset",
          "permissible-value overlap diagnostics (Table S3)"),
    Stage("characterize", "demap_repro.data.characterization", "A. Dataset",
          "dataset composition and shift statistics (Tables S1, S2)"),

    # B — bi-encoder
    Stage("biencoder", "demap_repro.biencoder.cli", "B. Bi-encoder",
          "representation screen, Phase 1 and Phase 2 fine-tuning, selection"),
    Stage("deep-retrieval", "demap_repro.biencoder.deep_retrieval", "B. Bi-encoder",
          "FT-MPNet deep catalog retrieval feeding the candidate pool"),

    # C — lexical
    Stage("bm25", "demap_repro.lexical.bm25.cli", "C. Lexical",
          "BM25 baseline"),
    Stage("cdematch-candidates", "demap_repro.lexical.cde_match.build_candidates", "C. Lexical",
          "Python CDE Match approximation, or CDE Match-Fuzzy with --fuzzy-fallback"),
    Stage("non-exact-eval", "demap_repro.lexical.non_exact_subset", "C. Lexical",
          "Recall on queries with no exact-match evidence (Figure 5C)"),

    # D — candidate pool
    Stage("select-k", "demap_repro.pool.select_k", "D. Candidate pool",
          "choose the pool size K from the ceiling grid"),
    Stage("coverage", "demap_repro.pool.coverage", "D. Candidate pool",
          "Table S5: gold coverage of each retrieval arm and their union"),

    # E — cross-encoder
    Stage("ce-pool-train", "demap_repro.crossencoder.pool_train", "E. Cross-encoder",
          "corrected training pool, including the 172-query exclusion"),
    Stage("ce-pool-eval", "demap_repro.crossencoder.pool_eval", "E. Cross-encoder",
          "corrected evaluation pool"),
    Stage("ce-pairs", "demap_repro.crossencoder.pairs", "E. Cross-encoder",
          "cross-encoder training pairs"),
    Stage("ce-train", "demap_repro.crossencoder.train", "E. Cross-encoder",
          "fine-tune one backbone under the frozen protocol"),
    Stage("ce-score", "demap_repro.crossencoder.score", "E. Cross-encoder",
          "score a candidate pool with a cross-encoder"),
    Stage("ce-select", "demap_repro.crossencoder.select", "E. Cross-encoder",
          "aggregate the bake-off and write CE_WINNER.json"),
    Stage("ce-evaluate", "demap_repro.crossencoder.evaluate", "E. Cross-encoder",
          "cross-encoder reranking evaluation"),

    # F — reranker
    Stage("fixed-k-features", "demap_repro.reranker.fixed_k_features", "F. Reranker",
          "assemble the fixed-K (K=30) candidate feature table"),
    Stage("merge-ce-features", "demap_repro.reranker.merge_crossenc_features", "F. Reranker",
          "merge the six crossenc_* features into the feature table"),
    Stage("train-hgbc", "demap_repro.reranker.train", "F. Reranker",
          "train and select the 117-feature HGBC reranker"),
    Stage("eval-by-dataset", "demap_repro.evaluation.by_dataset", "F. Reranker",
          "evaluate the reranker per evaluation dataset"),
    Stage("family-ablation", "demap_repro.reranker.family_ablation", "F. Reranker",
          "Figure S6: the three-family drop-lists and their frozen-config commands"),

    # G — reporting
    Stage("table4", "demap_repro.reporting.table4", "G. Reporting",
          "assemble Table 4, the six-method comparison"),
    Stage("figure4-inputs", "demap_repro.reporting.inputs.build_crossencoder", "G. Reporting",
          "Figure 4 input tables from the corrected roots"),
    Stage("figure5-inputs", "demap_repro.reporting.inputs.build_keyword", "G. Reporting",
          "Figure 5 input tables from the corrected roots"),
    Stage("figureS6", "demap_repro.reporting.figures.make_figureS6_broad_family_ablation",
          "G. Reporting", "Figure S6: render the broad evidence-family ablation heatmap"),

    # H — sensitivity
    Stage("leakage-sensitivity", "demap_repro.sensitivity.leakage", "H. Sensitivity",
          "Table S6: rescore saved rankings on the filtered query lists"),
    Stage("allowance-sensitivity", "demap_repro.sensitivity.allowance.aggregate", "H. Sensitivity",
          "Table S7 / Figures S7-S8: aggregate the allowance sweep"),
]

BY_NAME: Dict[str, Stage] = {s.name: s for s in STAGES}


def _print_stages() -> None:
    group = None
    for stage in STAGES:
        if stage.group != group:
            group = stage.group
            print(f"\n{group}")
        mark = " [gated]" if stage.gated else ""
        print(f"  {stage.name:22s} {stage.summary}{mark}")
    if any(s.gated for s in STAGES):
        print("\n[gated] needs a component whose redistribution status is unresolved; "
              "see docs/provenance.md")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="demap", description=__doc__, add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", help="stage name; omit with --list to see all")
    ap.add_argument("--list", action="store_true", help="list the stages in pipeline order")
    ap.add_argument("-h", "--help", action="store_true")
    args, rest = ap.parse_known_args(argv)

    if args.list or (not args.stage and args.help) or not args.stage:
        ap.print_help()
        _print_stages()
        return 0

    stage = BY_NAME.get(args.stage)
    if stage is None:
        print(f"unknown stage {args.stage!r}\n", file=sys.stderr)
        _print_stages()
        return 2

    module = importlib.import_module(stage.module)
    entry = getattr(module, "main", None)
    if entry is None:
        print(f"{stage.module} has no main(); run it as a module", file=sys.stderr)
        return 2
    if args.help:
        rest = rest + ["--help"]
    return int(entry(rest) or 0)


if __name__ == "__main__":
    sys.exit(main())
