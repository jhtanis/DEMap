"""Item 1: Phase 0 off-the-shelf 4x10 representation grids for all three models.

Extraction logic copied verbatim in spirit from
notebooks/phase0_all_mpnet_representation_heatmap.ipynb (same assertions).
READ-ONLY on artifacts. Writes only under .scratch/demap/paper_v10_revision/data/.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path("/data/nextgen2/james/tasks/cde_project/demap")
PHASE0_ROOT = REPO / "artifacts/phase0_representation_sweep_manuscript_v5"
OUT = REPO / ".scratch/demap/paper_v10_revision/data"

EXPECT_SPLIT = "val_dev"
EXPECT_N = 4234
EXPECT_CATALOG = ("data/processed/cadsr_xml_2026-06-18/"
                  "cde_master_enriched_eval_production_cde_match.parquet")
EXPECT_CATALOG_N = 62976
METRIC = "recall@5"

MODELS = [
    ("all-MPNet", "all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"),
    ("BioSimCSE", "biosimcse-biolinkbert-base", "kamalkraj/BioSimCSE-BioLinkBERT-BASE"),
    ("PubMedBERT", "pubmedbert-base-embeddings", "NeuML/pubmedbert-base-embeddings"),
]

QUERY_ORDER = [("Q1", "RAW"), ("Q3", "RAW_PV"), ("Q4", "RAW_PV_PH"), ("Q2", "PREF_PV")]
RECIPE_ORDER = [
    ("v3", "PQT"),
    ("v2", "LN_DEF"),
    ("v1_v3", "SN_PQT"),
    ("v1_v2a_v2b_v3", "SN_LN_DEF_PQT"),
    ("v1_v2a_v2b_v5", "SN_LN_DEF_PV"),
    ("v1_v2a_v3_v5", "SN_LN_PQT_PV"),
    ("v1_v2a_v2b_v3_v5", "SN_LN_DEF_PQT_PV"),
    ("v1_v6_v2b_v3_v5", "SN_DEC_DEF_PQT_PV"),
    ("v1_v2a_v2b_v3_v4_v5", "SN_LN_DEF_PQT_VD_PV"),
    ("v1_v2a_v6_v2b_v3_v5", "SN_LN_DEC_DEF_PQT_PV"),
]
ANCHOR = ("Q3", "v1_v6_v2b_v3_v5")
QLABEL, RLABEL = dict(QUERY_ORDER), dict(RECIPE_ORDER)


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


all_rows, sources, summary = [], [], {}


def main(argv=None) -> int:
    """Build phase0_heatmaps.csv, the input to Figures 2, S1 and S2."""
    for display, dirname, model_id in MODELS:
        model_dir = PHASE0_ROOT / dirname
        assert model_dir.is_dir(), model_dir
        rows = []
        for cell_dir in sorted(model_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            for mpath in sorted(cell_dir.glob("*/metrics.json")):
                run_dir = mpath.parent
                cpath = run_dir / "run_config.json"
                m = json.loads(mpath.read_text())
                c = json.loads(cpath.read_text())

                assert c["model_name"] == model_id, (run_dir.name, c["model_name"])
                assert "__ots__" in m["run_id"], f"not off-the-shelf: {m['run_id']}"
                splits = list(m["metrics_by_split"])
                assert splits == [EXPECT_SPLIT], (run_dir.name, splits)
                blk = m["metrics_by_split"][EXPECT_SPLIT]
                assert int(blk["n"]) == EXPECT_N, (run_dir.name, blk["n"])
                assert c["inputs"]["cde_master_enriched"] == EXPECT_CATALOG, run_dir.name
                assert int(c["catalog"]["n_cdes"]) == EXPECT_CATALOG_N, run_dir.name

                spec = m.get("spec", c.get("spec", {}))
                rows.append({
                    "model": display,
                    "model_id": model_id,
                    "query_variant": spec["query_variant"],
                    "query_label": QLABEL[spec["query_variant"]],
                    "cde_recipe": spec["recipe"],
                    "cde_recipe_label": RLABEL[spec["recipe"]],
                    "recall_at_5": float(blk[METRIC]),
                    "n_queries": int(blk["n"]),
                    "run_dir": str(run_dir.relative_to(REPO)),
                    "run_id": m["run_id"],
                })
                sources.append({"model": display,
                                "path": str(mpath.relative_to(REPO)),
                                "sha256": sha256(mpath)})

        df = pd.DataFrame(rows)
        assert len(df) == 40, (display, len(df))
        assert df.duplicated(["query_variant", "cde_recipe"]).sum() == 0
        want = {(q, r) for q, _ in QUERY_ORDER for r, _ in RECIPE_ORDER}
        have = set(map(tuple, df[["query_variant", "cde_recipe"]].values))
        assert have == want, f"{display}: missing={want-have} extra={have-want}"

        ranked = df.sort_values("recall_at_5", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        best = ranked.iloc[0]
        anch = ranked[(ranked.query_variant == ANCHOR[0]) &
                      (ranked.cde_recipe == ANCHOR[1])].iloc[0]
        rank_map = {(r.query_variant, r.cde_recipe): int(r["rank"])
                    for _, r in ranked.iterrows()}
        df["rank_within_model"] = [rank_map[(q, r)] for q, r in
                                   zip(df.query_variant, df.cde_recipe)]
        summary[display] = {
            "model_id": model_id,
            "n_cells": int(len(df)),
            "anchor": {"query_variant": ANCHOR[0], "cde_recipe": ANCHOR[1],
                       "label": f"{QLABEL[ANCHOR[0]]} x {RLABEL[ANCHOR[1]]}",
                       "recall_at_5": float(anch["recall_at_5"]),
                       "rank": int(anch["rank"]), "of": 40},
            "best": {"query_variant": str(best.query_variant),
                     "cde_recipe": str(best.cde_recipe),
                     "label": f"{QLABEL[best.query_variant]} x {RLABEL[best.cde_recipe]}",
                     "recall_at_5": float(best["recall_at_5"]),
                     "rank": int(best["rank"]), "of": 40},
            "best_minus_anchor": float(best["recall_at_5"] - anch["recall_at_5"]),
        }
        all_rows.append(df)

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["model", "query_variant", "cde_recipe"]).reset_index(drop=True)
    csv_path = OUT / "phase0_heatmaps.csv"
    out.to_csv(csv_path, index=False)

    prov = {
        "output": "phase0_heatmaps.csv",
        "item": "1 - Phase 0 off-the-shelf 4x10 query-representation x CDE-recipe grids",
        "metric": "val_dev Recall@5",
        "stage": "off-the-shelf (no fine-tuning); run_id contains '__ots__'",
        "extraction": ("per-run metrics.json + run_config.json, replicating the assertions in "
                       "notebooks/phase0_all_mpnet_representation_heatmap.ipynb"),
        "assertions_per_cell": {
            "model_name": "== model_id",
            "off_the_shelf": "'__ots__' in run_id",
            "splits": [EXPECT_SPLIT],
            "denominator_n": EXPECT_N,
            "catalog_path": EXPECT_CATALOG,
            "catalog_n_cdes": EXPECT_CATALOG_N,
        },
        "source_artifact_root": str(PHASE0_ROOT.relative_to(REPO)),
        "grid": {"query_order": [q for q, _ in QUERY_ORDER],
                 "query_labels": [l for _, l in QUERY_ORDER],
                 "recipe_order": [r for r, _ in RECIPE_ORDER],
                 "recipe_labels": [l for _, l in RECIPE_ORDER]},
        "anchor_cell": {"query_variant": ANCHOR[0], "cde_recipe": ANCHOR[1],
                        "label": "RAW_PV x SN_DEC_DEF_PQT_PV"},
        "per_model_summary": summary,
        "n_rows": int(len(out)),
        "n_source_files": len(sources),
        "source_files": sources,
        "missing_cells": [],
    }
    (OUT / "phase0_heatmaps_provenance.json").write_text(json.dumps(prov, indent=1))

    print(f"wrote {csv_path} rows={len(out)}")
    for k, v in summary.items():
        print(f"{k:11s} anchor {v['anchor']['recall_at_5']:.6f} rank {v['anchor']['rank']}/40 | "
              f"best {v['best']['label']:24s} {v['best']['recall_at_5']:.6f} rank {v['best']['rank']}/40")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
