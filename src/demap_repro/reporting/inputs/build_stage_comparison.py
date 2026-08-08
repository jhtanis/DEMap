"""Items 5 and 6: stage comparison, and retained Phase 2 bi-encoder R@5 on six canonical sets.

READ-ONLY on artifacts. Writes only under .scratch/demap/paper_v10_revision/data/.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from demap_repro.utils.paths import data_root

REPO = data_root()
OUT = REPO / ".scratch/demap/paper_v10_revision/data"
def _code_commit() -> str:
    """Git HEAD of the tree the inputs were built from, for the provenance sidecar."""
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def sha256(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def src(p):
    p = Path(p)
    ap = p if p.is_absolute() else REPO / p
    return {"path": str(ap.relative_to(REPO)), "sha256": sha256(ap),
            "bytes": ap.stat().st_size}



def main(argv=None) -> int:
    """Build stage_comparison.csv and s5_retained_biencoder_recall5.csv."""
    ORDER = {"all-MPNet": 0, "BioSimCSE": 1, "PubMedBERT": 2}

    # ===================================================================== item 5
    p0 = pd.read_csv(OUT / "phase0_heatmaps.csv")
    p1 = pd.read_csv(OUT / "phase1_winners.csv")
    p2 = pd.read_csv(OUT / "phase2_winners.csv")

    rows5, src5 = [], [src(OUT.relative_to(REPO) / "phase0_heatmaps.csv") if False else None]
    src5 = []

    for model in ["all-MPNet", "BioSimCSE", "PubMedBERT"]:
        # --- off_the_shelf: best cell of that model's Phase 0 4x10 grid (deterministic, 1 run)
        sub = p0[p0.model == model].sort_values("recall_at_5", ascending=False)
        b = sub.iloc[0]
        mpath = Path(b["run_dir"]) / "metrics.json"
        rows5.append({
            "model": model, "stage": "off_the_shelf",
            "val_dev_recall5_mean": float(b["recall_at_5"]), "val_dev_recall5_sd": "",
            "n_seeds": 1, "value_type": "deterministic single off-the-shelf run (no fine-tuning)",
            "config": f"{b['query_variant']} ({b['query_label']}) x {b['cde_recipe']} ({b['cde_recipe_label']})",
            "n_queries": int(b["n_queries"]),
            "source": str(mpath),
        })
        src5.append({"model": model, "stage": "off_the_shelf", **src(mpath)})

        # --- phase1
        r = p1[p1.model == model].iloc[0]
        rows5.append({
            "model": model, "stage": "phase1",
            "val_dev_recall5_mean": float(r["val_dev_recall5_mean"]),
            "val_dev_recall5_sd": float(r["val_dev_recall5_sd"]),
            "n_seeds": 2, "value_type": "seed mean over seeds {0,1}; SD sample ddof=1",
            "config": (f"{r['query_rep']} x {r['cde_recipe']} / {r['loss']} / lr {r['lr']} / "
                       f"t {r['temperature']} / {r['epochs']}ep"),
            "n_queries": 4234, "source": r["source"],
        })

        # --- phase2
        r = p2[p2.model == model].iloc[0]
        rows5.append({
            "model": model, "stage": "phase2",
            "val_dev_recall5_mean": float(r["val_dev_recall5_mean"]),
            "val_dev_recall5_sd": float(r["val_dev_recall5_sd"]),
            "n_seeds": 2, "value_type": "seed mean over seeds {0,1}; SD sample ddof=1",
            "config": (f"{r['negative_strategy']} / lr {r['lr']} / t {r['temperature']} / "
                       f"{r['epochs']}ep"),
            "n_queries": 4234,
            "source": f"{r['runs_root']} (select.select_phase2_seed_mean, phase2_seed_mean_lexicographic_v1)",
        })

    df5 = pd.DataFrame(rows5)
    df5["_o"] = df5.model.map(ORDER)
    df5 = df5.sort_values(["_o", "stage"], key=lambda s: s if s.name == "_o" else s.map(
        {"off_the_shelf": 0, "phase1": 1, "phase2": 2})).drop(columns="_o").reset_index(drop=True)
    df5.to_csv(OUT / "stage_comparison.csv", index=False)

    handoff = {
        ("all-MPNet", "off_the_shelf"): (0.267596, None),
        ("BioSimCSE", "off_the_shelf"): (0.292395, None),
        ("PubMedBERT", "off_the_shelf"): (0.265234, None),
        ("all-MPNet", "phase1"): (0.854983, 0.004342),
        ("BioSimCSE", "phase1"): (0.843647, 0.001670),
        ("PubMedBERT", "phase1"): (0.818611, 0.022713),
        ("all-MPNet", "phase2"): (0.901393, 0.000835),
    }
    verif = []
    for _, r in df5.iterrows():
        k = (r["model"], r["stage"])
        if k in handoff:
            hm, hs = handoff[k]
            okm = abs(float(r["val_dev_recall5_mean"]) - hm) < 5e-7
            oks = (hs is None) or abs(float(r["val_dev_recall5_sd"]) - hs) < 5e-7
            verif.append({"model": r["model"], "stage": r["stage"], "handoff_mean": hm,
                          "artifact_mean": float(r["val_dev_recall5_mean"]),
                          "mean_matches": bool(okm), "handoff_sd": hs,
                          "artifact_sd": (None if r["val_dev_recall5_sd"] == ""
                                          else float(r["val_dev_recall5_sd"])),
                          "sd_matches": bool(oks)})
        else:
            verif.append({"model": r["model"], "stage": r["stage"],
                          "handoff_mean": None, "artifact_mean": float(r["val_dev_recall5_mean"]),
                          "note": "not stated in the handoff; newly aggregated from artifacts"})

    prov5 = {
        "output": "stage_comparison.csv",
        "item": "5 - S3.6 / Section 4.4 stage figure: model x stage val_dev Recall@5",
        "metric": "Validation Dev Recall@5, n = 4234, June 62,976-CDE production catalog",
        "stage_definitions": {
            "off_the_shelf": ("best cell of that model's Phase 0 4x10 off-the-shelf representation "
                              "grid (artifacts/phase0_representation_sweep_manuscript_v5). A single "
                              "deterministic inference run, so no seed SD exists; sd is left empty."),
            "phase1": ("canonical Phase 1 seed-mean winner (rule phase1_seed_mean_lexicographic_v1). "
                       "BioSimCSE/PubMedBERT from configs/paper/phase1_winners_for_phase2_v1.yaml; "
                       "all-MPNet re-aggregated from artifacts/phase1_finetuning/all-mpnet-base-v2."),
            "phase2": ("canonical Phase 2 seed-mean winner (rule phase2_seed_mean_lexicographic_v1, "
                       "configuration identity INCLUDING negative strategy). all-MPNet from "
                       "artifacts/phase2_finetuning/all-mpnet-base-v2; BioSimCSE/PubMedBERT newly "
                       "aggregated from artifacts/phase2_canonical_v1 (108 runs, Slurm 26053167)."),
        },
        "handoff_verification": verif,
        "handoff_verification_summary": (
            "All seven values quoted in HANDOFF_2026-07-26_DEMAP_ONLY.md were reproduced exactly "
            "from artifacts. The two remaining cells (BioSimCSE and PubMedBERT Phase 2) were not in "
            "the handoff and were aggregated here for the first time."),
        "code_commit": _code_commit(),
        "source_files": src5,
        "derived_from": ["phase0_heatmaps.csv", "phase1_winners.csv", "phase2_winners.csv"],
        "n_rows": int(len(df5)),
        "missing_cells": [],
    }
    (OUT / "stage_comparison_provenance.json").write_text(json.dumps(prov5, indent=1))
    print("item5 rows", len(df5))
    print(df5[["model", "stage", "val_dev_recall5_mean", "val_dev_recall5_sd"]].to_string(index=False))

    # ===================================================================== item 6
    DATASETS = [("test", 3959), ("cctg", 1097), ("oid_alt", 1766),
                ("cdash", 324), ("gdc_combined", 72), ("cimac_v2", 131)]

    BASE = REPO / "artifacts/final_reranker/hgbc_reranker_medcpt/with_ce/baseline_eval_by_split.csv"
    DEEP = REPO / "artifacts/final_reranker/biencoder_deep/eval_canonical/biencoder_deep_rankings_top1000.parquet"
    CMP = REPO / "artifacts/final_reranker/final_report/FINAL_METHOD_COMPARISON.csv"
    MANIFEST = REPO / "artifacts/final_reranker/final_report/final_model_manifest.json"

    base = pd.read_csv(BASE)
    be = base[base.method == "biencoder"].set_index("split")

    # independent recomputation from the depth-1000 bi-encoder rankings
    deep = pd.read_parquet(DEEP, columns=["split", "query_id", "biencoder_rank", "is_label"])
    best_rank = deep[deep.is_label].groupby(["split", "query_id"])["biencoder_rank"].min()
    nq = deep.groupby("split")["query_id"].nunique()
    recomputed = {s: float((best_rank.loc[s] <= 5).sum()) / int(nq[s]) for s, _ in DATASETS}

    cmp_df = pd.read_csv(CMP)
    cmp_ft = cmp_df[cmp_df.method == "ft_mpnet"].set_index("dataset")

    rows6, missing6 = [], []
    MPNET_CKPT = ("artifacts/phase2_finetuning/all-mpnet-base-v2/runs/"
                  "20260705_203956__phase2_finetuning__ft2__20260703_142804__phase1_finetuning__ft__"
                  "all-mpnet-base-v2__Q-cc2ba6ac__Q3__v1_v6_v2b_v3_v5__labeled__R0__symmetric_mnrl__"
                  "lr5e-05__bs11__t0.03__ep1__-6f14e0fbed/model")

    for ds, denom in DATASETS:
        r = be.loc[ds]
        assert int(r["n_queries"]) == denom, (ds, r["n_queries"], denom)
        assert abs(float(r["recall@5"]) - recomputed[ds]) < 1e-12, ds
        assert abs(float(r["recall@5"]) - float(cmp_ft.loc[ds, "recall@5"])) < 5e-5, ds
        rows6.append({
            "model": "all-MPNet", "dataset": ds, "n_queries": denom,
            "recall_at_5": float(r["recall@5"]),
            "value_type": ("retained-seed deterministic checkpoint evaluation "
                           "(Phase 2 seed 1, -6f14e0fbed); NOT a seed mean"),
            "n_seeds": 1,
            "checkpoint": MPNET_CKPT,
            "source": str(BASE.relative_to(REPO)),
            "independent_recomputation": recomputed[ds],
            "recall_at_1": float(r["recall@1"]), "recall_at_10": float(r["recall@10"]),
            "mrr_at_100": float(r["mrr@100"]),
        })

    for model in ["BioSimCSE", "PubMedBERT"]:
        for ds, denom in DATASETS:
            rows6.append({
                "model": model, "dataset": ds, "n_queries": denom,
                "recall_at_5": "MISSING", "value_type": "MISSING", "n_seeds": "",
                "checkpoint": "MISSING", "source": "MISSING",
                "independent_recomputation": "", "recall_at_1": "MISSING",
                "recall_at_10": "MISSING", "mrr_at_100": "MISSING",
            })
        missing6.append(model)

    df6 = pd.DataFrame(rows6)
    df6["_o"] = df6.model.map(ORDER)
    df6 = df6.sort_values(["_o"]).drop(columns="_o").reset_index(drop=True)
    df6.to_csv(OUT / "s5_retained_biencoder_recall5.csv", index=False)

    prov6 = {
        "output": "s5_retained_biencoder_recall5.csv",
        "item": "6 - Recall@5 of the retained Phase 2 bi-encoder on the six canonical eval datasets",
        "canonical_registry": "configs/evaluation/canonical_eval_datasets.yaml",
        "canonical_data_root": "data/processed/eval_canonical/",
        "denominators_query_level": dict(DATASETS),
        "all_mpnet": {
            "status": "COMPLETE for all six datasets",
            "checkpoint": MPNET_CKPT,
            "value_type": ("Deterministic evaluation of the SINGLE retained Phase 2 checkpoint "
                           "(seed 1, suffix -6f14e0fbed). It is a retained-seed evaluation, NOT a "
                           "seed mean. The corresponding val_dev seed mean (0.901393 +/- 0.000835) "
                           "is a different quantity and must not be mixed with these."),
            "primary_source": str(BASE.relative_to(REPO)),
            "primary_source_note": ("rows with method == 'biencoder'; carries an explicit n_queries "
                                    "column matching the canonical query-level denominators exactly. "
                                    "Only recall@1/5/10/mrr@100 are meaningful in this file; its "
                                    "recall@100 column is really the candidate-pool ceiling."),
            "cross_check_1": {
                "source": str(CMP.relative_to(REPO)),
                "rows": "method == 'ft_mpnet' (deep top-1000 retrieval)",
                "agreement": "identical to 4 dp on all six datasets",
            },
            "cross_check_2": {
                "source": str(DEEP.relative_to(REPO)),
                "method": ("independently recomputed query-level Recall@5 as "
                           "min(biencoder_rank | is_label) <= 5 grouped by (split, query_id) "
                           "over the 7,349,000-row depth-1000 ranking table"),
                "agreement": "exact to <1e-12 on all six datasets",
                "recomputed_values": recomputed,
            },
            "model_identity_evidence": str(MANIFEST.relative_to(REPO)),
        },
        "MISSING": {
            "models": missing6,
            "datasets": [d for d, _ in DATASETS],
            "n_missing_cells": 12,
            "reason": (
                "No evaluation of the CANONICAL Phase 2 retained checkpoints for BioSimCSE or "
                "PubMedBERT on the six canonical datasets exists anywhere in artifacts/."),
            "what_was_searched": [
                "artifacts/phase2_canonical_v1/ — all 108 metrics.json files were opened; every one "
                "contains metrics_by_split == {'val_dev'} only. No canonical-dataset evaluation was "
                "ever run for these checkpoints.",
                "results/ — contains only results/phase1_canonical_v1/ (val_dev-only Phase 1 sweep) "
                "and results/section4_3/. No results/phase2_canonical_v1/ exists.",
                "artifacts/final_reranker/ — bi-encoder canonical-dataset evaluation exists for the "
                "FT-MPNet retained checkpoint only (it is the promoted candidate generator).",
                "artifacts/evaluation/phase1_phase2_seedavg_model_winners_canonical_eval/ — DOES "
                "contain phase2__biosimcse-biolinkbert-base__hard_top25__seed{0,1} and "
                "phase2__pubmedbert-base-embeddings__semihard_1_50__seed{0,1}, BUT these evaluate "
                "the HISTORICAL 2026-07-05 artifacts/phase2_finetuning/ checkpoints, not the "
                "canonical Phase 2 winners. For BioSimCSE the representation even differs (historical "
                "Q3 x v1_v6_v2b_v3_v5 vs canonical winner Q2 x v1_v2a_v3_v5), and they are scored at "
                "PAIR level (test 4258, cctg 1159, oid_alt 1908, cdash 526), not at the canonical "
                "query-level denominators. They are therefore NOT substitutes and were not used.",
                "artifacts/publication_artifact_package_20260713/, "
                "artifacts/production_handoff_reranker_20260713/, artifacts/reports/, "
                "artifacts/summaries/, artifacts/tables/, artifacts/figures/, notebooks/data/, "
                "notebooks/figures/, notebooks/paper_figures/ — no canonical-dataset bi-encoder "
                "metrics for the canonical Phase 2 BioSimCSE/PubMedBERT checkpoints.",
            ],
            "what_would_be_required": (
                "Running eval_checkpoint on data/processed/eval_canonical for the two canonical "
                "Phase 2 retained checkpoints listed in phase2_winners.csv, at query level. That is "
                "new computation and was deliberately NOT performed here."),
        },
        "code_commit": _code_commit(),
        "source_files": [src(BASE), src(CMP), src(MANIFEST),
                         {"path": str(DEEP.relative_to(REPO)),
                          "sha256": "not hashed (2.0+ GB parquet); verified by recomputation",
                          "bytes": DEEP.stat().st_size},
                         src(REPO / "artifacts/final_reranker/biencoder_deep/eval_canonical/recall_summary.csv")],
        "n_rows": int(len(df6)),
        "missing_cells": [f"{m}/{d} recall_at_5" for m in missing6 for d, _ in DATASETS],
    }
    (OUT / "s5_retained_biencoder_recall5_provenance.json").write_text(json.dumps(prov6, indent=1))
    print("item6 rows", len(df6), "missing", len(prov6["missing_cells"]))
    print(df6[df6.model == "all-MPNet"][["model", "dataset", "n_queries", "recall_at_5"]].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
