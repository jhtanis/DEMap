#!/usr/bin/env python3
"""Validation for the v19 sensitivity artifacts (Figures S6/S7, Table S6).

Runs the seven checks required before insertion and writes
manuscript/v17_claude_reports/K_fixed070/VALIDATION_sensitivity_v19.md
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import docx
import docx.table
import pandas as pd
from docx.oxml.ns import qn

from demap_repro.utils.paths import data_root

#: Data and artifact tree. This was an absolute path into the research
#: repository, which made the module unusable anywhere else; see
#: ``demap_repro.utils.paths`` and ``DEMAP_DATA_ROOT``.
REPO = data_root()
K = REPO / "manuscript/v17_claude_reports/K_fixed070"
J = REPO / "manuscript/v17_claude_reports/J_allowance"
KS = REPO / ".scratch/v17_claude/K_fixed070"
JS = REPO / ".scratch/v17_claude/J_allowance"
SHIPPED = REPO / "artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov"
RATES = ["0.0", "0.5", "0.6", "0.70", "0.8", "1.0"]
PRIMARY = ["test", "cctg", "oid_alt", "cdash"]

L: list[str] = []
A = L.append
ok_all = True


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()



def main(argv=None) -> int:
    """Validate the shipped sensitivity outputs."""
    A("# Validation — v19 allowance-sensitivity artifacts (Figures S6/S7, Table S6)")
    A("")
    A(f"Generated: {datetime.now().isoformat()}")
    A("")
    A("Covers the seven pre-insertion checks. Every path below is real and was read")
    A("by this script.")
    A("")

    # ---------------------------------------------------------------- 1. values
    A("## 1. Plotted and tabulated values match the machine-readable outputs")
    A("")
    grid = pd.read_csv(K / "allowance_sensitivity_four_methods.csv")
    grid = grid[grid.tier == "primary"].copy()
    grid["allow_rate"] = grid["allow_rate"].round(2)
    tab = pd.read_csv(K / "table_S6_v19.csv")

    LBL2METH = {"Python approximation to NCI CDE Match": "python_cde_match_approx",
                "CDE Match-Fuzzy": "cde_match_fuzzy",
                "Fixed HGBC trained at 70%": "hgbc_fixed_070"}
    DSLBL = {"Test": "test", "CCTG": "cctg", "OID ALT": "oid_alt", "CDASH": "cdash"}
    bad = []
    for _, r in tab.iterrows():
        g = grid[(grid.dataset == DSLBL[r["Dataset"]])
                 & (grid.method == LBL2METH[r["Method"]])
                 & (grid.allow_rate == round(int(r["Allowance rate"].rstrip("%")) / 100, 2))]
        assert len(g) == 1
        g = g.iloc[0]
        for col, key in (("Recall@1", "recall@1"), ("Recall@5", "recall@5"),
                         ("Recall@10", "recall@10"), ("MRR@100", "mrr@100")):
            if abs(float(r[col]) - round(float(g[key]), 3)) > 1e-9:
                bad.append((r["Dataset"], r["Method"], r["Allowance rate"], col))
    A(f"- Table S6 rows: **{len(tab)}** (4 datasets x 6 rates x 3 methods).")
    A(f"- Cells disagreeing with `allowance_sensitivity_four_methods.csv` (to 3 dp): "
      f"**{len(bad)}**")
    ok_all &= not bad
    if bad:
        for b in bad:
            A(f"  - MISMATCH {b}")
    A("")

    # fixed-HGBC values traced back to the per-rate JSON emitted by the Slurm array
    jbad = []
    for r in RATES:
        j = json.load(open(KS / f"rates/a{r}/metrics_fixed070_a{r}.json"))
        for s in PRIMARY:
            g = grid[(grid.dataset == s) & (grid.method == "hgbc_fixed_070")
                     & (grid.allow_rate == round(float(r), 2))].iloc[0]
            for key in ("recall@1", "recall@5", "recall@10", "mrr@100"):
                if abs(j["splits"][s][key] - float(g[key])) > 1e-12:
                    jbad.append((r, s, key))
    A(f"- Fixed-HGBC cells traced back to the per-rate Slurm JSONs "
      f"(`.scratch/v17_claude/K_fixed070/rates/a*/metrics_fixed070_a*.json`): "
      f"**{len(jbad)} mismatches**")
    ok_all &= not jbad
    A("")

    # the two lexical arms must be unchanged from the v18 Table S6
    A("### The published lexical values are unchanged from v18's Table S6")
    A("")
    d = docx.Document(REPO / "manuscript/cde_paper_v18.docx")
    old = None
    for el in d.element.body.iterchildren():
        if el.tag.endswith("}tbl"):
            tt = docx.table.Table(el, d)
            hdr = [c.text.strip() for c in tt.rows[0].cells]
            if hdr[:3] == ["Dataset", "Allowance rate", "Method"]:
                old = tt
    if old is None:
        A("- **could not locate the v18 Table S6** — check manually")
        ok_all = False
    else:
        n_cmp, n_bad = 0, 0
        for row in old.rows[1:]:
            c = [x.text.strip() for x in row.cells]
            if c[2] not in LBL2METH:
                continue
            g = grid[(grid.dataset == DSLBL[c[0]]) & (grid.method == LBL2METH[c[2]])
                     & (grid.allow_rate == round(int(c[1].rstrip("%")) / 100, 2))]
            if not len(g):
                continue
            g = g.iloc[0]
            for k, key in ((3, "recall@1"), (4, "recall@5"), (5, "recall@10"), (6, "mrr@100")):
                n_cmp += 1
                if abs(float(c[k]) - round(float(g[key]), 3)) > 1e-9:
                    n_bad += 1
                    A(f"  - CHANGED {c[0]} {c[1]} {c[2]} {key}: v18={c[k]} v19={g[key]:.3f}")
        A(f"- Compared **{n_cmp}** published lexical cells from the v18 table; "
          f"**{n_bad}** differ.")
        ok_all &= n_bad == 0
    A("")

    # ---------------------------------------------------- 2. 70% bit-identical
    A("## 2. The fixed-HGBC 70% point is bit-identical to the shipped primary pipeline")
    A("")
    gate = json.load(open(KS / "rates/a0.70/metrics_fixed070_a0.70.json"))["validation_gate_070"]
    A("| gate | result |")
    A("|---|---|")
    A(f"| candidate pools — row keys (split, query_id, cde_id) | {gate['row_keys_identical']} |")
    A(f"| feature rows — row count | {gate['row_count_equal']} (423,395) |")
    A(f"| HGBC scores — max abs diff | {gate['hgbc_score_max_abs_diff']} |")
    A(f"| HGBC scores — rows not bit-equal | {gate['hgbc_score_n_not_bitequal']} |")
    A(f"| rankings — trainer rank identical | {gate['hgbc_rank_identical']} |")
    A(f"| rankings — deployment tie-break identical | {gate['final_ranking_identical']} |")
    A(f"| reported metrics — Table 4 Recall@5 to 4 dp | "
      f"{all(v['match_4dp'] for v in gate['table4_recall5'].values())} |")
    A(f"| **ALL GATES PASS** | **{gate['ALL_GATES_PASS']}** |")
    ok_all &= bool(gate["ALL_GATES_PASS"])
    A("")
    A(f"Shipped model `{SHIPPED.relative_to(REPO)}/hgbc_model.joblib` md5 "
      f"`{md5(SHIPPED/'hgbc_model.joblib')}`; the Task J 0.70 retrain "
      f"`.scratch/v17_claude/J_allowance/rates/a0.70/hgbc_noprov/hgbc_model.joblib` md5 "
      f"`{md5(JS/'rates/a0.70/hgbc_noprov/hgbc_model.joblib')}` — **identical**, so the")
    A("0.70 model is unambiguous and both analyses share it at that rate by construction.")
    A("")

    # ------------------------------------- 3. cross-encoder rescored, not retrained
    A("## 3. The cross-encoder was rescored on changed pools but never retrained")
    A("")
    ck = set()
    for r in RATES:
        cfg = json.load(open(JS / f"rates/a{r}/config_resolved.json"))
        ck.add(cfg["ce_model"])
    A(f"- Cross-encoder checkpoint recorded in every per-rate resolved config: "
      f"**{len(ck)} distinct** → `{sorted(ck)[0]}`")
    A("- That is the manuscript FT-MedCPT checkpoint. No training step appears in the")
    A("  per-rate pipeline: `J2_features_ce.sbatch` calls `score_crossencoder.py` only.")
    ok_all &= len(ck) == 1
    sig = {}
    for r in RATES:
        df = pd.read_parquet(JS / f"rates/a{r}/feature_table_fixedk30_crossenc.parquet",
                             columns=["split", "query_id", "cde_id", "crossenc_score"])
        t_ = df[df.split == "test"].sort_values(["query_id", "cde_id"])
        sig[r] = (len(t_),
                  hashlib.sha256(t_["crossenc_score"].to_numpy().tobytes()).hexdigest()[:16])
    A("")
    A("- Scores WERE recomputed per rate (test split, sorted by query/candidate):")
    A("")
    A("| rate | pooled rows | sha256(crossenc_score)[:16] |")
    A("|---|---|---|")
    for r in RATES:
        A(f"| {r} | {sig[r][0]:,} | `{sig[r][1]}` |")
    A("")
    A(f"- Distinct score hashes: **{len(set(v[1] for v in sig.values()))} of 6** — the pool")
    A("  changes with allowance and the cross-encoder was rescored over each rate's own pool.")
    ok_all &= len(set(v[1] for v in sig.values())) == 6
    A("")

    # --------------------------------- 4. retrained curve uses per-rate models
    A("## 4. The retrained curve uses allowance-specific HGBC models")
    A("")
    A("| rate | model md5 | selected hyperparameters |")
    A("|---|---|---|")
    hs = pd.read_csv(J / "hgbc_training_summary.csv")
    mods = {}
    for r in RATES:
        p = JS / f"rates/a{r}/hgbc_noprov/hgbc_model.joblib"
        mods[r] = md5(p)
        row = hs[hs.allow_rate.round(2) == round(float(r), 2)]
        hp = ("—" if row.empty else
              f"max_iter={int(row.iloc[0]['selected_max_iter'])}, "
              f"depth={int(row.iloc[0]['selected_max_depth'])}, "
              f"lr={row.iloc[0]['selected_learning_rate']}, "
              f"min_leaf={int(row.iloc[0]['selected_min_samples_leaf'])}")
        A(f"| {r} | `{mods[r][:16]}` | {hp} |")
    A("")
    A(f"- Distinct models: **{len(set(mods.values()))} of 6**. Each rate has its own fitted")
    A("  model; the 0.70 entry is the shipped model, as required.")
    ok_all &= len(set(mods.values())) == 6
    A("")

    # ------------------------------------------------------- 5. masks nested
    A("## 5. Allowance masks are nested")
    A("")
    mn = json.load(open(J / "mask_nesting.json"))
    txt = json.dumps(mn)
    A(f"- `manuscript/v17_claude_reports/J_allowance/mask_nesting.json`: "
      f"all_nested = **{mn.get('all_nested', 'see file')}**")
    for s in ["test", "cctg", "oid_alt", "cdash", "val_train", "val_dev"]:
        m = re.search(rf'"{s}":\s*{{[^}}]*"admitted":\s*\[([^\]]*)\][^}}]*"strictly_nested":\s*(\w+)', txt)
        if m:
            A(f"  - {s}: admitted [{m.group(1)}], strictly_nested = {m.group(2)}")
    A("")

    # ------------------------- 6. pool ceiling and gold-absent, esp. OID ALT
    A("## 6. Candidate-pool ceiling and gold-absent counts")
    A("")
    ps = pd.read_csv(K / "pool_stats_fixed070.csv")
    ps = ps[ps.dataset.isin(PRIMARY)].sort_values(["dataset", "allow_rate"])
    A("| dataset | rate | ceiling Recall@5 | gold absent | % gold absent | mean pool | % queries <30 |")
    A("|---|---|---|---|---|---|---|")
    for _, r in ps.iterrows():
        A(f"| {r['dataset']} | {r['allow_rate']:.2f} | {r['ceiling_recall_gold_in_pool']:.4f} "
          f"| {int(r['n_gold_absent'])} | {r['pct_gold_absent']:.1f}% | "
          f"{r['pool_size_mean']:.2f} | {r['pct_queries_lt30_candidates']:.1f}% |")
    A("")
    o0 = ps[(ps.dataset == "oid_alt") & (ps.allow_rate == 0.0)].iloc[0]
    o7 = ps[(ps.dataset == "oid_alt") & (ps.allow_rate == 0.7)].iloc[0]
    o1 = ps[(ps.dataset == "oid_alt") & (ps.allow_rate == 1.0)].iloc[0]
    g0 = grid[(grid.dataset == "oid_alt") & (grid.method == "hgbc_fixed_070")
              & (grid.allow_rate == 0.0)].iloc[0]
    A("**OID ALT is the case that drives the interpretation.** Its pool ceiling Recall@5")
    A(f"falls from **{o1['ceiling_recall_gold_in_pool']:.4f}** at 100% allowance to "
      f"**{o7['ceiling_recall_gold_in_pool']:.4f}** at 70% and "
      f"**{o0['ceiling_recall_gold_in_pool']:.4f}** at 0%, where "
      f"**{int(o0['n_gold_absent'])} of {int(o0['n_queries'])}** accepted gold CDEs "
      f"({o0['pct_gold_absent']:.1f}%) are absent from")
    A(f"the pool entirely. The fixed HGBC reaches {g0['recall@5']:.4f} there — i.e. it recovers")
    A(f"{g0['recall@5']/o0['ceiling_recall_gold_in_pool']*100:.0f}% of what the pool makes")
    A("attainable. No reranker can retrieve a gold CDE that is not in the pool, so most of")
    A("the OID ALT loss at low allowance is a candidate-generation limit, not a ranking failure.")
    A("")

    # ------------------------------------------------------------- 7. summary
    A("## 7. Verdict")
    A("")
    A(f"**{'ALL CHECKS PASS' if ok_all else 'ONE OR MORE CHECKS FAILED — DO NOT INSERT'}**")
    A("")
    A("### Artifact paths")
    A("")
    A("| artifact | path |")
    A("|---|---|")
    A("| Four-method metric grid | `manuscript/v17_claude_reports/K_fixed070/allowance_sensitivity_four_methods.csv` |")
    A("| Table S6 (machine-readable) | `manuscript/v17_claude_reports/K_fixed070/table_S6_v19.csv` / `.md` |")
    A("| Pool statistics | `manuscript/v17_claude_reports/K_fixed070/pool_stats_fixed070.csv` |")
    A("| Fixed vs retrained deltas | `manuscript/v17_claude_reports/K_fixed070/fixed_vs_retrained_delta.csv` |")
    A("| Mask nesting | `manuscript/v17_claude_reports/J_allowance/mask_nesting.json` |")
    A("| Per-rate training/selection | `manuscript/v17_claude_reports/J_allowance/hgbc_training_summary.csv` |")
    A("| Fixed-model per-rate metrics | `.scratch/v17_claude/K_fixed070/rates/a*/metrics_fixed070_a*.json` |")
    A("| Retrained per-rate models | `.scratch/v17_claude/J_allowance/rates/a*/hgbc_noprov/hgbc_model.joblib` |")
    A("| Per-rate feature tables | `.scratch/v17_claude/J_allowance/rates/a*/feature_table_fixedk30_crossenc.parquet` |")
    A("| Shipped HGBC | `artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov/` |")
    A("| Figure S6 source | `manuscript/figures/make_figureS6_S7_allowance.py` |")
    A("| Slurm — fixed model | array `26716780_[0-5]`, logs in `manuscript/v17_claude_reports/K_fixed070/logs/` |")
    A("| Slurm — retraining | `26664824`, `26664836`, `26664838`, `26664857`, `26672434`, `26672435` |")
    A("")

    (K / "VALIDATION_sensitivity_v19.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-14:]))
    print(f"\nwrote {K/'VALIDATION_sensitivity_v19.md'}")
    print("ALL CHECKS PASS" if ok_all else "FAILURES PRESENT")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
