#!/usr/bin/env bash
# Migrated Slurm driver. Paths are resolved from DEMAP_DATA_ROOT; the
# original hardcoded research-repository root has been parameterized.
# SBATCH --output/--error are static directives and cannot expand shell
# variables: pass them on the command line, e.g.
#   sbatch --output=<log>/%A_%a.out --error=<log>/%A_%a.err <this file>
#SBATCH --job-name=J_allow_hgbc_archft
#SBATCH --partition=norm
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gres=lscratch:20
#SBATCH --time=08:00:00

# Task J Phase-2 CONTROL: retrain the Variant-B HGBC on the ARCHIVED manuscript
# feature table (bit-identical input, no rebuilt features). Isolates trainer
# determinism from any CE-scoring float jitter in the rebuilt 0.70 chain.
# Expected: metrics identical to
# artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov/.
set -euo pipefail
REPO="${DEMAP_DATA_ROOT:?set DEMAP_DATA_ROOT to the data/artifact root}"
J="$REPO/.scratch/v17_claude/J_allowance"
WT="$REPO/.scratch/demap/paper_v13_scientific_audit/wt_hgbc_2903203"
cd "$REPO"
[[ "$(git -C "$WT" rev-parse --short=7 HEAD)" == "2903203" ]] || { echo "FATAL: worktree moved" >&2; exit 2; }

FT="$REPO/artifacts/final_reranker/hgbc_features_v2_eligible/feature_table_fixedk30_crossenc_v2.parquet"
CFG_B="$REPO/configs/hgbc_feature_sets/fixedk30_ablation_C_drop_final_rank_noprov.json"
OUT="$J/rates/a0.70/hgbc_noprov_archivedft"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$WT/src:$WT/scripts"
LS="/lscratch/${SLURM_JOB_ID:?}/J3b"; mkdir -p "$LS"; export TMPDIR="$LS" TEMP="$LS" TMP="$LS"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

echo "[$(date)] J3b retrain on ARCHIVED manuscript feature table"
srun "$PY" "$WT/scripts/train_hgbc_reranker.py" \
  --feature-table "$FT" --categorical-mode stable \
  --exclude-features-file "$CFG_B" \
  --out "$OUT"

echo "=== compare vs archived with_ce_noprov ==="
"$PY" - <<'PYEOF'
import pandas as pd, json
from pathlib import Path
REPO = Path("${DEMAP_DATA_ROOT}")
J = REPO / ".scratch/v17_claude/J_allowance"
new = pd.read_csv(J / "rates/a0.70/hgbc_noprov_archivedft/hgbc_eval_by_split.csv")
old = pd.read_csv(REPO / "artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov/hgbc_eval_by_split.csv")
m = old.merge(new, on="split", suffixes=("_arch", "_rerun"))
cols = [c for c in old.columns if c != "split" and pd.api.types.is_numeric_dtype(old[c])]
rep = {}
for _, r in m.iterrows():
    rep[r["split"]] = {c: {"archived": float(r[f"{c}_arch"]), "rerun": float(r[f"{c}_rerun"]),
                           "equal": bool(abs(r[f"{c}_arch"] - r[f"{c}_rerun"]) < 1e-12)}
                       for c in cols}
ok = all(v["equal"] for s in rep.values() for v in s.values())
print(json.dumps({"all_equal": ok}, indent=2))
for s, d in rep.items():
    bad = {c: v for c, v in d.items() if not v["equal"]}
    if bad:
        print(s, "DIFFS:", bad)
with open(J / "rates/a0.70/PARITY_TRAINER_ARCHIVEDFT.json", "w") as fh:
    json.dump({"all_equal": ok, "detail": rep}, fh, indent=2)
print("wrote", J / "rates/a0.70/PARITY_TRAINER_ARCHIVEDFT.json")
PYEOF
echo "[$(date)] J3b DONE"
