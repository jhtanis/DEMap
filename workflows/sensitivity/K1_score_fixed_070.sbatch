#!/usr/bin/env bash
#SBATCH --job-name=K_fixed070
#SBATCH --output=/vf/users/nextgen2/james/tasks/cde_project/demap/.scratch/v17_claude/K_fixed070/logs/K1_fixed070_%A_%a.out
#SBATCH --error=/vf/users/nextgen2/james/tasks/cde_project/demap/.scratch/v17_claude/K_fixed070/logs/K1_fixed070_%A_%a.err
#SBATCH --partition=norm
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --gres=lscratch:20
#SBATCH --time=01:00:00

# Task K -- FIXED-MODEL inference-allowance sensitivity.
#
# Applies the UNCHANGED 0.70-trained HGBC to each allowance rate's regenerated
# feature table. No retraining, no refitting, no model selection at any rate.
#
# This is the DEPLOYMENT-ROBUSTNESS counterpart to Task J. Task J retrained the
# HGBC at every rate (pipeline adaptability). Task K freezes the shipped
# classifier and varies only the inference-time exact-match evidence.
#
# The per-rate feature tables were built by Task J stage J2 at that rate --
# nested exact-match mask, regenerated CDE Match-Fuzzy candidates, FT-MPNet
# union pool with public-id dedup, ranks, keyword/rule features, and FT-MedCPT
# cross-encoder scores computed over that rate's own pool. They are reused
# here rather than rebuilt: the 0.70 table was verified bit-identical to the
# shipped 423,395-row manuscript artifact, so rebuilding would be a no-op that
# costs ~6 h of GPU time. The ONLY thing that changes in Task K is which model
# scores them.
#
# Array index -> rate: 0=0.0  1=0.5  2=0.6  3=0.70  4=0.8  5=1.0
set -euo pipefail
REPO="/vf/users/nextgen2/james/tasks/cde_project/demap"
J="$REPO/.scratch/v17_claude/J_allowance"
K="$REPO/.scratch/v17_claude/K_fixed070"
WT="$REPO/.scratch/demap/paper_v13_scientific_audit/wt_hgbc_2903203"
SHIPPED="$REPO/artifacts/final_reranker/hgbc_reranker_v2_eligible/with_ce_noprov"
cd "$REPO"

# Hard pin: refuse to run if the manuscript trainer worktree has moved.
[[ "$(git -C "$WT" rev-parse --short=7 HEAD)" == "2903203" ]] || {
  echo "FATAL: trainer worktree moved" >&2; exit 2; }

RATES=(0.0 0.5 0.6 0.70 0.8 1.0)
R="${RATES[${SLURM_ARRAY_TASK_ID:?}]}"
FT="$J/rates/a${R}/feature_table_fixedk30_crossenc.parquet"
OUT="$K/rates/a${R}"

for f in "$FT" "$SHIPPED/hgbc_model.joblib" "$SHIPPED/feature_set.json" \
         "$SHIPPED/categorical_vocab.json"; do
  [[ -e "$f" ]] || { echo "FATAL: missing $f" >&2; exit 4; }
done

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$WT/src:$WT/scripts"
export PYTHONNOUSERSITE=1
LS="/lscratch/${SLURM_JOB_ID:?}/K1"; mkdir -p "$LS"
export TMPDIR="$LS" TEMP="$LS" TMP="$LS" JOBLIB_TEMP_FOLDER="$LS"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "[$(date)] K1 inference_allow_rate=$R  frozen model=$SHIPPED"
echo "  feature table: $FT"
echo "  git HEAD:      $(git -C "$REPO" rev-parse HEAD)"
echo "  trainer wt:    $(git -C "$WT" rev-parse HEAD)"

srun "$PY" "$REPO/manuscript/v17_claude_reports/K_fixed070/scripts/K1_score_fixed070.py" \
  --rate "$R" \
  --feature-table "$FT" \
  --model-dir "$SHIPPED" \
  --out-dir "$OUT"

echo "[$(date)] K1 rate=$R DONE"
