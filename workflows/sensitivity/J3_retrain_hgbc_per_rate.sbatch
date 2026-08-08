#!/usr/bin/env bash
#SBATCH --job-name=J_allow_hgbc
#SBATCH --output=/vf/users/nextgen2/james/tasks/cde_project/demap/.scratch/v17_claude/J_allowance/logs/J3_hgbc_%A_%a.out
#SBATCH --error=/vf/users/nextgen2/james/tasks/cde_project/demap/.scratch/v17_claude/J_allowance/logs/J3_hgbc_%A_%a.err
#SBATCH --partition=norm
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gres=lscratch:20
#SBATCH --time=08:00:00

# Task J stage 3 (per allowance rate): retrain the final-generation HGBC
# (Variant B "no_provenance", 117 features) from scratch on the rate-specific
# feature table. EXACT frozen Step-H protocol of the manuscript model
# (chain_G arm 4): train=val_train only, tune/select=val_dev only
# (deployment-style Recall@5), the SAME predefined 16-config grid
# (max_iter {200,500} x max_depth {3,5} x lr {0.05,0.1} x min_samples_leaf
# {10,30}), random_state=42, --categorical-mode stable,
# --exclude-features-file fixedk30_ablation_C_drop_final_rank_noprov.json.
# NO per-rate hyperparameter retuning beyond this identical predefined grid.
# Code = pinned worktree wt_hgbc_2903203 (the manuscript trainer).
set -euo pipefail
REPO="/vf/users/nextgen2/james/tasks/cde_project/demap"
J="$REPO/.scratch/v17_claude/J_allowance"
WT="$REPO/.scratch/demap/paper_v13_scientific_audit/wt_hgbc_2903203"
cd "$REPO"
[[ "$(git -C "$WT" rev-parse --short=7 HEAD)" == "2903203" ]] || { echo "FATAL: worktree moved" >&2; exit 2; }

RATES=(0.0 0.5 0.6 0.70 0.8 1.0)
R="${RATES[${SLURM_ARRAY_TASK_ID:?}]}"
RD="$J/rates/a${R}"
FT="$RD/feature_table_fixedk30_crossenc.parquet"
CFG_B="$REPO/configs/hgbc_feature_sets/fixedk30_ablation_C_drop_final_rank_noprov.json"
OUT="$RD/hgbc_noprov"
for f in "$FT" "$CFG_B"; do [[ -e "$f" ]] || { echo "FATAL: missing $f" >&2; exit 4; }; done

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$WT/src:$WT/scripts"
LS="/lscratch/${SLURM_JOB_ID:?}/J3"; mkdir -p "$LS"; export TMPDIR="$LS" TEMP="$LS" TMP="$LS"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

echo "[$(date)] J3 rate=$R train_hgbc_reranker (Variant B noprov, stable) ft=$FT"
srun "$PY" "$WT/scripts/train_hgbc_reranker.py" \
  --feature-table "$FT" --categorical-mode stable \
  --exclude-features-file "$CFG_B" \
  --out "$OUT"

echo "=== post metrics (deploy tie-break, MRR@100, 0.70 reproduction check) ==="
"$PY" "$J/scripts/post_metrics.py" --rate "$R" --run-dir "$RD"
echo "[$(date)] J3 rate=$R DONE"
