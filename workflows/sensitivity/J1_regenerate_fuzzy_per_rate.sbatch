#!/usr/bin/env bash
# Migrated Slurm driver. Paths are resolved from DEMAP_DATA_ROOT; the
# original hardcoded research-repository root has been parameterized.
# SBATCH --output/--error are static directives and cannot expand shell
# variables: pass them on the command line, e.g.
#   sbatch --output=<log>/%A_%a.out --error=<log>/%A_%a.err <this file>
#SBATCH --job-name=J_allow_fuzzyval
#SBATCH --partition=norm
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=lscratch:50
#SBATCH --time=06:00:00
#SBATCH --array=0-5

# Task J (allowance-specific HGBC sensitivity), stage 1:
# Regenerate the CDE Match-Fuzzy (keyword_v1) candidate tables for the HGBC
# TRAINING/SELECTION splits (val_train, val_dev) at each allowance rate.
# Mirrors kwfuzzy_v2_eligible_array.sbatch task 2 EXACTLY (frozen 2026-06-18
# snapshot inputs, production_cde_match eligibility, exact_sql, ref_docs,
# fuzzy-top-k-per-rule=100, seed 42) -- only --exact-query-match-allow-rate varies.
# Code = pinned worktree wt_ce_5f06f10 (verified byte-identical to the 84049ed
# lexical code that generated the manuscript keyword_fuzzy_v2_eligible tables).
# Array index 3 (rate 0.70) is a DETERMINISM CHECK: regenerates val_train/val_dev
# at 0.70 into fuzzy_val_check/ and diffs against the archived manuscript tables.
set -euo pipefail
REPO="${DEMAP_DATA_ROOT:?set DEMAP_DATA_ROOT to the data/artifact root}"
J="$REPO/.scratch/v17_claude/J_allowance"
WT="$REPO/.scratch/demap/paper_v13_scientific_audit/wt_ce_5f06f10"
cd "$REPO"
[[ "$(git -C "$WT" rev-parse --short=7 HEAD)" == "5f06f10" ]] || { echo "FATAL: worktree moved" >&2; exit 2; }

RATES=(0.0 0.5 0.6 0.70 0.8 1.0)
R="${RATES[${SLURM_ARRAY_TASK_ID:?}]}"
TAG="_allow${R}"
if [[ "$R" == "0.70" ]]; then
  OUT="$J/rates/a0.70/fuzzy_val_check"
else
  OUT="$J/rates/a${R}/fuzzy_val"
fi
mkdir -p "$OUT"

CDE_MASTER="data/processed/cadsr_xml_2026-06-18/cde_master_enriched_eval.parquet"
ALT="data/interim/cadsr_xml_2026-06-18/cadsr_merged/cde_alternate_names.parquet"
PVS="data/interim/cadsr_xml_2026-06-18/cadsr_merged/cde_permissible_values.parquet"
REF="data/interim/cadsr_xml_2026-06-18/cadsr_merged/cde_reference_documents.parquet"

if [[ -n "${SLURM_JOB_ID:-}" && -d "/lscratch/${SLURM_JOB_ID}" ]]; then
  export TMPDIR="/lscratch/${SLURM_JOB_ID}/J1"
else
  export TMPDIR="$J/tmp"
fi
mkdir -p "$TMPDIR"; export TEMP="$TMPDIR" TMP="$TMPDIR" JOBLIB_TEMP_FOLDER="$TMPDIR"
export PYTHONPATH="$WT/src"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY="$REPO/.venv/bin/python"

echo "[$(date)] J1 rate=$R splits=val_train,val_dev topk=100 out=$OUT code=$WT"
srun "$PY" "$WT/scripts/build_cde_match_candidates.py" \
  --splits "val_train,val_dev" --split-dir "data/processed/splits" \
  --cde-master "$CDE_MASTER" --alt-names "$ALT" --pvs "$PVS" --ref-docs "$REF" \
  --eligibility production_cde_match --normalization exact_sql --question-source ref_docs \
  --fuzzy-fallback keyword_v1 \
  --fuzzy-top-k-per-rule 100 \
  --exact-query-match-allow-rate "$R" \
  --out-dir "$OUT" --tag "$TAG" --overwrite

if [[ "$R" == "0.70" ]]; then
  echo "=== determinism check vs archived keyword_fuzzy_v2_eligible (0.70) ==="
  "$PY" "$J/scripts/compare_fuzzy_070.py"
fi
echo "[$(date)] J1 rate=$R DONE"
