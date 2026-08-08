#!/usr/bin/env bash
#SBATCH --job-name=J_allow_feat_ce
#SBATCH --output=/vf/users/nextgen2/james/tasks/cde_project/demap/.scratch/v17_claude/J_allowance/logs/J2_featce_%A_%a.out
#SBATCH --error=/vf/users/nextgen2/james/tasks/cde_project/demap/.scratch/v17_claude/J_allowance/logs/J2_featce_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1,lscratch:100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=04:00:00

# Task J stage 2 (per allowance rate): fixed-K30 base feature table
# (FT-MPNet top20 UNION keyword_v1-fuzzy top10, public-id dedup, no gold
# injection) + MedCPT fulltrain cross-encoder scoring + crossenc_* merge.
# Mirrors chain_D/chain_F5 exactly (worktree wt_ce_5f06f10, run_fixedk wrapper,
# SN_DEC_DEF_PQT_PV recipe, query_text_q3, batch 128, --require-candidate-text);
# the ONLY experimental variable is the allowance rate, which enters via
#   (a) the keyword_v1 fuzzy candidate tables routed into the pool, and
#   (b) --keyword-allow-rate for the kw_* provenance exact-gate (seed 42).
# External splits (gdc_combined, cimac_v2) stay at allowance 1.0 at every rate
# (manuscript policy).
set -euo pipefail
REPO="/vf/users/nextgen2/james/tasks/cde_project/demap"
J="$REPO/.scratch/v17_claude/J_allowance"
SCRATCH="$REPO/.scratch/demap/paper_v13_scientific_audit"
WT="$SCRATCH/wt_ce_5f06f10"
cd "$REPO"
[[ "$(git -C "$WT" rev-parse --short=7 HEAD)" == "5f06f10" ]] || { echo "FATAL: worktree moved" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: must run under Slurm" >&2; exit 2; }

RATES=(0.0 0.5 0.6 0.70 0.8 1.0)
R="${RATES[${SLURM_ARRAY_TASK_ID:?}]}"
RD="$J/rates/a${R}"
mkdir -p "$RD/fixedk30_base" "$RD/scores" \
         "$RD/adapter/splits_dir" "$RD/adapter/fuzzy_cadsr" "$RD/adapter/fuzzy_ext"

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$WT/src:$WT/scripts"
export DEMAP_CODE_DIR="$WT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
LS="/lscratch/${SLURM_JOB_ID}/J2"; mkdir -p "$LS"; export TMPDIR="$LS" TEMP="$LS" TMP="$LS" JOBLIB_TEMP_FOLDER="$LS"

PROD_CAT="$REPO/data/processed/cadsr_xml_2026-06-18/cde_master_enriched_eval_production_cde_match.parquet"
KW_INDEX="$REPO/artifacts/final_reranker/keyword_fuzzy_v2_eligible/keyword_index/keyword_index_v3_production_cde_match.joblib"
KWV2="$REPO/artifacts/final_reranker/keyword_fuzzy_v2_eligible"
SENS="$REPO/artifacts/paper_v13_lexical_sensitivity/kwfuzzy"
RANKINGS="$SCRATCH/adapter_v2/rankings/phase2_allmpnet_rep_6f14e0fbed/biencoder_deep_rankings_top1000.parquet"
CE_CKPT="$REPO/artifacts/final_reranker/crossencoder_fulltrain_v2_eligible/runs/ncbi_MedCPT-Cross-Encoder"
SPLITS="val_train,val_dev,test,cctg,oid_alt,cdash,gdc_combined,cimac_v2"
BASE_FT="$RD/fixedk30_base/feature_table.parquet"
CE_SCORES="$RD/scores/crossenc_scores_MedCPT_fulltrain.parquet"
MERGED_FT="$RD/feature_table_fixedk30_crossenc.parquet"

for f in "$PROD_CAT" "$KW_INDEX" "$RANKINGS" "$CE_CKPT/model.safetensors"; do
  [[ -e "$f" ]] || { echo "FATAL: missing input: $f" >&2; exit 4; }
done

echo "=== GPU fail-fast preflight ==="
srun "$PY" - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA unavailable - refusing to run"
print(f"[gpu-guard] {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}")
PYEOF

echo "=== stage 0: adapter (rate $R) ==="
for s in val_train val_dev test; do
  ln -sfn "$REPO/data/processed/splits/${s}.parquet" "$RD/adapter/splits_dir/${s}.parquet"
done
for s in cctg oid_alt cdash gdc_combined cimac_v2; do
  ln -sfn "$REPO/data/processed/eval_canonical/${s}.parquet" "$RD/adapter/splits_dir/${s}.parquet"
done
# caDSR-derived splits: rate-specific fuzzy tables.
for s in test cctg oid_alt cdash; do
  if [[ "$R" == "0.70" ]]; then src="$KWV2/cde_match_clone_candidates_${s}__allow0.70.parquet"
  else src="$SENS/cde_match_clone_candidates_${s}__allow${R}.parquet"; fi
  [[ -e "$src" ]] || { echo "FATAL: missing fuzzy table: $src" >&2; exit 4; }
  ln -sfn "$src" "$RD/adapter/fuzzy_cadsr/cde_match_clone_candidates_${s}_fuzzy_a070.parquet"
done
for s in val_train val_dev; do
  if [[ "$R" == "0.70" ]]; then src="$KWV2/cde_match_clone_candidates_${s}__allow0.70.parquet"
  else src="$J/rates/a${R}/fuzzy_val/cde_match_clone_candidates_${s}__allow${R}.parquet"; fi
  [[ -e "$src" ]] || { echo "FATAL: missing fuzzy table: $src" >&2; exit 4; }
  ln -sfn "$src" "$RD/adapter/fuzzy_cadsr/cde_match_clone_candidates_${s}_fuzzy_a070.parquet"
done
# externals: allowance 1.0 always (manuscript policy).
for s in gdc_combined cimac_v2; do
  src="$KWV2/cde_match_clone_candidates_${s}__allow1.0.parquet"
  [[ -e "$src" ]] || { echo "FATAL: missing external fuzzy: $src" >&2; exit 4; }
  ln -sfn "$src" "$RD/adapter/fuzzy_ext/cde_match_clone_candidates_${s}_fuzzy_a10.parquet"
done

echo "=== stage A: fixed-K30 base feature table (keyword-allow-rate $R) ==="
srun "$PY" "$SCRATCH/run_fixed_k_stepG_v2.py" \
  --splits "$SPLITS" \
  --splits-dir "$RD/adapter/splits_dir" \
  --cde-master "$PROD_CAT" \
  --rankings "$RANKINGS" \
  --keyword-index "$KW_INDEX" \
  --fuzzy-dir-cadsr "$RD/adapter/fuzzy_cadsr" \
  --fuzzy-dir-external "$RD/adapter/fuzzy_ext" \
  --biencoder-k 20 --keyword-fuzzy-k 10 \
  --keyword-allow-rate "$R" --keyword-allow-seed 42 \
  --out "$BASE_FT" --overwrite

echo "=== stage A5: is_injected_gold + key uniqueness ==="
"$PY" - "$BASE_FT" <<'PYEOF'
import sys, pandas as pd
p = sys.argv[1]
df = pd.read_parquet(p)
if "is_injected_gold" not in df.columns:
    df["is_injected_gold"] = False
    df.to_parquet(p, index=False)
assert df.duplicated(["split","query_id","cde_id"]).sum() == 0
assert "query_text_q3" in df.columns
print(f"[A5] base rows={len(df)} splits={sorted(df['split'].unique())}")
PYEOF

echo "=== stage A6: eligibility assertion ==="
"$PY" - "$BASE_FT" "$PROD_CAT" <<'PYEOF'
import sys, pandas as pd
ft, prod = sys.argv[1], sys.argv[2]
df = pd.read_parquet(ft, columns=["cde_id"])
prod_pubs = set(pd.read_parquet(prod, columns=["cde_id"])["cde_id"].astype(str)
                .str.split("::").str[0])
pubs = df["cde_id"].astype(str).str.split("::").str[0]
inel = int((~pubs.isin(prod_pubs)).sum())
print(f"[A6] off-production rows: {inel}")
assert inel == 0
PYEOF

echo "=== stage B: MedCPT fulltrain CE scoring (GPU) ==="
srun "$PY" "$WT/scripts/score_crossencoder.py" \
  --model "$CE_CKPT" \
  --feature-table "$BASE_FT" \
  --cde-master "$PROD_CAT" \
  --splits "$SPLITS" \
  --cde-text-recipe "SN_DEC_DEF_PQT_PV" \
  --query-text-col "query_text_q3" \
  --require-candidate-text \
  --device cuda --batch-size 128 \
  --out "$CE_SCORES"

echo "=== stage C: merge crossenc_* features ==="
srun "$PY" "$WT/scripts/build_hgbc_feature_table_with_crossenc.py" \
  --feature-table "$BASE_FT" \
  --crossenc-scores "$CE_SCORES" \
  --out "$MERGED_FT" --overwrite

echo "=== stage D: candidate-pool statistics ==="
"$PY" "$J/scripts/pool_stats.py" --rate "$R" --table "$BASE_FT" --out "$RD/pool_stats.json"

if [[ "$R" == "0.70" ]]; then
  echo "=== stage E (0.70 only): feature parity vs archived manuscript table ==="
  "$PY" "$J/scripts/parity_070_features.py"
fi

echo "=== resolved config ==="
"$PY" - "$RD" "$R" <<PYEOF
import json, sys, subprocess
rd, rate = sys.argv[1], sys.argv[2]
cfg = {
  "allow_rate": rate,
  "mask": {"impl": "exact_match_mask_v1 (SHA-1, seed 42, nested)", "seed": 42,
           "fields": ["P","Q","A"], "scope": "gold", "level": "auto",
           "external_splits_allow": 1.0},
  "pool": {"biencoder_k": 20, "keyword_fuzzy_k": 10, "dedup": "public_id",
           "inject_gold": False},
  "biencoder_rankings": "$RANKINGS",
  "keyword_index": "$KW_INDEX",
  "cde_master": "$PROD_CAT",
  "ce_model": "$CE_CKPT",
  "ce_recipe": "SN_DEC_DEF_PQT_PV", "ce_query_col": "query_text_q3",
  "keyword_allow_rate_cadsr": float(rate), "keyword_allow_seed": 42,
  "code": {"features_ce_worktree": "5f06f10", "trainer_worktree": "2903203"},
  "splits": "$SPLITS",
  "slurm_job": "${SLURM_JOB_ID}",
}
with open(f"{rd}/config_resolved.json", "w") as fh:
    json.dump(cfg, fh, indent=2)
print("wrote", f"{rd}/config_resolved.json")
PYEOF
echo "[$(date)] J2 rate=$R DONE  merged=$MERGED_FT"
