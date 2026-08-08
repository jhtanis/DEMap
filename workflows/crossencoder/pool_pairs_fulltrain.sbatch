#!/usr/bin/env bash
# Migrated Slurm driver. Paths are resolved from DEMAP_DATA_ROOT; the
# original hardcoded research-repository root has been parameterized.
# SBATCH --output/--error are static directives and cannot expand shell
# variables: pass them on the command line, e.g.
#   sbatch --output=<log>/%A_%a.out --error=<log>/%A_%a.err <this file>
#SBATCH --job-name=demap_v13_F2_pool
#SBATCH --partition=norm
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=lscratch:20
#SBATCH --time=02:00:00

# Paper v13 chain F2 (CPU): corrected FULLTRAIN CE pool (train + val_dev) and
# corrected fulltrain pairs. Pool semantics identical to the original Stage C
# builder; only the keyword inputs change (corrected index). Pairs via the
# canonical builder with --train-split train and the PRODUCTION catalog for
# candidate text (all candidates are production-eligible after the correction);
# empty candidate text fails closed (hardening commit 8b06ad9).
set -euo pipefail
REPO="${DEMAP_DATA_ROOT:?set DEMAP_DATA_ROOT to the data/artifact root}"
CODE="$REPO/.scratch/demap/paper_v13_scientific_audit/wt_ce_5f06f10"
cd "$REPO"
[[ "$(git -C "$CODE" rev-parse --short=7 HEAD)" == "5f06f10" ]] || { echo "FATAL: worktree moved" >&2; exit 2; }
LS="/lscratch/${SLURM_JOB_ID:?}/v13F2"; mkdir -p "$LS"; export TMPDIR="$LS" TEMP="$LS" TMP="$LS"
export PYTHONPATH="$CODE/src:$CODE/scripts"
PY="$REPO/.venv/bin/python"

FT_ROOT="$REPO/artifacts/final_reranker/crossencoder_fulltrain_v2_eligible"
POOL_DIR="$FT_ROOT/candidate_pool_se20_kwfuzzy10"
POOL="$POOL_DIR/pool_se20_kwfuzzy10.parquet"
PAIRS_DIR="$FT_ROOT/train_pairs"
PROD_CAT="$REPO/data/processed/cadsr_xml_2026-06-18/cde_master_enriched_eval_production_cde_match.parquet"
RECIPE="SN_DEC_DEF_PQT_PV"

echo "=== stage 1: corrected fulltrain pool ==="
srun "$PY" "$REPO/.scratch/demap/paper_v13_scientific_audit/build_ce_pool_train_v2.py" \
  --out-dir "$POOL_DIR" --overwrite

echo "=== stage 1b: pool eligibility assertion (0 off-production rows) ==="
"$PY" - "$POOL" "$PROD_CAT" <<'PYEOF'
import sys, pandas as pd
pool, prod = sys.argv[1], sys.argv[2]
p = pd.read_parquet(pool, columns=["split", "public_id", "is_label"])
pubs = set(pd.read_parquet(prod, columns=["cde_id"])["cde_id"].astype(str)
           .str.split("::").str[0])
off = int((~p["public_id"].astype(str).isin(pubs)).sum())
print(f"[validate] fulltrain pool rows={len(p)} off_production={off}")
assert off == 0, f"{off} off-production rows in the corrected fulltrain pool"
PYEOF

echo "=== stage 2: corrected fulltrain pairs (train split; frozen protocol) ==="
srun "$PY" "$CODE/scripts/build_ce_training_pairs.py" \
  --feature-table "$POOL" \
  --cde-master "$PROD_CAT" \
  --cde-text-recipe "$RECIPE" \
  --train-split train --dev-split val_dev \
  --out-dir "$PAIRS_DIR" --overwrite

echo "=== stage 3: pair eligibility validation ==="
"$PY" - "$PAIRS_DIR" "$PROD_CAT" <<'PYEOF'
import sys, pandas as pd
from pathlib import Path
pairs_dir, prod = Path(sys.argv[1]), sys.argv[2]
prod_ids = set(pd.read_parquet(prod, columns=["cde_id"])["cde_id"].astype(str))
prod_pubs = {c.split("::")[0] for c in prod_ids}
for name in ("train_SN_DEC_DEF_PQT_PV.parquet", "dev_SN_DEC_DEF_PQT_PV.parquet"):
    df = pd.read_parquet(pairs_dir / name)
    pubs = df["cde_id"].astype(str).str.split("::").str[0]
    inel = int((~pubs.isin(prod_pubs)).sum())
    empty = int(df["candidate_text"].fillna("").str.strip().eq("").sum())
    print(f"[validate] {name}: rows={len(df)} off_production={inel} empty_text={empty} "
          f"pos={int((df['label']>0).sum())}")
    assert inel == 0 and empty == 0
print("[validate] corrected fulltrain pairs OK")
PYEOF
echo "[$(date)] F2 DONE"
