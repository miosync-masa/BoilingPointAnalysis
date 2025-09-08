#!/usr/bin/env bash
# Complete MD Pipeline - FAST VERSION (Skip Validation Only)
set -uo pipefail

########################
# User Configuration
########################

GMX=${GMX:-gmx}
GPU_ID=${GPU_ID:-0}
NT_CPU=${NT_CPU:-8}

# Temperature sets (°C)
H2O_TEMPS_TIP3P="95 98 100 102 105"
H2O_TEMPS_SPCE="105 108 109 110 111"
H2O_TEMPS_TIP4P="97 99 100 101 103"

# Simulation parameters
NREP=5
BULK_NS=1.0
SLAB_NS=4.0
ANALYSIS_SKIP_NS=0.5
ANALYSIS_TOT_NS=4.0

# Directory structure
ROOT="$(pwd)"
OUT_SIM="$ROOT/sim_out"
OUT_RES="$ROOT/results"
LOG_DIR="$ROOT/logs"
mkdir -p "$OUT_SIM" "$OUT_RES" "$LOG_DIR"

# MDP golden templates
MDP_WF_SLAB="$ROOT/mdp_gold/water_flex_slab_4ns.mdp.gold"
MDP_WR_SLAB="$ROOT/mdp_gold/water_rigid_slab_4ns.mdp.gold"
MDP_WF_BULK="$ROOT/mdp_gold/water_flex_bulk_1ns.mdp.gold"
MDP_WR_BULK="$ROOT/mdp_gold/water_rigid_bulk_1ns.mdp.gold"

# Utility Functions
function K() {
  echo "scale=2; $1 + 273.15" | bc
}

function log_msg() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline.log"
}

# Simulation Function
function run_one() {
  local species="$1"
  local model="$2"
  local phase="$3"
  local tempC="$4"
  local rep="$5"

  local Tref=$(K "$tempC")
  local base="$OUT_SIM/$species/$model/$phase/${tempC}C/rep$rep"
  mkdir -p "$base"
  cd "$base"

  local COMMON="$ROOT/common/$species/$model"
  cp "$COMMON/conf_${phase}.gro" conf.gro
  cp "$COMMON/topol_${phase}.top" topol.top

  local MDP_GOLD=""
  if [[ "$model" == "TIP4P2005" ]]; then
    [[ "$phase" == "slab" ]] && MDP_GOLD="$MDP_WR_SLAB" || MDP_GOLD="$MDP_WR_BULK"
  else
    [[ "$phase" == "slab" ]] && MDP_GOLD="$MDP_WF_SLAB" || MDP_GOLD="$MDP_WF_BULK"
  fi

  sed "s/__REF_T__/${Tref}/g" "$MDP_GOLD" > nvt.mdp

  log_msg "Preparing $model $phase ${tempC}C rep${rep}..."
  $GMX grompp -f nvt.mdp -c conf.gro -p topol.top -o nvt.tpr -maxwarn 1 &>> grompp.log

  log_msg "Starting MD: $model $phase ${tempC}C rep${rep} on GPU ${GPU_ID}..."
  local start_time=$(date +%s)

  $GMX mdrun -deffnm nvt \
    -nb gpu -pme gpu \
    -gpu_id $GPU_ID \
    -ntmpi 1 -ntomp $NT_CPU \
    -cpt 10 -append \
    &>> mdrun.log

  local end_time=$(date +%s)
  local elapsed=$((end_time - start_time))

  log_msg "Completed: $model $phase ${tempC}C rep${rep} (${elapsed}s)"
  cd "$ROOT"
}

# Analysis Function
function analyze_one() {
  local species="$1"
  local model="$2"
  local phase="$3"
  local tempC="$4"

  local dir="$OUT_SIM/$species/$model/$phase/${tempC}C"
  local out="$OUT_RES/$species/$model/$phase/${tempC}C"
  mkdir -p "$out"

  log_msg "Analyzing $model $phase ${tempC}C..."

  if [[ -f "$ROOT/analysis/analyze_alpha.py" ]]; then
    python3 "$ROOT/analysis/analyze_alpha.py" \
      --simdir "$dir" \
      --model "$model" \
      --temperature "$tempC" \
      --skip "${ANALYSIS_SKIP_NS}" \
      --total "${BULK_NS}" \
      --out "$out/alpha_summary.json" \
      &>> "$LOG_DIR/analysis.log"
  fi

  if [[ "$phase" == "slab" ]] && [[ -f "$ROOT/analysis/analyze_slab.py" ]]; then
    python3 "$ROOT/analysis/analyze_slab.py" \
      --simdir "$dir" \
      --tempC "$tempC" \
      --skip "${ANALYSIS_SKIP_NS}" \
      --total "${SLAB_NS}" \
      --out "$out/slab_summary.json" \
      &>> "$LOG_DIR/analysis.log"
  fi
}

# Main Execution
log_msg "=== Complete MD Pipeline Started (FAST VERSION) ==="
log_msg "GPU ID: $GPU_ID"
log_msg "Output: $OUT_SIM"
log_msg "Results: $OUT_RES"

declare -A H2O_MODELS=(
  [TIP3P]="$H2O_TEMPS_TIP3P"
  [SPC_E]="$H2O_TEMPS_SPCE"
  [TIP4P2005]="$H2O_TEMPS_TIP4P"
)

TOTAL_JOBS=0
for model in "${!H2O_MODELS[@]}"; do
  temps="${H2O_MODELS[$model]}"
  for T in $temps; do
    TOTAL_JOBS=$((TOTAL_JOBS + NREP * 2))
  done
done

log_msg "Total jobs to run: $TOTAL_JOBS"
CURRENT_JOB=0

for model in "${!H2O_MODELS[@]}"; do
  temps="${H2O_MODELS[$model]}"

  for T in $temps; do
    for rep in $(seq 1 $NREP); do
      CURRENT_JOB=$((CURRENT_JOB + 1))
      log_msg "[$CURRENT_JOB/$TOTAL_JOBS] Running $model bulk ${T}C rep${rep}..."
      run_one "H2O" "$model" "bulk" "$T" "$rep"

      CURRENT_JOB=$((CURRENT_JOB + 1))
      log_msg "[$CURRENT_JOB/$TOTAL_JOBS] Running $model slab ${T}C rep${rep}..."
      run_one "H2O" "$model" "slab" "$T" "$rep"
    done

    analyze_one "H2O" "$model" "bulk" "$T"
    analyze_one "H2O" "$model" "slab" "$T"
  done
done

log_msg "=== All Simulations Complete ==="

if [[ -f "$ROOT/analysis/generate_report.py" ]]; then
  python3 "$ROOT/analysis/generate_report.py" \
    --results "$OUT_RES" \
    --output "$ROOT/final_report.pdf" \
    &>> "$LOG_DIR/analysis.log"
  log_msg "Final report generated: final_report.pdf"
fi

total_time=$(grep "Completed:" "$LOG_DIR/pipeline.log" 2>/dev/null | awk '{sum+=$NF} END {print sum}')
if [[ -n "$total_time" ]]; then
  log_msg "Total computation time: ${total_time}s"
  log_msg "Average time per job: $(echo "scale=2; $total_time / $TOTAL_JOBS" | bc)s"
fi

log_msg "=== Pipeline Finished Successfully ==="
echo "Results available in: $OUT_RES"
echo "Logs available in: $LOG_DIR"
