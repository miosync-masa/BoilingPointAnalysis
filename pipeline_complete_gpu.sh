#!/usr/bin/env bash
# Complete MD Pipeline for Water Models Boiling Point Analysis
# GPU-accelerated Sequential Execution with Integrated Analysis
# Version: 2.0 (Frozen Spec v1.0 compliant)
# Authors: Iizumi M. & Iizumi T. (2025)

set -euo pipefail

########################
# User Configuration
########################

# GROMACS command and GPU settings
GMX=${GMX:-gmx}
GPU_ID=${GPU_ID:-0}
NT_CPU=${NT_CPU:-8}  # CPU threads for non-GPU tasks

# Temperature sets (°C) - Frozen Spec v1.0
H2O_TEMPS_TIP3P="95 98 100 102 105"
H2O_TEMPS_SPCE="105 108 109 110 111"
H2O_TEMPS_TIP4P="97 99 100 101 103"

# Organic solvents (for future extension)
EtOH_TEMPS="76 78 79 80 82"           # Boiling point ~78.4°C
DECANE_TEMPS="170 172 174 176 178"    # Boiling point ~174.1°C

# Simulation parameters
NREP=5                  # Number of replicas
BULK_NS=1.0            # Bulk simulation time (ns)
SLAB_NS=4.0            # Slab simulation time (ns)

# Analysis parameters
ANALYSIS_SKIP_NS=0.5   # Skip initial equilibration
ANALYSIS_TOT_NS=4.0    # Total analysis window

# Directory structure
ROOT="$(pwd)"
OUT_SIM="$ROOT/sim_out"
OUT_RES="$ROOT/results"
LOG_DIR="$ROOT/logs"
mkdir -p "$OUT_SIM" "$OUT_RES" "$LOG_DIR"

# MDP golden templates (Frozen Spec v1.0)
MDP_WF_SLAB="$ROOT/mdp_gold/water_flex_slab_4ns.mdp.gold"
MDP_WR_SLAB="$ROOT/mdp_gold/water_rigid_slab_4ns.mdp.gold"
MDP_WF_BULK="$ROOT/mdp_gold/water_flex_bulk_1ns.mdp.gold"
MDP_WR_BULK="$ROOT/mdp_gold/water_rigid_bulk_1ns.mdp.gold"
MDP_ORG_SLAB="$ROOT/mdp_gold/organic_slab_4ns.mdp.gold"
MDP_ORG_BULK="$ROOT/mdp_gold/organic_bulk_1ns.mdp.gold"

#################################
# Frozen Spec v1.0 Validation
#################################

req_keys=(
  "cutoff-scheme=Verlet"
  "rlist=1.2" "rcoulomb=1.2" "rvdw=1.2"
  "coulombtype=PME" "pme-order=4" "fourierspacing=0.12"
  "DispCorr=EnerPres" "pcoupl=no" "pbc=xyz" 
  "tcoupl=nose-hoover" "tau-t=0.5"
)
req_slab_keys=("ewald-geometry=3dc" "epsilon-surface=0")

function check_mdp() {
  local file="$1"
  local miss=0
  
  for kv in "${req_keys[@]}"; do
    local k=${kv%%=*}
    local v=${kv#*=}
    local gv=$(grep -E "^[[:space:]]*$k[[:space:]]*=" "$file" 2>/dev/null | tail -1 | awk -F= '{gsub(/[[:space:]]/,"",$2);print $2}')
    [[ "$gv" == "$v" ]] || { echo "[NG] $k=$gv (need $v) in $file"; miss=1; }
  done
  
  if grep -q "slab" <<< "$file"; then
    for kv in "${req_slab_keys[@]}"; do
      local k=${kv%%=*}
      local v=${kv#*=}
      local gv=$(grep -E "^[[:space:]]*$k[[:space:]]*=" "$file" 2>/dev/null | tail -1 | awk -F= '{gsub(/[[:space:]]/,"",$2);print $2}')
      [[ "$gv" == "$v" ]] || { echo "[NG] $k=$gv (need $v) in $file"; miss=1; }
    done
  fi
  
  [[ $miss -eq 0 ]] || { echo "Frozen Spec violation in $file"; exit 2; }
}

# Validate all MDP templates
echo "=== Validating MDP templates (Frozen Spec v1.0) ==="
for f in "$MDP_WF_SLAB" "$MDP_WR_SLAB" "$MDP_WF_BULK" "$MDP_WR_BULK"; do
  [[ -f "$f" ]] || { echo "Missing MDP gold: $f"; exit 2; }
  check_mdp "$f"
done
echo "All MDP templates validated successfully"

#################################
# Utility Functions
#################################

# Convert Celsius to Kelvin
function K() { 
  echo "scale=2; $1 + 273.15" | bc
}

# Log with timestamp
function log_msg() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline.log"
}

#################################
# Simulation Function
#################################

function run_one() {
  local species="$1"     # H2O / EtOH / Decane
  local model="$2"       # TIP3P / SPC_E / TIP4P2005 / OPLS-AA
  local phase="$3"       # slab / bulk
  local tempC="$4"       # Temperature in Celsius
  local rep="$5"         # Replica number

  local Tref=$(K "$tempC")
  local base="$OUT_SIM/$species/$model/$phase/${tempC}C/rep$rep"
  mkdir -p "$base"
  cd "$base"

  # Copy input files
  local COMMON="$ROOT/common/$species/$model"
  cp "$COMMON/conf_${phase}.gro" conf.gro
  cp "$COMMON/topol.top" topol.top
  
  # Copy additional .itp files if present
  shopt -s nullglob
  for itp in "$COMMON"/*.itp "$COMMON"/*.prm; do 
    cp "$itp" ./
  done
  shopt -u nullglob

  # Select appropriate MDP template
  local MDP_GOLD=""
  if [[ "$species" == "H2O" ]]; then
    if [[ "$model" == "TIP4P2005" ]]; then
      [[ "$phase" == "slab" ]] && MDP_GOLD="$MDP_WR_SLAB" || MDP_GOLD="$MDP_WR_BULK"
    else
      [[ "$phase" == "slab" ]] && MDP_GOLD="$MDP_WF_SLAB" || MDP_GOLD="$MDP_WF_BULK"
    fi
  else
    [[ "$phase" == "slab" ]] && MDP_GOLD="$MDP_ORG_SLAB" || MDP_GOLD="$MDP_ORG_BULK"
  fi

  # Generate MDP with temperature
  sed "s/__REF_T__/${Tref}/g" "$MDP_GOLD" > nvt.mdp

  # Run grompp
  log_msg "Preparing $model $phase ${tempC}C rep${rep}..."
  $GMX grompp -f nvt.mdp -c conf.gro -p topol.top -o nvt.tpr -maxwarn 1 &>> grompp.log

  # Run MD on GPU
  log_msg "Starting MD: $model $phase ${tempC}C rep${rep} on GPU ${GPU_ID}..."
  local start_time=$(date +%s)
  
  # GPU execution (water doesn't need -bonded gpu)
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

#################################
# Analysis Function
#################################

function analyze_one() {
  local species="$1"
  local model="$2"
  local phase="$3"
  local tempC="$4"
  
  local dir="$OUT_SIM/$species/$model/$phase/${tempC}C"
  local out="$OUT_RES/$species/$model/$phase/${tempC}C"
  mkdir -p "$out"

  log_msg "Analyzing $model $phase ${tempC}C..."

  # Alpha coefficient analysis
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

  # Slab-specific analysis
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

#################################
# Main Execution
#################################

log_msg "=== Complete MD Pipeline Started ==="
log_msg "GPU ID: $GPU_ID"
log_msg "Output: $OUT_SIM"
log_msg "Results: $OUT_RES"

# Define water models
declare -A H2O_MODELS=(
  [TIP3P]="$H2O_TEMPS_TIP3P"
  [SPC_E]="$H2O_TEMPS_SPCE"
  [TIP4P2005]="$H2O_TEMPS_TIP4P"
)

# Count total jobs
TOTAL_JOBS=0
for model in "${!H2O_MODELS[@]}"; do
  temps="${H2O_MODELS[$model]}"
  for T in $temps; do
    TOTAL_JOBS=$((TOTAL_JOBS + NREP * 2))  # bulk + slab
  done
done

log_msg "Total jobs to run: $TOTAL_JOBS"
CURRENT_JOB=0

# Sequential execution (GPU-accelerated)
for model in "${!H2O_MODELS[@]}"; do
  temps="${H2O_MODELS[$model]}"
  
  for T in $temps; do
    for rep in $(seq 1 $NREP); do
      # Bulk simulation
      ((CURRENT_JOB++))
      log_msg "[$CURRENT_JOB/$TOTAL_JOBS] Running bulk..."
      run_one "H2O" "$model" "bulk" "$T" "$rep"
      
      # Slab simulation
      ((CURRENT_JOB++))
      log_msg "[$CURRENT_JOB/$TOTAL_JOBS] Running slab..."
      run_one "H2O" "$model" "slab" "$T" "$rep"
    done
    
    # Analyze immediately after each temperature
    analyze_one "H2O" "$model" "bulk" "$T"
    analyze_one "H2O" "$model" "slab" "$T"
  done
done

#################################
# Final Summary
#################################

log_msg "=== All Simulations Complete ==="

# Generate summary report
if [[ -f "$ROOT/analysis/generate_report.py" ]]; then
  python3 "$ROOT/analysis/generate_report.py" \
    --results "$OUT_RES" \
    --output "$ROOT/final_report.pdf" \
    &>> "$LOG_DIR/analysis.log"
  log_msg "Final report generated: final_report.pdf"
fi

# Statistics
total_time=$(grep "Completed:" "$LOG_DIR/pipeline.log" | awk '{sum+=$NF} END {print sum}')
log_msg "Total computation time: ${total_time}s"
log_msg "Average time per job: $(echo "scale=2; $total_time / $TOTAL_JOBS" | bc)s"

log_msg "=== Pipeline Finished Successfully ==="
echo "Results available in: $OUT_RES"
echo "Logs available in: $LOG_DIR"
