#!/bin/bash
# Unified MTP workflow for WC W-vacancy reconstruction (alpha-WC -> planar C-C dimer).
#
# Runs the full pipeline in one command:
#   1. Prepare MTP template (level 20, W-C, 6.0 A cutoff)
#   2. Build training dataset from W-vacancy AIMD OUTCARs + sources.conf
#   3. Train MTP (linear fit on fixed basis)
#   4. Validate against convergence thresholds
#   5. (optional) Active-learning loop on MD/exploratory candidate pool
#   6. (optional) Per-trajectory MTP fits
#
# Usage:
#   cd /slask/mehdin/dynamics/wc
#   ./run.sh
#
# With active learning (auto-finds datasets/candidates/*.cfg):
#   ./run.sh --al
#   ./run.sh --al --candidates datasets/candidates/md_frames.cfg
#
# Skip steps when resuming:
#   ./run.sh --skip-dataset          # train + validate only
#   ./run.sh --skip-train            # dataset + validate only
#   ./run.sh --only dataset          # single step
#
# Environment overrides: see scripts/mtp_config.sh
#   MTP_LEVEL=22 MTP_MAX_DIST=6.5 FORCE_WEIGHT=0.15 ./run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="${SCRIPT_DIR}/scripts"
# shellcheck source=scripts/mtp_config.sh
source "${SCRIPTS}/mtp_config.sh"

RUN_LOG="${MTP_LOGS_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${MTP_LOGS_DIR}" "${MTP_AL_DIR}" "${MTP_DATASETS_DIR}/candidates"

DO_DATASET=1
DO_TRAIN=1
DO_VALIDATE=1
DO_AL=0
DO_PER_TRAJ=0
ONLY_STEP=""
CANDIDATES_CFG=""
LABELED_CFG=""

usage() {
    cat <<'EOF'
Unified MTP workflow for WC W-vacancy reconstruction.

Usage:
  ./run.sh [options]

Default (no options): dataset -> train -> validate

Options:
  --al                 After train, run active-learning loop (auto-finds candidate .cfg)
  --candidates FILE    Override candidate pool for active learning
  --labeled FILE       Merge FILE into train.cfg before retrain
  --per-traj           Also train per-AIMD-trajectory MTPs (vac_W_* folders)
  --skip-dataset       Use existing datasets/initial/train.cfg
  --skip-train         Build dataset only, do not train
  --skip-validate      Skip error-threshold check
  --only STEP          Run one step: template | dataset | train | validate | al | per-traj
  -h, --help           Show this help

Examples:
  ./run.sh
  ./run.sh --al
  ./run.sh --skip-dataset
  MTP_LEVEL=22 MTP_MAX_DIST=6.5 ./run.sh

Active-learning loop (after initial fit on bulk + relaxed defect configs):
  1. Run MTP relaxation/MD on unreconstructed W-vacancy supercell
  2. Dump frames to datasets/candidates/
  3. ./run.sh --skip-dataset --al
  4. Label selected configs with VASP (PBE, ENCUT 450-500 eV), save to datasets/labeled/
  5. Repeat until reconstructed dimer is stable ground state (~3-4 eV lowering)

After labeling, continue with:
  ./run.sh --skip-dataset --al
EOF
}

log() {
    echo "$@" | tee -a "${RUN_LOG}"
}

step_header() {
    local n="$1"
    local title="$2"
    log ""
    log "============================================================"
    log " Step ${n}: ${title}"
    log "============================================================"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --al)
            DO_AL=1
            shift
            ;;
        --candidates)
            CANDIDATES_CFG="${2:?--candidates requires a file path}"
            DO_AL=1
            shift 2
            ;;
        --labeled)
            LABELED_CFG="${2:?--labeled requires a file path}"
            shift 2
            ;;
        --per-traj|--per-temp)
            DO_PER_TRAJ=1
            shift
            ;;
        --skip-dataset)
            DO_DATASET=0
            shift
            ;;
        --skip-train)
            DO_TRAIN=0
            shift
            ;;
        --skip-validate)
            DO_VALIDATE=0
            shift
            ;;
        --only)
            ONLY_STEP="${2:?--only requires a step name}"
            DO_DATASET=0
            DO_TRAIN=0
            DO_VALIDATE=0
            DO_AL=0
            DO_PER_TRAJ=0
            case "${ONLY_STEP}" in
                template) DO_DATASET=0 ;;
                dataset)  DO_DATASET=1 ;;
                train)    DO_TRAIN=1 ;;
                validate) DO_VALIDATE=1 ;;
                al)       DO_AL=1 ;;
                per-traj|per-temp) DO_PER_TRAJ=1 ;;
                *)
                    echo "Unknown step: ${ONLY_STEP}" >&2
                    echo "Valid steps: template dataset train validate al per-traj" >&2
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -x "${MLP}" ]]; then
    echo "MLIP not found: ${MLP}" >&2
    exit 1
fi

log "WC MTP workflow started: $(date)"
log "Project: ${MTP_PROJECT_ROOT}"
log "Level:   ${MTP_LEVEL}  Cutoff: ${MTP_MIN_DIST}-${MTP_MAX_DIST} A  RB: ${MTP_RADIAL_BASIS_SIZE}"
log "Log:     ${RUN_LOG}"

# --- Step 1: Template ---
if [[ -z "${ONLY_STEP}" || "${ONLY_STEP}" == "template" || "${DO_DATASET}" == "1" || "${DO_TRAIN}" == "1" ]]; then
    step_header 1 "Prepare MTP template (level ${MTP_LEVEL}, W-C)"
    ensure_mtp_template
    log "Template: ${MTP_TEMPLATE}"
fi
[[ "${ONLY_STEP}" == "template" ]] && exit 0

# --- Step 2: Dataset ---
if [[ "${DO_DATASET}" == "1" ]]; then
    step_header 2 "Build initial training dataset"
    bash "${SCRIPTS}/build_initial_dataset.sh" 2>&1 | tee -a "${RUN_LOG}"
    if [[ ! -f "${TRAIN_CFG}" ]]; then
        echo "Dataset build failed: ${TRAIN_CFG} not created" >&2
        exit 1
    fi
    N_CFG="$(python3 - "${TRAIN_CFG}" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1]) if line.strip() == "BEGIN_CFG"))
PY
)"
    log "Training set: ${TRAIN_CFG} (${N_CFG} configurations)"
fi
[[ "${ONLY_STEP}" == "dataset" ]] && exit 0

# --- Merge labeled configs (active-learning augmentation) ---
if [[ -n "${LABELED_CFG}" ]]; then
    step_header "2b" "Merge new DFT labels into training set"
    if [[ ! -f "${LABELED_CFG}" ]]; then
        echo "Labeled cfg not found: ${LABELED_CFG}" >&2
        exit 1
    fi
    MERGED="${MTP_DATASETS_DIR}/initial/train_merged.cfg"
    python3 "${SCRIPTS}/merge_cfg.py" "${MERGED}" "${TRAIN_CFG}" "${LABELED_CFG}" --dedupe \
        | tee -a "${RUN_LOG}"
    cp "${MERGED}" "${TRAIN_CFG}"
    log "Updated training set: ${TRAIN_CFG}"
fi

# --- Step 3: Train ---
if [[ "${DO_TRAIN}" == "1" ]]; then
    step_header 3 "Train MTP"
    bash "${SCRIPTS}/train_mtp.sh" 2>&1 | tee -a "${RUN_LOG}"
fi
[[ "${ONLY_STEP}" == "train" ]] && exit 0

# --- Step 4: Validate ---
if [[ "${DO_VALIDATE}" == "1" ]]; then
    step_header 4 "Validate MTP errors"
    ERR_LOG="${MTP_AL_DIR}/logs/calc_errors_valid.log"
    [[ -f "${ERR_LOG}" ]] || ERR_LOG="${MTP_AL_DIR}/logs/calc_errors_train.log"
    if [[ -f "${ERR_LOG}" ]]; then
        VALID_ARGS=(
            "${ERR_LOG}"
            --force-rms-max "${VAL_FORCE_RMS_MAX}"
            --force-mae-max "${VAL_FORCE_MAE_MAX}"
            --energy-per-atom-max "${VAL_ENERGY_PER_ATOM_MAX}"
            --stress-rms-max "${VAL_STRESS_RMS_MAX}"
        )
        if python3 "${SCRIPTS}/validate_mtp.py" "${VALID_ARGS[@]}" 2>&1 | tee -a "${RUN_LOG}"; then
            log "Validation: PASSED"
        else
            log "Validation: NOT converged (add data via active learning or increase MTP_LEVEL if defect forces > ${VAL_DEFECT_FORCE_RMS_MAX} eV/A)"
        fi
    else
        log "Validation skipped: no errors log found"
    fi
fi
[[ "${ONLY_STEP}" == "validate" ]] && exit 0

# --- Step 5: Active learning ---
if [[ "${DO_AL}" == "1" ]]; then
    step_header 5 "Active learning (grade threshold ${AL_SELECT_THRESHOLD})"
    bash "${SCRIPTS}/run_active_learning.sh" ${CANDIDATES_CFG:+"${CANDIDATES_CFG}"} \
        2>&1 | tee -a "${RUN_LOG}"
fi
[[ "${ONLY_STEP}" == "al" ]] && exit 0

# --- Step 6: Per-trajectory fits ---
if [[ "${DO_PER_TRAJ}" == "1" ]]; then
    step_header 6 "Per-AIMD-trajectory MTP training"
    bash "${SCRIPTS}/run_all_trajectories.sh" --wait 2>&1 | tee -a "${RUN_LOG}"
fi
[[ "${ONLY_STEP}" == "per-traj" || "${ONLY_STEP}" == "per-temp" ]] && exit 0

log ""
log "Workflow complete: $(date)"
log "  Trained MTP: ${TRAINED_MTP}"
log "  Training set: ${TRAIN_CFG}"
log "  Full log:     ${RUN_LOG}"