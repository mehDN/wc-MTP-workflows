#!/bin/bash
# Run one active-learning iteration: grade candidates, select for DFT, retrain.
#
# Usage:
#   ./scripts/active_learning.sh <candidate.cfg> [iteration_label]
#
# Prerequisite: trained MTP and train.cfg must exist.
# After this script, label selected configs with DFT, append to train.cfg,
# then call again or use run_active_learning.sh for the full loop.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <candidate.cfg> [iteration_label]" >&2
    exit 1
fi

CANDIDATE_CFG="$1"
ITER_LABEL="${2:-iter}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mtp_config.sh
source "${SCRIPT_DIR}/mtp_config.sh"

ITER_DIR="${MTP_AL_DIR}/${ITER_LABEL}"
mkdir -p "${ITER_DIR}" "${MTP_AL_DIR}/logs"

if [[ ! -f "${TRAINED_MTP}" ]]; then
    echo "Trained MTP not found: ${TRAINED_MTP}" >&2
    echo "Run ./scripts/train_mtp.sh first." >&2
    exit 1
fi

if [[ ! -f "${TRAIN_CFG}" ]]; then
    echo "Training set not found: ${TRAIN_CFG}" >&2
    exit 1
fi

if [[ ! -f "${CANDIDATE_CFG}" ]]; then
    echo "Candidate pool not found: ${CANDIDATE_CFG}" >&2
    exit 1
fi

GRADED_CFG="${ITER_DIR}/graded.cfg"
SELECTED_CFG="${ITER_DIR}/selected.cfg"
DFT_QUEUE="${ITER_DIR}/dft_queue.cfg"
ALS_ITER="${ITER_DIR}/state.als"

echo "=== WC active learning iteration: ${ITER_LABEL} ==="
echo "Grade threshold: ${AL_SELECT_THRESHOLD} (add configs above this)"
echo "MTP:        ${TRAINED_MTP}"
echo "Train set:  ${TRAIN_CFG}"
echo "Candidates: ${CANDIDATE_CFG}"
echo "Parallel:   nice -n ${NICE_N} ${MPIRUN} -np ${MPI_NPROCS}"
echo

echo "[1/3] Calculating maxvol grades (np=${MPI_NPROCS})"
run_mlp calc-grade "${TRAINED_MTP}" "${TRAIN_CFG}" "${CANDIDATE_CFG}" "${GRADED_CFG}" \
    --als-filename="${ALS_ITER}" \
    --init-threshold="${AL_INIT_THRESHOLD}" \
    --select-threshold="${AL_SELECT_THRESHOLD}" \
    --swap-threshold="${AL_SWAP_THRESHOLD}" \
    2>&1 | tee "${ITER_DIR}/calc_grade.log"

echo "[2/3] Selecting configurations to add (gamma / maxvol, np=${MPI_NPROCS})"
SELECT_ARGS=(
    select-add "${TRAINED_MTP}" "${TRAIN_CFG}" "${CANDIDATE_CFG}" "${DFT_QUEUE}"
    --als-filename="${ALS_ITER}"
    --selected-filename="${SELECTED_CFG}"
    --init-threshold="${AL_INIT_THRESHOLD}"
    --select-threshold="${AL_SELECT_THRESHOLD}"
    --swap-threshold="${AL_SWAP_THRESHOLD}"
)

if [[ "${AL_SELECTION_LIMIT}" != "0" ]]; then
    SELECT_ARGS+=(--selection-limit="${AL_SELECTION_LIMIT}")
fi

run_mlp "${SELECT_ARGS[@]}" \
    2>&1 | tee "${ITER_DIR}/select_add.log"

# Persist ALS state for subsequent iterations
cp "${ALS_ITER}" "${ALS_FILE}" 2>/dev/null || true

echo "[3/3] Summary"
if [[ -f "${DFT_QUEUE}" ]]; then
    N_SELECT="$(python3 - "${DFT_QUEUE}" <<'PY'
import sys
path = sys.argv[1]
count = 0
with open(path) as fh:
    for line in fh:
        if line.strip() == "BEGIN_CFG":
            count += 1
print(count)
PY
)"
    echo "  Selected for DFT labeling: ${N_SELECT} configs"
    echo "  DFT queue:  ${DFT_QUEUE}"
    echo "  Selected:   ${SELECTED_CFG}"
    echo
    echo "Next steps:"
    echo "  1. Run VASP on structures in ${DFT_QUEUE}"
    echo "  2. Convert OUTCARs and merge into ${TRAIN_CFG}"
    echo "  3. Retrain: ./scripts/train_mtp.sh"
else
    echo "  No new configurations selected (model may be converged for this pool)."
fi