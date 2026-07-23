#!/bin/bash
# Launch MTP training for one W-vacancy AIMD trajectory in the background.
#
# Usage:
#   ./scripts/submit_train.sh vac_W_2500_ML
#
# The job runs as: nohup nice -n 19 train_trajectory.sh <folder>

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <trajectory_folder>" >&2
    exit 1
fi

TRAJ="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_trajectory.sh"
LOG="${PROJECT_ROOT}/logs/mtp_WC_${TRAJ}.out"

mkdir -p "${PROJECT_ROOT}/logs"

nohup nice -n 19 bash "${TRAIN_SCRIPT}" "${TRAJ}" >> "${LOG}" 2>&1 &
PID=$!

echo "Started MTP training for ${TRAJ}"
echo "  PID: ${PID}"
echo "  Log: ${LOG}"