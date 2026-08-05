#!/bin/bash
# Run MTP training for every W-vacancy AIMD trajectory folder.
#
# Usage:
#   ./scripts/run_all_trajectories.sh            # up to 3 at a time (default)
#   ./scripts/run_all_trajectories.sh --wait   # same, then wait for all to finish
#   ./scripts/run_all_trajectories.sh --sequential
#
# Environment overrides:
#   MAX_PARALLEL   - concurrent jobs (default: 1 when using MPI train; each job uses MPI_NPROCS)
#   MPI_NPROCS     - ranks per train job (default: 19; see mtp_config.sh)
#   (plus all train_trajectory.sh overrides)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_trajectory.sh"

# shellcheck source=mtp_config.sh
source "${SCRIPT_DIR}/mtp_config.sh"

# Default one concurrent MPI train job so total ranks ≈ MPI_NPROCS (not MAX_PARALLEL * MPI_NPROCS).
MAX_PARALLEL="${MAX_PARALLEL:-1}"

MODE="parallel"
if [[ "${1:-}" == "--sequential" ]]; then
    MODE="sequential"
elif [[ "${1:-}" == "--wait" ]]; then
    MODE="wait"
fi

declare -A PID_TO_TRAJ=()
ALL_PIDS=()
RUNNING=0
LAUNCHED=0

echo "AIMD trajectories: ${AIMD_SOURCES[*]}"
echo "Mode: ${MODE}"
echo "Max parallel jobs: ${MAX_PARALLEL}"
echo

mkdir -p "${PROJECT_ROOT}/logs"

wait_for_slot() {
    local finished_pid=""
    while (( RUNNING >= MAX_PARALLEL )); do
        if wait -n -p finished_pid 2>/dev/null; then
            RUNNING=$((RUNNING - 1))
            echo "Finished ${PID_TO_TRAJ[$finished_pid]} (PID ${finished_pid})"
            unset "PID_TO_TRAJ[$finished_pid]"
        else
            sleep 1
        fi
    done
}

launch_job() {
    local traj="$1"
    local log="${PROJECT_ROOT}/logs/mtp_WC_${traj}.out"
    local pid=""

    nohup nice -n 19 bash "${TRAIN_SCRIPT}" "${traj}" >> "${log}" 2>&1 &
    pid=$!
    ALL_PIDS+=("${pid}")
    PID_TO_TRAJ["${pid}"]="${traj}"
    RUNNING=$((RUNNING + 1))
    LAUNCHED=$((LAUNCHED + 1))
    echo "Started ${traj} (PID ${pid}, running ${RUNNING}/${MAX_PARALLEL}, log: ${log})"
}

for TRAJ in "${AIMD_SOURCES[@]}"; do
    WORKDIR="${PROJECT_ROOT}/${TRAJ}"
    if [[ ! -d "${WORKDIR}" ]]; then
        echo "Skipping ${TRAJ}: folder not found (${WORKDIR})" >&2
        continue
    fi
    if [[ ! -f "${WORKDIR}/OUTCAR" ]]; then
        echo "Skipping ${TRAJ}: OUTCAR missing" >&2
        continue
    fi
    local_size=$(stat -c %s "${WORKDIR}/OUTCAR" 2>/dev/null || stat -f %z "${WORKDIR}/OUTCAR")
    if (( local_size < 1000000 )); then
        echo "Skipping ${TRAJ}: OUTCAR too small (${local_size} bytes)" >&2
        continue
    fi

    if [[ "${MODE}" == "sequential" ]]; then
        echo "=============================="
        echo "Training MTP for ${TRAJ}"
        echo "=============================="
        nice -n 19 bash "${TRAIN_SCRIPT}" "${TRAJ}"
        echo
    else
        wait_for_slot
        launch_job "${TRAJ}"
    fi
done

if [[ "${MODE}" != "sequential" ]]; then
    echo
    echo "Launched ${LAUNCHED} jobs (max ${MAX_PARALLEL} concurrent)."
    echo "Monitor logs in ${PROJECT_ROOT}/logs/"

    if [[ "${MODE}" == "wait" ]]; then
        echo
        echo "Waiting for remaining jobs to finish..."
        finished_pid=""
        while (( RUNNING > 0 )); do
            if wait -n -p finished_pid 2>/dev/null; then
                RUNNING=$((RUNNING - 1))
                echo "Finished ${PID_TO_TRAJ[$finished_pid]} (PID ${finished_pid})"
                unset "PID_TO_TRAJ[$finished_pid]"
            else
                sleep 1
            fi
        done
        echo "All jobs completed."
    else
        echo "Up to ${RUNNING} jobs may still be running in the background."
    fi
else
    echo "All sequential jobs finished."
fi