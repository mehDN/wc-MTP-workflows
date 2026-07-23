#!/bin/bash
# Active-learning loop driver (auto-discovers candidate .cfg files).
#
# Usage:
#   ./scripts/run_active_learning.sh
#   ./scripts/run_active_learning.sh [candidate.cfg]   # optional override
#
# Candidate pool (first match wins):
#   1. Optional command-line path
#   2. AL_CANDIDATE_CFG environment variable
#   3. All datasets/candidates/*.cfg (merged if multiple)
#   4. Full AIMD staging trajectories (datasets/initial/staging/aimd_*K.cfg)
#
# Labeled configs for auto-merge (datasets/labeled/*.cfg, newest first):
#   Place VASP-labeled cfg files in datasets/labeled/ to continue the loop.
#
# Each iteration:
#   1. select-add high-uncertainty configs from candidate pool
#   2. DFT-label queued structures -> save to datasets/labeled/
#   3. merge new labels into train.cfg
#   4. retrain MTP and validate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mtp_config.sh
source "${SCRIPT_DIR}/mtp_config.sh"

mkdir -p "${MTP_AL_DIR}/logs" "${AL_CANDIDATES_DIR}" "${AL_LABELED_DIR}"

CANDIDATE_CFG=""
if ! CANDIDATE_CFG="$(resolve_al_candidate_cfg "${1:-}")"; then
    cat >&2 <<EOF
No candidate .cfg pool found.

Drop MD/LAMMPS frames into:
  ${AL_CANDIDATES_DIR}/

Or ensure full AIMD staging configs exist under:
  ${MTP_DATASETS_DIR}/initial/staging/

Optional override:
  AL_CANDIDATE_CFG=/path/to/candidates.cfg $0
EOF
    exit 1
fi

echo "=== Active learning loop (max ${AL_MAX_ITERATIONS} iterations) ==="
echo "Candidates: ${CANDIDATE_CFG}"
echo "Labeled dir: ${AL_LABELED_DIR}/"
echo

for ((iter=1; iter<=AL_MAX_ITERATIONS; iter++)); do
    LABEL="iter_$(printf '%03d' "${iter}")"
    echo "=============================="
    echo "Iteration ${iter}/${AL_MAX_ITERATIONS}: ${LABEL}"
    echo "=============================="

    bash "${SCRIPT_DIR}/active_learning.sh" "${CANDIDATE_CFG}" "${LABEL}"

    DFT_QUEUE="${MTP_AL_DIR}/${LABEL}/dft_queue.cfg"
    if [[ ! -f "${DFT_QUEUE}" ]]; then
        echo "No DFT queue produced; stopping."
        break
    fi

    N_SELECT="$(python3 - "${DFT_QUEUE}" <<'PY'
import sys
count = 0
with open(sys.argv[1]) as fh:
    for line in fh:
        if line.strip() == "BEGIN_CFG":
            count += 1
print(count)
PY
)"
    if [[ "${N_SELECT}" -eq 0 ]]; then
        echo "Zero selections; active learning converged for this candidate pool."
        break
    fi

    LABELED_CFG=""
    if LABELED_CFG="$(resolve_al_labeled_cfg)"; then
        echo "Auto-merging labeled configs from ${LABELED_CFG}"
        MERGED="${MTP_DATASETS_DIR}/initial/train_plus_${LABEL}.cfg"
        python3 "${SCRIPT_DIR}/merge_cfg.py" "${MERGED}" "${TRAIN_CFG}" "${LABELED_CFG}" --dedupe
        cp "${MERGED}" "${TRAIN_CFG}"
        # Avoid re-merging the same file on the next iteration
        mkdir -p "${AL_LABELED_DIR}/merged"
        mv "${LABELED_CFG}" "${AL_LABELED_DIR}/merged/$(basename "${LABELED_CFG}")"
    elif [[ "${AL_AUTO_MERGE:-0}" == "1" ]]; then
        echo "AL_AUTO_MERGE=1 but no labeled cfg found in ${AL_LABELED_DIR}/" >&2
        echo "Label ${DFT_QUEUE} with VASP, save to ${AL_LABELED_DIR}/, then rerun." >&2
        exit 0
    else
        echo
        echo ">>> PAUSE: Label structures in ${DFT_QUEUE} with VASP"
        echo ">>> Save labeled cfg to ${AL_LABELED_DIR}/ and rerun:"
        echo ">>>   ./scripts/run_active_learning.sh"
        echo
        exit 0
    fi

    bash "${SCRIPT_DIR}/train_mtp.sh"

    VALID_LOG="${MTP_AL_DIR}/logs/calc_errors_valid.log"
    if [[ -f "${VALID_LOG}" ]]; then
        if python3 "${SCRIPT_DIR}/validate_mtp.py" "${VALID_LOG}" \
            --force-rms-max "${VAL_FORCE_RMS_MAX}" \
            --force-mae-max "${VAL_FORCE_MAE_MAX}" \
            --energy-per-atom-max "${VAL_ENERGY_PER_ATOM_MAX}" \
            --stress-rms-max "${VAL_STRESS_RMS_MAX}"; then
            echo "Validation converged at iteration ${iter}."
            break
        fi
    fi
done

echo "Active learning loop finished."