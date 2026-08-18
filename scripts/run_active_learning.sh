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
#   4. Full AIMD staging trajectories (already DFT-labeled leftover frames)
#
# If selected configs already have Energy + forces (typical when the pool is
# unused AIMD/OUTCAR frames), they are merged into train.cfg and the MTP is
# retrained. New VASP is requested only for unlabeled selections.
#
# Resume: existing iter_NNN/dft_queue.cfg is reused (no re-grade). A
# iter_NNN/merged.ok stamp means that iteration already merged + retrained.
#
# Labeled configs for unlabeled queues (datasets/labeled/*.cfg, newest first):
#   Place VASP-labeled cfg files there to continue after a pause.
#
# Exit codes:
#   0  loop finished (converged or hit max iterations)
#   10 paused — unlabeled selections still need DFT (AL_PAUSE_EXIT)

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
# Prefer high-force-error subset from refine as extra candidate material
HIGH_ERR_CFG="${MTP_AL_DIR}/refine/high_force_error.cfg"
if [[ "${AL_PREFER_HIGH_FORCE_ERROR}" == "1" && -f "${HIGH_ERR_CFG}" && -s "${HIGH_ERR_CFG}" ]]; then
    echo "High-force-error subset available: ${HIGH_ERR_CFG}"
    echo "  (already DFT-labeled; merge into train if not already present, or use as AL focus)"
fi
echo

merge_into_train() {
    local src="$1"
    local label="$2"
    local merged="${MTP_DATASETS_DIR}/initial/train_plus_${label}.cfg"

    echo "Merging labeled configs into train.cfg from ${src}"
    python3 "${SCRIPT_DIR}/merge_cfg.py" "${merged}" "${TRAIN_CFG}" "${src}" --dedupe
    cp "${merged}" "${TRAIN_CFG}"
    mkdir -p "${AL_LABELED_DIR}/merged"
    cp "${src}" "${AL_LABELED_DIR}/merged/${label}_$(basename "${src}")"
}

for ((iter=1; iter<=AL_MAX_ITERATIONS; iter++)); do
    LABEL="iter_$(printf '%03d' "${iter}")"
    ITER_DIR="${MTP_AL_DIR}/${LABEL}"
    DFT_QUEUE="${ITER_DIR}/dft_queue.cfg"
    MERGED_OK="${ITER_DIR}/merged.ok"

    echo "=============================="
    echo "Iteration ${iter}/${AL_MAX_ITERATIONS}: ${LABEL}"
    echo "=============================="

    if [[ -f "${MERGED_OK}" ]]; then
        echo "Already merged and retrained (${MERGED_OK}); skipping."
        continue
    fi

    if [[ -s "${DFT_QUEUE}" ]]; then
        echo "Reusing existing selection (skip grade/select): ${DFT_QUEUE}"
    else
        bash "${SCRIPT_DIR}/active_learning.sh" "${CANDIDATE_CFG}" "${LABEL}"
    fi

    if [[ ! -f "${DFT_QUEUE}" ]]; then
        echo "No DFT queue produced; stopping."
        break
    fi

    eval "$(al_cfg_label_status "${DFT_QUEUE}")"
    echo "  Queue: ${N_CFG} selected, ${N_LABELED} already DFT-labeled, ${N_UNLABELED} need VASP"

    if [[ "${N_CFG}" -eq 0 ]]; then
        echo "Zero selections; active learning converged for this candidate pool."
        break
    fi

    LABELED_SRC=""
    UNLABELED_QUEUE="${ITER_DIR}/unlabeled_queue.cfg"

    if [[ "${N_LABELED}" -gt 0 && "${N_UNLABELED}" -eq 0 ]]; then
        echo "All selected configs already have Energy + forces (AIMD/OUTCAR); no new VASP."
        LABELED_SRC="${DFT_QUEUE}"
    elif [[ "${N_LABELED}" -gt 0 ]]; then
        LABELED_SRC="${ITER_DIR}/labeled_from_queue.cfg"
        python3 "${SCRIPT_DIR}/cfg_label_status.py" "${DFT_QUEUE}" \
            --extract-labeled "${LABELED_SRC}" \
            --extract-unlabeled "${UNLABELED_QUEUE}"
        echo "Extracted ${N_LABELED} already-labeled configs; ${N_UNLABELED} still need DFT."
    fi

    if [[ -z "${LABELED_SRC}" ]]; then
        if LABELED_SRC="$(resolve_al_labeled_cfg)"; then
            echo "Using newly labeled configs from ${LABELED_SRC}"
        fi
    fi

    if [[ -n "${LABELED_SRC}" ]]; then
        merge_into_train "${LABELED_SRC}" "${LABEL}"
        if [[ "${LABELED_SRC}" == "${AL_LABELED_DIR}"/* && -f "${LABELED_SRC}" ]]; then
            mkdir -p "${AL_LABELED_DIR}/merged"
            mv "${LABELED_SRC}" "${AL_LABELED_DIR}/merged/$(basename "${LABELED_SRC}")"
        fi
    elif [[ "${AL_AUTO_MERGE:-0}" == "1" ]]; then
        echo "AL_AUTO_MERGE=1 but no labeled cfg found in ${AL_LABELED_DIR}/" >&2
        echo "Label ${DFT_QUEUE} with VASP, save to ${AL_LABELED_DIR}/, then rerun." >&2
        exit "${AL_PAUSE_EXIT}"
    else
        echo
        echo ">>> PAUSE: Label structures in ${DFT_QUEUE} with VASP"
        echo ">>> Save labeled cfg to ${AL_LABELED_DIR}/ and rerun:"
        echo ">>>   ./scripts/run_active_learning.sh"
        echo
        exit "${AL_PAUSE_EXIT}"
    fi

    bash "${SCRIPT_DIR}/train_mtp.sh"
    : > "${MERGED_OK}"

    if [[ "${N_UNLABELED}" -gt 0 && -s "${UNLABELED_QUEUE}" ]]; then
        echo
        echo ">>> PAUSE: ${N_UNLABELED} selected configs still lack DFT labels"
        echo ">>> Unlabeled queue: ${UNLABELED_QUEUE}"
        echo ">>> Save labeled cfg to ${AL_LABELED_DIR}/ and rerun:"
        echo ">>>   ./scripts/run_active_learning.sh"
        echo
        exit "${AL_PAUSE_EXIT}"
    fi

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
