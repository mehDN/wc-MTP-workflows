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
# retrained with TRAIN_FORCE=1 (resume-skip of an existing pot does not apply).
# New VASP is requested only for unlabeled selections.
#
# Resume: existing iter_NNN/dft_queue.cfg is reused (no re-grade). A
# iter_NNN/merged.ok stamp means that iteration already merged + actually
# retrained (not written if BFGS was skipped). If BFGS finished but train
# postproc died (e.g. same-file cp of calc_errors_train.log), resume seals
# status only — it does not re-run BFGS or rewrite an unchanged train.cfg.
#
# The driver always proceeds to the next iteration after a verified retrain
# until AL_MAX_ITERATIONS or the pool yields zero selections. run.sh restarts
# this script after a crash so remaining iters continue without a manual rerun.
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
echo "Already completed: $(al_completed_iter_count)/${AL_MAX_ITERATIONS} (skip stamped iters, then continue)"
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
    # Do not touch train.cfg mtime when the merge is a no-op. Rewriting would
    # make train_set_current_for_pot fail and trigger a wasted TRAIN_FORCE BFGS
    # after a postproc-only crash.
    if [[ -f "${TRAIN_CFG}" ]] && cmp -s "${merged}" "${TRAIN_CFG}"; then
        echo "train.cfg already contains merged configs; not rewriting"
    else
        cp "${merged}" "${TRAIN_CFG}"
    fi
    mkdir -p "${AL_LABELED_DIR}/merged"
    cp "${src}" "${AL_LABELED_DIR}/merged/${label}_$(basename "${src}")"
}

stamp_al_converged() {
    local reason="$1"
    {
        echo "REASON=${reason}"
        echo "ITER=${LABEL:-}"
        echo "UPDATED=$(date -Iseconds 2>/dev/null || date)"
    } > "${MTP_AL_DIR}/al_converged.ok"
    al_write_progress "${iter:-0}" "${LABEL:-}" "converged"
}

# Merge/retrain for this iteration. Returns 0 if the pot matches train.cfg.
run_al_retrain() {
    local rc=0
    if train_fully_complete "${TRAINED_MTP}" "train" "${MTP_AL_DIR}/logs" "${TRAIN_CFG}"; then
        echo "Pot already fitted on merged train.cfg (n_cfg=$(cfg_n_configurations "${TRAIN_CFG}")); skipping BFGS."
        return 0
    elif train_postproc_pending "${TRAINED_MTP}" "train" "${MTP_AL_DIR}/logs" "${TRAIN_CFG}"; then
        echo "BFGS already finished on merged train.cfg; resuming post-processing only."
        TRAIN_FORCE=0 TRAIN_RESUME_POSTPROC=1 EFFECTIVE_MAX_ITER="${AL_RETRAIN_MAX_ITER}" \
            bash "${SCRIPT_DIR}/train_mtp.sh" || rc=$?
    else
        echo "Retraining MTP on updated train.cfg (TRAIN_FORCE=1, max_iter=${AL_RETRAIN_MAX_ITER}, n_cfg=$(cfg_n_configurations "${TRAIN_CFG}"))"
        TRAIN_FORCE=1 EFFECTIVE_MAX_ITER="${AL_RETRAIN_MAX_ITER}" \
            bash "${SCRIPT_DIR}/train_mtp.sh" || rc=$?
    fi
    if [[ "${rc}" -ne 0 ]] && train_postproc_pending \
            "${TRAINED_MTP}" "train" "${MTP_AL_DIR}/logs" "${TRAIN_CFG}"; then
        echo "WARNING: train_mtp exited rc=${rc}; retrying post-processing only" >&2
        rc=0
        TRAIN_FORCE=0 TRAIN_RESUME_POSTPROC=1 EFFECTIVE_MAX_ITER="${AL_RETRAIN_MAX_ITER}" \
            bash "${SCRIPT_DIR}/train_mtp.sh" || rc=$?
    fi
    return "${rc}"
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
        # Skip only when the pot already matches the merged train set.
        # A 0-byte stamp from the old "skip-complete then write merged.ok" bug
        # is ignored unless a later TRAIN_FORCE fit caught up.
        if train_set_current_for_pot \
            "${TRAINED_MTP}" "${TRAIN_CFG}" "${MTP_AL_DIR}/logs"; then
            echo "Already merged and retrained (${MERGED_OK}); skipping."
            continue
        fi
        echo "Stale merged.ok (pot does not match current train.cfg); will retrain."
        rm -f "${MERGED_OK}"
    fi

    if [[ -s "${DFT_QUEUE}" ]]; then
        echo "Reusing existing selection (skip grade/select): ${DFT_QUEUE}"
    else
        bash "${SCRIPT_DIR}/active_learning.sh" "${CANDIDATE_CFG}" "${LABEL}"
    fi

    if [[ ! -f "${DFT_QUEUE}" ]]; then
        echo "No DFT queue produced; candidate pool is covered."
        stamp_al_converged "no_dft_queue"
        break
    fi

    eval "$(al_cfg_label_status "${DFT_QUEUE}")"
    echo "  Queue: ${N_CFG} selected, ${N_LABELED} already DFT-labeled, ${N_UNLABELED} need VASP"

    if [[ "${N_CFG}" -eq 0 ]]; then
        echo "Zero selections; active learning converged for this candidate pool."
        stamp_al_converged "zero_selections"
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

    if ! run_al_retrain; then
        echo "ERROR: AL retrain failed for ${LABEL}." >&2
        echo "  train.cfg: ${TRAIN_CFG} (n_cfg=$(cfg_n_configurations "${TRAIN_CFG}"))" >&2
        echo "  pot:       ${TRAINED_MTP}" >&2
        echo "  Driver will retry this iteration (skip completed iters) on the next start." >&2
        al_write_progress "$((iter - 1))" "${LABEL}" "retrain_failed"
        exit 1
    fi
    if ! train_set_current_for_pot "${TRAINED_MTP}" "${TRAIN_CFG}" "${MTP_AL_DIR}/logs"; then
        echo "ERROR: AL retrain did not consume the merged train set." >&2
        echo "  train.cfg: ${TRAIN_CFG} (n_cfg=$(cfg_n_configurations "${TRAIN_CFG}"))" >&2
        echo "  pot:       ${TRAINED_MTP}" >&2
        echo "  Refusing to stamp merged.ok — driver will retry this iteration." >&2
        al_write_progress "$((iter - 1))" "${LABEL}" "retrain_unverified"
        exit 1
    fi
    {
        echo "LABEL=${LABEL}"
        echo "TRAIN_N_CFG=$(cfg_n_configurations "${TRAIN_CFG}")"
        echo "POT=${TRAINED_MTP}"
        echo "TRAIN_SET=${TRAIN_CFG}"
        echo "UPDATED=$(date -Iseconds 2>/dev/null || date)"
    } > "${MERGED_OK}"
    echo "Wrote ${MERGED_OK} (retrain verified against current train.cfg)"
    al_write_progress "${iter}" "${LABEL}" "iter_complete"

    if [[ "${N_UNLABELED}" -gt 0 && -s "${UNLABELED_QUEUE}" ]]; then
        echo
        echo ">>> PAUSE: ${N_UNLABELED} selected configs still lack DFT labels"
        echo ">>> Unlabeled queue: ${UNLABELED_QUEUE}"
        echo ">>> Save labeled cfg to ${AL_LABELED_DIR}/ and rerun:"
        echo ">>>   ./scripts/run_active_learning.sh"
        echo
        al_write_progress "${iter}" "${LABEL}" "paused_unlabeled"
        exit "${AL_PAUSE_EXIT}"
    fi

    VALID_LOG="${MTP_AL_DIR}/logs/calc_errors_valid.log"
    if [[ "${AL_STOP_ON_VALID}" == "1" && -f "${VALID_LOG}" ]]; then
        if python3 "${SCRIPT_DIR}/validate_mtp.py" "${VALID_LOG}" \
            --force-rms-max "${VAL_FORCE_RMS_MAX}" \
            --force-mae-max "${VAL_FORCE_MAE_MAX}" \
            --energy-per-atom-max "${VAL_ENERGY_PER_ATOM_MAX}" \
            --stress-rms-max "${VAL_STRESS_RMS_MAX}"; then
            echo "Validation converged at iteration ${iter}."
            stamp_al_converged "validation_passed"
            break
        fi
    fi

    if (( iter < AL_MAX_ITERATIONS )); then
        echo
        echo "Iteration ${iter}/${AL_MAX_ITERATIONS} complete. Starting iteration $((iter + 1))/${AL_MAX_ITERATIONS}..."
        echo
    fi
done

if al_loop_finished; then
    echo "Active learning loop finished ($(al_completed_iter_count)/${AL_MAX_ITERATIONS} iters)."
else
    echo "Active learning loop finished."
fi
