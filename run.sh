#!/bin/bash
# Unified MTP workflow for WC W-vacancy reconstruction (alpha-WC -> planar C-C dimer).
#
# Runs the full pipeline in one command:
#   1. Prepare MTP template (level 20, W-C, 6.0 A cutoff)
#   2. Build training dataset from W-vacancy AIMD OUTCARs + sources.conf
#   3. Train MTP (linear fit on fixed basis)
#   4. Validate against convergence thresholds
#   5. (optional) Active-learning loop: leftover AIMD frames are already
#      DFT-labeled and are merged + retrained (TRAIN_FORCE=1; existing pot
#      is not treated as done). New VASP only if unlabeled.
#   6. (optional) Per-trajectory MTP fits
#
# Resume (default):
#   On re-run after a crash/error, ./run.sh skips steps already completed
#   (dataset present, pot trained + errors logged, etc.) and continues from
#   the failed step. State: active_learning/workflow_state.env
#   Force a full re-run:  ./run.sh --fresh
#
# Usage:
#   cd /slask/mehdin/dynamics/wc_parallel_+
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
#   TRAIN_FRESH=1 ./run.sh --skip-dataset   # ignore existing pots; random init
# Existing fitted MTPs under active_learning/ are auto-discovered and continued.

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
DO_REFINE=0
DO_AL=0
DO_PER_TRAJ=0
ONLY_STEP=""
CANDIDATES_CFG=""
LABELED_CFG=""

# Track whether the user explicitly forced step selection (disables auto-skip
# for those steps). --only / --fresh also disable auto-resume planning.
USER_SET_DATASET=0
USER_SET_TRAIN=0
USER_SET_VALIDATE=0
USER_SET_REFINE=0
FRESH_RUN=0
# AUTO_RESUME from mtp_config (default 1); --fresh / --no-resume set to 0.

usage() {
    cat <<'EOF'
Unified MTP workflow for WC W-vacancy reconstruction.

Usage:
  ./run.sh [options]

Default (no options): dataset -> train -> validate
  If validation fails and AUTO_REFINE=1 (default): run refine sequence
  (more BFGS + rescale, force-weighted retrain, high-error subset, mindist check).

Resume (default AUTO_RESUME=1):
  Re-running ./run.sh after a crash continues from the step that failed
  (or the next incomplete step), using artifacts + active_learning/workflow_state.env.
  Example: train BFGS finished but calc-errors died → re-run finishes errors,
  then validates (and auto-refines if needed) without rebuilding the dataset
  or re-running BFGS.

Options:
  --al                 After train, run active-learning loop (auto-finds candidate .cfg)
  --candidates FILE    Override candidate pool for active learning
  --labeled FILE       Merge FILE into train.cfg and force-retrain (TRAIN_FORCE=1)
  --refine             Force refine sequence after train (also AUTO_REFINE=1 on validate fail)
  --skip-refine        Do not auto-refine when validation fails
  --per-traj           Also train per-AIMD-trajectory MTPs (vac_W_* folders)
  --skip-dataset       Use existing datasets/initial/train.cfg
  --skip-train         Do not train (validate / refine / al only)
  --skip-validate      Skip error-threshold check
  --only STEP          Run one step: template | dataset | train | validate | refine | al | per-traj
  --fresh              Ignore resume state; run all selected steps from scratch
                       (still continues BFGS from fitted pots unless TRAIN_FRESH=1)
  --no-resume          Same as --fresh for step selection
  --resume             Force auto-resume planning (default)
  -h, --help           Show this help

Examples:
  ./run.sh                         # full pipeline; auto-resumes if prior run failed
  ./run.sh --fresh                 # rebuild dataset + retrain path ignoring resume skips
  ./run.sh --al
  ./run.sh --skip-dataset          # continues from active_learning/WC_L20_trained.mtp if present
  ./run.sh --only refine           # continue BFGS / force retrain from existing pot
  TRAIN_FRESH=1 ./run.sh --skip-dataset   # force random init (ignore trained pots)
  MTP_LEVEL=22 MTP_MAX_DIST=6.5 ./run.sh

Refine sequence (when force RMS fails or BFGS hits step limit):
  0. Filter bad DFT (VASP_not_converged, huge forces, short mindist, E outliers)
  1. Continue BFGS from best pot (reset Hessian; each run rescales) until settled
  2. Force-weighted retrain from best intermediate
  3. Optional high-F fit on high-error subset + full-set polish
  4. Check min cutoff vs mindist; then AL or MTP_LEVEL=22 if still stuck

Active-learning loop (after initial fit on bulk + relaxed defect configs):
  1. Grade the candidate pool (MD frames in datasets/candidates/, or leftover
     already-labeled AIMD frames from datasets/initial/staging/)
  2. If selected configs already have Energy + forces: merge + retrain
     (BFGS is forced; resume-skip of an existing pot does not apply)
  3. Only unlabeled selections need new VASP (PBE, ENCUT 450-500 eV) —
     save those to datasets/labeled/ and rerun ./run.sh --al
  4. Repeat until reconstructed dimer is stable ground state (~3-4 eV lowering)

Resume after a pause or crash:
  ./run.sh --al
  Existing iter_NNN/dft_queue.cfg is reused (no re-grade).
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

# Mark workflow step transitions in workflow_state.env
WF_CURRENT=""
workflow_begin_step() {
    WF_CURRENT="$1"
    workflow_write_state \
        "CURRENT_STEP=${1}" \
        "CURRENT_STATUS=running" \
        "RUN_PID=$$" \
        "WANT_AL=${DO_AL}" \
        "WANT_REFINE=${DO_REFINE}" \
        "WANT_PER_TRAJ=${DO_PER_TRAJ}"
}

workflow_finish_step() {
    local step="$1"
    workflow_write_state \
        "LAST_COMPLETED_STEP=${step}" \
        "CURRENT_STEP=${step}" \
        "CURRENT_STATUS=done" \
        "FAIL_REASON=" \
        "WANT_AL=${DO_AL}" \
        "WANT_REFINE=${DO_REFINE}" \
        "WANT_PER_TRAJ=${DO_PER_TRAJ}"
    WF_CURRENT=""
}

workflow_fail_hook() {
    local rc=$?
    # Do not clobber a clean exit.
    if [[ ${rc} -eq 0 ]]; then
        return 0
    fi
    # AL pause (unlabeled selections need VASP) is not a crash.
    if [[ ${rc} -eq "${AL_PAUSE_EXIT}" && "${WF_CURRENT}" == "al" ]]; then
        workflow_write_state \
            "CURRENT_STEP=al" \
            "CURRENT_STATUS=paused" \
            "FAIL_REASON=awaiting_dft_labels" \
            "RUN_PID=$$" \
            "WANT_AL=${DO_AL}" \
            "WANT_REFINE=${DO_REFINE}" \
            "WANT_PER_TRAJ=${DO_PER_TRAJ}" \
            2>/dev/null || true
        echo "Active learning paused — waiting for DFT labels." >&2
        echo "  Save labeled cfg to ${AL_LABELED_DIR}/ then: ./run.sh --al" >&2
        echo "  State:  ${WORKFLOW_STATE_FILE}" >&2
        return 0
    fi
    if [[ -n "${WF_CURRENT}" ]]; then
        workflow_write_state \
            "CURRENT_STEP=${WF_CURRENT}" \
            "CURRENT_STATUS=failed" \
            "FAIL_REASON=exit_${rc}" \
            "RUN_PID=$$" \
            "WANT_AL=${DO_AL}" \
            "WANT_REFINE=${DO_REFINE}" \
            "WANT_PER_TRAJ=${DO_PER_TRAJ}" \
            2>/dev/null || true
        echo "Workflow failed at step '${WF_CURRENT}' (exit ${rc})." >&2
        echo "  Re-run: ./run.sh     # auto-resumes from this step" >&2
        echo "  State:  ${WORKFLOW_STATE_FILE}" >&2
    fi
    return "${rc}"
}
trap workflow_fail_hook EXIT

# Apply AUTO_RESUME planning: turn off steps already completed unless the user
# explicitly selected steps via --only / --skip-* / --fresh.
apply_auto_resume() {
    local hint
    # --only or --fresh: user fully controls the plan.
    if [[ -n "${ONLY_STEP}" || "${FRESH_RUN}" == "1" || "${AUTO_RESUME}" != "1" ]]; then
        return 0
    fi

    hint="$(workflow_infer_resume_point)"
    log "Auto-resume: inferred progress → ${hint}"

    case "${hint}" in
        complete)
            log "  All core steps already complete on disk."
            if [[ "${USER_SET_DATASET}" != "1" ]]; then DO_DATASET=0; fi
            if [[ "${USER_SET_TRAIN}" != "1" ]]; then DO_TRAIN=0; fi
            # Still validate unless user skipped (cheap sanity check).
            if [[ "${USER_SET_VALIDATE}" != "1" ]]; then DO_VALIDATE=1; fi
            ;;
        al|per-traj)
            if [[ "${USER_SET_DATASET}" != "1" ]]; then DO_DATASET=0; fi
            if [[ "${USER_SET_TRAIN}" != "1" ]]; then DO_TRAIN=0; fi
            if [[ "${USER_SET_VALIDATE}" != "1" ]]; then DO_VALIDATE=0; fi
            if [[ "${hint}" == "al" && "${DO_AL}" != "1" && "${WANT_AL:-0}" == "1" ]]; then
                DO_AL=1
                log "  Resuming planned active-learning step."
            fi
            if [[ "${hint}" == "al" && "${CURRENT_STATUS:-}" == "paused" ]]; then
                log "  Previous AL pause: will reuse existing iter queues if present."
            fi
            if [[ "${hint}" == "per-traj" && "${DO_PER_TRAJ}" != "1" && "${WANT_PER_TRAJ:-0}" == "1" ]]; then
                DO_PER_TRAJ=1
            fi
            ;;
        refine)
            if [[ "${USER_SET_DATASET}" != "1" ]]; then DO_DATASET=0; fi
            if [[ "${USER_SET_TRAIN}" != "1" ]]; then DO_TRAIN=0; fi
            if [[ "${USER_SET_VALIDATE}" != "1" ]]; then DO_VALIDATE=0; fi
            if [[ "${USER_SET_REFINE}" != "1" ]]; then
                DO_REFINE=1
                log "  Resuming refine sequence."
            fi
            ;;
        validate|validate_done)
            if [[ "${USER_SET_DATASET}" != "1" ]]; then DO_DATASET=0; fi
            if [[ "${USER_SET_TRAIN}" != "1" ]]; then DO_TRAIN=0; fi
            if [[ "${USER_SET_VALIDATE}" != "1" ]]; then DO_VALIDATE=1; fi
            log "  Skipping dataset + train (already complete); will validate."
            ;;
        train_postproc)
            if [[ "${USER_SET_DATASET}" != "1" ]]; then DO_DATASET=0; fi
            if [[ "${USER_SET_TRAIN}" != "1" ]]; then DO_TRAIN=1; fi
            log "  Train BFGS finished earlier; train step will only finish calc-errors/status."
            # train_mtp.sh detects postproc-pending and skips BFGS.
            export TRAIN_RESUME_POSTPROC=1
            ;;
        train)
            if [[ "${USER_SET_DATASET}" != "1" ]]; then DO_DATASET=0; fi
            if [[ "${USER_SET_TRAIN}" != "1" ]]; then DO_TRAIN=1; fi
            log "  Skipping dataset (present); will train/continue pot."
            ;;
        dataset)
            if [[ "${USER_SET_DATASET}" != "1" ]]; then DO_DATASET=1; fi
            log "  Resuming at dataset build."
            ;;
        start|*)
            log "  Starting from the beginning (no complete upstream artifacts)."
            ;;
    esac

    # Restore optional intents saved from the interrupted run when user did not
    # pass them on this CLI.
    if [[ -f "${WORKFLOW_STATE_FILE}" ]]; then
        # shellcheck disable=SC1090
        source "${WORKFLOW_STATE_FILE}" 2>/dev/null || true
        if [[ "${DO_AL}" != "1" && "${WANT_AL:-0}" == "1" && "${hint}" != "start" && "${hint}" != "dataset" ]]; then
            # Only auto-reenable AL if we are past train (AL is post-train).
            case "${hint}" in
                validate|validate_done|refine|al|complete|train_postproc)
                    DO_AL=1
                    log "  Restoring --al from previous run intent."
                    ;;
            esac
        fi
        if [[ "${DO_PER_TRAJ}" != "1" && "${WANT_PER_TRAJ:-0}" == "1" ]]; then
            case "${hint}" in
                validate|validate_done|refine|al|per-traj|complete)
                    DO_PER_TRAJ=1
                    log "  Restoring --per-traj from previous run intent."
                    ;;
            esac
        fi
    fi
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
        --refine)
            DO_REFINE=1
            USER_SET_REFINE=1
            shift
            ;;
        --skip-refine)
            AUTO_REFINE=0
            DO_REFINE=0
            USER_SET_REFINE=1
            shift
            ;;
        --per-traj|--per-temp)
            DO_PER_TRAJ=1
            shift
            ;;
        --skip-dataset)
            DO_DATASET=0
            USER_SET_DATASET=1
            shift
            ;;
        --skip-train)
            DO_TRAIN=0
            USER_SET_TRAIN=1
            shift
            ;;
        --skip-validate)
            DO_VALIDATE=0
            USER_SET_VALIDATE=1
            shift
            ;;
        --fresh|--no-resume)
            FRESH_RUN=1
            AUTO_RESUME=0
            shift
            ;;
        --resume)
            FRESH_RUN=0
            AUTO_RESUME=1
            shift
            ;;
        --only)
            ONLY_STEP="${2:?--only requires a step name}"
            DO_DATASET=0
            DO_TRAIN=0
            DO_VALIDATE=0
            DO_REFINE=0
            DO_AL=0
            DO_PER_TRAJ=0
            USER_SET_DATASET=1
            USER_SET_TRAIN=1
            USER_SET_VALIDATE=1
            USER_SET_REFINE=1
            case "${ONLY_STEP}" in
                template) DO_DATASET=0 ;;
                dataset)  DO_DATASET=1 ;;
                train)    DO_TRAIN=1 ;;
                validate) DO_VALIDATE=1 ;;
                refine)   DO_REFINE=1 ;;
                al)       DO_AL=1 ;;
                per-traj|per-temp) DO_PER_TRAJ=1 ;;
                *)
                    echo "Unknown step: ${ONLY_STEP}" >&2
                    echo "Valid steps: template dataset train validate refine al per-traj" >&2
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

# Stage mlp onto project FS so multi-day mpirun launches still work after
# Kerberos tickets for $HOME (NFS+krb5) expire.
if ! ensure_mlp || [[ ! -x "${MLP}" ]]; then
    echo "MLIP not found: ${MLP_SOURCE:-${MLP}}" >&2
    exit 1
fi

log "WC MTP workflow started: $(date)"
log "Project: ${MTP_PROJECT_ROOT}"
log "Level:   ${MTP_LEVEL}  Cutoff: ${MTP_MIN_DIST}-${MTP_MAX_DIST} A  RB: ${MTP_RADIAL_BASIS_SIZE}"
log "MPI:     nice -n ${NICE_N} ${MPIRUN} -np ${MPI_NPROCS} (train + active learning)"
log "Log:     ${RUN_LOG}"
log "Resume:  AUTO_RESUME=${AUTO_RESUME}  state=${WORKFLOW_STATE_FILE}"

# Refuse to start a second overlapping run when state says a step is live.
if [[ -f "${WORKFLOW_STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${WORKFLOW_STATE_FILE}" 2>/dev/null || true
    if [[ "${CURRENT_STATUS:-}" == "running" && -n "${RUN_PID:-}" ]]; then
        if kill -0 "${RUN_PID}" 2>/dev/null; then
            # Allow this process to continue (we are that PID); block a concurrent second run.
            if [[ "${RUN_PID}" != "$$" ]]; then
                echo "Another workflow appears to be running (PID ${RUN_PID}, step ${CURRENT_STEP:-?})." >&2
                echo "  State: ${WORKFLOW_STATE_FILE}" >&2
                echo "  Wait for it to finish, or: kill ${RUN_PID}  then  ./run.sh" >&2
                echo "  To ignore state and force a plan: ./run.sh --fresh" >&2
                exit 1
            fi
        fi
    fi
fi

# Plan which steps to run based on prior progress.
apply_auto_resume

# Announce whether we will continue from an existing fitted MTP
if FOUND_START="$(resolve_start_mtp 2>/dev/null)"; then
    log "Existing fitted MTP found — train/refine will continue from:"
    log "  ${FOUND_START}"
    if [[ "${TRAIN_FRESH:-0}" == "1" ]]; then
        log "  (but TRAIN_FRESH=1 is set — will start from template instead)"
    fi
else
    log "No fitted MTP found for level ${MTP_LEVEL}; train will start from template"
fi

log "Plan: dataset=${DO_DATASET} train=${DO_TRAIN} validate=${DO_VALIDATE} refine=${DO_REFINE} al=${DO_AL} per_traj=${DO_PER_TRAJ}"

# --- Step 1: Template ---
if [[ -z "${ONLY_STEP}" || "${ONLY_STEP}" == "template" || "${DO_DATASET}" == "1" || "${DO_TRAIN}" == "1" ]]; then
    step_header 1 "Prepare MTP template (level ${MTP_LEVEL}, W-C)"
    workflow_begin_step template
    ensure_mtp_template
    log "Template: ${MTP_TEMPLATE}"
    workflow_finish_step template
fi
[[ "${ONLY_STEP}" == "template" ]] && exit 0

# --- Step 2: Dataset ---
if [[ "${DO_DATASET}" == "1" ]]; then
    step_header 2 "Build initial training dataset"
    workflow_begin_step dataset
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
    workflow_finish_step dataset
fi
[[ "${ONLY_STEP}" == "dataset" ]] && exit 0

# --- Merge labeled configs (active-learning augmentation) ---
if [[ -n "${LABELED_CFG}" ]]; then
    step_header "2b" "Merge new DFT labels into training set"
    workflow_begin_step merge_labels
    if [[ ! -f "${LABELED_CFG}" ]]; then
        echo "Labeled cfg not found: ${LABELED_CFG}" >&2
        exit 1
    fi
    MERGED="${MTP_DATASETS_DIR}/initial/train_merged.cfg"
    python3 "${SCRIPTS}/merge_cfg.py" "${MERGED}" "${TRAIN_CFG}" "${LABELED_CFG}" --dedupe \
        | tee -a "${RUN_LOG}"
    cp "${MERGED}" "${TRAIN_CFG}"
    log "Updated training set: ${TRAIN_CFG}"
    export TRAIN_FORCE=1
    if [[ "${DO_TRAIN}" != "1" && "${DO_AL}" != "1" ]]; then
        if [[ "${USER_SET_TRAIN}" == "1" ]]; then
            log "WARNING: --labeled updated train.cfg but train is disabled; pot is now stale."
        else
            DO_TRAIN=1
            log "  Forcing train (train.cfg changed by --labeled merge)."
        fi
    elif [[ "${DO_TRAIN}" == "1" ]]; then
        log "  TRAIN_FORCE=1 (will not skip BFGS after labeled merge)"
    fi
    workflow_finish_step merge_labels
fi

# --- Step 3: Train ---
if [[ "${DO_TRAIN}" == "1" ]]; then
    step_header 3 "Train MTP"
    workflow_begin_step train
    if FOUND_START="$(resolve_start_mtp 2>/dev/null)"; then
        log "Continuing training from existing pot: ${FOUND_START}"
    else
        log "No existing fitted pot; fresh train from template"
    fi
    bash "${SCRIPTS}/train_mtp.sh" 2>&1 | tee -a "${RUN_LOG}"
    workflow_finish_step train
fi
[[ "${ONLY_STEP}" == "train" ]] && exit 0

# --- Step 4: Validate ---
VALIDATION_PASSED=1
if [[ "${DO_VALIDATE}" == "1" ]]; then
    step_header 4 "Validate MTP errors"
    workflow_begin_step validate
    ERR_LOG="${MTP_AL_DIR}/logs/calc_errors_valid.log"
    [[ -f "${ERR_LOG}" ]] || ERR_LOG="${MTP_AL_DIR}/logs/calc_errors_train.log"
    # If train just finished postproc, prefer the train errors log.
    if ! errors_report_ok "${ERR_LOG}" && errors_report_ok "${MTP_AL_DIR}/logs/calc_errors_train.log"; then
        ERR_LOG="${MTP_AL_DIR}/logs/calc_errors_train.log"
    fi
    if errors_report_ok "${ERR_LOG}"; then
        VALID_ARGS=(
            "${ERR_LOG}"
            --force-rms-max "${VAL_FORCE_RMS_MAX}"
            --force-mae-max "${VAL_FORCE_MAE_MAX}"
            --energy-per-atom-max "${VAL_ENERGY_PER_ATOM_MAX}"
            --stress-rms-max "${VAL_STRESS_RMS_MAX}"
        )
        if python3 "${SCRIPTS}/validate_mtp.py" "${VALID_ARGS[@]}" 2>&1 | tee -a "${RUN_LOG}"; then
            log "Validation: PASSED"
            VALIDATION_PASSED=1
        else
            VALIDATION_PASSED=0
            log "Validation: NOT converged (force RMS or energy thresholds)"
            # Auto-refine if BFGS hit step limit or force RMS failed
            TRAIN_LOG="${MTP_AL_DIR}/logs/train.log"
            STEP_HIT=0
            [[ -f "${TRAIN_LOG}" ]] && grep -q "step limit reached" "${TRAIN_LOG}" && STEP_HIT=1
            if [[ "${AUTO_REFINE}" == "1" && "${DO_REFINE}" != "1" ]]; then
                log "AUTO_REFINE=1: scheduling refine (more BFGS/rescale, force-weighted retrain)"
                DO_REFINE=1
            elif [[ "${STEP_HIT}" == "1" ]]; then
                log "BFGS step limit was reached; run: ./run.sh --only refine"
            else
                log "Next: ./run.sh --only refine   or  --al  or  MTP_LEVEL=22"
            fi
        fi
    else
        log "Validation skipped: no usable errors log found (need calc-errors Errors report)"
        # If pot exists but errors missing, steer user to re-run train postproc.
        if is_fitted_mtp "${TRAINED_MTP}"; then
            log "  Fitted pot exists; re-run: ./run.sh   # will finish train calc-errors then validate"
        fi
    fi
    workflow_finish_step validate
fi
[[ "${ONLY_STEP}" == "validate" ]] && exit 0

# --- Step 4b: Refine (continue BFGS / force retrain / high-error subset) ---
if [[ "${DO_REFINE}" == "1" ]]; then
    step_header "4b" "Refine MTP (BFGS continue + force retrain + high-error subset)"
    workflow_begin_step refine
    if bash "${SCRIPTS}/refine_mtp.sh" 2>&1 | tee -a "${RUN_LOG}"; then
        log "Refine: PASSED"
        VALIDATION_PASSED=1
    else
        log "Refine: still not at target — use active learning or MTP_LEVEL=${ESCALATE_LEVEL}"
        VALIDATION_PASSED=0
    fi
    workflow_finish_step refine
fi
[[ "${ONLY_STEP}" == "refine" ]] && exit 0

# --- Step 5: Active learning ---
if [[ "${DO_AL}" == "1" ]]; then
    step_header 5 "Active learning (grade threshold ${AL_SELECT_THRESHOLD})"
    workflow_begin_step al
    set +e
    # Line-buffer AL progress so "Retraining..." is not stuck behind tee.
    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf -oL -eL bash "${SCRIPTS}/run_active_learning.sh" \
            ${CANDIDATES_CFG:+"${CANDIDATES_CFG}"} \
            2>&1 | tee -a "${RUN_LOG}"
    else
        bash "${SCRIPTS}/run_active_learning.sh" ${CANDIDATES_CFG:+"${CANDIDATES_CFG}"} \
            2>&1 | tee -a "${RUN_LOG}"
    fi
    al_rc=${PIPESTATUS[0]}
    set -e
    if [[ "${al_rc}" -eq "${AL_PAUSE_EXIT}" ]]; then
        log "Active learning paused — unlabeled selections need DFT labels."
        log "  Save labeled cfg to ${AL_LABELED_DIR}/ then: ./run.sh --al"
        workflow_write_state \
            "CURRENT_STEP=al" \
            "CURRENT_STATUS=paused" \
            "FAIL_REASON=awaiting_dft_labels" \
            "WANT_AL=1" \
            "WANT_REFINE=${DO_REFINE}" \
            "WANT_PER_TRAJ=${DO_PER_TRAJ}"
        WF_CURRENT=""
        log ""
        log "Workflow paused at active learning: $(date)"
        log "  State: ${WORKFLOW_STATE_FILE}"
        exit 0
    elif [[ "${al_rc}" -ne 0 ]]; then
        exit "${al_rc}"
    fi
    workflow_finish_step al
fi
[[ "${ONLY_STEP}" == "al" ]] && exit 0

# --- Step 6: Per-trajectory fits ---
if [[ "${DO_PER_TRAJ}" == "1" ]]; then
    step_header 6 "Per-AIMD-trajectory MTP training"
    workflow_begin_step per-traj
    bash "${SCRIPTS}/run_all_trajectories.sh" --wait 2>&1 | tee -a "${RUN_LOG}"
    workflow_finish_step per-traj
fi
[[ "${ONLY_STEP}" == "per-traj" || "${ONLY_STEP}" == "per-temp" ]] && exit 0

workflow_write_state \
    "LAST_COMPLETED_STEP=complete" \
    "CURRENT_STEP=complete" \
    "CURRENT_STATUS=done" \
    "WANT_AL=${DO_AL}" \
    "WANT_REFINE=${DO_REFINE}" \
    "WANT_PER_TRAJ=${DO_PER_TRAJ}"
WF_CURRENT=""

log ""
log "Workflow complete: $(date)"
log "  Trained MTP: ${TRAINED_MTP}"
log "  Training set: ${TRAIN_CFG}"
log "  Full log:     ${RUN_LOG}"
log "  State:        ${WORKFLOW_STATE_FILE}"
