#!/bin/bash
# Train an MTP on a given .cfg training set.
#
# Usage:
#   ./scripts/train_mtp.sh [train.cfg] [output.mtp]
#
# Environment (see mtp_config.sh):
#   TRAIN_START_MTP   explicit starting potential
#   TRAIN_RESUME=auto search active_learning for fitted MTPs and continue (default)
#   TRAIN_FRESH=1     ignore existing pots; start from untrained template
#   TRAIN_FORCE=1     always re-run BFGS (AL merge / --labeled set this)
#   MAX_ITER, ENERGY_WEIGHT, FORCE_WEIGHT, STRESS_WEIGHT, ...
#
# Writes:
#   ${LOGDIR}/train.log
#   ${LOGDIR}/train_status.env   # STEP_LIMIT=0|1  FORCE_RMS=... etc.
#   ${LOGDIR}/calc_errors_train.log
#
# Defaults come from mtp_config.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mtp_config.sh
source "${SCRIPT_DIR}/mtp_config.sh"

TRAIN_SET="${1:-${TRAIN_CFG}}"
OUTPUT_MTP="${2:-${TRAINED_MTP}}"
LOGDIR="${MTP_AL_DIR}/logs"
mkdir -p "${LOGDIR}" "${MTP_AL_DIR}"

# Stage mlp onto project FS (survives Kerberos expiry on $HOME NFS).
if ! ensure_mlp || [[ ! -x "${MLP}" ]]; then
    echo "MLIP executable not found: ${MLP_SOURCE:-${MLP}}" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_SET}" ]]; then
    echo "Training set not found: ${TRAIN_SET}" >&2
    exit 1
fi

ensure_mtp_template

INIT_MTP="${MTP_AL_DIR}/init_L${MTP_LEVEL}.mtp"
# Per-iteration checkpoint (survives MPI crashes; resume via TRAIN_RESUME=1).
CURR_MTP="${CURR_MTP:-${MTP_AL_DIR}/WC_L${MTP_LEVEL}_curr.mtp}"
cp "${MTP_TEMPLATE}" "${INIT_MTP}"

EFFECTIVE_MAX_ITER="${EFFECTIVE_MAX_ITER:-${MAX_ITER}}"
EFFECTIVE_ENERGY_WEIGHT="${EFFECTIVE_ENERGY_WEIGHT:-${ENERGY_WEIGHT}}"
EFFECTIVE_FORCE_WEIGHT="${EFFECTIVE_FORCE_WEIGHT:-${FORCE_WEIGHT}}"
EFFECTIVE_STRESS_WEIGHT="${EFFECTIVE_STRESS_WEIGHT:-${STRESS_WEIGHT}}"
TRAIN_TAG="${TRAIN_TAG:-train}"
TRAIN_LOG="${LOGDIR}/${TRAIN_TAG}.log"
TRAIN_N_CFG="$(cfg_n_configurations "${TRAIN_SET}")"

# ---------------------------------------------------------------------------
# Resume: skip finished sub-steps after a mid-train crash.
#   TRAIN_FORCE=1          always re-run BFGS
#   TRAIN_RESUME_POSTPROC=1 force postproc-only (set by run.sh auto-resume)
# Never skip BFGS when train.cfg is newer or larger than the last fit
# (AL merge / --labeled). That used to look "already complete" and stamp
# merged.ok without refitting the new configs.
# ---------------------------------------------------------------------------
SKIP_BFGS=0
if [[ "${TRAIN_FORCE:-0}" == "1" ]]; then
    echo "TRAIN_FORCE=1: BFGS will run even if a fitted pot already exists"
elif train_fully_complete "${OUTPUT_MTP}" "${TRAIN_TAG}" "${LOGDIR}" "${TRAIN_SET}"; then
    echo "=== WC MTP training (${TRAIN_TAG}) ==="
    echo "Already complete: ${OUTPUT_MTP}"
    echo "  errors + train_status.env present and train set unchanged — nothing to do."
    echo "  Re-run BFGS with: TRAIN_FORCE=1 ./scripts/train_mtp.sh ..."
    exit 0
else
    if ! train_set_current_for_pot "${OUTPUT_MTP}" "${TRAIN_SET}" "${LOGDIR}"; then
        echo "Train set changed since last fit — will re-run BFGS"
        echo "  train set: ${TRAIN_SET} (n_cfg=${TRAIN_N_CFG})"
        echo "  pot:       ${OUTPUT_MTP}"
    fi
    if [[ "${TRAIN_RESUME_POSTPROC:-0}" == "1" ]] || \
       train_postproc_pending "${OUTPUT_MTP}" "${TRAIN_TAG}" "${LOGDIR}" "${TRAIN_SET}"; then
        SKIP_BFGS=1
    fi
fi

# ---------------------------------------------------------------------------
# Resolve starting potential: search for fitted MTPs and continue from them.
# ---------------------------------------------------------------------------
TRAIN_START_MTP_RESOLVED="${INIT_MTP}"
START_SOURCE="untrained template (fresh)"
CONTINUING=0

if [[ "${SKIP_BFGS}" == "1" ]]; then
    TRAIN_START_MTP_RESOLVED="${OUTPUT_MTP}"
    START_SOURCE="resume post-processing (BFGS already finished)"
    CONTINUING=1
    echo "Resuming train post-processing only (skip BFGS):"
    echo "  pot: ${OUTPUT_MTP}"
    echo "  log: ${TRAIN_LOG}"
elif FOUND_MTP="$(resolve_start_mtp)"; then
    TRAIN_START_MTP_RESOLVED="${FOUND_MTP}"
    CONTINUING=1
    if [[ -n "${TRAIN_START_MTP:-}" ]] && \
       [[ "$(readlink -f "${TRAIN_START_MTP}")" == "$(readlink -f "${FOUND_MTP}")" ]]; then
        START_SOURCE="explicit TRAIN_START_MTP"
    elif [[ "$(readlink -f "${FOUND_MTP}")" == "$(readlink -f "${OUTPUT_MTP}")" ]]; then
        START_SOURCE="existing trained pot (continue)"
    elif [[ "$(readlink -f "${FOUND_MTP}")" == "$(readlink -f "${CURR_MTP}")" ]]; then
        START_SOURCE="curr checkpoint (continue)"
    else
        START_SOURCE="discovered fitted pot (continue)"
    fi
    echo "Found fitted MTP — continuing training from:"
    echo "  ${TRAIN_START_MTP_RESOLVED}"
    echo "  source: ${START_SOURCE}"
else
    if [[ "${TRAIN_FRESH:-0}" == "1" ]]; then
        echo "TRAIN_FRESH=1: starting from untrained template (ignoring existing pots)"
    else
        echo "No fitted MTP found for level ${MTP_LEVEL}; starting from untrained template"
        # List what we looked at for debugging
        echo "  searched under: ${MTP_AL_DIR}/ (WC_L${MTP_LEVEL}_*.mtp, refine/, curr, trained)"
    fi
fi

# If continuing from a pot that is also the output path, keep a pre-continue backup.
# Skip backup when only finishing post-processing (pot is already the final result).
if [[ "${CONTINUING}" == "1" && "${SKIP_BFGS}" != "1" ]]; then
    if [[ "$(readlink -f "${TRAIN_START_MTP_RESOLVED}")" == "$(readlink -f "${OUTPUT_MTP}")" ]]; then
        BACKUP="${MTP_AL_DIR}/WC_L${MTP_LEVEL}_trained.prev.mtp"
        cp "${OUTPUT_MTP}" "${BACKUP}"
        echo "  backup of previous trained pot: ${BACKUP}"
    fi
fi

echo "=== WC MTP training (${TRAIN_TAG}) ==="
echo "Level:        ${MTP_LEVEL} (increase to 22 if defect force RMSE > ${VAL_DEFECT_FORCE_RMS_MAX} eV/A)"
echo "Template:     ${MTP_TEMPLATE}"
echo "Start pot:    ${TRAIN_START_MTP_RESOLVED}"
echo "Start mode:   ${START_SOURCE}"
echo "Species:      W + C (distinct)"
echo "Cutoff (A):   ${MTP_MIN_DIST} - ${MTP_MAX_DIST}"
echo "Radial basis: ${MTP_RADIAL_BASIS_SIZE} (Chebyshev; moment rank 3-4 from level)"
echo "Train set:    ${TRAIN_SET}"
echo "Output MTP:   ${OUTPUT_MTP}"
echo "Checkpoint:   ${CURR_MTP} (saved every BFGS iter)"
echo "Weights:      E=${EFFECTIVE_ENERGY_WEIGHT} F=${EFFECTIVE_FORCE_WEIGHT} S=${EFFECTIVE_STRESS_WEIGHT}"
echo "Max iter:     ${EFFECTIVE_MAX_ITER}  bfgs-conv-tol=${BFGS_CONV_TOL}"
echo "Parallel:     nice -n ${NICE_N} ${MPIRUN} -np ${MPI_NPROCS} (I_MPI_FABRICS=${I_MPI_FABRICS_LOCAL:-shm})"
if [[ "${SKIP_BFGS}" == "1" ]]; then
    echo "Resume:       post-processing only (BFGS skipped)"
fi
echo

# mindist rewrites the .cfg in place and is not MPI-safe: all ranks would
# race on the same file (unknown token / truncated train.cfg on NFS).
if [[ "${SKIP_BFGS}" == "1" ]]; then
    echo "[1/4] Skipping mindist (resume post-processing)"
elif [[ "${SKIP_MINDIST:-0}" != "1" ]]; then
    echo "[1/4] Updating mindist in training set (serial)"
    run_mlp_serial mindist "${TRAIN_SET}" \
        2>&1 | tee "${LOGDIR}/mindist.log"
else
    echo "[1/4] Skipping mindist (SKIP_MINDIST=1)"
fi

if [[ "${SKIP_BFGS}" == "1" ]]; then
    echo "[2/4] Skipping BFGS training (already finished; pot on disk)"
else
    echo "[2/4] Training MTP (max_iter=${EFFECTIVE_MAX_ITER}, np=${MPI_NPROCS})"
    if [[ "${CONTINUING}" == "1" ]]; then
        echo "  Continuing from fitted pot (BFGS Hessian reset; no random re-init; ends with rescale)"
    fi

    TRAIN_ARGS=(
        train "${TRAIN_START_MTP_RESOLVED}" "${TRAIN_SET}"
        --trained-pot-name="${OUTPUT_MTP}"
        --curr-pot-name="${CURR_MTP}"
        --max-iter="${EFFECTIVE_MAX_ITER}"
        --energy-weight="${EFFECTIVE_ENERGY_WEIGHT}"
        --force-weight="${EFFECTIVE_FORCE_WEIGHT}"
        --stress-weight="${EFFECTIVE_STRESS_WEIGHT}"
        --bfgs-conv-tol="${BFGS_CONV_TOL}"
        --weighting="${WEIGHTING}"
        --init-params="${INIT_PARAMS}"
        --update-mindist
    )

    if [[ "${SCALE_BY_FORCE}" != "0" ]]; then
        TRAIN_ARGS+=(--scale-by-force="${SCALE_BY_FORCE}")
    fi

    if [[ -f "${VALID_CFG}" ]]; then
        TRAIN_ARGS+=(--valid-cfgs="${VALID_CFG}")
        echo "  Using validation set: ${VALID_CFG}"
    fi

    # Capture train rc separately so a non-zero mlp exit (common on step
    # limit) does not skip postproc under set -e / pipefail mid-script.
    TRAIN_RC=0
    # stdbuf on tee so BFGS iter lines show up during long steps (mlp is
    # wrapped in run_mlp).
    if command -v stdbuf >/dev/null 2>&1; then
        run_mlp "${TRAIN_ARGS[@]}" \
            2>&1 | stdbuf -oL -eL tee "${TRAIN_LOG}" || TRAIN_RC=$?
    else
        run_mlp "${TRAIN_ARGS[@]}" \
            2>&1 | tee "${TRAIN_LOG}" || TRAIN_RC=$?
    fi
    if [[ ${TRAIN_RC} -ne 0 ]]; then
        echo "WARNING: mlp train exited rc=${TRAIN_RC} (continuing postproc if pot exists)" >&2
    fi
fi

if [[ ! -f "${OUTPUT_MTP}" ]]; then
    echo "Training failed: ${OUTPUT_MTP} not created" >&2
    exit 1
fi

# Detect BFGS termination mode from log.
STEP_LIMIT=0
BFGS_SMALL_DECR=0
if [[ -f "${TRAIN_LOG}" ]]; then
    if grep -q "step limit reached" "${TRAIN_LOG}"; then
        STEP_LIMIT=1
    fi
    if grep -q "BFGS ended due to small decr" "${TRAIN_LOG}"; then
        BFGS_SMALL_DECR=1
    fi
fi
LAST_BFGS_F="$(grep -E 'BFGS iter [0-9]+:' "${TRAIN_LOG}" 2>/dev/null | tail -1 | sed -E 's/.*f=([0-9.eE+-]+).*/\1/' || true)"

# Early status checkpoint: if the shell dies during calc-errors (NFS, syntax
# after mid-run script edit, OOM), refine can still read STEP_LIMIT / pot path.
STATUS_ENV="${LOGDIR}/train_status.env"
{
    echo "TRAIN_TAG=$(printf '%q' "${TRAIN_TAG}")"
    echo "OUTPUT_MTP=$(printf '%q' "${OUTPUT_MTP}")"
    echo "TRAIN_SET=$(printf '%q' "${TRAIN_SET}")"
    echo "TRAIN_N_CFG=${TRAIN_N_CFG}"
    echo "START_MTP=$(printf '%q' "${TRAIN_START_MTP_RESOLVED}")"
    echo "START_SOURCE=$(printf '%q' "${START_SOURCE}")"
    echo "CONTINUING=${CONTINUING}"
    echo "STEP_LIMIT=${STEP_LIMIT}"
    echo "BFGS_SMALL_DECR=${BFGS_SMALL_DECR}"
    echo "LAST_BFGS_F=$(printf '%q' "${LAST_BFGS_F}")"
    echo "FORCE_RMS="
    echo "FORCE_MAE="
    echo "EPA_RMS="
    echo "MAX_ITER=${EFFECTIVE_MAX_ITER}"
    echo "ENERGY_WEIGHT=${EFFECTIVE_ENERGY_WEIGHT}"
    echo "FORCE_WEIGHT=${EFFECTIVE_FORCE_WEIGHT}"
    echo "STRESS_WEIGHT=${EFFECTIVE_STRESS_WEIGHT}"
    echo "POSTPROC_PENDING=1"
} > "${STATUS_ENV}"

echo "[3/4] Training-set errors"
ERR_LOG="${LOGDIR}/calc_errors_${TRAIN_TAG}.log"
# Prefer MPI; fall back to serial if mpirun cannot exec mlp (common after
# multi-day runs when NFS+krb5 tickets expire — see ensure_mlp in mtp_config.sh).
if ! run_mlp_or_serial calc-errors "${OUTPUT_MTP}" "${TRAIN_SET}" \
        2>&1 | tee "${ERR_LOG}"; then
    # Last resort: mlp train already printed "* * * TRAIN ERRORS * * *"
    if grep -q "Errors report" "${TRAIN_LOG}" 2>/dev/null; then
        echo "WARNING: calc-errors failed; salvaging TRAIN ERRORS from ${TRAIN_LOG}" >&2
        # Extract the errors report block from the end of the train log.
        awk '
            /\* \* \* TRAIN ERRORS \* \* \*/ {grab=1}
            grab {print}
            /^_{10,}/ && seen_header {if (++ends>=2) exit}
            grab && /Errors report/ {seen_header=1}
        ' "${TRAIN_LOG}" > "${ERR_LOG}"
    else
        echo "calc-errors failed and no TRAIN ERRORS in train log" >&2
        exit 1
    fi
fi
# Reject logs that only contain mpirun launch failures (incomplete prior runs).
if ! grep -q "Errors report" "${ERR_LOG}"; then
    # Second chance: salvage from train log even if tee wrote an error banner.
    if grep -q "Errors report" "${TRAIN_LOG}" 2>/dev/null; then
        echo "WARNING: calc-errors log incomplete; salvaging TRAIN ERRORS from ${TRAIN_LOG}" >&2
        awk '
            /\* \* \* TRAIN ERRORS \* \* \*/ {grab=1}
            grab {print}
            /^_{10,}/ && seen_header {if (++ends>=2) exit}
            grab && /Errors report/ {seen_header=1}
        ' "${TRAIN_LOG}" > "${ERR_LOG}"
    fi
fi
if ! grep -q "Errors report" "${ERR_LOG}"; then
    echo "calc-errors log has no Errors report: ${ERR_LOG}" >&2
    exit 1
fi
# Keep canonical name used by run.sh / validate
cp "${ERR_LOG}" "${LOGDIR}/calc_errors_train.log"

if [[ -f "${VALID_CFG}" ]]; then
    echo "[4/4] Validation-set errors"
    if ! run_mlp_or_serial calc-errors "${OUTPUT_MTP}" "${VALID_CFG}" \
            2>&1 | tee "${LOGDIR}/calc_errors_valid.log"; then
        echo "WARNING: validation-set calc-errors failed (non-fatal)" >&2
    fi
else
    echo "[4/4] No validation set (${VALID_CFG}); skipped"
fi

# Parse force RMS for downstream refine loop.
FORCE_RMS="$(python3 - "${ERR_LOG}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"Forces:[\s\S]*?RMS\s+absolute difference = ([0-9.eE+-]+)", text)
print(m.group(1) if m else "")
PY
)"
FORCE_MAE="$(python3 - "${ERR_LOG}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"Forces:[\s\S]*?Average absolute difference = ([0-9.eE+-]+)", text)
print(m.group(1) if m else "")
PY
)"
EPA_RMS="$(python3 - "${ERR_LOG}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"Energy per atom:[\s\S]*?RMS\s+absolute difference = ([0-9.eE+-]+)", text)
print(m.group(1) if m else "")
PY
)"

# Quote values so paths/spaces are safe when sourced. (STATUS_ENV set earlier.)
{
    echo "TRAIN_TAG=$(printf '%q' "${TRAIN_TAG}")"
    echo "OUTPUT_MTP=$(printf '%q' "${OUTPUT_MTP}")"
    echo "TRAIN_SET=$(printf '%q' "${TRAIN_SET}")"
    echo "TRAIN_N_CFG=${TRAIN_N_CFG}"
    echo "START_MTP=$(printf '%q' "${TRAIN_START_MTP_RESOLVED}")"
    echo "START_SOURCE=$(printf '%q' "${START_SOURCE}")"
    echo "CONTINUING=${CONTINUING}"
    echo "STEP_LIMIT=${STEP_LIMIT}"
    echo "BFGS_SMALL_DECR=${BFGS_SMALL_DECR}"
    echo "LAST_BFGS_F=$(printf '%q' "${LAST_BFGS_F}")"
    echo "FORCE_RMS=$(printf '%q' "${FORCE_RMS}")"
    echo "FORCE_MAE=$(printf '%q' "${FORCE_MAE}")"
    echo "EPA_RMS=$(printf '%q' "${EPA_RMS}")"
    echo "MAX_ITER=${EFFECTIVE_MAX_ITER}"
    echo "ENERGY_WEIGHT=${EFFECTIVE_ENERGY_WEIGHT}"
    echo "FORCE_WEIGHT=${EFFECTIVE_FORCE_WEIGHT}"
    echo "STRESS_WEIGHT=${EFFECTIVE_STRESS_WEIGHT}"
    echo "POSTPROC_PENDING=0"
} > "${STATUS_ENV}"

echo
echo "Training complete: ${OUTPUT_MTP}"
echo "  Started from: ${START_SOURCE}"
echo "  BFGS step_limit=${STEP_LIMIT} small_decr=${BFGS_SMALL_DECR} last_f=${LAST_BFGS_F}"
echo "  force_rms=${FORCE_RMS} force_mae=${FORCE_MAE} epa_rms=${EPA_RMS}"
echo "  Status: ${STATUS_ENV}"
