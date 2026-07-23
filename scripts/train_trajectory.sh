#!/bin/bash
# Train an MTP for one W-vacancy AIMD trajectory folder.
#
# Usage:
#   ./scripts/train_trajectory.sh vac_W_2500_ML
#   ./scripts/train_trajectory.sh /slask/mehdin/dynamics/wc/vac_W_2800_small_ML
#
# Uses recommended hyperparameters from mtp_config.sh (level 20, 6.0 A cutoff).
# For the unified defect model, prefer: ./run.sh

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <trajectory_folder_or_name>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mtp_config.sh
source "${SCRIPT_DIR}/mtp_config.sh"

TARGET="$1"
if [[ -d "${TARGET}" ]]; then
    WORKDIR="$(cd "${TARGET}" && pwd)"
    TRAJ="$(basename "${WORKDIR}")"
else
    TRAJ="${TARGET}"
    WORKDIR="${MTP_PROJECT_ROOT}/${TRAJ}"
fi

OUTCAR="${WORKDIR}/OUTCAR"
if [[ ! -f "${OUTCAR}" ]]; then
    echo "Missing OUTCAR in ${WORKDIR}" >&2
    exit 1
fi

ensure_mtp_template

LOGDIR="${WORKDIR}/logs"
mkdir -p "${LOGDIR}"

TRAIN_FULL="${WORKDIR}/train_full.cfg"
TRAIN_CFG_LOCAL="${WORKDIR}/train.cfg"
INIT_MTP="${WORKDIR}/init.mtp"
TRAINED_MTP_LOCAL="${WORKDIR}/WC_${TRAJ}_L${MTP_LEVEL}.mtp"
ERRORS_LOG="${LOGDIR}/calc_errors.log"

TEMP="${AIMD_SOURCE_TEMPS[$TRAJ]:-0}"

echo "=== MTP training workflow: ${TRAJ} ==="
echo "Workdir:      ${WORKDIR}"
echo "Temperature:  ${TEMP} K"
echo "MLIP:         ${MLP}"
echo "MTP level:    ${MTP_LEVEL}"
echo "Template:     ${MTP_TEMPLATE}"
echo "Cutoff (A):   ${MTP_MIN_DIST} - ${MTP_MAX_DIST}"
echo "Stride:       ${SUBSAMPLE_STRIDE}"
echo "Weights:      E=${ENERGY_WEIGHT} F=${FORCE_WEIGHT} S=${STRESS_WEIGHT}"
echo

if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
    echo "[1/5] Converting OUTCAR -> train_full.cfg"
    nice -n 19 "${MLP}" convert-cfg --input-format=vasp-outcar "${OUTCAR}" "${TRAIN_FULL}" \
        2>&1 | tee "${LOGDIR}/convert.log"
else
    echo "[1/5] Skipping conversion (SKIP_CONVERT=1)"
fi

if [[ ! -f "${TRAIN_FULL}" ]]; then
    echo "Conversion failed: ${TRAIN_FULL} not created" >&2
    exit 1
fi

if [[ "${SKIP_SUBSAMPLE:-0}" != "1" ]]; then
    echo "[2/5] Subsampling train_full.cfg -> train.cfg"
    STRIDE="${SUBSAMPLE_STRIDE}"
    if (( TEMP >= HIGH_T_THRESHOLD_K )); then
        STRIDE="${HIGH_T_STRIDE}"
        echo "  High-T subsample stride: ${STRIDE}"
    fi
    python3 "${SCRIPT_DIR}/subsample_cfg.py" \
        "${TRAIN_FULL}" "${TRAIN_CFG_LOCAL}" \
        --stride "${STRIDE}" \
        2>&1 | tee "${LOGDIR}/subsample.log"
else
    echo "[2/5] Skipping subsample (SKIP_SUBSAMPLE=1)"
fi

if [[ ! -f "${TRAIN_CFG_LOCAL}" ]]; then
    echo "Subsample failed: ${TRAIN_CFG_LOCAL} not created" >&2
    exit 1
fi

echo "[3/5] Preparing initial MTP"
cp "${MTP_TEMPLATE}" "${INIT_MTP}"
nice -n 19 "${MLP}" mindist "${TRAIN_CFG_LOCAL}" --update-mindist 2>&1 | tee "${LOGDIR}/mindist.log"

echo "[4/5] Training MTP"
TRAIN_ARGS=(
    train "${INIT_MTP}" "${TRAIN_CFG_LOCAL}"
    --trained-pot-name="${TRAINED_MTP_LOCAL}"
    --max-iter="${MAX_ITER}"
    --energy-weight="${ENERGY_WEIGHT}"
    --force-weight="${FORCE_WEIGHT}"
    --stress-weight="${STRESS_WEIGHT}"
    --bfgs-conv-tol="${BFGS_CONV_TOL}"
    --weighting="${WEIGHTING}"
    --update-mindist
)

if [[ "${SCALE_BY_FORCE}" != "0" ]]; then
    TRAIN_ARGS+=(--scale-by-force="${SCALE_BY_FORCE}")
fi

nice -n 19 "${MLP}" "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOGDIR}/train.log"

if [[ ! -f "${TRAINED_MTP_LOCAL}" ]]; then
    echo "Training failed: ${TRAINED_MTP_LOCAL} not created" >&2
    exit 1
fi

echo "[5/5] Calculating training errors"
nice -n 19 "${MLP}" calc-errors "${TRAINED_MTP_LOCAL}" "${TRAIN_CFG_LOCAL}" \
    2>&1 | tee "${ERRORS_LOG}"

echo
echo "Done."
echo "  Trained potential: ${TRAINED_MTP_LOCAL}"
echo "  Training set:      ${TRAIN_CFG_LOCAL}"
echo "  Logs:              ${LOGDIR}/"