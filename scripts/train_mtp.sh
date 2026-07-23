#!/bin/bash
# Train an MTP on a given .cfg training set.
#
# Usage:
#   ./scripts/train_mtp.sh [train.cfg] [output.mtp]
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

if [[ ! -x "${MLP}" ]]; then
    echo "MLIP executable not found: ${MLP}" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_SET}" ]]; then
    echo "Training set not found: ${TRAIN_SET}" >&2
    exit 1
fi

ensure_mtp_template

INIT_MTP="${MTP_AL_DIR}/init_L${MTP_LEVEL}.mtp"
cp "${MTP_TEMPLATE}" "${INIT_MTP}"

echo "=== WC MTP training ==="
echo "Level:        ${MTP_LEVEL} (increase to 22 if defect force RMSE > ${VAL_DEFECT_FORCE_RMS_MAX} eV/A)"
echo "Template:     ${MTP_TEMPLATE}"
echo "Species:      W + C (distinct)"
echo "Cutoff (A):   ${MTP_MIN_DIST} - ${MTP_MAX_DIST}"
echo "Radial basis: ${MTP_RADIAL_BASIS_SIZE} (Chebyshev; moment rank 3-4 from level)"
echo "Train set:    ${TRAIN_SET}"
echo "Output MTP:   ${OUTPUT_MTP}"
echo "Weights:      E=${ENERGY_WEIGHT} F=${FORCE_WEIGHT} S=${STRESS_WEIGHT}"
echo

echo "[1/4] Updating mindist in training set"
nice -n 19 "${MLP}" mindist "${TRAIN_SET}" --update-mindist \
    2>&1 | tee "${LOGDIR}/mindist.log"

echo "[2/4] Training MTP (max_iter=${MAX_ITER})"
TRAIN_ARGS=(
    train "${INIT_MTP}" "${TRAIN_SET}"
    --trained-pot-name="${OUTPUT_MTP}"
    --max-iter="${MAX_ITER}"
    --energy-weight="${ENERGY_WEIGHT}"
    --force-weight="${FORCE_WEIGHT}"
    --stress-weight="${STRESS_WEIGHT}"
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

nice -n 19 "${MLP}" "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOGDIR}/train.log"

if [[ ! -f "${OUTPUT_MTP}" ]]; then
    echo "Training failed: ${OUTPUT_MTP} not created" >&2
    exit 1
fi

echo "[3/4] Training-set errors"
nice -n 19 "${MLP}" calc-errors "${OUTPUT_MTP}" "${TRAIN_SET}" \
    2>&1 | tee "${LOGDIR}/calc_errors_train.log"

if [[ -f "${VALID_CFG}" ]]; then
    echo "[4/4] Validation-set errors"
    nice -n 19 "${MLP}" calc-errors "${OUTPUT_MTP}" "${VALID_CFG}" \
        2>&1 | tee "${LOGDIR}/calc_errors_valid.log"
else
    echo "[4/4] No validation set (${VALID_CFG}); skipped"
fi

echo
echo "Training complete: ${OUTPUT_MTP}"