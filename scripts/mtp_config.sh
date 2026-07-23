#!/bin/bash
# Shared MTP hyperparameters and paths for the WC W-vacancy reconstruction workflow.
# Source this file from other scripts:  source "$(dirname "$0")/mtp_config.sh"
#
# Recommended starting parameters (iteratively refined via active learning):
#   Level 20-22 | r_cut 6.0 A | radial basis 10-12 | W/C distinct species
#   See datasets/sources.conf for training-set composition targets.

# --- MLIP installation ---
MLIP_ROOT="${MLIP_ROOT:-${HOME}/software/mlip-2}"
MLP="${MLP:-${MLIP_ROOT}/bin/mlp}"
MLIP_UNTRAINED_MTPS="${MLIP_UNTRAINED_MTPS:-${MLIP_ROOT}/untrained_mtps}"

# --- MTP complexity (level 20-22; start at 20) ---
# Increase to 22 only if defect-config force RMSE remains >0.1 eV/A after AL.
MTP_LEVEL="${MTP_LEVEL:-20}"
MTP_SPECIES_COUNT="${MTP_SPECIES_COUNT:-2}"   # W and C as distinct species

# --- Radial basis / cutoff ---
# W-C NN ~2.1-2.2 A; lattice a~2.906, c~2.837 A; target 6.0 A (1st-3rd shells).
# Test variants: MTP_MAX_DIST=5.0 or 6.5
MTP_MIN_DIST="${MTP_MIN_DIST:-1.2}"            # C-C dimer well ~1.3-1.5 A
MTP_MAX_DIST="${MTP_MAX_DIST:-6.0}"
MTP_RADIAL_BASIS_SIZE="${MTP_RADIAL_BASIS_SIZE:-11}"   # 10-12 Chebyshev functions

# --- Fitting weights (tune on validation holdout) ---
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCE_WEIGHT="${FORCE_WEIGHT:-0.1}"            # 0.05-0.2; prioritize local accuracy
STRESS_WEIGHT="${STRESS_WEIGHT:-0.005}"        # 0.001-0.01
SCALE_BY_FORCE="${SCALE_BY_FORCE:-0}"

# --- Training control (linear fit on fixed basis; nonlinear only if needed) ---
MAX_ITER="${MAX_ITER:-2000}"
BFGS_CONV_TOL="${BFGS_CONV_TOL:-1e-4}"
WEIGHTING="${WEIGHTING:-vibrations}"
INIT_PARAMS="${INIT_PARAMS:-random}"

# --- Dataset subsampling (AIMD frames) ---
SUBSAMPLE_STRIDE="${SUBSAMPLE_STRIDE:-50}"
HIGH_T_STRIDE="${HIGH_T_STRIDE:-25}"
HIGH_T_THRESHOLD_K="${HIGH_T_THRESHOLD_K:-1800}"   # >1500-1800 K defect fluctuations

# --- Active learning (maxvol / extrapolation grade) ---
# Add configs with grade above threshold; start 2.0-4.0 (default 3.0).
AL_INIT_THRESHOLD="${AL_INIT_THRESHOLD:-1e-5}"
AL_SELECT_THRESHOLD="${AL_SELECT_THRESHOLD:-3.0}"
AL_SWAP_THRESHOLD="${AL_SWAP_THRESHOLD:-1.0000001}"
AL_GRADE_THRESHOLD="${AL_GRADE_THRESHOLD:-${AL_SELECT_THRESHOLD}}"
AL_SELECTION_LIMIT="${AL_SELECTION_LIMIT:-50}"
AL_MAX_ITERATIONS="${AL_MAX_ITERATIONS:-20}"

# --- Validation convergence targets (held-out test set) ---
VAL_FORCE_RMS_MAX="${VAL_FORCE_RMS_MAX:-0.08}"       # eV/A
VAL_FORCE_MAE_MAX="${VAL_FORCE_MAE_MAX:-0.08}"       # eV/A
VAL_ENERGY_PER_ATOM_MAX="${VAL_ENERGY_PER_ATOM_MAX:-0.005}"  # 5 meV/atom
VAL_STRESS_RMS_MAX="${VAL_STRESS_RMS_MAX:-0.5}"      # GPa
VAL_DEFECT_FORCE_RMS_MAX="${VAL_DEFECT_FORCE_RMS_MAX:-0.1}"  # trigger level increase

# --- Paths (override MTP_PROJECT_ROOT before sourcing to relocate) ---
if [[ -z "${MTP_PROJECT_ROOT:-}" ]]; then
    _CFG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MTP_PROJECT_ROOT="$(cd "${_CFG_DIR}/.." && pwd)"
fi

MTP_TEMPLATES_DIR="${MTP_TEMPLATES_DIR:-${MTP_PROJECT_ROOT}/templates}"
MTP_DATASETS_DIR="${MTP_DATASETS_DIR:-${MTP_PROJECT_ROOT}/datasets}"
MTP_AL_DIR="${MTP_AL_DIR:-${MTP_PROJECT_ROOT}/active_learning}"
MTP_LOGS_DIR="${MTP_LOGS_DIR:-${MTP_PROJECT_ROOT}/logs}"

MTP_TEMPLATE="${MTP_TEMPLATE:-${MTP_TEMPLATES_DIR}/WC_L${MTP_LEVEL}.mtp}"
TRAIN_CFG="${TRAIN_CFG:-${MTP_DATASETS_DIR}/initial/train.cfg}"
VALID_CFG="${VALID_CFG:-${MTP_DATASETS_DIR}/validation/holdout.cfg}"
TRAINED_MTP="${TRAINED_MTP:-${MTP_AL_DIR}/WC_L${MTP_LEVEL}_trained.mtp}"
ALS_FILE="${ALS_FILE:-${MTP_AL_DIR}/state.als}"

# Active-learning auto-discovery directories
AL_CANDIDATES_DIR="${AL_CANDIDATES_DIR:-${MTP_DATASETS_DIR}/candidates}"
AL_LABELED_DIR="${AL_LABELED_DIR:-${MTP_DATASETS_DIR}/labeled}"
AL_MERGED_CANDIDATES="${AL_MERGED_CANDIDATES:-${AL_CANDIDATES_DIR}/_merged_pool.cfg}"

# WC W-vacancy AIMD trajectory folders (folder name -> temperature K for subsampling)
# Multiple supercell sizes capture finite-size vacancy-vacancy effects.
declare -A AIMD_SOURCE_TEMPS=(
    [vac_W_2300]=2300
    [vac_W_2500_ML]=2500
    [vac_W_2500_small_ML]=2500
    [vac_W_2800_ML]=2800
    [vac_W_2800_small_ML]=2800
    [vac_W_2800_betaWC]=2800
)
AIMD_SOURCES=(
    vac_W_2300
    vac_W_2500_ML
    vac_W_2500_small_ML
    vac_W_2800_ML
    vac_W_2800_small_ML
)

# Optional external DFT sources with category tags for composition tracking.
# Format: category:kind:path  (see datasets/sources.conf)
SOURCES_CONF="${SOURCES_CONF:-${MTP_DATASETS_DIR}/sources.conf}"

mtp_template_name() {
    echo "WC_L${MTP_LEVEL}.mtp"
}

# Resolve candidate pool for active learning (optional explicit path).
resolve_al_candidate_cfg() {
    local explicit="${1:-}"

    if [[ -n "${explicit}" && -f "${explicit}" ]]; then
        echo "${explicit}"
        return 0
    fi
    if [[ -n "${AL_CANDIDATE_CFG:-}" && -f "${AL_CANDIDATE_CFG}" ]]; then
        echo "${AL_CANDIDATE_CFG}"
        return 0
    fi

    mkdir -p "${AL_CANDIDATES_DIR}"
    local -a files=()
    shopt -s nullglob
    for f in "${AL_CANDIDATES_DIR}"/*.cfg; do
        local base
        base="$(basename "${f}")"
        [[ "${base}" == "_merged_pool.cfg" || "${base}" == "_aimd_staging_pool.cfg" ]] && continue
        [[ -s "${f}" ]] && files+=("${f}")
    done
    shopt -u nullglob

    if [[ ${#files[@]} -eq 1 ]]; then
        echo "${files[0]}"
        return 0
    fi
    if [[ ${#files[@]} -gt 1 ]]; then
        python3 "${MTP_PROJECT_ROOT}/scripts/merge_cfg.py" \
            "${AL_MERGED_CANDIDATES}" "${files[@]}" --dedupe >/dev/null
        echo "${AL_MERGED_CANDIDATES}"
        return 0
    fi

    local staging="${MTP_DATASETS_DIR}/initial/staging"
    files=()
    shopt -s nullglob
    for f in "${staging}"/aimd_*.cfg; do
        [[ "$(basename "${f}")" =~ _s[0-9]+\.cfg$ ]] && continue
        [[ -s "${f}" ]] && files+=("${f}")
    done
    shopt -u nullglob

    if [[ ${#files[@]} -gt 0 ]]; then
        local fallback="${AL_CANDIDATES_DIR}/_aimd_staging_pool.cfg"
        python3 "${MTP_PROJECT_ROOT}/scripts/merge_cfg.py" \
            "${fallback}" "${files[@]}" --dedupe >/dev/null
        echo "${fallback}"
        return 0
    fi

    return 1
}

# Resolve newly labeled DFT configs for merge (optional explicit path).
resolve_al_labeled_cfg() {
    local explicit="${1:-}"

    if [[ -n "${explicit}" && -f "${explicit}" ]]; then
        echo "${explicit}"
        return 0
    fi
    if [[ -n "${AL_LABELED_CFG:-}" && -f "${AL_LABELED_CFG}" ]]; then
        echo "${AL_LABELED_CFG}"
        return 0
    fi

    mkdir -p "${AL_LABELED_DIR}"
    local -a files=()
    shopt -s nullglob
    files=("${AL_LABELED_DIR}"/*.cfg)
    shopt -u nullglob

    local newest="" newest_mtime=0 mtime
    for f in "${files[@]}"; do
        [[ ! -s "${f}" ]] && continue
        mtime=$(stat -c %Y "${f}" 2>/dev/null || stat -f %m "${f}")
        if (( mtime > newest_mtime )); then
            newest_mtime=$mtime
            newest="${f}"
        fi
    done

    if [[ -n "${newest}" ]]; then
        echo "${newest}"
        return 0
    fi
    return 1
}

ensure_mtp_template() {
    local template="${MTP_TEMPLATE}"
    local gen_script="${MTP_PROJECT_ROOT}/scripts/prepare_mtp_template.py"
    if [[ -f "${template}" ]]; then
        local rfc src_mtp src_rfc
        rfc=$(grep -m1 'radial_funcs_count' "${template}" | awk -F= '{gsub(/ /,"",$2); print $2}')
        src_mtp="${MLIP_UNTRAINED_MTPS}/$(printf '%02d' "${MTP_LEVEL}").mtp"
        [[ -f "${src_mtp}" ]] || src_mtp="${MLIP_UNTRAINED_MTPS}/${MTP_LEVEL}.mtp"
        if [[ -f "${src_mtp}" ]]; then
            src_rfc=$(grep -m1 'radial_funcs_count' "${src_mtp}" | awk -F= '{gsub(/ /,"",$2); print $2}')
            if [[ -n "${rfc}" && -n "${src_rfc}" && "${rfc}" == "${src_rfc}" ]]; then
                return 0
            fi
            echo "Regenerating stale MTP template (radial_funcs_count=${rfc}, expected ${src_rfc}): ${template}" >&2
        fi
        rm -f "${template}"
    fi
    if [[ ! -f "${gen_script}" ]]; then
        echo "Template missing and generator not found: ${gen_script}" >&2
        return 1
    fi
    python3 "${gen_script}" \
        --level "${MTP_LEVEL}" \
        --species-count "${MTP_SPECIES_COUNT}" \
        --min-dist "${MTP_MIN_DIST}" \
        --max-dist "${MTP_MAX_DIST}" \
        --radial-basis-size "${MTP_RADIAL_BASIS_SIZE}" \
        --output "${template}"
}