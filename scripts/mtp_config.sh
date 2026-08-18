#!/bin/bash
# Shared MTP hyperparameters and paths for the WC W-vacancy reconstruction workflow.
# Source this file from other scripts:  source "$(dirname "$0")/mtp_config.sh"
#
# Recommended starting parameters (iteratively refined via active learning):
#   Level 20-22 | r_cut 6.0 A | radial basis 10-12 | W/C distinct species
#   See datasets/sources.conf for training-set composition targets.

# --- MLIP installation ---
MLIP_ROOT="${MLIP_ROOT:-${HOME}/software/mlip-2}"
# Original install path (often under $HOME on /user3, NFS4+krb5).
MLP_SOURCE="${MLP_SOURCE:-${MLP:-${MLIP_ROOT}/bin/mlp}}"
MLP="${MLP:-${MLP_SOURCE}}"
MLIP_UNTRAINED_MTPS="${MLIP_UNTRAINED_MTPS:-${MLIP_ROOT}/untrained_mtps}"
# Stage mlp onto the project filesystem before launch (default on).
# /user3 is NFS4 with sec=krb5: after multi-day nohup jobs Kerberos tickets
# expire and mpirun dies with:
#   execvp error on file .../mlp (Permission denied)
# Project tree is typically /slask (sec=sys) and stays executable. Set
# MLP_STAGE=0 to disable; MLP_STAGED_BIN to override the staged path.
MLP_STAGE="${MLP_STAGE:-1}"

# --- Parallel execution (MPI) for train / active-learning stages ---
# Override with e.g. MPI_NPROCS=8 or MPI_NPROCS=1 (serial via mpirun).
MPI_NPROCS="${MPI_NPROCS:-19}"
MPIRUN="${MPIRUN:-mpirun}"
NICE_N="${NICE_N:-19}"
# Extra mpirun args, e.g. MPI_EXTRA_ARGS="-genv I_MPI_DEBUG 0"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"

# Intel MPI single-node fabric. Default "shm" avoids the long-run crash:
#   Assertion failed ... nemesis/netmod/tcp/socksm.c: (revents & POLLERR) == 0
# The login env often sets I_MPI_FABRICS=shm:tcp; TCP sockets then fail after
# multi-day BFGS. Force shared-memory only for local jobs (this machine is
# single-node). Override with I_MPI_FABRICS=shm:tcp for multi-node runs.
I_MPI_FABRICS_LOCAL="${I_MPI_FABRICS_LOCAL:-shm}"

# Copy/refresh MLP onto a non-krb path so mpirun can still exec after ticket expiry.
# Safe to call repeatedly; no-op when staging is off or source is already the dest.
ensure_mlp() {
    local src="${MLP_SOURCE}"
    if [[ ! -x "${src}" ]]; then
        # Source may itself be a staged path if user set MLP=...
        src="${MLP}"
    fi
    if [[ ! -x "${src}" ]]; then
        echo "MLIP executable not found or not executable: ${src}" >&2
        echo "  (If this is NFS+krb5, renew tickets with 'kinit' and retry.)" >&2
        return 1
    fi
    if [[ "${MLP_STAGE}" != "1" ]]; then
        MLP="${src}"
        return 0
    fi
    # MTP_PROJECT_ROOT is set later in this file; fall back if ensure_mlp is
    # called before that block (should not happen in normal scripts).
    local root="${MTP_PROJECT_ROOT:-}"
    if [[ -z "${root}" ]]; then
        local _cfg
        _cfg="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        root="$(cd "${_cfg}/.." && pwd)"
    fi
    local dest="${MLP_STAGED_BIN:-${root}/.bin/mlp}"
    local src_real dest_real
    src_real="$(readlink -f "${src}" 2>/dev/null || echo "${src}")"
    dest_real="$(readlink -f "${dest}" 2>/dev/null || echo "${dest}")"
    if [[ "${src_real}" == "${dest_real}" ]]; then
        MLP="${dest}"
        return 0
    fi
    if [[ ! -x "${dest}" || "${src}" -nt "${dest}" ]]; then
        mkdir -p "$(dirname "${dest}")"
        # Atomic replace so concurrent ranks never see a partial binary.
        cp -f "${src}" "${dest}.tmp" && mv -f "${dest}.tmp" "${dest}"
        chmod a+x "${dest}"
    fi
    if [[ ! -x "${dest}" ]]; then
        echo "Failed to stage mlp to ${dest}; using source ${src}" >&2
        MLP="${src}"
        return 0
    fi
    MLP="${dest}"
}

# Parallel MLP: nice -n NICE_N mpirun -np MPI_NPROCS mlp ...
# Uses shm fabric + higher nofile so long trainings do not die on POLLERR.
run_mlp() {
    ensure_mlp || return 1
    # Soft nofile is often 1024; many MPI ranks need more sockets/fds.
    ulimit -n 65536 2>/dev/null || true

    # -genv forces fabric into all ranks (login env often has I_MPI_FABRICS=shm:tcp).
    # shellcheck disable=SC2086
    nice -n "${NICE_N}" "${MPIRUN}" -np "${MPI_NPROCS}" \
        -genv I_MPI_FABRICS "${I_MPI_FABRICS_LOCAL}" \
        -genv I_MPI_FALLBACK "${I_MPI_FALLBACK:-0}" \
        -genv I_MPI_DYNAMIC_CONNECTION "${I_MPI_DYNAMIC_CONNECTION:-0}" \
        ${MPI_EXTRA_ARGS} \
        "${MLP}" "$@"
}

# Serial MLP under nice only (convert-cfg and other light ops).
run_mlp_serial() {
    ensure_mlp || return 1
    nice -n "${NICE_N}" "${MLP}" "$@"
}

# Run calc-errors (or similar) with MPI, falling back to serial on launch failure
# (e.g. residual Permission denied if staging was disabled).
run_mlp_or_serial() {
    local log_tmp rc=0
    log_tmp="$(mktemp)" || return 1
    # `|| rc=$?` keeps set -e from aborting on parallel launch failure.
    run_mlp "$@" >"${log_tmp}" 2>&1 || rc=$?
    if [[ ${rc} -eq 0 ]]; then
        cat "${log_tmp}"
        rm -f "${log_tmp}"
        return 0
    fi
    # Detect exec/permission failures from Hydra and retry serial.
    if grep -qE 'Permission denied|execvp error' "${log_tmp}"; then
        echo "WARNING: parallel mlp failed to launch; retrying serial:" >&2
        grep -E 'Permission denied|execvp error' "${log_tmp}" | head -3 >&2 || true
        cat "${log_tmp}"
        rm -f "${log_tmp}"
        run_mlp_serial "$@"
        return $?
    fi
    cat "${log_tmp}"
    rm -f "${log_tmp}"
    return "${rc}"
}

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
# Continue from existing fitted MTPs when found (trained / curr / refine stages).
# Set TRAIN_FRESH=1 to ignore them and start from the untrained template.
TRAIN_FRESH="${TRAIN_FRESH:-0}"
# TRAIN_FORCE=1 always re-run BFGS even if pot + errors look complete.
# The AL driver and --labeled merge set this so a grown train.cfg is refit.
TRAIN_FORCE="${TRAIN_FORCE:-0}"
# TRAIN_RESUME: auto | 1 | 0
#   auto (default) = search for fitted pots and continue from the newest
#   1              = prefer curr checkpoint
#   0              = do not auto-pick (still honors TRAIN_START_MTP / TRAIN_FRESH)
TRAIN_RESUME="${TRAIN_RESUME:-auto}"

# --- Continue / refine until BFGS + force RMS converge ---
# After "step limit reached", restart train from best pot (BFGS Hessian resets;
# each restart ends with linear re-fit + radial rescaling in mlp train).
MAX_ITER_CONTINUE="${MAX_ITER_CONTINUE:-2000}"
TRAIN_CONTINUE_MAX_ROUNDS="${TRAIN_CONTINUE_MAX_ROUNDS:-3}"
# Stage 1 after BFGS is settled: force-weighted retrain from best intermediate.
FORCE_WEIGHT_RETRAIN="${FORCE_WEIGHT_RETRAIN:-0.5}"
FORCE_RETRAIN_ENERGY_WEIGHT="${FORCE_RETRAIN_ENERGY_WEIGHT:-${ENERGY_WEIGHT}}"
FORCE_RETRAIN_STRESS_WEIGHT="${FORCE_RETRAIN_STRESS_WEIGHT:-${STRESS_WEIGHT}}"
FORCE_RETRAIN_MAX_ITER="${FORCE_RETRAIN_MAX_ITER:-1500}"
# Optional stage: high-F (or force-only) fit on high-error subset, then full-set polish.
ENABLE_HIGH_ERROR_RETRAIN="${ENABLE_HIGH_ERROR_RETRAIN:-1}"
HIGH_ERROR_TOP_FRAC="${HIGH_ERROR_TOP_FRAC:-0.15}"      # top fraction by force RMSE
HIGH_ERROR_MIN_FORCE_RMSE="${HIGH_ERROR_MIN_FORCE_RMSE:-0.15}"  # eV/A absolute floor
HIGH_ERROR_ENERGY_WEIGHT="${HIGH_ERROR_ENERGY_WEIGHT:-0.1}"
HIGH_ERROR_FORCE_WEIGHT="${HIGH_ERROR_FORCE_WEIGHT:-1.0}"
HIGH_ERROR_STRESS_WEIGHT="${HIGH_ERROR_STRESS_WEIGHT:-0.0}"
HIGH_ERROR_MAX_ITER="${HIGH_ERROR_MAX_ITER:-500}"
HIGH_ERROR_POLISH_MAX_ITER="${HIGH_ERROR_POLISH_MAX_ITER:-1000}"
# Exclude bad / irrelevant DFT before (re)training.
FILTER_BAD_DFT="${FILTER_BAD_DFT:-1}"
FILTER_MAX_FORCE="${FILTER_MAX_FORCE:-50.0}"            # eV/A: drop configs with larger |F|
FILTER_MIN_DIST_ABS="${FILTER_MIN_DIST_ABS:-0.50}"      # A: drop unphysical short pairs
FILTER_MAX_EPA_OUTLIER="${FILTER_MAX_EPA_OUTLIER:-5.0}"  # eV/atom vs median energy/atom
# If force RMS still fails after data+refine: bump level (off by default).
AUTO_ESCALATE_LEVEL="${AUTO_ESCALATE_LEVEL:-0}"
ESCALATE_LEVEL="${ESCALATE_LEVEL:-22}"
# Auto-run refine sequence after initial train when validation fails (run.sh).
AUTO_REFINE="${AUTO_REFINE:-1}"

# --- Dataset subsampling (AIMD frames) ---
SUBSAMPLE_STRIDE="${SUBSAMPLE_STRIDE:-50}"
HIGH_T_STRIDE="${HIGH_T_STRIDE:-25}"
HIGH_T_THRESHOLD_K="${HIGH_T_THRESHOLD_K:-1800}"   # >1500-1800 K defect fluctuations

# --- Active learning (maxvol / extrapolation grade) ---
# Add configs with grade above threshold; start 2.0-4.0 (default 3.0).
# Prefer high-force-error / high-grade candidates when force RMS is stuck.
AL_INIT_THRESHOLD="${AL_INIT_THRESHOLD:-1e-5}"
AL_SELECT_THRESHOLD="${AL_SELECT_THRESHOLD:-3.0}"
AL_SWAP_THRESHOLD="${AL_SWAP_THRESHOLD:-1.0000001}"
AL_GRADE_THRESHOLD="${AL_GRADE_THRESHOLD:-${AL_SELECT_THRESHOLD}}"
AL_SELECTION_LIMIT="${AL_SELECTION_LIMIT:-50}"
AL_MAX_ITERATIONS="${AL_MAX_ITERATIONS:-20}"
AL_PREFER_HIGH_FORCE_ERROR="${AL_PREFER_HIGH_FORCE_ERROR:-1}"
# Exit code from run_active_learning.sh when selections still need VASP.
# Already-labeled queues (AIMD/OUTCAR leftover frames) are merged instead.
AL_PAUSE_EXIT="${AL_PAUSE_EXIT:-10}"

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

# Workflow resume state (written by run.sh / train_mtp.sh after each step).
# AUTO_RESUME=1 (default): ./run.sh skips steps already completed on disk.
# --fresh / AUTO_RESUME=0: ignore state and re-run selected steps.
WORKFLOW_STATE_FILE="${WORKFLOW_STATE_FILE:-${MTP_AL_DIR}/workflow_state.env}"
AUTO_RESUME="${AUTO_RESUME:-1}"

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

# True if path looks like a fitted MTP (has linear coeffs), not an untrained template.
is_fitted_mtp() {
    local path="${1:-}"
    [[ -n "${path}" && -f "${path}" && -s "${path}" ]] || return 1
    # Untrained templates / init copies lack species_coeffs + moment_coeffs.
    grep -q '^species_coeffs' "${path}" 2>/dev/null || return 1
    grep -q '^moment_coeffs' "${path}" 2>/dev/null || return 1
    return 0
}

# True if MTP level/basis is compatible with current MTP_LEVEL template.
mtp_level_compatible() {
    local path="${1:-}"
    local template="${2:-${MTP_TEMPLATE}}"
    [[ -f "${path}" ]] || return 1
    # Prefer matching alpha_moments_count (level fingerprint) when template exists.
    if [[ -f "${template}" ]]; then
        local a b
        a=$(grep -m1 'alpha_moments_count' "${path}" 2>/dev/null | awk -F= '{gsub(/[ \r]/,"",$2); print $2}')
        b=$(grep -m1 'alpha_moments_count' "${template}" 2>/dev/null | awk -F= '{gsub(/[ \r]/,"",$2); print $2}')
        if [[ -n "${a}" && -n "${b}" && "${a}" != "${b}" ]]; then
            return 1
        fi
        local ra rb
        ra=$(grep -m1 'radial_funcs_count' "${path}" 2>/dev/null | awk -F= '{gsub(/[ \r]/,"",$2); print $2}')
        rb=$(grep -m1 'radial_funcs_count' "${template}" 2>/dev/null | awk -F= '{gsub(/[ \r]/,"",$2); print $2}')
        if [[ -n "${ra}" && -n "${rb}" && "${ra}" != "${rb}" ]]; then
            return 1
        fi
    fi
    return 0
}

# Collect fitted MTP candidates for the current level (absolute paths).
# Searches active_learning/, refine/, and optional extra dirs.
list_fitted_mtp_candidates() {
    local level="${1:-${MTP_LEVEL}}"
    local -a candidates=()
    local f base

    # Canonical names first (order is only for discovery; newest wins later).
    for f in \
        "${TRAINED_MTP}" \
        "${MTP_AL_DIR}/WC_L${level}_trained.mtp" \
        "${MTP_AL_DIR}/WC_L${level}_curr.mtp" \
        "${CURR_MTP:-}"
    do
        [[ -n "${f}" && -f "${f}" && -s "${f}" ]] || continue
        candidates+=("$(readlink -f "${f}")")
    done

    shopt -s nullglob
    for f in \
        "${MTP_AL_DIR}/WC_L${level}"*.mtp \
        "${MTP_AL_DIR}/refine/WC_L${level}"*.mtp \
        "${MTP_AL_DIR}"/*/WC_L${level}*.mtp
    do
        base="$(basename "${f}")"
        # Skip untrained init copies and bare templates
        [[ "${base}" == init_L* || "${base}" == init.mtp ]] && continue
        [[ -s "${f}" ]] || continue
        candidates+=("$(readlink -f "${f}")")
    done
    shopt -u nullglob

    # Deduplicate while preserving order
    if [[ ${#candidates[@]} -eq 0 ]]; then
        return 0
    fi
    printf '%s\n' "${candidates[@]}" | awk 'NF && !seen[$0]++'
}

# Resolve the best pot to continue training from.
# Prints absolute path and returns 0 if a fitted pot was found; 1 otherwise.
#
# Priority:
#   1. Explicit argument or TRAIN_START_MTP (if fitted)
#   2. TRAIN_RESUME=1 → curr checkpoint if fitted
#   3. Newest fitted MTP for this level (trained / curr / refine / ...)
#   4. (none → caller uses untrained template)
#
# Honors TRAIN_FRESH=1 (always fail so caller starts fresh).
resolve_start_mtp() {
    local explicit="${1:-}"
    local level="${MTP_LEVEL}"
    local f best="" best_mtime=0 mtime

    if [[ "${TRAIN_FRESH:-0}" == "1" ]]; then
        return 1
    fi

    # Explicit path wins when fitted + compatible
    if [[ -z "${explicit}" && -n "${TRAIN_START_MTP:-}" ]]; then
        explicit="${TRAIN_START_MTP}"
    fi
    if [[ -n "${explicit}" && -f "${explicit}" && -s "${explicit}" ]]; then
        if is_fitted_mtp "${explicit}" && mtp_level_compatible "${explicit}"; then
            readlink -f "${explicit}"
            return 0
        fi
        # Explicit but unfitted: still use it only if user forced TRAIN_START_MTP
        if [[ -n "${TRAIN_START_MTP:-}" && "$(readlink -f "${explicit}")" == "$(readlink -f "${TRAIN_START_MTP}")" ]]; then
            readlink -f "${explicit}"
            return 0
        fi
    fi

    # Prefer curr when TRAIN_RESUME=1
    local curr="${CURR_MTP:-${MTP_AL_DIR}/WC_L${level}_curr.mtp}"
    if [[ "${TRAIN_RESUME:-auto}" == "1" ]]; then
        if is_fitted_mtp "${curr}" && mtp_level_compatible "${curr}"; then
            readlink -f "${curr}"
            return 0
        fi
    fi

    if [[ "${TRAIN_RESUME:-auto}" == "0" ]]; then
        return 1
    fi

    # Scan all fitted candidates; pick newest by mtime
    local -a found=()
    mapfile -t found < <(list_fitted_mtp_candidates "${level}" || true)
    for f in "${found[@]}"; do
        is_fitted_mtp "${f}" || continue
        mtp_level_compatible "${f}" || continue
        mtime=$(stat -c %Y "${f}" 2>/dev/null || stat -f %m "${f}")
        if (( mtime >= best_mtime )); then
            best_mtime=$mtime
            best="${f}"
        fi
    done

    if [[ -n "${best}" ]]; then
        echo "${best}"
        return 0
    fi
    return 1
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
        local need_rebuild=1
        if [[ -s "${fallback}" ]]; then
            need_rebuild=0
            local f
            for f in "${files[@]}"; do
                if [[ "${f}" -nt "${fallback}" ]]; then
                    need_rebuild=1
                    break
                fi
            done
        fi
        if [[ "${need_rebuild}" == "1" ]]; then
            python3 "${MTP_PROJECT_ROOT}/scripts/merge_cfg.py" \
                "${fallback}" "${files[@]}" --dedupe >/dev/null
        fi
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

# Print N_CFG / N_LABELED / N_UNLABELED for a .cfg (Energy + forces = labeled).
al_cfg_label_status() {
    python3 "${MTP_PROJECT_ROOT}/scripts/cfg_label_status.py" "$1"
}

ensure_mtp_template() {
    local template="${MTP_TEMPLATE}"
    local gen_script="${MTP_PROJECT_ROOT}/scripts/prepare_mtp_template.py"
    if [[ -f "${template}" ]]; then
        local rfc src_mtp src_rfc
        # Strip CR (Windows line endings in some untrained .mtp files) so
        # "5" vs "5\r" does not force endless template regeneration.
        rfc=$(grep -m1 'radial_funcs_count' "${template}" | awk -F= '{gsub(/[ \r]/,"",$2); print $2}')
        src_mtp="${MLIP_UNTRAINED_MTPS}/$(printf '%02d' "${MTP_LEVEL}").mtp"
        [[ -f "${src_mtp}" ]] || src_mtp="${MLIP_UNTRAINED_MTPS}/${MTP_LEVEL}.mtp"
        if [[ -f "${src_mtp}" ]]; then
            src_rfc=$(grep -m1 'radial_funcs_count' "${src_mtp}" | awk -F= '{gsub(/[ \r]/,"",$2); print $2}')
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

# ---------------------------------------------------------------------------
# Workflow resume helpers
# ---------------------------------------------------------------------------
# Step order used by run.sh auto-resume:
#   template → dataset → train → validate → refine → al → per-traj → complete

cfg_has_configurations() {
    local cfg="${1:-${TRAIN_CFG}}"
    [[ -f "${cfg}" && -s "${cfg}" ]] || return 1
    grep -q '^BEGIN_CFG' "${cfg}" 2>/dev/null
}

# Number of MLIP configurations in a .cfg (BEGIN_CFG count). Prints 0 if missing.
cfg_n_configurations() {
    local cfg="${1:-}"
    if [[ -z "${cfg}" || ! -f "${cfg}" ]]; then
        echo 0
        return 1
    fi
    grep -c '^BEGIN_CFG' "${cfg}" 2>/dev/null || true
}

# Return 0 if pot + last train_status still match this training set.
# Fail when train.cfg was rewritten or grew after the last fit (AL merge,
# --labeled, dataset rebuild). Resume skip must not hide a new train set.
train_set_current_for_pot() {
    local pot="${1:-${TRAINED_MTP}}"
    local train_set="${2:-${TRAIN_CFG}}"
    local logdir="${3:-${MTP_AL_DIR}/logs}"
    local status_env="${logdir}/train_status.env"
    local pot_m set_m status_n now_n

    [[ -f "${pot}" && -f "${train_set}" ]] || return 1

    pot_m=$(stat -c %Y "${pot}" 2>/dev/null || stat -f %m "${pot}" 2>/dev/null || echo 0)
    set_m=$(stat -c %Y "${train_set}" 2>/dev/null || stat -f %m "${train_set}" 2>/dev/null || echo 0)
    # train.cfg rewritten after pot (typical AL merge). Allow 30s NFS skew.
    if (( set_m > pot_m + 30 )); then
        return 1
    fi

    now_n="$(cfg_n_configurations "${train_set}")"
    if [[ -f "${status_env}" ]]; then
        status_n="$(grep -E '^TRAIN_N_CFG=' "${status_env}" 2>/dev/null | head -1 \
            | cut -d= -f2- | tr -d "\"'\\\\" || true)"
        if [[ -n "${status_n}" && "${status_n}" != "${now_n}" ]]; then
            return 1
        fi
    fi
    return 0
}

# True if a calc-errors log contains a real Errors report (not just mpirun failures).
errors_report_ok() {
    local log="${1:-}"
    [[ -n "${log}" && -f "${log}" && -s "${log}" ]] || return 1
    grep -q 'Errors report' "${log}" 2>/dev/null
}

# True if an mlp train log shows BFGS/training finished (pot should exist).
train_log_bfgs_finished() {
    local log="${1:-}"
    [[ -n "${log}" && -f "${log}" && -s "${log}" ]] || return 1
    grep -qE 'MTPR training ended|step limit reached|BFGS ended due to small decr|\* \* \* TRAIN ERRORS \* \* \*' \
        "${log}" 2>/dev/null
}

# Canonical calc-errors log for a train tag (train → calc_errors_train.log).
train_errors_log_for_tag() {
    local tag="${1:-train}"
    local logdir="${2:-${MTP_AL_DIR}/logs}"
    if [[ "${tag}" == "train" ]]; then
        echo "${logdir}/calc_errors_train.log"
    else
        echo "${logdir}/calc_errors_${tag}.log"
    fi
}

# Training fully finished for pot + tag: fitted pot + matching calc-errors log
# written at/after the pot (so a newer refine pot does not look "done" from an
# old errors file), and the training set has not grown/been rewritten since.
train_fully_complete() {
    local pot="${1:-${TRAINED_MTP}}"
    local tag="${2:-train}"
    local logdir="${3:-${MTP_AL_DIR}/logs}"
    local train_set="${4:-${TRAIN_CFG}}"
    local err pot_m err_m status_env status_tag status_rms
    is_fitted_mtp "${pot}" || return 1
    err="$(train_errors_log_for_tag "${tag}" "${logdir}")"
    if ! errors_report_ok "${err}"; then
        err="${logdir}/calc_errors_${tag}.log"
        errors_report_ok "${err}" || return 1
    fi
    pot_m=$(stat -c %Y "${pot}" 2>/dev/null || stat -f %m "${pot}" 2>/dev/null || echo 0)
    err_m=$(stat -c %Y "${err}" 2>/dev/null || stat -f %m "${err}" 2>/dev/null || echo 0)
    # Allow small clock/NFS skew; errors must not predate the pot substantially.
    if (( err_m + 30 < pot_m )); then
        return 1
    fi
    # If train_status.env is from this same tag, require FORCE_RMS present.
    status_env="${logdir}/train_status.env"
    if [[ -f "${status_env}" ]]; then
        status_tag="$(grep -E '^TRAIN_TAG=' "${status_env}" 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"'\\\\" || true)"
        if [[ -z "${status_tag}" || "${status_tag}" == "${tag}" ]]; then
            status_rms="$(grep -E '^FORCE_RMS=' "${status_env}" 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"'\\\\" || true)"
            [[ -n "${status_rms}" ]] || return 1
        fi
    else
        return 1
    fi
    # A newer / larger train.cfg (AL merge) is not "already complete".
    if [[ -n "${train_set}" && -f "${train_set}" ]]; then
        train_set_current_for_pot "${pot}" "${train_set}" "${logdir}" || return 1
    fi
    return 0
}

# BFGS finished and pot written, but calc-errors / status not done (mid-train crash).
# Not pending if the train set changed — that needs a full BFGS, not postproc-only.
train_postproc_pending() {
    local pot="${1:-${TRAINED_MTP}}"
    local tag="${2:-train}"
    local logdir="${3:-${MTP_AL_DIR}/logs}"
    local train_set="${4:-${TRAIN_CFG}}"
    local tlog="${logdir}/${tag}.log"
    is_fitted_mtp "${pot}" || return 1
    train_fully_complete "${pot}" "${tag}" "${logdir}" "${train_set}" && return 1
    if [[ -n "${train_set}" && -f "${train_set}" ]]; then
        train_set_current_for_pot "${pot}" "${train_set}" "${logdir}" || return 1
    fi
    train_log_bfgs_finished "${tlog}" || return 1
    return 0
}

# Dataset step complete on disk.
dataset_step_complete() {
    cfg_has_configurations "${TRAIN_CFG}"
}

# Train step complete for the default trained pot.
train_step_complete() {
    train_fully_complete "${TRAINED_MTP}" "train" "${MTP_AL_DIR}/logs"
}

# Persist workflow state (safe to call from subshells; overwrites file).
# Usage: workflow_write_state KEY=val KEY=val ...
# Always refreshes UPDATED and MTP_LEVEL.
workflow_write_state() {
    local state_file="${WORKFLOW_STATE_FILE}"
    local -A kv=()
    local k v line key val
    mkdir -p "$(dirname "${state_file}")"

    # Load existing keys so we can merge.
    if [[ -f "${state_file}" ]]; then
        while IFS= read -r line || [[ -n "${line}" ]]; do
            [[ "${line}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
            key="${line%%=*}"
            val="${line#*=}"
            # Strip surrounding single quotes from printf %q style if present.
            kv["${key}"]="${val}"
        done < "${state_file}"
    fi

    for pair in "$@"; do
        k="${pair%%=*}"
        v="${pair#*=}"
        kv["${k}"]="${v}"
    done
    kv["UPDATED"]="$(date '+%Y-%m-%dT%H:%M:%S%z' 2>/dev/null || date)"
    kv["MTP_LEVEL"]="${MTP_LEVEL}"
    kv["PROJECT"]="${MTP_PROJECT_ROOT}"

    {
        echo "# WC MTP workflow state — auto-managed; used by ./run.sh resume"
        for k in LAST_COMPLETED_STEP CURRENT_STEP CURRENT_STATUS \
                 WANT_AL WANT_REFINE WANT_PER_TRAJ \
                 FAIL_REASON RUN_PID UPDATED MTP_LEVEL PROJECT; do
            if [[ -n "${kv[$k]+x}" ]]; then
                printf '%s=%s\n' "${k}" "$(printf '%q' "${kv[$k]}")"
            fi
        done
        # Preserve any extra keys not in the ordered list.
        for k in "${!kv[@]}"; do
            case "${k}" in
                LAST_COMPLETED_STEP|CURRENT_STEP|CURRENT_STATUS|WANT_AL|WANT_REFINE|WANT_PER_TRAJ|FAIL_REASON|RUN_PID|UPDATED|MTP_LEVEL|PROJECT) continue ;;
            esac
            printf '%s=%s\n' "${k}" "$(printf '%q' "${kv[$k]}")"
        done
    } > "${state_file}.tmp"
    mv -f "${state_file}.tmp" "${state_file}"
}

# Load workflow state into the caller's shell (variables become global).
# Returns 1 if no state file.
workflow_load_state() {
    local state_file="${WORKFLOW_STATE_FILE}"
    [[ -f "${state_file}" ]] || return 1
    # shellcheck disable=SC1090
    source "${state_file}"
    return 0
}

# Infer how far the pipeline has progressed from artifacts + state file.
# Prints one of: start | dataset | train | train_postproc | validate | refine | al | per-traj | complete
# and sets RESUME_HINT (same value) for callers.
workflow_infer_resume_point() {
    local last="" status="" current=""
    RESUME_HINT="start"

    if workflow_load_state 2>/dev/null; then
        last="${LAST_COMPLETED_STEP:-}"
        status="${CURRENT_STATUS:-}"
        current="${CURRENT_STEP:-}"
    fi

    # Interrupted step takes priority over "last completed", but advance past
    # steps that finished on disk after the crash (e.g. manual recovery).
    if [[ "${status}" == "running" || "${status}" == "failed" || "${status}" == "paused" ]]; then
        case "${current}" in
            dataset)
                if dataset_step_complete; then
                    RESUME_HINT="train"
                else
                    RESUME_HINT="dataset"
                fi
                echo "${RESUME_HINT}"
                return 0
                ;;
            train)
                if train_step_complete; then
                    RESUME_HINT="validate"
                elif train_postproc_pending; then
                    RESUME_HINT="train_postproc"
                else
                    RESUME_HINT="train"
                fi
                echo "${RESUME_HINT}"
                return 0
                ;;
            validate)
                RESUME_HINT="validate"
                echo "${RESUME_HINT}"
                return 0
                ;;
            refine|al|per-traj)
                RESUME_HINT="${current}"
                echo "${RESUME_HINT}"
                return 0
                ;;
        esac
    fi

    # Prefer explicit last-completed from state when present.
    case "${last}" in
        complete)
            RESUME_HINT="complete"
            echo "${RESUME_HINT}"
            return 0
            ;;
        refine)
            # Refine finished according to state; optional AL/per-traj still open.
            RESUME_HINT="al"
            echo "${RESUME_HINT}"
            return 0
            ;;
        validate)
            RESUME_HINT="validate_done"
            echo "${RESUME_HINT}"
            return 0
            ;;
        train)
            RESUME_HINT="validate"
            echo "${RESUME_HINT}"
            return 0
            ;;
        dataset)
            RESUME_HINT="train"
            echo "${RESUME_HINT}"
            return 0
            ;;
        template)
            RESUME_HINT="dataset"
            echo "${RESUME_HINT}"
            return 0
            ;;
    esac

    # Artifact fallback (state missing or incomplete after manual runs / old jobs).
    if train_step_complete; then
        RESUME_HINT="validate"
    elif train_postproc_pending; then
        RESUME_HINT="train_postproc"
    elif is_fitted_mtp "${TRAINED_MTP}" || is_fitted_mtp "${MTP_AL_DIR}/WC_L${MTP_LEVEL}_curr.mtp"; then
        # Pot exists but training not fully sealed → re-enter train (continue BFGS).
        RESUME_HINT="train"
    elif dataset_step_complete; then
        RESUME_HINT="train"
    elif [[ -f "${MTP_TEMPLATE}" ]]; then
        RESUME_HINT="dataset"
    else
        RESUME_HINT="start"
    fi

    # If refine was in progress (refine dir has stage pots newer than train status),
    # prefer resuming refine over re-validating from scratch.
    if [[ "${RESUME_HINT}" == "validate" || "${RESUME_HINT}" == "validate_done" ]]; then
        if [[ "${WANT_REFINE:-0}" == "1" ]] || \
           [[ -d "${MTP_AL_DIR}/refine" && -n "$(ls -A "${MTP_AL_DIR}/refine"/WC_L${MTP_LEVEL}_*.mtp 2>/dev/null || true)" ]]; then
            # Only auto-jump to refine if last train hit step limit or force RMS failed.
            local fr=""
            if [[ -f "${MTP_AL_DIR}/logs/train_status.env" ]]; then
                # shellcheck disable=SC1090
                source "${MTP_AL_DIR}/logs/train_status.env" 2>/dev/null || true
                fr="${FORCE_RMS:-}"
            fi
            if [[ "${STEP_LIMIT:-0}" == "1" ]] || \
               { [[ -n "${fr}" ]] && python3 -c "import sys; sys.exit(0 if float('${fr}') > float('${VAL_FORCE_RMS_MAX}') else 1)" 2>/dev/null; }; then
                if [[ -d "${MTP_AL_DIR}/refine" ]]; then
                    RESUME_HINT="refine"
                fi
            fi
        fi
    fi

    echo "${RESUME_HINT}"
    return 0
}
