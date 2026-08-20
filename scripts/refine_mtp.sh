#!/bin/bash
# Practical MTP refinement sequence after an initial (or partial) fit.
#
# Addresses the common failure mode from nohup.out:
#   - BFGS hits "step limit reached" (not fully converged)
#   - force RMS slightly above VAL_FORCE_RMS_MAX
#   - post-train rescaling done only once
#
# Sequence (each stage restarts BFGS from the best intermediate pot;
# mlp train always ends with linear re-fit + radial rescaling):
#
#   0. Optional: filter bad / irrelevant DFT (VASP_not_converged, huge F, ...)
#   1. Continue BFGS (+ rescale) from best pot until step limit clears
#      or TRAIN_CONTINUE_MAX_ROUNDS exhausted
#   2. Force-weighted retrain from best intermediate (reset BFGS)
#   3. Optional: high-F / force-focused fit on high-error subset, then
#      polish on the full training set
#   4. Revisit min cutoff vs mindist if short pairs remain
#   5. If still stuck: recommend AL on high-force-error / high-grade configs;
#      optionally escalate MTP_LEVEL (AUTO_ESCALATE_LEVEL=1)
#
# Usage:
#   ./scripts/refine_mtp.sh
#   ./scripts/refine_mtp.sh [train.cfg] [start.mtp]
#
# Environment: see mtp_config.sh (FORCE_WEIGHT_RETRAIN, MAX_ITER_CONTINUE, ...)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mtp_config.sh
source "${SCRIPT_DIR}/mtp_config.sh"

TRAIN_SET="${1:-${TRAIN_CFG}}"
START_POT="${2:-}"
LOGDIR="${MTP_AL_DIR}/logs"
REFINE_DIR="${MTP_AL_DIR}/refine"
mkdir -p "${LOGDIR}" "${REFINE_DIR}" "${MTP_AL_DIR}"

CURR_MTP="${CURR_MTP:-${MTP_AL_DIR}/WC_L${MTP_LEVEL}_curr.mtp}"

# Search for any fitted MTP (trained / curr / refine stages) and continue from it.
BEST_MTP=""
START_SOURCE=""
if [[ -n "${START_POT}" && -f "${START_POT}" ]]; then
    if is_fitted_mtp "${START_POT}"; then
        BEST_MTP="$(readlink -f "${START_POT}")"
        START_SOURCE="explicit argument"
    else
        echo "Warning: ${START_POT} does not look fitted; searching for alternatives..." >&2
    fi
fi
if [[ -z "${BEST_MTP}" ]]; then
    if FOUND_MTP="$(resolve_start_mtp "${START_POT}")"; then
        BEST_MTP="${FOUND_MTP}"
        START_SOURCE="auto-discovered fitted MTP"
    fi
fi
if [[ -z "${BEST_MTP}" ]]; then
    echo "No fitted MTP found for level ${MTP_LEVEL}." >&2
    echo "Searched: ${MTP_AL_DIR}/WC_L${MTP_LEVEL}_*.mtp, refine/, curr, trained" >&2
    echo "Run ./scripts/train_mtp.sh first, or set TRAIN_START_MTP=/path/to.mtp" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_SET}" ]]; then
    echo "Training set not found: ${TRAIN_SET}" >&2
    exit 1
fi

echo "=== WC MTP refine sequence ==="
echo "Train set:  ${TRAIN_SET}"
echo "Start pot:  ${BEST_MTP}"
echo "Source:     ${START_SOURCE}"
echo "Target:     force_rms <= ${VAL_FORCE_RMS_MAX} eV/A"
echo "Continue:   up to ${TRAIN_CONTINUE_MAX_ROUNDS} x ${MAX_ITER_CONTINUE} BFGS iters"
echo "Force retrain F-weight: ${FORCE_WEIGHT_RETRAIN}"
# Show other fitted pots found (for transparency)
if mapfile -t _all_fitted < <(list_fitted_mtp_candidates 2>/dev/null || true); then
    if [[ ${#_all_fitted[@]} -gt 1 ]]; then
        echo "Other fitted MTPs on disk (newest selected above):"
        for f in "${_all_fitted[@]}"; do
            [[ "$(readlink -f "${f}")" == "$(readlink -f "${BEST_MTP}")" ]] && continue
            is_fitted_mtp "${f}" || continue
            echo "  - ${f}"
        done
    fi
fi
echo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_train_stage() {
    # run_train_stage <tag> <start_mtp> <train_cfg> <max_iter> <E> <F> <S>
    # Promotes pot even if train_mtp postproc fails (calc-errors / script
    # interrupted after BFGS), so refine can resume from the new pot.
    local tag="$1" start="$2" cfg="$3" maxit="$4" ew="$5" fw="$6" sw="$7"
    local out="${REFINE_DIR}/WC_L${MTP_LEVEL}_${tag}.mtp"
    local rc=0
    echo
    echo "---------- stage: ${tag} ----------"
    TRAIN_START_MTP="${start}" \
    TRAIN_RESUME=0 \
    EFFECTIVE_MAX_ITER="${maxit}" \
    EFFECTIVE_ENERGY_WEIGHT="${ew}" \
    EFFECTIVE_FORCE_WEIGHT="${fw}" \
    EFFECTIVE_STRESS_WEIGHT="${sw}" \
    TRAIN_TAG="${tag}" \
    SKIP_MINDIST="${SKIP_MINDIST_STAGE:-0}" \
    bash "${SCRIPT_DIR}/train_mtp.sh" "${cfg}" "${out}" || rc=$?

    if [[ -f "${out}" ]] && is_fitted_mtp "${out}"; then
        BEST_MTP="${out}"
        cp "${out}" "${TRAINED_MTP}"
        echo "  Promoted stage pot -> ${TRAINED_MTP}"
    elif [[ ${rc} -ne 0 ]]; then
        echo "  train_mtp stage ${tag} failed (rc=${rc}) and no fitted pot at ${out}" >&2
        return "${rc}"
    fi

    if [[ -f "${LOGDIR}/train_status.env" ]]; then
        # shellcheck disable=SC1090
        source "${LOGDIR}/train_status.env" || true
    fi

    # If BFGS finished but status/errors were not written, salvage from train log
    # so the refine loop still sees STEP_LIMIT / FORCE_RMS.
    if [[ ${rc} -ne 0 ]] || [[ -z "${FORCE_RMS:-}" ]]; then
        local tlog="${LOGDIR}/${tag}.log"
        if train_log_bfgs_finished "${tlog}"; then
            echo "  WARNING: stage ${tag} incomplete postproc — salvaging status from ${tlog}" >&2
            if _refine_salvage_status_from_log "${tag}" "${out}" "${cfg}" "${tlog}"; then
                # shellcheck disable=SC1090
                source "${LOGDIR}/train_status.env" 2>/dev/null || true
                # Pot + metrics recovered: allow refine to continue.
                rc=0
            fi
        fi
    fi

    # Successful pot with metrics counts as stage success even if train_mtp
    # exited non-zero after writing the pot (e.g. mid-script crash in postproc).
    if [[ ${rc} -ne 0 && -n "${FORCE_RMS:-}" && -f "${out}" ]] && is_fitted_mtp "${out}"; then
        echo "  WARNING: train_mtp rc=${rc} but pot+FORCE_RMS present; continuing refine" >&2
        rc=0
    fi

    return "${rc}"
}

# Write calc_errors_*.log + train_status.env from an mlp train log that already
# contains "* * * TRAIN ERRORS * * *" (e.g. after a mid-script crash).
_refine_salvage_status_from_log() {
    local tag="$1" pot="$2" cfg="$3" tlog="$4"
    local err="${LOGDIR}/calc_errors_${tag}.log"
    local force_rms force_mae epa_rms last_f step_limit=0 small_decr=0

    [[ -f "${tlog}" ]] || return 1
    awk '
        /\* \* \* TRAIN ERRORS \* \* \*/ {grab=1}
        grab {print}
        /^_{10,}/ && seen_header {if (++ends>=2) exit}
        grab && /Errors report/ {seen_header=1}
    ' "${tlog}" > "${err}"
    if ! grep -q "Errors report" "${err}" 2>/dev/null; then
        rm -f "${err}"
        return 1
    fi
    cp_unless_same "${err}" "${LOGDIR}/calc_errors_train.log"

    force_rms="$(python3 - "${err}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"Forces:[\s\S]*?RMS\s+absolute difference = ([0-9.eE+-]+)", text)
print(m.group(1) if m else "")
PY
)"
    force_mae="$(python3 - "${err}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"Forces:[\s\S]*?Average absolute difference = ([0-9.eE+-]+)", text)
print(m.group(1) if m else "")
PY
)"
    epa_rms="$(python3 - "${err}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"Energy per atom:[\s\S]*?RMS\s+absolute difference = ([0-9.eE+-]+)", text)
print(m.group(1) if m else "")
PY
)"
    last_f="$(grep -E 'BFGS iter [0-9]+:' "${tlog}" 2>/dev/null | tail -1 \
        | sed -E 's/.*f=([0-9.eE+-]+).*/\1/' || true)"
    grep -q "step limit reached" "${tlog}" 2>/dev/null && step_limit=1
    grep -q "BFGS ended due to small decr" "${tlog}" 2>/dev/null && small_decr=1

    {
        echo "TRAIN_TAG=$(printf '%q' "${tag}")"
        echo "OUTPUT_MTP=$(printf '%q' "${pot}")"
        echo "TRAIN_SET=$(printf '%q' "${cfg}")"
        echo "START_MTP=$(printf '%q' "${pot}")"
        echo "START_SOURCE=$(printf '%q' "salvaged from ${tag}.log")"
        echo "CONTINUING=1"
        echo "STEP_LIMIT=${step_limit}"
        echo "BFGS_SMALL_DECR=${small_decr}"
        echo "LAST_BFGS_F=$(printf '%q' "${last_f}")"
        echo "FORCE_RMS=$(printf '%q' "${force_rms}")"
        echo "FORCE_MAE=$(printf '%q' "${force_mae}")"
        echo "EPA_RMS=$(printf '%q' "${epa_rms}")"
        echo "MAX_ITER=${EFFECTIVE_MAX_ITER:-${MAX_ITER:-2000}}"
        echo "ENERGY_WEIGHT=${EFFECTIVE_ENERGY_WEIGHT:-${ENERGY_WEIGHT}}"
        echo "FORCE_WEIGHT=${EFFECTIVE_FORCE_WEIGHT:-${FORCE_WEIGHT}}"
        echo "STRESS_WEIGHT=${EFFECTIVE_STRESS_WEIGHT:-${STRESS_WEIGHT}}"
        echo "POSTPROC_PENDING=0"
        echo "TRAIN_N_CFG=$(cfg_n_configurations "${cfg}" 2>/dev/null || echo "")"
    } > "${LOGDIR}/train_status.env"

    FORCE_RMS="${force_rms}"
    FORCE_MAE="${force_mae}"
    EPA_RMS="${epa_rms}"
    STEP_LIMIT="${step_limit}"
    BFGS_SMALL_DECR="${small_decr}"
    LAST_BFGS_F="${last_f}"
    echo "  Salvaged force_rms=${force_rms} step_limit=${step_limit} last_f=${last_f}"
}

force_rms_ok() {
    local rms="${FORCE_RMS:-}"
    if [[ -z "${rms}" ]]; then
        return 1
    fi
    python3 -c "import sys; sys.exit(0 if float('${rms}') <= float('${VAL_FORCE_RMS_MAX}') else 1)"
}

bfgs_needs_continue() {
    # Continue if step limit hit (and not already small-decr converged).
    [[ "${STEP_LIMIT:-0}" == "1" ]]
}

report_mindist_vs_cutoff() {
    local cfg="$1"
    echo
    echo "---------- stage: mindist vs min cutoff ----------"
    # Global mindist from mlp mindist log or Feature mindist scan.
    local global_md
    global_md="$(python3 - "${cfg}" <<'PY'
import re, sys
md = None
pat = re.compile(r"Feature\s+mindist\s+([0-9.eE+-]+)", re.I)
# also tab-separated
pat2 = re.compile(r"Feature\s+mindist\s*[\t ]+([0-9.eE+-]+)", re.I)
for line in open(sys.argv[1]):
    m = pat2.search(line) or pat.search(line)
    if m:
        v = float(m.group(1))
        md = v if md is None else min(md, v)
print(f"{md:.8g}" if md is not None else "")
PY
)"
    if [[ -z "${global_md}" ]]; then
        echo "  Could not read mindist Features; running mlp mindist..."
        run_mlp_serial mindist "${cfg}" 2>&1 | tee "${LOGDIR}/mindist_refine.log" || true
        global_md="$(grep -iE 'Global mindist|mindist' "${LOGDIR}/mindist_refine.log" 2>/dev/null \
            | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    echo "  Dataset global mindist: ${global_md:-unknown} A"
    echo "  MTP template min_dist:  ${MTP_MIN_DIST} A"
    echo "  Trained pot min_dist:   $(grep -m1 'min_dist' "${BEST_MTP}" | awk -F= '{print $2}' | tr -d ' ')"
    if [[ -n "${global_md}" ]]; then
        python3 - "${global_md}" "${MTP_MIN_DIST}" <<'PY'
import sys
md, cut = float(sys.argv[1]), float(sys.argv[2])
if md < cut:
    print(f"  WARNING: short pairs remain (mindist={md:.4f} < MTP_MIN_DIST={cut}).")
    print("  Action: lower MTP_MIN_DIST toward ~0.99*mindist, or exclude those configs,")
    print("  then retrain (mlp train --update-mindist already lowers pot min_dist).")
    sys.exit(2)
else:
    print(f"  OK: mindist ({md:.4f} A) >= MTP_MIN_DIST ({cut} A).")
    sys.exit(0)
PY
        return $?
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Stage 0: filter bad / irrelevant DFT
# ---------------------------------------------------------------------------
WORK_CFG="${TRAIN_SET}"
if [[ "${FILTER_BAD_DFT}" == "1" ]]; then
    echo "---------- stage 0: filter bad / irrelevant DFT ----------"
    CLEAN_CFG="${REFINE_DIR}/train_filtered.cfg"
    REJECT_CFG="${REFINE_DIR}/train_rejected.cfg"
    if python3 "${SCRIPT_DIR}/filter_cfg.py" \
        "${TRAIN_SET}" "${CLEAN_CFG}" \
        --rejected-cfg "${REJECT_CFG}" \
        --max-force "${FILTER_MAX_FORCE}" \
        --min-dist "${FILTER_MIN_DIST_ABS}" \
        --max-epa-outlier "${FILTER_MAX_EPA_OUTLIER}"; then
        N_CLEAN="$(python3 - "${CLEAN_CFG}" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1]) if line.strip() == "BEGIN_CFG"))
PY
)"
        N_ORIG="$(python3 - "${TRAIN_SET}" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1]) if line.strip() == "BEGIN_CFG"))
PY
)"
        if [[ "${N_CLEAN}" -lt "${N_ORIG}" ]]; then
            echo "  Filtered ${N_ORIG} -> ${N_CLEAN}; using cleaned set for refine."
            # Backup original once, install clean as active train.cfg if paths match
            if [[ "$(readlink -f "${TRAIN_SET}")" == "$(readlink -f "${TRAIN_CFG}")" ]]; then
                if [[ ! -f "${MTP_DATASETS_DIR}/initial/train_before_filter.cfg" ]]; then
                    cp "${TRAIN_SET}" "${MTP_DATASETS_DIR}/initial/train_before_filter.cfg"
                fi
                cp "${CLEAN_CFG}" "${TRAIN_CFG}"
                WORK_CFG="${TRAIN_CFG}"
            else
                WORK_CFG="${CLEAN_CFG}"
            fi
        else
            echo "  No configs filtered."
            WORK_CFG="${TRAIN_SET}"
        fi
    else
        echo "  Filter failed; continuing with original train set." >&2
        WORK_CFG="${TRAIN_SET}"
    fi
fi

# Seed status from existing errors / train log (nohup: step limit + force RMS fail)
FORCE_RMS=""
STEP_LIMIT=0
BFGS_SMALL_DECR=0
if [[ -f "${LOGDIR}/calc_errors_train.log" ]]; then
    FORCE_RMS="$(python3 - "${LOGDIR}/calc_errors_train.log" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"Forces:[\s\S]*?RMS\s+absolute difference = ([0-9.eE+-]+)", text)
print(m.group(1) if m else "")
PY
)"
fi
if [[ -f "${LOGDIR}/train.log" ]] && grep -q "step limit reached" "${LOGDIR}/train.log"; then
    STEP_LIMIT=1
fi
if [[ -f "${LOGDIR}/train_status.env" ]]; then
    # shellcheck disable=SC1090
    source "${LOGDIR}/train_status.env" || true
fi

# Bootstrap errors if we only have a pot (e.g. resume refine after crash)
if [[ -z "${FORCE_RMS}" && -f "${BEST_MTP}" ]]; then
    echo "Computing baseline errors for ${BEST_MTP}..."
    run_mlp calc-errors "${BEST_MTP}" "${WORK_CFG}" \
        2>&1 | tee "${LOGDIR}/calc_errors_train.log"
    FORCE_RMS="$(python3 - "${LOGDIR}/calc_errors_train.log" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"Forces:[\s\S]*?RMS\s+absolute difference = ([0-9.eE+-]+)", text)
print(m.group(1) if m else "")
PY
)"
fi

echo "Current force_rms=${FORCE_RMS:-unknown}  step_limit=${STEP_LIMIT}"

# If force RMS already OK and BFGS did not hit step limit, nothing to do.
if force_rms_ok && [[ "${STEP_LIMIT}" != "1" ]]; then
    echo "Already within force RMS threshold and BFGS settled — nothing to refine."
    python3 "${SCRIPT_DIR}/validate_mtp.py" "${LOGDIR}/calc_errors_train.log" \
        --force-rms-max "${VAL_FORCE_RMS_MAX}" \
        --force-mae-max "${VAL_FORCE_MAE_MAX}" \
        --energy-per-atom-max "${VAL_ENERGY_PER_ATOM_MAX}" \
        --stress-rms-max "${VAL_STRESS_RMS_MAX}" || true
    exit 0
fi

# ---------------------------------------------------------------------------
# Stage 1: more BFGS iterations + rescaling until step limit clears
# ---------------------------------------------------------------------------
# Skip continue rounds that already finished (resume after crash / postproc salvage).
round=0
while [[ "${round}" -lt "${TRAIN_CONTINUE_MAX_ROUNDS}" ]]; do
    next=$((round + 1))
    next_tag="continue_r${next}"
    next_pot="${REFINE_DIR}/WC_L${MTP_LEVEL}_${next_tag}.mtp"
    if train_fully_complete "${next_pot}" "${next_tag}" "${LOGDIR}"; then
        echo "---------- stage: ${next_tag} (already complete — skip) ----------"
        BEST_MTP="${next_pot}"
        cp "${next_pot}" "${TRAINED_MTP}"
        status_tag=""
        if [[ -f "${LOGDIR}/train_status.env" ]]; then
            status_tag="$(grep -E '^TRAIN_TAG=' "${LOGDIR}/train_status.env" 2>/dev/null \
                | head -1 | cut -d= -f2- | tr -d "\"'\\\\" || true)"
        fi
        if [[ "${status_tag}" != "${next_tag}" ]]; then
            _refine_salvage_status_from_log "${next_tag}" "${next_pot}" "${WORK_CFG}" \
                "${LOGDIR}/${next_tag}.log" 2>/dev/null || true
        fi
        # shellcheck disable=SC1090
        source "${LOGDIR}/train_status.env" 2>/dev/null || true
        echo "  Resumed from ${next_tag}: step_limit=${STEP_LIMIT:-?} force_rms=${FORCE_RMS:-?}"
        round="${next}"
        continue
    fi
    break
done

while bfgs_needs_continue && [[ "${round}" -lt "${TRAIN_CONTINUE_MAX_ROUNDS}" ]]; do
    round=$((round + 1))
    echo
    echo "BFGS step limit was hit — continue round ${round}/${TRAIN_CONTINUE_MAX_ROUNDS}"
    echo "(restart from best pot; BFGS Hessian resets; ends with linear fit + rescaling)"
    SKIP_MINDIST_STAGE=1 \
    run_train_stage \
        "continue_r${round}" \
        "${BEST_MTP}" \
        "${WORK_CFG}" \
        "${MAX_ITER_CONTINUE}" \
        "${ENERGY_WEIGHT}" \
        "${FORCE_WEIGHT}" \
        "${STRESS_WEIGHT}"
    # shellcheck disable=SC1090
    source "${LOGDIR}/train_status.env"
    echo "  After continue: step_limit=${STEP_LIMIT} force_rms=${FORCE_RMS} last_f=${LAST_BFGS_F}"
    if force_rms_ok && [[ "${STEP_LIMIT}" != "1" ]]; then
        echo "Force RMS met after BFGS continue."
        break
    fi
done

if [[ "${STEP_LIMIT}" == "1" ]]; then
    echo "WARNING: BFGS still hits step limit after ${TRAIN_CONTINUE_MAX_ROUNDS} continue rounds."
    echo "  Consider larger MAX_ITER_CONTINUE or accept and rely on force retrain / more data."
fi

# ---------------------------------------------------------------------------
# Stage 2: force-weighted retrain from best intermediate (reset BFGS)
# ---------------------------------------------------------------------------
if ! force_rms_ok; then
    echo
    echo "Force RMS ${FORCE_RMS:-?} > ${VAL_FORCE_RMS_MAX} — force-weighted retrain"
    SKIP_MINDIST_STAGE=1 \
    run_train_stage \
        "force_w" \
        "${BEST_MTP}" \
        "${WORK_CFG}" \
        "${FORCE_RETRAIN_MAX_ITER}" \
        "${FORCE_RETRAIN_ENERGY_WEIGHT}" \
        "${FORCE_WEIGHT_RETRAIN}" \
        "${FORCE_RETRAIN_STRESS_WEIGHT}"
    # shellcheck disable=SC1090
    source "${LOGDIR}/train_status.env"
    echo "  After force retrain: force_rms=${FORCE_RMS} step_limit=${STEP_LIMIT}"
fi

# ---------------------------------------------------------------------------
# Stage 3: optional high-error subset force-focused fit + full-set polish
# ---------------------------------------------------------------------------
if ! force_rms_ok && [[ "${ENABLE_HIGH_ERROR_RETRAIN}" == "1" ]]; then
    echo
    echo "---------- stage 3: high-error subset force fit ----------"
    PRED_CFG="${REFINE_DIR}/train_mtp_efs.cfg"
    HIGH_CFG="${REFINE_DIR}/high_force_error.cfg"
    echo "  calc-efs for per-config force errors..."
    run_mlp calc-efs "${BEST_MTP}" "${WORK_CFG}" "${PRED_CFG}" \
        2>&1 | tee "${LOGDIR}/calc_efs_refine.log"

    if python3 "${SCRIPT_DIR}/extract_high_error_cfg.py" \
        "${WORK_CFG}" "${PRED_CFG}" "${HIGH_CFG}" \
        --top-frac "${HIGH_ERROR_TOP_FRAC}" \
        --min-force-rmse "${HIGH_ERROR_MIN_FORCE_RMSE}" \
        --scores "${REFINE_DIR}/force_error_scores.tsv"; then
        N_HIGH="$(python3 - "${HIGH_CFG}" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1]) if line.strip() == "BEGIN_CFG"))
PY
)"
        if [[ "${N_HIGH}" -gt 0 ]]; then
            echo "  Force-focused train on ${N_HIGH} high-error configs"
            SKIP_MINDIST_STAGE=1 \
            run_train_stage \
                "higherr" \
                "${BEST_MTP}" \
                "${HIGH_CFG}" \
                "${HIGH_ERROR_MAX_ITER}" \
                "${HIGH_ERROR_ENERGY_WEIGHT}" \
                "${HIGH_ERROR_FORCE_WEIGHT}" \
                "${HIGH_ERROR_STRESS_WEIGHT}"

            echo "  Polish on full training set from high-error pot"
            SKIP_MINDIST_STAGE=1 \
            run_train_stage \
                "polish" \
                "${BEST_MTP}" \
                "${WORK_CFG}" \
                "${HIGH_ERROR_POLISH_MAX_ITER}" \
                "${ENERGY_WEIGHT}" \
                "${FORCE_WEIGHT_RETRAIN}" \
                "${STRESS_WEIGHT}"
            # shellcheck disable=SC1090
            source "${LOGDIR}/train_status.env"
        else
            echo "  No high-error configs selected; skipping."
        fi
    else
        echo "  High-error extraction failed; skipping subset retrain." >&2
    fi
fi

# ---------------------------------------------------------------------------
# Stage 4: mindist vs min cutoff
# ---------------------------------------------------------------------------
MD_STATUS=0
report_mindist_vs_cutoff "${WORK_CFG}" || MD_STATUS=$?

# ---------------------------------------------------------------------------
# Stage 5: final validation + next-step recommendations
# ---------------------------------------------------------------------------
echo
echo "---------- final validation ----------"
ERR_LOG="${LOGDIR}/calc_errors_train.log"
if [[ -f "${ERR_LOG}" ]]; then
    if python3 "${SCRIPT_DIR}/validate_mtp.py" "${ERR_LOG}" \
        --force-rms-max "${VAL_FORCE_RMS_MAX}" \
        --force-mae-max "${VAL_FORCE_MAE_MAX}" \
        --energy-per-atom-max "${VAL_ENERGY_PER_ATOM_MAX}" \
        --stress-rms-max "${VAL_STRESS_RMS_MAX}"; then
        echo
        echo "REFINE: PASSED — force RMS within threshold."
        echo "  Best MTP: ${TRAINED_MTP}"
        exit 0
    fi
fi

echo
echo "REFINE: NOT converged (force_rms=${FORCE_RMS:-unknown} threshold=${VAL_FORCE_RMS_MAX})"
echo
echo "Practical next steps:"
echo "  2. Active learning on high-force-error / high-grade configs:"
echo "       ./run.sh --skip-dataset --al"
if [[ -f "${REFINE_DIR}/high_force_error.cfg" ]]; then
    echo "     High-error DFT subset (for labeling prioritization / candidates):"
    echo "       ${REFINE_DIR}/high_force_error.cfg"
fi
echo "  3. If still stuck after more labeled data: raise level"
echo "       MTP_LEVEL=${ESCALATE_LEVEL} ./run.sh --skip-dataset"
if [[ "${MD_STATUS}" == "2" ]]; then
    echo "  4. Short pairs remain: lower MTP_MIN_DIST or exclude them, then retrain"
    echo "       MTP_MIN_DIST=<slightly below mindist> ./scripts/refine_mtp.sh"
fi

if [[ "${AUTO_ESCALATE_LEVEL}" == "1" && "${MTP_LEVEL}" -lt "${ESCALATE_LEVEL}" ]]; then
    echo
    echo "AUTO_ESCALATE_LEVEL=1: retraining at level ${ESCALATE_LEVEL}"
    # Fresh level-22 template cannot reuse level-20 coeffs.
    MTP_LEVEL="${ESCALATE_LEVEL}" \
    TRAIN_START_MTP="" \
    TRAIN_RESUME=0 \
    EFFECTIVE_MAX_ITER="${MAX_ITER}" \
    TRAIN_TAG="level${ESCALATE_LEVEL}" \
    bash "${SCRIPT_DIR}/train_mtp.sh" "${WORK_CFG}" \
        "${MTP_AL_DIR}/WC_L${ESCALATE_LEVEL}_trained.mtp"
    # Point default trained pot if user wants L22 as current
    if [[ -f "${MTP_AL_DIR}/WC_L${ESCALATE_LEVEL}_trained.mtp" ]]; then
        echo "  Level ${ESCALATE_LEVEL} pot: ${MTP_AL_DIR}/WC_L${ESCALATE_LEVEL}_trained.mtp"
        echo "  Re-run refine with MTP_LEVEL=${ESCALATE_LEVEL} if needed."
    fi
fi

exit 1
