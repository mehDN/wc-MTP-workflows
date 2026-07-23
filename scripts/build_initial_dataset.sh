#!/bin/bash
# Assemble the initial DFT training set from W-vacancy AIMD trajectories and
# categorized external sources (bulk, defect, high-T, close C-C, surfaces, NEB).
#
# Usage:
#   ./scripts/build_initial_dataset.sh
#
# Environment overrides: see mtp_config.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mtp_config.sh
source "${SCRIPT_DIR}/mtp_config.sh"

WORK="${MTP_DATASETS_DIR}/initial"
STAGING="${WORK}/staging"
mkdir -p "${STAGING}" "${WORK}/logs"

TRAIN_FULL="${WORK}/train_full.cfg"
TRAIN_OUT="${TRAIN_CFG}"
MANIFEST="${WORK}/manifest.txt"
COMPOSITION="${WORK}/composition.txt"
: > "${MANIFEST}"
: > "${COMPOSITION}"

echo "=== Building initial WC training dataset ==="
echo "Project:  ${MTP_PROJECT_ROOT}"
echo "Output:   ${TRAIN_OUT}"
echo "Targets:  500-2000 structures; see ${SOURCES_CONF}"
echo

CFG_PARTS=()
declare -A CATEGORY_COUNTS=()
PART_IDX=0

add_cfg_part() {
    local part="$1"
    local label="$2"
    local category="${3:-aimd}"
    if [[ -f "${part}" ]]; then
        CFG_PARTS+=("${part}")
        local n
        n="$(python3 - "${part}" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1]) if line.strip() == "BEGIN_CFG"))
PY
)"
        echo "${label} [${category}] (${n} cfg) -> ${part}" >> "${MANIFEST}"
        CATEGORY_COUNTS["${category}"]=$(( ${CATEGORY_COUNTS["${category}"]:-0} + n ))
        PART_IDX=$((PART_IDX + 1))
    fi
}

convert_outcar() {
    local outcar="$1"
    local tag="$2"
    local out="${STAGING}/${tag}.cfg"
    if [[ ! -f "${outcar}" ]]; then
        echo "  skip (missing): ${outcar}" >&2
        return 0
    fi
    if [[ -s "${out}" ]]; then
        echo "[convert] reuse ${out}"
        return 0
    fi
    echo "[convert] ${outcar}"
    rm -f "${out}"
    if ! nice -n 19 "${MLP}" convert-cfg --input-format=vasp-outcar "${outcar}" "${out}" \
        >> "${WORK}/logs/convert.log" 2>&1; then
        echo "  WARNING: conversion failed for ${outcar}" >&2
        rm -f "${out}"
        return 0
    fi
    if [[ ! -s "${out}" ]]; then
        echo "  WARNING: empty cfg from ${outcar}" >&2
        rm -f "${out}"
    fi
}

subsample_aimd() {
    local cfg_in="$1"
    local stride="$2"
    local tag="$3"
    local category="$4"
    local out="${STAGING}/${tag}_s${stride}.cfg"
    if [[ ! -s "${cfg_in}" ]]; then
        echo "  skip subsample (invalid input): ${cfg_in}" >&2
        return 0
    fi
    if [[ -s "${out}" ]]; then
        echo "[subsample] reuse ${out}"
    else
        python3 "${SCRIPT_DIR}/subsample_cfg.py" "${cfg_in}" "${out}" --stride "${stride}" \
            >> "${WORK}/logs/subsample.log" 2>&1
    fi
    add_cfg_part "${out}" "${tag} (stride=${stride})" "${category}"
}

# Infer composition category from W-vacancy AIMD folder name.
aimd_category() {
    local folder="$1"
    local temp="${AIMD_SOURCE_TEMPS[$folder]:-0}"
    if (( temp >= HIGH_T_THRESHOLD_K )); then
        echo "high_t"
    else
        echo "defect"
    fi
}

# --- 1. W-vacancy AIMD trajectories ---
echo "[1/3] W-vacancy AIMD OUTCARs (${#AIMD_SOURCES[@]} sources)"
for FOLDER in "${AIMD_SOURCES[@]}"; do
    OUTCAR="${MTP_PROJECT_ROOT}/${FOLDER}/OUTCAR"
    if [[ ! -f "${OUTCAR}" ]]; then
        echo "  skip ${FOLDER} (no OUTCAR)"
        continue
    fi
    # Skip truncated/failed runs (< 1 MB)
    local_size=$(stat -c %s "${OUTCAR}" 2>/dev/null || stat -f %z "${OUTCAR}")
    if (( local_size < 1000000 )); then
        echo "  skip ${FOLDER} (OUTCAR too small: ${local_size} bytes)"
        continue
    fi

    TEMP="${AIMD_SOURCE_TEMPS[$FOLDER]:-unknown}"
    RAW="${STAGING}/aimd_${FOLDER}.cfg"
    convert_outcar "${OUTCAR}" "aimd_${FOLDER}"
    if [[ -s "${RAW}" ]]; then
        STRIDE="${SUBSAMPLE_STRIDE}"
        if [[ "${TEMP}" =~ ^[0-9]+$ ]] && (( TEMP >= HIGH_T_THRESHOLD_K )); then
            STRIDE="${HIGH_T_STRIDE}"
        fi
        CAT="$(aimd_category "${FOLDER}")"
        subsample_aimd "${STAGING}/aimd_${FOLDER}.cfg" "${STRIDE}" "aimd_${FOLDER}" "${CAT}"
    fi
done

# --- 2. External sources from sources.conf (category:kind:path) ---
echo
echo "[2/3] External DFT sources (${SOURCES_CONF})"
if [[ -f "${SOURCES_CONF}" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%%#*}"
        line="$(echo "${line}" | xargs)"
        [[ -z "${line}" ]] && continue

        category="external"
        kind=""
        path=""

        IFS=':' read -r f1 f2 f3 <<< "${line}"
        if [[ -n "${f3}" ]]; then
            category="${f1}"
            kind="${f2}"
            path="${f3}"
        else
            kind="${f1}"
            path="${f2}"
        fi

        # Expand project-root placeholder
        path="${path//\$\{MTP_PROJECT_ROOT\}/${MTP_PROJECT_ROOT}}"

        case "${kind}" in
            outcar)
                convert_outcar "${path}" "ext_${category}_${PART_IDX}"
                add_cfg_part "${STAGING}/ext_${category}_${PART_IDX}.cfg" \
                    "external outcar: ${path}" "${category}"
                ;;
            cfg)
                add_cfg_part "${path}" "external cfg: ${path}" "${category}"
                ;;
            dir)
                if [[ ! -d "${path}" ]]; then
                    echo "  skip dir (missing): ${path}" >&2
                    continue
                fi
                shopt -s nullglob
                for f in "${path}"/*.cfg; do
                    add_cfg_part "${f}" "external cfg: ${f}" "${category}"
                done
                for f in "${path}"/*/OUTCAR "${path}"/OUTCAR; do
                    [[ -f "${f}" ]] || continue
                    convert_outcar "${f}" "ext_${category}_dir_${PART_IDX}_$(basename "$(dirname "${f}")")"
                    add_cfg_part "${STAGING}/ext_${category}_dir_${PART_IDX}_$(basename "$(dirname "${f}")").cfg" \
                        "external outcar: ${f}" "${category}"
                    PART_IDX=$((PART_IDX + 1))
                done
                shopt -u nullglob
                ;;
            *)
                echo "  unknown source type '${kind}' in: ${line}" >&2
                ;;
        esac
    done < "${SOURCES_CONF}"
else
    echo "  No sources.conf found; AIMD-only dataset."
    echo "  Add bulk/defect/static configs to ${SOURCES_CONF}"
fi

# --- 3. Merge ---
echo
echo "[3/3] Merging ${#CFG_PARTS[@]} cfg parts"
if [[ ${#CFG_PARTS[@]} -eq 0 ]]; then
    echo "No configurations collected." >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/merge_cfg.py" "${TRAIN_FULL}" "${CFG_PARTS[@]}" --dedupe \
    | tee "${WORK}/logs/merge.log"

cp "${TRAIN_FULL}" "${TRAIN_OUT}"

TOTAL=0
{
    echo "WC training-set composition"
    echo "========================="
    for cat in bulk defect high_t close_cc surface pathway aimd external; do
        count="${CATEGORY_COUNTS[$cat]:-0}"
        if (( count > 0 )); then
            TOTAL=$((TOTAL + count))
            printf "  %-12s %6d\n" "${cat}:" "${count}"
        fi
    done
    echo "  ----------------------"
    printf "  %-12s %6d\n" "TOTAL:" "${TOTAL}"
    echo
    echo "Target fractions: bulk 25-30%, defect 20-25%, high_t 15-20%,"
    echo "  close_cc 10-15%, surface 10%, pathway remainder."
} | tee "${COMPOSITION}"

echo
echo "Initial dataset ready:"
echo "  ${TRAIN_OUT}"
echo "  manifest:    ${MANIFEST}"
echo "  composition: ${COMPOSITION}"
echo "  logs:        ${WORK}/logs/"