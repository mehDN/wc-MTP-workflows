#!/bin/bash
# Grade exploratory MD/LAMMPS frames and select high-uncertainty configs for DFT.
#
# Usage:
#   ./scripts/select_from_md.sh [md_frames.cfg] [iteration_label]
#
# If md_frames.cfg is omitted, uses the newest file in datasets/candidates/.
#
# Typical workflow:
#   1. Run NVT/NPT MD with the current MTP (LAMMPS-MLIP)
#   2. Dump frames to datasets/candidates/md_frames.cfg
#   3. Run this script (no arguments needed if file is in candidates/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mtp_config.sh
source "${SCRIPT_DIR}/mtp_config.sh"

MD_CFG=""
if [[ $# -ge 1 && -f "${1}" ]]; then
    MD_CFG="$1"
    shift
elif ! MD_CFG="$(resolve_al_candidate_cfg)"; then
    echo "No MD frames found. Place .cfg files in ${AL_CANDIDATES_DIR}/" >&2
    exit 1
fi

LABEL="${1:-md_$(date +%Y%m%d_%H%M%S)}"

bash "${SCRIPT_DIR}/active_learning.sh" "${MD_CFG}" "${LABEL}"