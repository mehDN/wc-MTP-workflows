#!/bin/bash
# Backward-compatible wrapper — delegates to ../run.sh
#
# Usage:
#   ./scripts/run_workflow.sh [options]
#   ./scripts/run_workflow.sh init          -> full pipeline
#   ./scripts/run_workflow.sh train         -> skip dataset
#   ./scripts/run_workflow.sh al-loop FILE  -> full + active learning

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="${SCRIPT_DIR}/../run.sh"

if [[ ! -x "${RUN_SH}" ]]; then
    chmod +x "${RUN_SH}"
fi

CMD="${1:-}"
shift || true

case "${CMD}" in
    ""|init|all|full)
        exec "${RUN_SH}" "$@"
        ;;
    train)
        exec "${RUN_SH}" --skip-dataset "$@"
        ;;
    dataset|build)
        exec "${RUN_SH}" --only dataset "$@"
        ;;
    validate)
        exec "${RUN_SH}" --skip-dataset --skip-train "$@"
        ;;
    al|al-loop)
        if [[ -n "${1:-}" ]]; then
            exec "${RUN_SH}" --skip-dataset --al --candidates "${1}"
        else
            exec "${RUN_SH}" --skip-dataset --al
        fi
        ;;
    per-traj|per-temp)
        exec "${RUN_SH}" --skip-dataset --per-traj "$@"
        ;;
    -h|--help|help)
        exec "${RUN_SH}" --help
        ;;
    *)
        echo "Unknown command: ${CMD}" >&2
        echo "Use ./run.sh --help or ./scripts/run_workflow.sh --help" >&2
        exit 1
        ;;
esac