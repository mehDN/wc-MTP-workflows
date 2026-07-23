#!/bin/bash
# Backward-compatible alias for run_all_trajectories.sh (TiC-era name).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_all_trajectories.sh" "$@"