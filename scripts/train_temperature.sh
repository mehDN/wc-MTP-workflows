#!/bin/bash
# Backward-compatible alias for train_trajectory.sh (TiC-era name).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_trajectory.sh" "$@"