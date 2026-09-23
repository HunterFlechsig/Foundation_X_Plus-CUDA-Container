#!/usr/bin/env bash
set -euo pipefail

# Build CANDID-PTX split lists and localization PNGs from the raw release.
#   ./scripts-apptainer/prepare_candidptx_splits.sh /scratch/$USER/CANDID-PTX
#   ./scripts-apptainer/prepare_candidptx_splits.sh /scratch/$USER/CANDID-PTX --skip-png

ROOT="/scratch/${USER}/CANDID-PTX"
if [[ $# -gt 0 && "$1" != --* ]]; then
    ROOT="$1"
    shift
fi

exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prepare_candidptx_splits.py" --root "$ROOT" "$@"
