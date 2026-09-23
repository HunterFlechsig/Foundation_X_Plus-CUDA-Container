#!/usr/bin/env bash
set -euo pipefail

# Build CANDID-PTX split lists and localization PNGs from the raw release.
#   ./scripts-apptainer/prepare_candidptx_splits.sh /data/jliang12/shared/dataset/CANDID-PTX
#   ./scripts-apptainer/prepare_candidptx_splits.sh /data/jliang12/shared/dataset/CANDID-PTX --skip-png
#
# Host python on SOL does not have pydicom. This falls through to the CUDA
# Apptainer image, which installs it from requirements-cuda.txt.

ROOT="/data/jliang12/shared/dataset/CANDID-PTX"
if [[ $# -gt 0 && "$1" != --* ]]; then
    ROOT="$1"
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$SCRIPT_DIR/prepare_candidptx_splits.py"

if python3 -c "import pydicom, numpy, PIL" >/dev/null 2>&1; then
    exec python3 "$PY" --root "$ROOT" "$@"
fi

IMAGE="$REPO_DIR/apptainer-cuda.sif"
if [[ ! -f "$IMAGE" ]]; then
    echo "Host python3 cannot import pydicom, and $IMAGE is missing." >&2
    echo "Build it with ./cuda-apptainer.sh build, or install pydicom, numpy, and Pillow for this python3." >&2
    exit 1
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "Host python3 cannot import pydicom, and apptainer is not on PATH." >&2
    exit 1
fi

echo "Host python3 has no pydicom. Running inside $IMAGE."
binds=()
for dir in /scratch /data; do
    if [[ -d "$dir" ]]; then
        binds+=(--bind "$dir:$dir:rw")
    fi
done
exec apptainer exec "${binds[@]}" --bind "$REPO_DIR:/workspace" --pwd /workspace "$IMAGE" \
    python3 scripts-apptainer/prepare_candidptx_splits.py --root "$ROOT" "$@"
