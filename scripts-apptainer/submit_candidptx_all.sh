#!/usr/bin/env bash
set -euo pipefail

# Submit the seven CANDID-PTX-only Foundation X+ Apptainer jobs.
# Usage, from the container repo root or anywhere:
#   ./scripts-apptainer/submit_candidptx_all.sh

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNNER="$ROOT/scripts-apptainer/run_apptainer_candidptx.sh"

experiments=(
	candidptx_cls
	candidptx_loc
	candidptx_seg
	candidptx_cls_loc
	candidptx_cls_seg
	candidptx_loc_seg
	candidptx_cls_loc_seg
)

cd "$ROOT"

for experiment in "${experiments[@]}"; do
	echo "Submitting $experiment"
	sbatch --job-name="fx_${experiment}" "$RUNNER" "$experiment"
done
