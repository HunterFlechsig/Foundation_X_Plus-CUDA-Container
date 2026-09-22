#!/usr/bin/env bash
set -euo pipefail

# Submit the seven CANDID-PTX-only Foundation X+ Apptainer jobs
# on SOL public/public under grp_jliang12 (not preemptable).
# Usage, from the container repo root or anywhere:
#   ./scripts-apptainer/submit_candidptx_all.sh

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNNER="$ROOT/scripts-apptainer/run_apptainer_candidptx.sh"
ACCOUNT="${ACCOUNT:-grp_jliang12}"
PARTITION="${PARTITION:-public}"
QOS="${QOS:-public}"

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
	echo "Submitting $experiment  (-A ${ACCOUNT} -p ${PARTITION} -q ${QOS})"
	sbatch -A "$ACCOUNT" -p "$PARTITION" -q "$QOS" \
		--chdir="$ROOT" --export=ALL,REPO_ROOT="$ROOT" \
		--job-name="fx_${experiment}" "$RUNNER" "$experiment"
done
