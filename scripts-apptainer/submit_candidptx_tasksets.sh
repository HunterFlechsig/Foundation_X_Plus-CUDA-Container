#!/bin/bash
# Submit the seven CANDID-PTX task-set jobs on public/public under grp_jliang12.
# Each job uses two A100s. Wall time is 1 day for one task, 2 days for two, 3 days for all three.
#
#   ./scripts-apptainer/submit_candidptx_tasksets.sh

set -euo pipefail

cd "$(dirname "$0")/.."

submit() {
	local name="$1"
	local wall="$2"
	sbatch \
		--job-name="candidptx_${name}" \
		--time="$wall" \
		--account=grp_jliang12 \
		--partition=public \
		--qos=public \
		--gres=gpu:a100:2 \
		--cpus-per-task=10 \
		--mem=100G \
		scripts-apptainer/run_apptainer_candidptx_taskset.sh "$name"
}

submit a 1-00:00:00
submit b 1-00:00:00
submit c 1-00:00:00
submit d 2-00:00:00
submit e 2-00:00:00
submit f 2-00:00:00
submit g 3-00:00:00
