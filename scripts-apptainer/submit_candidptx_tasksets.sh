#!/bin/bash
# Submit fresh 50-cycle CANDID-PTX jobs for f and g only.
# a–c are resumed by hand. d and e wait until their 10-cycle jobs finish.
#
#   ./scripts-apptainer/submit_candidptx_tasksets.sh

set -euo pipefail

cd "$(dirname "$0")/.."

submit() {
	local name="$1"
	sbatch \
		--job-name="candidptx_${name}" \
		--time=7-00:00:00 \
		--account=grp_jliang12 \
		--partition=public \
		--qos=public \
		--gres=gpu:a100:2 \
		--cpus-per-task=10 \
		--mem=100G \
		scripts-apptainer/run_apptainer_candidptx_taskset.sh "$name"
}

submit f
submit g
