#!/bin/bash
# Pretrain one CANDID-PTX task set with the v110 Apptainer launcher.
# Two A100s, torchrun, batch size 12 per GPU.
#
# Build the split files once, on a login node, before the first run:
#   python scripts/build_candidptx_splits.py /scratch/hflechsi/CANDID-PTX
#
# Smoke (loc then seg, own output dir):
#   sbatch scripts-apptainer/sbatch_candidptx_smoke.sh
#   interactive -p htc -q public -A grp_jliang12 -G a100:2 -c 10 --mem=100G -t 0-4
#   LOGFILE=.../smoke_f TOTAL_EPOCHS=3 ./scripts-apptainer/run_apptainer_candidptx_taskset.sh smoke_f
#
# SOL command sequence:
#   scripts-apptainer/candidptx_sol_commands.md
#
# Production jobs (50 cycles, 7-day wall; f and g only — a–e are resumed by hand):
#   ./scripts-apptainer/submit_candidptx_tasksets.sh
#   RESUME=/scratch/hflechsi/FoundationX/candidptx_tasksets/a/ckpt_E10_TH9.pth \
#     TOTAL_EPOCHS=51 sbatch --time=7-00:00:00 scripts-apptainer/run_apptainer_candidptx_taskset.sh a
#
# The training loop starts at epoch 1 and stops before total_epochs, so
# total_epochs is one past the last epoch. Fifty cycles:
#   a  candidptxCLS                              epochs 1-50    total_epochs 51    wall 7 days
#   b  candidptxLOC                              epochs 1-50    total_epochs 51    wall 7 days
#   c  candidptxSEG                              epochs 1-50    total_epochs 51    wall 7 days
#   d  candidptxCLS_candidptxLOC                 epochs 1-100   total_epochs 101   wall 7 days
#   e  candidptxCLS_candidptxSEG                 epochs 1-100   total_epochs 101   wall 7 days
#   f  candidptxLOC_candidptxSEG                 epochs 1-100   total_epochs 101   wall 7 days
#   g  candidptxCLS_candidptxLOC_candidptxSEG    epochs 1-150   total_epochs 151   wall 7 days
#   smoke_f  same tasks as f, one cycle          epochs 1-2     total_epochs 3
#   smoke    same tasks as g, one cycle          epochs 1-3     total_epochs 4

#SBATCH --job-name=candidptx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH --gres=gpu:a100:2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

NAME="${1:-}"
case "$NAME" in
	a) cyclictask=candidptxCLS ;;
	b) cyclictask=candidptxLOC ;;
	c) cyclictask=candidptxSEG ;;
	d) cyclictask=candidptxCLS_candidptxLOC ;;
	e) cyclictask=candidptxCLS_candidptxSEG ;;
	f) cyclictask=candidptxLOC_candidptxSEG ;;
	g) cyclictask=candidptxCLS_candidptxLOC_candidptxSEG ;;
	smoke_f) cyclictask=candidptxLOC_candidptxSEG ;;
	smoke) cyclictask=candidptxCLS_candidptxLOC_candidptxSEG ;;
	*)
		echo "Usage: $0 {a|b|c|d|e|f|g|smoke_f|smoke}" >&2
		exit 1
		;;
esac

IFS=_ read -ra TOKENS <<< "$cyclictask"
ntasks="${#TOKENS[@]}"
if [[ "$NAME" == "smoke_f" ]]; then
	total_epochs="${TOTAL_EPOCHS:-3}"
elif [[ "$NAME" == "smoke" ]]; then
	total_epochs="${TOTAL_EPOCHS:-4}"
else
	total_epochs="${TOTAL_EPOCHS:-$((50 * ntasks + 1))}"
fi

CONFIGFILE=config/DINO/DINO_4scale_swinBASE.py
LOGFILE="${LOGFILE:-${SCRATCH:-/scratch/$USER}/FoundationX/candidptx_tasksets/${NAME}}"
backbone_dir=/scratch/sejong/class-dataset/models/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar
BACKBONEMODEL=Swin-B
IMGSIZE=224
coco_path=/scratch/sejong/class-dataset/VinDr-CXR/
DATASETFILE=foundation6Ark6_datasets
lr_backbone=1e-5
lr_locEnc=1e-4
lr_locDec=1e-4
lr_segmentor=1e-4
BATCHSIZE="${BATCHSIZE:-12}"
num_workers="${num_workers:-12}"
INIT="${INIT:-ark}"
opt="${opt:-adamw}"
EMAMODE="${EMAMODE:-True_Epoch}"
DATASET_LOCATIONS_YML="${DATASET_LOCATIONS_YML:-config/dataset_locations_asu_sol.yml}"

MOUNT_DATA="${MOUNT_DATA:-false}"
DEFAULT_BIND_DIRS="/scratch"
if [[ "$MOUNT_DATA" == "true" ]]; then
	DEFAULT_BIND_DIRS="/scratch /data/jliang12"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
else
	export MASTER_PORT="${MASTER_PORT:-29501}"
fi
export MASTER_ADDR=127.0.0.1

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
	RUN_NPROC_PER_NODE="$NPROC_PER_NODE"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
	RUN_NPROC_PER_NODE="$(echo "$SLURM_GPUS_ON_NODE" | grep -Eo '[0-9]+' | head -n 1)"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
	RUN_NPROC_PER_NODE="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
elif command -v nvidia-smi >/dev/null 2>&1; then
	RUN_NPROC_PER_NODE="$(nvidia-smi -L | wc -l | awk '{print $1}')"
else
	RUN_NPROC_PER_NODE=1
fi

if [[ -z "${RUN_NPROC_PER_NODE}" || "${RUN_NPROC_PER_NODE}" -lt 1 ]]; then
	RUN_NPROC_PER_NODE=1
fi

LAUNCHER=(python)
ENV_UNSET_ARGS=(-u WORLD_SIZE -u RANK -u LOCAL_RANK -u SLURM_PROCID -u SLURM_LOCALID -u SLURM_NPROCS)
if [[ "${RUN_NPROC_PER_NODE}" -gt 1 ]]; then
	echo "Distributed launch enabled: torchrun nproc_per_node=${RUN_NPROC_PER_NODE}"
	LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node "${RUN_NPROC_PER_NODE}")
	ENV_UNSET_ARGS=()
else
	echo "Distributed launch disabled: single-process (nproc_per_node=${RUN_NPROC_PER_NODE})"
fi

mkdir -p "$LOGFILE"
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
	RESUME_ARGS+=(--resume "$RESUME")
fi

echo "Task set ${NAME}: ${cyclictask}"
echo "total_epochs ${total_epochs} (epochs 1-$((total_epochs - 1)))"
echo "batch_size ${BATCHSIZE} per GPU, nproc_per_node ${RUN_NPROC_PER_NODE}"
echo "output ${LOGFILE}"

DEFAULT_BIND_DIRS="$DEFAULT_BIND_DIRS" ./cuda-apptainer.sh exec env \
	"${ENV_UNSET_ARGS[@]}" \
	FOUNDATION_X_DATASET_LOCATIONS_YML="$DATASET_LOCATIONS_YML" \
	"${LAUNCHER[@]}" main_Consolidated.py --taskcomponent foundation_x5_pretraining \
	--train --numClasses 1 --dataset_file "$DATASETFILE" --classification_dataset "$DATASETFILE" --num_workers "$num_workers" \
	--coco_path "$coco_path" --weight-decay 0.0001 \
	--output_dir "$LOGFILE" -c "$CONFIGFILE" --imgsize "$IMGSIZE" --backbonemodel "$BACKBONEMODEL" --init "$INIT" \
	--total_epochs "$total_epochs" --batch_size "$BATCHSIZE" --opt "$opt" \
	--finetune_ignore label_enc.weight class_embed \
	--backbone_dir "$backbone_dir" --lr_backbone "$lr_backbone" --lr_locEnc "$lr_locEnc" --lr_locDec "$lr_locDec" --lr_segmentor "$lr_segmentor" \
	--cyclictask "$cyclictask" --modelEMA "$EMAMODE" --lockrelease --saveAllModel --find_unused_params \
	"${RESUME_ARGS[@]}" \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

echo "Execution finished."
