#!/bin/bash

#SBATCH --job-name=foundationx_candidptx
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

# CANDID-PTX-only Foundation X+ run inside the CUDA Apptainer image.
# Based on scripts-apptainer/run_apptainer_v108.sh.
#
# Trains only the tasks named by the experiment. Every epoch also evaluates
# classification, localization, and segmentation so focused and unfocused
# scores land in export_csvFile.csv. --saveAllModel writes
# ckpt_E<epoch>_TH<head>.pth after each trained task.
#
#   sbatch ./scripts-apptainer/run_apptainer_candidptx.sh candidptx_cls
#   TOTAL_EPOCHS=51 sbatch ./scripts-apptainer/run_apptainer_candidptx.sh candidptx_loc
#   DEBUG_RUN=true ./scripts-apptainer/run_apptainer_candidptx.sh candidptx_cls
#
# Experiments:
#   candidptx_cls          candidptxCLS
#   candidptx_loc          candidptxLOC
#   candidptx_seg          candidptxSEG
#   candidptx_cls_loc      candidptxCLS + candidptxLOC
#   candidptx_cls_seg      candidptxCLS + candidptxSEG
#   candidptx_loc_seg      candidptxLOC + candidptxSEG
#   candidptx_cls_loc_seg  candidptxCLS + candidptxLOC + candidptxSEG

set -euo pipefail

# SLURM copies this file to /var/spool/.../slurm_script, so BASH_SOURCE is
# not the repo path. Prefer an explicit REPO_ROOT, then the sbatch cwd.

cd "/scratch/$USER/Foundation_X_Plus-CUDA-Container"
echo "Repo root: ${PWD}"

declare -A TRAIN_TAGS=(
	[candidptx_cls]="candidptxCLS"
	[candidptx_loc]="candidptxLOC"
	[candidptx_seg]="candidptxSEG"
	[candidptx_cls_loc]="candidptxCLS_candidptxLOC"
	[candidptx_cls_seg]="candidptxCLS_candidptxSEG"
	[candidptx_loc_seg]="candidptxLOC_candidptxSEG"
	[candidptx_cls_loc_seg]="candidptxCLS_candidptxLOC_candidptxSEG"
)

EVAL_SUFFIX="TESTcls_candidptxCLS_TESTloc_candidptxLOC_TESTseg_candidptxSEG"

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 <experiment>" >&2
	echo "Experiments:" >&2
	printf '  %s\n' "${!TRAIN_TAGS[@]}" | sort >&2
	exit 2
fi

experiment="$1"
train_tags="${TRAIN_TAGS[$experiment]:-}"
if [[ -z "$train_tags" ]]; then
	echo "Unknown experiment: $experiment" >&2
	echo "Expected one of:" >&2
	printf '  %s\n' "${!TRAIN_TAGS[@]}" | sort >&2
	exit 2
fi

cyclictask="${train_tags}_${EVAL_SUFFIX}"

CONFIGFILE=config/DINO/DINO_4scale_swinBASE.py
LOGFILE=${LOGFILE:-${SCRATCH:-/scratch/$USER}/FoundationX/candidptx/${experiment}}

backbone_dir=/scratch/sejong/class-dataset/models/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar

BACKBONEMODEL=Swin-B # Swin-T, Swin-B, Swin-L
IMGSIZE=224 # 448

coco_path=/scratch/sejong/class-dataset/VinDr-CXR/

DATASETFILE=foundation6Ark6_datasets

lr_backbone=1e-5
lr_locEnc=1e-4
lr_locDec=1e-4
lr_segmentor=1e-4

BATCHSIZE=${BATCHSIZE:-12}
num_workers=${num_workers:-12}
INIT=${INIT:-ark}
# Epoch loop is range(1, total_epochs), so 51 yields epochs 1-50.
total_epochs=${TOTAL_EPOCHS:-51}
opt=${opt:-adamw} # sgd adamw
EMAMODE=${EMAMODE:-True_Epoch}

DATASET_LOCATIONS_YML=${DATASET_LOCATIONS_YML:-config/dataset_locations_asu_sol.yml}

# Lightweight sanity run mode:
# DEBUG_RUN=true ./scripts-apptainer/run_apptainer_candidptx.sh candidptx_cls
# - enables --debug
# - limits to one training step (range(1, 2))
# - uses small batch and workers
# - disables EMA deepcopy to reduce GPU memory pressure
DEBUG_RUN=${DEBUG_RUN:-false}
EXTRA_ARGS=()
if [[ "$DEBUG_RUN" == "true" ]]; then
	total_epochs=2
	BATCHSIZE=2
	num_workers=0
	EMAMODE=None
	EXTRA_ARGS+=(--debug)
fi

# Mount policy:
# - default: bind /scratch (not just /scratch/sejong — binding a subdirectory when the
#   parent doesn't exist in the SIF overlay causes unreliable mounts in worker processes)
# - set MOUNT_DATA=true to also bind /data/jliang12
MOUNT_DATA=${MOUNT_DATA:-false}
DEFAULT_BIND_DIRS="/scratch"
if [[ "$MOUNT_DATA" == "true" ]]; then
	DEFAULT_BIND_DIRS="/scratch /data/jliang12"
fi

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

# Determine workers per node (priority: explicit override -> Slurm allocation -> visible devices)
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
# Cyclic multi-task batches leave some task heads unused per localization step,
# producing undefined (None) gradients that DDP would still try to allreduce.
# Always pass --find_unused_params for this model regardless of GPU count.
DDP_EXTRA_ARGS=(--find_unused_params)
ENV_UNSET_ARGS=(-u WORLD_SIZE -u RANK -u LOCAL_RANK -u SLURM_PROCID -u SLURM_LOCALID -u SLURM_NPROCS)
if [[ "${RUN_NPROC_PER_NODE}" -gt 1 ]]; then
	echo "Distributed launch enabled: torchrun nproc_per_node=${RUN_NPROC_PER_NODE}"
	LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node "${RUN_NPROC_PER_NODE}")
	ENV_UNSET_ARGS=()
else
	echo "Distributed launch disabled: single-process (nproc_per_node=${RUN_NPROC_PER_NODE})"
fi
echo "DDP setting: --find_unused_params enabled (always on for cyclic multi-task model)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
	echo "Running outside SLURM. This mode assumes GPU resources are already available (interactive session)."
else
	echo "Running under SLURM job ${SLURM_JOB_ID}"
fi

echo "Experiment : ${experiment}"
echo "Cyclic task: ${cyclictask}"
echo "Output     : ${LOGFILE}"
echo "Epochs     : ${total_epochs}"

# Execute the python script within the Apptainer container
# The --nv flag enables NVIDIA GPU support
echo "Launching training script inside Apptainer..."
DEFAULT_BIND_DIRS="$DEFAULT_BIND_DIRS" ./cuda-apptainer.sh exec env \
	"${ENV_UNSET_ARGS[@]}" \
	FOUNDATION_X_DATASET_LOCATIONS_YML="$DATASET_LOCATIONS_YML" \
	"${LAUNCHER[@]}" main_Consolidated.py --taskcomponent foundation_x5_pretraining \
	--train --numClasses 1 --dataset_file $DATASETFILE --classification_dataset $DATASETFILE --num_workers $num_workers \
	--coco_path $coco_path --weight-decay 0.0001 \
	--output_dir $LOGFILE -c $CONFIGFILE --imgsize $IMGSIZE --backbonemodel $BACKBONEMODEL --init $INIT \
	--total_epochs $total_epochs --batch_size $BATCHSIZE --opt $opt \
	--finetune_ignore label_enc.weight class_embed \
	--backbone_dir $backbone_dir --lr_backbone $lr_backbone --lr_locEnc $lr_locEnc --lr_locDec $lr_locDec  --lr_segmentor $lr_segmentor \
	--cyclictask $cyclictask --modelEMA $EMAMODE --lockrelease --saveAllModel \
	"${DDP_EXTRA_ARGS[@]}" \
	${EXTRA_ARGS[@]} \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

echo "Execution finished."
