#!/bin/bash
# Pretrain one CANDID-PTX task set inside the Apptainer container.
#
# Usage:
#   ./scripts/run_candidptx_taskset.sh a
#   RESUME=/scratch/hflechsi/FoundationX/candidptx_tasksets/a/ckpt_E4_TH9.pth ./scripts/run_candidptx_taskset.sh a
#
# Task sets and the epochs the loop actually runs (range starts at 1 and stops
# before total_epochs, so total_epochs is one past the last epoch):
#   a  candidptxCLS                              epochs 1-10   total_epochs 11
#   b  candidptxLOC                              epochs 1-10   total_epochs 11
#   c  candidptxSEG                              epochs 1-10   total_epochs 11
#   d  candidptxCLS_candidptxLOC                 epochs 1-20   total_epochs 21
#   e  candidptxCLS_candidptxSEG                 epochs 1-20   total_epochs 21
#   f  candidptxLOC_candidptxSEG                 epochs 1-20   total_epochs 21
#   g  candidptxCLS_candidptxLOC_candidptxSEG    epochs 1-30   total_epochs 31
#
# Build the split files once before the first run:
#   python scripts/build_candidptx_splits.py /scratch/hflechsi/CANDID-PTX

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
	*)
		echo "Usage: $0 {a|b|c|d|e|f|g}" >&2
		exit 1
		;;
esac

IFS=_ read -ra TOKENS <<< "$cyclictask"
ntasks="${#TOKENS[@]}"
total_epochs="${TOTAL_EPOCHS:-$((10 * ntasks + 1))}"

CONFIGFILE=config/DINO/DINO_4scale_swinBASE.py
LOGFILE=/scratch/hflechsi/FoundationX/candidptx_tasksets/${NAME}
backbone_dir=/scratch/sejong/class-dataset/models/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar
BACKBONEMODEL=Swin-B
IMGSIZE=224
coco_path=/scratch/sejong/class-dataset/VinDr-CXR/
DATASETFILE=foundation6Ark6_datasets
lr_backbone=1e-5
lr_locEnc=1e-4
lr_locDec=1e-4
lr_segmentor=1e-4
BATCHSIZE="${BATCHSIZE:-24}"
num_workers="${num_workers:-12}"
INIT=ark
opt=adamw
EMAMODE=True_Epoch
DATASET_LOCATIONS_YML="${DATASET_LOCATIONS_YML:-config/dataset_locations_asu_sol.yml}"

mkdir -p "$LOGFILE"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29501}"

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
	RESUME_ARGS+=(--resume "$RESUME")
fi

echo "Task set ${NAME}: ${cyclictask}"
echo "total_epochs ${total_epochs} (epochs 1-$((total_epochs - 1)))"
echo "output ${LOGFILE}"

DEFAULT_BIND_DIRS="/scratch" ./cuda-apptainer.sh exec env FOUNDATION_X_DATASET_LOCATIONS_YML="$DATASET_LOCATIONS_YML" python main_Consolidated.py \
	--taskcomponent foundation_x5_pretraining --train --numClasses 1 \
	--dataset_file "$DATASETFILE" --classification_dataset "$DATASETFILE" \
	--num_workers "$num_workers" --coco_path "$coco_path" --weight-decay 0.0001 \
	--output_dir "$LOGFILE" -c "$CONFIGFILE" --imgsize "$IMGSIZE" --backbonemodel "$BACKBONEMODEL" \
	--init "$INIT" --total_epochs "$total_epochs" --batch_size "$BATCHSIZE" --opt "$opt" \
	--finetune_ignore label_enc.weight class_embed \
	--backbone_dir "$backbone_dir" --lr_backbone "$lr_backbone" --lr_locEnc "$lr_locEnc" \
	--lr_locDec "$lr_locDec" --lr_segmentor "$lr_segmentor" \
	--cyclictask "$cyclictask" --modelEMA "$EMAMODE" --lockrelease --saveAllModel \
	"${RESUME_ARGS[@]}" \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
