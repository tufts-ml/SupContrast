#!/usr/bin/env bash

#SBATCH -p hugheslab
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH -n 8
#SBATCH --mem 16G
#SBATCH -t 0-23:00
#SBATCH --job-name="robustness imagenet100"
#SBATCH -o /cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/notimportant/%j.out
#SBATCH -e /cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/notimportant/%j.err


ENCODER_CKPT_1="/cluster/tufts/hugheslab/code/SINCERE_models/SupCon/imagenet100_models/SINCERE_imagenet100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_18/last.pth"
ENCODER_CKPT_2="/cluster/tufts/hugheslab/code/SINCERE_models/SupCon/imagenet100_models/SupCon_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_20/last.pth"
ENCODER_CKPT_3="/cluster/tufts/hugheslab/code/SINCERE_models/SupCon/imagenet100_models/EpsSupInfoNCE_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_22-09_31_46/last.pth"

HEAD_CKPT_1="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/cynthia/imagenet100/size-32/cynthia-imagenet100/25_epochs/imagenet100_lr_7.196856730011514_decay_0.0001_bsz_128_SINCERE_imagenet100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_18_cosine/last.pth"
HEAD_CKPT_2="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/cynthia/imagenet100/size-32/cynthia-imagenet100/25_epochs/imagenet100_lr_0.13894954943731375_decay_0.0001_bsz_128_SupCon_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_20_cosine/last.pth"
HEAD_CKPT_3="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/cynthia/imagenet100/size-32/cynthia-imagenet100/25_epochs/imagenet100_lr_0.517947467923121_decay_0.0001_bsz_128_EpsSupInfoNCE_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_22-09_31_46_cosine/last.pth"

LABELS="SINCERE SupCon EpsSupInfoNCE"

SUBDIR="cynthia-imagenet100/"

DATASET="imagenet100"
MODEL_ARCH="resnet50"
IMG_SIZE=32

echo "Output subdirectory: $SUBDIR"

echo "Starting robustness analysis..."

pipenv run python analyze_robustness.py \
    --encoder_ckpts "$ENCODER_CKPT_1" "$ENCODER_CKPT_2" "$ENCODER_CKPT_3" \
    --head_ckpts "$HEAD_CKPT_1" "$HEAD_CKPT_2" "$HEAD_CKPT_3" \
    --labels $LABELS \
    --dataset "$DATASET" \
    --model "$MODEL_ARCH" \
    --size "$IMG_SIZE" \
    --subdir "$SUBDIR" \
    --limit 1000000

echo "Starting adversarial attack on IMAGE space..."

pipenv run python attack_image_space.py \
    --encoder_ckpts "$ENCODER_CKPT_1" "$ENCODER_CKPT_2" "$ENCODER_CKPT_3" \
    --head_ckpts "$HEAD_CKPT_1" "$HEAD_CKPT_2" "$HEAD_CKPT_3" \
    --labels $LABELS \
    --dataset "$DATASET" \
    --model "$MODEL_ARCH" \
    --size "$IMG_SIZE" \
    --subdir "$SUBDIR" \
    --epsilons $(seq 0 0.0025 0.025)

echo "Starting adversarial attack on EMBEDDING space..."

pipenv run python attack_embedding_space.py \
    --encoder_ckpts "$ENCODER_CKPT_1" "$ENCODER_CKPT_2" "$ENCODER_CKPT_3" \
    --head_ckpts "$HEAD_CKPT_1" "$HEAD_CKPT_2" "$HEAD_CKPT_3" \
    --labels $LABELS \
    --dataset "$DATASET" \
    --model "$MODEL_ARCH" \
    --size "$IMG_SIZE" \
    --subdir "$SUBDIR" \
    --epsilons $(seq 0 0.05 1.0)

echo "All scripts completed."
