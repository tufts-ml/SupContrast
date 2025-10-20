#!/usr/bin/env bash

#SBATCH -p hugheslab
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH -n 8
#SBATCH --mem 16G
#SBATCH -t 0-23:00
#SBATCH --job-name="robustness cifar10"
#SBATCH -o /cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/notimportant/%j.out
#SBATCH -e /cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/notimportant/%j.err


ENCODER_CKPT_1="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/cynthia/cifar10_models/SINCERE_cifar10_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_20-22_04_43/last.pth"
ENCODER_CKPT_2="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/cynthia/cifar10_models/SupCon_cifar10_resnet50_lr_0.35_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_19-15_04_54/last.pth"
ENCODER_CKPT_3="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/cynthia/cifar10_models/EpsSupInfoNCE_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_21-12_28_30/last.pth"

HEAD_CKPT_1="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/cynthia/cifar10/size-32/cynthia-cifar10/25_epochs/cifar10_lr_26.826957952797244_decay_0.0001_bsz_128_SINCERE_cifar10_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_20-22_04_43_cosine/last.pth"
HEAD_CKPT_2="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/cynthia/cifar10/size-32/cynthia-cifar10/25_epochs/cifar10_lr_26.826957952797244_decay_0.0001_bsz_128_SupCon_cifar10_resnet50_lr_0.35_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_19-15_04_54_cosine/last.pth"
HEAD_CKPT_3="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/cynthia/cifar10/size-32/cynthia-cifar10/25_epochs/cifar10_lr_26.826957952797244_decay_0.0001_bsz_128_EpsSupInfoNCE_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_21-12_28_30_cosine/last.pth"

LABELS="SINCERE SupCon EpsSupInfoNCE"

SUBDIR="cynthia-cifar10/"

DATASET="cifar10"
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
    --epsilons $(seq 0 0.02 0.3)

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
