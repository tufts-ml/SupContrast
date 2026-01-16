#!/usr/bin/env bash

#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -n 4
#SBATCH --mem 8G
#SBATCH -t 0-23:00
#SBATCH --job-name="evaluate final resnet50 cifar2"
#SBATCH -o /cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/notimportant/%j.out
#SBATCH -e /cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/notimportant/%j.err

PARENT_DIR="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp-final-cifar2/cifar2_models/"

DATASET="cifar2"
MODEL_ARCH="resnet50"
LAST_EPOCH=800
BATCH_SIZE=512
IMG_SIZE=32

echo "Running evaluation on BOTH Validation and Test sets..."

pipenv run python evaluate_models.py \
    --parent_dir "$PARENT_DIR" \
    --dataset "$DATASET" \
    --model_arch "$MODEL_ARCH" \
    --last_epoch "$LAST_EPOCH" \
    --batch_size "$BATCH_SIZE" \
    --size "$IMG_SIZE" \
    --separation_percentiles 0.99 \
    --separation_topk 1 \
    --num_workers 4

echo "Evaluation complete."