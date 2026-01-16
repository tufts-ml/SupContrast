#!/usr/bin/env bash

#SBATCH -p hugheslab
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH -n 4
#SBATCH --mem 8G
#SBATCH -t 0-23:00
#SBATCH --job-name="attack embedding cifar2"
#SBATCH -o /cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/notimportant/%j.out
#SBATCH -e /cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/notimportant/%j.err


# SINCERE Paths
ENCODER_SINCERE="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp-final-cifar2/cifar2_models/SINCERE_cifar2_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_12_21-03_36_45/last.pth"
HEAD_SINCERE="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-cifar2/january/cifar2/size-32/best_val_separation_t1/100_epochs_temp_no_decay_no_cache_match_batch_size/cifar2_lr_0.01_decay_0.0_bsz_512_SINCERE_cifar2_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_12_21-03_36_45_cosine_warm/last.pth"

# SupCon Paths
ENCODER_SUPCON="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp-final-cifar2/cifar2_models/SupCon_cifar2_resnet50_lr_0.19036539387158782_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_12_21-01_14_28/last.pth"
HEAD_SUPCON="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-cifar2/january/cifar2/size-32/best_val_separation_t1/100_epochs_temp_no_decay_no_cache_match_batch_size/cifar2_lr_0.01_decay_0.0_bsz_512_SupCon_cifar2_resnet50_lr_0.19036539387158782_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_12_21-01_14_28_cosine_warm/last.pth"

# Configuration

DATASET="cifar2"
MODEL_ARCH="resnet50"
OUTPUT_DIR="results_attack/january-no-cache-lowest-val-loss/"

EPSILONS=$(seq -s " " 0 0.05 1.5)

echo "Running embedding space attack..."

pipenv run python attack_embedding_space.py \
    --dataset "$DATASET" \
    --model_arch "$MODEL_ARCH" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 4 \
    --epsilons $EPSILONS \
    --loss_names SINCERE SupCon \
    --encoder_paths "$ENCODER_SINCERE" "$ENCODER_SUPCON" \
    --head_paths "$HEAD_SINCERE" "$HEAD_SUPCON"

# Configuration

DATASET="cifar2"
MODEL_ARCH="resnet50"
OUTPUT_DIR="results_robustness/january-no-cache-lowest-val-loss/"
LIMIT=10000000

echo "Running robustness analysis (Distance vs Epsilon)..."

pipenv run python attack_robustness.py \
    --dataset "$DATASET" \
    --model_arch "$MODEL_ARCH" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 4 \
    --limit "$LIMIT" \
    --loss_names SINCERE SupCon \
    --encoder_paths "$ENCODER_SINCERE" "$ENCODER_SUPCON" \
    --head_paths "$HEAD_SINCERE" "$HEAD_SUPCON"

echo "Attack evaluation complete."