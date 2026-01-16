#!/usr/bin/env bash

#SBATCH -p hugheslab
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH -n 4
#SBATCH --mem 8G
#SBATCH -t 0-23:00
#SBATCH --job-name="attack embedding cifar100"
#SBATCH -o /cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/notimportant/%j.out
#SBATCH -e /cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/notimportant/%j.err


# SINCERE Paths
ENCODER_SINCERE="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp-final-cifar100/cifar100_models/SINCERE_cifar100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_23-01_54_13/last.pth"
HEAD_SINCERE="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-cifar100/january/cifar100/size-32/best_val_separation_t1/100_epochs_temp_no_decay_no_cache_match_batch_size/cifar100_lr_0.03949885128725748_decay_0.0_bsz_512_SINCERE_cifar100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_23-01_54_13_cosine_warm/last.pth"

# SupCon Paths
ENCODER_SUPCON="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp-final-cifar100/cifar100_models/SupCon_cifar100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_22-23_19_21/last.pth"
HEAD_SUPCON="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-cifar100/january/cifar100/size-32/best_val_separation_t1/100_epochs_temp_no_decay_no_cache_match_batch_size/cifar100_lr_0.03949885128725748_decay_0.0_bsz_512_SupCon_cifar100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_22-23_19_21_cosine_warm/last.pth"

# Configuration

DATASET="cifar100"
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

DATASET="cifar100"
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