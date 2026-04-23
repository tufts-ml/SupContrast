#!/usr/bin/env bash

#SBATCH -p AAAAAAlab
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH -n 4
#SBATCH --mem 8G
#SBATCH -t 0-23:00
#SBATCH --job-name="attack embedding imagenet100"
#SBATCH -o /cluster/s/AAAAAAlab/mDDD01/Git/SupContrast/slurm_out/notimportant/%j.out
#SBATCH -e /cluster/s/AAAAAAlab/mDDD01/Git/SupContrast/slurm_out/notimportant/%j.err


# SINCERE Paths
ENCODER_SINCERE="/cluster/s/AAAAAAlab/mDDD01/Git/SupContrast/save/SupCon/exp-final-imagenet100/imagenet100_models/SINCERE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_cosine_warm_2025_11_27-22_17_28/last.pth"
HEAD_SINCERE="/cluster/s/AAAAAAlab/mDDD01/Git/SupContrast/save/linear/exp-final-imagenet100/january/imagenet100/size-32/best_val_separation_t1/100_epochs_temp_no_decay_no_cache_match_batch_size/imagenet100_lr_0.01_decay_0.0_bsz_512_SINCERE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_cosine_warm_2025_11_27-22_17_28_cosine_warm/last.pth"

# SupCon Paths
ENCODER_SUPCON="/cluster/s/AAAAAAlab/mDDD01/Git/SupContrast/save/SupCon/exp-final-imagenet100/imagenet100_models/SupCon_imagenet100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_29-05_55_46/last.pth"
HEAD_SUPCON="/cluster/s/AAAAAAlab/mDDD01/Git/SupContrast/save/linear/exp-final-imagenet100/january/imagenet100/size-32/best_val_separation_t1/100_epochs_temp_no_decay_no_cache_match_batch_size/imagenet100_lr_0.01_decay_0.0_bsz_512_SupCon_imagenet100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_29-05_55_46_cosine_warm/last.pth"

# Configuration

DATASET="imagenet100"
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

Configuration

DATASET="imagenet100"
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