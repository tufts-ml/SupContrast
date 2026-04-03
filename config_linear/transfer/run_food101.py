import itertools
import subprocess

from pathlib import Path

import numpy as np


launch_cmd = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True cd /cluster/tufts/hugheslab/mlao01/Git/SupContrast; nvidia-smi; pipenv run python main_linear.py"


checkpoint_dict = {
    "best_val_separation_t1": [
        Path(
            "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp-final-imagenet100/imagenet100_models/SINCERE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_cosine_warm_2025_11_27-22_17_28/last.pth",
        ),
        Path(
            "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp-final-imagenet100/imagenet100_models/SupCon_imagenet100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_29-05_55_46/last.pth",
        ),
        Path(
            "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp-final-imagenet100/imagenet100_models/EpsSupInfoNCE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_eps_0.1_cosine_warm_2025_12_12-16_04_00/last.pth"
        )
    ],
}


## settings ##

NUM_WORKER = 4
DATASET = "food101"
IMAGE_SIZE = 224

CHECKPOINT_NAME = "best_val_separation_t1"

PRJ_HEAD = False

EPOCH = 100
EPOCH_NAME = f"{EPOCH}_epochs_final"

if PRJ_HEAD:
    SUB_DIR = f"exp-final-imagenet100/transfer-final/{DATASET}/size-{IMAGE_SIZE}/{CHECKPOINT_NAME}/{EPOCH_NAME}/"
else:
    SUB_DIR = f"exp-final-imagenet100/transfer-final/{DATASET}/size-{IMAGE_SIZE}/{CHECKPOINT_NAME}_wo_projection_head/{EPOCH_NAME}/"

## end settings ##

search_dict = {
    "--batch_size": [128],
    "--model": "resnet50",
    "--use_cache_features": "",
    "--cosine": "",
    "--warm": "",
    "--epochs": EPOCH,
    "--weight_decay": [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    "--learning_rate":  np.logspace(0.01, np.log10(50), num=8).tolist(),
    "--valid_split": 0.1,
    "--size": IMAGE_SIZE,
    "--dataset": DATASET,
    "--save_sub_dir": SUB_DIR,
    "--ckpt": checkpoint_dict[CHECKPOINT_NAME],
    "--num_workers": NUM_WORKER,
    "--print_freq": 100,
}

if PRJ_HEAD:
    search_dict["--use_projection_head"] = ""

slurm_dict = {
    "-J": SUB_DIR,
    "-p": "preempt",
    "-t": "0-00:15:0",
    "--gres": "gpu:l40:1",
    "-c": NUM_WORKER,
    "-o": f"/cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/linear/{SUB_DIR}%A_%a.out",
}


def arg_dict_to_strs(arg_dict):
    val_lists = [val if type(val) is list else [val] for val in arg_dict.values()]
    return [
        " ".join([f"{key} {val}" for key, val in zip(arg_dict.keys(), vals)])
        for vals in itertools.product(*val_lists)
    ]


if __name__ == "__main__":
    search_strs = arg_dict_to_strs(search_dict)
    slurm_str = arg_dict_to_strs(slurm_dict)[0]
    print("Search Flags:", *search_strs, "Slurm Flags:", slurm_str, sep="\n")
    for search_str in search_strs:
        cmd = f'sbatch {slurm_str} --wrap "{launch_cmd} {search_str}"'
        subprocess.run(cmd, shell=True)
