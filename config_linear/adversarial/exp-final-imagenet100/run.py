import itertools
import subprocess

from pathlib import Path

import numpy as np


launch_cmd = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True cd /cluster/tufts/AAAAAAlab/mDDD01/Git/SupContrast; nvidia-smi; pipenv run python main_linear.py"


checkpoint_dict = {
    "best_val_separation_t1": [
        Path(
            "/cluster/tufts/AAAAAAlab/mDDD01/Git/SupContrast/save/SupCon/exp-final-imagenet100/imagenet100_models/SINCERE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_cosine_warm_2025_11_27-22_17_28/last.pth",
        ),
        Path(
            "/cluster/tufts/AAAAAAlab/mDDD01/Git/SupContrast/save/SupCon/exp-final-imagenet100/imagenet100_models/SupCon_imagenet100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_29-05_55_46/last.pth",
        ),
    ],
}


## settings ##

NUM_WORKER = 4
DATASET = "imagenet100"
IMAGE_SIZE = 32

CHECKPOINT_NAME = "best_val_separation_t1"

PRJ_HEAD = True

EPOCH = 100
EPOCH_NAME = f"{EPOCH}_epochs_temp_no_decay_no_cache_match_batch_size"

if PRJ_HEAD:
    SUB_DIR = f"exp-final-imagenet100/january/{DATASET}/size-{IMAGE_SIZE}/{CHECKPOINT_NAME}/{EPOCH_NAME}/"
else:
    SUB_DIR = f"exp-final-imagenet100/january/{DATASET}/size-{IMAGE_SIZE}/{CHECKPOINT_NAME}_wo_projection_head/{EPOCH_NAME}/"

## end settings ##


search_dict = {
    "--batch_size": 512,
    "--model": "resnet50",
    # "--use_cache_features": "",
    "--cosine": "",
    "--warm": "",
    "--epochs": EPOCH,
    "--weight_decay": [0],
    "--learning_rate":  np.logspace(np.log10(0.01), np.log10(150), 8).tolist(),
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
    "-p": "AAAAAAlab",
    "-t": "2-00:00:0",
    "--gres": "gpu:rtx_a6000:1",
    "-c": NUM_WORKER,
    "-o": f"/cluster/tufts/AAAAAAlab/mDDD01/Git/SupContrast/slurm_out/linear/{SUB_DIR}%A_%a.out",
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
