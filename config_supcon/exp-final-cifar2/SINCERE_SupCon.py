import itertools
import subprocess

import numpy as np


launch_cmd = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True cd /cluster/s/AAAAAAlab/mDDD01/Git/SupContrast; nvidia-smi; pipenv run python main_supcon.py"

SUB_DIR = "exp-final-cifar2/"
NUM_WORKDERS = 8

search_dict = {
    "--batch_size": 512,
    "--model": "resnet50",
    "--size": 32,
    "--mixed_precision": "",
    # "--jit": "",
    "--learning_rate": list(np.logspace(np.log10(0.1), np.log10(2.5), num=6)),
    "--weight_decay": [1e-4],
    "--temp": [0.08, 0.10, 0.12],
    "--cosine": "",
    "--warm": "",
    "--epochs": [800],
    "--method": ["SINCERE", "SupCon"],
    "--valid_split": 0.1,
    "--dataset": "cifar2",
    "--print_freq": 50,
    "--save_freq": 1000,
    "--save_sub_dir": SUB_DIR,
    "--num_workers": NUM_WORKDERS,
    # "--no_projection_head": "",
    "--feat_dim": 128,
}

slurm_dict = {
    "-J": f"exp-final-cifar2",
    "-p": "AAAAAAlab",
    "-t": "12:00:00",
    "--gres": "gpu:rtx_6000:1",
    "-c": NUM_WORKDERS,
    "-o": f"/cluster/s/AAAAAAlab/mDDD01/Git/SupContrast/slurm_out/{SUB_DIR}%A_%a.out",
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
