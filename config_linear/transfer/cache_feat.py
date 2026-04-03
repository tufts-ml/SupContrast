import itertools
import subprocess

launch_cmd = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True cd /cluster/tufts/AAAAAAlab/mDDD01/Git/SupContrast; nvidia-smi; pipenv run python main_linear_cache_features.py"


search_dict = {
    "--batch_size": 4,
    "--model": "resnet50",
    "--size": 224,
    "--dataset": ["pet", "dtd", "aircraft", "food101", "flowers", "cars"],
    "--num_workers": 8,
    "--ckpt": [
        "/cluster/tufts/AAAAAAlab/mDDD01/Git/SupContrast/save/SupCon/exp-final-imagenet100/imagenet100_models/SINCERE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_cosine_warm_2025_11_27-22_17_28/last.pth",
        "/cluster/tufts/AAAAAAlab/mDDD01/Git/SupContrast/save/SupCon/exp-final-imagenet100/imagenet100_models/SupCon_imagenet100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_29-05_55_46/last.pth",
        "/cluster/tufts/AAAAAAlab/mDDD01/Git/SupContrast/save/SupCon/exp-final-imagenet100/imagenet100_models/EpsSupInfoNCE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_eps_0.1_cosine_warm_2025_12_12-16_04_00/last.pth"
    ],
}


## check on --use_projection_head

PRJ_HEAD = False
if PRJ_HEAD:
    search_dict["--use_projection_head"] = ""


slurm_dict = {
    "-p": "preempt",
    "-t": "0-2:00:00",
    "--gres": "gpu:l40:1",
    "-c": 8,
    "-o": "/cluster/tufts/AAAAAAlab/mDDD01/Git/SupContrast/slurm_out/transfer-feb/%A_%a.out",
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
