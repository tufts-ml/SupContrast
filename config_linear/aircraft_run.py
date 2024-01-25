import itertools
import subprocess


launch_cmd = "cd ~/Git/SupContrast; nvidia-smi; pipenv run python main_linear.py"

search_dict = {
    "--batch_size": 512,
    "--learning_rate": 5,
    "--dataset": "aircraft",
    "--ckpt": [
        "save/SupCon/aircraft_models/SINCERE_aircraft_resnet50_lr_0.85_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_24-20_32_11/last.pth",  # noqa: E501
        "save/SupCon/aircraft_models/SupCon_aircraft_resnet50_lr_0.85_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_24-20_32_11/last.pth",  # noqa: E501
        "save/SupCon/imagenet100_models/SINCERE_imagenet100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_18/last.pth",  # noqa: E501
        "save/SupCon/imagenet100_models/SupCon_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_20/last.pth",  # noqa: E501
    ],
}

slurm_dict = {
    "-p": "hugheslab",
    "-t": "0-20:0:0",
    "--gres": "gpu:rtx_a6000:1",
    "-c": 16,
    "-o": "~/Git/SupContrast/slurm_out/%A_%a.out",
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
        cmd = f"sbatch {slurm_str} --wrap \"{launch_cmd} {search_str}\""
        subprocess.run(cmd, shell=True)