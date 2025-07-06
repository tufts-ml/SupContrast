import itertools
import subprocess

launch_cmd = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True cd /cluster/tufts/hugheslab/mlao01/Git/SupContrast; nvidia-smi; pipenv run python main_linear_cache_features.py"

search_dict = {
    "--batch_size": 512,
    # make sure the model name is consistent with the ckpt_path and linear probing later on
    "--model": "resnet18",
    # make sure the image size is consistent with the ckpt_path and linear probing later on
    "--size": 32,
    "--dataset": ["imagenet100"],
    "--num_workers": 16,
    "--ckpt_path": [
        # ## exp1
        # # batch-1024
        # # resnet-18-imagenet-100-112x112
        # "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-112x112/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_27-05_20_54",
        # "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-112x112/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_27-22_12_17",
        # # batch-1024
        # # resnet-18-imagenet-100-64x64
        # "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-64x64/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_20-20_49_04",
        # "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-64x64/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_20-20_22_04",
        # # batch-1024
        # # resnet-18-imagenet-100-32x32
        # "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_40",
        # "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_30",
        ## exp2
        # batch-100
        # resnet-18-imagenet-100-32x32
        "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-100/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.045_decay_0.0001_bsz_100_temp_0.1_trial_0_cosine_2025_06_27-19_00_45/last.pth",
        "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-100/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.045_decay_0.0001_bsz_100_temp_0.1_trial_0_cosine_2025_06_27-19_00_45/last.pth",
        # # batch-1024
        # # resnet-18-imagenet-100-32x32
        "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_40/last.pth",
        "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_30/last.pth",
        # batch-5000
        # resnet-18-imagenet-100-32x32
        "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-5000/imagenet100_models/SINCERE_imagenet100_resnet18_lr_2.75_decay_0.0001_bsz_5000_temp_0.1_trial_0_cosine_warm_2025_06_28-10_22_38/last.pth",
        "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-5000/imagenet100_models/SupCon_imagenet100_resnet18_lr_2.15_decay_0.0001_bsz_5000_temp_0.1_trial_0_cosine_warm_2025_06_28-10_18_52/last.pth",
    ],
}

slurm_dict = {
    "-p": "hugheslab",
    "-t": "0-6:00:00",
    "--gres": "gpu:rtx_a6000:1",
    "-c": 16,
    "-o": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/notimportant/%A_%a.out",
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
