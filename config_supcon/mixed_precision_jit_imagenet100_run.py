import itertools
import subprocess


launch_cmd = "export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache_${SLURM_JOB_ID}_$$; cd /cluster/tufts/hugheslab/mlao01/Git/SupContrast; nvidia-smi; pipenv run python main_supcon_fp16_jit.py"

search_dict = {
    "--batch_size": 1024,
    "--model": "resnet18",
    "--size": 32,
    "--mixed_precision": "",
    "--learning_rate": [0.10, 0.35, 0.5, 0.65, 0.75],
    "--temp": [0.1],
    "--cosine": "",
    "--epochs": 350,
    "--method": ["SINCERE", "SupCon"],
    "--valid_split": 0.1,
    "--dataset": "imagenet100",
    "--print_freq": 20,
}

slurm_dict = {
    "-p": "preempt",
    "-t": "1-00:00:00",
    "--gres": "gpu:rtx_a5000:1",
    "-c": 16,
    "-o": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/resnet-18-imagenet-100-32x32/%A_%a.out",
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
