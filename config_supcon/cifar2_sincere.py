import itertools
import subprocess


launch_cmd = "cd ~/Git/SupContrast; nvidia-smi; pipenv run python main_supcon.py"

search_dict = {
    "--batch_size": 512,
    "--learning_rate": 0.65,
    "--temp": 0.1,
    "--cosine": "",
    "--epochs": 800,
    "--method": "SINCERE",
    "--valid_split": 0.1,
    "--dataset": "cifar2",
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
