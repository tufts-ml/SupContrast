import itertools
import subprocess

# interactivate session

# srun -p gpu --gres=gpu:a100:1 -t 0-08:00 -c 16 --pty bash

# pipenv run python main_supcon.py \
#   --model resnet18 \
#   --batch_size 64 \
#   --learning_rate 0.65 \
#   --temp 0.1 \
#   --cosine \
#   --epochs 1 \
#   --print_freq 100 \
#   --method SINCERE \
#   --valid_split 0 \
#   --dataset imagenet \
#   --size 224 



launch_cmd = "cd /cluster/tufts/hugheslab/mlao01/Git/SupContrast; nvidia-smi; pipenv run python main_supcon.py"

search_dict = {
    "--batch_size": 512,
    "--learning_rate": 0.65,
    "--temp": 0.05,
    "--cosine": "",
    "--epochs": 800,
    "--method": "SINCERE",
    "--valid_split": 0,
    "--dataset": "imagenet",
}

slurm_dict = {
    "-p": "ccgpu",
    "-t": "2-0:0:0",
    "--gres": "gpu:a100:1",
    "-c": 16,
    "-o": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/slurm_out/%A_%a.out",
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
