import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  
from utils import test_contrastive_pred_probs_knn, read_embeds
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    model_folders_group = [
        # resnet-18-imagenet-100-64x64
        [
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/resnet-18-imagenet-100-64x64/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_20-20_49_04"),
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/resnet-18-imagenet-100-64x64/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_20-20_22_04"),
        ],
        # resnet-18-imagenet-100-32x32
        [
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/resnet-18-imagenet-100-32x32/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_40"),
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/resnet-18-imagenet-100-32x32/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_30")
        ],
    ]

    SELECT_FOLDER_GROUP = 0

    ## nll

    result = []

    embeds = read_embeds(model_folders_group[SELECT_FOLDER_GROUP])

    count_train = embeds["SINCERE"][0].shape[0]

    def calc_nll(k):
        storage = dict()

        for model_folder in model_folders_group[SELECT_FOLDER_GROUP]:
            # SINCERE or SupCon
            model_name = model_folder.name.split('_')[0]

            # get embed from saved dict
            train_embeds, test_embeds, train_labels, test_labels = embeds[model_name]

            y_pred_probs = test_contrastive_pred_probs_knn(
                train_embeds, test_embeds, train_labels, test_labels, k
            )

            nll = F.nll_loss(torch.log(y_pred_probs), test_labels.long(), reduction="none")

            storage[model_name] = {
                "nll": nll.numpy() 
            }

        return {"k-NN": k, "SINCERE": storage["SINCERE"]["nll"], "SupCon": storage["SupCon"]["nll"]}

    k_values = range(1, 11)

    result = []
    for k in tqdm(k_values, desc="Calculating NLL for k-NN"):
        result.append(calc_nll(k))

    result_df = pd.DataFrame(result)
    result_df = result_df.set_index('k-NN')

    plot_df_long = result_df.reset_index().melt(
        id_vars='k-NN', 
        value_vars=['SINCERE', 'SupCon'], 
        var_name='Model', 
        value_name='NLL_array'
    )

    plot_df_long = plot_df_long.explode('NLL_array').astype({'NLL_array': float})

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.violinplot(
        data=plot_df_long,
        x='k-NN',
        y='NLL_array',
        hue='Model',
        split=True,       
        inner=None,    
        ax=ax
    )

    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("Negative Log-Likelihood (NLL)")
    ax.set_title("NLL Distribution for SINCERE and SupCon vs. k-NN", fontsize=16)

    sns.despine(left=True)
    plt.tight_layout()

    output_path = Path('eval/figures/knn-vs-nll/violin/knn-vs-nll.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)