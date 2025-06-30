import torch
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

from utils import test_contrastive_pred_probs_knn, read_embeds, compute_frequency_baseline_nll

from tqdm import tqdm
from pathlib import Path


def calc_nll(k, model_folders, group_name_str, embeds):
    storage = dict()

    for model_folder in model_folders:
        # SINCERE or SupCon
        model_name = model_folder.name.split('_')[0]

        # get embed from saved dict
        train_embeds, test_embeds, train_labels, test_labels = embeds[model_name]

        y_pred_probs = test_contrastive_pred_probs_knn(
            train_embeds, test_embeds, train_labels, test_labels, k
        )

        nll = F.nll_loss(torch.log(y_pred_probs), test_labels.long(), reduction="none")

        storage[model_name] = {"nll": nll.numpy()}

    return {"k-NN": k, "Experiment": group_name_str, "SINCERE": storage["SINCERE"]["nll"], "SupCon": storage["SupCon"]["nll"]}


def get_baseline_nll(model_folders):

    embeds = read_embeds(model_folders)

    # embeds is a dict with key "SINCERE" and SupCon
    _, _, train_labels, test_labels = embeds["SINCERE"]

    return compute_frequency_baseline_nll(train_labels, test_labels)


if __name__ == "__main__":
    model_folders_group = [

        ## exp1

        # batch-1024
        # resnet-18-imagenet-100-32x32
        [
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_40"),
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_30")
        ],
        # batch-1024
        # resnet-18-imagenet-100-64x64
        [
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-64x64/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_20-20_49_04"), 
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-64x64/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_20-20_22_04"), 
        ],
        # batch-1024
        # resnet-18-imagenet-100-112x112
        [
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-112x112/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_27-05_20_54"),
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-112x112/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_27-22_12_17")
        ],

        # ## exp2

        # # batch-100
        # # resnet-18-imagenet-100-32x32
        # [
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-100/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.045_decay_0.0001_bsz_100_temp_0.1_trial_0_cosine_2025_06_27-19_00_45"),
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-100/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.045_decay_0.0001_bsz_100_temp_0.1_trial_0_cosine_2025_06_27-19_00_45"),
        # ],
        # # batch-1024
        # # resnet-18-imagenet-100-32x32
        # [
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_40"),
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_30")
        # ],
        # # batch-5000
        # # resnet-18-imagenet-100-32x32
        # [
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-5000/imagenet100_models/SINCERE_imagenet100_resnet18_lr_2.75_decay_0.0001_bsz_5000_temp_0.1_trial_0_cosine_warm_2025_06_28-10_22_38"),
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-5000/imagenet100_models/SupCon_imagenet100_resnet18_lr_2.15_decay_0.0001_bsz_5000_temp_0.1_trial_0_cosine_warm_2025_06_28-10_18_52")
        # ],
    ]

    # SELECT_FOLDER_INDICES = [0, 1, 2]
    # GROUP_NAMES = ["Image-Size-32", "Image-Size-64", "Image-Size-112"]

    SELECT_FOLDER_INDICES = [0, 1, 2]
    GROUP_NAMES = ["Batch-100", "Batch-1024", "Batch-5000"]

    # doesn't matter which group use to compute, all have the same train and test labels
    baseline_nll = get_baseline_nll(model_folders_group[0])

    all_results = []

    k_values = range(1, 6, 1)

    for i, group_index in enumerate(tqdm(SELECT_FOLDER_INDICES, desc="Processing Experiment Groups")):
        current_model_folders = model_folders_group[group_index]
        group_name = GROUP_NAMES[i]

        embeds = read_embeds(current_model_folders)

        for k in tqdm(k_values, desc=f"Calculating NLL for {group_name}", leave=False):
            all_results.append(
                calc_nll(k, current_model_folders, group_name, embeds)
            )

    result_df = pd.DataFrame(all_results)

    plot_df_long = result_df.melt(
        id_vars=['k-NN', 'Experiment'],
        value_vars=['SINCERE', 'SupCon'],
        var_name='Model',
        value_name='NLL_array'
    )

    plot_df_long = plot_df_long.explode('NLL_array').astype({'NLL_array': float})

    palette = {"SINCERE": "#4c72b0", "SupCon": "#dd8452"}

    # use catplot to create faceted violin plots
    # this creates subplots for each Experiment group
    g = sns.catplot(
        data=plot_df_long,
        x='k-NN',
        y='NLL_array',
        hue='Model',
        col='Experiment', 
        kind='violin',
        split=True,        
        inner=None,     
        palette=palette,
    )

    g.fig.suptitle("NLL Distribution Comparison: SINCERE vs. SupCon")
    g.set_axis_labels("Number of Neighbors (k)", "Negative Log-Likelihood (NLL)")
    g.set_titles("{col_name}")

    g.despine(left=True)

    # add the baseline NLL horizontal line to each subplot
    for ax in g.axes.flat:
        ax.axhline(y=baseline_nll, color='gray', linestyle='--', label='Baseline NLL')

    g.legend.set_title("Model")
    plt.tight_layout()

    output_path = Path('eval/figures/knn-vs-nll/violin/comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=500)