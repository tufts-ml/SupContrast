from pathlib import Path

from utils import read_embeds, test_contrastive_pred_probs_knn

import torch

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


class PlotKNNProba:
    def __init__(
        self,
        group_name_str: str,
        model_folders: list,
        k: int,
        output_dir="eval-new/figures",
        kde=False,
    ):
        self.group_name_str = group_name_str
        self.k = k

        self.kde = kde

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_folders = model_folders

        self.embeds = read_embeds(self.model_folders)

        self.compute_predict_proba()
        self.compute_max_proba_and_correct_mask()

        self.plot_correctness_dist()
        self.plot_nll_dist()

    def compute_predict_proba(self):
        self.test_predit_probas = {}
        self.test_true_labels = {}

        for model_folder in self.model_folders:
            # SINCERE or SupCon
            model_name = model_folder.name.split("_")[0]

            train_embeds, test_embeds, train_labels, test_labels = self.embeds[
                model_name
            ]
            
            y_pred_proba = test_contrastive_pred_probs_knn(
                train_embeds,
                test_embeds,
                train_labels,
                test_labels,
                self.k,
                raw=False,
            )

            self.test_predit_probas[model_name] = y_pred_proba
            self.test_true_labels[model_name] = test_labels

    def compute_max_proba_and_correct_mask(self):
        self.max_probas = {}
        self.correct_masks = {}
        self.labels_with_acc = {}

        for model_name, y_pred_proba in self.test_predit_probas.items():
            true_labels = self.test_true_labels[model_name]

            max_proba_values, predicted_labels = torch.max(y_pred_proba, dim=1)

            correct_mask = predicted_labels == true_labels

            self.max_probas[model_name] = max_proba_values.numpy()
            self.correct_masks[model_name] = correct_mask.numpy()

            accuracy = correct_mask.sum().item() / len(correct_mask)
            self.labels_with_acc[model_name] = f"{model_name} (Acc: {accuracy:.3f})"

    def plot_correctness_dist(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        fig.suptitle(
            f"Distribution of Max Prediction Probabilities (k={self.k}, {self.group_name_str})",
            fontsize=16,
        )

        uniform_bins = np.linspace(0, 1, 46)

        # Plot for Correct Predictions
        ax_correct = axes[0]
        for model_name in self.test_predit_probas.keys():
            max_proba = self.max_probas[model_name]
            correct_mask = self.correct_masks[model_name]
            label = self.labels_with_acc[model_name]

            sns.histplot(
                max_proba[correct_mask],
                label=label,
                bins=uniform_bins,
                alpha=0.35,
                ax=ax_correct,
                stat="density" if self.kde else "count",
            )

            if self.kde:
                sns.kdeplot(max_proba[correct_mask], ax=ax_correct, lw=2, alpha=0.35)

        ax_correct.set_title("Correct Predictions")
        ax_correct.set_xlabel("Maximum Probability")
        ax_correct.set_ylabel("Probability Density" if self.kde else "Count")
        ax_correct.legend()
        ax_correct.grid(alpha=0.3)

        # Plot for Incorrect Predictions
        ax_incorrect = axes[1]
        for model_name in self.test_predit_probas.keys():
            max_proba = self.max_probas[model_name]
            correct_mask = self.correct_masks[model_name]
            label = self.labels_with_acc[model_name]

            sns.histplot(
                max_proba[~correct_mask],
                label=label,
                bins=uniform_bins,
                alpha=0.35,
                ax=ax_incorrect,
                stat="density" if self.kde else "count",
            )

            if self.kde:
                sns.kdeplot(max_proba[~correct_mask], ax=ax_incorrect, lw=2, alpha=0.35)

        ax_incorrect.set_title("Incorrect Predictions")
        ax_incorrect.set_xlabel("Maximum Probability" if self.kde else "Count")
        ax_incorrect.grid(alpha=0.3)

        plt.tight_layout()

        output_filename = self.output_dir / f"k_{self.k}_max_prob_dist.png"
        plt.savefig(output_filename, dpi=300)

    def plot_nll_dist(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        fig.suptitle(
            f"Distribution of Negative Log-Likelihood (k={self.k}, {self.group_name_str})",
            fontsize=16,
        )

        uniform_bins = np.linspace(0, 23, 126)

        for model_name, y_pred_proba in self.test_predit_probas.items():
            true_labels = self.test_true_labels[model_name]
            label = self.labels_with_acc[model_name]

            # Convert labels to the required dtype for gather()
            true_labels = true_labels.long()

            # Get the predicted probability of the true class for each sample
            # We use torch.gather to select the probability from the column specified by true_labels
            true_class_probas = torch.gather(
                y_pred_proba, 1, true_labels.unsqueeze(1)
            ).squeeze()

            # Calculate Negative Log-Likelihood: -log(p_true)
            # Add a small epsilon to avoid log(0)
            nll_values = -torch.log(true_class_probas + 1e-9)

            # Plot the distributions
            sns.histplot(
                nll_values.numpy(),
                label=label,
                bins=uniform_bins,
                alpha=0.35,
                ax=ax,
                stat="density" if self.kde else "count",
            )

            if self.kde:
                sns.kdeplot(nll_values.numpy(), ax=ax, lw=2, alpha=0.35)

        ax.set_title("NLL of True Class Probabilities")
        ax.set_xlabel("Negative Log-Likelihood")
        ax.set_ylabel("Probability Density" if self.kde else "Count")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        output_filename = self.output_dir / f"k_{self.k}_nll_dist.png"
        plt.savefig(output_filename, dpi=300)


if __name__ == "__main__":

    # exp4
    comparison = {
        "ResNet-18, Cifar-10, 32x32, Batch-100": [
            Path(
                "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp4/resnet-18-cifar10-32x32-batch-100/cifar10_models/SINCERE_cifar10_resnet18_lr_0.04185432668024418_decay_0.0001_bsz_100_temp_0.1_trial_0_cosine_2025_07_14-00_35_03"
            ),
            Path(
                "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp4/resnet-18-cifar10-32x32-batch-100/cifar10_models/SupCon_cifar10_resnet18_lr_0.01_decay_0.001_bsz_100_temp_0.1_trial_0_cosine_2025_07_13-20_23_02"
            ),
        ],
    }

    # # exp2
    # comparison = {
    #     "ResNet-18, ImageNet-100, 32x32, Batch-1024": [
    #         Path(
    #             "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_40"
    #         ),
    #         Path(
    #             "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_30"
    #         ),
    #     ],
    # }

    for group_name_str, model_folders in comparison.items():
        for k in [1, 5, 15, 20]:
            output_dir = f"eval-new/figures/idw/exp4/{group_name_str}"

            PlotKNNProba(group_name_str, model_folders, k, output_dir)
