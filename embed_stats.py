import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import torch
from torch.nn.functional import cosine_similarity


def cos_sim_conf_mat(embeds, labels):
    n_labels = len(torch.unique(labels))
    conf_mat = torch.empty((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(i, n_labels):
            # compute cosine similarities between pairs from classes i and j
            i_embeds = embeds[labels == i].unsqueeze(0)
            j_embeds = embeds[labels == j].unsqueeze(1)
            conf_entries = cosine_similarity(i_embeds, j_embeds, dim=2)
            if i == j:
                # remove diagonal entries
                conf_entries = conf_entries[~torch.eye(conf_entries.shape[0], dtype=bool)]
            # take mean of cosine similarity
            conf_val = torch.mean(conf_entries)
            conf_mat[i, j] = conf_val
            conf_mat[j, i] = conf_val
    return conf_mat


def plot_conf_mat(conf_mat, labels="auto"):
    fig_folder = Path("figures/confusion")
    fig_folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax = sns.heatmap(conf_mat, vmin=0.25, vmax=0.85, cmap="Blues", annot=True, fmt=".2f",
                     square=True, ax=ax, xticklabels=labels, yticklabels=labels)
    plt.title("SINCERE Loss" if "new" in out_folder.name else "SupCon Loss")
    fig.savefig(fig_folder / (out_folder.name + ".pdf"), bbox_inches='tight')
    plt.close()


def pair_sim_mat(embeds):
    n_embeds = embeds.shape[0]
    pair_mat = torch.empty((n_embeds, n_embeds))
    # break up computation for each image for less memory usage
    for i in range(n_embeds):
        cur_pairs = cosine_similarity(embeds[i], embeds, dim=1)
        pair_mat[i] = cur_pairs
    return pair_mat


def pair_mat_to_target_noise(pair_mat, labels, target_label):
    # get top similarity from target distribution to target distribution
    target_mask = labels == target_label
    target_sim = pair_mat[target_mask][:, target_mask]
    # remove diagonal entries and flatten
    target_sim = target_sim[~torch.eye(target_sim.shape[0], dtype=bool)]
    # get top similarity from target distribution to noise distribution
    noise_sim = pair_mat[target_mask][:, ~target_mask].flatten()
    return target_sim, noise_sim


def pair_sim_hist(pred_dict, labels, class_labels, out_folder):
    fig_folder = Path("figures/hist") / out_folder.name
    fig_folder.mkdir(exist_ok=True)
    n_labels = len(torch.unique(labels))
    for label in range(n_labels):
        target_sim = pred_dict["target_sim"][pred_dict["target_label"] == label]
        noise_sim = pred_dict["noise_sim"][pred_dict["target_label"] == label]
        # plot histogram and save
        fig, ax = plt.subplots()
        sns.histplot(
            x=torch.hstack((target_sim, noise_sim)),
            hue=[class_labels[label]] * len(target_sim) + ["Noise"] * len(noise_sim),
            ax=ax, binrange=[0, 1], binwidth=1/100, element="step", stat="proportion",
            common_bins=True, common_norm=False)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Test Set Proportion")
        if "cifar2" in out_folder.name:
            ax.set_xlim(0, 1)
        else:
            ax.set_xlim(0.15, 1)
        ax.set_ylim(0, .085)
        ax.set_title("SINCERE Loss" if "SINCERE" in out_folder.name else "SupCon Loss")
        fig.savefig(fig_folder / (class_labels[label].lower() + ".pdf"))
        plt.close()


def pair_sim_curves(pair_mat, labels, class_labels, out_folder):
    roc_fig_folder = Path("figures/roc")
    roc_fig_folder.mkdir(exist_ok=True)
    pr_fig_folder = Path("figures/pr")
    pr_fig_folder.mkdir(exist_ok=True)
    n_labels = len(torch.unique(labels))
    for label in range(n_labels):
        target_sim, noise_sim = pair_mat_to_target_noise(pair_mat, labels, label)
        # plot ROC and save
        fig, ax = plt.subplots()
        sklearn.metrics.RocCurveDisplay.from_predictions(
            [class_labels[label]] * len(target_sim) + ["Noise"] * len(noise_sim),
            torch.hstack((target_sim, noise_sim)),
            pos_label=class_labels[label],
            name="SINCERE Loss" if "new" in out_folder.name else "SupCon Loss",
            ax=ax,
        )
        fig.savefig(roc_fig_folder / (out_folder.name + "_" + class_labels[label].lower() + ".pdf"))
        plt.close()
        # plot PR and save
        fig, ax = plt.subplots()
        sklearn.metrics.PrecisionRecallDisplay.from_predictions(
            [class_labels[label]] * len(target_sim) + ["Noise"] * len(noise_sim),
            torch.hstack((target_sim, noise_sim)),
            pos_label=class_labels[label],
            name="SINCERE Loss" if "new" in out_folder.name else "SupCon Loss",
            ax=ax,
        )
        fig.savefig(pr_fig_folder / (out_folder.name + "_" + class_labels[label].lower() + ".pdf"))
        plt.close()


def expected_bound(pair_mat, labels, temp=0.1):
    # apply temperature to cosine similarities
    pair_mat /= temp

    n_embeds = pair_mat.shape[0]
    mean_bound = 0
    # break up computation for each image for less memory usage
    for i in range(n_embeds):
        # separate out numerator terms and terms with other labels
        same_label = labels == labels[i]
        in_numer = same_label
        in_numer[i] = False
        # vector of all numerator terms
        log_numer = pair_mat[i][in_numer]
        # common terms for denominator
        log_base_denom = torch.logsumexp(pair_mat[i][~same_label], dim=0)
        # denominator with term from numerator
        log_denom = torch.logsumexp(
            torch.vstack((log_numer, torch.full_like(log_numer, log_base_denom))), dim=0)
        # bound for current image
        mean_bound += (torch.log(torch.sum(~same_label) + 1) + torch.mean(log_numer - log_denom))\
            / n_embeds
    return mean_bound.item()


def pred_dict(train_embeds, train_labels, pred_embeds, pred_labels):
    """Get vectors about 1NN predictions on pred set paseed on training set

    Args:
        train_embeds (torch.Tensor): (N1, D) embeddings of N1 images, normalized over D dimension.
        train_labels (torch.tensor): (N1,) integer class labels.
        pred_embeds (torch.Tensor): (N2, D) embeddings of N1 images, normalized over D dimension.
        pred_labels (torch.tensor): (N2,) integer class labels.

    Returns:
        dict: target_sim, target_label, noise_sim, noise_label, and nn_is_target tensors
    """
    # calculate logits (N2, N1)
    logits = pred_embeds @ train_embeds.T
    # calculate similarity for NN with same class
    target_mask = torch.logical_and(train_labels == pred_labels, logits < 1)
    target_sim = torch.max(logits[:, target_mask], dim=1)[0]
    # calculate similarity for NN with different class
    noise_sim, noise_nn_ind = torch.max(logits[:, train_labels != pred_labels], dim=1)
    # which label that NN has
    noise_label = train_labels[noise_nn_ind]
    return {
        "target_sim": target_sim,  # similarity to NN with same class
        "target_label": pred_labels,  # which label the target is
        "noise_sim": noise_sim,  # similarity to NN with different class
        "noise_label": noise_label,  # which label the NN with different class has
        "nn_is_target": target_sim > noise_sim,
    }


if __name__ == "__main__":
    from pathlib import Path

    out_folders = [Path("save/SupCon/cifar10_models/SINCERE_cifar10_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_20-22_04_43/"),  # noqa: E501
                   Path("save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.35_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_19-15_04_54/"),  # noqa: E501
                   Path("save/SupCon/cifar2_models/SINCERE_cifar2_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_22-09_32_40/"),  # noqa: E501
                   Path("save/SupCon/cifar2_models/SupCon_cifar2_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_22-09_32_42/"),  # noqa: E501
                   Path("save/SupCon/cifar100_models/SINCERE_cifar100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_28/"),  # noqa: E501
                   Path("save/SupCon/cifar100_models/SupCon_cifar100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_22-09_32_31/"),  # noqa: E501
                   Path("save/SupCon/imagenet100_models/SINCERE_imagenet100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_18/"),  # noqa: E501
                   Path("save/SupCon/imagenet100_models/SupCon_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_20/"),]  # noqa: E501
    # calculate embedding statistics
    for out_folder in out_folders:
        print(out_folder)
        # get prediction data for train to train and train to test
        # TODO update for test set
        if not (out_folder / "train_pred_dict.pth").exists():
            train_embeds = torch.load(out_folder / "train_embeds.pth")
            train_labels = torch.load(out_folder / "train_labels.pth")
            train_pred_dict = pred_dict(train_embeds, train_labels, train_embeds, train_labels)
            torch.save(train_pred_dict, out_folder / "train_pred_dict.pth")
        else:
            train_pred_dict = torch.load(out_folder / "train_pred_dict.pth")
        # datasets to skip following steps for
        if "cifar100" in out_folder.name or "imagenet100" in out_folder.name:
            continue
        if "cifar10" in out_folder.name:
            # CIFAR-10 labels
            class_labels = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog',
                            'Horse', 'Ship', 'Truck')
        else:
            # CIFAR-2 labels
            class_labels = ('Cat', 'Dog')
        # paired similarity histogram
        pair_sim_hist(train_pred_dict, train_labels, class_labels, out_folder)
        # # paired similarity ROC and PR curves
        # pair_sim_curves(pair_mat, labels, class_labels, out_folder)
        # # cosine similarity confusion matrix
        # if not (out_folder / "train_conf_mat.pth").exists():
        #     embeds = torch.load(out_folder / "train_embeds.pth")
        #     conf_mat = cos_sim_conf_mat(embeds, labels)
        #     torch.save(conf_mat, out_folder / "train_conf_mat.pth")
        # else:
        #     conf_mat = torch.load(out_folder / "train_conf_mat.pth")
        # print(conf_mat)
        # plot_conf_mat(conf_mat, class_labels)
        print()
