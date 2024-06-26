import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import torch


def pair_sim_hist(pred_dict, class_labels, out_folder):
    fig_folder = Path("figures/hist") / out_folder.name
    fig_folder.mkdir(exist_ok=True)
    n_labels = len(torch.unique(pred_dict["target_label"]))
    for label in range(n_labels):
        target_sim = pred_dict["target_sim"][pred_dict["target_label"] == label]
        noise_sim = pred_dict["noise_sim"][pred_dict["target_label"] == label]
        # plot histogram and save
        fig, ax = plt.subplots()
        sns_ax = sns.histplot(
            x=torch.hstack((target_sim, noise_sim)),
            hue=[class_labels[label]] * len(target_sim) + ["Noise"] * len(noise_sim),
            ax=ax, binrange=[-.1, 1], bins=100, element="step", stat="proportion",
            common_bins=True, common_norm=False)
        sns.move_legend(sns_ax, "upper left")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Test Set Proportion")
        ax.set_title("SINCERE Loss" if "SINCERE" in out_folder.name else "SupCon Loss")
        fig.savefig(fig_folder / (class_labels[label].lower() + ".pdf"))
        plt.close()


def pair_sim_hist_all(pred_dict, out_folder):
    fig_folder = Path("figures/hist") / out_folder.name
    fig_folder.mkdir(exist_ok=True)
    # plot histogram and save
    fig, ax = plt.subplots()
    sns_ax = sns.histplot(
        x=torch.hstack((pred_dict["target_sim"], pred_dict["noise_sim"])),
        hue=["Target"] * len(pred_dict["target_sim"]) + ["Noise"] * len(pred_dict["noise_sim"]),
        ax=ax, binrange=[-.1, 1], bins=100, element="step", stat="proportion",
        common_bins=True, common_norm=False)
    sns.move_legend(sns_ax, "upper left")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Test Set Proportion")
    ax.set_title("SINCERE Loss" if "SINCERE" in out_folder.name else "SupCon Loss")
    fig.savefig(fig_folder / "all.pdf")
    plt.close()


def pair_sim_curves(pred_dict, class_labels, out_folder):
    roc_fig_folder = Path("figures/roc") / out_folder.name
    roc_fig_folder.mkdir(exist_ok=True)
    pr_fig_folder = Path("figures/pr") / out_folder.name
    pr_fig_folder.mkdir(exist_ok=True)
    n_labels = len(torch.unique(pred_dict["target_label"]))
    for label in range(n_labels):
        target_sim = pred_dict["target_sim"][pred_dict["target_label"] == label]
        noise_sim = pred_dict["noise_sim"][pred_dict["target_label"] == label]
        # plot ROC and save
        fig, ax = plt.subplots()
        sklearn.metrics.RocCurveDisplay.from_predictions(
            [class_labels[label]] * len(target_sim) + ["Noise"] * len(noise_sim),
            torch.hstack((target_sim, noise_sim)),
            pos_label=class_labels[label],
            name="SINCERE Loss" if "SINCERE" in out_folder.name else "SupCon Loss",
            ax=ax,
        )
        fig.savefig(roc_fig_folder / (class_labels[label].lower() + ".pdf"))
        plt.close()
        # plot PR and save
        fig, ax = plt.subplots()
        sklearn.metrics.PrecisionRecallDisplay.from_predictions(
            [class_labels[label]] * len(target_sim) + ["Noise"] * len(noise_sim),
            torch.hstack((target_sim, noise_sim)),
            pos_label=class_labels[label],
            name="SINCERE Loss" if "SINCERE" in out_folder.name else "SupCon Loss",
            ax=ax,
        )
        fig.savefig(pr_fig_folder / (class_labels[label].lower() + ".pdf"))
        plt.close()


def expected_bound(pair_mat, labels, temp=0.1):
    # TODO revise to use train_embeds or pred_dict?
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


def pred_dict(train_embeds, train_labels, test_embeds, test_labels):
    """Get vectors about 1NN predictions on test set based on training set

    Args:
        train_embeds (torch.Tensor): (N1, D) embeddings of N1 images, normalized over D dimension.
        train_labels (torch.tensor): (N1,) integer class labels.
        test_embeds (torch.Tensor): (N2, D) embeddings of N1 images, normalized over D dimension.
        test_labels (torch.tensor): (N2,) integer class labels.

    Returns:
        dict: target_sim, target_label, noise_sim, noise_label, and nn_is_target tensors
    """
    # calculate logits (N2, N1)
    logits = test_embeds @ train_embeds.T
    # calculate similarity for NN with same class by masking different classes
    target_sim = torch.max(
        logits.masked_fill(train_labels.unsqueeze(0) != test_labels.unsqueeze(1), logits.min()),
        dim=1)[0]
    # calculate similarity for NN with different class by masking same class
    noise_sim, noise_nn_ind = torch.max(
        logits.masked_fill(train_labels.unsqueeze(0) == test_labels.unsqueeze(1), logits.min()),
        dim=1)
    # which label that NN has
    noise_label = train_labels[noise_nn_ind]
    return {
        "target_sim": target_sim,  # similarity to NN with same class
        "target_label": test_labels,  # which label the target is
        "noise_sim": noise_sim,  # similarity to NN with different class
        "noise_label": noise_label,  # which label the NN with different class has
        "nn_is_target": target_sim > noise_sim,
    }


if __name__ == "__main__":
    from pathlib import Path

    out_folders = [Path("save/SupCon/cifar10_models/SINCERE_cifar10_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_20-22_04_43/"),  # noqa: E501
                   Path("save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.35_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_19-15_04_54/"),  # noqa: E501
                   Path("save/SupCon/cifar10_models/EpsSupInfoNCE_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_21-12_28_30/"),  # noqa: E501
                   Path("save/SupCon/cifar2_models/SINCERE_cifar2_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_22-09_32_40/"),  # noqa: E501
                   Path("save/SupCon/cifar2_models/SupCon_cifar2_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_22-09_32_42/"),  # noqa: E501
                   Path("save/SupCon/cifar2_models/EpsSupInfoNCE_cifar2_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_21-12_52_23/"),  # noqa: E501
                   Path("save/SupCon/cifar100_models/SINCERE_cifar100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_28/"),  # noqa: E501
                   Path("save/SupCon/cifar100_models/SupCon_cifar100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_22-09_32_31/"),  # noqa: E501
                   Path("save/SupCon/cifar100_models/EpsSupInfoNCE_cifar100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_21-12_52_07/"),  # noqa: E501
                   Path("save/SupCon/imagenet100_models/SINCERE_imagenet100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_18/"),  # noqa: E501
                   Path("save/SupCon/imagenet100_models/SupCon_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_20/"),  # noqa: E501
                   Path("save/SupCon/imagenet100_models/EpsSupInfoNCE_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_22-09_31_46/"),]  # noqa: E501
    # calculate embedding statistics
    for out_folder in out_folders:
        print(out_folder)
        # get prediction data for test with 1NN on train
        if not out_folder.exists():
            print(f"Folder not found, skipping {out_folder}")
            continue
        if not (out_folder / "test_pred_dict.pth").exists():
            train_embeds = torch.load(out_folder / "train_embeds.pth")
            train_labels = torch.load(out_folder / "train_labels.pth")
            test_embeds = torch.load(out_folder / "test_embeds.pth")
            test_labels = torch.load(out_folder / "test_labels.pth")
            test_pred_dict = pred_dict(train_embeds, train_labels, test_embeds, test_labels)
            torch.save(test_pred_dict, out_folder / "test_pred_dict.pth")
        else:
            test_pred_dict = torch.load(out_folder / "test_pred_dict.pth")
        # print median target and noise similarities
        print(f"Median Target Similarity: {torch.median(test_pred_dict['target_sim'])}")
        print(f"Median Noise Similarity: {torch.median(test_pred_dict['noise_sim'])}")
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
        # paired similarity histogram for all classes
        pair_sim_hist_all(test_pred_dict, out_folder)
        # paired similarity histogram for individual classes
        pair_sim_hist(test_pred_dict, class_labels, out_folder)
        # paired similarity ROC and PR curves
        pair_sim_curves(test_pred_dict, class_labels, out_folder)
