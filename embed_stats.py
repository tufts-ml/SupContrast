import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.nn.functional import cosine_similarity


def cos_sim_per_class(embeds, labels):
    cos_dists = torch.empty((0,))
    for label in torch.unique(labels):
        l_embeds = embeds[labels == label]
        cos_dist_mat = cosine_similarity(l_embeds.unsqueeze(0), l_embeds.unsqueeze(1), dim=2)
        # remove diagonal and average
        cos_dist = torch.mean(cos_dist_mat[~torch.eye(cos_dist_mat.shape[0], dtype=bool)])
        cos_dists = torch.hstack((cos_dists, cos_dist))
    return cos_dists


def cos_sim_conf_mat(embeds, labels):
    n_labels = len(torch.unique(labels))
    conf_mat = torch.empty((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(i, n_labels):
            # compute cosine similarities between pairs from classes i and j
            i_embeds = embeds[labels == labels[i]].unsqueeze(0)
            j_embeds = embeds[labels == labels[j]].unsqueeze(1)
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
    fig, ax = plt.subplots()
    ax = sns.heatmap(conf_mat, vmin=0, vmax=1, cmap="Blues", annot=True, fmt=".2f", square=True,
                     ax=ax, xticklabels=labels, yticklabels=labels)
    return fig


def pair_sim_mat(embeds):
    n_embeds = embeds.shape[0]
    pair_mat = torch.empty((n_embeds, n_embeds))
    # break up computation for each image for less memory usage
    for i in range(n_embeds):
        cur_pairs = cosine_similarity(embeds[i], embeds, dim=1)
        pair_mat[i] = cur_pairs
    return pair_mat


def pair_sim_hist(pair_mat, labels, class_labels, fig_folder, out_folder):
    n_labels = len(torch.unique(labels))
    for label in range(n_labels):
        # get similarities from target distribution
        target_mask = labels == label
        target_sim = pair_mat[target_mask][:, target_mask]
        # remove diagonal entries and flatten
        target_sim = target_sim[~torch.eye(target_sim.shape[0], dtype=bool)]
        # get similarities from noise distribution
        noise_sim = pair_mat[target_mask][:, ~target_mask].flatten()
        # plot histogram and save
        fig, ax = plt.subplots()
        sns.histplot(
            x=torch.hstack((target_sim, noise_sim)),
            hue=[class_labels[label]] * len(target_sim) + ["noise"] * len(noise_sim),
            ax=ax, binrange=[0, 1], bins=100, element="step", stat="proportion", common_norm=False)
        fig.savefig(fig_folder / (out_folder.name + "_" + class_labels[label] + ".png"))


if __name__ == "__main__":
    from pathlib import Path

    out_folders = [Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_old/")]
    # CIFAR10 labels
    class_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # calculate embedding statistics
    for out_folder in out_folders:
        print(out_folder)
        # cosine similarity pair matrix
        if not (out_folder / "pair_mat.pth").exists():
            embeds = torch.load(out_folder / "embeds.pth")
            pair_mat = pair_sim_mat(embeds)
            torch.save(pair_mat, out_folder / "pair_mat.pth")
        else:
            pair_mat = torch.load(out_folder / "pair_mat.pth")
        labels = torch.load(out_folder / "labels.pth")
        fig_folder = Path("figures/hist")
        fig_folder.mkdir(exist_ok=True)
        pair_sim_hist(pair_mat, labels, class_labels, fig_folder, out_folder)
        # cosine similarity confusion matrix
        if not (out_folder / "conf_mat.pth").exists():
            embeds = torch.load(out_folder / "embeds.pth")
            conf_mat = cos_sim_conf_mat(embeds, labels)
            torch.save(conf_mat, out_folder / "conf_mat.pth")
        else:
            conf_mat = torch.load(out_folder / "conf_mat.pth")
        print(conf_mat)
        fig_folder = Path("figures/confusion")
        fig_folder.mkdir(exist_ok=True)
        plot_conf_mat(conf_mat, class_labels).savefig(fig_folder / (out_folder.name + ".png"))
        print()
