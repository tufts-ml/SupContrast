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
    fig, ax = plt.subplots()
    ax = sns.heatmap(conf_mat, vmin=0.25, vmax=0.85, cmap="Blues", annot=True, fmt=".2f",
                     square=True, ax=ax, xticklabels=labels, yticklabels=labels)
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
            hue=[class_labels[label]] * len(target_sim) + ["Noise"] * len(noise_sim),
            ax=ax, binrange=[0.15, 1], binwidth=1/100, element="step", stat="proportion",
            common_bins=True, common_norm=False)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Test Set Proportion")
        ax.set_ylim(0, .085)
        ax.set_title("SINCERE Loss" if "new" in out_folder.name else "SupCon Loss")
        fig.savefig(fig_folder / (out_folder.name + "_" + class_labels[label].lower() + ".pdf"))
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


if __name__ == "__main__":
    from pathlib import Path

    out_folders = [Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_old/"),
                   Path("save/linear/cifar100_models/cifar100_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar100_models/cifar100_lr_5.0_bsz_512_old/"),]
    # CIFAR10 labels
    class_labels = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
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
        # bound expectation
        print(f"Expectation of Bound: {expected_bound(pair_mat, labels)}")
        if "cifar100" in out_folder.name:
            continue
        # paired similarity histogram
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
        fig = plot_conf_mat(conf_mat, class_labels)
        plt.title("SINCERE Loss" if "new" in out_folder.name else "SupCon Loss")
        fig.savefig(fig_folder / (out_folder.name + ".pdf"), bbox_inches='tight')
        print()
