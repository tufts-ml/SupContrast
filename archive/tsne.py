from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch


if __name__ == "__main__":
    out_folders = [Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_old/")]
    fig_folder = Path("figures/tnse")
    fig_folder.mkdir(exist_ok=True)
    # CIFAR10 labels
    class_labels = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    # calculate embedding statistics
    for out_folder in out_folders:
        print(out_folder)
        dist_pair_mat = 1 - torch.load(out_folder / "pair_mat.pth")
        dist_pair_mat[torch.logical_and(dist_pair_mat > -1e5, dist_pair_mat < 1e-5)] = 0
        labels = torch.load(out_folder / "labels.pth")
        proj_embedding = TSNE(perplexity=50, init="random", metric="precomputed").fit_transform(
            dist_pair_mat.numpy())
        # labeled scatter plot
        fig, ax = plt.subplots()
        for label in torch.unique(labels):
            is_label = label == labels
            plt.scatter(proj_embedding[is_label, 0], proj_embedding[is_label, 1],
                        label=class_labels[int(label)], alpha=.5)
        plt.legend()
        plt.title("SINCERE Loss" if "new" in out_folder.name else "SupCon Loss")
        plt.savefig(fig_folder / (out_folder.name + ".pdf"), bbox_inches='tight')
