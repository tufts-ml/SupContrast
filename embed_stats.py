import torch
from torch.nn.functional import cosine_similarity


def cos_dist_per_class(embeds, labels):
    cos_dists = torch.empty((0,))
    for label in torch.unique(labels):
        l_embeds = embeds[labels == label]
        cos_dist_mat = cosine_similarity(l_embeds.unsqueeze(0), l_embeds.unsqueeze(1), dim=2)
        # remove diagonal and average
        cos_dist = torch.mean(cos_dist_mat[~torch.eye(cos_dist_mat.shape[0], dtype=bool)])
        cos_dists = cos_dists.hstack((cos_dists, cos_dist))
    return cos_dists


if __name__ == "__main__":
    from pathlib import Path

    out_folders = [Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_old/"),
                   Path("save/linear/cifar100_models/cifar100_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar100_models/cifar100_lr_5.0_bsz_512_old/")]
    # calculate embedding statistics
    for out_folder in out_folders:
        embeds = torch.load(out_folder / "embeds.pth")
        labels = torch.load(out_folder / "labels.pth")
        print(out_folder)
        print(cos_dist_per_class(embeds, labels))
        print()
