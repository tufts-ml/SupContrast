import torch
from tqdm import tqdm

from networks.resnet_big import SupConResNet


def load_supcon_model(encoder_path, arch):
    """
    Loads a SupConResNet model from a checkpoint path, assuming
    evaluation is after the 128-dim projection head.
    """

    # Load Encoder + Projection Head
    embedding_model = SupConResNet(name=arch, head="mlp", feat_dim=128)
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)

    state_dict = ckpt["model"]
    if list(state_dict.keys())[0].startswith("module."):
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

    embedding_model.load_state_dict(state_dict, strict=True)

    return embedding_model


def compute_features(model, data_loader, split_name, device):
    """
    Computes features for a given data split and returns them.
    Assumes features are from after the projection head.
    """

    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for data, labels in tqdm(
            data_loader, desc=f"Computing {split_name} features", leave=False
        ):
            images = data[0]
            images = images.to(device)

            # model(images) gives post-head features
            features = model(images)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def calculate_target_noise_separation(
    train_embeds, train_labels, eval_embeds, eval_labels, percentile=0.99
):
    """
    Calculates the separation between target and noise similarities at a given percentile.
    """
    similarities = eval_embeds @ train_embeds.T
    target_percentile_sims, noise_percentile_sims = [], []

    for i in range(len(eval_labels)):
        eval_label = eval_labels[i]
        sims = similarities[i]
        target_mask = train_labels == eval_label

        target_sims = sims[target_mask]
        if len(target_sims) > 0:
            target_percentile_sims.append(torch.quantile(target_sims, percentile))

        noise_sims = sims[~target_mask]
        if len(noise_sims) > 0:
            noise_percentile_sims.append(torch.quantile(noise_sims, percentile))

    if not target_percentile_sims or not noise_percentile_sims:
        return 0.0

    target_median = torch.median(torch.stack(target_percentile_sims))
    noise_median = torch.median(torch.stack(noise_percentile_sims))

    return abs(target_median - noise_median).item()


def calculate_target_noise_separation_topk(
    train_embeds, train_labels, eval_embeds, eval_labels, k=1
):
    """
    Calculates the separation between the k-th target and k-th noise similarities.
    """
    similarities = eval_embeds @ train_embeds.T
    target_kth_sims, noise_kth_sims = [], []

    for i in range(len(eval_labels)):
        eval_label = eval_labels[i]
        sims = similarities[i]
        target_mask = train_labels == eval_label

        target_sims = sims[target_mask]
        if len(target_sims) > 0:
            target_kth_sims.append(torch.topk(target_sims, k).values[-1])

        noise_sims = sims[~target_mask]
        if len(noise_sims) > 0:
            noise_kth_sims.append(torch.topk(noise_sims, k).values[-1])

    if not target_kth_sims or not noise_kth_sims:
        return 0.0

    target_median = torch.median(torch.stack(target_kth_sims))
    noise_median = torch.median(torch.stack(noise_kth_sims))

    return abs(target_median - noise_median).item()
