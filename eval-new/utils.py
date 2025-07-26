import torch
import torch.nn.functional as F


def read_embeds(model_folders):
    save = {}
    for model_folder in model_folders:
        # SINCERE or SupCon
        model_name = model_folder.name.split("_")[0]

        train_embeds = torch.load(model_folder / "train_embeds.pth", weights_only=False)
        test_embeds = torch.load(model_folder / "test_embeds.pth", weights_only=False)
        train_labels = torch.load(model_folder / "train_labels.pth", weights_only=False)
        test_labels = torch.load(model_folder / "test_labels.pth", weights_only=False)

        save[model_name] = [train_embeds, test_embeds, train_labels, test_labels]

    return save


def test_contrastive_pred_probs_knn(
    train_embeds: torch.Tensor,
    test_embeds: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    knn: int,
    raw: bool = False,
):
    """
    Calculates class probabilities using a weighted k-NN classifier with
    Inverse Distance Weighting.

    1. Converts cosine similarity to cosine distance (d = 1 - sim).
    2. Weights each neighbor vote by the inverse of its distance.
    3. Normalizes the scores to get probability distribution.
    """
    num_classes = int(train_labels.max().item() + 1)
    similarities = test_embeds @ train_embeds.T

    top_similarities, indices = torch.topk(similarities, knn, dim=1)
    epsilon = 1e-6

    pred = torch.zeros((len(test_labels), num_classes), device=test_embeds.device)
    for i in range(len(test_labels)):
        distances = 1.0 - top_similarities[i]

        neighbor_labels = train_labels[indices[i]]
        neighbor_weights = 1.0 / (distances + epsilon)

        scores = torch.zeros(num_classes, device=test_embeds.device)

        # Without for loop: add each neighbor's weight to the total score for its class
        scores.index_add_(0, neighbor_labels.long(), neighbor_weights)

        if raw:
            pred[i] = scores
        else:
            total_weight = torch.sum(neighbor_weights)
            pred[i] = scores / total_weight

    return pred
