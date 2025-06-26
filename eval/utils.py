import torch
import torch.nn.functional as F


def test_contrastive_pred_probs_knn(train_embeds: torch.Tensor, test_embeds: torch.Tensor,
                                    train_labels: torch.Tensor, test_labels: torch.Tensor,
                                    knn: int):
    """Weighted KNN accuracy on test set given training set, returning class probability"""
    num_classes = int(train_labels.max().item() + 1)
    logits = test_embeds @ train_embeds.T
    weights, indices = torch.topk(logits, knn, dim=1)

    pred_probs = torch.zeros((len(test_labels), num_classes))
    for i in range(len(test_labels)):
        neighbor_labels = train_labels[indices[i]]
        neighbor_weights = weights[i]

        scores = torch.zeros(num_classes)
        for j in range(len(neighbor_labels)):
            label = int(neighbor_labels[j])
            weight = neighbor_weights[j]
            scores[label] += weight

        pred_probs[i] = F.softmax(scores, dim=0)

    return pred_probs


def read_embeds(model_folders):

    save = {}
    for model_folder in model_folders:
        # SINCERE or SupCon
        model_name = model_folder.name.split('_')[0]

        train_embeds = torch.load(model_folder / "train_embeds.pth", weights_only=False)
        test_embeds = torch.load(model_folder / "test_embeds.pth", weights_only=False)
        train_labels = torch.load(model_folder / "train_labels.pth", weights_only=False)
        test_labels = torch.load(model_folder / "test_labels.pth", weights_only=False)

        save[model_name] = [train_embeds, test_embeds, train_labels, test_labels]

    return save