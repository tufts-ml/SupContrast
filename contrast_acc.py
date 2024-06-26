import torch


def contrastive_acc(embeds: torch.Tensor, labels: torch.tensor):
    """Get accuracy of supervised contrastive learning predictions

    Args:
        embeds (torch.Tensor): (B, V, D) embeddings of V augmented views of B images,
                               normalized over D dimension.
        labels (torch.tensor): (B,) integer class labels.


    Returns:
        torch.Tensor: Scalar accuracy.
    """
    batch_size = labels.shape[0]
    view_count = embeds.shape[1]
    # collapse view dimension, leaving views next to each other
    reshaped_embeds = torch.reshape(embeds, (-1, embeds.shape[2]))
    # repeat labels to account for views
    reshaped_labels = labels.repeat_interleave(embeds.shape[1])
    # calculate logits (BV, BV)
    logits = reshaped_embeds @ reshaped_embeds.T
    # prevent diagonal from being max while keeping same shape
    logits[torch.eye(batch_size * view_count, dtype=bool)] = -1e5
    # indices with greatest cosine similarity
    pred = torch.argmax(logits, dim=1)
    # 1 if predicted image with same label, otherwise 0
    acc_vec = (reshaped_labels == reshaped_labels[pred]).float()
    return torch.mean(acc_vec)


def test_contrastive_acc(train_embeds: torch.Tensor, test_embeds: torch.Tensor,
                         train_labels: torch.Tensor, test_labels: torch.Tensor):
    """1NN accuracy on test set given training set

    Args:
        train_embeds (torch.Tensor): (N1, D) embeddings of N1 images, normalized over D dimension.
        test_embeds (torch.Tensor): (N2, D) embeddings of N2 images, normalized over D dimension.
        train_labels (torch.Tensor): (N1,) integer class labels.
        test_labels (torch.Tensor): (N2,) integer class labels.
    """
    # calculate logits (N2, N1)
    logits = test_embeds @ train_embeds.T
    # indices with greatest cosine similarity
    pred = torch.argmax(logits, dim=1)
    # 1 if predicted image with same label, otherwise 0
    acc_vec = (test_labels == train_labels[pred]).float()
    return torch.mean(acc_vec)


def test_contrastive_acc_knn(train_embeds: torch.Tensor, test_embeds: torch.Tensor,
                             train_labels: torch.Tensor, test_labels: torch.Tensor,
                             knn: int):
    """Weighted KNN accuracy on test set given training set

    Args:
        train_embeds (torch.Tensor): (N1, D) embeddings of N1 images, normalized over D dimension.
        test_embeds (torch.Tensor): (N2, D) embeddings of N2 images, normalized over D dimension.
        train_labels (torch.Tensor): (N1,) integer class labels.
        test_labels (torch.Tensor): (N2,) integer class labels.
        knn (int): number of neighbors to use.
    """
    # assumes class labels are zero indexed
    num_classes = int(train_labels.max().item() + 1)
    # calculate logits (N2, N1)
    logits = test_embeds @ train_embeds.T
    # indices with greatest cosine similarity
    weights, indices = torch.topk(logits, knn, dim=1)
    # aggregate weights based on training class labels, with small uninitialized values
    pred = torch.zeros_like(test_labels)
    for i in range(len(test_labels)):
        pred_array = torch.empty((num_classes,))
        for label in range(num_classes):
            if label not in train_labels[indices[i]]:
                pred_array[label] = -1e5
            else:
                pred_array[label] = weights[i, label == train_labels[indices[i]]].sum()
        # select class with most weight as prediction
        pred[i] = torch.argmax(pred_array)
    # 1 if predicted image with same label, otherwise 0
    acc_vec = (test_labels == pred).float()
    return torch.mean(acc_vec)
