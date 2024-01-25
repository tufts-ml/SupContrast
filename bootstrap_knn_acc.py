import torch


def test_contrastive_pred_knn(train_embeds: torch.Tensor, test_embeds: torch.Tensor,
                              train_labels: torch.Tensor, test_labels: torch.Tensor,
                              knn: int):
    """Weighted KNN accuracy on test set given training set, returning class prediction

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
    return pred


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor):
    return (y_true == y_pred).float().mean()


if __name__ == "__main__":
    from pathlib import Path

    from bootstrap_lin_acc import bootstrap_metric, bootstrap_dif

    # out_folders should have pairs of models to compare
    out_folders = [Path("save/SupCon/cifar10_models/SINCERE_cifar10_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_20-22_04_43/"),  # noqa: E501
                   Path("save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.35_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_19-15_04_54/"),  # noqa: E501
                   Path("save/SupCon/cifar2_models/SINCERE_cifar2_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_22-09_32_40/"),  # noqa: E501
                   Path("save/SupCon/cifar2_models/SupCon_cifar2_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_22-09_32_42/"),  # noqa: E501
                   Path("save/SupCon/cifar100_models/SINCERE_cifar100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_28/"),  # noqa: E501
                   Path("save/SupCon/cifar100_models/SupCon_cifar100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_22-09_32_31/"),  # noqa: E501
                   Path("save/SupCon/imagenet100_models/SINCERE_imagenet100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_18/"),  # noqa: E501
                   Path("save/SupCon/imagenet100_models/SupCon_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_20/"),  # noqa: E501
                   Path("save/SupCon/cars_models/SINCERE_cars_resnet50_lr_1.0_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_24-15_29_37/"),  # noqa: E501
                   Path("save/SupCon/cars_models/SupCon_cars_resnet50_lr_1.0_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_24-15_29_43/"),  # noqa: E501
                   Path("save/SupCon/aircraft_models/SINCERE_aircraft_resnet50_lr_0.85_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_24-20_32_11/"),  # noqa: E501
                   Path("save/SupCon/aircraft_models/SupCon_aircraft_resnet50_lr_0.85_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_24-20_32_11/")]  # noqa: E501
    for k in [1, 5]:
        print(f"{k}NN Evaluation:")
        b_scores_cache = []
        # print bootstrapped accuracy CIs
        for out_folder in out_folders:
            y_pred = test_contrastive_pred_knn(
                torch.load(out_folder / "train_embeds.pth"),
                torch.load(out_folder / "test_embeds.pth"),
                torch.load(out_folder / "train_labels.pth"),
                torch.load(out_folder / "test_labels.pth"),
                k
            )
            y_true = torch.load(out_folder / "test_labels.pth")
            print(out_folder)
            print("Means, 95% CI Low, 95% CI High")
            metric_mean, ci_low, ci_high, b_scores = bootstrap_metric(y_pred, y_true, accuracy)
            b_scores_cache.append(b_scores)
            print(metric_mean, ci_low, ci_high)
            print()
        # print accuracy difference for each pair of models
        for i in range(len(out_folders) // 2):
            i1 = 2 * i
            i2 = 2 * i + 1
            print("Accuracy Difference 95% CI for:")
            print(out_folders[i1])
            print(out_folders[i2])
            print(bootstrap_dif(b_scores_cache[i1], b_scores_cache[i2]))
            print()
