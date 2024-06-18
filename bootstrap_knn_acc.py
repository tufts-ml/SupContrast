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

    # model_folders_group should have lists of models to compare
    model_folders_group = [
        [
            Path("2024_03_save/SupCon/cifar10_models/SINCERE_cifar10_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_20-22_04_43/"),  # noqa: E501
            Path("2024_03_save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.35_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_19-15_04_54/"),  # noqa: E501
            Path("2024_03_save/SupCon/cifar10_models/EpsSupInfoNCE_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_21-12_28_30/"),  # noqa: E501
            Path("save/SupCon/cifar10_models/SINCERE_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_act_arccos_trial_0_cosine_warm_2024_05_17-15_14_23/"),  # noqa: E501
        ],
        [
            Path("2024_03_save/SupCon/cifar100_models/SINCERE_cifar100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_28/"),  # noqa: E501
            Path("2024_03_save/SupCon/cifar100_models/SupCon_cifar100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_22-09_32_31/"),  # noqa: E501
            Path("2024_03_save/SupCon/cifar100_models/EpsSupInfoNCE_cifar100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_21-12_52_07/"),  # noqa: E501
            Path("save/SupCon/cifar100_models/SINCERE_cifar100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_act_arccos_trial_0_cosine_warm_2024_05_17-15_15_38/"),  # noqa: E501
        ],
    ]
    for k in [1, 5]:
        print(f"{k}NN Evaluation:")
        for model_folders in model_folders_group:
            b_scores_cache = []
            # print bootstrapped accuracy CIs
            for out_folder in model_folders:
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
            for i in range(1, len(model_folders)):
                for j in range(i):
                    print("Accuracy Difference 95% CI for:")
                    print(model_folders[j])
                    print(model_folders[i])
                    print(bootstrap_dif(b_scores_cache[j], b_scores_cache[i]))
                    print()
