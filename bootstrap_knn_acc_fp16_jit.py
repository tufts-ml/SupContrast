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

        # ## exp1

        # # batch-1024
        # # resnet-18-imagenet-100-112x112
        # [
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-112x112/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_27-05_20_54"),
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-112x112/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_27-22_12_17")
        # ],
        # # batch-1024
        # # resnet-18-imagenet-100-64x64
        # [
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-64x64/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_20-20_49_04"), 
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-64x64/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_20-20_22_04"), 
        # ],
        # # batch-1024
        # # resnet-18-imagenet-100-32x32
        # [
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_40"),
        #     Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_30")
        # ],

        ## exp2

        # batch-100
        # resnet-18-imagenet-100-32x32
        [
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-100/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.045_decay_0.0001_bsz_100_temp_0.1_trial_0_cosine_2025_06_27-19_00_45"), 
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-100/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.045_decay_0.0001_bsz_100_temp_0.1_trial_0_cosine_2025_06_27-19_00_45"), 
        ],
        # batch-1024
        # resnet-18-imagenet-100-32x32
        [
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SINCERE_imagenet100_resnet18_lr_0.75_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_40"),
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp1/resnet-18-imagenet-100-32x32/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.65_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_24-23_13_30")
        ],
        # batch-5000
        # resnet-18-imagenet-100-32x32
        [
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-5000/imagenet100_models/SINCERE_imagenet100_resnet18_lr_2.75_decay_0.0001_bsz_5000_temp_0.1_trial_0_cosine_warm_2025_06_28-10_22_38"),
            Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-5000/imagenet100_models/SupCon_imagenet100_resnet18_lr_2.15_decay_0.0001_bsz_5000_temp_0.1_trial_0_cosine_warm_2025_06_28-10_18_52")
        ],

    ]
    for k in [1, 5]:
        print(f"{k}NN Evaluation:")
        for model_folders in model_folders_group:
            b_scores_cache = []
            # print bootstrapped accuracy CIs
            for out_folder in model_folders:
                y_pred = test_contrastive_pred_knn(
                    torch.load(out_folder / "train_embeds.pth", weights_only=False),
                    torch.load(out_folder / "test_embeds.pth", weights_only=False),
                    torch.load(out_folder / "train_labels.pth", weights_only=False),
                    torch.load(out_folder / "test_labels.pth", weights_only=False),
                    k
                )
                y_true = torch.load(out_folder / "test_labels.pth", weights_only=False)
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
