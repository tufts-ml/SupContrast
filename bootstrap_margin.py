import torch


def sim_median_margin(target_sim: torch.Tensor, noise_sim: torch.Tensor):
    return torch.abs(torch.median(target_sim) - torch.median(noise_sim))


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
    for model_folders in model_folders_group:
        b_scores_cache = []
        # print bootstrapped accuracy CIs
        for out_folder in model_folders:
            test_pred_dict = torch.load(out_folder / "test_pred_dict.pth")
            print(out_folder)
            print("Means, 95% CI Low, 95% CI High")
            metric_mean, ci_low, ci_high, b_scores = bootstrap_metric(
                test_pred_dict["target_sim"], test_pred_dict["noise_sim"], sim_median_margin)
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
