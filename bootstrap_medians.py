import torch


def sim_median(sim: torch.Tensor, empty: torch.Tensor):
    return torch.median(sim)


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
                   Path("save/SupCon/imagenet100_models/SupCon_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_20/"),]  # noqa: E501
    for key in ["target_sim", "noise_sim"]:
        print(f"{key} Evaluation:")
        b_scores_cache = []
        # print bootstrapped accuracy CIs
        for out_folder in out_folders:
            sim = torch.load(out_folder / "test_pred_dict.pth")[key]
            print(out_folder)
            print("Means, 95% CI Low, 95% CI High")
            metric_mean, ci_low, ci_high, b_scores = bootstrap_metric(
                sim, torch.empty_like(sim), sim_median)
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
