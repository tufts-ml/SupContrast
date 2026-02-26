import torch


def bootstrap_metric(y_pred, y_true, metric_func,
                     n_bootstraps=1000, rng_seed=123):
    """Compute test set boostrapping of a metric
    Args:
        y_pred (tensor): Model predictions for some output y
        y_true (tensor): True value of output y
        metric_func (function): function with parameters (y_pred, y_true)
                                returning a Tensor castable metric
        n_bootstraps (int, optional): Number of bootstrap samples to take.
                                      Defaults to 200.
        rng_seed (int, optional): Random seed for reproducibility.
                                  Defaults to 123.
    Returns:
        tuple: metric_mean: Tensor with bootstrapped mean of metric
               ci_low: Low value from 95% confidence interval
               ci_high: High value from 95% confidence interval
               b_scores: Bootstrapped metric outputs
    """
    b_scores = None
    rng = torch.random.manual_seed(rng_seed)
    # bootstrap
    for _ in range(n_bootstraps):
        sample_idx = torch.randint(y_pred.shape[0], size=(y_pred.shape[0],), generator=rng)
        score = torch.Tensor(metric_func(y_pred[sample_idx], y_true[sample_idx]))
        # store results from each run along axis 0, with other axes' shape determined by metric
        if b_scores is None:
            b_scores = score.unsqueeze(0)
        else:
            b_scores = torch.vstack((b_scores, score))
    # compute mean and confidence interval
    metric_mean = torch.mean(b_scores, dim=0)
    ci_low = torch.quantile(b_scores, 0.025, dim=0)
    ci_high = torch.quantile(b_scores, 0.975, dim=0)
    return (metric_mean, ci_low, ci_high, b_scores)


def bootstrap_dif(b_scores_1, b_scores_2):
    """Examine the difference of two bootstrapped metrics

    Args:
        b_scores_1 (Tensor): Bootstrapped metric outputs, with same seed as 2
        b_scores_2 (Tensor): Bootstrapped metric outputs, with same seed as 1
    Returns:
        tensor: True if 95% CI does not contain 0 so result is statistically significant
                False if 95% CI contains 0 so result is not statistically significant
    """
    dif_scores = b_scores_1 - b_scores_2
    # compute confidence interval of the difference
    ci_low = torch.quantile(dif_scores, 0.025, dim=0)
    ci_high = torch.quantile(dif_scores, 0.975, dim=0)
    return ~torch.logical_and(ci_low <= 0, ci_high >= 0)


if __name__ == "__main__":
    from functools import partial
    from pathlib import Path
    from util import accuracy
    import itertools

    out_folders = {
        "pet": {
            "SupCon         ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/pet/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/pet_lr_16.459140251421243_decay_0.0_bsz_128_SupCon_imagenet100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_29-05_55_46_cosine_warm/",
            "SINCERE        ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/pet/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/pet_lr_49.99999999999999_decay_0.0_bsz_128_SINCERE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_cosine_warm_2025_11_27-22_17_28_cosine_warm/",
            "EpsSupInfoNCE  ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/pet/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/pet_lr_49.99999999999999_decay_1e-05_bsz_128_EpsSupInfoNCE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_eps_0.1_cosine_warm_2025_12_12-16_04_00_cosine_warm/",
        },
        "dtd": {
            "SupCon         ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/dtd/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/dtd_lr_49.99999999999999_decay_5e-05_bsz_128_SupCon_imagenet100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_29-05_55_46_cosine_warm/",
            "SINCERE        ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/dtd/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/dtd_lr_1.023292992280754_decay_1e-05_bsz_128_SINCERE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_cosine_warm_2025_11_27-22_17_28_cosine_warm/",
            "EpsSupInfoNCE  ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/dtd/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/dtd_lr_1.783534149330136_decay_0.0001_bsz_128_EpsSupInfoNCE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_eps_0.1_cosine_warm_2025_12_12-16_04_00_cosine_warm/",
        },
        "aircraft": {
            "SupCon         ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/aircraft/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/aircraft_lr_49.99999999999999_decay_0.0_bsz_128_SupCon_imagenet100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_29-05_55_46_cosine_warm/",
            "SINCERE        ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/aircraft/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/aircraft_lr_49.99999999999999_decay_0.0_bsz_128_SINCERE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_cosine_warm_2025_11_27-22_17_28_cosine_warm/",
            "EpsSupInfoNCE  ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/aircraft/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/aircraft_lr_28.687227341990756_decay_0.0_bsz_128_EpsSupInfoNCE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_eps_0.1_cosine_warm_2025_12_12-16_04_00_cosine_warm/",
        },
        "food101": {
            "SupCon         ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/food101/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/food101_lr_5.418065956319098_decay_0.0_bsz_128_SupCon_imagenet100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_29-05_55_46_cosine_warm/",
            "SINCERE        ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/food101/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/food101_lr_5.418065956319098_decay_0.0_bsz_128_SINCERE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_cosine_warm_2025_11_27-22_17_28_cosine_warm/",
            "EpsSupInfoNCE  ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/food101/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/food101_lr_5.418065956319098_decay_0.0_bsz_128_EpsSupInfoNCE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_eps_0.1_cosine_warm_2025_12_12-16_04_00_cosine_warm/",
        },
        "flowers": {
            "SupCon         ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/flowers/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/flowers_lr_28.687227341990756_decay_0.0_bsz_128_SupCon_imagenet100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_29-05_55_46_cosine_warm/",
            "SINCERE        ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/flowers/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/flowers_lr_28.687227341990756_decay_1e-05_bsz_128_SINCERE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_cosine_warm_2025_11_27-22_17_28_cosine_warm/",
            "EpsSupInfoNCE  ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/flowers/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/flowers_lr_49.99999999999999_decay_5e-05_bsz_128_EpsSupInfoNCE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_eps_0.1_cosine_warm_2025_12_12-16_04_00_cosine_warm/",
        },
        "cars": {
            "SupCon         ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/cars/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/cars_lr_28.687227341990756_decay_0.0_bsz_128_SupCon_imagenet100_resnet50_lr_0.6898648307306074_decay_0.0001_bsz_512_temp_0.12_trial_0_cosine_warm_2025_11_29-05_55_46_cosine_warm/",
            "SINCERE        ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/cars/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/cars_lr_49.99999999999999_decay_0.0_bsz_128_SINCERE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_cosine_warm_2025_11_27-22_17_28_cosine_warm/",
            "EpsSupInfoNCE  ": "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp-final-imagenet100/transfer-final/cars/size-224/best_val_separation_t1_wo_projection_head/100_epochs_final/cars_lr_49.99999999999999_decay_0.0_bsz_128_EpsSupInfoNCE_imagenet100_resnet50_lr_0.36238983183884776_decay_0.0001_bsz_512_temp_0.08_trial_0_eps_0.1_cosine_warm_2025_12_12-16_04_00_cosine_warm/",
        },
    }

    for dataset_name, models_dict in out_folders.items():

        print("\n")
        print(dataset_name)
        print("\n")
        
        dataset_b_scores = {}
        metric = partial(accuracy, topk=(1,))

        for model_name, model_path in models_dict.items():

            model_path = Path(model_path)
            y_pred = torch.load(model_path / "preds.pth", weights_only=False)
            y_true = torch.load(model_path / "labels.pth", weights_only=False)
            
            metric_mean, _, _, b_scores = bootstrap_metric(y_pred, y_true, metric)
            dataset_b_scores[model_name] = b_scores
            
            print(f"\t{model_name} \t : {metric_mean.item()}")

        model_names = list(dataset_b_scores.keys())

        print("\n")
        for name1, name2 in itertools.combinations(model_names, 2):
            scores1 = dataset_b_scores[name1]
            scores2 = dataset_b_scores[name2]
            
            is_significant = bootstrap_dif(scores1, scores2)
            
            if is_significant.item():
                status = "SIGNIFICANT"
            else:
                status = "Not Significant"
                
            print(f"\t{name1} vs {name2} \t : {status}")
        
        print("\n")