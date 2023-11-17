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

    b_scores_cache = []
    metric = partial(accuracy, topk=(1, 5))
    # out_folders should have pairs of models to compare
    out_folders = [Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_old/"),
                   Path("save/linear/cifar100_models/cifar100_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar100_models/cifar100_lr_5.0_bsz_512_old/"),
                   Path("save/linear/imagenet100_models/imagenet100_lr_5.0_bsz_512_new/"),
                   Path("save/linear/imagenet100_models/imagenet100_lr_5.0_bsz_512_old/")]
    # print bootstrapped accuracy CIs
    for out_folder in out_folders:
        y_pred = torch.load(out_folder / "preds.pth")
        y_true = torch.load(out_folder / "labels.pth")
        print(out_folder)
        print("Means, 95% CI Low, 95% CI High")
        metric_mean, ci_low, ci_high, b_scores = bootstrap_metric(y_pred, y_true, metric)
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
