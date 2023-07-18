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
    """
    bootstrapped_scores = None
    rng = torch.random.manual_seed(rng_seed)
    # bootstrap
    for _ in range(n_bootstraps):
        sample_idx = torch.randint(y_pred.shape[0], size=(y_pred.shape[0],), generator=rng)
        score = torch.Tensor(metric_func(y_pred[sample_idx], y_true[sample_idx]))
        # store results from each run along axis 0, with other axes' shape determined by metric
        if bootstrapped_scores is None:
            bootstrapped_scores = score.unsqueeze(0)
        else:
            bootstrapped_scores = torch.vstack((bootstrapped_scores, score))
    # compute mean and confidence interval
    metric_mean = torch.mean(bootstrapped_scores, dim=0)
    ci_low = torch.quantile(bootstrapped_scores, 0.025, dim=0)
    ci_high = torch.quantile(bootstrapped_scores, 0.975, dim=0)
    return (metric_mean, ci_low, ci_high)


if __name__ == "__main__":
    from functools import partial
    from pathlib import Path

    from util import accuracy

    out_folders = [Path("save/linear/cifar100_models/cifar100_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar100_models/cifar100_lr_5.0_bsz_512_old/")]
    metric = partial(accuracy, topk=(1, 5))
    for out_folder in out_folders:
        y_pred = torch.load(out_folder / "preds.pth")
        y_true = torch.load(out_folder / "labels.pth")
        print(out_folder)
        print("Means, 95% CI Low, 95% CI High")
        print(bootstrap_metric(y_pred, y_true, metric))
        print()
