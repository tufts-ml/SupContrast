import sklearn.metrics
import torch

import bootstrap_acc
import embed_stats


def bootstrap_nce_metric(pair_mat, labels, metric_func,
                         n_bootstraps=100, rng_seed=123):
    b_scores = None
    rng = torch.random.manual_seed(rng_seed)
    # bootstrap
    for _ in range(n_bootstraps):
        # random indices for 90% of the set without repeats
        sample_idx = torch.randperm(pair_mat.shape[0], generator=rng)[:pair_mat.shape[0]//10 * 9]
        score = torch.Tensor(metric_func(pair_mat[sample_idx][:, sample_idx], labels[sample_idx]))
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


def nce_av_precision(pair_mat, labels):
    av_precisions = []
    for label in range(len(torch.unique(labels))):
        target_sim, noise_sim = embed_stats.pair_mat_to_target_noise(pair_mat, labels, label)
        av_precisions.append(sklearn.metrics.average_precision_score(
            [1] * len(target_sim) + [0] * len(noise_sim),
            torch.hstack((target_sim, noise_sim))))
    return torch.mean(torch.Tensor(av_precisions))


if __name__ == "__main__":
    from pathlib import Path

    b_scores_cache = []
    # out_folders should have pairs of models to compare
    out_folders = [Path("save/linear/cifar2_models/cifar2_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar2_models/cifar2_lr_5.0_bsz_512_old/"),
                   Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_old/"),
                   Path("save/linear/cifar100_models/cifar100_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar100_models/cifar100_lr_5.0_bsz_512_old/")]
    # print bootstrapped average precision CIs
    for out_folder in out_folders:
        pair_mat = torch.load(out_folder / "pair_mat.pth")
        labels = torch.load(out_folder / "labels.pth")
        print(out_folder)
        print("Means, 95% CI Low, 95% CI High")
        metric_mean, ci_low, ci_high, b_scores = bootstrap_nce_metric(
            pair_mat, labels, nce_av_precision)
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
        print(bootstrap_acc.bootstrap_dif(b_scores_cache[i1], b_scores_cache[i2]))
        print()
