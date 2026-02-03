from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats


def mean_dif(x: np.ndarray, y: np.ndarray, axis=-1):
    return x.mean(axis=axis) - y.mean(axis=axis)


def bootstrap_adv_acc_diff(acc_val_1, acc_val_2, dataset_size):
    # reconstruct per-prediction accuracy
    acc_vec_1 = np.zeros(dataset_size)
    acc_vec_1[:int(acc_val_1 * dataset_size)] = 1
    acc_vec_2 = np.zeros(dataset_size)
    acc_vec_2[:int(acc_val_2 * dataset_size)] = 1
    np.random.shuffle(acc_vec_2)
    return scipy.stats.bootstrap((acc_vec_1, acc_vec_2), statistic=mean_dif, paired=True,
                                 n_resamples=200, method="percentile", random_state=0)


if __name__ == "__main__":
    dataset_test_sizes = {
        "cifar2": 2000,
        "cifar10": 10000,
        "cifar100": 10000,
        "imagenet100": 5000,
    }
    dataset_titles = {
        "cifar2": "CIFAR-2",
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "imagenet100": "ImageNet-100",
    }
    folder = Path("./results_attack_no_cache/")
    for dataset in ["cifar2", "cifar10", "cifar100", "imagenet100"]:
        df = pd.read_csv(folder / Path(f"attack_embedding_results_{dataset}.csv"))
        # accuracy diff bootstrap results
        diff_results = [bootstrap_adv_acc_diff(acc_val_1, acc_val_2, dataset_test_sizes[dataset])
                        for acc_val_1, acc_val_2 in
                        zip(df[df["model_name"] == "SINCERE"]["accuracy"],
                            df[df["model_name"] == "SupCon"]["accuracy"])]
        # array of significance (True >= .95)
        diff_sigs = np.array(
            [result.confidence_interval.low > 0 or result.confidence_interval.high < 0
             for result in diff_results])
        # make array include insignificance gaps
        last_true_ind = len(diff_sigs) - 1 - diff_sigs[::-1].argmax()
        diff_sigs[diff_sigs.argmax():last_true_ind] = True
        for method_name in ["SINCERE", "SupCon"]:
            plt.plot(df[df["model_name"] == method_name]["epsilon"][diff_sigs],
                     df[df["model_name"] == method_name]["accuracy"][diff_sigs],
                     label=method_name,)
            plt.title(dataset_titles[dataset])
        plt.legend()
        plt.savefig(folder / f"{dataset}.pdf")
        plt.close()
