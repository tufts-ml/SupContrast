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
    datasets = ["cifar2", "cifar10", "cifar100", "imagenet100"]
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

    # init figure
    plt.figure(figsize=(16, 4))
    plt.subplots_adjust(left=0.05,
                        right=0.99,
                        bottom=0.13,
                        top=0.92)
    plt.rcParams.update({'font.size': 16})
    # subplot spanning all 4 without any components besides x label
    ax = plt.subplot(1, 4, (1, 4))
    ax.yaxis.set_visible(False)
    plt.setp(ax.spines.values(), visible=False)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.set_xlabel(r"Perturbation $\epsilon$", labelpad=24)

    for i, dataset in enumerate(datasets):
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

        # init subplot
        ax = plt.subplot(1, 4, i + 1)
        ax.set_title(dataset_titles[dataset])
        for method_name in ["SINCERE", "SupCon"]:
            ax.fill_between(df[df["model_name"] == method_name]["epsilon"],
                            1,
                            where=list(~diff_sigs),
                            facecolor="lightgrey")
            ax.plot(df[df["model_name"] == method_name]["epsilon"],
                    df[df["model_name"] == method_name]["accuracy"],
                    label=method_name,)
        # add axes labels on left
        if i == 0:
            ax.set_ylabel("Accuracy")
    plt.legend()
    plt.savefig(folder / "adversarial.pdf")
    plt.close()
