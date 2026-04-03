import matplotlib.pyplot as plt
import torch


if __name__ == "__main__":
    for i, acc_cache in enumerate([torch.load("cifar10c_acc.pth"),
                                   torch.load("cifar100c_acc.pth")]):
        # acc_cache starts as 3 by 5 * corruption_num, so take average over corruption types
        y = torch.reshape(acc_cache, (3, -1, 5)).mean(dim=1)
        x = [1, 2, 3, 4, 5]
        fig, ax = plt.subplots()
        plt.plot(x, y[0], label="SINCERE")
        plt.plot(x, y[1], label="SupCon")
        plt.plot(x, y[2], label="Eps-SupInfoNCE")
        fig.legend()
        if i == 0:
            ax.set_ylim(0.75, 1)
        else:
            ax.set_ylim(0.5, 0.75)
        ax.set_xlabel("Corruption Severity")
        ax.set_xticks(x)
        ax.set_ylabel("1-NN Accuracy")
        ax.set_title("CIFAR-10-C" if i == 0 else "CIFAR-100-C")
        fig.savefig(("CIFAR-10-C" if i == 0 else "CIFAR-100-C") + ".pdf")
        plt.close()
