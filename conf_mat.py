from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay
import torch


if __name__ == "__main__":
    out_folders = [Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_old/")]
    fig_folder = Path("figures/confusion_acc")
    fig_folder.mkdir(exist_ok=True)
    # CIFAR10 labels
    class_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # calculate embedding statistics
    for out_folder in out_folders:
        print(out_folder)
        preds = torch.argmax(torch.load(out_folder / "preds.pth"), dim=1)
        labels = torch.load(out_folder / "labels.pth")
        ConfusionMatrixDisplay.from_predictions(labels, preds, display_labels=class_labels)\
            .figure_.savefig(fig_folder / (out_folder.name + ".png"))