from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch


if __name__ == "__main__":
    out_folders = [Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_new/"),
                   Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_old/")]
    fig_folder = Path("figures/confusion_acc")
    fig_folder.mkdir(exist_ok=True)
    # CIFAR10 labels
    class_labels = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    # calculate embedding statistics
    for out_folder in out_folders:
        print(out_folder)
        preds = torch.argmax(torch.load(out_folder / "preds.pth"), dim=1)
        labels = torch.load(out_folder / "labels.pth")
        disp = ConfusionMatrixDisplay.from_predictions(
            labels, preds, display_labels=class_labels, cmap="Blues")
        plt.title("SINCERE Loss" if "new" in out_folder.name else "SupCon Loss")
        disp.figure_.savefig(fig_folder / (out_folder.name + ".pdf"))
