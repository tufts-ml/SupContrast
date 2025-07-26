import os
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from main_ce import set_loader
from networks.resnet_big import SupConResNet, LinearClassifier


class ChooseLinearModel:
    def __init__(
        self, heads_dir, sincere_encoder_ckpt, supcon_encoder_ckpt, dataset_opt
    ):
        self.heads_dir = Path(heads_dir)
        self.sincere_encoder_ckpt = Path(sincere_encoder_ckpt)
        self.supcon_encoder_ckpt = Path(supcon_encoder_ckpt)
        self.opt = dataset_opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.opt["dataset"] == "imagenet100":
            self.opt["data_folder"] = (
                "/cluster/tufts/hugheslab/datasets/ImageNet100/train/"
            )
        elif self.opt["dataset"] == "imagenet":
            self.opt["data_folder"] = (
                "/cluster/tufts/hugheslab/datasets/ImageNet/train/"
            )
        else:
            self.opt["data_folder"] = "./datasets/"

        self.head_folders = []
        for folder_name in os.listdir(self.heads_dir):
            full_path = self.heads_dir / folder_name
            if full_path.is_dir() and (full_path / "last.pth").exists():
                self.head_folders.append(full_path)

        self.results = []

        _, self.val_loader, self.test_loader = self.__load_data()

    def __load_data(self):
        class Options:
            def __init__(self, opt_dict):
                self.__dict__.update(opt_dict)
            def __contains__(self, key):
                return key in self.__dict__

        opt_obj = Options(self.opt)
        _, val_loader, test_loader = set_loader(
            opt_obj, contrast_trans=False, for_cache=True
        )
        return _, val_loader, test_loader

    def process_runs(self):
        if not self.head_folders:
            print("No valid head folders found.")
            return

        for head_folder in self.head_folders:
            try:
                head_path = head_folder / "last.pth"

                if "SINCERE" in head_folder.name.upper():
                    encoder_path = self.sincere_encoder_ckpt
                    encoder_type = "SINCERE"
                else:
                    encoder_path = self.supcon_encoder_ckpt
                    encoder_type = "SupCon"

                lr, decay = self.__get_run_details(head_folder.name)

                val_loss, test_acc = self.__evaluate_model(encoder_path, head_path)

                self.results.append(
                    {
                        "filename": head_folder,
                        "encoder_type": encoder_type,
                        "lr": lr,
                        "decay": decay,
                        "val_loss": val_loss,
                        "test_accuracy": test_acc,
                    }
                )
            except Exception as e:
                print(f"Could not process {head_folder.name}. Error: {e}")

    def __evaluate_model(self, encoder_path, head_path):
        class FullModel(nn.Module):
            def __init__(self, enc, head):
                super().__init__()
                self.enc, self.head = enc, head

            def forward(self, x):
                return self.head(self.enc(x))

        encoder = SupConResNet(name=self.opt["model"])
        ckpt = torch.load(encoder_path, map_location=self.device, weights_only=False)
        sd = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        encoder.load_state_dict(sd, strict=False)

        classifier = LinearClassifier(
            name=self.opt["model"], num_classes=self.opt["n_cls"]
        )
        classifier_ckpt = torch.load(head_path, map_location=self.device, weights_only=False)
        classifier.load_state_dict(classifier_ckpt["model"])

        model = FullModel(encoder.encoder, classifier).to(self.device)
        model.eval()

        total_loss, total_samples = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                bsz = labels.shape[0]
                output = model(images)
                loss = F.cross_entropy(output, labels)
                total_loss += loss.item() * bsz
                total_samples += bsz
        avg_val_loss = total_loss / total_samples

        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = correct / total

        return avg_val_loss, test_accuracy

    def __get_run_details(self, folder_name):
        parts = folder_name.split("_")
        lr, decay = "N/A", "N/A"
        if "lr" in parts:
            try:
                lr = float(parts[parts.index("lr") + 1])
            except (ValueError, IndexError):
                pass
        if "decay" in parts:
            try:
                decay = float(parts[parts.index("decay") + 1])
            except (ValueError, IndexError):
                pass
        return lr, decay

    def save_results_to_csv(self, filename="choose_model_linear_results.csv"):
        if not self.results:
            print("No results to save.")
            return

        output_path = self.heads_dir / filename
        fieldnames = self.results[0].keys()

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"Successfully saved evaluation results to '{output_path}'")
        except IOError as e:
            print(f"Error saving CSV file: {e}")


if __name__ == "__main__":
    heads_directory = (
        "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/exp5/cifar10/size-32/resnet-18-cifar10-32x32-batch-1000/"
    )

    sincere_encoder_checkpoint = "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp5/resnet-18-cifar10-32x32-batch-1000/cifar10_models/SINCERE_cifar10_resnet18_lr_0.2859301839839105_decay_0.001_bsz_1000_temp_0.1_trial_0_cosine_warm_2025_07_24-02_32_48/last.pth"
    supcon_encoder_checkpoint = "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp5/resnet-18-cifar10-32x32-batch-1000/cifar10_models/SupCon_cifar10_resnet18_lr_0.2859301839839105_decay_0.001_bsz_1000_temp_0.1_trial_0_cosine_warm_2025_07_24-02_38_26/last.pth"

    dataset_options = {
        "dataset": "cifar10",
        "model": "resnet18",
        "size": 32,
        "batch_size": 512,
        "num_workers": 2,
        "valid_split": 0.1,
        "n_cls": 10,
    }

    evaluator = ChooseLinearModel(
        heads_dir=heads_directory,
        sincere_encoder_ckpt=sincere_encoder_checkpoint,
        supcon_encoder_ckpt=supcon_encoder_checkpoint,
        dataset_opt=dataset_options,
    )
    evaluator.process_runs()
    evaluator.save_results_to_csv()
