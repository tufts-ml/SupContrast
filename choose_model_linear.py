import os
import csv
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from main_ce import set_loader
from networks.resnet_big import SupConResNet, LinearClassifier


class ChooseLinearModel:
    def __init__(
        self, heads_dir, sincere_encoder_ckpt, supcon_encoder_ckpt, dataset_opt, epssupinfonce_encoder_ckpt=None,
    ):
        self.heads_dir = Path(heads_dir)
        self.sincere_encoder_ckpt = Path(sincere_encoder_ckpt)
        self.supcon_encoder_ckpt = Path(supcon_encoder_ckpt)
        self.epssupinfonce_encoder_ckpt = (
            Path(epssupinfonce_encoder_ckpt) if epssupinfonce_encoder_ckpt else None
        )

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

        progress_bar = tqdm(self.head_folders, desc="Model evaluation")
        for head_folder in progress_bar:
            try:
                head_path = head_folder / "last.pth"

                if "SINCERE" in head_folder.name:
                    encoder_path = self.sincere_encoder_ckpt
                    encoder_type = "SINCERE"
                elif "EpsSupInfoNCE" in head_folder.name:
                    if not self.epssupinfonce_encoder_ckpt:
                        print(
                            f"Skipping {head_folder.name}: EpsSupInfoNCE checkpoint not provided."
                        )
                        continue
                    encoder_path = self.epssupinfonce_encoder_ckpt
                    encoder_type = "EpsSupInfoNCE"
                else:
                    encoder_path = self.supcon_encoder_ckpt
                    encoder_type = "SupCon"

                progress_bar.set_description(f"Processing {encoder_type}")

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
        encoder.load_state_dict(ckpt["model"])

        classifier = LinearClassifier(
            name=self.opt["model"], num_classes=self.opt["n_cls"], feat_dim=128
        )
        classifier_ckpt = torch.load(head_path, map_location=self.device, weights_only=False)
        classifier.load_state_dict(classifier_ckpt["model"])

        model = FullModel(encoder, classifier).to(self.device)
        model.eval()

        all_losses_tensors = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = model(images)
                losses = F.cross_entropy(output, labels, reduction='none')
                all_losses_tensors.append(losses)

        all_losses = torch.cat(all_losses_tensors)
        avg_val_loss = all_losses.mean().item()

        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = model(images)

                predicted = output.argmax(dim=1, keepdim=True)
        
                correct += (predicted.squeeze(1) == labels).sum().item()
                total += labels.size(0)

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
        "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/linear/cynthia/imagenet100/size-32/cynthia-imagenet100/25_epochs/"
    )

    # cynthia imagenet100
    sincere_encoder_checkpoint = "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/cynthia/SINCERE_imagenet100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_18/last.pth"
    supcon_encoder_checkpoint = "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/cynthia/SupCon_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_20/last.pth"
    epssupinfonce_encoder_checkpoint = "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/cynthia/EpsSupInfoNCE_imagenet100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_22-09_31_46/last.pth"

    # # cynthia cifar10
    # sincere_encoder_checkpoint = "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/cynthia/cifar10_models/SINCERE_cifar10_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_20-22_04_43/last.pth"
    # supcon_encoder_checkpoint = "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/cynthia/cifar10_models/SupCon_cifar10_resnet50_lr_0.35_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_19-15_04_54/last.pth"
    # epssupinfonce_encoder_checkpoint = "/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/cynthia/cifar10_models/EpsSupInfoNCE_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_21-12_28_30/last.pth"

    # imagenet100
    dataset_options = {
        "dataset": "imagenet100",
        "model": "resnet50",
        "size": 32,
        "batch_size": 512,
        "num_workers": 4,
        "valid_split": 0.1,
        "n_cls": 100,
    }

    # # cifar10
    # dataset_options = {
    #     "dataset": "cifar10",
    #     "model": "resnet50",
    #     "size": 32,
    #     "batch_size": 1024,
    #     "num_workers": 8,
    #     "valid_split": 0.1,
    #     "n_cls": 10,
    # }

    evaluator = ChooseLinearModel(
        heads_dir=heads_directory,
        sincere_encoder_ckpt=sincere_encoder_checkpoint,
        supcon_encoder_ckpt=supcon_encoder_checkpoint,
        dataset_opt=dataset_options,
        epssupinfonce_encoder_ckpt=epssupinfonce_encoder_checkpoint,
    )
    evaluator.process_runs()
    evaluator.save_results_to_csv()
