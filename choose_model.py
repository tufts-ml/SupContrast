import os
from pathlib import Path

import torch

from bootstrap_knn_acc_fp16_jit import test_contrastive_pred_knn, accuracy
from bootstrap_lin_acc import bootstrap_metric
from embed_stats import make_test_pred_dict

import csv

from tbparse import SummaryReader


class choose_model:
    def __init__(self, k, save_dir, save_tb_dir):
        self.save_dir = Path(save_dir)
        self.save_tb_dir = Path(save_tb_dir)
        self.k = k

        self.filenames = {"SINCERE": [], "SupCon": []}
        for filename in os.listdir(self.save_dir):
            
            # incomplete run
            filename_saves = set(os.listdir(self.save_dir / filename))
            if "last.pth" not in filename_saves:
                continue

            if "SINCERE" in filename:
                self.filenames["SINCERE"].append(self.save_dir / filename)
            elif "SupCon" in filename:
                self.filenames["SupCon"].append(self.save_dir / filename)

        self.results = []

        self.read_tensorboards()

    def read_tensorboards(self):
        self.tensorboards = []
        for tensorboard in os.listdir(self.save_tb_dir):
            self.tensorboards.append(self.save_tb_dir / tensorboard)

        # this is going to have more keys than self.filenames
        # because failed run still save into this dict
        self.tensorboards_detail = {}
        for tensorboard in self.tensorboards:
            reader = SummaryReader(tensorboard)
            df = reader.scalars

            if "SINCERE" in tensorboard.name:
                loss_tag = "valid/SINCERE"
            else:
                loss_tag = "valid/SupCon"

            acc_tag = "valid/Top 1 Accuracy"

            # failed run
            if len(df) == 0:
                continue

            loss_df = df[df["tag"] == loss_tag]
            loss_best_row_idx = loss_df["value"].idxmin()
            loss_best_row = loss_df.loc[loss_best_row_idx]
            loss_last_row = loss_df.iloc[-1]

            acc_df = df[df["tag"] == acc_tag]
            acc_best_row_idx = acc_df["value"].idxmax()
            acc_best_row = acc_df.loc[acc_best_row_idx]
            acc_last_row = acc_df.iloc[-1]

            self.tensorboards_detail[tensorboard.name] = {
                "best loss (step, value)": (
                    loss_best_row["step"].item(),
                    loss_best_row["value"].item(),
                ),
                "last loss (step, value)": (
                    loss_last_row["step"].item(),
                    loss_last_row["value"].item(),
                ),
                "best accuracy (step, value)": (
                    acc_best_row["step"].item(),
                    acc_best_row["value"].item(),
                ),
                "last accuracy (step, value)": (
                    acc_last_row["step"].item(),
                    acc_last_row["value"].item(),
                ),
            }

    def process_result(self, loss_name):
        # loss_name = "SINCER" or "SupCon"

        for filename in self.filenames[loss_name]:
            try:
                # accuracy on test set
                test_metric_mean = self.__bootstrap_test_acc(filename)

                # mts = mean target similarity, mns = mean noise similarity
                mts, mns = self.__margin_separation(filename)

                lr, decay = self.__get_run_details(filename)

                self.results.append(
                    {
                        "loss": loss_name,
                        "filename": filename,
                        "lr": lr,
                        "decay": decay,
                        "k": int(self.k),
                        "best loss (step, value)": self.tensorboards_detail[
                            filename.name
                        ]["best loss (step, value)"],
                        "last loss (step, value)": self.tensorboards_detail[
                            filename.name
                        ]["last loss (step, value)"],
                        "best accuracy (step, value)": self.tensorboards_detail[
                            filename.name
                        ]["best accuracy (step, value)"],
                        "last accuracy (step, value)": self.tensorboards_detail[
                            filename.name
                        ]["last accuracy (step, value)"],
                        "test accuracy": test_metric_mean.item(),
                        "median target similarity": mts.item(),
                        "median noise similarity": mns.item(),
                        "margin of separation": abs(mts.item() - mns.item()),
                    }
                )

            except Exception as e:
                print(e)

    def __bootstrap_test_acc(self, filename):
        y_pred = test_contrastive_pred_knn(
            torch.load(filename / "train_embeds.pth", weights_only=False),
            torch.load(filename / "test_embeds.pth", weights_only=False),
            torch.load(filename / "train_labels.pth", weights_only=False),
            torch.load(filename / "test_labels.pth", weights_only=False),
            self.k,
        )

        y_true = torch.load(filename / "test_labels.pth", weights_only=False)

        metric_mean, _, _, _ = bootstrap_metric(y_pred, y_true, accuracy)

        return metric_mean

    def __margin_separation(self, filename):
        if not (filename / "test_pred_dict.pth").exists():
            make_test_pred_dict(filename)
        test_pred_dict = torch.load(filename / "test_pred_dict.pth", weights_only=False)

        target = torch.median(test_pred_dict["target_sim"])
        noise = torch.median(test_pred_dict["noise_sim"])

        return target, noise

    def __get_run_details(self, filename):
        filename = filename.name
        filename_part = filename.split("_")

        lr, decay = None, None

        if "lr" in filename_part:
            index_lr = filename_part.index("lr")
            lr = float(filename_part[index_lr + 1])
        if "decay" in filename_part:
            index_decay = filename_part.index("decay")
            decay = float(filename_part[index_decay + 1])

        return lr, decay

    def save_dict_list_to_csv(self, output_dir=None, filename=None):
        if not output_dir:
            output_dir = self.save_dir
        else:
            output_dir = -Path(output_dir)

        if not filename:
            filename = "choose_model.csv"

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)

        fieldnames = self.results[0].keys()

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)

            print(f"Successfully saved data to '{file_path}'")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    # choose = choose_model(
    #     k=1,
    #     save_dir="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp4.3/resnet-18-cifar10-32x32-batch-100/cifar10_models/",
    #     save_tb_dir="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp4.3/resnet-18-cifar10-32x32-batch-100/cifar10_tensorboard/",
    # )

    choose = choose_model(
        k=1,
        save_dir="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-5000/imagenet100_models/",
        save_tb_dir="/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/exp2/resnet-18-imagenet-100-32x32-batch-5000/imagenet100_tensorboard/"
    )

    choose.process_result("SINCERE")
    choose.process_result("SupCon")

    choose.save_dict_list_to_csv()
