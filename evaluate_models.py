import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse

from main_ce import set_loader
from contrast_acc import test_contrastive_acc

from evaluate_utils import (
    load_supcon_model,
    compute_features,
    calculate_target_noise_separation,
    calculate_target_noise_separation_topk,
)


def parse_option():
    parser = argparse.ArgumentParser(
        description="Evaluate contrastive models from a parent directory."
    )

    # Dataset and model architecture
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for dataloaders"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset"
    )
    parser.add_argument("--size", type=int, default=32, help="Image size")

    parser.add_argument(
        "--model_arch",
        choices=["resnet18", "resnet50", "resnet200"],
        default="resnet50",
    )

    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers for dataloaders"
    )

    # Arguments specific to this script
    parser.add_argument(
        "--parent_dir",
        type=str,
        required=True,
        help="Path to the parent directory containing model run subdirectories.",
    )
    parser.add_argument(
        "--last_epoch",
        type=int,
        required=True,
        help="The epoch number corresponding to 'last.pth' for logging purposes.",
    )
    parser.add_argument(
        "--separation_percentiles",
        type=float,
        nargs="+",
        default=[0.99],
        help="List of percentiles (0.0 to 1.0) for target-noise separation.",
    )
    parser.add_argument(
        "--separation_topk",
        type=int,
        nargs="+",
        default=[1],
        help="List of k values (e.g, 1 5) for k-th value target-noise separation.",
    )

    opt = parser.parse_args()

    opt.data_folder = (
        "/cluster/tufts/hugheslab/datasets/ImageNet100/train/"
        if opt.dataset == "imagenet100"
        else "./datasets/"
    )

    n_cls_map = {
        "cifar2": 2,
        "cifar10": 10,
        "cifar100": 100,
        "imagenet100": 100,
        "dtd": 47,
        "food101": 101,
        "pet": 37,
        "aircraft": 102,
        "cars": 196,
        "flowers": 102,
    }
    if opt.dataset not in n_cls_map:
        raise ValueError(f"Dataset '{opt.dataset}' not recognized in n_cls_map.")
    opt.n_cls = n_cls_map[opt.dataset]

    # Set valid_split for set_loader
    opt.valid_split = 0.1

    return opt


def compute_metrics(
    opt, train_embeds, train_labels, eval_embeds, eval_labels, prefix, device
):
    """
    Helper function to compute all metrics (kNN, Separation) for a specific split.
    """
    results = {}

    # Metric 1: 1-NN Accuracy
    knn_acc_1 = test_contrastive_acc(
        train_embeds.to(device),
        eval_embeds.to(device),
        train_labels.to(device),
        eval_labels.to(device),
    )
    tqdm.write(f"  {prefix.capitalize()} 1-NN Accuracy            \t: {knn_acc_1:.4f}")
    results[f"{prefix}_knn_acc_1"] = knn_acc_1.item()

    # Metric 2: Target-Noise Separation (percentile)
    for p in sorted(opt.separation_percentiles):
        percentile_key_str = f"p{p * 100:g}"
        separation_val = calculate_target_noise_separation(
            train_embeds, train_labels, eval_embeds, eval_labels, percentile=p
        )
        tqdm.write(
            f"  {prefix.capitalize()} Separation ({percentile_key_str}) \t: {separation_val:.4f}"
        )
        results[f"{prefix}_separation_{percentile_key_str}"] = separation_val

    # Metric 3: Target-Noise Separation (k-th Top Value)
    for k in sorted(opt.separation_topk):
        topk_key_str = f"t{k}"
        separation_val = calculate_target_noise_separation_topk(
            train_embeds, train_labels, eval_embeds, eval_labels, k=k
        )
        tqdm.write(
            f"  {prefix.capitalize()} Separation ({topk_key_str})      \t: {separation_val:.4f}"
        )
        results[f"{prefix}_separation_{topk_key_str}"] = separation_val

    return results


def main():
    opt = parse_option()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset: {opt.dataset}")
    train_loader, val_loader, test_loader = set_loader(
        opt,
        contrast_trans=True,
        for_test=True,
    )

    parent_dir = Path(opt.parent_dir)
    model_subdirs = sorted(
        d for d in parent_dir.iterdir() if d.is_dir() and (d / "last.pth").exists()
    )

    if not model_subdirs:
        print(f"No model subdirectories found in '{parent_dir}'. Exiting.")
        return

    print(f"\nFound {len(model_subdirs)} model directories to process.")
    all_results = []

    # Iterate and Evaluate Each Model
    for model_dir in tqdm(model_subdirs, desc="Total Progress"):
        tqdm.write(f"\nProcessing: {model_dir.name}")

        ckpt_path = model_dir / "last.pth"

        model = load_supcon_model(ckpt_path, opt.model_arch)
        model.to(device)

        # 1. Compute Train Features (Support Set)
        train_embeds, train_labels = compute_features(
            model, train_loader, "train", device
        )

        # 2. Compute Validation Features
        val_embeds, val_labels = compute_features(
            model, val_loader, "validation", device
        )

        # 3. Compute Test Features
        test_embeds, test_labels = compute_features(model, test_loader, "test", device)

        result_data = {
            "model_name": model_dir.name,
            "epoch": opt.last_epoch,
        }

        val_results = compute_metrics(
            opt, train_embeds, train_labels, val_embeds, val_labels, "val", device
        )
        result_data.update(val_results)

        test_results = compute_metrics(
            opt, train_embeds, train_labels, test_embeds, test_labels, "test", device
        )
        result_data.update(test_results)

        all_results.append(result_data)

        # Cleanup to free memory
        del (
            model,
            train_embeds,
            train_labels,
            val_embeds,
            val_labels,
            test_embeds,
            test_labels,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_results:
        print("\nEvaluation complete, but no results were generated.")
        return

    output_file = parent_dir / "evaluate_models_results.csv"

    final_df = pd.DataFrame(all_results).sort_values(by="model_name")
    final_df.to_csv(output_file, index=False, float_format="%.5f")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
