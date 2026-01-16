import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse

from main_ce import set_loader
from analysis_utils import (
    load_model_and_head,
    find_perturbation_for_distance,
    calculate_distance_to_hyperplane,
)


def parse_option():
    parser = argparse.ArgumentParser(
        description="Analyze robustness: Hyperplane Distance vs. Min Flipping Epsilon."
    )

    # Dataset and model architecture
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

    # Analysis options
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max number of test images to analyze per model",
    )

    # Model checkpoints
    parser.add_argument(
        "--loss_names",
        type=str,
        nargs="+",
        required=True,
        help="List of names for the models (e.g. SupCon, SINCERE).",
    )
    parser.add_argument(
        "--encoder_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to encoder checkpoints.",
    )
    parser.add_argument(
        "--head_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to linear head checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_robustness/",
        help="Directory to save results.",
    )

    opt = parser.parse_args()

    # Validation
    if not (len(opt.loss_names) == len(opt.encoder_paths) == len(opt.head_paths)):
        raise ValueError(
            "The number of loss_names, encoder_paths, and head_paths must match."
        )

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

    opt.batch_size = 1
    opt.valid_split = 0.1

    return opt


def find_min_flipping_epsilon(
    embedding_model, classifier_head, image, true_label, original_pred
):
    """
    Uses binary search to find the smallest epsilon (Euclidean distance on hypersphere)
    that flips the prediction of the classifier.
    """
    low, high = 0.0, 2.0  # Max distance on unit sphere is 2.0
    min_eps = float("inf")

    # Get original embedding and gradient
    with torch.no_grad():
        embed = embedding_model(image).squeeze(0)

    # Calculate gradient for direction
    embed_for_attack = embed.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(classifier_head(embed_for_attack.unsqueeze(0)), true_label)
    loss.backward()
    grad = embed_for_attack.grad

    if grad is None:
        return None

    grad_sign = grad.sign()

    # Binary search for 20 iterations
    for _ in range(20):
        eps = (low + high) / 2.0

        # Get perturbed embedding at exactly distance `eps`
        perturbed_z = find_perturbation_for_distance(embed, grad_sign, eps)
        perturbed_embed = perturbed_z.unsqueeze(0)

        with torch.no_grad():
            current_pred = classifier_head(perturbed_embed).argmax(dim=1)

        if current_pred.item() != original_pred.item():
            min_eps = eps
            high = eps
        else:
            low = eps

    return min_eps if min_eps != float("inf") else None


def main():
    opt = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(opt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {opt.dataset}")
    _, _, test_loader = set_loader(opt, contrast_trans=False, for_test=True)

    all_results = []

    for i, model_name in enumerate(opt.loss_names):
        print(f"\n---> Processing Model: {model_name}")

        encoder_path = opt.encoder_paths[i]
        head_path = opt.head_paths[i]

        embedding_model, classifier_head = load_model_and_head(
            encoder_path, head_path, opt.model_arch, opt.n_cls
        )
        embedding_model.eval().to(device)
        classifier_head.eval().to(device)

        # Pre-fetch weights for distance calculation
        W = classifier_head.fc.weight.data

        # Limit the loop for speed if requested
        total_images = min(opt.limit, len(test_loader))
        progress_bar = tqdm(
            enumerate(test_loader), total=total_images, desc=f"Analyzing {model_name}"
        )

        for j, (image, target) in progress_bar:
            if j >= opt.limit:
                break
            image, target = image.to(device), target.to(device)

            with torch.no_grad():
                features = embedding_model(image)
                logits = classifier_head(features).squeeze()
                pred = torch.argmax(logits)

            # 1. Calculate Distance
            distance = calculate_distance_to_hyperplane(logits, W, target.item())

            # 2. Calculate Robustness (Min Epsilon)
            # Only compute epsilon if prediction is correct, otherwise epsilon is effectively 0
            if pred.item() == target.item():
                min_eps = find_min_flipping_epsilon(
                    embedding_model, classifier_head, image, target, pred
                )
                epsilon_val = min_eps if min_eps is not None else -1.0
            else:
                epsilon_val = 0.0

            all_results.append(
                {"model": model_name, "distance": distance, "epsilon": epsilon_val}
            )

        # Cleanup to save memory between models
        del embedding_model, classifier_head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_results:
        print("\nNo results generated.")
        return

    # Save results
    output_file = out_dir / f"robustness_distance_vs_epsilon_{opt.dataset}.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"\nRobustness analysis saved to {output_file}")


if __name__ == "__main__":
    main()
