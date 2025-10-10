import torch
import torch.nn.functional as F

import csv

import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm

from main_ce import set_loader
from util_eval import (
    load_model_and_head,
    find_perturbation_for_distance,
)


def calculate_distance_to_hyperplane(logits, W, true_label_idx):
    """
    Calculate the distance to the decision boundary.
    - For a correctly predicted sample, this is the distance between the true class (k)
      and the runner-up class (j). The result is positive.
    - For a misclassified sample, this is the distance between the true class (k)
      and the predicted (incorrect) class (p). The result is negative.
    """
    scores, indices = torch.topk(logits, 2)
    pred_idx = indices[0]

    if pred_idx == true_label_idx:
        s_k, s_j = scores[0], scores[1]
        k_idx, j_idx = indices[0], indices[1]
        w_k, w_j = W[k_idx, :], W[j_idx, :]

        numerator = s_k - s_j
        denominator = torch.norm(w_k - w_j)
    else:
        s_p = logits[pred_idx]
        w_p = W[pred_idx, :]

        s_k_true = logits[true_label_idx]
        w_k_true = W[true_label_idx, :]

        numerator = s_k_true - s_p
        denominator = torch.norm(w_k_true - w_p)

    if denominator > 1e-9:
        distance = (numerator / denominator).item()
        return distance
    else:
        return 0


def find_min_flipping_epsilon(
    embedding_model,
    classifier_head,
    image,
    true_label,
    original_pred,
    max_epsilon=2.0,
    max_iter=20,
):
    """
    Uses binary search to find the smallest epsilon that causes a prediction to flip.
    """
    low = 0.0
    high = max_epsilon
    min_eps = float("inf")

    # Original embedding
    with torch.no_grad():
        embed = embedding_model(image).squeeze(0)

    # To get gradient for the attack direction
    embed_for_attack = embed.clone().detach().requires_grad_(True)
    logit = classifier_head(embed_for_attack.unsqueeze(0))
    loss = F.cross_entropy(logit, true_label)
    loss.backward()
    grad = embed_for_attack.grad

    if grad is None:
        return None

    grad_sign = grad.sign()

    # Binary search for the flipping epsilon
    for _ in range(max_iter):
        eps = (low + high) / 2.0

        # If epsilon is effectively zero, no perturbation
        if eps < 1e-8:
            perturbed_embed = embed.unsqueeze(0)
        else:
            # This inner search finds the perturbation scale for a given distance eps
            perturbed_z = find_perturbation_for_distance(
                z=embed,
                grad_sign=grad_sign,
                target_epsilon=eps,
            )
            perturbed_embed = perturbed_z.unsqueeze(0)

        with torch.no_grad():
            output = classifier_head(perturbed_embed)
            current_pred = output.argmax(dim=1)

        if current_pred.item() != original_pred.item():
            # Prediction flipped, this epsilon is a potential minimum.
            # Try for an even smaller one.
            min_eps = eps
            high = eps
        else:
            # Prediction did not flip, need a larger epsilon.
            low = eps

    if min_eps == float("inf"):
        return (
            None  # Did not find an epsilon that flips the prediction in the given range
        )

    return min_eps


def parse_option():

    parser = argparse.ArgumentParser(
        "Robustness Analysis: Hyperplane Distance vs. Flipping Epsilon"
    )

    # Required
    parser.add_argument(
        "--encoder_ckpts",
        type=str,
        nargs="+",
        required=True,
        help="Path to one or more encoder checkpoints",
    )
    parser.add_argument(
        "--head_ckpts",
        type=str,
        nargs="+",
        required=True,
        help="Path to one or more linear head checkpoints",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="Label for the model, in the same order as checkpoints.",
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100", "imagenet100"],
        required=True,
    )
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument(
        "--subdir",
        type=str,
        default="",
        help="Subdirectory name for saving figures",
    )

    # Can default
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Limit the number of test images",
    )

    parser.add_argument(
        "--use-projection-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default assume using: encoder + projection head + linear classification head (distance + epsilon). Use --no-use-projection-head for encoder + linear classifcation head (distance only).",
    )

    opt = parser.parse_args()

    if not (len(opt.encoder_ckpts) == len(opt.head_ckpts) == len(opt.labels)):
        raise ValueError("The number of checkpoints and labels must be the same.")

    opt.valid_split = 0
    opt.num_workers = 4

    # Process one image at a time
    opt.batch_size = 1

    if opt.dataset == "imagenet100":
        opt.data_folder = "/cluster/tufts/hugheslab/datasets/ImageNet100/train/"
    else:
        opt.data_folder = "./datasets/"

    opt.n_cls = {"cifar10": 10, "cifar100": 100, "imagenet100": 100}[opt.dataset]

    return opt


def save_results_to_csv(all_results, out_dir, dataset_name):
    """Save the collected distance and epsilon data to a CSV file."""

    save_path = out_dir / f"results_distance_vs_epsilon_{dataset_name}.csv"

    sample_result = next(iter(all_results.values()))
    if "epsilons" in sample_result:
        header = ["model", "distance", "epsilon"]
        is_full_analysis = True
    else:
        header = ["model", "distance"]
        is_full_analysis = False

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for model_name, data in all_results.items():
            distances = data["distances"]
            if is_full_analysis:
                epsilons = data["epsilons"]
                for i in range(len(distances)):
                    writer.writerow([model_name, distances[i], epsilons[i]])
            else:
                for i in range(len(distances)):
                    writer.writerow([model_name, distances[i]])

    print(f"Saved results to {save_path}")


def generate_plot(all_results, out_dir, dataset_name):
    pass


def main():
    opt = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_out_dir = Path("large-margin/figures/robustness")
    out_dir = base_out_dir / opt.subdir if opt.subdir else base_out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results dict based on the mode
    all_results = {}
    for name in opt.labels:
        if opt.use_projection_head:
            all_results[name] = {"distances": [], "epsilons": []}
        else:
            all_results[name] = {"distances": []}

    # Load data
    _, _, test_loader = set_loader(opt, contrast_trans=False)

    for i, name in enumerate(opt.labels):
        print(f"\n---> Processing Model: {name}")

        # Load model for the current iteration
        embedding_model, classifier_head = load_model_and_head(
            opt.encoder_ckpts[i], opt.head_ckpts[i], opt.model, opt.n_cls, use_projection_head=opt.use_projection_head
        )
        embedding_model.eval().to(device)
        classifier_head.eval().to(device)
        W = classifier_head.fc.weight.data

        progress_bar = tqdm(
            enumerate(test_loader),
            total=min(opt.limit, len(test_loader)),
            desc=f"Analyzing {name}",
        )

        for j, (image, target) in progress_bar:
            if j >= opt.limit:
                break

            image, target = image.to(device), target.to(device)

            with torch.no_grad():
   
                if opt.use_projection_head:
                    features = embedding_model(image)
                else:
                    features = embedding_model.encoder(image)
                    
                logits = classifier_head(features).squeeze()
                pred = torch.argmax(logits)

            distance = calculate_distance_to_hyperplane(logits, W, target.item())
            all_results[name]["distances"].append(distance)
            
            # ONLY calculate epsilon if in projection head mode
            if opt.use_projection_head:
                if pred.item() == target.item():

                    if distance > 0:
                        min_eps = find_min_flipping_epsilon(
                            embedding_model,
                            classifier_head,
                            image,
                            target,
                            pred,
                        )

                        if min_eps is not None:
                            all_results[name]["epsilons"].append(min_eps)
                        else:
                            all_results[name]["epsilons"].append(-1.0)
                            print("  * main(), find_min_flipping_epsilon(), min_eps is None")

                    # Cannot be true 
                    else:
                        all_results[name]["epsilons"].append(-1.0)
                        print("  * main(), Correct prediction with negative distance")

                else:
                    all_results[name]["epsilons"].append(0.0)

    # generate_plot(all_results, out_dir, opt.dataset)
    save_results_to_csv(all_results, out_dir, opt.dataset)


if __name__ == "__main__":
    main()
