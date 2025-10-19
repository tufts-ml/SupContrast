import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from networks.resnet_big import SupConResNet, LinearClassifier


def load_model_and_head(encoder_path, head_path, arch, num_classes):
    """
    Loads the full model pipeline: Encoder + Projection Head + Classifier Head.
    The 'embedding_model' returned is the Encoder + Projection Head.
    The 'classifier_head' is trained on the 128-dim output of the projection head.
    """
    # Load Encoder + Projection Head 
    embedding_model = SupConResNet(name=arch, head="mlp")
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    embedding_model.load_state_dict(ckpt["model"], strict=False)

    # Classifier is on top of the 128-dim projection head output
    classifier_head = LinearClassifier(
        name=arch, num_classes=num_classes, feat_dim=128
    )
    head_ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    classifier_head.load_state_dict(head_ckpt["model"])

    return embedding_model, classifier_head


def is_distance_acceptable(current_dist, target_dist, tolerance=1e-4):
    """
    Checks if the current distance is within a given absolute tolerance
    of the target distance.
    """
    return abs(current_dist - target_dist) <= tolerance


def find_perturbation_for_distance(
    z,
    grad_sign,
    target_epsilon,
    max_search_eps=10.0,
    max_iter=20,
):
    """
    Uses binary search to find the ε' for a gradient sign
    perturbation that results in a perturbed embedding z' with a Euclidean
    distance of `target_epsilon` from the original embedding z.

    Note: This assumes z is L2 normalized.
    """
    low = 0.0
    high = max_search_eps

    # Ensure z and grad_sign are 1D for calculations
    z = z.squeeze()
    grad_sign = grad_sign.squeeze()

    perturbed_z = z 

    for _ in range(max_iter):
        eps_prime = (low + high) / 2.0
        if eps_prime < 1e-8:
            perturbed_z = z
        else:
            perturbed_z_unnormalized = z + eps_prime * grad_sign
            perturbed_z = F.normalize(perturbed_z_unnormalized, p=2, dim=0)

        current_distance = torch.norm(z - perturbed_z, p=2)

        if is_distance_acceptable(current_distance, target_epsilon):
            return perturbed_z

        if current_distance < target_epsilon:
            low = eps_prime
        else:
            high = eps_prime

    return perturbed_z


def denorm(batch, mean, std, device):
    """Convert normalised tensor back to [0,1] range."""
    mean_t = torch.tensor(mean).to(device)
    std_t = torch.tensor(std).to(device)
    return batch * std_t.view(1, -1, 1, 1) + mean_t.view(1, -1, 1, 1)


def fgsm_attack(image, epsilon, data_grad):
    """Perform one-step FGSM on a denormalised image."""
    sign_grad = data_grad.sign()
    perturbed = torch.clamp(image + epsilon * sign_grad, 0, 1)
    return perturbed


def get_data_normalization(dataset_name):
    """Return (mean, std) tuple for the given dataset."""
    normalization_map = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        "imagenet100": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "dtd": ((0.5301, 0.4734, 0.4243), (0.1250, 0.1246, 0.1211)),
        "food101": ((0.5456, 0.4430, 0.3423), (0.2096, 0.2186, 0.2165)),
        "aircraft": ((0.4804, 0.5115, 0.5348), (0.1561, 0.1555, 0.1810)),
        "cars": ((0.4707, 0.4601, 0.4549), (0.2319, 0.2318, 0.2373)),
        "flowers": ((0.4312, 0.3786, 0.2944), (0.2385, 0.1858, 0.1986)),
        "pet": ((0.4312, 0.3786, 0.2944), (0.2385, 0.1858, 0.1986)),
    }
    if dataset_name in normalization_map:
        return normalization_map[dataset_name]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_base_parser(description):
    """Gets a parser with common arguments for all analysis scripts."""
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--encoder_ckpts", type=str, nargs="+", required=True, help="Path to encoder checkpoints")
    parser.add_argument("--head_ckpts", type=str, nargs="+", required=True, help="Path to linear head checkpoints")
    parser.add_argument("--labels", type=str, nargs="+", required=True, help="Labels for models, in order")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--model", choices=["resnet18", "resnet50", "resnet200"], default="resnet50")
    parser.add_argument("--subdir", type=str, default="", help="Subdirectory for saving results")
    parser.add_argument("--size", type=int, default=32, help="Image size")
    return parser


def process_opt(opt):
    """Processes shared options, sets n_cls, data_folder."""
    if not (len(opt.encoder_ckpts) == len(opt.head_ckpts) == len(opt.labels)):
        raise ValueError("The number of checkpoints and labels must be the same.")

    opt.valid_split, opt.num_workers, opt.batch_size = 0, 4, 1

    opt.data_folder = "/cluster/tufts/hugheslab/datasets/ImageNet100/train/" if opt.dataset == "imagenet100" else "./datasets/"

    n_cls_map = {
        "cifar10": 10, "cifar100": 100, "imagenet100": 100, "dtd": 47,
        "food101": 101, "pet": 37, "aircraft": 102, "cars": 196, "flowers": 102,
    }
    opt.n_cls = n_cls_map[opt.dataset]
    return opt


def save_robustness_results_to_csv(all_results, out_dir, dataset_name):
    """Save distance and epsilon data to a CSV file."""
    save_path = out_dir / f"results_distance_vs_epsilon_{dataset_name}.csv"
    header = ["model", "distance", "epsilon"]
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for model_name, data in all_results.items():
            for i in range(len(data["distances"])):
                writer.writerow([model_name, data["distances"][i], data["epsilons"][i]])
    print(f"Saved robustness results to {save_path}")


def save_accuracy_results_to_csv(epsilons, accuracies_dict, out_dir, dataset_name):
    """Save accuracy and epsilon data to a CSV file."""
    save_path = out_dir / f"accuracy_vs_epsilon_{dataset_name}.csv"
    header = ["model", "epsilon", "accuracy"]
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for model_name, accuracies in accuracies_dict.items():
            for i, acc in enumerate(accuracies):
                writer.writerow([model_name, epsilons[i], acc])
    print(f"Saved accuracy results to {save_path}")


def plot_accuracy_vs_epsilon(epsilons, accuracies_dict, out_dir, title):
    """Plot accuracy vs epsilon for multiple models and save figure."""
    pass
