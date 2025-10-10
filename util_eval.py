import torch
import torch.nn.functional as F

from networks.resnet_big import SupConResNet, LinearClassifier


def load_model_and_head(encoder_path, head_path, arch, num_classes, use_projection_head=True):
    """Load a SupCon model and a linear classifier head."""
    embedding_model = SupConResNet(name=arch, head="mlp")
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    embedding_model.load_state_dict(ckpt["model"], strict=False)

    if use_projection_head:
        classifier_head = LinearClassifier(
            name=arch, num_classes=num_classes, feat_dim=128
        )
    else:
        classifier_head = LinearClassifier(name=arch, num_classes=num_classes)
        
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
    Uses binary search to find the scaling factor ε' for a gradient sign
    perturbation that results in a perturbed embedding z' with a Euclidean
    distance of `target_epsilon` from the original embedding z.
    """
    low = 0.0
    high = max_search_eps

    for _ in range(max_iter):
        eps_prime = (low + high) / 2.0
        if eps_prime == 0:
            perturbed_z = z
        else:
            perturbed_z_unnormalized = z + eps_prime * grad_sign
            perturbed_z = F.normalize(perturbed_z_unnormalized, p=2, dim=0)

        current_distance = torch.norm(z - perturbed_z, p=2)

        if is_distance_acceptable(
            current_distance, target_epsilon
        ):
            return perturbed_z

        if current_distance < target_epsilon:
            # Distance is too small, need a larger perturbation
            low = eps_prime
        else:
            # Distance is too large, need a smaller perturbation
            high = eps_prime

    # If search finishes, return the last computed perturbation
    return perturbed_z