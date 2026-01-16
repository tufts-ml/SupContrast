import torch
import torch.nn.functional as F
from networks.resnet_big import SupConResNet, LinearClassifier


def load_model_and_head(encoder_path, head_path, arch, num_classes):
    """
    Loads the full model pipeline: Encoder + Projection Head + Classifier Head.
    
    Returns:
        embedding_model: SupConResNet (Encoder + MLP head)
        classifier_head: LinearClassifier
    """
    # Load Encoder + Projection Head 
    # The attack assumes we are attacking the output of the projection head (z)
    embedding_model = SupConResNet(name=arch, head="mlp", feat_dim=128)
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    embedding_model.load_state_dict(ckpt["model"], strict=True)

    # Classifier is on top of the 128-dim projection head output
    classifier_head = LinearClassifier(
        name=arch, num_classes=num_classes, feat_dim=128
    )
    head_ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    classifier_head.load_state_dict(head_ckpt["model"], strict=True)

    return embedding_model, classifier_head


def calculate_distance_to_hyperplane(logits, W, true_label_idx):
    """
    Calculate the geometric distance from an embedding to the decision boundary.
    
    Args:
        logits: Output scores from the classifier.
        W: Weight matrix of the linear classifier.
        true_label_idx: The index of the ground truth label.
        
    Returns:
        float: The distance to the decision boundary.
    """
    scores, indices = torch.topk(logits, 2)
    pred_idx = indices[0]

    if pred_idx == true_label_idx:
        # Correct prediction: distance to runner-up class boundary (positive margin)
        # Boundary defined by w_correct - w_runner
        s_correct, s_runner = scores[0], scores[1]
        w_correct, w_runner = W[indices[0], :], W[indices[1], :]
        numerator = s_correct - s_runner
        denominator = torch.norm(w_correct - w_runner)
    else:
        # Incorrect prediction: distance to predicted class boundary (negative margin)
        # Boundary defined by w_true - w_pred
        s_pred = logits[pred_idx]
        w_pred = W[pred_idx, :]
        s_true = logits[true_label_idx]
        w_true = W[true_label_idx, :]
        
        # We want distance to cross back to the truth
        numerator = s_true - s_pred
        denominator = torch.norm(w_true - w_pred)

    return (numerator / denominator).item() if denominator > 1e-9 else 0.0


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
    Uses binary search to find the scalar weight for a gradient sign
    perturbation that results in a perturbed embedding z' with a Euclidean
    distance of `target_epsilon` from the original embedding z.

    Args:
        z: Original embedding (assumed L2 normalized).
        grad_sign: Sign of the gradient.
        target_epsilon: The desired Euclidean distance ||z - z'||.
    
    Returns:
        perturbed_z: The perturbed embedding.
    """
    low = 0.0
    high = max_search_eps

    # Ensure z and grad_sign are 1D for calculations (batch_size=1)
    z = z.squeeze()
    grad_sign = grad_sign.squeeze()

    perturbed_z = z 

    for _ in range(max_iter):
        eps_prime = (low + high) / 2.0
        
        # Avoid division by zero or no-op
        if eps_prime < 1e-8:
            perturbed_z = z
        else:
            # Add perturbation and re-normalize to stay on hypersphere
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