import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from main_ce import set_loader
from analysis_util import (
    load_model_and_head,
    find_perturbation_for_distance,
    get_base_parser,
    process_opt,
    save_robustness_results_to_csv,
)


def calculate_distance_to_hyperplane(logits, W, true_label_idx):
    """Calculate the distance from an embedding to the decision boundary."""
    scores, indices = torch.topk(logits, 2)
    pred_idx = indices[0]

    if pred_idx == true_label_idx:
        # Correct prediction: distance to runner-up class (positive)
        s_k, s_j = scores[0], scores[1]
        w_k, w_j = W[indices[0], :], W[indices[1], :]
        numerator = s_k - s_j
        denominator = torch.norm(w_k - w_j)
    else:
        # Incorrect prediction: distance to predicted class (negative)
        s_p = logits[pred_idx]
        w_p = W[pred_idx, :]
        s_k_true = logits[true_label_idx]
        w_k_true = W[true_label_idx, :]
        numerator = s_k_true - s_p
        denominator = torch.norm(w_k_true - w_p)

    return (numerator / denominator).item() if denominator > 1e-9 else 0


def find_min_flipping_epsilon(
    embedding_model, classifier_head, image, true_label, original_pred
):
    """Uses binary search to find the smallest epsilon that flips a prediction."""
    low, high, min_eps = 0.0, 2.0, float("inf")

    with torch.no_grad():
        embed = embedding_model(image).squeeze(0)

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
        if eps < 1e-8:
            perturbed_embed = embed.unsqueeze(0)
        else:
            perturbed_z = find_perturbation_for_distance(embed, grad_sign, eps)
            perturbed_embed = perturbed_z.unsqueeze(0)

        with torch.no_grad():
            current_pred = classifier_head(perturbed_embed).argmax(dim=1)

        if current_pred.item() != original_pred.item():
            min_eps, high = eps, eps
        else:
            low = eps

    return min_eps if min_eps != float("inf") else None


def parse_option():
    parser = get_base_parser("Robustness Analysis: Hyperplane Distance vs. Flipping Epsilon")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of test images")
    return process_opt(parser.parse_args())


def main():
    opt = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path("large-margin/figures/robustness") / (opt.subdir or "")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {name: {"distances": [], "epsilons": []} for name in opt.labels}
    _, _, test_loader = set_loader(opt, contrast_trans=False)

    for i, name in enumerate(opt.labels):
        print(f"\n---> Processing Model: {name}")
        embedding_model, classifier_head = load_model_and_head(
            opt.encoder_ckpts[i], opt.head_ckpts[i], opt.model, opt.n_cls
        )
        embedding_model.eval().to(device)
        classifier_head.eval().to(device)
        W = classifier_head.fc.weight.data

        progress_bar = tqdm(enumerate(test_loader), total=min(opt.limit, len(test_loader)), desc=f"Analyzing {name}")

        for j, (image, target) in progress_bar:
            if j >= opt.limit:
                break
            image, target = image.to(device), target.to(device)

            with torch.no_grad():
                features = embedding_model(image)
                logits = classifier_head(features).squeeze()
                pred = torch.argmax(logits)

            distance = calculate_distance_to_hyperplane(logits, W, target.item())
            all_results[name]["distances"].append(distance)

            if pred.item() == target.item():
                min_eps = find_min_flipping_epsilon(embedding_model, classifier_head, image, target, pred)
                all_results[name]["epsilons"].append(min_eps if min_eps is not None else -1.0)
            else:
                all_results[name]["epsilons"].append(0.0) 

    save_robustness_results_to_csv(all_results, out_dir, opt.dataset)


if __name__ == "__main__":
    main()
