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
    save_accuracy_results_to_csv,
    plot_accuracy_vs_epsilon,
)


def parse_option():
    """Parse CLI arguments for embedding space attack."""
    parser = get_base_parser("FGSM on models in embedding space (z) using CE loss")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0, 0.005, 0.01, 0.02, 0.05, 0.1])
    return process_opt(parser.parse_args())


def test(embedding_model, classifier_head, device, test_data, epsilon):
    """Run FGSM attack on the embedding space for one epsilon value."""
    correct, total = 0, 0
    for image, target in tqdm(test_data, desc=f"ε={epsilon:.3f}"):
        image, target = image.to(device), target.to(device)

        with torch.no_grad():
            # embed = z
            embed = embedding_model(image) 

        if epsilon == 0:
            perturbed_embed = embed.clone()
        else:
            embed_for_attack = embed.clone().detach().requires_grad_(True)
            loss = F.cross_entropy(classifier_head(embed_for_attack), target)
            loss.backward()
            grad = embed_for_attack.grad

            perturbed_z = find_perturbation_for_distance(embed, grad.sign(), epsilon)
            perturbed_embed = perturbed_z.unsqueeze(0)

        with torch.no_grad():
            pred = classifier_head(perturbed_embed).argmax(dim=1, keepdim=True)
            if pred.item() == target.item():
                correct += 1
        total += 1
    return correct / total


def main():
    opt = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path("large-margin/figures/adversarial_z") / (opt.subdir or "")
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = set_loader(opt, contrast_trans=False)
    all_accuracies = {name: [] for name in opt.labels}

    for i, name in enumerate(opt.labels):
        print(f"\n---> Attacking Model: {name}")
        embedding_model, classifier_head = load_model_and_head(
            opt.encoder_ckpts[i], opt.head_ckpts[i], opt.model, opt.n_cls
        )
        embedding_model.eval().to(device)
        classifier_head.eval().to(device)

        model_accuracies = []
        for eps in opt.epsilons:
            acc = test(embedding_model, classifier_head, device, test_loader, eps)
            print(f"  ε={eps:.4f}  => Acc={acc:.4f}")
            model_accuracies.append(acc)
        all_accuracies[name] = model_accuracies

    plot_accuracy_vs_epsilon(opt.epsilons, all_accuracies, out_dir, "Accuracy vs Embedding Attack (z)")
    save_accuracy_results_to_csv(opt.epsilons, all_accuracies, out_dir, opt.dataset)


if __name__ == "__main__":
    main()
