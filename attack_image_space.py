import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

from main_ce import set_loader
from analysis_util import (
    load_model_and_head,
    get_data_normalization,
    denorm,
    fgsm_attack,
    get_base_parser,
    process_opt,
    save_accuracy_results_to_csv,
    plot_accuracy_vs_epsilon,
)


def parse_option():
    """Parse CLI arguments for image space attack."""
    parser = get_base_parser("FGSM on models in image space (x)")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0, 0.05, 0.1, 0.15, 0.2])
    return process_opt(parser.parse_args())


def test(model, device, test_data, epsilon, mean, std):
    """Run FGSM attack on the image space for one epsilon value."""
    correct, total = 0, len(test_data)
    for data, target in tqdm(test_data, desc=f"ε={epsilon:.3f}"):
        data, target = data.to(device), target.to(device)

        if epsilon == 0:
            with torch.no_grad():
                pred = model(data).max(1, keepdim=True)[1]
                if pred.item() == target.item():
                    correct += 1
            continue

        data.requires_grad = True
        output = model(data)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        grad = data.grad.data

        data_denorm = denorm(data, mean, std, device)
        perturbed_image = fgsm_attack(data_denorm, epsilon, grad)
        perturbed_norm = transforms.Normalize(mean, std)(perturbed_image)

        with torch.no_grad():
            final_pred = model(perturbed_norm).max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                correct += 1

    acc = correct / float(total)

    return acc


def main():
    opt = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path("large-margin/figures/adversarial_x") / (opt.subdir or "")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Wrapper to combine the two loaded parts into a single sequential model
    class FullModel(nn.Module):
        def __init__(self, embedding_model, classifier_head):
            super().__init__()
            self.embedding_model = embedding_model
            self.classifier_head = classifier_head
        def forward(self, x):
            return self.classifier_head(self.embedding_model(x))

    _, _, test_loader = set_loader(opt, contrast_trans=False)
    mean, std = get_data_normalization(opt.dataset)
    test_data_list = list(test_loader)
    all_accuracies = {name: [] for name in opt.labels}

    for i, name in enumerate(opt.labels):
        print(f"\n---> Attacking Model: {name}")
        embedding_model, classifier_head = load_model_and_head(
            opt.encoder_ckpts[i], opt.head_ckpts[i], opt.model, opt.n_cls
        )
        model = FullModel(embedding_model, classifier_head).eval().to(device)

        model_accuracies = []
        for eps in opt.epsilons:
            acc = test(model, device, test_data_list, eps, mean, std)
            print(f"  ε={eps:.4f}  => Acc={acc:.4f}")
            
            model_accuracies.append(acc)
        all_accuracies[name] = model_accuracies

    plot_accuracy_vs_epsilon(opt.epsilons, all_accuracies, out_dir, "Accuracy vs Image Attack (x)")
    save_accuracy_results_to_csv(opt.epsilons, all_accuracies, out_dir, opt.dataset)


if __name__ == "__main__":
    main()
