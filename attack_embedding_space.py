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
)


def parse_option():
    parser = argparse.ArgumentParser(
        description="Run FGSM attack on embedding space (z) using CE loss."
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

    # Attack specific arguments
    parser.add_argument(
        "--epsilons", 
        nargs="+", 
        type=float, 
        default=[0, 0.005, 0.01, 0.02, 0.05, 0.1],
        help="List of epsilon values for the attack."
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
        default="./results_attack/",
        help="Directory to save results.",
    )

    opt = parser.parse_args()

    # Validation
    if not (len(opt.loss_names) == len(opt.encoder_paths) == len(opt.head_paths)):
        raise ValueError("The number of loss_names, encoder_paths, and head_paths must match.")

    opt.data_folder = (
        "/cluster/s/AAAAAAlab/datasets/ImageNet100/train/"
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

    # Force batch_size to 1 for easy gradient attack implementation
    opt.batch_size = 1

    opt.valid_split = 0.1

    return opt


def run_attack_for_epsilon(embedding_model, classifier_head, device, test_loader, epsilon):
    """
    Run FGSM attack on the embedding space for a specific epsilon.
    Returns the accuracy.
    """
    correct = 0
    total = 0

    # Iterate over test data (batch_size=1)
    for image, target in tqdm(test_loader, desc=f"  ε={epsilon:.3f}", leave=False):
        image, target = image.to(device), target.to(device)

        # 1. Get original embedding (z)
        with torch.no_grad():
            embed = embedding_model(image) 

        # 2. Perturb embedding
        if epsilon == 0:
            perturbed_embed = embed.clone()
        else:
            # Prepare for gradient calculation
            embed_for_attack = embed.clone().detach().requires_grad_(True)
            
            # Forward through classifier to get loss
            outputs = classifier_head(embed_for_attack)
            loss = F.cross_entropy(outputs, target)
            loss.backward()
            
            # Get gradient sign
            grad = embed_for_attack.grad.sign()

            # Find perturbation that satisfies distance constraint
            # Note: find_perturbation_for_distance handles the normalization logic
            perturbed_z = find_perturbation_for_distance(embed, grad, epsilon)
            perturbed_embed = perturbed_z.unsqueeze(0) # Add batch dim back

        # 3. Predict on perturbed embedding
        with torch.no_grad():
            pred = classifier_head(perturbed_embed).argmax(dim=1, keepdim=True)
            if pred.item() == target.item():
                correct += 1
        
        total += 1

    return correct / total


def main():
    opt = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    out_dir = Path(opt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {opt.dataset}")
    _, _, test_loader = set_loader(opt, contrast_trans=False, for_test=True)
    
    print(f"Evaluating on Test set ({len(test_loader.dataset)} images).")

    all_results = []

    # Iterate through each model configuration
    for i, model_name in enumerate(opt.loss_names):
        print(f"\nProcessing Model: {model_name}")
        
        encoder_path = opt.encoder_paths[i]
        head_path = opt.head_paths[i]

        # Load models
        embedding_model, classifier_head = load_model_and_head(
            encoder_path, head_path, opt.model_arch, opt.n_cls
        )
        embedding_model.eval().to(device)
        classifier_head.eval().to(device)

        # Run attack for each epsilon
        for eps in opt.epsilons:
            acc = run_attack_for_epsilon(
                embedding_model, classifier_head, device, test_loader, eps
            )
            tqdm.write(f"    ε={eps:.4f} \t=> Accuracy: {acc:.4f}")

            all_results.append({
                "model_name": model_name,
                "epsilon": eps,
                "accuracy": acc
            })

        # Cleanup
        del embedding_model, classifier_head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_results:
        print("\nNo results generated.")
        return

    # Save results
    output_file = out_dir / f"attack_embedding_results_{opt.dataset}.csv"
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(output_file, index=False, float_format="%.5f")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()