from __future__ import print_function

import torch
from pathlib import Path
from main_ce import set_loader
from networks.resnet_big import SupConResNet
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm
import os
import sys


def parse_option():
    """
    Parses command-line arguments for feature extraction.
    """
    parser = argparse.ArgumentParser('argument for training')

    # required arguments
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='path to a single pre-trained model checkpoint')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'resnet200'],
                        help='model architecture name')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet100', 'imagenet', 'cifar2',
                                 'aircraft', 'cars', 'path'],
                        help='dataset')
    parser.add_argument('--size', type=int, default=224,
                        help='size of images after resizing')

    # dataloader settings
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')

    opt = parser.parse_args()

    # set the data folder path based on the dataset
    if opt.dataset == 'imagenet100':
        opt.data_folder = '/cluster/tufts/hugheslab/datasets/ImageNet100/train/'
    elif opt.dataset == 'imagenet':
        opt.data_folder = '/cluster/tufts/hugheslab/datasets/ImageNet/train/'
    else:
        opt.data_folder = './datasets/'

    # override image size for CIFAR datasets
    if 'cifar' in opt.dataset:
        opt.size = 32 if opt.size > 32 else opt.size

    # validation split is not needed for feature extraction, so it's fixed at 0
    opt.valid_split = 0

    print(opt)
    return opt


def set_model(opt):
    """Initializes the model for feature extraction."""
    model = SupConResNet(name=opt.model)
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
    return model


def get_features(model, dataloader, desc):
    """
    Extracts and returns features and labels from a given dataloader.
    The model is set to evaluation mode, and gradients are not computed.
    """
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            
            # forward pass through the encoder to get features
            features = model.encoder(images)

            # move features and labels to CPU and append to lists
            features_list.append(features.cpu())
            labels_list.append(labels.cpu())
            
    # concatenate all batches
    return torch.cat(features_list), torch.cat(labels_list)


def main(opt):
    """Main function to orchestrate the feature extraction process."""
    ckpt_path = Path(opt.ckpt_path)
    
    # define the output directory relative to the checkpoint's location
    output_dir_name = ckpt_path.parent

    print(f"Processing checkpoint: {ckpt_path.parent.name}/{ckpt_path.name}")
    print(f"Dataset: {opt.dataset}")
    sys.stdout.flush()

    # build data loaders for train and test sets
    # contrastive transformations are disabled as we only need standard augmentations
    print("Loading data...")
    train_loader, _, test_loader = set_loader(opt, contrast_trans=False)
    print("Data done")

    # build and load model from the specified checkpoint
    model = set_model(opt)
    print("Loading model weights from checkpoint...")
    sys.stdout.flush()
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint["model"])

    # create a specific output directory for the extracted features
    output_dir = Path(output_dir_name) / f"{opt.dataset}_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==> Get and save features for the training set
    train_features, train_labels = get_features(model, train_loader, "Extracting train features")
    torch.save(
        {"features": train_features, "labels": train_labels},
        output_dir / "train_features.pt",
    )
    print("Training features saved.")
    sys.stdout.flush()

    # ==> Get and save features for the test set
    test_features, test_labels = get_features(model, test_loader, "Extracting test features")
    torch.save(
        {"features": test_features, "labels": test_labels},
        output_dir / "test_features.pt",
    )
    print("Test features saved.")
    sys.stdout.flush()

    print(f"\nFeature extraction complete for {ckpt_path.name}")
    print(f"Features saved to: {output_dir}\n")


if __name__ == "__main__":
    main(parse_option())
