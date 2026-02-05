from __future__ import print_function

import sys
import argparse
import time
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from main_ce import set_loader
from networks.resnet_big import SupConResNet


def parse_option():
    parser = argparse.ArgumentParser('argument for feature extraction')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet100', 'imagenet', 'cifar2',
                                 'aircraft', 'cars', 'food101', 'pet', 'dtd', 'flowers', 'path'],
                        help='dataset')
    parser.add_argument('--size', type=int, default=32,
                        help='size of images after resizing')

    # other setting
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--use_projection_head', action='store_true',
                        help='use projection head for feature extraction')

    opt = parser.parse_args()

    if opt.dataset == 'imagenet100':
        opt.data_folder = '/cluster/tufts/hugheslab/datasets/ImageNet100/train/'
    elif opt.dataset == 'imagenet':
        opt.data_folder = '/cluster/tufts/hugheslab/datasets/ImageNet/train/'
    else:
        opt.data_folder = './datasets/'

    # override image size for CIFAR datasets
    if 'cifar' in opt.dataset:
        opt.size = 32 if opt.size > 32 else opt.size

    # IMPORTANT
    opt.valid_split = 0

    if not os.path.isfile(opt.ckpt):
        raise ValueError('checkpoint not found: {}'.format(opt.ckpt))

    print(opt)
    return opt


def set_model(opt):
    if "resnet200" in opt.ckpt:
        opt.model = "resnet200"
    model = SupConResNet(name=opt.model)

    ckpt = torch.load(opt.ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict)

    return model


def get_features(val_loader, model, opt):
    """
    Extracts and returns features and labels from a given dataloader.
    """
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(val_loader, desc="Extracting features")):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            
            if opt.use_projection_head:
                features = model(images)
            else:
                features = model.encoder(images)

            features_list.append(features.cpu())
            labels_list.append(labels.cpu())

    return torch.cat(features_list), torch.cat(labels_list)


def main():
    time_start_main = time.time()

    opt = parse_option()

    train_loader, _, test_loader = set_loader(opt, contrast_trans=False, for_cache=True)

    model = set_model(opt)

    ckpt_path = Path(opt.ckpt)
    output_dir = ckpt_path.parent / '{}_features'.format(opt.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print('Processing checkpoint: {}'.format(ckpt_path.name))
    print('Output directory: {}'.format(output_dir))

    train_features, train_labels = get_features(train_loader, model, opt)
    torch.save(
        {"features": train_features, "labels": train_labels},
        output_dir / "train_features.pt",
    )
    print("Train features saved: shape {}".format(train_features.shape))

    test_features, test_labels = get_features(test_loader, model, opt)
    torch.save(
        {"features": test_features, "labels": test_labels},
        output_dir / "test_features.pt",
    )
    print("Test features saved: shape {}".format(test_features.shape))

    time_end_main = time.time()
    print('\nTotal Time {:.2f} minute'.format((time_end_main - time_start_main) / 60))


if __name__ == '__main__':
    main()