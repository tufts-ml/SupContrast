from __future__ import print_function

import argparse

import torch
from torchvision import transforms, datasets

from main_linear import set_model
from losses import SupConLoss
from revised_losses import MultiviewSINCERELoss, InfoNCELoss
from util import AverageMeter


def parse_option():
    parser = argparse.ArgumentParser('argument for testing')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str,
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str,
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str,
                        default=None, help='path to custom dataset')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--implementation', type=str, default='old',
                        choices=['old', 'new'], help='loss implemenation version')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.)),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        train=False,
                                        transform=test_transform)
    elif opt.dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         train=False,
                                         transform=test_transform)
    else:
        raise ValueError(opt.dataset)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return test_loader


def test(test_loader, model, criterion, opt):
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = torch.cat([images[0], images[1]], dim=0)
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            features = model.encoder(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, labels)

            # update metric
            losses.update(loss.item(), bsz)
    return losses.avg


def main(opt):
    # build data loader
    _, test_loader = set_loader(opt)
    # build model and criterion
    model, _, _ = set_model(opt)
    if opt.implementation == 'old':
        # original implementation does not set base_temperature, but setting here to make
        # hyperparameters comparable between implementations
        criterion = SupConLoss(temperature=opt.temp, base_temperature=opt.temp)
    else:
        if opt.method == 'SupCon':
            criterion = MultiviewSINCERELoss(temperature=opt.temp)
        elif opt.method == 'SimCLR':
            criterion = InfoNCELoss(temperature=opt.temp)
    return test(test_loader, model, criterion, opt)


if __name__ == '__main__':
    models = [
        "save/SupCon/cifar10_models/SupCon_new_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth",
        "save/SupCon/cifar10_models/SupCon_old_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth",
        "save/SupCon/cifar100_models/SupCon_new_cifar100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth",
        "save/SupCon/cifar100_models/SupCon_old_cifar100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth"
    ]
    for model in models:
        print(model)
        opt = parse_option()
        opt.ckpt = model
        av_loss = main(opt)
        print(f"Average Test Loss:{av_loss}\n")
