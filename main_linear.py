from __future__ import print_function

import sys
import argparse
import time
import math
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import save_model, set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier


class CachedDataset(Dataset):
    """Dataset for loading pre-computed features."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet100', 'imagenet', 'cifar2',
                                 'aircraft', 'cars', 'food101', 'pet', 'dtd', 'flowers', 'path'],
                        help='dataset')
    parser.add_argument('--valid_split', type=float, default=0,
                        help="proportion of train data to use for validation set")
    parser.add_argument('--size', type=int, default=32,
                        help='size of images after resizing')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--use_cache_features', action='store_true',
                        help='load pre-computed features from cache')
    parser.add_argument('--save_sub_dir', type=str, default='',
                        help='create sub directory in save/SupCon/ for model and tensorboard')
    parser.add_argument('--use_projection_head', action='store_true',
                        help='use projection head for feature extraction')

    opt = parser.parse_args()

    # set the path according to the environment
    if opt.dataset == 'imagenet100':
        opt.data_folder = '/cluster/tufts/hugheslab/datasets/ImageNet100/train/'
    elif opt.dataset == 'imagenet':
        opt.data_folder = '/cluster/tufts/hugheslab/datasets/ImageNet/train/'
    else:
        opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # get the method used by the checkpoint by grabbing everything before first _ in folder name
    ckpt_method = Path(opt.ckpt).parent.name
    opt.model_name = '{}_lr_{}_decay_{}_bsz_{}_{}'.format(
        opt.dataset, opt.learning_rate, opt.weight_decay, opt.batch_size, ckpt_method)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'cifar2':
        opt.n_cls = 2
    elif opt.dataset == 'imagenet100':
        opt.n_cls = 100
    elif opt.dataset == 'imagenet':
        opt.n_cls = 1000
    elif opt.dataset == 'aircraft':
        opt.n_cls = 102
    elif opt.dataset == 'cars':
        opt.n_cls = 196
    elif opt.dataset == 'food101':
        opt.n_cls = 101
    elif opt.dataset == 'pet':
        opt.n_cls = 37
    elif opt.dataset == 'dtd':
        opt.n_cls = 47
    elif opt.dataset == 'flowers':
        opt.n_cls = 102
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.model_path = './save/linear/{}'.format(opt.save_sub_dir) if opt.save_sub_dir \
        else './save/linear/{}_models'.format(opt.dataset)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    print(opt)
    return opt


def set_model(opt):
    # hack to load the correct model for the checkpoint
    if "resnet200" in opt.ckpt:
        opt.model = "resnet200"
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    if opt.use_projection_head:
        # feature dimension is 128 when using the projection head
        classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls, feat_dim=128)
    else:
        # feature dimension is inferred from encoder (e.g., 2048 for ResNet50)
        classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    # only load encoder weights if NOT using cached features
    if not opt.use_cache_features:
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
            model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, classifier, criterion


def set_cached_loader(opt):
    """
    Creates and returns dataloaders for cached features, handling validation split.
    """
    cache_path = Path(opt.ckpt).parent / '{}_features'.format(opt.dataset)
    train_file = cache_path / 'train_features.pt'
    test_file = cache_path / 'test_features.pt'

    if not train_file.is_file() or not test_file.is_file():
        print('Cached features not found at {}'.format(cache_path))
        print('Please run precompute_features.py first.')
        sys.exit(1)

    train_data = torch.load(train_file)
    test_data = torch.load(test_file)

    train_dataset = CachedDataset(train_data['features'], train_data['labels'])
    test_dataset = CachedDataset(test_data['features'], test_data['labels'])

    val_loader = None
    if opt.valid_split > 0:
        train_indices, val_indices = train_test_split(
            list(range(len(train_dataset))),
            test_size=opt.valid_split,
            stratify=train_dataset.labels,
            random_state=42,
        )
        val_dataset = Subset(train_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)

        val_loader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        if not opt.use_cache_features:
            if opt.use_projection_head:
                features = model(images)
            else:
                features = model.encoder(images)
        else:
            features = images

        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, topk=(1,))[0]
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if not opt.use_cache_features:
                if opt.use_projection_head:
                    features = model(images)
                else:
                    features = model.encoder(images)
            else:
                features = images

            output = classifier(features)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            if opt.n_cls > 4:
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top5.update(acc5[0], bsz)
            else:
                acc1 = accuracy(output, labels, topk=(1,))[0]
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f} | Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return losses.avg, top1.avg


def cache_outputs(val_loader, model, classifier, opt):
    # save model outputs for analysis and bootstrapping
    model.eval()
    classifier.eval()
    # caches for outputs
    embeds = []
    preds = []
    labels_list = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.float().cuda()

            if not opt.use_cache_features:
                if opt.use_projection_head:
                    b_embeds = model(images)
                else:
                    b_embeds = model.encoder(images)
            else:
                b_embeds = images

            b_preds = classifier(b_embeds)

            embeds.append(b_embeds.cpu())
            preds.append(b_preds.cpu())
            labels_list.append(labels.cpu())

    embeds = torch.cat(embeds)
    preds = torch.cat(preds)
    labels = torch.cat(labels_list)

    # save caches
    torch.save(embeds, os.path.join(opt.save_folder, "embeds.pth"))
    torch.save(preds, os.path.join(opt.save_folder, "preds.pth"))
    torch.save(labels, os.path.join(opt.save_folder, "labels.pth"))
    return


def main():
    time_start_main = time.time()

    opt = parse_option()

    # build data loader
    if opt.use_cache_features:
        train_loader, val_loader, test_loader = set_cached_loader(opt)
    else:
        train_loader, val_loader, _ = set_loader(opt, contrast_trans=False)
        _, _, test_loader = set_loader(opt, contrast_trans=False, for_test=True)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    best_acc, val_acc = 0, 0
    val_loss = 0
    test_acc_last = 0

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        if val_loader is not None:
            val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
            if val_acc > best_acc:
                best_acc = val_acc
        # print final accuracy for the test set evaluation run
        if epoch == opt.epochs:
            _, test_acc_last = validate(test_loader, model, classifier, criterion, opt)

    print('-' * 25)
    print('save folder        \t: {}'.format(opt.save_folder))
    print('best val accuracy  \t: {:.5f}'.format(best_acc))
    print('last val accuracy  \t: {:.5f}'.format(val_acc))
    print('last val loss      \t: {:.5f}'.format(val_loss))
    print('last test accuracy \t: {:.5f}'.format(test_acc_last))
    print('-' * 25)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(classifier, optimizer, opt, opt.epochs, save_file)

    # save features and predictions
    cache_outputs(test_loader, model, classifier, opt)

    time_end_main = time.time()
    print('\nTotal Time {:.2f} minute'.format((time_end_main - time_start_main) / 60))


if __name__ == '__main__':
    main()