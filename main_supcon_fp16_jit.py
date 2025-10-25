from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from contrast_acc import contrastive_acc, test_contrastive_acc, test_contrastive_acc_knn
from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from revised_losses import MultiviewSINCERELoss, MultiviewEpsSupInfoNCELoss

from torch.amp import autocast, GradScaler


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="num of workers to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of training epochs"
    )

    # optimization
    parser.add_argument(
        "--learning_rate", type=float, default=0.05, help="learning rate"
    )
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="700,800,900",
        help="where to decay lr, can be a list",
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

    # model dataset
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50", "resnet200"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=[
            "pet",
            "dtd",
            "cifar10",
            "cifar100",
            "imagenet100",
            "imagenet",
            "cifar2",
            "aircraft",
            "cars",
            "path",
        ],
        help="dataset",
    )
    parser.add_argument(
        "--valid_split",
        type=float,
        default=0,
        help="proportion of train data to use for validation set",
    )
    parser.add_argument(
        "--mean", type=str, help="mean of dataset in path in form of str tuple"
    )
    parser.add_argument(
        "--std", type=str, help="std of dataset in path in form of str tuple"
    )
    parser.add_argument(
        "--data_folder", type=str, default=None, help="path to custom dataset"
    )
    parser.add_argument(
        "--size", type=int, default=32, help="size of images after resizing"
    )

    # method
    parser.add_argument(
        "--method",
        type=str,
        default="SupCon",
        choices=["SINCERE", "SupCon", "SimCLR", "EpsSupInfoNCE"],
        help="choose method",
    )

    # temperature
    parser.add_argument(
        "--temp", type=float, default=0.07, help="temperature for loss function"
    )

    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument(
        "--warm", action="store_true", help="warm-up for large batch training"
    )
    parser.add_argument(
        "--trial", type=str, default="0", help="id for recording multiple runs"
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use torch.amp for mixed precision"
    )
    parser.add_argument(
        "--jit", action="store_true", help="use jit compiler on model, loss"
    )
    parser.add_argument(
        "--save_sub_dir", type=str, default="", help="create sub directory in save/SupCon/ for model and tensorboard"
    )

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == "path":
        assert (
            opt.data_folder is not None and opt.mean is not None and opt.std is not None
        )

    # set the path according to the environment
    if opt.data_folder is None:
        if opt.dataset == "imagenet100":
            opt.data_folder = "/cluster/tufts/hugheslab/datasets/ImageNet100/train/"
        elif opt.dataset == "imagenet":
            opt.data_folder = "/cluster/tufts/hugheslab/datasets/ImageNet/train/"
        else:
            opt.data_folder = "./datasets/"
    opt.model_path = f"./save/SupCon/{opt.save_sub_dir}{opt.dataset}_models"
    opt.tb_path = f"./save/SupCon/{opt.save_sub_dir}{opt.dataset}_tensorboard"

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = "{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}".format(
        opt.method,
        opt.dataset,
        opt.model,
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.temp,
        opt.trial,
    )

    if opt.cosine:
        opt.model_name = "{}_cosine".format(opt.model_name)

    # mixed precision training
    opt.scaler = GradScaler(device="cuda") if opt.mixed_precision else None

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = "{}_warm".format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate**3)
            opt.warmup_to = (
                eta_min
                + (opt.learning_rate - eta_min)
                * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs))
                / 2
            )
        else:
            opt.warmup_to = opt.learning_rate
    # add time to model name
    opt.model_name += "_" + time.strftime("%Y_%m_%d-%H_%M_%S")

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    # write args to log
    print(opt)
    return opt


def get_loss_funcs(opt):
    sincere_loss_func = (
        MultiviewSINCERELoss(temperature=opt.temp)
        if opt.method != "EpsSupInfoNCE"
        else MultiviewEpsSupInfoNCELoss(temperature=opt.temp)
    )
    # original implementation does not set base_temperature, but setting here to make
    # hyperparameters comparable between implementations
    supcon_loss_func = SupConLoss(temperature=opt.temp, base_temperature=opt.temp)

    return {
        "sincere": torch.compile(sincere_loss_func) if opt.jit else sincere_loss_func,
        "supcon": torch.compile(supcon_loss_func) if opt.jit else supcon_loss_func,
    }


def set_model(opt):
    model = SupConResNet(name=opt.model)
    if torch.cuda.is_available():
        if "device" not in opt:
            model = model.cuda()
        else:
            model = model.to(opt.device)
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.parallel.DistributedDataParallel(model.encoder)
        cudnn.benchmark = True
    return torch.compile(model) if opt.jit else model


# def check_gradients(model):
#     nan_found = False
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             if torch.isnan(param.grad).any():
#                 print(f"NaN gradient in {name}")
#                 nan_found = True
#             if torch.isinf(param.grad).any():
#                 print(f"Inf gradient in {name}")
#                 nan_found = True
#     return nan_found


def train(loss_funcs, train_loader, model, optimizer, epoch, opt, logger):
    """one epoch training"""
    sincere_loss_func = loss_funcs["sincere"]
    supcon_loss_func = loss_funcs["supcon"]
    model.train()

    av_batch_time = AverageMeter()
    av_data_time = AverageMeter()
    av_sincere = AverageMeter()
    av_supcon = AverageMeter()
    av_acc = AverageMeter()
    av_grad = AverageMeter()

    end = time.time()
    # change reshuffle split of data across GPUs
    if "device" in opt:
        train_loader.sampler.set_epoch(epoch)
    for idx, (image_aug_tuple, labels) in enumerate(train_loader):
        av_data_time.update(time.time() - end)

        images = torch.cat([image_aug_tuple[0], image_aug_tuple[1]], dim=0)
        if torch.cuda.is_available():
            if "device" not in opt:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            else:
                images = images.to(opt.device, non_blocking=True)
                labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        with autocast("cuda", enabled=opt.mixed_precision):
            # forward
            flat_embeds = model(images)
            # reshape from (2B, D) to (B, 2, D)
            embeds = torch.cat(
                [
                    aug.unsqueeze(1)
                    for aug in torch.split(flat_embeds, [bsz, bsz], dim=0)
                ],
                dim=1,
            )
            # compute losses
            # loss is averaged across GPU-specific batches if using multiple GPUs, as in SupCon
            # see MoCo v3 for full batch size parallelization with torch's all_gather
            sincere_loss = sincere_loss_func(embeds, labels)
            supcon_loss = supcon_loss_func(embeds, labels)

        # update averages
        av_sincere.update(sincere_loss.item(), bsz)
        av_supcon.update(supcon_loss.item(), bsz)

        # SGD
        # always zero in case grad accidentally calculated for non-train epoch
        optimizer.zero_grad()

        if not opt.mixed_precision:
            if opt.method == "SINCERE" or opt.method == "EpsSupInfoNCE":
                sincere_loss.backward()
            elif opt.method == "SupCon":
                supcon_loss.backward()
            else:
                raise ValueError(
                    "contrastive method not supported: {}".format(opt.method)
                )

            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                float('inf')
            ).item()    

            optimizer.step()

        else:
            if opt.method == "SINCERE" or opt.method == "EpsSupInfoNCE":
                opt.scaler.scale(sincere_loss).backward()
            elif opt.method == "SupCon":
                opt.scaler.scale(supcon_loss).backward()
            else:
                raise ValueError(
                    "contrastive method not supported: {}".format(opt.method)
                )

            opt.scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                float('inf')
            ).item()

            opt.scaler.step(optimizer)
            opt.scaler.update()
        
        # check_gradients(model)
        if total_norm is not None and math.isfinite(total_norm):
            av_grad.update(total_norm, 1)
        else:
            print("\n**Gradient NaN**\n")

        # compute accuracy
        with torch.no_grad():
            acc = contrastive_acc(embeds, labels)
            av_acc.update(acc.item(), bsz)

        # measure elapsed time
        av_batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=av_batch_time,
                    data_time=av_data_time,
                )
            )

            # print(f"flat_embeds \t {flat_embeds.dtype}")            # torch.float.32
            # print(f"embeds      \t {embeds.dtype}")                 # torch.float.32

            sys.stdout.flush()

    # tensorboard logger
    if "device" not in opt or opt.device == 0:
        log_folder = "train/"
        logger.add_scalar(f"{log_folder}SINCERE", av_sincere.avg, epoch)
        logger.add_scalar(f"{log_folder}SupCon", av_supcon.avg, epoch)
        logger.add_scalar(f"{log_folder}Accuracy", av_acc.avg, epoch)
        logger.add_scalar(f"{log_folder}Gradient Norm", av_grad.avg, epoch)
    # log values independent of forward passes
    logger.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
    return


def valid(loss_funcs, train_loader, valid_loader, model, epoch, opt, logger):
    """validation"""
    # loggger is given if valid_loader is validation set, otherwise is test set
    val_is_test = logger is None

    sincere_loss_func = loss_funcs["sincere"]
    supcon_loss_func = loss_funcs["supcon"]

    # caches for data
    train_embeds = torch.empty((0, 128))
    train_labels = torch.empty((0,))
    # caches for test data
    if val_is_test:
        test_embeds = torch.empty((0, 128))
        test_labels = torch.empty((0,))

    for i, loader in enumerate([train_loader, valid_loader]):
        is_train = i == 0
        model.eval()

        av_batch_time = AverageMeter()
        av_data_time = AverageMeter()
        av_sincere = AverageMeter()
        av_supcon = AverageMeter()
        av_acc_top_1 = AverageMeter()
        av_acc_top_5 = AverageMeter()

        end = time.time()
        # change reshuffle split of data across GPUs
        if "device" in opt:
            loader.sampler.set_epoch(epoch)
        for idx, (image_aug_tuple, labels) in enumerate(loader):
            av_data_time.update(time.time() - end)

            images = torch.cat([image_aug_tuple[0], image_aug_tuple[1]], dim=0)
            if torch.cuda.is_available():
                if "device" not in opt:
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                else:
                    images = images.to(opt.device, non_blocking=True)
                    labels = labels.to(opt.device, non_blocking=True)
            bsz = labels.shape[0]

            with autocast("cuda", enabled=opt.mixed_precision):
                # forward
                with torch.no_grad():
                    flat_embeds = model(images)
                # reshape from (2B, D) to (B, 2, D)
                embeds = torch.cat(
                    [
                        aug.unsqueeze(1)
                        for aug in torch.split(flat_embeds, [bsz, bsz], dim=0)
                    ],
                    dim=1,
                )

            # cache train outputs
            if is_train:
                train_embeds = torch.vstack((train_embeds, embeds[:, 0].cpu()))
                train_labels = torch.hstack((train_labels, labels.cpu()))
            else:
                # cache test outputs
                if val_is_test:
                    test_embeds = torch.vstack((test_embeds, embeds[:, 0].cpu()))
                    test_labels = torch.hstack((test_labels, labels.cpu()))
                # compute validation accuracy
                av_acc_top_1.update(
                    test_contrastive_acc(
                        train_embeds.cuda(),
                        embeds[:, 0].cuda(),
                        train_labels.cuda(),
                        labels.cuda(),
                    ).item(),
                    bsz,
                )
                av_acc_top_5.update(
                    test_contrastive_acc_knn(
                        train_embeds.cuda(),
                        embeds[:, 0].cuda(),
                        train_labels.cuda(),
                        labels.cuda(),
                        5,
                    ).item(),
                    bsz,
                )
            # compute losses (note there's no class balancing sampler for test)
            # loss is averaged across GPU-specific batches if using multiple GPUs, as in SupCon
            # see MoCo v3 for full batch size parallelization with torch's all_gather
            with autocast("cuda", enabled=opt.mixed_precision):
                sincere_loss = sincere_loss_func(embeds, labels)
                supcon_loss = supcon_loss_func(embeds, labels)

            # update averages
            av_sincere.update(sincere_loss.item(), bsz)
            av_supcon.update(supcon_loss.item(), bsz)

            # measure elapsed time
            av_batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "DT {data_time.val:.3f} ({data_time.avg:.3f})\t".format(
                        epoch,
                        idx + 1,
                        len(loader),
                        batch_time=av_batch_time,
                        data_time=av_data_time,
                    )
                )

                # print(f"flat_embeds \t {flat_embeds.dtype}")            # torch.float.32
                # print(f"embeds      \t {embeds.dtype}")                 # torch.float.32

                sys.stdout.flush()
    if "device" not in opt or opt.device == 0 and not is_train:
        # tensorboard logger
        if not val_is_test:
            log_folder = "valid/"
            logger.add_scalar(f"{log_folder}SINCERE", av_sincere.avg, epoch)
            logger.add_scalar(f"{log_folder}SupCon", av_supcon.avg, epoch)
            logger.add_scalar(f"{log_folder}Top 1 Accuracy", av_acc_top_1.avg, epoch)
            logger.add_scalar(f"{log_folder}Top 5 Accuracy", av_acc_top_5.avg, epoch)
        else:
            # print output
            print(f"Test SINCERE: {av_sincere.avg}")
            print(f"Test SupCon: {av_supcon.avg}")
            print(f"Test Top 1 Accuracy: {av_acc_top_1.avg}")
            print(f"Test Top 5 Accuracy: {av_acc_top_5.avg}")
            # save caches
            torch.save(train_embeds, os.path.join(opt.save_folder, "train_embeds.pth"))
            torch.save(train_labels, os.path.join(opt.save_folder, "train_labels.pth"))
            torch.save(test_embeds, os.path.join(opt.save_folder, "test_embeds.pth"))
            torch.save(test_labels, os.path.join(opt.save_folder, "test_labels.pth"))


def test(loss_funcs, model, opt):
    train_loader, _, test_loader = set_loader(opt, contrast_trans=True, for_test=True)
    valid(loss_funcs, train_loader, test_loader, model, 0, opt, None)


def main(opt):
    # get loss functions
    loss_funcs = get_loss_funcs(opt)

    # build data loader
    train_loader, valid_loader, _ = set_loader(opt, contrast_trans=True)

    # build model
    model = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard, only for first process if multiple
    if "device" not in opt or opt.device == 0:
        logger = SummaryWriter(log_dir=opt.tb_folder)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train(loss_funcs, train_loader, model, optimizer, epoch, opt, logger)
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # use valid_loader if present
        if epoch % opt.save_freq == 0 and valid_loader is not None:
            time1 = time.time()
            valid(loss_funcs, train_loader, valid_loader, model, epoch, opt, logger)
            time2 = time.time()
            print("valid {}, total time {:.2f}".format(epoch, time2 - time1))

        # checkpoint
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # print test statistics
    test(loss_funcs, model, opt)


def launch_parallel(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # need to use gloo instead of nccl for Windows, but nccl faster on Linux
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    opt = parse_option()
    # modify options for parallel processing
    opt.device = rank  # device not in opt if not using parallel processing
    opt.batch_size = opt.batch_size // world_size
    main(opt)


if __name__ == "__main__":
    parallel = False
    if not parallel:
        main(parse_option())
    else:
        world_size = 2
        torch.multiprocessing.spawn(launch_parallel, (world_size,), world_size)
