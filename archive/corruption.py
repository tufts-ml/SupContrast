from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from bootstrap_knn_acc import accuracy, test_contrastive_pred_knn
from bootstrap_lin_acc import bootstrap_metric, bootstrap_dif
from main_supcon import parse_option, set_model


class NumpyTransformDataset(Dataset):
    def __init__(self, np_data, transform=None):
        self.data = np_data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.data)


def corrupt_filter(np_data, corruption_level):
    corruption_start_ind = (corruption_level - 1) * 10000
    return np_data[corruption_start_ind:corruption_start_ind + 10000]


def test_dataloader(distortion_name, corruption_level, opt):
    # dataset specific normalization
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'cifar2':
        mean = (0.4977, 0.4605, 0.4160)
        std = (0.2537, 0.2481, 0.2535)
    elif opt.dataset == 'aircraft':
        mean = (0.4804, 0.5115, 0.5348)
        std = (0.1561, 0.1555, 0.1810)
    elif opt.dataset == 'cars':
        mean = (0.4707, 0.4601, 0.4549)
        std = (0.2319, 0.2318, 0.2373)
    elif opt.dataset == 'food101':
        mean = (0.5456, 0.4430, 0.3423)
        std = (0.2096, 0.2186, 0.2165)
    elif opt.dataset == 'pet':
        mean = (0.4786, 0.4462, 0.3962)
        std = (0.2073, 0.2042, 0.2053)
    elif opt.dataset == 'dtd':
        mean = (0.5301, 0.4734, 0.4243)
        std = (0.1250, 0.1246, 0.1211)
    elif opt.dataset == 'flowers':
        mean = (0.4312, 0.3786, 0.2944)
        std = (0.2385, 0.1858, 0.1986)
    elif opt.dataset == 'imagenet100' or opt.dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    # image is non-augmented
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([opt.size, opt.size]),
            transforms.ToTensor(),
            normalize,
        ])
    np_data = corrupt_filter(np.load(Path(opt.data_folder) / (distortion_name + ".npy")),
                             corruption_level)
    dataset = NumpyTransformDataset(np_data, transform=transform)
    dataloader = DataLoader(
            dataset, num_workers=opt.num_workers, pin_memory=True,
            batch_size=opt.batch_size)
    return dataloader


def corruption_forward(distortion_name, corruption_level, model, model_folder, opt):
    if (model_folder / (distortion_name + f"_{corruption_level}_embeds.pth")).exists():
        embeds = torch.load(model_folder / (distortion_name + f"_{corruption_level}_embeds.pth"))
    else:
        dataloader = test_dataloader(distortion_name, corruption_level, opt)
        embeds = torch.empty((0, 128))
        for images in dataloader:
            with torch.no_grad():
                cur_embeds = model(images.cuda())
            embeds = torch.vstack((embeds, cur_embeds.cpu()))
        torch.save(embeds, model_folder / (distortion_name + f"_{corruption_level}_embeds.pth"))
    return embeds, torch.tensor(corrupt_filter(np.load(Path(opt.data_folder) / "labels.npy"),
                                               corruption_level))


if __name__ == "__main__":
    # grab default options
    opt = parse_option()
    opt.valid_split = 0

    model_folders_groups = [
        [
            # standard CIFAR-10
            Path("2024_03_save/SupCon/cifar10_models/SINCERE_cifar10_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_20-22_04_43/"),  # noqa: E501
            Path("2024_03_save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.35_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_19-15_04_54/"),  # noqa: E501
            Path("2024_03_save/SupCon/cifar10_models/EpsSupInfoNCE_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_21-12_28_30/"),  # noqa: E501
        ],
        [
            # standard CIFAR-100
            Path("2024_03_save/SupCon/cifar100_models/SINCERE_cifar100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.05_trial_0_cosine_warm_2024_01_22-09_32_28/"),  # noqa: E501
            Path("2024_03_save/SupCon/cifar100_models/SupCon_cifar100_resnet50_lr_0.65_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_01_22-09_32_31/"),  # noqa: E501
            Path("2024_03_save/SupCon/cifar100_models/EpsSupInfoNCE_cifar100_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm_2024_03_21-12_52_07/"),  # noqa: E501
        ],
    ]

    # corruption distortions
    distortions = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression",
        "speckle_noise", "gaussian_blur", "spatter", "saturate"
    ]
    for model_folders in model_folders_groups:
        b_scores_cache = [[] for _ in range(len(model_folders))]
        acc_cache = [[] for _ in range(len(model_folders))]
        for folder_ind, model_folder in enumerate(model_folders):
            # model loading
            if "resnet50" in model_folder.name:
                opt.model = "resnet50"
            elif "resnet200" in model_folder.name:
                opt.model = "resnet200"
            model = set_model(opt).cuda()
            model.load_state_dict(torch.load(model_folder / "last.pth")["model"])
            # training output loading
            train_embeds = torch.load(model_folder / "train_embeds.pth")
            train_labels = torch.load(model_folder / "train_labels.pth")

            # dataset loading
            # note that for both, first 10k images are corrupted 1 and last 10k are corrupted 5
            # (10k + 1 to 20k are corrupted 2, etc.)
            if "cifar10_" in model_folder.name:
                opt.dataset = "cifar10"
                opt.data_folder = "/cluster/s/AAAAAAlab/datasets/CIFAR-10-C"
            if "cifar100_" in model_folder.name:
                opt.dataset = "cifar100"
                opt.data_folder = "/cluster/s/AAAAAAlab/datasets/CIFAR-100-C"
            # loop over the distortions
            print(model_folder)
            for distortion_name in distortions:
                for corruption_level in range(1, 6):
                    test_embeds, test_labels = corruption_forward(
                        distortion_name, corruption_level, model, model_folder, opt)
                    y_pred = test_contrastive_pred_knn(
                        train_embeds, test_embeds, train_labels, test_labels, 1)
                    acc_cache[folder_ind].append(accuracy(y_pred, test_labels))
                    # print("Means, 95% CI Low, 95% CI High")
                    # metric_mean, ci_low, ci_high, b_scores = bootstrap_metric(
                    #     y_pred, test_labels, accuracy)
                    # b_scores_cache[folder_ind].append(b_scores)
                    # print(metric_mean, ci_low, ci_high)
                    # print()
        # save acc caches
        if "cifar10_" in model_folders[0].name:
            torch.save(torch.Tensor(acc_cache), "cifar10c_acc.pth")
        if "cifar100_" in model_folders[0].name:
            torch.save(torch.Tensor(acc_cache), "cifar100c_acc.pth")
        # print accuracy difference for each pair of models
        # for i in range(1, len(model_folders)):
        #     for j in range(i):
        #         print("Accuracy Difference 95% CI for:")
        #         print(model_folders[j])
        #         print(torch.mean(torch.vstack(b_scores_cache[j])))
        #         print(model_folders[i])
        #         print(torch.mean(torch.vstack(b_scores_cache[i])))
        #         print(bootstrap_dif(torch.vstack(b_scores_cache[j]),
        #                             torch.vstack(b_scores_cache[i])))
        #         print()
