from pathlib import Path

import torch

from main_ce import set_loader
from main_supcon_fp16_jit import parse_option, get_loss_funcs, set_model, valid


if __name__ == "__main__":
    # grab default options
    opt = parse_option()
    opt.valid_split = 0

    # ------ set arg here -------
    opt.model = "resnet18"
    opt.dataset = "imagenet100"
    opt.method = "SupCon"
    opt.batch_size = 512
    opt.size = 64
    opt.mixed_precision = ""
    opt.temp = 0.1
    # ---------------------------


    # edit this section to set model and dataset

    model_loc = Path("/cluster/tufts/hugheslab/mlao01/Git/SupContrast/save/SupCon/imagenet100_models/SupCon_imagenet100_resnet18_lr_0.35_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_2025_06_19-15_37_54/ckpt_epoch_250.pth")

    opt.save_folder = model_loc.parent
    if opt.dataset == "imagenet100":
        opt.data_folder = '/cluster/tufts/hugheslab/datasets/ImageNet100/train/'
    
    print(opt)

    train_loader, _, test_loader = set_loader(opt, contrast_trans=True, for_test=True)
    model = set_model(opt).cuda()
    model.load_state_dict(torch.load(model_loc)["model"])
    valid(get_loss_funcs(opt), train_loader, test_loader, model, 0, opt, None)
