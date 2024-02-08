from pathlib import Path

import torch

from main_ce import set_loader
from main_supcon import parse_option, set_model, valid


if __name__ == "__main__":
    # grab default options
    opt = parse_option()
    opt.valid_split = 0

    # edit this section to set model and dataset
    opt.dataset = "EDIT"
    model_loc = Path(f"save/SupCon/{opt.dataset}_models/EDIT/EDIT.pth")

    opt.save_folder = model_loc.parent
    if opt.dataset == "imagenet100":
        opt.data_folder = '/cluster/tufts/hugheslab/datasets/ImageNet100/train/'
    print(opt)

    train_loader, _, test_loader = set_loader(opt, contrast_trans=True, for_test=True)
    model = set_model(opt).cuda()
    model.load_state_dict(torch.load(model_loc)["model"])
    valid(train_loader, test_loader, model, 0, opt, None)
