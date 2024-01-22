import torch

from main_ce import set_loader
from main_supcon import parse_option, set_model, valid


if __name__ == "__main__":
    # grab default options
    opt = parse_option()
    opt.valid_split = 0

    # edit this section to set model and dataset
    model_loc = "save/SupCon/EDIT/EDIT.pth"
    opt.dataset = "save/SupCon/EDIT/last.pth"

    train_loader, _, test_loader = set_loader(opt, contrast_trans=True, for_test=True)
    model = set_model(opt).cuda()
    model.load_state_dict(torch.load(model_loc)["model"])
    valid(train_loader, test_loader, model, 0, opt, None)
