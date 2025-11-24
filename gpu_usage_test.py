import torch
import torch.optim as optim

import losses
from networks.resnet_big import SupConResNet
import revised_losses

from test_losses import spoof_sup_embeds


def dummy_loss(embeds, labels):
    # embeds are B, 2, D; labels are B
    # mean L1 distance
    return torch.mean(embeds[:, 0] - embeds[:, 1])


if __name__ == "__main__":
    old_loss = losses.SupConLoss()
    new_loss = revised_losses.MultiviewSINCERELoss()
    model = SupConResNet(name="resnet50").cuda()
    # optimizer doesn't seem to affect GPU usage, but included for completeness
    optimizer = optim.SGD(model.parameters(),
                          lr=.001)
    for i in range(15, 50):
        print(10 * i)
        # shape B, 2, D where B = 10 * i
        _, labels = spoof_sup_embeds(10, i, embed_dim=128)
        bsz = labels.shape[0]
        flat_embeds = torch.rand((2 * bsz, 3, 32, 32))
        print(flat_embeds.shape)
        flat_embeds = flat_embeds.cuda()
        labels = labels.cuda()
        # for loss in [old_loss, new_loss]:
        loss = dummy_loss
        embeds = torch.cat([aug.unsqueeze(1) for aug in
                            torch.split(model(flat_embeds), [bsz, bsz], dim=0)], dim=1)
        loss_val = loss(embeds, labels)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
