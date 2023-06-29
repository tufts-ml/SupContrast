import torch

import losses
import revised_losses


def spoof_embeds(batch_size=10, embed_dim=10, n_views=2):
    # construct random normalized vectors
    embeds = torch.nn.functional.normalize(torch.rand((batch_size, embed_dim)) - .5)
    # return duplicates for each view
    return torch.stack((embeds,) * n_views, dim=1)


def spoof_self_sup_embeds(batch_size=10, embed_dim=10):
    return spoof_embeds(batch_size, embed_dim, 2)


def test_self_sup():
    embeds = spoof_self_sup_embeds()
    old_loss = losses.SupConLoss(contrast_mode="one")
    new_loss = revised_losses.InfoNCELoss()
    assert torch.isclose(old_loss(embeds), new_loss(embeds), atol=1e-7)
