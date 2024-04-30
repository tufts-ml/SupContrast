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


def spoof_sup_embeds(n_labels=4, n_per_label=3, embed_dim=10):
    # use self-supervised embeds as base
    embeds = spoof_self_sup_embeds(n_labels, embed_dim)
    # construct labels with repeats next to each other
    labels = torch.arange(n_labels)
    labels = torch.repeat_interleave(labels, n_per_label, dim=0)
    # duplicate random vectors with the same label
    embeds = torch.repeat_interleave(embeds, n_per_label, dim=0)
    return embeds, labels


def test_self_sup():
    embeds = spoof_self_sup_embeds()
    # use default "all" contrast mode, which computes loss for all views instead of single view
    old_loss = losses.SupConLoss()
    new_loss = revised_losses.InfoNCELoss()
    # tolerance raised slightly due to old loss not using logsumexp function
    assert torch.isclose(old_loss(embeds), new_loss(embeds), atol=5e-7)


def test_sup():
    embeds, labels = spoof_sup_embeds()
    # use default "all" contrast mode, which computes loss for all views instead of single view
    old_loss = losses.SupConLoss()
    new_loss = revised_losses.MultiviewSINCERELoss()
    old_val = old_loss(embeds, labels)
    new_val = new_loss(embeds, labels)
    # new loss always strictly less than old loss due to the correction of the softmax denominator
    # old_loss usually greater than 1.5
    # new loss usually much less than 0.1, but varies more from random samples
    assert old_val > new_val


def test_sup_arccos():
    # fuzzing with the arccos_sim
    embeds, labels = spoof_sup_embeds()
    new_loss = revised_losses.MultiviewSINCERELoss(activation_func=revised_losses.arccos_sim)
    new_loss(embeds, labels)


def test_eps_0():
    # test that SINCERE and EpsSupInfoNCE are equivalent with epsilon=0
    embeds, labels = spoof_sup_embeds()
    # use default "all" contrast mode, which computes loss for all views instead of single view
    old_loss = revised_losses.MultiviewEpsSupInfoNCELoss(epsilon=0)
    new_loss = revised_losses.MultiviewSINCERELoss()
    old_val = old_loss(embeds, labels)
    new_val = new_loss(embeds, labels)
    assert torch.isclose(old_val, new_val)


def test_eps_non_0():
    # test that EpsSupInfoNCE is smaller than SINCERE with epsilon=0
    embeds, labels = spoof_sup_embeds()
    # use default "all" contrast mode, which computes loss for all views instead of single view
    old_loss = revised_losses.MultiviewEpsSupInfoNCELoss(epsilon=0.25)
    new_loss = revised_losses.MultiviewSINCERELoss()
    old_val = old_loss(embeds, labels)
    new_val = new_loss(embeds, labels)
    assert old_val < new_val
