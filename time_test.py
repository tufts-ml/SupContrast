import cProfile
import timeit

import losses
import revised_losses
from test_losses import spoof_sup_embeds


if __name__ == "__main__":
    embeds, labels = spoof_sup_embeds(8, 8, 64)
    old_loss = losses.SupConLoss()
    new_loss = revised_losses.MultiviewSINCERELoss()

    # computation time spent on max, exp, and matmul
    cProfile.run("old_loss(embeds, labels)", sort="cumtime")
    # computation time spent on logsumexp and repeat_interleave
    cProfile.run("new_loss(embeds, labels)", sort="cumtime")

    # on a laptop CPU, new loss is 2-3x slower
    print(timeit.timeit("old_loss(embeds, labels)", number=10000, globals=globals()))
    print(timeit.timeit("new_loss(embeds, labels)", number=10000, globals=globals()))
