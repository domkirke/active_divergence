import torch, torch.nn as nn

class Loss(nn.Module):

    def __init__(self, reduction=None):
        super().__init__()
        self.reduction = None

    def reduce(self, loss, **kwargs):
        if "reduction" in kwargs:
            reduction = kwargs['reduction']
        else:
            reduction = self.reduction
        if reduction is None:
            return loss.mean(0).sum()
        elif reduction == "mean":
            return loss.mean()
        elif reduction == "seq":
            return loss.mean(0).mean(1).sum()

