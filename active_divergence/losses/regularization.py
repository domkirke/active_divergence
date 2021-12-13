import sys
sys.path.append('../')
import torch, torch.nn as nn
from active_divergence.losses.loss import Loss
from torch.distributions import Distribution, kl_divergence
from typing import Callable

class KLD(Loss):

    def forward(self, params1: Distribution, params2: Distribution, **kwargs) -> torch.Tensor:
        """
        Wrapper for Kullback-Leibler Divergence.
        Args:
            params1 (Distribution): first distribution
            params2: (Distribution) second distribution

        Returns:
            kld (torch.Tensor): divergence output

        """
        reduction = kwargs.get('reduction', self.reduction)
        assert isinstance(params1, Distribution) and isinstance(params2, Distribution), \
            "KLD only works with two distributions"
        ld = self.reduce(kl_divergence(params1, params2), reduction=reduction)
        return -ld


def l2_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    loss = torch.exp(-(x.unsqueeze(1).expand(x_size, y_size, dim) - y.unsqueeze(0).expand(x_size, y_size, dim)).pow(2) / float(dim))
    return loss


class MMD(Loss):
    def __repr__(self):
        return "MMD(kernel=%s)"%self.kernel

    def __init__(self, kernel: Callable = l2_kernel, *args, reduction=None, **kwargs):
        """
        Maximum Mean Discrepency (MMD) performs global distribution matching, in order to regularize q(z) rather that
        q(z|x). Used in Wasserstein Auto-Encoders.
        Args:
            kernel (Callable): kernel used (default: l2_kernel)
        """
        super(MMD, self).__init__(reduction=reduction)
        self.kernel = kernel

    def forward(self, params1=None, params2=None, **kwargs) -> torch.Tensor:
        assert params1, params2
        reduction = kwargs.get('reduction', self.reduction)
        sample1 = params1.sample() if not params1.has_rsample else params1.rsample()
        sample2 = params2.sample() if not params2.has_rsample else params2.rsample()
        sample1 = sample1.view(-1, sample1.shape[-1])
        sample2 = sample2.view(-1, sample2.shape[-1])

        x_kernel = self.reduce(self.kernel(sample1, sample1), reduction)
        y_kernel = self.reduce(self.kernel(sample2, sample2), reduction)
        xy_kernel = self.reduce(self.kernel(sample1, sample2), reduction)
        loss = x_kernel + y_kernel - 2*xy_kernel

        return self.reduce(loss)




