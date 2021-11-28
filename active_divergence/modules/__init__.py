from torch import sqrt, tensor
from torch.nn.init import calculate_gain as cg
import torch, torch.nn as nn
from typing import Iterable

def calculate_gain(nnlin, param=None):
    try:
        gain = cg(nnlin, param)
    except ValueError:
        if nnlin == "SiLU":
            gain  = sqrt(tensor(2.))
        else:
            gain = 1.0
        pass
    return gain


class Reshape(nn.Module):
    def __repr__(self):
        return "Reshape%s"%(self.target_shape,)
    def __init__(self, *args):
        """
        A module reshaping incoming data into a target form.
        Args:
            *args: target shape
        """
        super(Reshape, self).__init__()
        self.target_shape = tuple([int(a) for a in args])

    def forward(self, x: torch.Tensor, batch_shape: Iterable[int] = tuple(), **kwargs) -> torch.Tensor:
        return torch.reshape(x, (*batch_shape, *self.target_shape))

class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        """
        A module unsqueezing incoming data among a given dimension.
        Args:
            dim (int): dimension index
        """
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.unsqueeze(x, self.dim)

from .layers import conv_hash, deconv_hash, gated_conv_hash, gated_deconv_hash, GatedConv1d, GatedConv2d, GatedConv3d, \
    GatedConvTranspose1d, GatedConvTranspose2d, GatedConvTranspose3d, GatedMLPLayer, MLP, GRU, RNN
from .encoders import MLPEncoder, MLPDecoder, ConvEncoder, DeconvEncoder