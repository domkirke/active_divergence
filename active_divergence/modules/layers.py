import sys, pdb
sys.path.append('../')
import torch, torch.nn as nn
import numpy as np
from active_divergence.modules import calculate_gain, layers, norm
from active_divergence.utils import checklist, checktuple, print_stats
from collections import OrderedDict
from typing import Union, Tuple, List, Iterable, Callable, NoReturn, Type
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

DEFAULT_NNLIN = "ELU"
dropout_hash = {1: nn.Dropout, 2:nn.Dropout2d, 3:nn.Dropout3d}

bn_hash = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
in_hash = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}
pixel_hash = {1: norm.PixelNorm1d, 2: norm.PixelNorm2d, 3: norm.PixelNorm3d}
norm_hash = {'batch':bn_hash, 'instance':in_hash, 'pixel': pixel_hash}
conv_hash = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
deconv_hash = {1: nn.ConvTranspose1d, 2:nn.ConvTranspose2d, 3:nn.ConvTranspose3d}
pooling_hash = {'avg': {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d},
                'max': {1: nn.MaxPool1d, 3: nn.MaxPool2d, 3: nn.MaxPool3d}}

class GatedConv(nn.Module):
    conv_hash = conv_hash
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, dim: int =None, nn_lin=None, 
                 padding: _size_1_t = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', **kwargs):
        """
        The Gated Convolutional Module uses two different convolutional operators, one used for features and one used
        for masking.
        (must not be used directly : instantiate GatedConv1d, GatedConv2d, or GatedConv3d instead)
        Args:
            in_channels: input channels
            out_channels: output channels
            kernel_size: kernel size
            stride: stride
            dim: input dimensionality
            padding: padding
            dilation: dilation
            groups: groups
            bias: bias
            padding_mode: padding mode (default: 'zeros')
        """

        super(GatedConv, self).__init__()
        self.conv_module = self.conv_hash[dim](in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, **kwargs)
        self.gated_module = self.conv_hash[dim](in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode, **kwargs)
        if isinstance(nn_lin, str):
            self.nn_lin = getattr(nn, nn_lin)
        elif isinstance(nn_lin, nn.Module):
            self.nn_lin = nn_lin
        else:
            self.nn_lin = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.nn_lin is not None:
            return self.nn_lin(self.conv_module(x)) * torch.sigmoid(self.gated_module(x))
        else:
            return self.conv_module(x) * torch.sigmoid(self.gated_module(x))

class GatedConv1d(GatedConv):
    def __init__(self, *args, **kwargs):
        super(GatedConv1d, self).__init__(*args, dim=1, **kwargs)

class GatedConv2d(GatedConv):
    def __init__(self, *args, **kwargs):
        super(GatedConv2d, self).__init__(*args, dim=2, **kwargs)

class GatedConv3d(GatedConv):
    def __init__(self, *args, **kwargs):
        super(GatedConv3d, self).__init__(*args, dim=3, **kwargs)

class GatedConvTranspose(GatedConv):
    conv_hash = deconv_hash

class GatedConvTranspose1d(GatedConvTranspose):
    def __init__(self, *args, **kwargs):
        super(GatedConvTranspose1d, self).__init__(*args, dim=1, **kwargs)

class GatedConvTranspose2d(GatedConvTranspose):
    def __init__(self, *args, **kwargs):
        super(GatedConvTranspose2d, self).__init__(*args, dim=2, **kwargs)

class GatedConvTranspose3d(GatedConvTranspose):
    def __init__(self, *args, **kwargs):
        super(GatedConvTranspose3d, self).__init__(*args, dim=3, **kwargs)


class ConvLayer(nn.Module):
    conv_hash = conv_hash
    def __init__(self, channels: Tuple[int, int],
                 kernel_size: Union[List[int], int] = 7, stride: Union[List[int], int] = 1, dim: int = 2,
                 nn_lin: str = DEFAULT_NNLIN, norm: bool = None, padding: Union[List[int], int] = None,
                 dilation: Union[List[int], int] = 1, dropout: float = None, bias: bool = True, **kwargs):
        """
        Stackable convolutional layer.
        Args:
            channels: channels
            kernel_size: kernel size (default: 7)
            stride: stride (default: 1)
            dim: input dimensionality (default: 2)
            nn_lin: non linearity (default: SiLU)
            norm: normalization (default: None)
            padding: padding (default: None)
            dilation: dilation (default: 1)
            dropout: dropout (default: None)
        """
        super(ConvLayer, self).__init__()
        self.channels = channels
        self.kernel_size =  np.array(checklist(kernel_size, n=dim), dtype=np.int)
        if padding is None:
            self.padding = np.floor(self.kernel_size/2).astype(np.int)
        else:
            self.padding = np.array(checklist(padding, n=dim))
        self.dilation = np.array(checklist(dilation, n=dim))
        self.stride = np.array(checklist(stride, n=dim))
        self.dim = dim
        if kwargs.get('output_padding') is not None:
            kwargs['output_padding'] = tuple(checklist(kwargs['output_padding'].tolist()))

        # init module
        self.conv = self.conv_hash[dim](in_channels=self.channels[0],
                                         out_channels=self.channels[1],
                                         kernel_size=tuple(self.kernel_size.tolist()),
                                         stride=tuple(self.stride.tolist()),
                                         padding=tuple(self.padding.tolist()),
                                         dilation =tuple(self.dilation.tolist()),
                                         bias=bias if bias is not None else False,
                                         **kwargs)
        if dropout is not None:
            self.dropout = dropout_hash[dim](dropout)
        if norm is not None and norm != "none":
            self.norm = norm_hash[norm][dim](channels[1])
        if nn_lin is not None:
            self.activation= getattr(nn, nn_lin)()
        self._init_modules()

    def _init_modules(self) -> NoReturn:
        nn.init.normal_(self.conv.weight, 0.0, 0.02)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        if hasattr(self, "norm"):
            if isinstance(self.norm, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.normal_(self.norm.weight, 1.0, 0.02)
                nn.init.zeros_(self.norm.bias)
        

    def input_shape(self, output_dim: Union[Iterable[int], int])->np.ndarray:
        """
        Returns layer's input shape required to obtain the target output shape.
        Args:
            output_dim (iterable[int]): target output shape

        Returns:
            input_dim (iterable[int]): required input shape
        """
        if not isinstance(output_dim, np.ndarray):
            output_dim = np.array(checklist(output_dim))
        return self.stride * (output_dim-1) - 2*self.padding + self.dilation*(self.kernel_size-1) + 1

    def output_shape(self, input_dim: Union[Iterable[int], int])->np.ndarray:
        """
        Returns layer's output shape obtained with a given input shape.
        Args:
            input_dim (iterable[int]): shape of the data input

        Returns:
            output_dim (iterable[int]): obtained output shape
        """
        if not isinstance(input_dim, np.ndarray):
            input_dim = np.array(checklist(input_dim))
        return np.floor((input_dim + 2*self.padding - self.dilation * (self.kernel_size - 1) - 1)/self.stride + 1)

    def forward(self, x: torch.Tensor, mod_closure=None)->torch.Tensor:
        """Performs convolution."""
        out = self.conv(x)
        if hasattr(self, "dropout"):
            out = self.dropout(out)
        if hasattr(self, "norm"):
            out = self.norm(out)
        if mod_closure is not None:
            out = mod_closure(out)
        if hasattr(self, "activation"):
            out = self.activation(out)
        return out


class DeconvLayer(ConvLayer):
    conv_hash = deconv_hash

    def __init__(self, *args, **kwargs):
        """
        Stackable deconvolutional layer.
        Args:
            channels: channels
            kernel_size: kernel size (default: 7)
            stride: stride (default: 1)
            dim: input dimensionality (default: 2)
            nn_lin: non linearity (default: SiLU)
            norm: normalization (default: None)
            padding: padding (default: None)
            dilation: dilation (default: 1)
            dropout: dropout (default: None)
            output_padding: output padding (default: None)
        """
        super(DeconvLayer, self).__init__(*args, **kwargs)
        self.output_padding = kwargs.get('output_padding', 0)
        if self.output_padding is None:
            self.output_padding = (0,) * self.dim

    @classmethod
    def get_output_padding(cls, output_dim: Iterable[int], kernel_size: Union[int, Iterable[int]] = None,
                           padding: Union[int, Iterable[int]] = None,
                           dilation: Union[int, Iterable[int]] = None,
                           stride: Union[int, Iterable[int]] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        gets the output padding required to preserve a target output dimension
        Args:
            output_dim: target output dimension
            ks (int or tuple[int]): kernel size
            padding (int or tuple[int]): padding
            dilation (int or tuple[int]): dilation
            stride (int or tuple[int]): stride

        Returns:
            output_padding (tuple[int]): required output padding
            out_shape (tuple[int]): output shape
        """
        if padding is None:
            padding = np.floor(kernel_size / 2)
        out_shape = np.floor((np.array(output_dim) + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        reversed_shape = stride * (out_shape-1) - 2*padding + dilation*(kernel_size-1) + 1
        # print(output_dim, reversed_shape, out_shape)
        return (output_dim - reversed_shape).astype(np.int), out_shape

    def output_shape(self, input_dim: Union[Iterable[int], int]) -> np.ndarray:
        """
        Returns layer's input shape required to obtain the target output shape.
        Args:
            output_dim (iterable[int]): target output shape

        Returns:
            input_dim (iterable[int]): required input shape
        """
        return self.stride * (input_dim-1) - 2*self.padding + self.dilation*(self.kernel_size-1) + 1 + self.output_padding

    def input_shape(self, output_dim: Union[Iterable[int], int]) -> np.ndarray:
        """
        Returns layer's output shape obtained with a given input shape.
        Args:
            input_dim (iterable[int]): shape of the data input

        Returns:
            output_dim (iterable[int]): obtained output shape
        """
        if not isinstance(output_dim, np.ndarray):
            output_dim = np.array(checklist(output_dim))
        out_shape = np.floor((output_dim + 2*self.padding - self.output_padding - self.dilation * (self.kernel_size - 1) - 1)/self.stride + 1)
        return out_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs convolution."""
        return super(DeconvLayer, self).forward(x)


gated_conv_hash = {1: GatedConv1d, 2:GatedConv2d, 3:GatedConv3d}
gated_deconv_hash = {1: GatedConvTranspose1d, 2:GatedConvTranspose2d, 3:GatedConvTranspose3d}

class GatedConvLayer(ConvLayer):
    conv_hash = gated_conv_hash
    def _init_modules(self):
        nn.init.xavier_normal_(self.conv.conv_module.weight)
        nn.init.xavier_normal_(self.conv.gated_module.weight)
        if self.conv.conv_module.bias is not None:
            nn.init.zeros_(self.conv.conv_module.bias)
        if self.conv.gated_module.bias is not None:
            nn.init.zeros_(self.conv.gated_module.bias)

class GatedDeconvLayer(DeconvLayer):
    conv_hash = gated_deconv_hash
    def _init_modules(self):
        nn.init.xavier_normal_(self.conv.conv_module.weight)
        nn.init.xavier_normal_(self.conv.gated_module.weight)
        if self.conv.conv_module.bias is not None:
            nn.init.zeros_(self.conv.conv_module.bias)
        if self.conv.gated_module.bias is not None:
            nn.init.zeros_(self.conv.gated_module.bias)




## Convolutional Blocks for specific applications

# Upsampling / Pooling blocks for progressive GANs
class UpsamplingConvBlock(nn.Module):
    def __init__(self, channels, *args, upsample=2, n_convs_per_block=2, up_pos="first", **kwargs):
        super().__init__()
        modules = []

        self.upsample = nn.Upsample(scale_factor=upsample)
        if up_pos == "first":
            modules.append(self.upsample)
        for n in range(n_convs_per_block):
            if n == 0:
                modules.append(DeconvLayer([channels[0], channels[1]], *args, **kwargs))
            else:
                modules.append(DeconvLayer([channels[1], channels[1]], *args, **kwargs)) 
        if up_pos == "last":
            modules.append(self.upsample)
        self.module_list= nn.ModuleList(modules)

    def forward(self, x, **kwargs):
        out = x
        for m in self.module_list:
            out = m(out)
        return out

    def output_shape(self, input_dim):
        for m in self.module_list:
            if isinstance(m, nn.Upsample):
                input_dim = input_dim * m.scale_factor
            else:
                input_dim = m.output_shape(input_dim)
        return input_dim

    def input_shape(self, output_dim):
        for m in reversed(self.module_list):
            if isinstance(m, nn.Upsample):
                input_dim = input_dim / m.scale_factor
            else:
                input_dim = m.input_shape(input_dim)
        return input_dim

    @classmethod
    def get_output_padding(cls, out_shape, up_pos='first', downsample=2, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if up_pos == "last":
            out_shape = out_shape / downsample
        output_padding, out_shape  = DeconvLayer.get_output_padding(out_shape, **kwargs)
        if up_pos == "first":
            out_shape = out_shape / downsample
        return output_padding, out_shape


# WaveNet blocks for TCNs
class WaveNetBlock(nn.Module):
    default_layer = GatedConv1d
    def __init__(self, dilation_channels=32, residual_channels=32, skip_channels=32, kernel_size=2, 
                 layer=None, n_convs_per_block=9, nnlin="Tanh", bias=False, dilation_rate=2, snap="right", **kwargs) -> None:
        super().__init__()
        self.gated_convs = [None] *  n_convs_per_block
        self.residual_convs = [None] * n_convs_per_block
        self.skip_convs = [None] * n_convs_per_block
        layer = getattr(layers, layer or self.default_layer)
        self.dilation_rate = dilation_rate
        for n in range(n_convs_per_block):
            current_dilation = dilation_rate ** n
            self.gated_convs[n] = layer([residual_channels, dilation_channels], kernel_size=kernel_size, nn_lin=nnlin, dilation = current_dilation, bias=bias, padding=0, dim=1)
            self.residual_convs[n] = nn.Conv1d(dilation_channels, residual_channels, kernel_size = 1, bias=bias, dilation=current_dilation)
            self.skip_convs[n] = nn.Conv1d(dilation_channels, skip_channels, kernel_size = 1, bias = bias, dilation=current_dilation)
        self.gated_convs = nn.ModuleList(self.gated_convs)
        self.residual_convs = nn.ModuleList(self.residual_convs)
        self.skip_convs = nn.ModuleList(self.skip_convs)
        self.snap = snap

    def snap_tensor(self, gated_out, residual_out):
        gated_shape = gated_out.shape[-1]
        if self.snap == "right":
            return residual_out[..., -gated_shape:]
        elif self.snap == "left":
            return residual_out[..., :gated_shape]

    def write_skip_buffer(self, skip_buffer, new_skip):
        skip_buffer = skip_buffer[..., -new_skip.shape[-1]:] + new_skip

    def __call__(self, x, skip_buffer=None, return_skip = True):
        out = x
        for i in range(len(self.gated_convs)):
            gated_out = self.gated_convs[i](out)
            residual_out = self.residual_convs[i](out)
            if skip_buffer is None:
                skip_buffer = self.skip_convs[i](gated_out)
            else:
                self.write_skip_buffer(skip_buffer, self.skip_convs[i](residual_out))
            out = gated_out + self.snap_tensor(gated_out, residual_out)
        if return_skip:
            return out, skip_buffer
        else:
            return out


class DownsamplingConvBlock(nn.Module):
    def __init__(self, channels, *args, downsample=2, pooling="avg", n_convs_per_block=2, up_pos="last", **kwargs):
        super().__init__()
        modules = []
        dim = kwargs.get('dim')
        self.downsample = pooling_hash[pooling][dim](downsample)
        if up_pos == "first":
            modules.append(self.downsample)
        for n in range(n_convs_per_block):
            if n == 0:
                modules.append(ConvLayer([channels[0], channels[1]], *args, **kwargs))
            else:
                modules.append(ConvLayer([channels[1], channels[1]], *args, **kwargs)) 
        if up_pos == "last":
            modules.append(self.downsample)
        self.module_list= nn.ModuleList(modules)

    def forward(self, x, **kwargs):
        out = x
        for m in self.module_list:
            out = m(out)
        return out

    def output_shape(self, input_dim):
        for m in self.module_list:
            if isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
                input_dim = input_dim / m.kernel_size
            else:
                input_dim = m.output_shape(input_dim)
        return input_dim

    def input_shape(self, output_dim):
        for m in reversed(self.module_list):
            if isinstance(m, nn.Upsample):
                input_dim = input_dim * m.kernel_size
            else:
                input_dim = m.input_shape(input_dim)
        return input_dim


## Fully-connected layers & blocks
class WeightedLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True) -> None:
        super().__init__()
        self._weight = nn.Parameter(torch.zeros(output_dim, input_dim))
        nn.init.xavier_normal_(self._weight)
        self._weight_mul = nn.Parameter(torch.ones(1, 1))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.ones(output_dim))

    def __repr__(self):
        return f"WeightedLinear(in_features={self._weight.shape[1]}, "\
               f"out_features={self._weight.shape[0]}, "\
               f"bias={self.bias is not None})"

    @property
    def weight(self):
        return self._weight_mul * self._weight / self._weight.norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, bias=self.bias)

class MLPLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, norm: str = "batch", dropout: float = None, weight_norm: bool = False, bias=True, **kwargs):
        """
        Stackable fully-connected layers.
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            norm (str): normalization (default: "batch")
            dropout (float): dropout (default: None)
            **kwargs:
        """
        super().__init__()
        modules = OrderedDict()
        self.input_dim = input_dim; self.output_dim = output_dim
        if weight_norm:
            self.linear = WeightedLinear(np.prod(input_dim), np.prod(output_dim), bias=bias)
        else:
            self.linear = nn.Linear(np.prod(input_dim), np.prod(output_dim), bias=bias)
        if dropout is not None:
            self.dropout = dropout_hash[1](dropout)
        if norm is not None:
            self.norm = norm_hash[norm][1](output_dim)
        self.nnlin = kwargs.get('nnlin', DEFAULT_NNLIN)
        if self.nnlin is not None:
            if isinstance(self.nnlin, str):
                self.activation = getattr(nn, self.nnlin)()
            elif isinstance(self.nnlin, nn.Module):
                self.activation = self.nnlin
        self.modules = modules
        self._init_modules()

    def forward(self, x, mod_closure=None):
        out = self.linear(x)
        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        if mod_closure is not None:
            out = mod_closure(out)
        if hasattr(self, 'activation'):
            out = self.activation(out)
        return out

    def _init_modules(self):
        if self.nnlin:
            gain = calculate_gain(self.nnlin, param=None)
        else:
            gain = 1.0
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)


class BilinearLayer(nn.Module):
    def __init__(self, input_dim_1: int, input_dim_2: int, output_dim: int, norm: str = "batch", dropout: float = None, bias=True, **kwargs):
        """
        Stackable fully-connected layers.
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            norm (str): normalization (default: "batch")
            dropout (float): dropout (default: None)
            **kwargs:
        """
        super().__init__()
        modules = OrderedDict()
        self.input_dim_1 = input_dim_1; self.input_dim_2 = input_dim_2; self.output_dim = output_dim
        self.bilinear = nn.Bilinear(np.prod(input_dim_1), np.prod(input_dim_2), np.prod(output_dim), bias=bias)
        if dropout is not None:
            self.dropout = dropout_hash[1](dropout)
        if norm is not None:
            self.norm = norm_hash[norm][1](output_dim)
        self.nn_lin = kwargs.get('nn_lin', DEFAULT_NNLIN)
        if self.nn_lin is not None:
            self.activation = getattr(nn, self.nn_lin)()
        self.modules = modules
        self._init_modules()

    def forward(self, x1, x2, mod_closure=None):
        out = self.bilinear(x1, x2)
        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        if mod_closure is not None:
            out = mod_closure(out)
        if hasattr(self, 'activation'):
            out = self.activation(out)
        return out

    def _init_modules(self):
        if self.nn_lin:
            gain = calculate_gain(self.nn_lin, param=None)
        else:
            gain = 1.0
        torch.nn.init.xavier_uniform_(self.bilinear.weight, gain=gain)
        if self.bilinear.bias is not None:
            torch.nn.init.zeros_(self.bilinear.bias)


class GatedMLPLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, norm: str = "batch", dropout: float = None, bias=True, **kwargs):
        """
        Stackable fully-connected layers.
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            norm (str): normalization (default: "batch")
            dropout (float): dropout (default: None)
        """
        super(GatedMLPLayer, self).__init__()
        self.input_dim = input_dim; self.output_dim = output_dim
        self.linear = nn.Linear(np.prod(input_dim), np.prod(output_dim), bias=bias)
        self.mask = nn.Linear(np.prod(input_dim), np.prod(input_dim), bias=False)
        if dropout is not None:
            self.dropout = dropout_hash[1](dropout)
        if norm is not None:
            self.norm = norm_hash[norm][1](output_dim)
        self.nn_lin = kwargs.get('nn_lin', DEFAULT_NNLIN)
        if self.nn_lin is not None:
            self.activation = getattr(nn, self.nn_lin)()
        self._init_modules()

    def _init_modules(self):
        if self.nn_lin:
            gain = calculate_gain(self.nn_lin, param=None)
        else:
            gain = 1.0
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.mask.weight, gain=gain)
        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, mod_closure=None, **args) -> torch.Tensor:
        out = self.linear(x * torch.sigmoid(self.mask(x)))
        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        if mod_closure is not None:
            out = mod_closure(out)
        if hasattr(self, 'activation'):
            out = self.activation(out)
        return out


class MLP(nn.Module):
    Layer = MLPLayer
    def __repr__(self):
        return self.module.__repr__()

    def __init__(self, input_dim: int, output_dim: int, nlayers: int = 1,
                 dim: int = 800, nnlin: str = DEFAULT_NNLIN, norm: str = "batch", bias=True, 
                 dropout: float = None, layer: Type = None, weight_norm: bool = False, **kwargs):
        """
        Multi-Layer fully-connected module.
        Args:
            input_dim (int): input dimensions
            output_dim (int): output dimensions
            nlayers (int): number of layers (default: 1)
            hidden_dims (int): number of hidden units (default: 800)
            nn_lin (str): non-linearity (default: "SiLU")
            norm (str): normalization (default: "batch")
            dropout (float): dropout rate (default: None)
        """
        super(MLP, self).__init__()
        self.input_dim = checktuple(input_dim)
        dims = [np.prod(input_dim)]+checklist(dim, n=nlayers)+[output_dim]
        layers = []
        layer_class = layer or self.Layer
        if isinstance(nnlin, list):
            assert len(nnlin) in (nlayers, nlayers+1)
            if len(nnlin) == nlayers:
                nnlin.append(None)
        else:
            nnlin = [nnlin] * nlayers + [None]
        dropout = checklist(dropout, n=len(dims))
        weight_norm = checklist(weight_norm, n=len(dims))
        bias = checklist(bias, n=len(dims))
        for i in range(len(dims)-1):
            current_norm = norm if i<len(dims)-2 else None
            layers.append(layer_class(dims[i], dims[i+1], nnlin=nnlin[i], norm=current_norm, dropout=dropout[i], bias=bias[i], weight_norm=weight_norm[i]))
        self.module = nn.Sequential(*tuple(layers))

    def forward(self, x: torch.Tensor, mod_closure=None, return_hidden=False, trace=None) -> torch.Tensor:
        batch_size = x.shape[:-(len(self.input_dim))]
        out = x.reshape(np.prod(batch_size), np.prod(self.input_dim))
        hidden = []
        for i, mod in enumerate(self.module):
            current_mod_closure = None if mod_closure is None else mod_closure[i]
            out = self.module[i](out, mod_closure=current_mod_closure)
            hidden.append(out.view(*batch_size, out.shape[-1]))
        out = out.view(*batch_size, out.shape[-1])
        if trace is not None: 
            trace['out'] = out
        if return_hidden:
            return out, hidden
        else:
            return out


class RNN(nn.RNN):
    def __init__(self, *args, **kwargs):
        """RNN module. Similar to torch.nn implementation, just force the batch_first argument and
        do not return the hidden vector for global coherence."""
        super().__init__(*args, batch_first=True, **kwargs)

    def forward(self, *args, return_hidden: bool = True, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out, h = super().forward(*args, **kwargs)
        if return_hidden:
            return out, h
        else:
            return out

class GRU(nn.GRU):
    def  __init__(self, input_dim: int, output_dim: int, nlayers=2, bidirectionnal=False, dropout=0.0, bias=True, **kwargs):
        """RNN module. Similar to torch.nn implementation, just force the batch_first argument and
        do not return the hidden vector for global coherence."""
        self.input_dim = checktuple(input_dim)
        super().__init__(np.prod(input_dim), output_dim, num_layers=nlayers, dropout=dropout, bias=bias, batch_first = True) 

    def forward(self, *args, return_hidden=True, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out, h = super(GRU, self).forward(*args, **kwargs)
        if return_hidden:
            return out, h
        else:
            return out

