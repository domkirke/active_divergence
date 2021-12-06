import sys, pdb
sys.path.append('../')
import torch, torch.nn as nn
import numpy as np
from active_divergence.modules import calculate_gain
from active_divergence.utils import checklist, checktuple, print_stats
from collections import OrderedDict
from typing import Union, Tuple, List, Iterable, Callable, NoReturn, Type
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

DEFAULT_NNLIN = "ELU"
dropout_hash = {1: nn.Dropout, 2:nn.Dropout2d, 3:nn.Dropout3d}

bn_hash = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
in_hash = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}
norm_hash = {'batch':bn_hash, 'instance':in_hash}
conv_hash = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
deconv_hash = {1: nn.ConvTranspose1d, 2:nn.ConvTranspose2d, 3:nn.ConvTranspose3d}

class GatedConv(nn.Module):
    conv_hash = conv_hash
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, dim: int =None,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
                 nn_lin: str = DEFAULT_NNLIN, norm: str = None, padding: Union[List[int], int] = None,
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
        super().__init__()
        self.channels = channels
        self.kernel_size =  np.array(checklist(kernel_size, n=dim), dtype=np.int)
        if padding is None:
            self.padding = np.ceil(self.kernel_size/2).astype(np.int)
        else:
            self.padding = np.array(checklist(padding, n=dim))
        self.dilation = np.array(checklist(dilation, n=dim))
        self.stride = np.array(checklist(stride, n=dim))
        self.dim = dim
        if kwargs.get('output_padding') is not None:
            kwargs['output_padding'] = tuple(checklist(kwargs['output_padding'].tolist()))


        # init module
        modules = OrderedDict()
        modules['conv'] = self.conv_hash[dim](in_channels=self.channels[0],
                                         out_channels=self.channels[1],
                                         kernel_size=tuple(self.kernel_size.tolist()),
                                         stride=tuple(self.stride.tolist()),
                                         padding=tuple(self.padding.tolist()),
                                         dilation =tuple(self.dilation.tolist()),
                                         bias=bias,
                                         **kwargs)
        if dropout is not None:
            modules['dropout'] = dropout_hash[dim](dropout)
        if norm is not None and norm != "none":
            modules['norm'] = norm_hash[norm][dim](channels[1])
        if nn_lin is not None:
            modules['nn_lin'] = getattr(nn, nn_lin)()
        self.module = nn.Sequential(modules)
        self._init_modules()

    def _init_modules(self) -> NoReturn:
        if self.dim > 1:
            nn.init.kaiming_normal_(self.module.conv.weight)
        else:
            nn.init.uniform_(self.module.conv.weight, -0.1, 0.1)
        if self.module.conv.bias is not None:
            nn.init.zeros_(self.module.conv.bias)

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
        conv_out = self.module['conv'](x)
        if 'dropout' in self.module:
            conv_out = self.module['dropout'](conv_out)
        if 'norm' in self.module:
            conv_out = self.module['norm'](conv_out)
        if mod_closure is not None:
            conv_out = mod_closure(conv_out)
        if 'nn_lin' in self.module:
            conv_out = self.module['nn_lin'](conv_out)
        return conv_out

class DummyConvLayer(ConvLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.dirac_(self.module.conv.weight)
        self.module.conv.weight.requires_grad_(False)
        if self.module.conv.bias:
            nn.init.zeros_(self.module.conv.bias)        
            self.module.conv.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor, mod_closure=None)->torch.Tensor:
        """Performs convolution."""
        if mod_closure is None:
            conv_out = self.module['conv'](x)
        else:
            conv_out = mod_closure(x)
        if 'dropout' in self.module:
            conv_out = self.module['dropout'](conv_out)
        if 'norm' in self.module:
            conv_out = self.module['norm'](conv_out)
        if 'nn_lin' in self.module:
            conv_out = self.module['nn_lin'](conv_out)
        return conv_out


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

    @classmethod
    def get_output_padding(cls, output_dim: Iterable[int], ks: Union[int, Iterable[int]],
                           padding: Union[int, Iterable[int]], dilation: Union[int, Iterable[int]],
                           stride: Union[int, Iterable[int]]) -> Tuple[np.ndarray, np.ndarray]:
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
        out_shape = np.floor((np.array(output_dim) + 2 * padding - dilation * (ks - 1) - 1) / stride + 1)
        reversed_shape = stride * (out_shape-1) - 2*padding + dilation*(ks-1) + 1
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Performs convolution."""
        return super().forward(self, x, **kwargs)


gated_conv_hash = {1: GatedConv1d, 2:GatedConv2d, 3:GatedConv3d}
gated_deconv_hash = {1: GatedConvTranspose1d, 2:GatedConvTranspose2d, 3:GatedConvTranspose3d}

class GatedConvLayer(ConvLayer):
    conv_hash = gated_conv_hash
    def _init_modules(self):
        nn.init.xavier_normal_(self.module.conv.conv_module.weight)
        nn.init.xavier_normal_(self.module.conv.gated_module.weight)
        if self.module.conv.conv_module.bias is not None:
            nn.init.zeros_(self.module.conv.conv_module.bias)

class GatedDeconvLayer(DeconvLayer):
    conv_hash = gated_deconv_hash
    def _init_modules(self):
        nn.init.xavier_normal_(self.module.conv.conv_module.weight)
        nn.init.xavier_normal_(self.module.conv.gated_module.weight)
        if self.module.conv.conv_module.bias is not None:
            nn.init.zeros_(self.module.conv.conv_module.bias)


class MLPLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, norm: str = "batch", dropout: float = None, **kwargs):
        """
        Stackable fully-connected layers.
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            norm (str): normalization (default: "batch")
            dropout (float): dropout (default: None)
            **kwargs:
        """
        super(MLPLayer, self).__init__()
        self.input_dim = input_dim; self.output_dim = output_dim
        self.linear = nn.Linear(np.prod(input_dim), np.prod(output_dim))
        if dropout is not None:
            self.dropout = dropout_hash[1](dropout)
        if norm is not None:
            self.norm = norm_hash[norm][1](output_dim)
        self.nn_lin = kwargs.get('nn_lin', DEFAULT_NNLIN)
        if self.nn_lin is not None:
            self.activation = getattr(nn, self.nn_lin)()
        self._init_modules()

    def forward(self, x, mod_closure=None):
        out = self.linear(x)
        if hasattr(self, "dropout"):
            out = self.dropout(out)
        if hasattr(self, "norm"):
            out = self.norm(out)
        if mod_closure is not None:
            out = mod_closure(out)
        if hasattr(self, "activation"):
            out = self.activation(out)
        return out

    def _init_modules(self):
        if self.nn_lin:
            gain = calculate_gain(self.nn_lin, param=None)
        else:
            gain = 1.0
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        torch.nn.init.zeros_(self.linear.bias)


class GatedMLPLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, norm: str = "batch", dropout: float = None, **kwargs):
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
        self.linear = nn.Linear(np.prod(input_dim), np.prod(output_dim))
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

class GatedConv(nn.Module):
    conv_hash = conv_hash
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, dim: int =None,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            self.padding = np.ceil(self.kernel_size/2).astype(np.int)
        else:
            self.padding = np.array(checklist(padding, n=dim))
        self.dilation = np.array(checklist(dilation, n=dim))
        self.stride = np.array(checklist(stride, n=dim))
        self.dim = dim
        if kwargs.get('output_padding') is not None:
            kwargs['output_padding'] = tuple(checklist(kwargs['output_padding'].tolist()))

        # init module
        modules = OrderedDict()
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
        if self.dim > 1:
            nn.init.kaiming_normal_(self.conv.weight)
        else:
            nn.init.uniform_(self.conv.weight, -0.1, 0.1)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

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

    @classmethod
    def get_output_padding(cls, output_dim: Iterable[int], ks: Union[int, Iterable[int]],
                           padding: Union[int, Iterable[int]], dilation: Union[int, Iterable[int]],
                           stride: Union[int, Iterable[int]]) -> Tuple[np.ndarray, np.ndarray]:
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
        out_shape = np.floor((np.array(output_dim) + 2 * padding - dilation * (ks - 1) - 1) / stride + 1)
        reversed_shape = stride * (out_shape-1) - 2*padding + dilation*(ks-1) + 1
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
        out = self.module(x)
        return out


gated_conv_hash = {1: GatedConv1d, 2:GatedConv2d, 3:GatedConv3d}
gated_deconv_hash = {1: GatedConvTranspose1d, 2:GatedConvTranspose2d, 3:GatedConvTranspose3d}

class GatedConvLayer(ConvLayer):
    conv_hash = gated_conv_hash
    def _init_modules(self):
        nn.init.xavier_normal_(self.module.conv.conv_module.weight)
        nn.init.xavier_normal_(self.module.conv.gated_module.weight)
        if self.module.conv.conv_module.bias is not None:
            nn.init.zeros_(self.module.conv.conv_module.bias)

class GatedDeconvLayer(DeconvLayer):
    conv_hash = gated_deconv_hash
    def _init_modules(self):
        nn.init.xavier_normal_(self.module.conv.conv_module.weight)
        nn.init.xavier_normal_(self.module.conv.gated_module.weight)
        if self.module.conv.conv_module.bias is not None:
            nn.init.zeros_(self.module.conv.conv_module.bias)


class MLPLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, norm: str = "batch", dropout: float = None, **kwargs):
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
        self.linear = nn.Linear(np.prod(input_dim), np.prod(output_dim))
        if dropout is not None:
            self.dropout = dropout_hash[1](dropout)
        if norm is not None:
            self.norm = norm_hash[norm][1](output_dim)
        self.nn_lin = kwargs.get('nn_lin', DEFAULT_NNLIN)
        if self.nn_lin is not None:
            self.activation = getattr(nn, self.nn_lin)()
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
        if self.nn_lin:
            gain = calculate_gain(self.nn_lin, param=None)
        else:
            gain = 1.0
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=gain)
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
    def __init__(self, input_dim: int, output_dim: int, norm: str = "batch", dropout: float = None, **kwargs):
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
        self.linear = nn.Linear(np.prod(input_dim), np.prod(output_dim))
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
                 dim: int = 800, nn_lin: str = DEFAULT_NNLIN, norm: str = "batch",
                 dropout: float = None, layer: Type = None, **kwargs):
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
        dims = [input_dim]+checklist(dim, n=nlayers)+[output_dim]
        layers = []
        layer_class = layer or self.Layer
        if isinstance(nn_lin, list):
            assert len(nn_lin) in (nlayers, nlayers+1)
            if len(nn_lin) == nlayers:
                nn_lin.append(None)
        else:
            nn_lin = [nn_lin] * nlayers + [None]
        for i in range(len(dims)-1):
            current_norm = norm if i<len(dims)-2 else None
            layers.append(layer_class(dims[i], dims[i+1], nn_lin=nn_lin[i], norm=current_norm, dropout=dropout))
        self.module = nn.Sequential(*tuple(layers))

    def forward(self, x: torch.Tensor, mod_closure=None, trace=None) -> torch.Tensor:
        batch_size = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        out = x
        for i, mod in enumerate(self.module):
            current_mod_closure = None if mod_closure is None else mod_closure[i]
            out = self.module[i](out, mod_closure=current_mod_closure)
        out = out.view(*batch_size, out.shape[-1])
        if trace is not None: 
            trace['out'] = out
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
    def __init__(self, *args, dim=None, **kwargs):
        """RNN module. Similar to torch.nn implementation, just force the batch_first argument and
        do not return the hidden vector for global coherence."""
        super().__init__(*args, batch_first = True, **kwargs)

    def forward(self, *args, return_hidden=True, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out, h = super().forward(*args, **kwargs)
        if return_hidden:
            return out, h
        else:
            return out

