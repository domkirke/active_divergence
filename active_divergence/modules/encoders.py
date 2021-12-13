import sys, pdb
sys.path.append('../')
import torch, torch.nn as nn, numpy as np
from active_divergence.utils import checklist, checktuple, print_stats, checkdist
from omegaconf import OmegaConf
import torch.distributions as dist
from active_divergence.modules import layers as layers, Reshape
from typing import Tuple, Union


class MLPEncoder(nn.Module):
    Layer = layers.MLP
    available_distributions = [dist.Bernoulli, dist.Categorical, dist.Normal]
    def __init__(self, config: OmegaConf, **kwargs):
        """
        Feed-forward encoder for auto-encoding architectures. OmegaConfuration may include:
        input_dim : input dimensionality
        nlayers : number of layers (default: 3)
        hidden_dims: hidden dimensions (default: 800)
        nn_lin: non-linearity (default : SiLU)
        norm: normalization ("batch" for batch norm)
        target_shape: target shape of encoder
        target_dist:  target distribution of encoder
        Args:
            config (OmegaConf): encoder configuration.
        """
        super(MLPEncoder, self).__init__()
        self.input_size = checktuple(config.input_dim)
        self.nlayers = config.get('nlayers', 3)
        self.hidden_dims = config.get('hidden_dims', 800)
        self.nn_lin = checklist(config.get('nn_lin') or layers.DEFAULT_NNLIN, n=self.nlayers)
        self.nn_lin.append(None)
        self.out_nnlin = None if config.get('out_nnlin') is None else getattr(nn, config.get('out_nnlin'))()
        self.norm = config.get('norm')
        self.dropout = config.get('dropout')
        self.target_dist = config.get('target_dist')
        if self.target_dist is not None:
            self.target_dist = checkdist(self.target_dist)
            if self.target_dist not in self.available_distributions:
                return NotImplementedError('MLPEncoder does not support the distribution %s'%self.target_dist)
        else:
            self.target_dist = None
        self.target_shape = config.get('target_shape')
        if hasattr(self.target_shape, "__iter__"):
            self.target_shape = tuple([d for d in self.target_shape])
        elif isinstance(self.target_shape, int):
            self.target_shape = (config.target_shape,)
        else:
            raise TypeError('target shape of %d module must be int / iterable ints'%(type(self)))
        self.layer = self.Layer if config.get('layer') is None else getattr(layers, config.layer)
        self._init_modules()

    def _init_modules(self):
        input_size = self.input_size
        if isinstance(input_size, torch.Size):
            input_size = np.cumprod(list(input_size))[-1]
        target_shape = np.cumprod(list(self.target_shape))[-1]
        target_shape = target_shape if self.target_dist not in [dist.Normal] else target_shape * 2
        self.mlp = self.Layer(input_size, target_shape, nlayers=self.nlayers, hidden_dims=self.hidden_dims,
                              nn_lin=self.nn_lin, norm=self.norm, dropout=self.dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, dist.Distribution]:
        """
        Encodes incoming tensor.
        Args:
            x (torch.Tensor): data to encode

        Returns:
            y (torch.Tensor or Distribution): encoded data.
        """
        batch_shape = x.shape[:-len(self.input_size)]
        hidden = self.mlp(x.reshape(-1, np.prod(self.input_size)))
        if self.target_dist == dist.Normal:
            out = list(hidden.split(hidden.shape[-1]//2, -1))
            if self.target_shape:
                out[0] = out[0].reshape(*batch_shape, *checktuple(self.target_shape))
                out[1] = torch.clamp(out[1].reshape(*batch_shape, *checktuple(self.target_shape)), -5)
            if not self.out_nnlin is None:
                out[0] = self.out_nnlin(out[0])
            out = dist.Normal(out[0], torch.exp(out[1]).sqrt())
        elif self.target_dist in [dist.Bernoulli]:
            if self.target_shape:
                hidden = hidden.reshape(*batch_shape, *checktuple(self.target_shape))
            out = self.target_dist(probs=torch.sigmoid(hidden))
        elif self.target_dist in [dist.Categorical]:
            if self.target_shape:
                hidden = hidden.reshape(*batch_shape, *checktuple(self.target_shape))
            out = self.target_dist(probs=torch.softmax(hidden))
        else:
            out = hidden.reshape(*batch_shape, *checktuple(self.target_shape))
        return out


class ConvEncoder(nn.Module):
    Layer = layers.ConvLayer
    Flatten = "MLP"
    def __init__(self, config):
        """
        Convolutional encoder for auto-encoding architectures. OmegaConfuration may include:
        input_dim (Iterable[int]): input dimensionality
        layer (type): convolutional layer (ConvLayer, GatedConvLayer)
        channels (Iterable[int]): sequence of channels
        kernel_size (int, Iterable[int]): sequence of kernel sizes (default: 7)
        dilation (int, Iterable[int]): sequence of dilations (default: 1)
        stride (int, Interable[int]): sequence of strides (default: 1)
        bias (bool): convolution with bias
        dim (int): input dimensionality (1, 2, 3)
        hidden_dims (int, Iterable[int]): hidden dimensions
        nn_lin (str): non-linearity (default : SiLU)
        norm (str): normalization
        target_shape (Iterable[int]): target shape of encoder
        target_dist: (type) target distribution of encoder
        reshape_method (str): how is convolutional output reshaped (default: 'flatten')
        flatten_args (OmegaConf): keyword arguments for flatten module

        Args:
            config (OmegaConf): encoder configuration.
        """
        super(ConvEncoder, self).__init__()
        # convolutional parameters
        self.input_size = config.input_dim
        if self.input_size is None:
            print('[Warning] : input dimension not known, may lead to degenerated dimensions')
        self.channels = ([config.input_dim[0]] or [1]) + checklist(config.channels)
        self.n_layers = len(self.channels) - 1
        self.kernel_size = checklist(config.get('kernel_size', 7), n=self.n_layers)
        self.dilation = checklist(config.get('dilation', 1), n=self.n_layers)
        self.padding = checklist(config.get('padding'), n=self.n_layers)
        self.dropout = checklist(config.get('dropout'), n=self.n_layers)
        self.stride = checklist(config.get('stride',1), n=self.n_layers)
        self.dim = config.get('dim', len(config.get('input_dim', [None]*3)) - 1)
        self.nn_lin = checklist(config.get('nn_lin'), n=self.n_layers)
        if config.get('layer') is not None:
            self.Layer = getattr(layers, config.layer)
        self.config_flatten = dict(config.get('flatten_args', {}))
        self.flatten_type = getattr(layers, config.get('flatten_module', self.Flatten))
        self.norm = checklist(config.get('norm'), n=self.n_layers)
        self.bias = config.get('bias', True)


        # flattening parameters
        self.target_shape = config.get('target_shape')
        self.target_dist = config.get('target_dist')
        if self.target_dist:
            self.target_dist = checkdist(self.target_dist)
        self.reshape_method = config.get('reshape_method', "flatten")
        if self.target_dist and not self.target_shape:
            if isinstance(self.target_dist, dist.Normal):
                self.channels[-1] *= 2

        # init modules
        # self.Flatten = self.Flatten if config.flatten is None else getattr(layers, config.flatten)
        self.aggregate = config.get('aggregate')
        self._init_modules()

    def _init_conv_modules(self):
        modules = []
        for n in range(self.n_layers):
            current_layer = self.Layer([self.channels[n], self.channels[n+1]],
                                      kernel_size=self.kernel_size[n],
                                      dilation=self.dilation[n],
                                      padding=self.padding[n],
                                      dim=self.dim,
                                      stride=self.stride[n],
                                      norm=self.norm[n],
                                      dropout=self.dropout[n],
                                      bias=self.bias,
                                      nn_lin=self.nn_lin[n])
            modules.append(current_layer)
        self.conv_modules = nn.ModuleList(modules)

    def _init_flattening_modules(self):
        self.flatten_module = None
        if self.target_shape is not None:
            assert self.input_size is not None, "if target_shape of ConvEncoder is not None, input_size must not be None"
            current_shape = np.array(self.input_size[1:])
            for c in self.conv_modules:
                current_shape = c.output_shape(current_shape)
            target_shape = int(self.target_shape)
            if self.target_dist == dist.Normal:
                target_shape *= 2
            if self.reshape_method == "flatten":
                flatten_shape = self.channels[-1] * int(np.cumprod(current_shape)[-1])
                self.flatten_module = self.flatten_type(flatten_shape, target_shape, **self.config_flatten)
            elif self.reshape_method == "reshape":
                self.flatten_module = Reshape(self.channels[-1])

    def _init_modules(self):
        self._init_conv_modules()
        self._init_flattening_modules()


    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, dist.Distribution]:
        """
        Encodes incoming tensor.
        Args:
            x (torch.Tensor): data to encode

        Returns:
            y (torch.Tensor or Distribution): encoded data.
        """
        hidden = []
        batch_shape = x.shape[:-(self.dim + 1)]
        x = x.reshape(-1, *x.shape[-(self.dim + 1):])
        out = x
        for conv_module in self.conv_modules:
            out = conv_module(out)
            hidden.append(out)
        out = out.view(*batch_shape, *out.shape[1:])

        if self.flatten_module is not None:
            if self.reshape_method == "flatten":
                out = self.flatten_module(out.view(*batch_shape, -1))
            elif self.reshape_method == "reshape":
                out = self.flatten_module(out, batch_shape = batch_shape)
            if self.aggregate == "last":
                out = out[:, -1]
            elif self.aggregate == "mean":
                out = out.mean(1)
            if self.target_dist == dist.Normal:
                mu, std = out.split(out.shape[-1]//2, dim=-1)
                out = dist.Normal(mu, torch.sigmoid(std))
        else:
            if self.target_dist == dist.Normal:
                mu, std = out.split(out.shape[1]//2, dim=1)
                out = dist.Normal(mu, torch.sigmoid(std))

        return out



class MLPDecoder(MLPEncoder):
    pass

class DeconvEncoder(nn.Module):
    Layer = layers.DeconvLayer
    Flatten = "MLP"
    def __init__(self, config: OmegaConf, encoder: nn.Module = None):
        """
        Convolutional encoder for auto-encoding architectures. OmegaConfuration may include:
        input_dim (Iterable[int]): input dimensionality
        layer (type): convolutional layer (ConvLayer, GatedConvLayer)
        channels (Iterable[int]): sequence of channels
        kernel_size (int, Iterable[int]): sequence of kernel sizes (default: 7)
        dilation (int, Iterable[int]): sequence of dilations (default: 1)
        stride (int, Interable[int]): sequence of strides (default: 1)
        dim (int): input dimensionality (1, 2, 3)
        hidden_dims (int, Iterable[int]): hidden dimensions
        nn_lin (str): non-linearity (default : SiLU)
        norm (str): normalization
        bias (bool): convolutional bias
        target_shape (Iterable[int]): target shape of encoder
        target_dist: (type) target distribution of encoder
        reshape_method (str): how is convolutional output reshaped (flatten or reshape, default: 'flatten')
        flatten_args (OmegaConf): keyword arguments for flatten module

        Args:
            config (OmegaConf): decoder configuration.
            encoder (nn.Module): corresponding encoder
        """
        super(DeconvEncoder, self).__init__()
        # access to encoder may be useful for skip-connection / pooling operations
        if encoder is not None:
            self.__dict__['encoder'] = encoder
        # convolutional parameters
        self.input_size = config.get('input_dim')
        if self.input_size is None:
            print('[Warning] : input dimension not known, may lead to degenerated dimensions')
        self.target_shape = config.get('target_shape')
        if self.target_shape:
            self.out_channels = self.target_shape[0]
        else:
            self.out_channels = config.get('out_channels')
        self.target_dist = config.get('target_dist')
        if self.target_dist:
            self.target_dist = checkdist(self.target_dist)
            self.out_channels = self.get_channels_from_dist(self.out_channels or 1)

        self.channels = list(reversed(config.get('channels'))) + checklist(self.out_channels)
        self.n_layers = len(self.channels) - 1
        self.kernel_size = list(reversed(checklist(config.get('kernel_size', 7), n=self.n_layers)))
        self.dilation = list(reversed(checklist(config.get('dilation', 1), n=self.n_layers)))
        if config.get('padding') is None:
            self.padding = [int(np.ceil(k/2)) for k in self.kernel_size]
        else:
            self.padding = list(reversed(checklist(config.get('padding'), n=self.n_layers)))
        self.dropout = list(reversed(checklist(config.get('dropout'), n=self.n_layers)))
        self.stride = list(reversed(checklist(config.get('stride', 1), n=self.n_layers)))
        self.dim = config.get('dim') or len(self.target_shape or [None] * 3) - 1
        self.output_padding = [np.array(0,)]*self.n_layers
        self.nn_lin = list(reversed(checklist(config.get('nn_lin', layers.DEFAULT_NNLIN), n=self.n_layers)))
        self.out_nnlin = config.get('out_nnlin')
        if self.out_nnlin is not None:
            self.out_nnlin = getattr(nn, self.out_nnlin)()
        self.norm = list(reversed(checklist(config.get('norm'), n=self.n_layers)))
        self.bias = config.get('bias',True)
        if config.get('layer') is not None:
            self.Layer = getattr(layers, config.layer)

        # flattening parameters
        self.reshape_method = config.get('reshape_method') or "flatten"
        self.config_flatten = dict(config.get('flatten_args', {}))
        self.flatten_type = getattr(layers, config.get('flatten_module') or self.Flatten)

        self.aggregate = config.get('aggregate')
        # init modules
        self._init_modules()

    def get_channels_from_dist(self, out_channels: int, target_dist: dist.Distribution=None):
        """returns channels from output distribution."""
        target_dist = target_dist or self.target_dist
        if hasattr(target_dist, "get_nchannels_params"):
            return target_dist.get_nchannels_params()
        elif target_dist == dist.Normal:
            return out_channels * 2
        else:
            return out_channels

    def _init_modules(self):
        # init convolutional modules
        self._init_conv_modules()
        self._init_unfold_modules()

    def _init_conv_modules(self):
        modules = []
        if self.target_shape is not None:
            # retrieve output paddings
            current_shape = np.array(self.target_shape[1:])
            output_padding = [None]*self.n_layers
            for n in reversed(range(self.n_layers)):
                output_padding[n], current_shape = self.Layer.get_output_padding(current_shape, self.kernel_size[n],
                                                               self.padding[n], self.dilation[n], self.stride[n])
        else:
            output_padding = self.output_padding
        for n in range(self.n_layers):
            current_layer = self.Layer([self.channels[n], self.channels[n + 1]],
                                      kernel_size=self.kernel_size[n],
                                      dilation=self.dilation[n],
                                      padding=self.padding[n],
                                      dropout=self.dropout[n],
                                      stride=self.stride[n],
                                      norm=self.norm[n],
                                      dim=self.dim,
                                      bias = self.bias,
                                      nn_lin = self.nn_lin[n] if n<self.n_layers-1 else None,
                                      output_padding=output_padding[n])
            modules.append(current_layer)
        self.conv_modules = nn.Sequential(*tuple(modules))

    def _init_unfold_modules(self):
        # init flattening modules
        if self.target_shape is not None:
            # retrieve output paddings
            current_shape = np.array(self.target_shape[1:])
            for n in reversed(range(self.n_layers)):
                _, current_shape = self.Layer.get_output_padding(current_shape, self.kernel_size[n],
                                                               self.padding[n], self.dilation[n], self.stride[n])
        self.flatten_module = None
        if self.target_shape is not None:
            assert self.input_size is not None, "if target_shape of ConvEncoder is not None, input_size must not be None"
            if self.reshape_method == "flatten":
                final_shape = tuple(current_shape.tolist())
                flatten_shape = int(np.cumprod(current_shape)[-1]) * self.channels[0]
                self.flatten_module = self.flatten_type(self.input_size, flatten_shape, **self.config_flatten)
                self.reshape_module = Reshape(self.channels[0], *final_shape)
            elif self.reshape_method == "reshape":
                self.reshape_module = Reshape(self.channels[0], *(1,)*self.dim)

    def forward(self, x: torch.Tensor, seq_shape=None, **kwargs) -> Union[torch.Tensor, dist.Distribution]:
        """
        decodes an incoming tensor.
        Args:
            x (torch.Tensor): incoming tensor
            seq_shape (int, optional): number of decoded elements (if recurrent flattening module)

        Returns:
            out (torch.Tensor): decoded output

        """
        batch_shape = x.shape[:-1]
        if self.flatten_module is not None:
            if self.aggregate is not None:
                x = torch.stack([x]*seq_shape, 1)
                batch_shape = (*batch_shape, seq_shape)
            x = self.flatten_module(x)
            if seq_shape:
                x = self.reshape_module(x, batch_shape=batch_shape)
            else:
                x = self.reshape_module(x, batch_shape=batch_shape)
        else:
            if self.reshape_module is not None:
                x = self.reshape_module(x, batch_shape=batch_shape)
            else:
                batch_shape = x.shape[-(self.dim+1):]
        x = x.reshape(-1, *x.shape[-(self.dim+1):])
        out = self.conv_modules(x)
        out = out.view(*batch_shape, *out.shape[-(self.dim+1):])
        if issubclass(self.target_dist, dist.Normal):
            mu, std = out.split(out.shape[1] // 2, dim=1)
            if self.out_nnlin is not None:
                mu = self.out_nnlin(mu)
            out = dist.Normal(mu, std.clamp(-4).exp().sqrt())
        elif issubclass(self.target_dist, dist.Bernoulli):
            if self.out_nnlin is not None:
                out = self.out_nnlin(out)
            out = dist.Bernoulli(torch.sigmoid(out))
        elif issubclass(self.target_dist, dist.Categorical):
            if self.out_nnlin is not None:
                out = self.out_nnlin(out)
            out = dist.Categorical(probs=torch.softmax(out, dim=-1))
        return out


