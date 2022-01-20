from collections import namedtuple
import sys, pdb
sys.path.append('../')
import torch, torch.nn as nn, numpy as np
from active_divergence.utils import checklist, checktuple, print_stats, checkdist, parse_slice
from omegaconf import OmegaConf
import torch.distributions as dist
from active_divergence.modules import conv, layers as layers, mlp_dist_hash, conv_dist_hash, Reshape
from typing import Tuple, Union, Iterable, List


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
        self.weight_norm = config.get('weight_norm', False)
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
            raise TypeError('target shape of %s module must be int / iterable ints'%(type(self)))
        self.layer = self.Layer if config.get('layer') is None else getattr(layers, config.layer)
        self._init_modules()

    def _init_modules(self):
        input_size = self.input_size
        if isinstance(input_size, torch.Size):
            input_size = np.cumprod(list(input_size))[-1]
        target_shape = np.cumprod(list(self.target_shape))[-1]
        target_shape = target_shape if self.target_dist not in [dist.Normal] else target_shape * 2
        self.mlp = self.Layer(input_size, target_shape, nlayers=self.nlayers, hidden_dims=self.hidden_dims,
                              nn_lin=self.nn_lin, norm=self.norm, dropout=self.dropout, weight_norm=self.weight_norm)

    def forward(self, x: torch.Tensor, return_hidden=False, **kwargs) -> Union[torch.Tensor, dist.Distribution]:
        """
        Encodes incoming tensor.
        Args:
            x (torch.Tensor): data to encode

        Returns:
            y (torch.Tensor or Distribution): encoded data.
        """
        batch_shape = x.shape[:-len(self.input_size)]
        hidden = self.mlp(x, return_hidden=return_hidden)
        if return_hidden:
            hidden, hidden_history = hidden
        if self.target_dist == dist.Normal:
            out = list(hidden.split(hidden.shape[-1]//2, -1))
            if self.target_shape:
                out[0] = out[0].reshape(*batch_shape, *checktuple(self.target_shape))
                out[1] = torch.clamp(out[1].reshape(*batch_shape, *checktuple(self.target_shape)), -5)
            if not self.out_nnlin is None:
                out[0] = self.out_nnlin(out[0])
            out = dist.Normal(out[0], torch.exp(out[1]))
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
            if self.out_nnlin is not None:
                out = self.out_nnlin(out)
        if return_hidden:
            return out, hidden_history
        else:
            return out


class ConvEncoder(nn.Module):
    Layer = "ConvLayer"
    Flatten = "MLP"
    available_modes = ['forward', 'forward+', 'skip', 'residual']

    def __len__(self):
        return len(self.conv_modules) 

    def __getitem__(self, item):
        return self.get_submodule(item)

    def __init__(self, config, init_modules=True):
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
        self.input_size = config.get('input_dim')
        self.mode = config.get('mode', "forward")
        self.channels = checklist(config.channels)
        self.n_layers = len(self.channels) - 1
        self.kernel_size = checklist(config.get('kernel_size', 7), n=self.n_layers)
        self.dilation = checklist(config.get('dilation', 1), n=self.n_layers)
        self.padding = checklist(config.get('padding'), n=self.n_layers)
        self.dropout = checklist(config.get('dropout'), n=self.n_layers)
        self.stride = checklist(config.get('stride',1), n=self.n_layers)
        self.dim = config.get('dim', len(config.get('input_dim', [None]*3)) - 1)
        self.nn_lin = checklist(config.get('nn_lin'), n=self.n_layers)
        
        self.Layer = checklist(config.get('layer', self.Layer), n=self.n_layers)
        self.config_flatten = dict(config.get('flatten_args', {}))
        self.flatten_type = getattr(layers, config.get('flatten_module', self.Flatten))
        self.norm = checklist(config.get('norm'), n=self.n_layers)
        self.bias = config.get('bias', True)
        self.block_args = checklist(config.get('block_args', {}), n=self.n_layers)

        # flattening parameters
        self.target_shape = config.get('target_shape')
        self.target_dist = config.get('target_dist')
        self.reshape_method = config.get('reshape_method', "flatten")

        # distribution parameters
        if init_modules:
            self.out_nnlin = config.get('out_nnlin')
            if self.out_nnlin is not None:
                self.out_nnlin = getattr(nn, self.out_nnlin)()
        if self.target_dist:
            self.target_dist = checkdist(self.target_dist)
            if init_modules:
                self.dist_module = mlp_dist_hash[self.target_dist](out_nnlin=self.out_nnlin)
            self.channels[-1] *= self.dist_module.required_dim_upsampling
        else:
            if init_modules:
                self.dist_module = self.out_nnlin if self.out_nnlin is not None else None

        # init modules
        self.aggregate = config.get('aggregate')
        if init_modules:
            self._init_modules()

    def _init_conv_modules(self):
        modules = []
        if self.mode in ["forward"]:
            self.pre_conv = layers.conv_hash['conv'][self.dim](self.input_size[0], self.channels[0], 1)
        elif self.mode in ["forward+", "residual", "skip"]:
            self.pre_conv = nn.ModuleList([layers.conv_hash['conv'][self.dim](self.input_size[0], c, 1) for c in self.channels[:-1]])
        for n in range(self.n_layers):
            Layer = getattr(layers, self.Layer[n])
            if n > 0 and self.mode == "skip":
                in_channels, out_channels = self.channels[n], self.channels[n+1]
            else:
                in_channels, out_channels = self.channels[n], self.channels[n+1]
            current_layer = Layer([in_channels, out_channels],
                                      kernel_size=self.kernel_size[n],
                                      dilation=self.dilation[n],
                                      padding=self.padding[n],
                                      dim=self.dim,
                                      stride=self.stride[n],
                                      norm=self.norm[n],
                                      dropout=self.dropout[n],
                                      bias=self.bias,
                                      nn_lin=self.nn_lin[n], 
                                      **self.block_args[n])
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
                self.flatten_module = nn.Sequential(Reshape(flatten_shape, incoming_dim=self.dim+1), self.flatten_type(flatten_shape, target_shape, **self.config_flatten))
            elif self.reshape_method == "reshape":
                self.flatten_module = Reshape(self.channels[-1])

    def _init_modules(self):
        self._init_conv_modules()
        self._init_flattening_modules()


    def forward(self, x: torch.Tensor, return_hidden=False, transition=None, **kwargs) -> Union[torch.Tensor, dist.Distribution]:
        """
        Encodes incoming tensor.
        Args:
            x (torch.Tensor): data to encode
            return_hidden (bool): return intermediate hidden vectors
            transition (float): transition factor for progressive learning

        Returns:
            y (torch.Tensor or Distribution): encoded data.
        """
        batch_shape = x.shape[:-(self.dim + 1)]
        out = x.reshape(-1, *x.shape[-(self.dim + 1):])
        out_orig = out

        # set inputs
        hidden = []
        # fill buffers
        if self.mode == "skip":
            buffer = out
        if hasattr(self, "pre_conv"):
            pre_conv = self.pre_conv if not isinstance(self.pre_conv, nn.ModuleList) else self.pre_conv[0]
            out = pre_conv(out)
        if self.mode == "residual":
            buffer = out
        # compute convs
        for i, conv_module in enumerate(self.conv_modules):
            if self.mode in ['skip']:
                if i > 0:
                    if hasattr(conv_module, "downsample"):
                        buffer = self.conv_modules[i-1].downsample(buffer)
                    out = out + self.pre_conv[i](buffer)
            out = conv_module(out)
            if i == 0 and transition:
                if hasattr(conv_module, "downsample"):
                    out_orig = conv_module.downsample(out_orig)
                out = transition * out + (1 - transition) * self.pre_conv[1](out_orig)

            if self.mode in ['residual']:
                if hasattr(conv_module, "downsample"):
                    buffer = self.conv_modules[i-1].downsample(buffer)
                buffer = out + buffer
                out = buffer
            hidden.append(out)

        out = out.view(*batch_shape, *out.shape[1:])
        if hasattr(self, "flatten_module"):
            if self.flatten_module is not None:
                out = self.flatten_module(out)
        hidden.append(out)
        if hasattr(self, "dist_module"):
            if self.dist_module is not None:
                out = self.dist_module(out)

        if return_hidden:
            return out, hidden
        else:
            return out

    def get_submodule(self, items: Union[int, List[int], range]) -> nn.Module:
        if isinstance(items, slice):
            items = list(range(len(self.conv_modules)))[items]
        elif isinstance(items, int):
            if items < 0:
                items = len(self) + items
            items = [items]
        config = OmegaConf.create()
        
        if 0 in items:
            config.input_dim = self.input_size
        if len(self.conv_modules) - 1 in items:
            config.target_shape = self.target_shape

        # set convolution parameters
        config.dim = self.dim
        config.channels = [self.channels[i] for i in items]
        config.n_layers = len(items)
        config.kernel_size = [self.kernel_size[i] for i in items]
        config.dilation = [self.dilation[i] for i in items]
        config.padding = [self.padding[i] for i in items]
        config.dropout = [self.dropout[i] for i in items]
        config.stride = [self.stride[i] for i in items]
        config.nn_lin = [self.nn_lin[i] for i in items]
        config.mode = self.mode
        
        config.norm = [self.norm[i] for i in items]
        config.bias = self.bias
        config.Layer = self.Layer
        config.block_args = [self.block_args[i] for i in items]
        module = type(self)(config, init_modules=False)

        conv_modules = []
        pre_convs = []
        for i in items:
            if i == 0:
                module.dist_module = self.dist_module
                if self.mode in ["forward"]:
                    module.pre_conv = self.pre_conv
            conv_modules.append(self.conv_modules[i])
            if self.mode in ["skip", "residual", "forward+"]:
                pre_convs.append(self.pre_conv[i])
            if i == len(self) - 1:
                module.flatten_type = self.flatten_type
                module.flatten_module = self.flatten_module
                if hasattr(self, "dist_module"):
                    module.dist_module = self.dist_module
                if hasattr(self, "out_nnlin"):
                    module.out_nnlin = self.out_nnlin
        module.conv_modules = nn.ModuleList(conv_modules)
        if self.mode in ["skip", "residual", "forward+"]:
            module.pre_conv = nn.ModuleList(pre_convs)
        return module


class MLPDecoder(MLPEncoder):
    pass


class DeconvEncoder(nn.Module):
    Layer = "DeconvLayer"
    Flatten = "MLP"
    available_modes = ['forward', 'forward+', 'residual', 'skip', 'parallel']
    def __init__(self, config: OmegaConf, encoder: nn.Module = None, init_modules=True):
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
        # set dimensionality parameters
        self.input_size = config.get('input_dim')
        self.target_shape = config.get('target_shape')
        self.out_channels = self.target_shape[0] if self.target_shape else config.get('out_channels')

        # set convolution parameters
        self.dim = config.get('dim') or len(self.target_shape or [None] * 3) - 1
        self.channels = list(reversed(config.get('channels'))) 
        self.n_layers = len(self.channels) - 1
        self.kernel_size = list(reversed(checklist(config.get('kernel_size', 7), n=self.n_layers)))
        self.dilation = list(reversed(checklist(config.get('dilation', 1), n=self.n_layers)))
        self.padding = list(reversed(checklist(config.get('padding'), n=self.n_layers)))
        self.dropout = list(reversed(checklist(config.get('dropout'), n=self.n_layers)))
        self.stride = list(reversed(checklist(config.get('stride', 1), n=self.n_layers)))
        self.output_padding = [np.zeros(self.dim)]*self.n_layers
        self.nn_lin = list(reversed(checklist(config.get('nn_lin', layers.DEFAULT_NNLIN), n=self.n_layers)))
        self.mode = config.get('mode', 'forward')
        assert self.mode in self.available_modes
        
        self.norm = list(reversed(checklist(config.get('norm'), n=self.n_layers)))
        self.bias = config.get('bias',True)
        self.Layer = checklist(config.get('layer', self.Layer), n=self.n_layers)
        self.block_args = checklist(config.get('block_args', {}), n=self.n_layers)

        # output parameters
        self.target_dist = config.get('target_dist')
        self.out_nnlin = config.get('out_nnlin')
        if self.out_nnlin is not None:
            self.out_nnlin = getattr(nn, self.out_nnlin)()
        if self.target_dist:
            self.target_dist = checkdist(self.target_dist)
            if init_modules:
                self.dist_module = conv_dist_hash[self.target_dist](out_nnlin=self.out_nnlin, dim=self.dim)
        else:
            if init_modules:
                self.dist_module = self.out_nnlin if self.out_nnlin is not None else None

        # flattening parameters
        self.reshape_method = config.get('reshape_method') or "flatten"
        self.config_flatten = dict(config.get('flatten_args', {}))
        self.flatten_type = getattr(layers, config.get('flatten_module') or self.Flatten)

        self.aggregate = config.get('aggregate')
        # init modules
        if init_modules:
            self._init_modules()

    def __len__(self):
        return len(self.conv_modules) 

    def __getitem__(self, item):
        return self.get_submodule(item)

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
        Layers = [getattr(layers, l) for l in self.Layer]
        if self.target_shape is not None:
            # retrieve output paddings
            current_shape = np.array(self.target_shape[1:])
            output_padding = [None]*self.n_layers
            for n in reversed(range(self.n_layers)):
                output_padding[n], current_shape = Layers[n].get_output_padding(current_shape, kernel_size=self.kernel_size[n],
                                                               padding=self.padding[n], dilation=self.dilation[n],
                                                               stride=self.stride[n], **self.block_args[n])
                output_padding[n] = tuple(output_padding[n].astype(np.int).tolist())
            self.output_padding = output_padding
        else:
            output_padding = self.output_padding

        for n in range(self.n_layers):
            current_layer = Layers[n]([self.channels[n], self.channels[n + 1]],
                                    kernel_size=self.kernel_size[n],
                                    dilation=self.dilation[n],
                                    padding=self.padding[n],
                                    dropout=self.dropout[n],
                                    stride=self.stride[n],
                                    norm=self.norm[n],
                                    dim=self.dim,
                                    bias = self.bias,
                                    nn_lin = self.nn_lin[n] if n<self.n_layers else None,
                                    output_padding=output_padding[n],
                                    **self.block_args[n])
            modules.append(current_layer)
        self.conv_modules = nn.ModuleList(modules)
        out_channels = self.out_channels
        if self.dist_module is not None:
            if hasattr(self.dist_module, "required_channel_upsampling"):
                out_channels *= self.dist_module.required_channel_upsampling
        if self.mode in ["forward"]:
            if len(self.kernel_size) == len(self.channels):
                self.final_conv = layers.conv_hash['conv'][self.dim](self.channels[-1], out_channels, self.kernel_size[-1], padding=int(np.floor(self.kernel_size[-1]/2)))
            else:
                self.final_conv = layers.conv_hash['conv'][self.dim](self.channels[-1], out_channels, 1)
        else:
            final_convs = []
            for n in range(self.n_layers):
                final_convs.append(layers.conv_hash['conv'][self.dim](self.channels[n], self.out_channels, 1))
            final_convs.append(layers.conv_hash['conv'][self.dim](self.channels[-1], self.out_channels, 1))
            self.final_conv = nn.ModuleList(final_convs)

    def _init_unfold_modules(self):
        # init flattening modules
        if self.target_shape is not None:
            # retrieve output paddings
            current_shape = np.array(self.target_shape[1:])
            Layers = [getattr(layers, l) for l in self.Layer]
            for n in reversed(range(self.n_layers)):
                _, current_shape = Layers[n].get_output_padding(current_shape, kernel_size=self.kernel_size[n],
                                                               padding=self.padding[n], dilation=self.dilation[n], stride=self.stride[n],
                                                               **self.block_args[n])
        self.flatten_module = None
        self.reshape_module = None
        input_shape = (self.channels[0], *([int(i) for i in current_shape]))
        if self.target_shape is not None:
            final_shape = tuple(current_shape.tolist())
            if self.reshape_method == "flatten":
                assert self.input_size, "flattening modules needs the input dimensionality."
                flatten_module = self.flatten_type(self.input_size, int(np.cumprod(input_shape)[-1]), **self.config_flatten) 
                reshape_module = Reshape(self.channels[0], *final_shape, incoming_dim=1)
                self.flatten_module = nn.Sequential(flatten_module, reshape_module)
            elif self.reshape_method == "channel":
                kernel_size = np.array(input_shape[1:])
                padding = kernel_size - 1
                flatten_module = layers.conv_hash['conv'][self.dim](self.channels[0], self.channels[0], tuple(kernel_size), padding=tuple(padding))
                reshape_module = Reshape(self.channels[0], *final_shape, incoming_dim=self.dim+1)
                self.flatten_module = nn.Sequential(flatten_module, reshape_module)
                self.input_size = (self.channels[0], *(1,)*len(final_shape))
            else:
                self.flatten_module = Reshape(self.channels[0], *final_shape, incoming_dim=self.dim)
        if self.input_size is None:
            self.input_size = input_shape
            

    def forward(self, x: torch.Tensor, mod=None, transition=None, **kwargs) -> Union[torch.Tensor, dist.Distribution]:
        """
        decodes an incoming tensor.
        Args:
            x (torch.Tensor): incoming tensor
            seq_shape (int, optional): number of decoded elements (if recurrent flattening module)

        Returns:
            out (torch.Tensor): decoded output
        """
        batch_shape = x.shape[:-len(checktuple(self.input_size))]
        # process flattening
        if self.flatten_module is not None:
            out = self.flatten_module(x)
        out = out.reshape(-1, *out.shape[-(self.dim+1):])

        # process convolutions
        if self.mode == "skip":
            buffer = None
        elif self.mode == "residual":
            buffer = out
        for i, conv_module in enumerate(self.conv_modules):
            # write buffer before convolution for skip connections
            if self.mode == "skip":
                out_side = self.final_conv[i](out)
                buffer = out_side if buffer is None else buffer + out_side
                if hasattr(conv_module, "upsample"):
                    buffer = conv_module.upsample(buffer)
            # keep last output for transition
            last_out = out if self.mode == "skip" else out
            # perform cocnv
            if mod is not None:
                if isinstance(mod, (list, tuple)):
                    out = conv_module(out, mod=mod[i])
                else:
                    out = conv_module(out, mod=mod)
            else:
                out = conv_module(out)
            # add to previous outputs for residual connections
            if self.mode == "residual":
                if hasattr(conv_module, "upsample"):
                    buffer = conv_module.upsample(buffer)
                out = out + buffer
                buffer = out
        # if transitioning, past previous output in last layer upsampling
        if transition and isinstance(self.conv_modules, nn.ModuleList):
            if hasattr(self.conv_modules[-1], "upsample"):
                last_out = self.conv_modules[-1].upsample(last_out)

        # process output
        if hasattr(self, "final_conv"):
            final_conv = self.final_conv if not isinstance(self.final_conv, nn.ModuleList) else self.final_conv[-1]
            if self.mode in ['skip']:
                out = final_conv(out)
                if transition and isinstance(self.final_conv, nn.ModuleList):
                    if len(self.final_conv) >= 2:
                        out = buffer + (1-transition) * self.final_conv[-2](last_out) + transition * out
            else:
                out = final_conv(out)
                if transition and isinstance(self.final_conv, nn.ModuleList):
                    if len(self.final_conv) >= 2:
                        out = (1-transition) * self.final_conv[-2](last_out) + transition * out

        out = out.view(*batch_shape, *out.shape[-(self.dim+1):])
        if hasattr(self, "dist_module"):
            if self.dist_module is not None:
                out = self.dist_module(out)
        return out
    
    def get_submodule(self, items: Union[int, List[int], range]) -> nn.Module:
        if isinstance(items, slice):
            items = list(range(len(self.conv_modules)))[items]
        elif isinstance(items, int):
            if items < 0:
                items = len(self) + items
            items = [items]
        config = OmegaConf.create()
        
        if 0 in items:
            config.input_dim = self.input_size
        if len(self.conv_modules) - 1 in items:
            config.target_shape = self.target_shape
            config.out_channels = self.out_channels

        # set convolution parameters
        config.dim = self.dim
        config.channels = [self.channels[i] for i in items]
        config.n_layers = len(items)
        config.kernel_size = [self.kernel_size[i] for i in items]
        config.dilation = [self.dilation[i] for i in items]
        config.padding = [self.padding[i] for i in items]
        config.dropout = [self.dropout[i] for i in items]
        config.stride = [self.stride[i] for i in items]
        config.output_padding = [self.output_padding[i] for i in items]
        config.nn_lin = [self.nn_lin[i] for i in items]
        config.mode = self.mode
        
        config.norm = [self.norm[i] for i in items]
        config.bias = self.bias
        config.Layer = self.Layer
        config.block_args = [self.block_args[i] for i in items]
        module = type(self)(config, init_modules=False)

        conv_modules = []
        final_convs = []
        for i in items:
            if i == 0:
                module.flatten_type = self.flatten_type
                module.flatten_module = self.flatten_module
            conv_modules.append(self.conv_modules[i])
            if self.mode in ["skip", "residual", "forward+"]:
                final_convs.append(self.final_conv[i])
            if i == len(self) - 1:
                if self.mode in ["forward", "residual"]:
                    module.final_conv = self.final_conv
                if hasattr(self, "dist_module") and self.mode in ['forward']:
                    module.dist_module = self.dist_module

        module.conv_modules = nn.ModuleList(conv_modules)
        if self.mode in ["skip", "forward+", "residual"]:
            final_convs.append(self.final_conv[i+1])
            module.final_conv = nn.ModuleList(final_convs)
            if hasattr(self, 'dist_module'):
                module.dist_module = self.dist_module
        return module
            