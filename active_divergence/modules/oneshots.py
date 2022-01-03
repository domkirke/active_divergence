from typing import ForwardRef
from active_divergence.modules import layers
import numpy as np, torch, torch.nn as nn, pdb
from omegaconf import OmegaConf
from math import sqrt
from active_divergence.utils.misc import checklist
from active_divergence.modules.layers import DeconvLayer, MLP, GatedMLPLayer

# Regression models from a limited amount of data.


class TemporalEncoding(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.mode = config.get('mode', 'concat')
        self.input_size = (1,)
        self.dim = config.get('dim', 128)
        self.target_shape = self.dim
        if self.mode == "concat":
            self.target_shape += 1
        self.input_conv = nn.Conv1d(1, self.dim, (1,))
        self.weights = nn.Parameter(torch.rand(1, self.dim, 1))
        self.bias = nn.Parameter(torch.rand(1, self.dim, 1))
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-2)
        out = self.input_conv(x) * self.weights + self.bias
        out_sin, out_cos = out.split(out.shape[-2]//2, -2)
        out_sin = torch.sin(out_sin)
        out_cos = torch.cos(out_cos)
        if self.mode == "concat":
            out = torch.cat([x, out_sin, out_cos], dim=-2)
        elif self.mode == "residual":
            out = x.expand_as(out) + torch.cat([out_sin, out_cos], dim=-2)
        else:
            out = torch.cat([out_sin, out_cos], dim=-2)
        return out

class ScaledSin(nn.Module):
    def __init__(self, dim, bias = True):
        super().__init__()
        self.affine = nn.Linear(dim, dim, bias=bias)
        nn.init.uniform_(self.affine.weight, -sqrt(6/dim), sqrt(6/dim))
        if bias:
            nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.affine(x))

class ChanneledScaledSin(nn.Module):
    def __init__(self, channels, channel_wise=True, bias = True):
        super().__init__()
        dims = 1 if not channel_wise else channels
        self.affine = nn.Conv1d(channels, channels, (1,), bias=bias)
        nn.init.uniform_(self.affine.weight, -sqrt(6/channels), sqrt(6/channels))
        if bias:
            nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.affine(x))


class MLPRegressor(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.input_size = config.input_dim
        encoding_config = config.get('encoding')
        if encoding_config is not None:
            self.time_encoding = TemporalEncoding(dict(encoding_config))
            mlp_input_shape = encoding_config.target_shape
        else:
            mlp_input_shape = 1
        self.nlayers = config.get('nlayers', 3)
        self.nnlin = checklist(config.get('nn_lin', "ScaledSin"), n=self.nlayers)
        self.dims = checklist(config.get('dim'), n=self.nlayers)
        self.out_nnlin = config.get('out_nnlin', 'Tanh')
        self.init_modules()

    def init_modules(self):
        for i, nnlin in enumerate(self.nnlin):
            if nnlin == "ScaledSin":
                self.nnlin[i] = ScaledSin(self.dims[i])
        self.mlp = MLP(self.input_size, self.input_size, nlayers = self.nlayers, dim = self.dims, nnlin=self.nnlin, layer=layers.GatedMLPLayer)
        if self.out_nnlin:
            self.out_nnlin = getattr(nn, self.out_nnlin)()

    def forward(self, x):
        if hasattr(self, "time_encoding"):
            out = self.time_encoding(x)
            out = out.view(out.shape[0], -1)
        else:
            out = x
        out = self.mlp(out)
        if self.out_nnlin:
            out = self.out_nnlin(out)
        return out


class ConvRegressor(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        encoding_config = config.get('encoding')
        if encoding_config is not None:
            self.time_encoding = TemporalEncoding(dict(encoding_config))
        self.channels = checklist(config.get('channels'), [64, 128, 256, 128, 64, 32])
        self.kernel_size = checklist(config.get('kernel_size', 13), n=len(self.channels)+1)
        self.dilation = checklist(config.get('dilation', [1, 2, 4, 8, 4, 2, 1]), n=len(self.channels)+1)
        self.padding = [int(np.floor(ks / 2)) for ks in self.kernel_size]
        self.nn_lin= checklist(config.get('nn_lin', "ScaledSin"), n=len(self.channels)+1)
        self.dropout = checklist(config.get('dropout', 0.0), n=len(self.channels)+1)
        self.bias = checklist(config.get('bias', False), n=len(self.channels)+1)
        self.init_modules()

    def init_modules(self):
        if hasattr(self, "time_encoding"): 
            channels = [self.time_encoding.target_shape] + self.channels + [1]
        else:
            channels = [1] + self.channels + [1]
        modules = []
        for l in range(len(channels) - 1):
            current_module = DeconvLayer([channels[l], channels[l+1]], self.kernel_size[l], dilation=self.dilation[l], padding=self.padding[l], dim=1, bias=self.bias[l], nn_lin=None)
            modules.append(current_module)
            if self.nn_lin[l] is not None:
                if self.nn_lin[l] == "ScaledSin":
                    modules.append(ChanneledScaledSin(channels[l+1], channel_wise=True, bias=True))
                else:
                    modules.append(getattr(nn, self.nn_lin[l])())
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        if hasattr(self, "time_encoding"):
            out = self.time_encoding(x)
        else: 
            out = x if x.ndim == 3 else x.unsqueeze(-2)
            out = out * 10
        for m in self.module_list:
            out = m(out)
        #TODO baaaah
        oh = out.shape[-1] - x.shape[-1]
        if oh != 0:
            if oh % 2 == 0:
                out = out[..., int(oh/2):-int(oh/2)]
            else:
                out = out[..., int(oh/2):-int(oh/2+1)]
        return out[..., 0, :]


