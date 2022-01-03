import torch, torch.nn as nn, pdb
from math import log, floor, ceil
from omegaconf import OmegaConf
from active_divergence.data.audio import parse_transforms, transforms as ad_transforms
from active_divergence.modules.activations import ScaledSoftSign
from active_divergence.modules import layers
from active_divergence.utils import checklist


class MCNN(nn.Module):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__()
        self.input_size = config.input_dim
        assert isinstance(self.input_size, int)
        n_modules = floor(log(self.input_size) / log(2))
        self.n_modules = int(n_modules)

        self.channels = [self.input_size] + [2**(i) for i in reversed(range(self.n_modules))]
        self.stride = [2] * int(self.n_modules)
        self.kernel_size = checklist(config.get('kernel_size', 13), n=self.n_modules)
        self.padding = [ceil(ks / 2) for ks in self.kernel_size]
        self.bias = checklist(config.get('bias', False), n=self.n_modules)
        self.nnlin = checklist(config.get('nnlin', 'ELU'), n=self.n_modules)
        self.n_heads = config.get('n_heads', 8)
        self.layer = checklist(config.get('layers', 'DeconvLayer'), n=self.n_modules)
        self.init_modules()

    def init_modules(self):
        heads = []
        for n in range(self.n_heads):
            modules = []
            for i in range(self.n_modules): 
                layer = getattr(layers, self.layer[n])
                current_module = layer([self.channels[i], self.channels[i+1]],
                                        kernel_size=self.kernel_size[n],
                                        stride = self.stride[n],
                                        bias = self.bias[n],
                                        padding = self.padding[n],
                                        nn_lin = self.nnlin[n], 
                                        dim = 1)
                modules.append(current_module)
            current_head = nn.Sequential(*modules)
            heads.append(current_head)
        self.heads = nn.ModuleList(heads)
        self.head_weights = nn.Parameter(torch.full((self.n_heads,1, 1, 1),  1.))
        self.scaled_softsign = ScaledSoftSign()

    def forward(self, x) -> torch.Tensor:
        x_t = x.transpose(-1, -2)
        weights = nn.functional.softmax(self.head_weights, dim=-1)
        out_heads = []
        for i, h in enumerate(self.heads):
            out_heads.append(h(x_t))
        out = torch.stack(out_heads)
        out = self.scaled_softsign((weights * out).sum(0))
        return out.squeeze(-2)


        




