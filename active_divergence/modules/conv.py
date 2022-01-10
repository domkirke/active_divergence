import torch, torch.nn as nn, pdb
from typing import Optional, List
from torch import Tensor, ones
from torch.nn import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d 


### weighted convolutions
### Convolution Modules

class WeightedConv1d(Conv1d):
    def __init__(self, *args, **kwargs):
        super(WeightedConv1d, self).__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.tensor(1.))

    def _get_normalized_weight(self):
        return self.scale * self.weight / self.weight.norm()

    def forward(self, input: Tensor) -> Tensor:
        weight = self._get_normalized_weight(self.weight)
        return self._conv_forward(input, weight, self.bias)

class WeightedConv2d(Conv2d):
    def __init__(self, *args, **kwargs):
        super(WeightedConv1d, self).__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.tensor(1.))

    def _get_normalized_weight(self):
        return self.scale * self.weight / self.weight.norm()

    def forward(self, input: Tensor) -> Tensor:
        weight = self._get_normalized_weight(self.weight)
        return self._conv_forward(input, weight, self.bias)

class WeightedConv3d(Conv3d):
    def __init__(self, *args, **kwargs):
        super(WeightedConv3d, self).__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.tensor(1.))

    def _get_normalized_weight(self):
        return self.scale * self.weight / self.weight.norm()

    def forward(self, input: Tensor) -> Tensor:
        weight = self._get_normalized_weight(self.weight)
        return self._conv_forward(input, weight, self.bias)


class WeightedConvTranspose1d(ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(WeightedConvTranspose1d, self).__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.tensor(1.))

    def _get_normalized_weight(self):
        return self.scale * self.weight / self.weight.norm()

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')
        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]
        weight = self._get_normalized_weight()
        return nn.functional.conv_transpose1d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
            
class WeightedConvTranspose2d(ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super(WeightedConvTranspose2d, self).__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.tensor(1.))

    def _get_normalized_weight(self):
        return self.scale * self.weight / self.weight.norm()

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')
        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]
        weight = self._get_normalized_weight()
        return nn.functional.conv_transpose2d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

class WeightedConvTranspose3d(ConvTranspose3d):
    def __init__(self, *args, **kwargs):
        super(WeightedConvTranspose3d, self).__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.tensor(1.))

    def _get_normalized_weight(self):
        return self.scale * self.weight / self.weight.norm()

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')
        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]
        weight = self._get_normalized_weight()
        return nn.functional.conv_transpose3d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


## Modulated convolutions (StyleGAN2-like)

eps = 1e-9
class ModConvTranspose2d(ConvTranspose2d):
    def __init__(self, *args, mod_input=None, noise_amp=1.0, **kwargs):
        super(ModConvTranspose2d, self).__init__(*args, **kwargs)
        in_channels = mod_input or kwargs.get('in_channels')
        self.affine_module = nn.Linear(in_channels, kwargs['in_channels'])
        self.noise_weight = nn.Parameter(torch.ones(1, kwargs["out_channels"], 1, 1))
        self.noise_amp = noise_amp
    
    @property
    def std_dims(self):
        return (1, 3, 4)

    def _expand_mod(self, mod):
        return mod.view(mod.shape[0], mod.shape[1], 1, 1, 1)
    def _expand_demod(self, demod):
        return demod.view(demod.shape[0], 1, demod.shape[1], 1, 1)

    def _get_modulated_weight(self, input, mod):
        if mod is None:
            mod = torch.zeros(input.shape[0], self.affine_module.weight.shape[1]).to(input.device)
        mod = self.affine_module(mod)
        mod = self.weight.unsqueeze(0) * self._expand_mod(mod)
        mod_std = self._expand_demod((mod.pow(2).sum(self.std_dims) + eps).sqrt())
        return mod / mod_std

    def _get_bias(self, out, noise):
        if noise is not None:
            bias = self.noise_weight * noise
        else:
            bias = self.noise_amp * self.noise_weight * torch.randn(out.shape[0], self.weight.shape[1], 1, 1).to(out.device)
        if self.bias is not None:
            bias = bias + self.bias.unsqueeze(0)
        return bias

    def forward(self, input: Tensor, mod: Tensor = None, noise: Tensor=None, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')
        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]
        weight = self._get_modulated_weight(input, mod)
        #weight = self._get_demodulated_weight(weight)
        weight = weight.view(weight.shape[0]*weight.shape[1], *weight.shape[2:])
        input_r = input.view(1, input.shape[0]*input.shape[1], *input.shape[2:])
        out = nn.functional.conv_transpose2d(input_r, weight, self.bias, self.stride, self.padding, output_padding, input.shape[0], self.dilation)
        out = out.view(input.shape[0], weight.shape[1], *out.shape[2:])
        # apply bias
        bias = self._get_bias(out, noise)
        out = out + bias
        return out