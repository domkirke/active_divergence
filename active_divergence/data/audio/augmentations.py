import torch, random, numpy as np, sys, abc
sys.path.append('../')
from active_divergence.data.audio.utils import random_phase_mangle
from typing import Tuple, Dict
from scipy.signal import lfilter

class AudioAugmentation(object):
    def __init__(self, prob: float=0.5, sr: int=None, bitrate: int=None):
        """
        A base class for audio augmentation objects.
        Args:
            prob (float): augmentation probability (default: 0.5)
            sr (int): sampling rate (default: None)
            bitrate (int): bitrate (default: None)
        """
        self.prob = prob
        self.sr = sr
        self.bitrate = bitrate

    @abc.abstractmethod
    def augment(self, x, y=None):
        raise NotImplementedError

    def __call__(self, x: torch.Tensor, y: dict) -> Tuple[torch.Tensor, Dict]:
        """
        AudioAugmentation calls callback "augment" with the given probability.
        Args:
            x (torch.Tensor): incoming tensor
            y (dict): incoming metadata

        Returns:

        """
        if random.random() < self.prob:
            return self.augment(x, y=y)
        else:
            return x, y


# RAW WAVEFORM AUGMENTATIONS
class RandomPhase(AudioAugmentation):
    def __init__(self, fmin: float=20., fmax: float=2000., amp: float=.99, **kwargs):
        """
        Randomize the phase of an incoming raw audio chunk.
        Args:
            fmin(float): minimum frequency (default: 20)
            fmax (float): maximum frequency (default: 2000)
            amp (float): noise amplitude (default: 0.99)
            **kwargs:
        """
        super(RandomPhase, self).__init__(**kwargs)
        self.fmin = fmin
        self.fmax = fmax
        self.amp = amp

    def augment(self, x: torch.Tensor, y=None, **kwargs) -> Tuple[torch.Tensor, Dict]:
        x = random_phase_mangle(x, self.fmin, self.fmax, self.amp, self.sr)
        return x, y


class Dequantize(AudioAugmentation):
    """from RAVE, Antoine Caillon https://github.com/caillonantoine/RAVE"""
    def __init__(self, prob: float=1.0, **kwargs):
        """
        Dequantize incoming raw audio chunk.
        Args:
            prob: dequantization probability (default: 1.0)
            **kwargs: AudioAugmentation keywords
        """
        super(Dequantize, self).__init__(prob=prob, **kwargs)

    def augment(self, x: torch.Tensor, y: dict=None) -> Tuple[torch.Tensor, Dict]:
        x += np.random.rand(len(x)) / 2**self.bitrate
        return x, y



class Shift(AudioAugmentation):
    def __init__(self, roll_range=[-2, 2], prob=0.5):
        super(Shift, self).__init__(prob=prob)
        self.roll_range = roll_range
        self.prob = prob

    def augment(self, x, **kwargs):
        if isinstance(x, list):
            return [self(x_tmp, **kwargs) for x_tmp in x]
        shift = torch.randint(self.roll_range[0], self.roll_range[1], (1,)).item()
        if shift > 0:
            x = torch.cat([torch.zeros((*x.shape[:-1], shift)), x[..., :-shift]], -1)
        elif shift < 0:
            x = torch.cat([x[..., abs(shift):], torch.zeros((*x.shape[:-1], abs(shift)))], -1)
        if kwargs.get("y") is None:
            return x
        else:
            return x, kwargs.get('y')



class Amplitude(AudioAugmentation):
    def __init__(self, amp_range=0.05, mode="gaussian", batch=False, prob=0.5):
        super(Amplitude, self).__init__(prob=prob)
        self.amp_range = amp_range
        self.mode = mode
        self.batch = batch

    def augment(self, x, **kwargs):
        if isinstance(x, list):
            return [self(x_tmp, **kwargs) for x_tmp in x]
        if not self.batch:
            x = x.unsqueeze(0)
        if self.mode == "gaussian":
            gain = self.amp_range * torch.randn(x.shape[0], x.shape[-1])
            gain = gain.view((x.shape[0], *(1,) * (len(x.shape) - 2), x.shape[-1])).expand_as(x)
            x = x + gain
        elif self.mode == "constant":
            gain = self.amp_range * torch.randn(x.shape[0],)
            gain = gain.view(x.shape[0], *(1,) * (len(x.shape) - 1)).expand_as(x)
            x = x + gain
        elif self.mode == "prop":
            gain = self.amp_range * torch.randn(x.shape[0], x.shape[-1])
            gain = gain.view((x.shape[0], *(1,) * (len(x.shape) - 2), x.shape[-1])).expand_as(x) + 1
            x = x * gain
        elif self.mode == "prop_constant":
            gain = self.amp_range * torch.randn(x.shape[0],)
            gain = gain.view(x.shape[0], *(1,) * (len(x.shape) - 1)).expand_as(x) + 1
            x = x * gain
        if not self.batch:
            x = x[0]
        if kwargs.get("y") is None:
            return x
        else:
            return x, kwargs.get('y')

