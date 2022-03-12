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
            return self.augment(x, y)
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




