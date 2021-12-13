import torch, random, numpy as np, sys
sys.path.append('../')
from active_divergence.data.audio.utils import random_phase_mangle
from scipy.signal import lfilter

class AudioAugmentation(object):
    def __init__(self, prob=0.5, sr=None, bitrate=None):
        self.prob = prob
        self.sr = sr
        self.bitrate = bitrate

    def augment(self, x, y=None):
        raise NotImplementedError

    def __call__(self, x: torch.Tensor, y: dict):
        if random.random() < self.prob:
            return self.augment(x, y)
        else:
            return x, y


# RAW WAVEFORM AUGMENTATIONS
class RandomPhase(AudioAugmentation):
    def __init__(self, fmin=20, fmax=2000, amp=.99, **kwargs):
        super(RandomPhase, self).__init__(**kwargs)
        self.fmin = fmin
        self.fmax = fmax
        self.amp = amp

    def augment(self, x: torch.Tensor, y=None, **kwargs):
        x = random_phase_mangle(x, self.fmin, self.fmax, self.amp, self.sr)
        return x, y


class Dequantize(AudioAugmentation):
    """from RAVE, Antoine Caillon https://github.com/caillonantoine/RAVE"""
    def __init__(self, prob=1.0, **kwargs):
        super(Dequantize, self).__init__(prob=prob, **kwargs)

    def augment(self, x, y=None):
        x += np.random.rand(len(x)) / 2**self.bitrate
        return x, y




