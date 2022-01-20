import torch, torchaudio, abc, numpy as np, librosa, re, scipy, pdb, math, random
import sys; sys.path.append('../')
from active_divergence.utils import *
from scipy.fft import dct, dst, idct, idst
from mdct import mdct, mdst, imdct, imdst
from scipy.interpolate import interp1d
from nsgt import NSGT as NNSGT, LogScale, LinScale, MelScale, OctScale
try:
    from tifresi.stft import GaussTF, GaussTruncTF
    from tifresi.utils import preprocess_signal
    TIFRESI_AVAILABLE = True
except ImportError:
    TIFRESI_AVAILABLE = False
from torchaudio import transforms as ta_transforms


class NotInvertibleError(Exception):
    pass


def preprocess_signal_stft(y, M=1024, trim=False):
    """Trim and cut signal.
    
    The function ensures that the signal length is a multiple of M.
    (taken from tifresi library, but with optional trimming)
    """
    # Trimming
    if trim: 
        y, _ = librosa.effects.trim(y)

    # Preemphasis
    # y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # Padding
    left_over = np.mod(len(y), M)
    extra = M - left_over
    y = np.pad(y, (0, extra))
    assert (np.mod(len(y), M) == 0)

    return y

## MAIN CLASS

class AudioTransform(object):
    invertible = True
    needs_scaling = False
    @abc.abstractmethod
    def __init__(self, sr=44100):
        self.sr = sr

    def __repr_(self):
        return "AudioTransform()"

    def __add__(self, transform):
        if isinstance(transform, ComposeAudioTransform):
            return ComposeAudioTransform(transforms=[self] + transform.transforms)
        elif isinstance(transform, AudioTransform):
            return ComposeAudioTransform(transforms=[self, transform])
        else:
            raise TypeError('AudioTransform cannot be added to type: %s'%type(transform))

    def __call__(self, x, time=None, *args, **kwargs):
        if time is None:
            return x
        else:
            return x, time

    def scale(self, x, *args, **kwargs):
        pass

    @abc.abstractmethod
    def invert(self, x, time=None, *args, **kwargs):
        if time is None:
            return x
        else:
            return x, time

## CONTAINERS

class ComposeAudioTransform(AudioTransform):
    @property
    def invertible(self):
        return not False in [t.invertible for t in self.transforms]

    @property
    def needs_scaling(self):
        return True in [t.needs_scaling for t in self.transforms]

    def __getitem__(self, item):
        return self.transforms[item]

    def __getattr__(self, item):
        if item in self.__dict__.keys():
            return self.__dict__('item')
        else:
            values = []
            if "transforms" in self.__dict__:
                for t in self.transforms:
                    if hasattr(t, item):
                        values.append(getattr(t, item))
            if len(values) == 0:
                raise AttributeError
            else:
                if len(values) == 1:
                    return values[0]
                else:
                    return values

    def __init__(self, transforms = [], sr=44100):
        self.transforms = transforms

    def __repr__(self) -> str:
        return "ComposeAudioTransform(%s)"%[t.__repr__()+"\n" for t in self.transforms]

    def __add__(self, itm):
        if not isinstance(itm, AudioTransform):
            raise TypeError("ComposeAudioTransform can only be added to other AudioTransforms")
        if isinstance(itm, ComposeAudioTransform):
            return ComposeAudioTransform(self.transforms + itm.transforms)
        else:
            return ComposeAudioTransform(self.transforms + [itm])

    def __radd__(self, other):
        if not isinstance(other, AudioTransform):
            raise TypeError("ComposeAudioTransform can only be added to other AudioTransforms")
        if isinstance(other, ComposeAudioTransform):
            return ComposeAudioTransform(other.transforms + self.transforms)
        else:
            return ComposeAudioTransform([other] + self.transforms)

    def scale(self, x, *args, **kwargs):
        for t in self.transforms:
            t.scale(x)
            x = t(x, **kwargs)

    def __call__(self, x, *args, time=None, sr=None, **kwargs):
        if time is None:
            for t in self.transforms:
                x = t(x, sr=sr)
            return x
        else:
            for t in self.transforms:
                x, time = t(x, time=time, sr=sr)
            return x, time

    def invert(self, x, *args, time=None, sr=None, **kwargs):
        if time is None:
            for t in reversed(self.transforms):
                x = t.invert(x, time=time, sr=sr)
        else:
            for t in reversed(self.transforms):
                x, time = t.invert(x, time=time, sr=sr)
        return x


## HELPER FUNCTIONS

def apply_transform_to_list(transform, data, time=None, **kwargs):
    if time is None:
        outs = [transform(data[i], **kwargs) for i in range(len(data))]
        return outs
    else:
        outs = [transform(data[i], time=time[i], **kwargs) for i in range(len(data))]
        return [o[0] for o in outs], [o[1] for o in outs]


def apply_inver_transform_to_list(transform, data, time=None, **kwargs):
    if time is None:
        outs = [transform.invert(data[i], **kwargs) for i in range(len(data))]
        return outs
    else:
        outs = [transform.invert(data[i], time=time[i], **kwargs) for i in range(len(data))]
        return [o[0] for o in outs], [o[1] for o in outs]



class Mono(AudioTransform):
    def __init__(self, mode="mix", normalize=False, squeeze=False, dim=-2, invert_as_stereo=True, sr=44100):
        self.mode = mode
        self.normalize = normalize
        self.squeeze = squeeze
        self.dim = dim
        self.invert_as_stereo = invert_as_stereo

    def __repr__(self):
        return "Mono(mode=%s, squeeze=%s, normalize=%s, dim=%s, invert_as_stereo=%s)"%(self.mode, self.squeeze, self.normalize, self.dim, self.invert_as_stereo)

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return [self(x_tmp) for x_tmp in x]
        if x.shape[self.dim] == 2:
            if self.mode == "mix":
                x = (x.sum(self.dim) / 2)[np.newaxis]
            elif self.mode == "right":
                x = x.index_select(torch.tensor(1), self.dim).unsqueeze(self.dim)
            elif self.mode == "left":
                x = x.index_select(torch.tensor(0), self.dim).unsqueeze(self.dim)
        if self.normalize:
            x = x / x.max()
        if self.squeeze:
            x = x.squeeze(self.dim)
        if time is None:
            return x
        else:
            return x, time

    def invert(self, x, **kwargs):
        if self.squeeze:
            x = x.unsqueeze(self.dim)
        if x.shape[self.dim] == 1 and self.invert_as_stereo:
            x = torch.cat([x, x], dim=self.dim)
        return x


class Stereo(AudioTransform):
    def __init__(self, normalize=False, sr=44100):
        self.normalize = normalize

    def __repr__(self):
        return "Stereo(normalize: %s)"%self.normalize

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return [self(x_tmp) for x_tmp in x]
        if x.shape[0] == 1:
            x = torch.cat([x, x], dim=0)
        if self.normalize:
            x = x / x.max()
        if time is None:
            return x
        else:
            return x, time

    def invert(self, x, time=None, **kwargs):
        if x.shape[0] == 1:
            x = torch.cat([x, x], dim=0)
        if time is None:
            return x
        else:
            return x, time


class Window(AudioTransform):
    invertible = True
    def __init__(self, window_size, hop_size=None, dim=-1, batch_dim=0, split_time=False, pad=False, inversion="crop"):
        self.window_size = window_size
        self.hop_size = hop_size or self.window_size
        assert self.window_size >= self.hop_size
        self.dim = dim
        self.batch_dim = batch_dim
        self.split_time = split_time
        self.pad = pad
        self.inversion = inversion

    def __repr__(self):
        return f"Window(ws={self.window_size}, hs={self.hop_size}, dim={self.dim}, pad={self.pad}, inversion={self.inversion})"

    def _apply_pad(self, chunks):
        if self.pad:
            missing_dims = self.window_size - chunks[-1].shape[self.dim]
            cat_dim = self.dim
            if cat_dim < 0:
                cat_dim = len(chunks[-1].shape) + cat_dim
            zeros_shape = list(chunks[-1].shape[:cat_dim]) + [missing_dims] + list(chunks[-1].shape[cat_dim + 1:])
            chunks[-1] = torch.cat([chunks[-1], torch.zeros(*zeros_shape).to(chunks[-1].device)], self.dim)

        else:
            if self.window_size != chunks[-1].shape[self.dim]:
                chunks = chunks[:-1]
        return chunks

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return [self(x_tmp, *args, time=time, **kwargs) for x_tmp in x]
        chunks = frame(x, self.window_size, self.hop_size, self.dim)
        if time is not None:
            if self.split_time:
                time = torch.stack(self.pad(list(time.split(self.window_size))), 0)
            return chunks, time
        else:
            return chunks

    def invert(self, x, time = None, **kwargs):
        if isinstance(x, list):
            return [self(x_tmp, time=time, **kwargs) for x_tmp in x]

        if self.dim >= 0:
            window_dim = self.dim - 1
        else:
            window_dim = len(x.shape) + self.dim - 1

        if self.window_size == self.hop_size:
            splits = [x_tmp.squeeze(window_dim) for x_tmp in x.split(1, self.dim-1)]
            x = torch.cat(splits, window_dim+1)
        else:
            if self.inversion == "crop":
                idx = [slice(None)] * len(x.shape)
                idx[window_dim+1] = slice(0, self.hop_size)
                x = x.__getitem__(idx)
                x = x.reshape(*x.shape[:window_dim], -1, *x.shape[window_dim+2:])
        if time is not None:
            if self.split_time:
                time = torch.cat([list(time)], -1)
            return x, time
        else:
            return x

class MuLaw(AudioTransform):
    def __init__(self, channels=256, one_hot=None, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.one_hot = one_hot
        self.encoding = torchaudio.transforms.MuLawEncoding(channels)
        self.decoding = torchaudio.transforms.MuLawDecoding(channels)

    def encode(self, x):
        out = self.encoding(x)
        if self.one_hot == "channel":
            out = torch.nn.functional.one_hot(out, self.channels).transpose(-1, -2).contiguous()
        elif self.one_hot == "categorical":
            out = torch.nn.functional.one_hot(out, self.channels)
        return out

    def decode(self, x):
        x = x.long()
        if self.one_hot == "channel":
            x = x.transpose(-2, -1)
            batch_shape = x.shape[:-2]
            idx = x.view(-1, x.shape[-2], x.shape[-1]).nonzero()[:, -1]
            out = idx.reshape(*batch_shape, -1)
        elif self.ont_hot == "categorical":
            batch_shape = x.shape[:-2]
            idx = x.view(-1, x.shape[-2], x.shape[-1]).nonzero()[:, 1]
            out = idx.reshape(*batch_shape, -1)
        else:
            out = x
        out = self.decoding(out)
        return out

    def __call__(self, x, time=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time)
        if time is None:
            return self.encode(x)
        else:
            return self.encode(x), time

    def invert(self, x, time=None, **kwargs):
        if isinstance(x, list):
            return apply_inver_transform_to_list(self, x, time=time)
        if time is None:
            return self.decoding(x)
        else:
            return self.decoding(x), time


class Normalize(AudioTransform):
    def __init__(self, mode="minmax", scale="bipolar", max=None):
        super(Normalize, self).__init__()
        self.mode = mode or "minmax"
        self.polarity = scale or "bipolar"
        self.mean = None
        self.max = max

    def __repr__(self):
        return f"Normalize(mode={self.mode}, scale={self.scale})"

    @property
    def needs_scaling(self):
        return self.mean is None or self.max is None

    def get_stats(self, data, mode="gaussian", polarity="bipolar"):
        if mode == "minmax":
            if polarity == "bipolar":
                mean = (torch.max(data) - torch.sign(torch.min(data))*torch.min(data)) / 2
                max = torch.max(torch.abs(data - mean))
            elif polarity == "unipolar":
                mean = torch.min(data); 
            max = torch.max(torch.abs(data)) if self.max is None else self.max
        elif mode == "gaussian":
            if polarity=='bipolar':
                mean = torch.mean(data); max = torch.std(data)
            if polarity=='unipolar':
                mean = torch.min(data); 
                max = torch.std(data) if self.max is None else torch.std(data) * self.max
        return mean, max

    def scale(self, data):
        if issubclass(type(data), list):
            stats = torch.Tensor([self.get_stats(d, self.mode, self.polarity) for d in data])
            if self.mode == "minmax":
                self.mean = torch.mean(stats[:,0]); self.max = torch.max(stats[:,1])
            else:
                self.mean = torch.min(stats[:, 0])
                # recompose overall variance from element-wise ones
                n_elt = torch.Tensor([torch.prod(torch.Tensor(list(x.size()))) for x in data])
                std_unscaled = ((stats[:,1]**2) / (stats.shape[0]))
                self.max = torch.sqrt(torch.sum(std_unscaled))
        else:
            self.mean, self.max = self.get_stats(data, self.mode, self.polarity)

    def __call__(self, x, batch_first=True):
        if issubclass(type(x), list):
            return [self(x[i]) for i in range(len(x))]
        if self.mean is None:
            mean, max = self.get_stats(x, self.mode, self.polarity)
        else:
            mean, max = self.mean, self.max
        out = torch.true_divide(x - mean, max)
        # if self.polarity == "unipolar":
        #     out = out + eps
        return out

    def invert(self, x):
        if self.mean is None or self.max is None:
            raise NotInvertibleError()
        return (x - eps) * self.max + self.mean


def apply_hpss(tensor, margin=1.0, power=2, transpose=True):
    if len(tensor.shape) > 2:
        batch_size = checktuple(tensor.shape[:-2])
        tensor = np.reshape(tensor, (-1, tensor.shape[-2], tensor.shape[-1]))
        tensor = np.stack([apply_hpss(t, margin=margin, power=power, transpose=transpose) for t in tensor])
        tensor = np.reshape(tensor, (*batch_size, tensor.shape[-2], tensor.shape[-1]))
        return tensor
    if transpose:
        tensor = np.transpose(tensor, (-1, -2))
    tensor = librosa.decompose.hpss(tensor,  margin=margin, power=power)
    if margin == 1.0:
        tensor, _ = tensor
    elif margin > 1.0:
        tensor, _, _ = tensor
    if transpose:
        tensor = np.transpose(tensor, (-1, -2))
    return tensor

def cast_complex(out):
    if torch.get_default_dtype() == torch.float32:
        out = out.to(torch.complex64)
    elif torch.get_default_dtype() == torch.float64:
        out = out.to(torch.complex128)
    return out

# Short-Term Frequency Time Representation (STFT) & derived

class STFT(AudioTransform):
    """ Short-Term Frequency Time Representation"""
    available_backends = ['torch', 'tifresi']

    def __init__(self, nfft=2048, hop_size=None, win_size=None, sr=44100,
                 hps=False, hps_margin= 1.0, hps_power=2, backend="tifresi"):
        self.nfft = nfft
        self.hop_size = hop_size or nfft // 4
        self.win_size = win_size or self.nfft
        self.sr = sr
        self.hps = hps
        self.hps_margin = hps_margin
        self.hps_power = hps_power
        self.backend = backend or "tifresi"
        assert self.backend in self.available_backends
        if self.backend == "tifresi":
            if not TIFRESI_AVAILABLE:
                print("tifresi could not be imported ; switching to torch")
                self.backend = "torch"
        if self.backend == "torch":
            self.griffin_lim = torchaudio.transforms.GriffinLim(self.nfft, win_length=self.win_size,
                                                                hop_length=self.hop_size, power=1.0)

    def __repr__(self):
        repr = f"STFT(nfft={self.nfft}, hop_size={self.hop_size}"
        if self.win_size != self.nfft:
            repr += f", {self.window_size}"
        repr += f', sr={self.sr}'
        if self.hps:
            repr += f", hps_margin{self.hps_margin}, hps_power={self.hps_power}"
        repr += f", backend={self.backend})"
        return repr

    @property
    def frame_dim(self):
        return self.nfft // 2 + 1

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time, sr=self.sr)
        if self.backend == "torch":
            out = torch.stft(x, self.nfft, self.hop_size, win_length=self.win_size, return_complex=True).transpose(-2, -1)
        elif self.backend == "tifresi":
            stft = GaussTF(self.hop_size, self.nfft)
            if x.ndim > 1:
                batch_shape = x.shape[:-1]
                x = x.view(-1, x.shape[-1])
                x = [torch.from_numpy(stft.dgt(preprocess_signal_stft(x_tmp.numpy(), self.nfft)).T) for x_tmp in x]
                out = torch.stack(x).reshape(*batch_shape, -1, x[0].shape[0]).transpose(-2, -1)
            else:
                out = torch.from_numpy(stft.dgt(preprocess_signal_stft(x.numpy(), self.nfft)).T)

        if time is not None:
            new_time = self.get_time(out, time, self.sr)
        else:
            new_time = None

        if self.hps:
            out = torch.from_numpy(apply_hpss(out.numpy(), self.hps_margin, self.hps_power))
        if time is None:
            return out
        else:
            return out, new_time

    def get_time(self, out, time, sr):
        return time + (torch.arange(out.shape[-2]) * self.hop_size / sr)#.expand(*out.shape[:-1])

    def invert_time(self, out, time, sr):
        return time[..., 0]

    def invert(self, x, *args, time=None, **kwargs):
        if torch.is_complex(x):
            if self.backend == "torch":
                out = torch.istft(x.transpose(-2, -1), self.nfft, self.hop_size, self.win_size)
            elif self.backend == "tifresi":
                stft = GaussTF(hop_size=self.hop_size, stft_channels=self.nfft)
                if x.ndim > 2:
                    batch_shape = x.shape[:-2]
                    x = [torch.from_numpy(stft.idgt(x_tmp.numpy().T)) for x_tmp in x.view(-1, *x.shape[-2:])]
                    out = torch.stack(x).reshape(*batch_shape, x[0].shape[0])
                else:
                    out = torch.from_numpy(stft.idgt(x.numpy().T))
        else:
            if self.backend == "torch":
                out = self.griffin_lim(x.transpose(-2, -1))
            elif self.backend == "tifresi":
                stft = GaussTF(hop_size=self.hop_size, stft_channels=self.nfft)
                x = x.clamp(0, None)
                if x.ndim > 2:
                    batch_shape = x.shape[:-2]
                    x = [torch.from_numpy(stft.idgt(x_tmp.numpy().T)) for x_tmp in x.view(-1, *x.shape[-2:])]
                    out = torch.stack(x).reshape(*batch_shape, x[0].shape[0]).float()
                else:
                    out = torch.from_numpy(stft.invert_spectrogram(x.numpy().T)).float()
        if time is not None:
            invert_time = self.invert_time(out, time, self.sr)
            return out, invert_time
        else:
            return out

    def get_frequencies(self, as_notes=False, as_midi=False):
        freq = librosa.fft_frequencies(self.nfft, self.sr)
        if as_notes:
            freq = librosa.hz_to_note(freq)
        elif as_midi:
            freq = librosa.hz_to_midi(freq)
        return freq


class MultiResolutionSTFT(AudioTransform):
    invertible = False

    def __init__(self, resolutions=[256, 512, 1024, 2048], overlap=2, hps=False, hps_margin=1.0, hps_power=2,
                 sr=44100):
        hop_size = max(resolutions) // overlap
        self.stfts = [None] * len(resolutions)
        self.sr = sr
        for i, n in enumerate(resolutions):
            self.stfts[i] = STFT(nfft=max(resolutions), hop_size=hop_size, win_size=resolutions[i],
                                 hps=hps, hps_margin=hps_margin, hps_power=hps_power)

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time)
        transforms = []
        for stft in self.stfts:
            transforms.append(stft(x, time=time))
        if time is not None:
            time = [t[1] for t in transforms]
            transforms = [t[0] for t in transforms]
            time = time[0]
            transforms = torch.stack(transforms, -3)
            return transforms, time
        else:
            transforms = torch.stack(transforms, -3)
            return transforms


class HPS(STFT):
    invertible = False
    """Harmonic Power Spectrum"""
    def __init__(self, nfft, hop_size=None, win_size=None, sr=44100, threshold1=1e-3, threshold2=1e-2, octave=False):
        super(HPS, self).__init__(nfft, hop_size, win_size, sr=sr)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.octave = octave

    def __call__(self, x, *args, time=None, sr=None, **kwargs):
        if isinstance(x, list):
            return [self(x_tmp) for x_tmp in x]
        fft = super(HPS, self).__call__(x, *args, time=time, **kwargs).abs()
        if time is not None:
            fft, time= fft
        zeros = torch.zeros_like(fft)
        fft = torch.where(fft > self.threshold1*fft.max(), fft, zeros)
        hps = torch.zeros_like(fft)
        if self.octave:
            power2 = torch.log2(torch.tensor(float(hps.shape[-1]))).floor()
        for f in range(1, hps.shape[-1]-1):
            if self.octave:
                idxs = (2**torch.range(0, power2)*f)
                idxs = idxs[idxs < hps.shape[-1]]
            else:
                idxs = torch.range(f, hps.shape[-1]-1, f)
            prod = torch.prod(fft[..., idxs.long()], -1)
            hps[..., f] = prod
        hps = torch.where(hps > self.threshold2*hps.max(), hps, zeros)
        if time is None:
            return hps
        else:
            return hps, time

    def invert(self, x, *args, **kwargs):
        raise NotInvertibleError


class ChromagramSTFT(STFT):
    """Chromagram based on STFT"""
    invertible = False

    def __init__(self, nfft, hop_size=None, win_size=None, sr=44100, filter=True, **kwargs):
        super(ChromagramSTFT, self).__init__(nfft, hop_size, win_size, sr)
        self.chroma_args = kwargs
        self.filter = filter

    def __call__(self, x, *args, time=None, **kwargs):
        stft = super(ChromagramSTFT, self).__call__(x, *args, **kwargs).transpose(-2, -1).numpy()
        if time is not None:
            stft, time = stft
        chroma = np.abs(librosa.feature.chroma_stft(S=stft, **self.chroma_args))
        if self.filter:
            chroma = np.minimum(chroma, librosa.decompose.nn_filter(chroma, aggregate=np.median, metric='cosine'))
            chroma = scipy.ndimage.median_filter(chroma, size=(1, 9))
        out = torch.from_numpy(chroma).to(x.device).transpose(-2, -1)
        if time is None:
            return out
        else:
            return out, time

    def invert(self, x, *args, **kwargs):
        raise NotInvertibleError

    def get_frequencies(self, as_notes=False, as_midi=False):
        raise NotImplementedError


class EPCP(HPS):
    """Chromagram based on PowerSpectrum"""
    invertible = False

    def __init__(self, nfft, hop_size=None, win_size=None, sr=44100, threshold1=1e-5, threshold2=0, octave=False, filter=False, **kwargs):
        super(EPCP, self).__init__(nfft, hop_size=hop_size, win_size=win_size, sr=sr,
                                   threshold1=threshold1, threshold2=threshold2, octave=octave)
        self.chroma_args = kwargs
        self.filter = filter

    def __call__(self, x, *args, time=None, **kwargs):
        hps = super(EPCP, self).__call__(x, *args, time=time, **kwargs).transpose(-2, -1).numpy()
        if time is not None:
            hps, time = hps
        chroma = librosa.feature.chroma_stft(S=hps, **self.chroma_args)
        if self.filter:
            chroma = np.minimum(chroma, librosa.decompose.nn_filter(chroma, aggregate=np.median, metric='cosine'))
            chroma = scipy.ndimage.median_filter(chroma, size=(1, 9))
        out = torch.from_numpy(chroma).to(x.device).transpose(-2, -1)
        if time is None:
            return out
        else:
            return out, time

    def invert(self, x, *args, **kwargs):
        raise NotInvertibleError

    def get_frequencies(self, as_notes=False, as_midi=False):
        raise NotImplementedError


class PowerSpectrum(STFT):
    """Power spectrum"""

    def __init__(self, nfft=2048, hop_size=None, win_size=None, sr=44100, hps=False, hps_margin=1.0, hps_power=2):
        super().__init__(nfft, hop_size, win_size, sr, hps, hps_margin, hps_power)
        self.griffin_lim = torchaudio.transforms.GriffinLim(self.nfft, win_length=self.win_size, hop_length = self.hop_size, power=2.0)

    def __call__(self, x, *args, time=None, **kwargs):
        stft = super(PowerSpectrum, self).__call__(x, *args, time=time, **kwargs)
        if time is None:
            return stft.abs().pow(2)
        else:
            return stft.abs().pow(2), time

    def invert(self, x, time=None):
        inv_x = x.sqrt()
        return super(PowerSpectrum,self).invert(inv_x, time=time)


class DCT(AudioTransform):
    """ Short-Term Frequency Time Representation"""
    def __init__(self, nfft=2048, hop_size=None, win_size=None, mode=2, sr=44100, hps=False, hps_margin= 1.0, hps_power=2):
        self.nfft = nfft
        self.hop_size = hop_size or nfft // 4
        self.win_size = win_size or self.nfft
        self.window = np.hanning(self.win_size)
        self.mode = mode
        self.sr = sr
        self.hps = hps
        self.hps_margin = hps_margin
        self.hps_power = hps_power

    @property
    def frame_dim(self):
        return len(self.nfft // 2)

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time, sr=self.sr)
        x_framed = frame(x, self.nfft, self.hop_size, -1).numpy()
        if self.win_size > self.nfft:
            x_framed = np.pad(x_framed, self.win_size - self.nfft, axis=-1)
        # x_framed *= self.window
        out = torch.from_numpy(dct(x_framed, type=self.mode))
        if time is not None:
            new_time = self.get_time(out, time, self.sr)
            return out, new_time
        else:
            return out

    def get_time(self, out, time, sr):
        return time + (torch.arange(out.shape[-2]) * self.hop_size / sr)#.expand(*out.shape[:-1])

    def invert_time(self, out, time, sr):
        return time[..., 0]

    def invert(self, x, *args, time=None, **kwargs):
        x_inv = idct(x.numpy(), type=self.mode)
        x_inv *= self.window
        out = torch.from_numpy(overlap_add(x_inv, self.nfft, self.hop_size))
        if time is not None:
            invert_time = self.invert_time(out, time, self.sr)
            return out, invert_time
        else:
            return out

    def get_frequencies(self, as_notes=False, as_midi=False):
        freq = librosa.fft_frequencies(self.nfft, self.sr)
        if as_notes:
            freq = librosa.hz_to_note(freq)
        elif as_midi:
            freq = librosa.hz_to_midi(freq)
        return freq


class MelSpectrogram(AudioTransform, ta_transforms.MelSpectrogram):
    invertible = None

    def __init__(self, *args, sr=None, inversion_module=None, **kwargs):
        if kwargs.get('sample_rate') is not None:
            del kwargs['sample_rate']
        AudioTransform.__init__(self, sr=sr)
        ta_transforms.MelSpectrogram.__init__(self, sample_rate=sr, **kwargs)
        self.inversion_module = inversion_module

    def __call__(self, x, *args, time=None, sr=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time)
        out = ta_transforms.MelSpectrogram.__call__(self, x, *args, **kwargs).transpose(-2, -1)
        if time is None:
            return out
        else:
            time = self.get_time(out, time, self.sr)
            return out, time

    def get_time(self, out, time, sr):
        return time + (torch.arange(out.shape[-2]) * self.hop_size / sr)

    def invert_time(self, out, time, sr):
        return time[..., 0]

    @property
    def invertible(self):
        return self.inversion_module is not None

    def invert(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return apply_inver_transform_to_list(self, x, time=time)

        if self.inversion_module is None:
            raise NotInvertibleError()
        else:
            out = self.inversion_module.invert_melspectrum(x.transpose(-2, -1))
            if time is None:
                return out
            else:
                return out, self.invert_time(out, time, self.sr)


class MFCC(AudioTransform, ta_transforms.MFCC):
    invertible = None

    def __init__(self, *args, sr=None, inversion_module=None, **kwargs):
        if kwargs.get('sample_rate') is not None:
            del kwargs['sample_rate']
        AudioTransform.__init__(self, sr=sr)
        ta_transforms.MFCC.__init__(self, sample_rate=sr, **kwargs)
        self.inversion_module = inversion_module

    def __call__(self, x, *args, time=None, sr=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time)
        out = ta_transforms.MFCC.__call__(self, x, *args, **kwargs).transpose(-2, -1)
        if time is None:
            return out
        else:
            time = self.get_time(out, time, self.sr)
            return out, time

    def get_time(self, out, time, sr):
        return time + (torch.arange(out.shape[-2]) * self.hop_size / sr)

    def invert_time(self, out, time, sr):
        return time[..., 0]

    @property
    def invertible(self):
        return self.inversion_module is not None

    def invert(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return apply_inver_transform_to_list(self, x, time=time)

        if self.inversion_module is None:
            raise NotInvertibleError()
        else:
            out = self.inversion_module.invert_melspectrum(x.transpose(-2, -1))
            if time is None:
                return out
            else:
                return out, self.invert_time(out, time, self.sr)



class LFCC(AudioTransform, ta_transforms.LFCC):
    invertible = None

    def __init__(self, *args, sr=None, inversion_module=None, **kwargs):
        if kwargs.get('sample_rate') is not None:
            del kwargs['sample_rate']
        AudioTransform.__init__(self, sr=sr)
        ta_transforms.LFCC.__init__(self, sample_rate=sr, **kwargs)
        self.inversion_module = inversion_module

    def __call__(self, x, *args, time=None, sr=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time)
        out = ta_transforms.LFCC.__call__(self, x, *args, **kwargs).transpose(-2, -1)
        if time is None:
            return out
        else:
            time = self.get_time(out, time, self.sr)
            return out, time

    def get_time(self, out, time, sr):
        return time + (torch.arange(out.shape[-2]) * self.hop_size / sr)

    def invert_time(self, out, time, sr):
        return time[..., 0]

    @property
    def invertible(self):
        return self.inversion_module is not None

    def invert(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return apply_inver_transform_to_list(self, x, time=time)

        if self.inversion_module is None:
            raise NotInvertibleError()
        else:
            out = self.inversion_module.invert_melspectrum(x.transpose(-2, -1))
            if time is None:
                return out
            else:
                return out, self.invert_time(out, time, self.sr)


class MDCT(AudioTransform):
    """ Modified Discrete Cosine Transform"""

    def __init__(self, win_size=2048, hop_size=None, odd=True, sr=44100, padding=0, centered=True):
        self.win_size = win_size or 2048
        self.hop_size = hop_size or self.win_size // 2
        self.window = np.hanning(self.win_size)
        self.sr = sr
        self.padding = padding
        self.centered = centered
        self.odd = odd

    @property
    def mdct_args(self):
        return {'framelength': self.win_size, 'hopsize': self.hop_size, 'odd': self.odd,
                 'centered': self.centered, 'padding': self.padding}

    @property
    def forward_callback(self):
        return mdct

    @property
    def invert_callback(self):
        return imdct

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time, sr=self.sr)

        if len(x.shape) > 2:
            batch_shape = x.shape[:-2]; event_shape = x.shape[-2:]
            out, time = apply_transform_to_list(self, list(x.view(-1, *event_shape)), list(time))
            out = torch.stack(out).reshape(*batch_shape, out[0].shape[1:])
        else:
            if x.ndim == 2:
                x = x.transpose(-2, -1)
            out = torch.from_numpy(self.forward_callback(x.numpy(), **self.mdct_args))
        if out.ndim == 2:
            out = out.transpose(-1, -2).unsqueeze(0).contiguous()
        elif out.ndim == 3:
            out = out.permute(-1, -3, -2)
        if time is not None:
            new_time = self.get_time(out, time, self.sr)
            return out, new_time
        else:
            return out

    def get_time(self, out, time, sr):
        return time + (torch.arange(out.shape[-2]) * self.hop_size / sr)  # .expand(*out.shape[:-1])

    def invert_time(self, out, time, sr):
        return time[..., 0]

    def invert(self, x, *args, time=None, **kwargs):

        if len(x.shape) > 3:
            batch_shape = x.shape[:-3]; event_shape = x.shape[-3:]
            out = [torch.from_numpy(self.invert_callback(x_tmp.numpy(), **self.mdct_args)) for x_tmp in x.view(-1, *event_shape)]
            out = torch.stack(out).reshape(*batch_shape, out[0].shape[1:])
        else:
            if x.ndim == 3:
                x = x.permute(-1, -2, -3)
            out = torch.from_numpy(self.invert_callback(x.numpy(), **self.mdct_args))
            return out.transpose(-1, -2)
        if time is not None:
            invert_time = self.invert_time(out, time, self.sr)
            return out, invert_time
        else:
            return out

    def get_frequencies(self, as_notes=False, as_midi=False):
        freq = librosa.fft_frequencies(self.nfft, self.sr)
        if as_notes:
            freq = librosa.hz_to_note(freq)
        elif as_midi:
            freq = librosa.hz_to_midi(freq)
        return freq


class MDST(MDCT):
    @property
    def forward_callback(self):
        return mdst

    @property
    def invert_callback(self):
        return mdst




# Constant-Q Transform
class CQT(AudioTransform):

    @staticmethod
    def callback(x, hps=False, hps_margin=1.0, hps_power=2, **kwargs):
        if not x.flags.f_contiguous:
            x = np.asfortranarray(x)
        transform = librosa.core.cqt(x, **kwargs)
        if hps:
            transform = torch.from_numpy(apply_hpss(transform, hps_margin, hps_power))
        return transform

    def __init__(self, sr=44100, hps=False, hps_margin= 1.0, hps_power=2, **kwargs):
        self.sr = sr
        self.cqt_args = kwargs
        self.hps = hps
        self.hps_margin = hps_margin
        self.hps_power = hps_power

    def __call__(self, x, *args, time=None, sr=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time)
        if len(x.shape) == 1:
            out = torch.from_numpy(self.callback(x.cpu().numpy(), sr=self.sr, **self.cqt_args)).to(torch.get_default_dtype()).transpose(-2, -1)
        if len(x.shape) > 1:
            batch_shape = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = np.stack([self.callback(x_tmp.cpu().numpy(), sr=self.sr, hps=self.hps, hps_margin=self.hps_margin, hps_power=self.hps_power, **self.cqt_args) for x_tmp in x])
            out = torch.from_numpy(np.reshape(out, (*batch_shape, *out.shape[-2:])))
            out = out.to(torch.get_default_dtype()).transpose(-2, -1)

        if time is None:
            return out
        else:
            new_time = self.get_time(out, time, self.sr)
            return out, new_time

    def get_time(self, out, time, sr):
        return time + (torch.arange(out.shape[-2]) * self.cqt_args.get('hop_length', 512) / sr)#.expand(*out.shape[:-1])

    def invert(self, x, *args, **kwargs):
        if isinstance(x, list):
            return [self.invert(x_tmp) for x_tmp in x]
        return self.invert_callback(x, **self.cqt_args)

    @property
    def frame_dim(self):
        return len(self.get_frequencies())

    def get_frequencies(self, as_notes=False, as_midi=False):
        freq = librosa.cqt_frequencies(self.cqt_args.get('n_bins', 84),
                                       self.cqt_args.get('fmin', librosa.note_to_hz('C1')),
                                       self.cqt_args.get('bins_per_octave', 12),
                                       self.cqt_args.get('tuning', 0.0))
        if as_notes:
            freq = librosa.hz_to_note(freq)
        elif as_midi:
            freq = librosa.hz_to_midi(freq)
        return freq


class HybridCQT(CQT):
    @staticmethod
    def callback(x, **kwargs):
        if not x.flags.f_contiguous:
            x = np.asfortranarray(x)

        return librosa.core.hybrid_cqt(x, **kwargs)

class PseudoCQT(CQT):
    @staticmethod
    def callback(x, **kwargs):
        if not x.flags.f_contiguous:
            x = np.asfortranarray(x)
        return librosa.core.pseudo_cqt(x, **kwargs)

class VariableQT(CQT):
    @staticmethod
    def callback(x, **kwargs):
        if not x.flags.f_contiguous:
            x = np.asfortranarray(x)
        return librosa.core.vqt(x, **kwargs)

class ChromagramCQT(CQT):
    def __init__(self, cens=False, filter=True, **kwargs):
        super(ChromagramCQT, self).__init__(**kwargs)
        self.cens = cens
        self.filter = filter

    def callback(self, x, **kwargs):
        if not x.flags.f_contiguous:
            x = np.asfortranarray(x)
        if self.cens:
            return librosa.feature.chroma_cqt(y=x, **kwargs)
        else:
            return librosa.feature.chroma_cens(y=x, **kwargs)

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time)
        pdb.set_trace()
        if len(x.shape) == 1:
            out = self.callback(x.cpu().numpy(), sr=self.sr, **self.cqt_args)
            if self.filter:
                out = np.minimum(out, librosa.decompose.nn_filter(out, aggregate=np.median, metric='cosine'))
                out = scipy.ndimage.median_filter(out, size=(1, 9))
            out = out.T
        if len(x.shape) > 1:
            batch_shape = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = [self.callback(x_tmp.cpu().numpy(), sr=self.sr, **self.cqt_args) for x_tmp in x]
            if self.filter:
                out = [np.minimum(o, librosa.decompose.nn_filter(o, aggregate=np.median, metric='cosine')) for o in out]
                out = [scipy.ndimage.median_filter(o, size=(1, 9)) for o in out]
            out = np.stack([o.T for o in out])
            out = np.reshape(out, (*batch_shape, *out.shape[-2:]))

        out = torch.from_numpy(out).to(torch.get_default_dtype())
        if time is None:
            return out
        else:
            new_time = self.get_time(out, time, self.sr)
            return out, new_time


class HarmonicCQT(CQT):
    invertible = False

    def __init__(self, harmonics: list = [0.5, 1, 2, 3, 4, 5], **kwargs):
        # original values from article
        kwargs['bins_per_octave'] = kwargs.get('bins_per_octave', 60)
        super(HarmonicCQT, self).__init__(**kwargs)
        self.harmonics = harmonics

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time)
        if len(x.shape) > 1:
            batch_shape = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = torch.stack([self(x_i, **kwargs) for x_i in x])
            out = out.reshape(*batch_shape, *out.shape[-3:])
            if time is None:
                return out
            else:
                time = self.get_time(out, time, self.sr)
                return out, time
        # compute harmonic CQT
        # if torch.is_tensor(x):
            # x = x.numpy()
        n_steps = np.ceil(x.shape[-1] / self.cqt_args.get('hop_length', 512)).astype(np.int) + 1
        n_bins = self.cqt_args.get('n_bins', 84)
        hcqt = np.zeros((*x.shape[:-1], len(self.harmonics), n_steps, n_bins), dtype=np.complex)
        freqs = super(HarmonicCQT, self).get_frequencies()
        for i, f in enumerate(freqs):
            for h in range(len(self.harmonics)):
                cqt_args = dict(self.cqt_args); cqt_args['fmin'] = self.harmonics[h] * f
                current_cqt = self.callback(x.cpu().numpy(), **cqt_args)
                hcqt[..., h, :current_cqt.shape[-1], :] = np.transpose(current_cqt, (-1, -2))
        return cast_complex(torch.from_numpy(hcqt))

    def invert(self, x, *args, **kwargs):
        raise NotInvertibleError()





def pitch_hash():
    return {'C':0, 'Db':1, 'D-':1, 'C#':1, 'C♯':1, 'D':2, 'E-':3, 'Eb':3, 'D♯':3, 'D#':3, 'E':4, 'F':5, 'F#':6, 'F♯':6,
            'Gb':6, 'G-':6, 'G':7, 'G#':8, 'G♯':8, 'A-':8, 'Ab':8, 'A':9, 'A♯':10, 'A#':10, 'B-':10, 'Bb':10, 'B':11, 'C-':11}

## Non-Stationary Gabor Transform

class NSGT(AudioTransform):
    scale_hash = {'oct':OctScale, 'mel':MelScale, 'lin':LinScale, 'log':LogScale}

    def __init__(self, ls=None, fmin=40, fmax=22050, scale="oct", bins=24, downsample=10, sr=44100, hps=False,
                 hps_margin= 1.0, hps_power=2, n_iter = 32):
        self.srcale = self.scale_hash[scale](fmin, fmax, bins)
        self.downsample = downsample
        self.ls = ls
        self.sr = sr
        self.hps = hps
        self.hps_margin = hps_margin
        self.hps_power = hps_power
        self.n_iter = n_iter or 32

    @property
    def ls_samples(self):
        if self.ls is not None:
            if isinstance(self.ls, int):
                return self.ls
            elif isinstance(self.ls, float):
                return int(self.ls * self.sr)
            else:
                raise ValueError("length of NSGT must be float or int, got : %s"%type(self.ls))

        else:
            return None

    @property
    def frame_dim(self):
        return len(self.srscale)

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return [self(x_tmp) for x_tmp in x]

        ls = self.ls_samples or x.shape[-1]
        nsgt = NNSGT(self.srcale, self.sr, ls, real=True, matrixform=True, reducedform=True)
        if torch.is_tensor(x):
            if x.grad is not None:
                print('[Warning] performing NSGT on a backwarded tensor')
            x = x.detach().cpu().numpy()
        # forward
        needs_squeeze = len(x.shape) == 1
        if len(x.shape) == 1:
            x = x[np.newaxis]
        out = np.stack([np.asarray(nsgt.forward(x_tmp)).T for x_tmp in x])
        # downsample
        if self.downsample:
            step = np.linspace(0, out.shape[-2]-1, int(out.shape[-2]/self.downsample)).astype(np.int)
            out = out[..., step, :]
        if self.hps:
            out = apply_hpss(out, self.hps_margin, self.hps_power)
        out = torch.from_numpy(out)
        if needs_squeeze:
            out = out[0]
        if torch.get_default_dtype() == torch.float32:
            out = out.to(torch.complex64)
        elif torch.get_default_dtype() == torch.float64:
            out = out.to(torch.complex128)

        if time is not None:
            time = self.get_time(out, ls, time)
            return out, time.to(torch.get_default_dtype())
        else:
            return out

    def get_time(self, out, ls, time):
        return (torch.arange(0, out.shape[-2]) / (out.shape[-2]-1) * ls) / self.sr + time

    def invert(self, x, *args, **kwargs):

        if self.ls is None:
            raise AttributeError("Inverting NSGT requires fixed length.")
        nsgt = NNSGT(self.srcale, self.sr, int(self.ls * self.sr), real=True, matrixform=True, reducedform=True)

        if torch.is_tensor(x):
            if x.grad is not None:
                print('[Warning] performing iNSGT on a backwarded tensor')
            x = x.detach().cpu().numpy()
        #TODO upsample

        needs_squeeze = len(x.shape) == 2
        if needs_squeeze:
            x = x[np.newaxis]
        if self.downsample:
            x_data = np.linspace(0, self.ls or 1.0, x.shape[-2])
            x_interp = np.linspace(0, self.ls or 1.0, x.shape[-2]*self.downsample)
            if np.iscomplexobj(x):
                interp_r = interp1d(x_data, np.abs(x), axis=-2)
                interp_p = interp1d(x_data, np.angle(x), axis=-2)
                x = interp_r(x_interp) * np.exp(interp_p(x_interp) * 1j)
            else:
                interp_r = interp1d(x_data, x, axis=-2)
                x = interp_r(x_interp)

        if not np.iscomplexobj(x):
            out = np.stack([self.griffin_lim(nsgt, x_tmp) for x_tmp in x])
        else:
            out = np.stack([nsgt.backward(x_tmp.T) for x_tmp in x])

        if needs_squeeze:
            out = out[0]
        return torch.from_numpy(out)

    def griffin_lim(self, nsgt, mag):
        mag = mag.T
        phase = np.random.uniform(0, 2*np.pi, mag.shape)
        n_iter = 32 if not hasattr(self, "n_iter") else self.n_iter
        for i in range(n_iter):
            spec = mag * np.exp(phase * 1j)
            inv = nsgt.backward(spec)
            phase = np.angle(np.stack(nsgt.forward(inv), 0))
            if phase.shape[-1] > mag.shape[-1]:
                phase = phase[..., :spec.shape[-1]]
        spec = mag * np.exp(phase * 1j)
        inv = np.real(nsgt.backward(spec))
        return inv

    def pghi(self, mag):
        return

    def get_frequencies(self, as_notes=False, as_midi=False):
        freq, _ = self.scale()
        if as_notes:
            freq = librosa.hz_to_note(freq)
        elif as_midi:
            freq = librosa.hz_to_midi(freq)
        return freq



class ChromagramNSGT(NSGT):

    def __init__(self, ls, fmin=80, fmax=22050, scale="oct", bins=24, downsample=10, sr=44100, reducedform=False, filter=True):
        scale = "oct"
        super(ChromagramNSGT, self).__init__(ls, fmin, fmax, scale, bins, downsample, sr, reducedform)
        self.filter = filter

    def __call__(self, x, *args, n_chromas=12, **kwargs):
        nsgt = super().__call__(x, *args, **kwargs).cpu().numpy()
        chromagram = np.zeros((*nsgt.shape[:-1], n_chromas))
        notes = self.get_frequencies(as_notes=True)
        freq = [re.match(r'([A-G]+♯*).*$', f).groups()[0] for f in notes]
        hash = pitch_hash()
        for i,f in enumerate(freq):
            chromagram[..., hash[f]] += np.abs(nsgt[..., i])
        chromagram = np.log10(chromagram + 1)
        if self.filter:
            chromagram = np.minimum(np.transpose(chromagram, axes=(-2, -1)),
                                    librosa.decompose.nn_filter(chromagram, aggregate=np.median, metric='cosine'))
            chromagram = np.transpose(chromagram, axes=(-2, -1))
            chromagram = scipy.ndimage.median_filter(chromagram, size=(1, 4))
        return torch.from_numpy(chromagram).to(x.device)

## Filtering
eps = 1e-9


class Magnitude(AudioTransform):
    def __init__(self, normalize=None, contrast="log1p", shrink=4, global_norm=True, log_clamp=-8, **kwargs):
        super(Magnitude, self).__init__()
        self.normalize = None
        self.constrast = contrast
        if normalize is not None:
            self.normalize = Normalize(**normalize)
        self.shrink = shrink
        self.global_norm = global_norm
        self.log_clamp = log_clamp

    @property
    def needs_scaling(self):
        if self.normalize is not None:
            return self.normalize.needs_scaling

    def preprocess(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.abs()
        if self.constrast is None:
            return x
        elif self.constrast == "log":
            return torch.clamp(torch.log(x/self.shrink), self.log_clamp)
        elif self.constrast == "log1p":
            return torch.log1p(x/self.shrink)
        else:
            raise ValueError('constrast %s not valid for Magnitude transform'%self.constrast)

    def scale(self, x):
        if self.normalize is not None and self.global_norm:
            if isinstance(x, list):
                x = [self.preprocess(x[i]) for i in range(len(x))]
            else:
                x = self.preprocess(x)
            self.normalize.scale(x)

    def __call__(self, x, time=None, sr=None):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time, sr=sr)
        out = self.preprocess(x)
        if self.normalize is not None:
            out = self.normalize(out)
        if time is None:
            return out
        else:
            return out, time

    def invert(self, x, time=None, sr=None):
        if isinstance(x, list):
            return [self.invert(x_i) for x_i in x]
        if self.normalize is not None:
            x = self.normalize.invert(x)
        if self.constrast == "log":
            x[x <= self.log_clamp] = -torch.inf
            x = torch.exp(x)*self.shrink
        elif self.constrast == "log1p":
            x = torch.expm1(x)*self.shrink
        else:
            raise ValueError('constrast %s not valid for Magnitude transform'%self.constrast)
        if time is None:
            return x
        else:
            return x, time


class Phase(AudioTransform):
    def __init__(self, unwrap=True, normalize=None,**kwargs):
        super(Phase, self).__init__()
        self.unwrap = unwrap
        self.normalize = None
        if normalize is not None:
            self.normalize = Normalize(**normalize)

    def scale(self, x):
        if isinstance(x, list):
            x = [x[i].angle() for i in range(len(x))]
        else:
            x = x.angle()
        if self.unwrap:
            x = unwrap(x)
        if self.normalize is not None:
            self.normalize.scale(x)

    def __call__(self, x, **kwargs):
        if isinstance(x, list):
            return [self(x_i) for x_i in x]
        phase = x.angle()

        if self.unwrap:
            # phase = torch.from_numpy(np.unwrap(phase.numpy()))
            phase = unwrap(phase)

        # ax[0].imshow(phase[0], aspect="auto")
        if self.normalize is not None:
            phase = self.normalize(phase)
        return phase

    def invert(self, x, *args, **kwargs):
        if self.normalize is not None:
            x = self.normalize.invert(x)

        # ax[1].imshow(x[0], aspect="auto")
        if self.unwrap:
            x = torch.fmod(x, 2*torch.pi)
        return x


class InstantaneousFrequency(AudioTransform):
    methods = ['backward', 'forward', 'central', 'fm-disc']

    def __repr__(self):
        return "<preprocessing InstantaneousFrequency with method: %s, normalize: %s, mode: %s>"%(self.method, self.normalize, self.mode)

    def __init__(self, method="backward", wrap=False, weighted = False, normalize=None, mode=None):
        assert method in self.methods
        self.method = method
        self.wrap = wrap
        self.weighted = weighted
        self.normalize = normalize
        self.mode = mode
        if self.normalize is not None:
            self.normalize = Normalize(**normalize)

    def scale(self, data):
        if self.normalize is not None:
            self.normalize.scale(self.get_if(data))

    def get_window(self, N):
        n = torch.arange(N)
        return (1.5 * N) / (N ** 2 - 1) * (1 - ((n - (N / 2 - 1)) / (N / 2)) ** 2)

    def get_if(self, data):
        if issubclass(type(data), list):
            return [self.get_if(i) for i in data]

        if self.method in ['forward', 'backward', 'central']:
            phase = unwrap(torch.angle(data))
            # mag = np.abs(data)
            if self.method == "backward":
                inst_f = fdiff(phase, order=1)
                inst_f[1:] /= torch.pi
            elif self.method == "forward":
                inst_f = torch.flip(fdiff(torch.flip(phase, axis=0), order=1), axis=0)
                inst_f[:-1] /= -torch.pi
            if self.method == "central":
                inst_f = fdiff(phase, order=2)
                inst_f[1:-1] /= torch.pi
            if self.weighted:
                window = self.get_window(inst_f.shape[0]).unsqueeze(1)
                inst_f = window * inst_f

        if self.method == "fm-disc":
            real = data.real; dreal = fdiff(real, order=1)
            imag = data.imag; dimag = fdiff(imag, order=1)
            inst_f = (real * dimag - dreal * imag) / (real**2 + imag**2)
            inst_f = inst_f / (2*torch.pi)

        if self.wrap:
            inst_f = torch.fmod(inst_f, 2*torch.pi)

        return inst_f

    def __call__(self, data, batch_first=False, **kwargs):
        if isinstance(data, list):
            return [self(x_i) for x_i in data]
        if batch_first:
            return torch.stack([self(data[i], batch_first=False) for i in range(data.shape[0])])
        inst_f = self.get_if(data)
        if self.normalize is not None:
            inst_f = self.normalize(inst_f)
        return inst_f

    def invert(self, data, batch_first=False, **kwargs):
        if issubclass(type(data), list):
            return [self.invert(x, batch_first=batch_first) for x in data]
        if batch_first:
            if torch.is_tensor(data):
                return torch.stack([self.invert(data[i], batch_first=False) for i in range(data.shape[0])])
            else:
                return torch.stack([self.invert(data[i], batch_first=False) for i in range(data.shape[0])])
        if self.normalize:
            data = self.normalize.invert(data)
        if self.wrap:
            data = unwrap(data)
        if self.method == "backward":
            data[1:] *= torch.pi
            phase = fint(data, order=1)
        if self.method == "forward":
            data[:-1] *= -torch.pi
            if torch.is_tensor(data):
                phase = torch.flip(fint(torch.flip(data, axis=0), order=1), axis=0)
            else:
                phase = torch.flip(fint(torch.flip(data, axis=0), order=1), axis=0)
        elif self.method == "central":
            data[1:-1] *= torch.pi
            phase = fint(data, order=2)

        return phase


class Polar(AudioTransform):
    magnitude_transform = Magnitude
    phase_transform = Phase
    invertible = True

    def __init__(self, *args, mag_options={}, phase_options={}, stack=True, **kwargs):
        super(Polar, self).__init__()
        self.transforms = [self.magnitude_transform(**mag_options),
                           self.phase_transform(**phase_options)]
        self.stack = stack

    def __getitem__(self, item):
        return self.transforms[item]

    def __call__(self, x, time=None, **kwargs):
        # fft = super(Polar, self).forward(x)
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time)
        out = [self.transforms[0](x), self.transforms[1](x)]
        if self.stack:
            out = torch.stack(out, dim=-3)
        if time is None:
            return out
        else:
            return out, time

    def invert(self, x, time=None, **kwargs):
        if self.stack:
            mag, phase = x[..., 0, :, :], x[..., 1, :, :]
            mag, phase = self.transforms[0].invert(mag), self.transforms[1].invert(phase)
        else:
            mag, phase = self.transforms[0].invert(x[0]), self.transforms[1].invert(x[1])
        fft = mag*torch.exp(1j*phase)
        if time is None:
            return fft
        else:
            return fft, time
        return fft

    def scale(self, x):
        self.transforms[0].scale(x)
        self.transforms[1].scale(x)


class PolarInst(Polar):
    phase_transform = InstantaneousFrequency


class Cartesian(AudioTransform):
    def __init__(self, normalize=None, stack=-3, cat=None):
        self.normalize = normalize
        if not normalize is None:
            self.normalize = Normalize(**normalize)
        self.stack = stack
        self.cat = cat
        if self.stack is not None and self.cat is not None:
            raise ValueError('stack and cat are not compatible'%(self.stack, self.cat))

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time)
        real, imag = x.real, x.imag
        if self.stack is not None:
            out = torch.stack([real, imag], dim=self.stack)
        elif self.cat is not None:
            out = torch.cat([real, imag], dim=self.cat)
        else:
            out = [real, imag]
        if time is None:
            return out
        else:
            return out, time

    def invert(self, x, time=None, *args, **kwargs):
        if self.stack is not None:
            real, imag = x.split(1, dim=self.stack)
            real = real.squeeze(self.stack); imag = imag.squeeze(self.stack)
        elif self.cat is not None:
            full_dim = x.shape[self.cat]
            real, imag = x.split(full_dim//2, dim=self.cat)
        out = real + imag * 1j
        if time is None:
            return out
        else:
            return out, time


class Harmonic(AudioTransform):
    def __init__(self, margin=8):
        self.margin = margin

    def __call__(self, x, *args, **kwargs):
        return torch.from_numpy(librosa.effects.harmonic(y=x.cpu().numpy(), margin=self.margin)).to(x.device)


## UTILS TRANSFORMS

class Unsqueeze(AudioTransform):
    invertible = True

    def __repr__(self):
        return "Unsqueeze(dim=%s)"%self.dim

    def __init__(self, dim):
        """
        Unsqueezes the incoming data
        Args:
            dim: dimension unsqueezed.
        """
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def __call__(self, x, time=None, sr=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time, sr=sr)
        if time is None:
            return x.unsqueeze(self.dim)
        else:
            return x.unsqueeze(self.dim), time

    def invert(self, x, *args, time=None, sr=None, **kwargs):
        if isinstance(x, list):
            return apply_inver_transform_to_list(self, x, time=time, sr=sr)
        if time is None:
            return x.squeeze(self.dim)
        else:
            return x.squeeze(self.dim), time
        

class Transpose(AudioTransform):
    invertible = True

    def __repr__(self):
        return "Unsqueeze(dim=%s)"%self.dim

    def __init__(self, dims, contiguous=True):
        """
        Unsqueezes the incoming data
        Args:
            dim: dimension unsqueezed.
        """
        super(Transpose, self).__init__()
        self.dims = dims
        self.contiguous = contiguous

    def __call__(self, x, time=None, sr=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time, sr=sr)
        data = x.transpose(*self.dims)
        if self.contiguous:
            data = data.contiguous()
        if time is None:
            return data
        else:
            return data, time

    def invert(self, x, **kwargs):
        return self(x, **kwargs)
