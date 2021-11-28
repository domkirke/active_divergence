import torch, torchaudio, abc, numpy as np, librosa, re, scipy, pdb, math
import sys; sys.path.append('../')
from active_divergence.utils import checklist, checktuple, frame, overlap_add
from scipy.fft import dct, dst, idct, idst
from scipy.interpolate import interp1d
from nsgt import NSGT as NNSGT, LogScale, LinScale, MelScale, OctScale


class NotInvertibleError(Exception):
    pass

class AudioTransform(object):
    invertible = True
    needs_scaling = False
    @abc.abstractmethod
    def __init__(self, sr=44100):
        pass

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

class ComposeAudioTransform(AudioTransform):
    @property
    def invertible(self):
        return not False in [t.invertible for t in self.transforms]

    @property
    def needs_scaling(self):
        return True in [t.needs_scaling for t in self.transforms]

    def __init__(self, transforms = [], sr=44100):
        self.transforms = transforms

    def __add__(self, itm):
        if not isinstance(itm, AudioTransform):
            raise TypeError("ComposeAudioTransform can only be added to other AudioTransforms")
        if isinstance(itm, ComposeAudioTransform):
            return ComposeAudioTransform(self.transforms + itm.transforms)
        else:
            return ComposeAudioTransform(self.transforms + [itm])

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


## UTILITY


def apply_transform_to_list(transform, data, time=None, **kwargs):
    if time is None:
        outs = [transform(data[i], **kwargs) for i in range(len(data))]
        return outs
    else:
        outs = [transform(data[i], time=time[i], **kwargs) for i in range(len(data))]
        return [o[0] for o in outs], [o[1] for o in outs]


class Mono(AudioTransform):
    def __init__(self, mode="mix", normalize=False, squeeze=True, sr=44100):
        self.mode = mode
        self.normalize = normalize
        self.squeeze = squeeze

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return [self(x_tmp) for x_tmp in x]
        if x.shape[0] == 2:
            if self.mode == "mix":
                x = (x.sum(0) / 2)[np.newaxis]
            elif self.mode == "right":
                x = x[1][np.newaxis]
            elif self.mode == "left":
                x = x[0][np.newaxis]
        if self.normalize:
            x = x / x.max()
        if self.squeeze:
            x = x.squeeze(0)
        if time is None:
            return x
        else:
            return x, time

    def invert(self, x, **kwargs):
        if self.squeeze:
            x = x[np.newaxis]
        if x.shape[0] == 1:
            x = torch.cat([x, x], dim=0)
        return x


class Stereo(AudioTransform):
    def __init__(self, normalize=False, sr=44100):
        self.normalize = normalize

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
    def __init__(self, nfft=2048, hop_size=None, win_size=None, sr=44100, hps=False, hps_margin= 1.0, hps_power=2):
        self.nfft = nfft
        self.hop_size = hop_size or nfft // 4
        self.win_size = win_size or self.nfft
        self.sr = sr
        self.hps = hps
        self.hps_margin = hps_margin
        self.hps_power = hps_power
        self.griffin_lim = torchaudio.transforms.GriffinLim(self.nfft, win_length=self.win_size, hop_length = self.hop_size, power=1.0)

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time, sr=self.sr)
        out = torch.stft(x, self.nfft, self.hop_size, win_length=self.win_size, return_complex=True).transpose(-2, -1)
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
            out = torch.istft(x.transpose(-2, -1), self.nfft, self.hop_size, self.win_size)
        else:

            out = self.griffin_lim(x.transpose(-2, -1))
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


class MultiResolutionSTFT(AudioTransform):
    invertible = False
    def __init__(self, resolutions=[256,512,1024,2048], overlap=2, hps=False, hps_margin= 1.0, hps_power=2, sr=44100):
        hop_size = max(resolutions)//overlap
        self.stfts = [None]*len(resolutions)
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

## Non-Stationary Gabor Transform

class NSGT(AudioTransform):
    scale_hash = {'oct':OctScale, 'mel':MelScale, 'lin':LinScale, 'log':LogScale}

    def __init__(self, ls=None, fmin=40, fmax=22050, scale="oct", bins=24, downsample=10, sr=44100, hps=False, hps_margin= 1.0, hps_power=2, n_iter = 32):
        self.srcale = self.scale_hash[scale](fmin, fmax, bins)
        self.downsample = downsample
        self.ls = ls
        self.sr = sr
        self.hps = hps
        self.hps_margin = hps_margin
        self.hps_power = hps_power
        self.n_iter = 32

    def __call__(self, x, *args, time=None, **kwargs):
        if isinstance(x, list):
            return [self(x_tmp) for x_tmp in x]

        ls = self.ls or x.shape[-1]
        nsgt = NNSGT(self.srcale, self.sr, int(ls * self.sr), real=False, matrixform=True)
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

        # as the signal is assumed real, we can split in half the spectrum
        out = out[..., :int(out.shape[-1]/2)]

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
        nsgt = NNSGT(self.srcale, self.sr, int(self.ls * self.sr), real=False, matrixform=True)

        if torch.is_tensor(x):
            if x.grad is not None:
                print('[Warning] performing iNSGT on a backwarded tensor')
            x = x.detach().cpu().numpy()
        #TODO upsample

        needs_squeeze = len(x.shape) == 2
        if needs_squeeze:
            x = x[np.newaxis]
        if self.downsample:
            x_data = np.linspace(0, 1, x.shape[-2])
            x_interp = np.linspace(0, 1, x.shape[-2]*self.downsample)
            interp = interp1d(x_data, x, axis=-2)
            x = interp(x_interp)
        x = np.concatenate([x, x[..., ::-1]], axis=-1)

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
        inv = nsgt.backward(spec)
        return inv


    def get_frequencies(self, as_notes=False, as_midi=False):
        freq, _ = self.scale()
        if as_notes:
            freq = librosa.hz_to_note(freq)
        elif as_midi:
            freq = librosa.hz_to_midi(freq)
        return freq


def pitch_hash():
    return {'C':0, 'Db':1, 'D-':1, 'C#':1, 'C♯':1, 'D':2, 'E-':3, 'Eb':3, 'D♯':3, 'D#':3, 'E':4, 'F':5, 'F#':6, 'F♯':6,
            'Gb':6, 'G-':6, 'G':7, 'G#':8, 'G♯':8, 'A-':8, 'Ab':8, 'A':9, 'A♯':10, 'A#':10, 'B-':10, 'Bb':10, 'B':11, 'C-':11}

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

class Harmonic(AudioTransform):
    def __init__(self, margin=8):
        self.margin = margin

    def __call__(self, x, *args, **kwargs):
        return torch.from_numpy(librosa.effects.harmonic(y=x.cpu().numpy(), margin=self.margin)).to(x.device)


class Normalize(AudioTransform):
    def __init__(self, mode="minmax", scale="bipolar"):
        super(Normalize, self).__init__()
        self.mode = mode or "minmax"
        self.polarity = scale or "bipolar"
        self.mean = None
        self.max = None

    @property
    def needs_scaling(self):
        return self.mean is None or self.max is None

    @staticmethod
    def get_stats(data, mode="gaussian", polarity="bipolar"):
        if mode == "minmax":
            if polarity == "bipolar":
                mean = (torch.max(data) - torch.sign(torch.min(data))*torch.min(data)) / 2
                max = torch.max(torch.abs(data - mean))
            elif polarity == "unipolar":
                mean = torch.min(data); max = torch.max(torch.abs(data))
        elif mode == "gaussian":
            if polarity=='bipolar':
                mean = torch.mean(data); max = torch.std(data)
            if polarity=='unipolar':
                mean = torch.min(data); max = torch.std(data)
        return mean, max

    def scale(self, data):
        if issubclass(type(data), list):
            stats = torch.Tensor([self.get_stats(d, self.mode, self.polarity) for d in data])
            if self.mode == "minmax":
                self.mean = torch.mean(stats[:,0]); self.max = torch.max(stats[:,1])
            else:
                self.mean = torch.min(stats[:, 0]);
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


class Magnitude(AudioTransform):
    def __init__(self, normalize=None, contrast=None, shrink=1, global_norm=True, **kwargs):
        super(Magnitude, self).__init__()
        self.normalize = None
        self.constrast = contrast
        if normalize is not None:
            self.normalize = Normalize(**normalize)
        self.shrink = shrink
        self.global_norm = global_norm

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
            return torch.log(x/self.shrink)
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
            x = torch.exp(x)*self.shrink
        elif self.constrast == "log1p":
            x = torch.expm1(x)*self.shrink
        if time is None:
            return x
        else:
            return x, time


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
        return x.unsqueeze(self.dim), time

    def invert(self, x, *args, time=None, sr=None, **kwargs):
        if isinstance(x, list):
            return apply_transform_to_list(self, x, time=time, sr=sr)
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
        return data, time

    def invert(self, x, **kwargs):
        return self(x, **kwargs)


class Window(AudioTransform):
    invertible = True
    def __init__(self, window_size, dim, hop_size=None, batch_dim=0, split_time=False, pad=False, inversion="crop"):
        self.window_size = window_size
        self.hop_size = hop_size or self.window_size
        assert self.window_size >= self.hop_size
        self.dim = dim
        self.batch_dim = batch_dim
        self.split_time = split_time
        self.pad = pad
        self.inversion = inversion

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
