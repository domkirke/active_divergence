import torch, torch.nn, sys, pdb
import torch.distributions as dist
sys.path.append('../')
from active_divergence.losses.loss import Loss
from active_divergence.losses import loss_utils as utils
from active_divergence.utils import checklist

class LogDensity(Loss):
    def forward(self, params1, params2, sample=True):
        if isinstance(params2, dist.Distribution):
            if sample:
                if hasattr(params2, "rsample"):
                    params2 = params2.rsample()
                else:
                    params2 = params2.sample()
            else:
                params2 = params2.mean
                if params2 is None:
                    raise ValueError('Could not sample distribution %s in LogDensity'%params2)
        ld = self.reduce(-params1.log_prob(params2))
        return ld


class MSE(Loss):
    def __init__(self, reduction=None):
        super().__init__(reduction=reduction)

    def forward(self, x, target, sample=False, **kwargs):
        if isinstance(x, dist.Distribution):
            if sample:
                if x.has_rsample:
                    x = x.rsample()
                else:
                    if x.grad_fn is not None:
                        print('[Warning] sampling a tensor in a graph will cut backpropagation' )
                    x = x.sample()
            else:
                if isinstance(x, dist.Normal):
                    x = x.mean
                elif isinstance(x, (dist.Bernoulli, dist.Categorical)):
                    x = x.probs
        return self.reduce(torch.nn.functional.mse_loss(x, target, reduction="none"))


## AUDIO LOSSES
class LogCosh(Loss):
    def __init__(self, freq_weight=2e-5, freq_exponent=2, nfft=1024, **kwargs):
        super().__init__(**kwargs)
        self.freq_weight = freq_weight
        self.freq_exponent = freq_exponent
        self.nfft = nfft

    def forward(self, x, target):
        loss = torch.cosh(x - target).log().sum(-1)
        x_fft = torch.stft(x, self.nfft, self.nfft//2, onesided=True, return_as_complex=True).abs()
        target_fft = torch.stft(target, self.nfft, self.nfft//2, onesided=True, return_as_complex=True).abs()
        l1_mag = torch.abs(x_fft - target_fft)
        freq_bins = torch.arange(target_fft.shape[-1]).pow(self.freq_exponent)
        freq_bins = freq_bins.view(*(1,)*(len(freq_bins.shape)-1))
        loss = loss + freq_bins * l1_mag
        return loss

class ESR(Loss):
    """Error-Signal Rate"""
    def forward(self, x, target):
        return self.reduce(torch.abs(x - target).pow(2).sum(-1) / target.pow(2).sum(-1))

class SpectralLoss(Loss):
    def __init__(self, input_type="raw", losses={'mag_l2':{}}, nfft=512, hop_size=None, normalized=False, reduction=None, drop_individual=False):
        super().__init__(reduction=reduction)
        self.input_type = input_type
        self.losses = losses
        self.nfft = nfft
        self.hop_size = hop_size or self.nfft // 4
        self.normalized = normalized
        self.drop_individual = drop_individual

    def forward(self, x, target):
        if self.input_type == "raw":
            x_f = torch.stft(x, self.nfft, self.hop_size, normalized=self.normalized, onesided=True, return_complex=True)
            target_f = torch.stft(target, self.nfft, self.hop_size, normalized=self.normalized, onesided=True, return_complex=True)
            x_f = x_f.transpose(-2, -1); target_f = target_f.transpose(-2, -1)
        elif self.input_type == "mag":
            x_f = x * torch.exp(torch.zeros_like(x) * 1j)
            target_f = x * torch.exp(torch.zeros_like(target) * 1j)
        elif self.input_type == "phase_channel":
            x_f = x[..., 0, :] * torch.exp(x[..., 1, :] * 1j)
            target_f = target[..., 0, :] * torch.exp(target[..., 1, :] * 1j)
        elif self.input_type == "complex":
            x_f = x; target_f = target
        losses = []
        for loss_name, loss_args in self.losses.items():
            current_loss = getattr(utils, loss_name)(x_f, target_f, **loss_args)
            losses.append(self.reduce(current_loss))
        if self.drop_individual:
            return losses
        else:
            return sum(losses)

class MultiResSpectraLoss(Loss):
    def __init__(self, nffts=[256, 1024, 2048], hop_sizes=None, weights = None, reduction=None, drop_individual=False, **kwargs):
        super(MultiResSpectraLoss, self).__init__(reduction=reduction)
        hop_size = checklist(hop_sizes, n=len(nffts))
        self.weights = checklist(weights or 1, n=len(nffts))
        self.spectral_losses = []
        self.drop_individual = drop_individual
        for i, nfft in enumerate(nffts):
            hs = hop_size[i] or nfft // 4
            self.spectral_losses.append(SpectralLoss(nfft=nfft, hop_size=hs, drop_individual=drop_individual, **kwargs))

    def forward(self, x, target):
        losses = []
        for l in self.spectral_losses:
            losses.append(l(x, target))
        if self.drop_individual:
            return [sum([losses[j][i] for j in range(len(losses))]) for i in range(len(losses[0]))]
        else:
            return sum(losses)


# Spectral Losses
#TODO Multi-resolution spectral Losses
#TODO Perceptual weighting (pre-emphasis)

