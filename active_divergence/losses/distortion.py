import torch, torch.nn, sys
import torch.distributions as dist
sys.path.append('../')
from active_divergence.losses.loss import Loss

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



## AUDIO LOSSES

#TODO Spectral Losses
#TODO Multi-resolution spectral Losses
#TODO Perceptual weighting (pre-emphasis)

