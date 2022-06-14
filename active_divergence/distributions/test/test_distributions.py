import sys
sys.path.append('../')
from active_divergence import distributions as dist
import torch, pytest, torch.nn as nn

def test_uniform():
    pass

class NormalModule(nn.Module):
    def __init__(self, dist_type = dist.Normal):
        super().__init__()
    @torch.jit.export
    def encode(self, x: torch.Tensor):
        return dist.Normal(torch.zeros(10, 10, 10), torch.ones(10, 10, 10))
    @torch.jit.export
    def decode(self, x: torch.Tensor):
        return dist.Normal(torch.zeros(10, 10, 10), torch.ones(10, 10, 10))
    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        z_sample = z.sample()
        x_rec = self.decode(z_sample)
        return x_rec

def test_normal():
    # isotropic normal
    norm_dist = dist.Normal(torch.zeros(1000), torch.ones(1000))
    norm_samples = norm_dist.rsample()
    histogram = torch.histogram(norm_samples, 100, range=[-4., 4.])
    # test scripting
    module = NormalModule()
    module_scripted = torch.jit.script(module)
    
    
    

def test_bernoulli():
    pass