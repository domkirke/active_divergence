import torch, torch.distributions as dist

def isotropic_gaussian(shape):
    return dist.Normal(torch.zeros(*shape), torch.ones(*shape))
