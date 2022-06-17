import torch, active_divergence.distributions as dist

def isotropic_gaussian(shape, device=torch.device('cpu')):
    return dist.Normal(torch.zeros(*shape, device=device), torch.ones(*shape, device=device))
