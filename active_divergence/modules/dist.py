import torch, torch.nn as nn, torch.distributions as dist, pdb

#__all__ = ['MLPNormal', 'ConvNormal', "conv_dist_hash"]

class MLPBernoulli(nn.Module):
    def __init__(self, out_nnlin=None, dim=None):
        super(ConvNormal, self).__init__()
        self.out_nnlin = out_nnlin or nn.Sigmoid()
        self.dim = dim

    @property
    def required_dim_upsampling(self):
        return 1
    
    @property
    def required_channel_upsampling(self):
        return 1

    def forward(self, out):
        if self.out_nnlin:
            out = self.out_nnlin(out)
        return dist.Bernoulli(out)

class ConvBernoulli(MLPBernoulli):
    pass

class MLPNormal(nn.Module):
    def __init__(self, out_nnlin=None, dim=None):
        super(MLPNormal, self).__init__()
        self.out_nnlin = out_nnlin
        self.dim = dim

    @property
    def required_dim_upsampling(self):
        return 2

    def forward(self, out):
        dim = self.dim or -1
        mu, std = out.split(out.shape[-1]//2, dim=dim)
        if self.out_nnlin:
            mu = self.out_nnlin(mu)
        std = torch.sigmoid(std-3) + 1e-6
        #std = torch.sqrt(torch.exp(std) + 1e-6)
        return dist.Normal(mu, std)


class ConvNormal(nn.Module):
    def __init__(self, out_nnlin=None, dim=None):
        super(ConvNormal, self).__init__()
        self.out_nnlin = out_nnlin
        self.dim = dim

    @property
    def required_channel_upsampling(self):
        return 2

    def forward(self, out, out_nnlin=None):
        dim = 1 if self.dim is None else -(self.dim + 1)
        mu, std = out.split(out.shape[dim]//2, dim=dim)
        if self.out_nnlin:
            mu = self.out_nnlin(mu)
        std = torch.sigmoid(std - 3) + 1e-6
        #std = torch.sqrt(torch.exp(std) + 1e-6)
        return dist.Normal(mu, std)

class MLPCategorical(nn.Module):
    def __init__(self, out_nnlin=None, dim=-1):
        super(MLPCategorical, self).__init__()
        self.out_nnlin = out_nnlin or nn.Softmax(dim=-1)
        self.dim = dim

    @property
    def required_dim_upsampling(self):
        return 1
    
    @property
    def required_channel_upsampling(self):
        return 1

    def forward(self, out):
        if self.out_nnlin:
            out = self.out_nnlin(out)
        return dist.Categorical(out)

class ConvCategorical(MLPCategorical):
    pass

conv_dist_hash = {dist.Bernoulli: ConvBernoulli, dist.Normal: ConvNormal, dist.Categorical: ConvCategorical}
mlp_dist_hash = {dist.Bernoulli: MLPBernoulli, dist.Normal: MLPNormal, dist.Categorical: MLPCategorical}