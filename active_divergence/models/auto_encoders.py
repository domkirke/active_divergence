import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist, sys, pdb
sys.path.append('../')
from active_divergence.modules import encoders
from active_divergence.utils import checklist
from omegaconf import OmegaConf
from active_divergence.losses import distortion, regularization, priors
import pytorch_lightning as pl


class AutoEncoder(pl.LightningModule):
    def __init__(self, config=None, encoder=None, decoder=None, training=None, latent=None, data=None, **kwargs):
        super().__init__()
        config = config or OmegaConf.create()
        if isinstance(config, dict):
            config = OmegaConf(config)
        # latent config
        config.latent = config.get('latent') or latent
        self.latent = config.latent
        # architecture
        config.encoder = config.get('encoder') or encoder
        config.encoder.args = config.encoder.get('args', {})
        config.encoder['args']['input_dim'] = list(data.shape)
        config.encoder['args']['target_shape'] = config.latent.dim
        config.encoder['args']['target_dist'] = config.latent.dist
        encoder_type = config.encoder.type or "MLPEncoder"
        self.encoder = getattr(encoders, encoder_type)(config.encoder.args)
        config.decoder = config.get('decoder', decoder)
        config.decoder.args = config.decoder.get('args', {})
        config.decoder.args.input_dim = config.latent.dim
        config.decoder.args.target_shape = data.shape
        decoder_type = config.decoder.type or "MLPDecoder"
        self.decoder = getattr(encoders, decoder_type)(config.decoder.args)

        # loss
        config.training = config.get('training', training)
        rec_config = config.training.get('reconstruction', OmegaConf.create())
        self.reconstruction_loss = getattr(distortion, rec_config.get('type', "LogDensity"))(**rec_config.get('args', {}))
        reg_config = config.training.get('regularization', OmegaConf.create())
        self.regularization_loss = getattr(regularization, reg_config.get('type', "KLD"))(**reg_config.get('args',{}))
        self.prior = getattr(priors, config.training.get('prior', "isotropic_gaussian"))
        # record config
        self.config = config
        self.save_hyperparameters(dict(self.config))

    @property
    def device(self):
        return next(self.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.training.lr or 1e-3)
        return optimizer

    def encode(self, x):
        return self.forward(x)

    def sample(self, z_params):
        if isinstance(z_params, dist.Distribution):
            if z_params.has_rsample:
                z = z_params.rsample()
            else:
                z = z_params.sample()
        else:
            z = z_params
        return z

    def decode(self, z):
        return self.decoder(z)

    def full_forward(self, x, batch_idx=None, trace=None, sample=True):
        if isinstance(x, (tuple, list)):
            x, y = x
        z_params = self.encoder(x, trace={})
        if sample:
            z = self.sample(z_params)
        else:
            z = z_params.mean
        x_hat = self.decoder(z, trace={})
        return x_hat, z_params, z

    def loss(self, batch, x, z_params, z, **kwargs):
        rec_loss = self.reconstruction_loss(x, batch)
        prior = self.prior(z.shape, device=batch.device)
        reg_loss = self.regularization_loss(prior, z_params)
        beta = self.config.training.beta if self.config.training.beta is not None else 1.0
        if self.config.training.warmup and (kwargs.get('epoch') is not None):
            beta = min(int(kwargs.get('epoch')) / self.config.training.warmup, beta)
        return rec_loss + beta * reg_loss, (rec_loss.detach(), reg_loss.detach())

    def training_step(self, batch, batch_idx):
        batch, y = batch
        # training_step defined the train loop.
        x, z_params, z = self.full_forward(batch, batch_idx)
        loss, (rec_loss, reg_loss) = self.loss(batch, x, z_params, z, epoch=self.trainer.current_epoch)
        self.log("rec_loss/train", rec_loss, prog_bar=True)
        self.log("reg_loss/train", reg_loss, prog_bar=True)
        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, y = batch
        x, z_params, z = self.full_forward(batch, batch_idx)
        loss, (rec_loss, reg_loss) = self.loss(batch, x, z_params, z)
        self.log("rec_loss/valid", rec_loss, prog_bar=True)
        self.log("reg_loss/valid", reg_loss, prog_bar=True)
        self.log("loss/valid", loss, prog_bar=True)
        return loss

    # External methods
    def forward(self, x, *args, **kwargs):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def reconstruct(self, x, *args, sample_latent=False, sample_data=False, only_data=False, **kwargs):
        if isinstance(x, (tuple, list)):
            x, y = x
        x_out, _, _ = self.full_forward(x.to(self.device), sample=sample_latent)
        if sample_data and isinstance(x_out, dist.Distribution):
            x_out = [x_out.sample()]
        elif isinstance(x_out, dist.Normal):
            x_out = [x_out.mean, x_out.stddev]
        elif isinstance(x_out, dist.Distribution):
            x_out = [x_out.mean]
        #TODO baaah
        if only_data:
            x_out = x_out[1]
        return x, x_out

    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        temperature = checklist(temperature)
        generations = []
        for t in temperature:
            z = torch.randn((n_samples, self.latent.dim), device=self.device) * t
            x = self.decode(z)
            if sample:
                x = x.sample()
            else:
                x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)

    def trace_from_inputs(self, x):
        if isinstance(x, (tuple, list)):
            x, y = x
        trace = {}
        reconstructions, z_params, z = self.full_forward(x.to(self.device), trace=trace)
        full_trace = {'embeddings':{'latent':z_params.mean},
                      'histograms':{'latent_std': z_params.stddev}}
        return full_trace

