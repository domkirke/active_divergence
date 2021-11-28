import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist, sys
sys.path.append('../')
from active_divergence.modules import encoders
from active_divergence.utils import Config, checklist
from active_divergence.losses import distortion, regularization, priors
import pytorch_lightning as pl


class AutoEncoder(pl.LightningModule):
    def __init__(self, config=None, encoder=None, decoder=None, training=None, latent=None, data=None, **kwargs):
        super().__init__()
        config = config or Config()
        if isinstance(config, dict):
            config = Config(config)
        # latent config
        config.latent = config.latent or Config(latent)
        self.latent = config.latent
        # architecture
        config.encoder = config.encoder or Config(encoder)
        config.encoder.args.input_dim = data.shape
        config.encoder.args.target_shape = config.latent.dim
        config.encoder.args.target_dist = config.latent.dist
        encoder_type = config.encoder.type or "MLPEncoder"
        self.encoder = getattr(encoders, encoder_type)(config.encoder.args)
        config.decoder = config.decoder or Config(decoder)
        config.decoder.args.input_dim = config.latent.dim
        config.decoder.args.target_shape = data.shape
        decoder_type = config.decoder.type or "MLPDecoder"
        self.decoder = getattr(encoders, decoder_type)(config.decoder.args)
        # loss
        config.training = config.training or Config(training)
        rec_config = config.training.reconstruction or Config()
        self.reconstruction_loss = getattr(distortion, rec_config.type or "LogDensity")(**(rec_config.args or Config()).dict())
        reg_config = config.training.regularization or Config()
        self.regularization_loss = getattr(regularization, reg_config.type or "KLD")(**(reg_config.args or Config()).dict())
        self.prior = getattr(priors, config.training.prior or "isotropic_gaussian")
        # record config
        self.config = config
        self.save_hyperparameters(self.config.dict())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr or 1e-3)
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
        prior = self.prior(z.shape)
        reg_loss = self.regularization_loss(prior, z_params)
        beta = self.config.training.beta or 1.0
        if self.config.training.warmup and kwargs.get('epoch'):
            beta = min(int(kwargs.get('epoch')) / self.config.training.warmup, beta)
        return rec_loss + beta * reg_loss, (rec_loss, reg_loss)

    def training_step(self, batch, batch_idx):
        batch, y = batch
        # training_step defined the train loop.
        x, z_params, z = self.full_forward(batch, batch_idx)
        loss, (rec_loss, reg_loss) = self.loss(batch, x, z_params, z)
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

    def reconstruct(self, x, *args, sample_latent=False, sample_data=False, **kwargs):
        if isinstance(x, (tuple, list)):
            x, y = x
        x_out, _, _ = self.full_forward(x, sample=sample_latent)
        if sample_data and isinstance(x_out, dist.Distribution):
            x_out = x_out.sample()
        elif isinstance(x_out, dist.Distribution):
            x_out = x_out.mean
        return x, x_out

    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        temperature = checklist(temperature)
        generations = []
        for t in temperature:
            z = torch.randn((n_samples, self.latent.dim)) * t
            x = self.decode(z)
            if sample:
                x = x.sample()
            else:
                x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)

    def trace_from_inputs(self, x):
        if isinstance(x, tuple):
            x, y = x
        trace = {}
        reconstructions, z_params, z = self.full_forward(x, trace=trace)
        full_trace = {'embeddings':{'latent':z_params.mean},
                      'histograms':{'latent_std': z_params.stddev}}
        return full_trace

