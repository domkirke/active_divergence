from active_divergence.utils.config import ConfigItem
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist, sys, pdb, re
sys.path.append('../')
from active_divergence.models.model import Model, ConfigType
from active_divergence.models.gans import GAN, parse_additional_losses
from active_divergence.modules import encoders
from active_divergence.utils import checklist, checkdir, trace_distribution
from omegaconf import OmegaConf, ListConfig
from active_divergence.losses import get_regularization_loss, get_distortion_loss, priors
from typing import Dict, Union, Tuple


class AutoEncoder(Model):
    def __init__(self, config: OmegaConf=None, **kwargs):
        """
        Base class for auto-encoding architectures. Configuration file must include:
        - encoder
        - decoder
        - latent
        - training

        Args:
            config (Config): full configuration
            **kwargs:
        """
        # manage config
        config = config or OmegaConf.create()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        config_checkpoint = config.get('config_checkpoint')
        super().__init__(config=config)

        # input config
        self.input_size = config.get('input_size') or kwargs.get('input_size')
        # latent configs
        config.latent = config.get('latent')
        self.latent = config.latent

        # encoder architecture
        config.encoder = config.get('encoder')
        config.encoder.args = config.encoder.get('args', {})
        if config.encoder['args'].get('input_size') is None:
            config.encoder['args']['input_size'] = self.input_size
        if config.encoder['args'].get('target_shape') is None:
            config.encoder['args']['target_shape'] = config.latent.dim
        config.encoder['args']['target_dist'] = config.latent.dist
        encoder_type = config.encoder.type or "MLPEncoder"
        self.encoder = getattr(encoders, encoder_type)(config.encoder.args)

        # decoder architecture
        config.decoder = config.get('decoder')
        config.decoder.args = config.decoder.get('args', {})
        config.decoder.args.input_size = config.latent.dim
        if config.decoder.args.get('input_size') is None:
            config.decoder.args.input_size = config.latent.dim
        if config.decoder.args.get('target_shape') is None:
            config.decoder.args.target_shape = self.input_size
        decoder_type = config.decoder.type or "MLPDecoder"
        self.decoder = getattr(encoders, decoder_type)(config.decoder.args)

        # loss
        config.training = config.get('training')
        rec_config = config.training.get('reconstruction', OmegaConf.create())
        self.reconstruction_loss = get_distortion_loss(rec_config)
        reg_config = config.training.get('regularization', OmegaConf.create())
        self.regularization_loss = get_regularization_loss(reg_config)
        self.prior = getattr(priors, config.training.get('prior', "isotropic_gaussian"))

        # load from checkpoint
        if config_checkpoint:
            self.import_checkpoint(config_checkpoint)
        # record configs
        self.save_config(self.config)

    def configure_optimizers(self):
        optimizer_config = self.config.training.get('optimizer', {'type':'Adam'})
        optimizer_args = optimizer_config.get('args', {'lr':1e-4})
        parameters = self.get_parameters(self.config.training.get('optim_params'))
        optimizer = getattr(torch.optim, optimizer_config['type'])(parameters, **optimizer_args)
        if self.config.training.get('scheduler'):
            scheduler_config = self.config.training.get('scheduler')
            scheduler_args = scheduler_config.get('args', {})
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_config.type)(optimizer, **scheduler_args)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss/valid"}
        else:
            return optimizer

    # External methods
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # in lightning, forward defines the prediction/inference actions
        return self.encode(x, *args, **kwargs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def sample(self, z_params: Union[dist.Distribution, torch.Tensor]) -> torch.Tensor:
        """Samples a latent distribution."""
        if isinstance(z_params, dist.Distribution):
            if z_params.has_rsample:
                z = z_params.rsample()
            else:
                z = z_params.sample()
        else:
            z = z_params
        return z

    def decode(self, z: torch.Tensor):
        """Decode an incoming tensor."""
        return self.decoder(z)

    def reinforce(self, x, z, mode="forward"):
        """callback used for adversarial reinforcement"""
        if mode == "forward":
            return self.reconstruct(x)[1]
        elif mode == "latent":
            return self.decode(z)

    def full_forward(self, x: torch.Tensor, batch_idx: int=None,  sample: bool=True) -> Dict:
        """encodes and decodes an incoming tensor."""
        if isinstance(x, (tuple, list)):
            x, y = x
        z_params = self.encoder(x)
        if sample:
            z = self.sample(z_params)
        else:
            z = z_params.mean
        x_hat = self.decoder(z)
        return x_hat, z_params, z

    def trace(self, x: torch.Tensor, sample: bool = False):
        trace_model = {'encoder': {}, 'decoder' : {}}
        if isinstance(x, (tuple, list)):
            x, y = x
        x = x.to(self.device)
        z_params = self.encoder(x, trace = trace_model['encoder'])
        if sample:
            z = self.sample(z_params)
        else:
            z = z_params.mean
        x_params = self.decoder(z, trace = trace_model['decoder'])
        trace = {}
        trace['embeddings'] = {'latent': z_params.mean, **trace.get('embeddings', {})}
        trace['histograms'] = {**trace.get('histograms', {}),
                               **trace_distribution(z_params, name="latent", scatter_dim=True),
                               **trace_distribution(x_params, name="out"),
                               **trace_model}
        return trace


    def loss(self, batch, x, z_params, z, drop_detail = False, **kwargs):
        rec_loss = self.reconstruction_loss(x, batch, drop_detail=drop_detail)
        prior = self.prior(z.shape, device=batch.device)
        reg_loss = self.regularization_loss(prior, z_params, drop_detail=drop_detail)
        if drop_detail:
            rec_loss, rec_losses = rec_loss
            reg_loss, reg_losses = reg_loss
        beta = self.config.training.get('beta', 1.0)
        if self.config.training.get('warmup') and (kwargs.get('epoch') is not None):
            beta = min(int(kwargs.get('epoch')) / self.config.training.warmup, beta)
        loss = rec_loss + beta * reg_loss
        if drop_detail:
            return loss, {"full_loss": loss.cpu().detach(), **reg_losses, **rec_losses}
        else:
            return loss

    def training_step(self, batch, batch_idx):
        batch, y = batch
        # training_step defined the train loop.
        x, z_params, z = self.full_forward(batch, batch_idx)
        loss, losses = self.loss(batch, x, z_params, z, epoch=self.trainer.current_epoch, drop_detail=True)
        self.log_losses(losses, "train", prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, y = batch
        x, z_params, z = self.full_forward(batch, batch_idx)
        loss, losses = self.loss(batch, x, z_params, z, drop_detail=True)
        self.log_losses(losses, "valid", prog_bar=True)
        return loss

    def reconstruct(self, x, *args, sample_latent=False, sample_data=False, **kwargs):
        if isinstance(x, (tuple, list)):
            x, y = x
        x_out, _, _ = self.full_forward(x.to(self.device), sample=sample_latent)
        if isinstance(x_out, dist.Distribution):
            if sample_data:
                x_out = x_out.sample()
            else:
                x_out = x_out.mean
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


class InfoGAN(GAN):
    def __init__(self, config=None, encoder=None, decoder=None, discriminator=None, training=None, latent=None, **kwargs) -> None:
        super(GAN, self).__init__()
        if isinstance(config, dict):
            config = OmegaConf(config)
        else:
            config = OmegaConf.create()

        input_size = config.get('input_size') or kwargs.get('input_size')
        # setup latent
        config.latent = config.get('latent') or latent or {}
        self.prior = getattr(priors, config.latent.get('prior', "isotropic_gaussian"))
       # latent configs
        config.latent = config.get('latent') or latent
        self.latent = config.latent
        # encoder architecture
        config.encoder = config.get('encoder') or encoder
        config.encoder.args = config.encoder.get('args', {})
        if config.encoder['args'].get('input_size') is None:
            config.encoder['args']['input_size'] = config.get('input_size') or kwargs.get('input_size')
        if config.encoder['args'].get('target_shape') is None:
            config.encoder['args']['target_shape'] = config.latent.dim
        config.encoder['args']['target_dist'] = config.latent.dist
        encoder_type = config.encoder.type or "MLPEncoder"
        self.encoder = getattr(encoders, encoder_type)(config.encoder.args)
        # decoder architecture
        config.decoder = config.get('decoder', decoder)
        config.decoder.args = config.decoder.get('args', {})
        config.decoder.args.input_size = config.latent.dim
        if config.decoder.args.get('input_size') is None:
            config.decoder.args.input_size = config.latent.dim
        if config.decoder.args.get('target_shape') is None:
            config.decoder.args.target_shape = config.get('input_size') or kwargs.get('input_size')
        decoder_type = config.decoder.type or "MLPDecoder"
        self.decoder = getattr(encoders, decoder_type)(config.decoder.args) 
        # setup discriminator
        config.discriminator = config.get('discriminator') or discriminator 
        config.discriminator.args.input_size = input_size
        self.init_discriminator(config.discriminator)
        # setup training
        config.training = config.get('training') or training
        config.training.mode = config.training.get('mode', 'adv')
        assert config.training.mode in self.gan_modes
        self.automatic_optimization = False
        self.reconstruction_losses = parse_additional_losses(config.training.get('rec_losses'))
        reg_config = config.training.get('regularization_loss', OmegaConf.create())
        self.regularization_loss = getattr(regularization, reg_config.get('type', "KLD"))(**reg_config.get('args',{}))
        self.prior = getattr(priors, config.training.get('prior', "isotropic_gaussian"))

        self.config = config
        self.save_hyperparameters(dict(self.config))
        self.__dict__['generator'] = self.decoder

    def get_parameters(self, parameters=None, model=None, prefix=None):
        model = model or self
        if parameters is None:
            params = list(model.parameters())
        else:
            full_params = dict(model.named_parameters())
            full_params_names = list(full_params.keys())
            full_params_names_prefix = full_params_names if prefix is None else [f"{prefix}.{n}" for n in full_params_names]
            params = []
            for param_regex in parameters:
                valid_names = list(filter(lambda x: re.match(param_regex, full_params_names_prefix[x]), range(len(full_params_names))))
                params.extend([full_params[full_params_names[i]] for i in valid_names])
        return params

    def configure_optimizers(self):
        gen_p = self.get_parameters(self.config.get('optim_params'), self.encoder, "encoder")
        gen_p += self.get_parameters(self.config.get('optim_params'), self.decoder, "decoder")
        dis_p = self.get_parameters(self.config.get('optim_params'), self.discriminator, "discriminator")
        if len(gen_p) == 0:
            dis_opt = torch.optim.Adam(dis_p, 1e-4, (.5, .9))
            return dis_opt
        elif len(dis_p) == 0:
            gen_opt = torch.optim.Adam(gen_p, 1e-4, (.5, .9))
            return gen_opt
        else:
            dis_opt = torch.optim.Adam(dis_p, 1e-4, (.5, .9))
            gen_opt = torch.optim.Adam(gen_p, 1e-4, (.5, .9))
            return gen_opt, dis_opt    

    def encode(self, x):
        return self.encoder(x)

    def sample_prior(self, batch=None, shape=None):
        if batch is None:
            return super().sample_prior(batch=batch, shape=shape)
        else:
            z = self.encode(batch)
            return z

    def generate(self, x=None, z=None, sample=True, **kwargs):
        if isinstance(z, dist.Distribution):
            z = z.rsample()
        out = self.decoder(z, **kwargs)
        return out

    def generator_loss(self, generator, batch, out, d_fake, z_params, hidden=None, **kwargs):
        adv_loss = super().generator_loss(generator, batch, out, d_fake, z_params, hidden=hidden)
        z = z_params.sample()
        prior = self.prior(z.shape, device=batch.device)
        reg_loss = self.regularization_loss(prior, z_params)
        beta = self.config.training.beta if self.config.training.beta is not None else 1.0
        if self.config.training.warmup:
            beta = min(int(self.trainer.current_epoch) / self.config.training.warmup, beta)
        return adv_loss + beta * reg_loss

        
        
