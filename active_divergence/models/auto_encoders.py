from active_divergence.utils.config import ConfigItem
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist, sys, pdb, re
sys.path.append('../')
from active_divergence.models.gans import GAN, parse_additional_losses
from active_divergence.modules import encoders
from active_divergence.utils import checklist, checkdir
from omegaconf import OmegaConf, ListConfig
from active_divergence.losses import distortion, regularization, priors
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage


class AutoEncoder(pl.LightningModule):
    def __init__(self, config=None, encoder=None, decoder=None, training=None, latent=None, checkpoint=None, **kwargs):
        super().__init__()
        config = config or OmegaConf.create()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        # load keys from external config in case
        config_checkpoint = config.get('checkpoint') or checkpoint
        if config_checkpoint:
            config = self.import_checkpoint_config(config, config_checkpoint)
        self.input_dim = config.get('input_dim') or kwargs.get('input_dim')
        # latent config
        config.latent = config.get('latent') or latent
        self.latent = config.latent
        # encoder architecture
        config.encoder = config.get('encoder') or encoder
        config.encoder.args = config.encoder.get('args', {})
        if config.encoder['args'].get('input_dim') is None:
            config.encoder['args']['input_dim'] = self.input_dim
        if config.encoder['args'].get('target_shape') is None:
            config.encoder['args']['target_shape'] = config.latent.dim
        config.encoder['args']['target_dist'] = config.latent.dist
        encoder_type = config.encoder.type or "MLPEncoder"
        self.encoder = getattr(encoders, encoder_type)(config.encoder.args)
        # decoder architecture
        config.decoder = config.get('decoder', decoder)
        config.decoder.args = config.decoder.get('args', {})
        config.decoder.args.input_dim = config.latent.dim
        if config.decoder.args.get('input_dim') is None:
            config.decoder.args.input_dim = config.latent.dim
        if config.decoder.args.get('target_shape') is None:
            config.decoder.args.target_shape = self.input_dim
        decoder_type = config.decoder.type or "MLPDecoder"
        self.decoder = getattr(encoders, decoder_type)(config.decoder.args)

        # loss
        config.training = config.get('training', training)
        rec_config = config.training.get('reconstruction', OmegaConf.create())
        self.reconstruction_loss = getattr(distortion, rec_config.get('type', "LogDensity"))(**rec_config.get('args', {}))
        reg_config = config.training.get('regularization', OmegaConf.create())
        self.regularization_loss = getattr(regularization, reg_config.get('type', "KLD"))(**reg_config.get('args',{}))
        self.prior = getattr(priors, config.training.get('prior', "isotropic_gaussian"))

        # load from checkpoint
        if config_checkpoint:
            self.import_checkpoint(config_checkpoint)

        # record config
        if config.get('checkpoint'):
            del config['checkpoint']
        self.config = config
        self.save_hyperparameters(dict(self.config))

    @property
    def device(self):
        return next(self.parameters()).device

    def import_checkpoint(self, config_checkpoint: OmegaConf):
        if isinstance(config_checkpoint, ListConfig):
            for c in config_checkpoint:
                self.import_checkpoint(c)
        else:
            if isinstance(config_checkpoint, str):
                config_checkpoint = OmegaConf.create({'path': config_checkpoint})
            model = self.load_from_checkpoint(config_checkpoint.path)
            model_params = dict(model.named_parameters())
            model_params_names = list(model_params.keys())
            if config_checkpoint.get('params'):
                params_to_import = []
                for target_params in checklist(config_checkpoint.params):
                    p = list(filter(lambda x: re.match(target_params, x), model_params_names))
                    params_to_import.extend(p)
            else:
                params_to_import = model_params_names
            current_params = dict(self.named_parameters())
            for param_name in params_to_import:
                if current_params[param_name].shape == model_params[param_name].shape:
                    print('loading paramater %s from %s...'%(param_name, config_checkpoint.path))
                    current_params[param_name] = model_params[param_name]
                else:
                    print('non-matching shape for parameter %s ; skipping'%param_name)
            try:
                self.load_state_dict(current_params, strict=True)
            except Exception as e:
                incompatible, unexpected = self.load_state_dict(current_params, strict=False) 
                if len(incompatible) != 0:
                    print("Found incompatible keys :", incompatible)
                if len(unexpected) != 0:
                    print("Unexpected keys :", unexpected)
            

    def import_checkpoint_config(self, config: OmegaConf, config_checkpoint: OmegaConf):
        if isinstance(config_checkpoint, ListConfig):
            for c in config_checkpoint:
                self.import_checkpoint_config(config, c)
        else:
            if config_checkpoint.get('config'):
                external_config = OmegaConf.load(config_checkpoint.config)
                if config_checkpoint.get('config_keys'):
                    for k in checklist(config_checkpoint.config_keys):
                        config[k] = external_config.model[k]
                else:
                    config = external_config.model
        return config

    def get_parameters(self, parameters=None, model=None):
        model = model or self
        if parameters is None:
            params = model.parameters()
        else:
            full_params = dict(model.named_parameters())
            full_params_names = list(full_params.keys())
            params = []
            for param_regex in self.config.training.optim_params:
                valid_names = list(filter(lambda x: re.match(param_regex, x), full_params_names))
                params.extend([full_params[k] for k in valid_names])
        return params

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

    def reconstruct(self, x, *args, sample_latent=False, sample_data=False, **kwargs):
        if isinstance(x, (tuple, list)):
            x, y = x
        x_out, _, _ = self.full_forward(x.to(self.device), sample=sample_latent)
        if sample_data and isinstance(x_out, dist.Distribution):
            x_out = [x_out.sample()]
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

    def reinforce(self, x, z, mode="forward"):
        """callback used for adversarial reinforcement"""
        if mode == "forward":
            return self.reconstruct(x)[1]
        elif mode == "latent":
            return self.decode(z)

    def trace_from_inputs(self, x):
        if isinstance(x, (tuple, list)):
            x, y = x
        trace = {}
        reconstructions, z_params, z = self.full_forward(x.to(self.device), trace=trace)
        full_trace = {'embeddings':{'latent':z_params.mean},
                      'histograms':{**{'latent_std/dim_%i'%i: z_params.stddev[..., i] for i in range(z_params.stddev.shape[-1])},
                                    **{'latent/dim_%i'%i: z_params.mean[..., i] for i in range(z_params.mean.shape[-1])}}}
        return full_trace


class InfoGAN(GAN):
    def __init__(self, config=None, encoder=None, decoder=None, discriminator=None, training=None, latent=None, **kwargs) -> None:
        super(GAN, self).__init__()
        if isinstance(config, dict):
            config = OmegaConf(config)
        else:
            config = OmegaConf.create()

        input_dim = config.get('input_dim') or kwargs.get('input_dim')
        # setup latent
        config.latent = config.get('latent') or latent or {}
        self.prior = getattr(priors, config.latent.get('prior', "isotropic_gaussian"))
       # latent config
        config.latent = config.get('latent') or latent
        self.latent = config.latent
        # encoder architecture
        config.encoder = config.get('encoder') or encoder
        config.encoder.args = config.encoder.get('args', {})
        if config.encoder['args'].get('input_dim') is None:
            config.encoder['args']['input_dim'] = config.get('input_dim') or kwargs.get('input_dim')
        if config.encoder['args'].get('target_shape') is None:
            config.encoder['args']['target_shape'] = config.latent.dim
        config.encoder['args']['target_dist'] = config.latent.dist
        encoder_type = config.encoder.type or "MLPEncoder"
        self.encoder = getattr(encoders, encoder_type)(config.encoder.args)
        # decoder architecture
        config.decoder = config.get('decoder', decoder)
        config.decoder.args = config.decoder.get('args', {})
        config.decoder.args.input_dim = config.latent.dim
        if config.decoder.args.get('input_dim') is None:
            config.decoder.args.input_dim = config.latent.dim
        if config.decoder.args.get('target_shape') is None:
            config.decoder.args.target_shape = config.get('input_dim') or kwargs.get('input_dim')
        decoder_type = config.decoder.type or "MLPDecoder"
        self.decoder = getattr(encoders, decoder_type)(config.decoder.args) 
        # setup discriminator
        config.discriminator = config.get('discriminator') or discriminator 
        config.discriminator.args.input_dim = input_dim
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

        
        
