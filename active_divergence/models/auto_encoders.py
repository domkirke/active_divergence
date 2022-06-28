from yaml import serialize
from active_divergence.utils.config import ConfigItem
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sys, pdb, re
sys.path.append('../')
from active_divergence.models.model import Model, ConfigType
from active_divergence.models.gans import GAN, parse_additional_losses
from active_divergence.modules import encoders, OverlapAdd
from active_divergence import distributions as dist
from active_divergence.utils import checklist, checkdir, trace_distribution, reshape_batch, flatten_batch
from omegaconf import OmegaConf, ListConfig
from sklearn import decomposition 
from active_divergence.losses import get_regularization_loss, get_distortion_loss, priors, regularization
from typing import Dict, Union, Tuple, Callable


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
        self.input_shape = config.get('input_shape') or kwargs.get('input_shape')
        # latent configs
        config.latent = config.get('latent')
        self.latent = config.latent

        # encoder architecture
        config.encoder = config.get('encoder')
        config.encoder.args = config.encoder.get('args', {})
        if config.encoder['args'].get('input_shape') is None:
            config.encoder['args']['input_shape'] = self.input_shape
        if config.encoder['args'].get('target_shape') is None:
            config.encoder['args']['target_shape'] = config.latent.dim
        config.encoder['args']['target_dist'] = config.latent.dist
        encoder_type = config.encoder.type or "MLPEncoder"
        self.encoder = getattr(encoders, encoder_type)(config.encoder.args)

        # decoder architecture
        config.decoder = config.get('decoder')
        config.decoder.args = config.decoder.get('args', {})
        config.decoder.args.input_shape = config.latent.dim
        if config.decoder.args.get('input_shape') is None:
            config.decoder.args.input_shape = config.latent.dim
        if config.decoder.args.get('target_shape') is None:
            config.decoder.args.target_shape = self.input_shape
        decoder_type = config.decoder.type or "MLPDecoder"
        self.decoder = getattr(encoders, decoder_type)(config.decoder.args)

        # loss
        config.training = config.get('training')
        rec_config = config.training.get('reconstruction', OmegaConf.create())
        self.reconstruction_loss = get_distortion_loss(rec_config)
        reg_config = config.training.get('regularization', OmegaConf.create())
        self.regularization_loss = get_regularization_loss(reg_config)
        self.prior = getattr(priors, config.training.get('prior', "isotropic_gaussian"))

        # dimensionality reduction
        dim_red_type = config.get('dim_red', "PCA")
        self.dimred_type = getattr(decomposition, dim_red_type)
        self.dimred_batches = config.get('dim_red_batches', 128)
        self.register_buffer("latent_dimred", torch.eye(config.latent.dim))
        self.register_buffer("latent_mean", torch.zeros(config.latent.dim))
        self.register_buffer("fidelity", torch.zeros(config.latent.dim))
        self._latent_buffer = []

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

    def get_beta(self):
        beta = self.config.training.beta if self.config.training.beta is not None else 1.0
        if self.config.training.warmup:
            beta_schedule_type = self.config.training.get('beta_schedule_type', "epoch")
            if beta_schedule_type == "epoch":
                beta = min(int(self.trainer.current_epoch) / self.config.training.warmup, beta)
            elif beta_schedule_type == "batch":
                beta = min(int(self.trainer.global_step) / self.config.training.warmup, beta)
        return beta

    # External methods
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # in lightning, forward defines the prediction/inference actions
        return self.encode(x, *args, **kwargs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def sample(self, z_params: Union[dist.Distribution, torch.Tensor]) -> torch.Tensor:
        """Samples a latent distribution."""
        if isinstance(z_params, dist.Distribution):
            z = z_params.rsample()
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

    @torch.jit.export
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
        beta = self.get_beta()
        loss = rec_loss + beta * reg_loss
        if drop_detail:
            return loss, {"full_loss": loss.cpu().detach(), **reg_losses, **rec_losses}
        else:
            return loss

    def _update_latent_buffer(self, z_params):
        if isinstance(z_params, dist.Normal):
            z_params = z_params.mean
        if len(self._latent_buffer) <= self.dimred_batches:
            self._latent_buffer.append(z_params.detach().cpu())

    def training_step(self, batch, batch_idx):
        batch, y = batch
        # training_step defined the train loop.
        x, z_params, z = self.full_forward(batch, batch_idx)
        loss, losses = self.loss(batch, x, z_params, z, epoch=self.trainer.current_epoch, drop_detail=True)
        losses['beta'] = self.get_beta()
        self.log_losses(losses, "train", prog_bar=True)
        self._update_latent_buffer(z_params)
        return loss

    def on_train_epoch_end(self):
        # shamelessly taken from Caillon's RAVE
        latent_pos = torch.cat(self._latent_buffer, 0).reshape(-1, self.config.latent.dim)
        self.latent_mean.copy_(latent_pos.mean(0))

        dimred = self.dimred_type(n_components=latent_pos.size(-1))
        dimred.fit(latent_pos.cpu().numpy())
        components = dimred.components_
        components = torch.from_numpy(components).to(latent_pos)
        self.latent_dimred.copy_(components)

        var = dimred.explained_variance_ / np.sum(dimred.explained_variance_)
        var = np.cumsum(var)
        self.fidelity.copy_(torch.from_numpy(var).to(self.fidelity))

        var_percent = [.8, .9, .95, .99]
        for p in var_percent:
            self.log(f"{p}%_manifold",
                    np.argmax(var > p).astype(np.float32))
        self.trainer.logger.experiment.add_image("latent_dimred", components.unsqueeze(0).numpy(), self.trainer.current_epoch)
        self._latent_buffer = []
        
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

    def get_scripted(self, mode: str = "audio", script: bool=True, **kwargs):
        if mode == "audio":
            scriptable_model = ScriptableAudioAutoEncoder(self, **kwargs)
        else:
            raise ValueError("error while scripting model %s : mode %s not found"%(type(self), mode))
        if script:
            return torch.jit.script(scriptable_model)
        else:
            return scriptable_model



class InfoGAN(GAN):
    def __init__(self, config=None, encoder=None, decoder=None, discriminator=None, training=None, latent=None, **kwargs) -> None:
        super(GAN, self).__init__()
        if isinstance(config, dict):
            config = OmegaConf(config)
        else:
            config = OmegaConf.create()

        input_shape = config.get('input_shape') or kwargs.get('input_shape')
        # setup latent
        config.latent = config.get('latent') or latent or {}
        self.prior = getattr(priors, config.latent.get('prior', "isotropic_gaussian"))
       # latent configs
        config.latent = config.get('latent') or latent
        self.latent = config.latent
        # encoder architecture
        config.encoder = config.get('encoder') or encoder
        config.encoder.args = config.encoder.get('args', {})
        if config.encoder['args'].get('input_shape') is None:
            config.encoder['args']['input_shape'] = config.get('input_shape') or kwargs.get('input_shape')
        if config.encoder['args'].get('target_shape') is None:
            config.encoder['args']['target_shape'] = config.latent.dim
        config.encoder['args']['target_dist'] = config.latent.dist
        encoder_type = config.encoder.type or "MLPEncoder"
        self.encoder = getattr(encoders, encoder_type)(config.encoder.args)
        # decoder architecture
        config.decoder = config.get('decoder', decoder)
        config.decoder.args = config.decoder.get('args', {})
        config.decoder.args.input_shape = config.latent.dim
        if config.decoder.args.get('input_shape') is None:
            config.decoder.args.input_shape = config.latent.dim
        if config.decoder.args.get('target_shape') is None:
            config.decoder.args.target_shape = config.get('input_shape') or kwargs.get('input_shape')
        decoder_type = config.decoder.type or "MLPDecoder"
        self.decoder = getattr(encoders, decoder_type)(config.decoder.args) 
        # setup discriminator
        config.discriminator = config.get('discriminator') or discriminator 
        config.discriminator.args.input_shape = input_shape
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

        
        
#%% Scriptable Auto-Encoders

class ScriptableAudioAutoEncoder(nn.Module):
    def __init__(self, auto_encoder: AutoEncoder, transform: Union[Callable, None] = None, 
                 use_oa: bool = False, win_length: int = None, hop_length: int = None,
                 use_dimred: bool = True, export_for_nn: bool = True):
        super().__init__()
        self.transform = transform
        self.encoder = auto_encoder.encoder
        self.encoder_type = auto_encoder.encoder.target_dist
        self.decoder = auto_encoder.decoder
        self.decoder_type = auto_encoder.encoder.target_dist
        # overlap add
        win_length = win_length or self.encoder.input_shape[-1]
        hop_length = hop_length or hop_length
        self.overlap_add = OverlapAdd(win_length, hop_length, transform=transform)
        self.use_oa = use_oa
        if self.use_oa:
            self.transform = self.transform.realtime()
        # nn~ parameters
        # by default, input is considered as raw; hence the ratio is the input
        self.register_buffer("forward_params", torch.tensor([self.encoder.input_shape[0], 1, self.decoder.target_shape[0], 1]))
        self.register_buffer("encode_params", torch.tensor([self.encoder.input_shape[0], 1, auto_encoder.config.latent.dim, hop_length]))
        self.register_buffer("decode_params", torch.tensor([auto_encoder.config.latent.dim, hop_length, self.decoder.target_shape[0], 1]))
        # dim reduction parameters
        self.register_buffer("latent_dimred", auto_encoder.latent_dimred)
        self.register_buffer("latent_mean", auto_encoder.latent_mean)
        self.register_buffer("fidelity", auto_encoder.fidelity)
        self.use_dimred = use_dimred
        self.export_for_nn = export_for_nn

    def project_z(self, z:torch.Tensor):
        z, batch_size = flatten_batch(z, dim=-2)
        z = z - self.latent_mean
        z = nn.functional.conv1d(z.transpose(-1,-2), self.latent_dimred.unsqueeze(-1)).transpose(-1,-2)
        z = reshape_batch(z, batch_size, dim=-2)
        return z

    def unproject_z(self, z:torch.Tensor):
        z, batch_size = flatten_batch(z, dim=-2)
        z = nn.functional.conv1d(z.transpose(-1,-2), self.latent_dimred.t().unsqueeze(-1)).transpose(-1,-2)
        z = z + self.latent_mean
        z = reshape_batch(z, batch_size, dim=-2)
        return z

    @torch.jit.export
    def encode(self, x: torch.Tensor, sample: bool = False):
        if self.use_oa:
            x = self.overlap_add(x)
            x_transformed = []
            for i in range(x.size(-2)):
                x_tmp = x[..., i, :]
                if self.transform is not None:
                    x_transformed.append(self.transform(x_tmp))
            x = torch.stack(x_transformed, dim=-(len(self.encoder.input_shape) + 1))
        else:
            if self.transform is not None:
                x = self.transform(x)
        z = self.encoder(x)
        if sample:
            out = z.sample()
        else:
            out = z.mean 
        out = self.project_z(out)
        if self.export_for_nn:
            out = out.transpose(-2, -1)
        return out

    @torch.jit.export
    def decode(self, z:torch.Tensor):
        if self.export_for_nn:
            z = z.transpose(-2, -1)
        if self.use_dimred:
            z = self.unproject_z(z)
        x = self.decoder(z).mean
        if self.use_oa:
            outs = []
            iter_dim = -(len(self.encoder.input_shape) + 1)
            for i in range(x.size(iter_dim)):
                x_tmp = x.index_select(-(len(self.encoder.input_shape) + 1), torch.tensor(i)).squeeze(iter_dim)
                if self.transform is not None:
                    x_tmp = self.transform.invert(x_tmp)
                outs.append(x_tmp)
            outs = torch.stack(outs, dim=-2)
            x = self.overlap_add.invert(outs)
        else:
            if self.transform is not None:
                x = self.transform.invert(x)
        return x 

    @torch.jit.export
    def forward(self, x: torch.Tensor, sample: bool = False):
        if self.use_oa:
            x = self.overlap_add(x)
            outs = []
            for i in range(x.size(-2)):
                x_tmp = x[..., i, :]
                if self.transform is not None:
                    x_tmp = self.transform(x_tmp)
                z = self.encoder(x_tmp)
                if sample:
                    decoder_input = z.sample()
                else:
                    decoder_input = z.mean
                x_rec = self.decoder(decoder_input).mean
                if self.transform is not None:
                    x_rec = self.transform.invert(x_rec)
                outs.append(x_rec)
            outs = torch.stack(outs, -2)
            outs = self.overlap_add.invert(outs)
        else:
            if self.transform is not None:
                x = self.transform(x)
            z = self.encoder(x)
            if sample:
                decoder_input = z.sample()
            else:
                decoder_input = z.mean
            outs = self.decoder(decoder_input).mean
            if self.transform is not None:
                outs = self.transform.invert(outs)
        return outs
