import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist, sys, pdb
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus

from active_divergence.modules import encoders 
from active_divergence.utils import checklist, config
from active_divergence import losses
from active_divergence.losses import priors


def parse_additional_losses(config):
    if config is None:
        return None
    if isinstance(config, list):
        return [parse_additional_losses[c] for c in config]
    return getattr(losses, config.type)(**config.get('args', {}))
    
class GAN(pl.LightningModule):
    gan_modes = ['adv', 'hinge', 'wasserstein']
    def __init__(self, config=None, generator=None, discriminator=None, training=None, latent=None, **kwargs) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = OmegaConf(config)
        else:
            config = OmegaConf.create()

        input_dim = config.get('input_dim') or kwargs.get('input_dim')
        # setup latent
        config.latent = config.get('latent') or latent or {}
        self.prior = getattr(priors, config.latent.get('prior', "isotropic_gaussian"))
        # setup generator
        config.generator = config.get('generator') or generator 
        if config.latent.get('dim'):
            config.generator.args.input_dim = config.latent.dim
        config.generator.args.target_shape = input_dim
        self.init_generator(config.generator)
        if config.latent.get('dim') is None:
            config.latent.dim = self.generator.input_size
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

        self.config = config
        self.save_hyperparameters(dict(self.config))

    def init_generator(self, config: OmegaConf) -> None:
        generator_type = config.type or "DeconvEncoder"
        self.generator = getattr(encoders, generator_type)(config.args)
    
    def init_discriminator(self, config: OmegaConf) -> None:
        disc_type= config.type or "ConvEncoder"
        config.args.target_shape = 1
        self.discriminator = getattr(encoders, disc_type)(config.args)
    
    def configure_optimizers(self):
        lr = checklist(self.config.training.get('lr', 1e-4), n=2)
        if self.config.training.mode == "wasserstein":
            betas = (0.5, 0.999)
        else:
            betas = (0.9, 0.999)
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=lr[0] or 1e-4, betas=betas)
        disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr[1] or 1e-4, betas=betas)
        self.config.training.balance = self.config.training.get('balance')
        return gen_optim, disc_optim

    """
    @property
    def device(self):
        return next(self.parameters.__iter__()).device
    """

    def sample_prior(self, batch=None, shape=None):
        if batch is not None:
            batch_len = len(batch.shape) - len(self.generator.target_shape)
            z = self.prior((*batch.shape[:batch_len], *checklist(self.config.latent.dim))).sample()
        else:
            z = self.prior(*shape, **checklist(self.config.latent.dim)).sample()
        return z.to(self.device)

    def full_forward(self, batch, batch_idx=None, trace=None, sample=True):
        batch = batch.to(self.device)
        batch_len = len(batch.shape) - len(self.generator.target_shape)
        z = self.prior((*batch.shape[:batch_len], *checklist(self.config.latent.dim))).sample()
        x = self.generator(z.to(self.device))
        y_fake = self.discriminator(x)
        y_true = self.discriminator(batch)
        return z, x, (y_fake, y_true)

    def get_labels(self, out, phase):
        margin = float(self.config.training.get('margin', 0))
        invert = bool(self.config.training.get('invert_labels', False))
        if phase == 0:
            # get generator lables
            labels = torch.ones((*out.shape[:-1], 1), device=self.device) - margin
            if invert:
                labels = 1 - labels
            return labels
        elif phase == 1:
            true_labels = torch.ones((*out.shape[:-1], 1), device=self.device) - margin
            fake_labels = torch.zeros((*out.shape[:-1], 1), device=self.device) 
            if invert:
                true_labels = 1 - true_labels
                fake_labels = 1 - fake_labels
            return true_labels, fake_labels

    def step(self, gen_loss, disc_loss):
        g_opt, d_opt = self.optimizers()
        if self._loss_phase == 0 or self._loss_phase is None:
            g_opt.zero_grad()
            self.manual_backward(gen_loss)
            g_opt.step()
        if self._loss_phase == 1 or self._loss_phase is None:
            d_opt.zero_grad()
            self.manual_backward(disc_loss)
            d_opt.step()

    def on_train_epoch_start(self):
        if self.config.training.balance is not None:
            self._loss_phase = 0
            self._loss_counter = 0
        else:
            self._loss_phase = None
            self._loss_counter = None

    def on_validation_epoch_start(self):
        self._loss_phase = None

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        if self.config.training.balance is not None:
            self._loss_counter += 1
            if self._loss_counter >= self.config.training.balance[self._loss_phase]:
                self._loss_phase = (self._loss_phase + 1)%2
                self._loss_counter = 0

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x, return_hidden=False):
        return self.discriminator(x, return_hidden = return_hidden)

    def discriminator_loss(self, discriminator, batch, generation, d_real, d_fake):
        if self.config.training.mode == "adv":
            labels_real, labels_fake = self.get_labels(d_real, phase=1)
            true_loss = nn.functional.binary_cross_entropy(d_real, labels_real)
            fake_loss = nn.functional.binary_cross_entropy(d_fake, labels_fake)
            disc_loss = (true_loss + fake_loss)/2
        elif self.config.training.mode == "hinge":
            true_loss = torch.relu(1 - d_real).mean()
            fake_loss = torch.relu(1 + d_fake).mean()
            disc_loss = (true_loss + fake_loss) / 2
        elif self.config.training.mode == "wasserstein":
            disc_loss =  - d_real.mean() + d_fake.mean()

        if self.config.training.get('wdiv'):
            if d_real.grad_fn is not None:
                p = self.config.training.get('wdiv_exp', 6)
                weight = float(self.config.training.wdiv)
                labels_real, labels_fake = self.get_labels(d_real, phase=1)
                real_grad = torch.autograd.grad(
                    d_real, batch, labels_real, create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                fake_grad = torch.autograd.grad(
                    d_fake, generation, labels_fake, create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
                fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
                div_gp = torch.mean(real_grad_norm + fake_grad_norm) * weight / 2
                disc_loss = disc_loss + div_gp
        if self.config.training.get('gp'):
            if d_real.grad_fn is not None:
                weight = float(self.config.training.gp)
                gradient_penalty = self.compute_gradient_penalty(batch, generation)
                disc_loss = disc_loss + weight * gradient_penalty
        if self.config.training.get('r1'):
            if d_real.grad_fn is not None:
                weight = float(self.config.training.r1) / 2
                labels, _ = self.get_labels(d_real, phase=1)
                r1_reg = self.gradient_regularization(batch, labels)
                disc_loss = disc_loss + weight * r1_reg
        if self.config.training.get('r2'):
            if d_real.grad_fn is not None:
                weight = float(self.config.training.r2) / 2
                _, labels = self.get_labels(generation, phase=1)
                r2_reg = self.gradient_regularization(generation)
                disc_loss = disc_loss + weight * r2_reg
        return disc_loss

    def regularize_discriminator(self, discriminator, d_real, d_fake, d_loss):
        if self.config.training.mode == "wasserstein":
            clip_value = self.config.training.get('wgan_clip')
            if clip_value is not None:
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

    def generator_loss(self, generator, batch, out, d_fake, hidden=None):
        if self.config.training.mode == "adv":
            labels_real = self.get_labels(d_fake, phase=0)
            gen_loss = nn.functional.binary_cross_entropy(d_fake, labels_real)
        elif self.config.training.mode == "hinge":
            gen_loss = -d_fake.mean()
        elif self.config.training.mode == "wasserstein":
            gen_loss = -d_fake.mean()
        if self.config.training.get('feature_matching'):
            fm_loss = 0.
            for i in range(len(hidden[0]) - 1):
                feature_dim = len(hidden[0][i].shape) - len(d_fake.shape[:-1])
                sum_dims = tuple(range(0, -feature_dim, -1))
                #fm_loss = fm_loss + (hidden[0][i] - hidden[1][i]).pow(2).sum(sum_dims).sqrt().mean()
                fm_loss = fm_loss + (hidden[0][i] - hidden[1][i]).abs().mean()
            gen_loss = gen_loss + float(self.config.training['feature_matching']) * fm_loss
        return gen_loss

    def training_step(self, batch, batch_idx):
        batch, y = batch
        batch = batch.to(self.device)
        g_opt, d_opt = self.optimizers()
        g_opt.zero_grad()
        d_opt.zero_grad()
        # generate 
        z = self.sample_prior(batch=batch)
        out = self.generate(z)
        # update discriminator
        g_loss = d_loss = None
        loss = 0
        if self._loss_phase == 0 or self._loss_phase is None:
            d_real = self.discriminate(batch)
            d_fake = self.discriminate(out.detach())
            d_loss = self.discriminator_loss(self.discriminator, batch, out, d_real, d_fake)
            loss = loss + d_loss
            self.manual_backward(d_loss)
            d_opt.step()
            self.regularize_discriminator(self.discriminator, d_real, d_fake, d_loss)
        # update generator
        if self._loss_phase == 1 or self._loss_phase is None:
            if self.config.training.get('feature_matching'):
                d_fake, hidden_fake = self.discriminate(out, return_hidden=True)
                d_true, hidden_true = self.discriminate(batch, return_hidden=True)
                g_loss = self.generator_loss(self.generator, batch, out, d_fake, hidden=[hidden_true, hidden_fake])
            else:
                d_fake = self.discriminate(out, return_hidden=False)
                g_loss = self.generator_loss(self.generator, batch, out, d_fake)
            loss = loss + g_loss
            self.manual_backward(g_loss)
            g_opt.step()
        if g_loss is not None:
            self.log("gen_loss/train", g_loss.detach().cpu(), prog_bar=True)
        if d_loss is not None:
            self.log("disc_loss/train", d_loss.detach().cpu(), prog_bar=True)
        self.log("loss/train", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, y = batch
        batch = batch.to(self.device)
        # generate 
        z = self.sample_prior(batch=batch)
        out = self.generate(z)
        # get discriminator loss
        g_loss = d_loss = None
        d_real, hidden_real = self.discriminate(batch, return_hidden=True)
        d_fake, hidden_fake  = self.discriminate(out, return_hidden=True)
        d_loss = self.discriminator_loss(self.discriminator, batch, out, d_real, d_fake)
        # get generator loss
        g_loss = self.generator_loss(self.generator, batch, out, d_fake, hidden=[hidden_real, hidden_fake])
        if g_loss is not None:
            self.log("gen_loss/valid", g_loss.detach().cpu(), prog_bar=True)
        if d_loss is not None:
            self.log("disc_loss/valid", d_loss.detach().cpu(), prog_bar=True)
        loss = (g_loss or 0) + (d_loss or 0)
        self.log("loss/valid", loss, prog_bar=False)
        return loss

    # External methods
    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        temperature = checklist(temperature)
        generations = []
        for t in temperature:
            z = torch.randn((n_samples, *checklist(self.config.latent.dim)), device=self.device) * t
            x = self.generator(z)
            if isinstance(x, dist.Distribution):
                if sample:
                    x = x.sample()
                else:
                    x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)

    def gradient_regularization(self, inputs, labels):
        inputs.requires_grad_(True)
        d_out = self.discriminate(inputs)
        gradients = torch.autograd.grad(outputs=d_out, inputs=inputs, grad_outputs=labels, allow_unused=True, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        data_shape = len(real_samples.shape[1:])
        alpha = torch.rand((*real_samples.shape[:-data_shape], *(1,) * data_shape), device=real_samples.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples.detach() + ((1 - alpha) * fake_samples.detach()))
        interpolates.requires_grad_(True)
        d_interpolates = self.discriminate(interpolates)
        fake = torch.full((real_samples.shape[0], 1) ,1.0, device=real_samples.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

class ProgressiveGAN(GAN):
    def __init__(self, config=None, generator=None, discriminator=None, training=None, latent=None, **kwargs) -> None:
        super().__init__(config=config, generator=generator, discriminator=discriminator, training=training, latent=latent, **kwargs)
        self.config.training.training_schedule = checklist(self.config.training.get('training_schedule'), n=len(self.generator))
        self.config.training.transition_schedule = checklist(self.config.training.get('transition_schedule'), n=len(self.generator)-1)
        #self.init_rgb_modules()
        self.save_hyperparameters(dict(self.config))
        self._current_phase = None
        self._transition = None

    """"
    def init_rgb_modules(self):
        # prepare "toRGB modules"
        toRGB_modules = []
        module = conv_hash[self.generator.dim]
        for i in self.generator.channels[:-1]:
            toRGB_modules.append(module(i, self.discriminator.input_size[0], 1))
        self.generator.toRGB_modules = nn.ModuleList(toRGB_modules)
        fromRGB_modules = []
        module = conv_hash[self.discriminator.dim]
        # prepare "fromRGB modules"
        for i in self.discriminator.channels[1:]:
            fromRGB_modules.append(module(self.discriminator.input_size[0], i, 1))
        self.discriminator.fromRGB_modules = nn.ModuleList(fromRGB_modules)
    """

    def configure_optimizers(self):
        lr = checklist(self.config.training.get('lr', 1e-4), n=2)
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=lr[0] or 1e-4)
        disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr[1] or 1e-4)
        self.config.training.balance = self.config.training.get('balance')
        return gen_optim, disc_optim

    def _get_mixing_factor(self):
        if self._phase_counter > self.config.training.transition_schedule[self._current_phase-1]:
            return None
        alpha = min(1, float(self._phase_counter / self.config.training.transition_schedule[self._current_phase]))
        return alpha

    def generate(self, z):
        if self._transition == 1:
            alpha = self._get_mixing_factor()
            """
            generation = self.current_generator(z)
            generation_new = self.generator[self._current_phase](generation)
            generation_old = self._get_generator_upsample(self.generator[self._current_phase])(generation)
            if self._current_phase != len(self.config.training.training_schedule):
                generation_new = self.generator.toRGB_modules[self._current_phase](generation_new) 
            generation_old = self.generator.toRGB_modules[self._current_phase-1](generation_old)
            generation = generation_old * (1-alpha) + alpha * generation_new
            """
            generation = self.current_generator(z, transition=alpha)
        else:
            generation = self.current_generator(z)
            #if self._current_phase is not None and (self._current_phase != len(self.config.training.training_schedule)):
            #    generation = self.generator.toRGB_modules[self._current_phase](generation)
        return generation

    def discriminate(self, x):
        if self._transition == 1:
            alpha = self._get_mixing_factor()
            """
            if self._current_phase != len(self.config.training.training_schedule):
                x_new = self.discriminator.fromRGB_modules[-(self._current_phase+1)](x)
            else:
                x_new = x
            x_new = self.discriminator[-(self._current_phase+1)](x_new)
            x_old = self._get_discriminator_downsample(self.discriminator[-(self._current_phase+1)])(x)
            x_old = self.discriminator.fromRGB_modules[-(self._current_phase)](x_old)
            """
            disc = self.current_discriminator(x, transition=alpha)
        else:
            disc = self.current_discriminator(x)
        return disc

    def _get_sub_generator(self, phase):
        generator = self.generator[:phase+1]
        return generator

    def _get_sub_discriminator(self, phase):
        return self.discriminator[-(phase+1):]

    def _get_generator_upsample(self, module):
        ds_module = None
        if hasattr(module, "upsample"):
            ds_module = module.upsample
        elif isinstance(module, nn.Sequential):
            for m in module:
                ds_module = m.__dict__['_modules'].get('upsample')
                if ds_module is not None:
                    break
        return ds_module

    def _get_discriminator_downsample(self, module):
        ds_module = None
        if hasattr(module, "downsample"):
            ds_module = module.downsample
        elif hasattr(module, "conv_modules"):
            for m in module.conv_modules:
                ds_module = m.__dict__['_modules'].get('downsample')
                if ds_module is not None:
                    break
        return ds_module

    def _get_discriminator_downsamples(self, phase):
        downsamples = []
        for i in range(len(self.discriminator) - (phase+1)):
            ds_module = self._get_discriminator_downsample(self.discriminator[i])
            if ds_module is not None:
                downsamples.append(ds_module)
        return nn.Sequential(*downsamples)

    def training_step(self, batch, batch_idx):
        batch, y = batch
        batch = batch.to(self.device)
        g_opt, d_opt = self.optimizers()
        # generate 
        z = self.sample_prior(batch=batch)
        out = self.generate(z)
        # update discriminator

        # downsample image in case
        if self._current_phase is not None and self._current_phase != len(self.config.training.training_schedule):
            batch = self._get_discriminator_downsamples(self._current_phase)(batch)

        g_loss = d_loss = None
        if self._loss_phase == 0 or self._loss_phase is None:
            d_real = self.discriminate(batch)
            d_fake = self.discriminate(out.detach())
            d_loss = self.discriminator_loss(self.current_discriminator, batch, out, d_real, d_fake)
            d_opt.zero_grad()
            self.manual_backward(d_loss)
            d_opt.step()
            self.regularize_discriminator(self.current_discriminator, d_real, d_fake, d_loss)
        # update generator
        if self._loss_phase == 1 or self._loss_phase is None:
            d_fake = self.discriminate(out)
            g_loss = self.generator_loss(self.current_generator, batch, out, d_fake)
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            g_opt.step()
        if g_loss is not None:
            self.log("gen_loss/train", g_loss.detach().cpu(), prog_bar=True)
        if d_loss is not None:
            self.log("disc_loss/train", d_loss.detach().cpu(), prog_bar=True)
        loss = (g_loss or 0) + (d_loss or 0)
        self.log("loss/train", loss, prog_bar=False)
        return loss        

    def validation_step(self, batch, batch_idx):
        batch, y = batch
        batch = batch.to(self.device)
        # generate 
        z = self.sample_prior(batch=batch)
        out = self.generate(z)
        # get discriminator loss
        if self._current_phase is not None and self._current_phase != len(self.config.training.training_schedule):
            batch = self._get_discriminator_downsamples(self._current_phase)(batch)
        g_loss = d_loss = None
        d_real = self.discriminate(batch)
        d_fake = self.discriminate(out)
        d_loss = self.discriminator_loss(self.current_discriminator, batch, out, d_real, d_fake)
        # get generator loss
        g_loss = self.generator_loss(self.current_generator, batch, out, d_fake)
        if g_loss is not None:
            self.log("gen_loss/valid", g_loss.detach().cpu(), prog_bar=True)
        if d_loss is not None:
            self.log("disc_loss/valid", d_loss.detach().cpu(), prog_bar=True)
        loss = (g_loss or 0) + (d_loss or 0)
        self.log("loss/valid", loss, prog_bar=False)
        return loss

    def on_train_start(self):
        self._current_phase = 0
        self._phase_counter = 0
        # no transition at first epoch
        if self._current_phase == 0:
            self._transition = 0
            self.current_generator = self._get_sub_generator(self._current_phase)
            self.current_discriminator = self._get_sub_discriminator(self._current_phase)
        else:
            self._transition = 1
            self.current_generator = self._get_sub_generator(self._current_phase-1)
            self.current_discriminator = self._get_sub_discriminator(self._current_phase-1)

    def on_validation_start(self):
        if self._current_phase is None:
            self.on_train_start()

    def on_train_epoch_start(self):
        super(ProgressiveGAN, self).on_train_epoch_start()
        # check learning phase
        if self._current_phase != len(self.config.training.training_schedule):
            if self._phase_counter >= self.config.training.training_schedule[self._current_phase]:
                self._current_phase += 1
                self._transition = 1
                self._phase_counter = 0
                if self._current_phase == len(self.config.training.training_schedule):
                    self.current_generator = self.generator
                    self.current_discriminator = self.discriminator
                else:
                    self.current_generator = self._get_sub_generator(self._current_phase)
                    self.current_discriminator = self._get_sub_discriminator(self._current_phase)
        # check transitions 
        """
        tr = self.config.training.transition_schedule[self._current_phase]
        if (self._phase_counter >= tr or tr == 0) and self._transition:
            # set modules at current phases
            if self._current_phase == len(self.config.training.training_schedule):
                self.current_generator = self.generator
                self.current_discriminator = self.discriminator
            else:
                self.current_generator = self._get_sub_generator(self._current_phase)
                self.current_discriminator = self._get_sub_discriminator(self._current_phase)
            self._transition = 0
        """

    def on_train_epoch_end(self):
        self._phase_counter += 1

    # External methods
    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        temperature = checklist(temperature)
        generations = []
        for t in temperature:
            mod = torch.randn((n_samples, *checklist(self.config.latent.dim)), device=self.device) * t
            if self._current_phase is not None:
                x = self._get_sub_generator(self._current_phase)(self.const, mod)
            else:
                x = self.generator(z)
            if isinstance(x, dist.Distribution):
                if sample:
                    x = x.sample()
                else:
                    x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)


class ModulatedGAN(ProgressiveGAN):
    def __init__(self, config=None, encoder=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        # build encoden
        self.config.encoder = self.config.get('encoder') or encoder  
        self.config.encoder['mode'] = self.config.encoder.get('mode', 'sequential')
        self.init_encoder(self.config.encoder)
        # build constant input
        self.const = nn.Parameter(torch.randn(self.generator.input_size))
        self.save_hyperparameters(dict(self.config))

    def init_encoder(self, encoder_config):
        encoder_args = encoder_config.get('args', {})
        if encoder_config.mode == "sequential":
            encoder_args['input_dim'] = self.config.latent.get('dim', 512)
            encoder_args['target_shape'] = encoder_args.get('target_shape', 512)
            self.encoder = getattr(encoders, encoder_config.get('type', 'MLPEncoder'))(encoder_args)
        else:
            dims = self.generator.channels
            encoder_list = []
            for d in dims:
                current_args = dict(encoder_args)
                current_args.target_shape = d
                encoder_list.append(getattr(encoders, encoder_config.get('type', 'MLPEncoder'))(encoder_args))
            self.encoder = nn.ModuleList(encoder_list)

    def get_modulations(self, z):
        if self.config.encoder.mode == "sequential":
            styles = self.encoder(z)
        else:
            styles = [enc(z) for enc in self.encoder]
        return styles

    def get_const(self, z):
        const = self.const.view(*(1,)*(z.ndim-1), *self.const.shape)
        return const.repeat(*z.shape[:-1], *(1,)*(self.generator.dim+1))

    def generate(self, z, eps=None):
        const = self.get_const(z)
        y = self.get_modulations(z)
        if self._transition == 1:
            alpha = self._get_mixing_factor()
            generation = self.current_generator(const, mod=y, transition=alpha)
        else:
            generation = self.current_generator(const, mod=y)
            #if self._current_phase is not None and (self._current_phase != len(self.config.training.training_schedule)):
            #    generation = self.generator.toRGB_modules[self._current_phase](generation)
        return generation

    def generator_loss(self, generator, batch, out, d_fake, hidden=None):
        g_loss = super(ModulatedGAN, self).generator_loss(generator, batch, out, d_fake, hidden=hidden)
        if self.config.training.path_length and g_loss.grad_fn is not None:
            path_length_penalty = self.path_length_penalty()
            g_loss = g_loss + float(self.config.training.path_penalty) * path_length_penalty
        return g_loss

    def path_length_penalty(self, n_batch=64, n_samples=8):
        z = torch.randn((n_batch*n_samples, *checklist(self.config.latent.dim)), device=self.device, requires_grad=True)
        y = self.get_modulations(z)
        const = self.get_const(z)
        generations = self.current_generator(const, mod=y)
        noise = torch.randn_like(generations) * np.sqrt(np.prod(generations.shape[-(self.generator.dim+1):]))
        corrupted = (generations * noise).sum()
        gradients = torch.autograd.grad(outputs=corrupted, inputs=y, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(n_batch, n_samples, gradients.shape[-1])
        path_lengths = gradients.pow(2).sum(-1).mean(-1).sqrt()
        self.path_means = (self.path_means + self.path_length_decay * (path_lengths.mean() - self.path_means)).detach()
        pl_penalty = (path_lengths - self.path_means).pow(2).mean()
        return pl_penalty

    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        temperature = checklist(temperature)
        generations = []
        for t in temperature:
            z = torch.randn((n_samples, *checklist(self.config.latent.dim)), device=self.device) * t
            if self._current_phase is not None:
                x = self._get_sub_generator(self._current_phase)(self.get_const(z), mod=z)
            else:
                x = self.generator(z)
            if isinstance(x, dist.Distribution):
                if sample:
                    x = x.sample()
                else:
                    x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)


