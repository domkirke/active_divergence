from math import floor, ceil
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist, sys, pdb
sys.path.append('../')
from active_divergence.modules import inversion as inv_models
from active_divergence.data.audio import parse_transforms, transforms as ad_transforms
from active_divergence.utils import checklist, checkdir
from omegaconf import OmegaConf
from active_divergence import losses 

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage


class Inversion(pl.LightningModule):
    def __init__(self, config=None, transforms=None, inversion=None, training=None, losses=None, input_dim=None, **kwargs):
        super().__init__()
        if config is None:
            config = OmegaConf.create()

        config.transforms = config.get('transforms') or transforms
        if config.transforms is None:
            transform = [ad_transforms.STFT(nfft=1024, hop_size=256), 
                         ad_transforms.Magnitude(normalize={"mode":"minmax", "scale":"bipolar"}, contrast="log")]
        else:
            transform = parse_transforms(transforms)
        self.transform = transform
        self.frame_dim = self.transform.frame_dim

        config.inversion = config.get('inversion') or inversion
        config.inversion.args.input_dim = self.transform.frame_dim
        self.input_dim = input_dim
        self.inversion = getattr(inv_models, config.inversion.type)(config.inversion.args)
        config.training = config.get('training') or training
        config.losses = config.get('losses') or losses
        # save configs and hyperparameters
        self.config = config
        self.save_hyperparameters(dict(self.config))
        self.configure_losses()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr or 1e-3)
        return optimizer

    def configure_losses(self):
        loss_list = []
        weights = []
        loss_labels = []
        for l in self.config.losses:
            current_loss = getattr(losses, l['type'])(**l.get('args', {}))
            loss_labels.append(l['type'])
            loss_list.append(current_loss)
            weights.append(float(l.get('weight', 1.0)))
        self.losses = loss_list
        self.losses_weight = weights
        self.loss_labels = loss_labels

    def invert(self, x):
        return self.inversion(x)

    def crop(self, original, generation):
        diff_shape = original.shape[-1] - generation.shape[-1]
        if diff_shape > 0:
            original = original[..., floor(diff_shape/2):-(ceil(diff_shape/2))]
        elif diff_shape < 0:
            generation = generation[..., floor(diff_shape/2):-(ceil(diff_shape/2))]
        return original, generation

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x, y = batch
        x_t = self.transform(x.cpu()).float().to(self.device)
        out = self.invert(x_t)
        x, out = self.crop(x, out)
        loss, individual_losses = self.get_loss(x, out)
        for i, l in enumerate(individual_losses):
            self.log("%s/train"%self.loss_labels[i], l, prog_bar=False)
        self.log("loss/train", loss, prog_bar=True) 
        return loss

    def get_loss(self, batch, out):
        losses = []
        for i, l in enumerate(self.losses):
            current_loss = l(out, batch)
            losses.append(self.losses_weight[i] * current_loss)
        return sum(losses), [l.detach() for l in losses]

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x, y = batch
        #TODO dumb
        x_t = self.transform(x.cpu()).float().to(self.device)
        out = self.invert(x_t)
        x, out = self.crop(x, out)
        loss, individual_losses = self.get_loss(x, out)
        for i, l in enumerate(individual_losses):
            self.log("%s/valid"%self.loss_labels[i], l, prog_bar=False)
        self.log("loss/valid", loss, prog_bar=True) 
        return loss

    def reconstruct(self, x, *args, sample_latent=False, sample_data=False, **kwargs):
        if isinstance(x, (list, tuple)):
            x, y = x
        x_t = self.transform(x).float()
        x_t = x_t.to(self.device)
        out = self.generate(x_t)
        x, out = self.crop(x, out)
        if isinstance(out, dist.Categorical):
            out = out.probs.argmax(dim=-1)
        if out.shape[-1] < x.shape[-1]:
            out = torch.cat([torch.zeros((*out.shape[:-1], x.shape[-1] - out.shape[-1])).to(out.device), out], -1)
        return x.cpu(), out

