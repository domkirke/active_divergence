from math import isnan
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist, sys, pdb
sys.path.append('../')
from active_divergence.modules import autoregressive as ar_models
from active_divergence.utils import checklist, checkdir
from omegaconf import OmegaConf
from active_divergence import losses 

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage


class AutoRegressive(pl.LightningModule):
    def __init__(self, config=None, autoregressive=None, training=None, losses=None, input_dim=None, **kwargs):
        super().__init__()
        config = config or OmegaConf.create()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        config.autoregressive = config.get('autoregressive') or autoregressive
        config.autoregressive.args.input_dim = input_dim
        self.input_dim = input_dim
        self.autoregressive = getattr(ar_models, config.autoregressive.type)(config.autoregressive.args)
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.generate(x)
        loss, individual_losses = self.get_loss(x, out)
        for i, l in enumerate(individual_losses):
            self.log("%s/train"%self.loss_labels[i], l, prog_bar=False)
        self.log("loss/train", loss, prog_bar=True) 
        return loss

    def generate(self, x):
        return self.autoregressive(x)

    def get_loss(self, batch, out):
        if isinstance(out, dist.Distribution):
            generation_size = out.batch_shape[-1]
        else:
            generation_size = out.shape[-(len(self.input_dim) + 1)]
        out_target = batch[..., -generation_size:]
        losses = []
        for i, l in enumerate(self.losses):
            current_loss = l(out, out_target)
            losses.append(self.losses_weight[i] * current_loss)
        return sum(losses), [l.detach() for l in losses]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.generate(x)
        loss, individual_losses = self.get_loss(x, out)
        for i, l in enumerate(individual_losses):
            self.log("%s/valid"%self.loss_labels[i], l, prog_bar=False)
        self.log("loss/valid", loss, prog_bar=True)
        return loss

    def reconstruct(self, x, *args, sample_latent=False, sample_data=False, chunk_size=None, **kwargs):
        if isinstance(x, (list, tuple)):
            x, y = x
        x = x.to(self.device)
        out = self.generate(x)
        if isinstance(out, dist.Categorical):
            out = out.probs.argmax(dim=-1)
        if out.shape[-1] < x.shape[-1]:
            out = torch.cat([torch.zeros((*out.shape[:-1], x.shape[-1] - out.shape[-1])).to(out.device), out], -1)
        return x.cpu(), out

