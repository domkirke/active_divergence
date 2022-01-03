import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist, sys, pdb
sys.path.append('../')
from active_divergence.modules import oneshots
from active_divergence.utils import checklist, checkdir
from omegaconf import OmegaConf
from active_divergence.losses import distortion, regularization, priors
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage


class Regressor(pl.LightningModule):
    def __init__(self, config=None, regressor=None, training=None, input_dim=None, **kwargs):
        super().__init__()
        if config is None:
            config = OmegaConf.create()
        config.regressor = config.get('regressor') or regressor 
        config.regressor.args.input_dim = input_dim
        self.regressor = getattr(oneshots, config.regressor.type)(config.regressor.args)
        config.training = config.get('training') or training
        # save config and hyperparameters
        self.config = config
        self.save_hyperparameters(dict(self.config))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr or 1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        time = y['time']
        time = time.to(self.device)
        out = self.regressor(time)
        loss = torch.nn.functional.mse_loss(out, x)
        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        time = y['time']
        out = self.regressor(time)
        loss = torch.nn.functional.mse_loss(out, x)
        return loss

    def reconstruct(self, x, *args, sample_latent=False, sample_data=False, **kwargs):
        x, y = x
        time = y['time']
        x = x.to(self.device)
        time = time.to(self.device)
        out = self.regressor(time)
        return x.cpu(), out

