import sys, torch, pdb
sys.path.append('../')
from active_divergence.models.rave.rave import model
from active_divergence.utils import checklist
from omegaconf import OmegaConf

class RAVE(model.RAVE):
    def __init__(self, config: OmegaConf = None, **kwargs):
        if config is None:
            config = OmegaConf.create()
        dict_config = {}
        dict_config['data_size'] = config.get('data_size', 16)
        dict_config['capacity'] = config.get('capacity', 64)
        dict_config['latent_size'] = config.get('latent_size', 128)
        dict_config['ratios'] = checklist(config.get('ratios', [4,4,4,2]))
        dict_config['bias'] = config.get('bias', True)
        dict_config['loud_stride'] = config.get('loud_stride', 1)
        dict_config['use_noise'] = config.get('use_noise', True)
        dict_config['noise_ratios'] = checklist(config.get('noise_ratios', [4,4,4]))
        dict_config['noise_bands'] = config.get('noise_bands', 5)
        dict_config['d_capacity'] = config.get('d_capacity', 16)
        dict_config['d_multiplier'] = config.get('d_multiplier', 4)
        dict_config['d_n_layers'] = config.get('d_n_layers', 4)
        dict_config['warmup'] = config.get('warmup', 1000000)
        dict_config['mode']  = config.get('mode', "hinge")
        dict_config['no_latency'] = config.get('no_latency', False)
        dict_config['sr'] = config.get('sr', 44100)
        super(RAVE, self).__init__(**dict_config)

    @property
    def device(self):
        return next(self.parameters().__iter__()).device

    @property
    def train_dataloader(self):
        return self.trainer.datamodule.train_dataloader

    def training_step(self, batch, batch_idx):
        x, y = batch
        return super(RAVE, self).training_step(x, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return  super(RAVE, self).validation_step(x, batch_idx)


    # external methods
    def encode(self, x, *args, **kwargs):
        x = x.unsqueeze(-2).to(self.device)
        if self.pqmf is not None:  # MULTIBAND DECOMPOSITION
            x = self.pqmf(x)
        z_params = self.encoder(x)
        #z, kl = self.reparametrize(*z_params)
        return z_params

    def decode(self, z, *args, **kwargs):
        x = self.decoder(z, add_noise=self.warmed_up)
        if self.pqmf is not None:
            x = self.pqmf.inverse(x)
        return x

    def forward(self, x, *args, **kwargs):
        return self.encode(x)

    def full_forward(self, x, add_noise=None):
        add_noise = add_noise if add_noise is not None else self.warmed_up
        z_params = self.encode(x)
        z, _ = self.reparametrize(*z_params)
        x = self.decode(z, add_noise=add_noise)
        return x, z_params, z

    def reconstruct(self, x, *args, **kwargs):
        if isinstance(x, (list, tuple)):
            x, y = x
        out, _, _ = self.full_forward(x, add_noise=self.warmed_up)
        return x, out

    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        temperature = checklist(temperature)
        generations = []
        for t in temperature:
            z = torch.randn((n_samples, self.latent.dim), device=self.device) * t
            x = self.decode(z, add_noise=self.warmed_up)
            if sample:
                x = x.sample()
            else:
                x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)

    def trace_from_inputs(self, x):
        if isinstance(x, (list, tuple)):
            x, y = x
        x, z_params, z = self.full_forward(x)
        full_trace = {'histograms': {**{'latent_std/dim_%i' % i: z_params[1][..., i] for i in
                                        range(z_params[1].shape[-1])},
                                     **{'latent/dim_%i' % i: z_params[0][..., i] for i in
                                        range(z_params[0].shape[-1])}}}
        return full_trace


