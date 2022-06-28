import sys, torch, pdb, re, os
sys.path.append(os.path.dirname(__file__)+'/RAVE')

from rave import model
from active_divergence.utils import checklist
from omegaconf import OmegaConf, ListConfig

class RAVE(model.RAVE):
    def __init__(self, config: OmegaConf = None, checkpoint=None, transfer=None, **kwargs):
        if config is None:
            config = OmegaConf.create()
        # load keys from external configs in case
        config_checkpoint = config.get('checkpoint') or checkpoint
        if (config_checkpoint is not None) and transfer:
            config = self.import_checkpoint_config(config, config_checkpoint)
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
        dict_config['warmup'] = config.get('warmup') or kwargs.get('warmup', 1000000)
        dict_config['mode']  = config.get('mode') or kwargs.get('mode', "hinge")
        dict_config['no_latency'] = config.get('no_latency', False)
        dict_config['sr'] = config.get('sr', 44100)
        # load from checkpoint
        super(RAVE, self).__init__(**dict_config)
        dict_config['optim_params'] = config.get('optim_params') or kwargs.get('optim_params')
        self.config = OmegaConf.create(dict_config)
        if config_checkpoint:# and transfer:
            self.import_checkpoint(config_checkpoint)

    @property
    def device(self):
        return next(self.parameters().__iter__()).device

    @property
    def train_dataloader(self):
        return self.trainer.datamodule.train_dataloader
    
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
            if config_checkpoint.get('configs'):
                external_config = OmegaConf.load(config_checkpoint.config)
                if config_checkpoint.get('config_keys'):
                    for k in checklist(config_checkpoint.config_keys):
                        config[k] = external_config.model[k]
                else:
                    config = external_config.model
        return config

    def training_step(self, batch, batch_idx):
        x, y = batch
        return super(RAVE, self).training_step(x, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return  super(RAVE, self).validation_step(x, batch_idx)

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


