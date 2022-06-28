from argparse import ArgumentError
import sys, torch, pdb, re, os, numpy as np, torchaudio
from typing import Tuple
import torchaudio
sys.path.append(os.path.dirname(__file__)+'/ddsp_pytorch')
from ddsp import model, core
from active_divergence.utils import checklist, pad
from omegaconf import OmegaConf, ListConfig

__all__ = ['DDSP']

class DDSP(model.DDSP):
    def __init__(self, config: OmegaConf = None, checkpoint=None, transfer=None, **kwargs):
        if config is None:
            config = OmegaConf.create()
        # load keys from external configs in case
        config_checkpoint = config.get('checkpoint') or checkpoint
        if (config_checkpoint is not None) and transfer:
            config = self.import_checkpoint_config(config, config_checkpoint)
        # load from checkpoint
        super(DDSP, self).__init__(**config['model'])
        config['optim_params'] = config.get('optim_params') or kwargs.get('optim_params')
        self.config = OmegaConf.create(config)
        if config_checkpoint:# and transfer:
            self.import_checkpoint(config_checkpoint)

    def import_checkpoint(self, config_checkpoint: OmegaConf):
        if isinstance(config_checkpoint, ListConfig):
            for c in config_checkpoint:
                self.import_checkpoint(c)
        else:
            if isinstance(config_checkpoint, str):
                config_checkpoint = OmegaConf.create({'path': config_checkpoint})
            model_params = torch.load(config_checkpoint.path)
            self.config_names = list(model_params.keys())
            if config_checkpoint.get('params'):
                params_to_import = []
                for target_params in checklist(config_checkpoint.params):
                    p = list(filter(lambda x: re.match(target_params, x), self.config_names))
                    params_to_import.extend(p)
            else:
                params_to_import = self.config_names
            current_params = dict(self.named_parameters())
            for param_name in params_to_import:
                if not param_name in current_params:
                    current_params[param_name] = model_params[param_name]
                elif current_params[param_name].shape == model_params[param_name].shape:
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

    @property
    def device(self):
        return next(self.parameters().__iter__()).device


    def encode(self, filepath: str = None, audio: Tuple[torch.Tensor, int] = None, oneshot: bool = False):
        assert (filepath is not None) != (audio is not None), "either filepath and audio keywrod must be given"
        # adjust audio
        if filepath is not None:
            audio, sr = torchaudio.load(filepath)
        if audio is not None:
            audio, sr = audio
        if audio.ndim > 1:
            audio = audio [0]
        if (sr != self.sampling_rate):
            audio = torchaudio.functional.resample(audio, sr, self.sampling_rate)

        signal_length = self.config.preprocess.signal_length
        if oneshot:
            audio = audio[:signal_length]
        N = (signal_length - len(audio) % signal_length) % signal_length
        audio = torch.cat([audio, torch.zeros(N, dtype=audio.dtype, device=audio.device)], -1)
        
        # extract informations
        loudness = core.extract_loudness(audio.numpy(), int(self.sampling_rate), int(self.block_size))
        pitch = core.extract_pitch(audio.numpy(), int(self.sampling_rate), int(self.block_size))
        audio = audio.reshape(-1, signal_length)
        pitch = torch.from_numpy(pitch).float().reshape(audio.shape[0], -1, 1).to(self.device)
        loudness = torch.from_numpy(loudness).float().reshape(audio.shape[0], -1, 1).to(self.device)
        return pitch, loudness

    def decode(self, pitch, loudness):
        loudness = (loudness - self.config.data.mean_loudness) / self.config.data.std_loudness
        return super(DDSP, self).forward(pitch, loudness)


