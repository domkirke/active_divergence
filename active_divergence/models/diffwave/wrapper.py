from argparse import ArgumentError
import sys, torch, pdb, re, os, numpy as np, torchaudio
import torchaudio
sys.path.append(os.path.dirname(__file__)+'/diffwave/src')
from diffwave import model, preprocess
from active_divergence.utils import checklist
from omegaconf import OmegaConf, ListConfig

__all__ = ['DiffWave']

class DiffWave(model.DiffWave):
    def __init__(self, config: OmegaConf = None, checkpoint=None, transfer=None, **kwargs):
        if config is None:
            config = OmegaConf.create()
        # load keys from external configs in case
        config_checkpoint = config.get('checkpoint') or checkpoint
        if (config_checkpoint is not None) and transfer:
            config = self.import_checkpoint_config(config, config_checkpoint)
        # load from checkpoint
        super(DiffWave, self).__init__(config)
        config['optim_params'] = config.get('optim_params') or kwargs.get('optim_params')
        self.config = OmegaConf.create(config)
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
            model_params = torch.load(config_checkpoint.path)['model']
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
                if current_params[param_name].shape == model_params[param_name].shape:
                    # print('loading paramater %s from %s...'%(param_name, config_checkpoint.path))
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
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

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
        raise NotImplemented

    def preprocess(self, filename: str=None, audio=None):
        if filename is not None:
            audio, sr = torchaudio.load(filename)
            audio = torch.clamp(audio[0], -1.0, 1.0)
        elif audio is not None:
            audio, sr = audio
        else:
            raise ArgumentError("either filename or audio keyword must be given for the preprocess function")
        if self.config.sample_rate != sr:
            resample = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            audio = resample(audio)
        mel_args = {
            'sample_rate': self.config.sample_rate,
            'win_length': self.config.hop_samples * 4,
            'hop_length': self.config.hop_samples,
            'n_fft': self.config.n_fft,
            'f_min': 20.0,
            'f_max': self.config.sample_rate / 2.0,
            'n_mels': self.config.n_mels,
            'power': 1.0,
            'normalized': True,
        }
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(**mel_args)
        with torch.no_grad():
            spectrogram = mel_spec_transform(audio)
            spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
            spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
        return spectrogram

    # external methods
    def encode(self, x, *args, **kwargs):
        raise NotImplemented

    def decode(self, audio=None, spectrogram=None, crop_audio=False, fast_sampling=True, noise_schedule=None, inference_noise_schedule=None):
        # Change in notation from the DiffWave paper for fast sampling.
        # DiffWave paper -> Implementation below
        # --------------------------------------
        # alpha -> talpha
        # beta -> training_noise_schedule
        # gamma -> alpha
        # eta -> beta
        device = self.device
        if not self.config.unconditional:
            assert (audio is not None) or (spectrogram is not None), "conditional models need audio or spectrogram inputs"

        if audio is not None:
            audio, sr = audio
            if sr != self.config.sample_rate:
                resample = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                audio = resample(audio)
            if crop_audio:
                audio = audio[..., :self.config.audio_len]
            audio = audio.to(self.device)

        training_noise_schedule = np.array(noise_schedule or self.config.noise_schedule)
        if fast_sampling:
            inference_noise_schedule = training_noise_schedule
        else:
            inference_noise_schedule = np.array(inference_noise_schedule or self.config.inference_noise_schedule)

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)

        if not self.config.unconditional:
            if spectrogram is None:
                spectrogram = self.preprocess(audio)
            if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
                spectrogram = spectrogram.unsqueeze(0)
            spectrogram = spectrogram.to(device)
            audio = torch.randn(spectrogram.shape[0], self.config.hop_samples * spectrogram.shape[-1], device=device)
        else:
            if audio is None:
                audio = 0.2 * torch.randn(1, self.config.audio_len, device=device)

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
            audio = c1 * (audio - c2 * super(DiffWave, self).forward(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)
        return audio, self.config.sample_rate

    def forward(self, x, *args, **kwargs):
        raise NotImplemented

    def full_forward(self, x, add_noise=None):
        raise NotImplemented

    def reconstruct(self, x, *args, **kwargs):
        raise NotImplemented

    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        raise NotImplemented

    def trace_from_inputs(self, x):
        raise NotImplemented


