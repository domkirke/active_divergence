import torch, torch.distributions as dist, torchvision as tv, pdb, random, matplotlib.pyplot as plt, os, trajectories, numpy as np, re
import torchaudio, tqdm
import dill
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from active_divergence.utils import checkdir

class ImgReconstructionMonitor(Callback):

    def __init__(self, n_reconstructions=5, n_morphings=5, n_samples=5, n_files=1,
                 temperature_range=None, monitor_epochs=1, reconstruction_epochs=5):
        self.n_reconstructions = n_reconstructions
        self.temperature_range = temperature_range or [0.1, 1.0, 3.0, 5.0, 10.0]
        self.n_samples = n_samples
        self.monitor_epochs = monitor_epochs

    def plot_reconstructions(self, model, loader):
        data = next(loader(batch_size=self.n_reconstructions).__iter__())
        x_original, x_out = model.reconstruct(data)
        value_range = [x_original.min(), x_original.max()]
        x_out = [x_tmp.cpu() for x_tmp in x_out]
        out = torch.stack([x_original, *x_out], 0).reshape((len(x_out) + 1) * x_original.shape[0], *x_original.shape[1:])
        return tv.utils.make_grid(out, nrow=x_original.shape[0], value_range=value_range)

    def plot_samples(self, model):
        out = model.sample_from_prior(n_samples=self.n_samples, temperature=self.temperature_range)
        if isinstance(out, dist.Distribution):
            out = out.mean
        #out = out.transpose(0, 1)
        out = out.reshape(out.size(0) * out.size(1), *out.shape[2:])
        full_img = tv.utils.make_grid(out, nrow=self.n_samples, value_range=[0, 1]) 
        return full_img

    def reconstruct_file(self, model, files, dataset):
        originals = []; generations = []
        for f in files:
            data, meta = dataset.transform_file(f"{dataset.root_directory}/{f}")
            original, generation = model.reconstruct(data)
            originals.append(dataset.invert_transform(original))
            generations.append(dataset.invert_transform(generation))
        originals, generations = torch.cat(originals, -1).squeeze(), torch.cat(generations, -1).squeeze()
        #pdb.set_trace()
        return check_mono(originals, normalize=True), check_mono(generations, normalize=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            if trainer.state.stage == RunningStage.SANITY_CHECKING:
                return
            if trainer.current_epoch % self.monitor_epochs != 0:
                return
            model = trainer.model
            dataset = None
            if hasattr(trainer.datamodule, "dataset"):
                dataset = trainer.datamodule.dataset
            # plot reconstructions
            if hasattr(model, 'reconstruct'):
                train_rec = self.plot_reconstructions(model, trainer.datamodule.train_dataloader)
                trainer.logger.experiment.add_image('reconstructions/train', train_rec, trainer.current_epoch)
                valid_rec = self.plot_reconstructions(model, trainer.datamodule.val_dataloader)
                trainer.logger.experiment.add_image('reconstructions/valid', valid_rec, trainer.current_epoch)
                if trainer.datamodule.test_dataset:
                    test_rec = self.plot_reconstructions(model, trainer.datamodule.test_dataloader)
                    trainer.logger.experiment.add_image('reconstructions/test', test_rec, trainer.current_epoch)

            # plot prior sampling
            if hasattr(model, 'sample_from_prior'):
                samples = self.plot_samples(model)
                trainer.logger.experiment.add_image('samples', samples, trainer.current_epoch)


def check_mono(sgl, normalize=True):
    sgl = sgl.squeeze()
    if sgl.ndim == 2:
        if sgl.shape[0] > 1:
            sgl = sgl.mean(0).unsqueeze(0)
    if normalize:
        sgl /= sgl.max()
    return sgl


class AudioReconstructionMonitor(Callback):

    def __init__(self, plot_reconstructions=True, plot_samples=True, 
                 generate_samples=True, generate_files=True, generate_trajs=True,
                 n_reconstructions=5, n_samples=5, n_files=3, files_path=None,
                 temperature_range=None, monitor_epochs=1, reconstruction_epochs=5,
                 sample_reconstruction=False,
                 traj_file=None, traj_sr = 172):
        # reconstruction arguments
        self.plot_reconstructions = plot_reconstructions
        self.n_reconstructions = n_reconstructions
        self.generate_files = generate_files
        self.files_path = files_path
        self.generate_trajs = int(generate_trajs or 0)
        self.traj_file = traj_file
        self.traj_sr = traj_sr
        self.n_files = n_files
        self.sample_reconstruction = sample_reconstruction
        # sample arguments
        self.plot_samples = plot_samples
        self.generate_samples = generate_samples
        self.n_samples = n_samples
        self.temperature_range = temperature_range or [0.01, 0.1, 0.5, 1.0, 1.5, 3.0, 5.0, 10.0]
        self.monitor_epochs = monitor_epochs
        self.reconstruction_epochs = reconstruction_epochs

    def plot_rec(self, model, loader):
        data = next(loader(batch_size=self.n_reconstructions).__iter__())
        x_original, x_out = model.reconstruct(data)
        x_original = x_original.squeeze()
        fig, ax = plt.subplots(len(x_original), 1)
        for i in range(len(x_original)):
            ax[i].plot(x_original[i], 'b', linewidth=1)
            if isinstance(x_out, dist.Normal):
                mean = x_out.mean[i].squeeze().cpu(); std = x_out.stddev[i].squeeze().cpu()
                ax[i].plot(mean, c="g", linewidth=0.7)
                ax[i].fill_between(np.arange(mean.shape[-1]), mean-std, mean+std, color="g", alpha=0.2)
            else:
                ax[i].plot(x_out[i].squeeze().cpu())
        return fig

    def plot_samps(self, model):
        out = model.sample_from_prior(n_samples=self.n_samples, temperature=self.temperature_range)
        if isinstance(out, dist.Distribution):
            out = out.mean
        fig, ax = plt.subplots(self.n_samples, len(self.temperature_range))
        for i in range(self.n_samples):
            for j in range(len(self.temperature_range)):
                ax[i, j].plot(out[i, j].squeeze().cpu())
        return fig

    def reconstruct_file(self, model, files, dataset):
        originals = []; generations = []
        for f in files:
            root_directory = self.files_path or dataset.root_directory
            data, meta = dataset.transform_file(f"{root_directory}/{f}")
            unsqueezed=False
            if len(data.shape) == 1:
                data = data.unsqueeze(0)
                unsqueezed = True
            original, generation = model.reconstruct(data) 
            if unsqueezed:
                original = original[0]; generation = generation[0]
            original = original.cpu()
            if isinstance(generation, dist.Normal):
                generation = generation.sample() if self.sample_reconstruction else generation.mean
            generation = generation.cpu()
            originals.append(dataset.invert_transform(original))
            generations.append(dataset.invert_transform(generation))
        originals, generations = torch.cat(originals, -1).squeeze(), torch.cat(generations, -1).squeeze()
        return check_mono(originals, normalize=True), check_mono(generations, normalize=True)

    def generate_trajectories(self, dataset, model, path):
        with open(self.traj_file, 'rb') as f:
            traj_dict = dill.load(f)
        path = re.sub("lightning_logs/", "trajectories/", path)
        checkdir(path)
        phase = np.linspace(0., 1., self.traj_sr)
        for traj_name, traj in tqdm.tqdm(traj_dict.items(), total=len(traj_dict)):
            out = traj(phase)
            out = torch.from_numpy(out)[np.newaxis].float().to(model.device)
            x = model.decode(out)
            if isinstance(x, dist.Distribution):
                x = x.mean
            x = check_mono(dataset.invert_transform(x), normalize=True)
            torchaudio.save(f"{path}/{traj_name}.wav", x, sample_rate = dataset.sr or 44100)
            
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            if trainer.state.stage == RunningStage.SANITY_CHECKING:
                return
            if trainer.current_epoch % self.monitor_epochs != 0:
                return
            model = trainer.model
            dataset = None
            if hasattr(trainer.datamodule, "dataset"):
                dataset = trainer.datamodule.dataset
            else:
                dataset = trainer.datamodule
            # plot and generate reconstructions
            if  hasattr(model, 'reconstruct'):
                if self.plot_reconstructions:
                    train_rec = self.plot_rec(model, trainer.datamodule.train_dataloader)
                    trainer.logger.experiment.add_figure('reconstructions/train', train_rec, trainer.current_epoch)
                    valid_rec = self.plot_rec(model, trainer.datamodule.val_dataloader)
                    trainer.logger.experiment.add_figure('reconstructions/valid', valid_rec, trainer.current_epoch)
                    if trainer.datamodule.test_dataset:
                        test_rec = self.plot_rec(model, trainer.datamodule.test_dataloader)
                        trainer.logger.experiment.add_figure('reconstructions/test', test_rec, trainer.current_epoch)
                if self.generate_files:
                    if trainer.current_epoch % self.reconstruction_epochs == 0:
                        if dataset is not None:
                            files = random.choices(dataset.files, k=self.n_files)
                            raw_original, raw_generation = self.reconstruct_file(model, files, dataset)
                            trainer.logger.experiment.add_audio('original', raw_original, global_step=trainer.current_epoch, sample_rate=dataset.sr)
                            trainer.logger.experiment.add_audio('generation', raw_generation, global_step=trainer.current_epoch,
                                                                sample_rate=dataset.sr)

            # plot prior sampling
            if hasattr(model, 'sample_from_prior'):
                if self.plot_samples:
                    samples = self.plot_samps(model)
                    trainer.logger.experiment.add_figure('samples', samples, trainer.current_epoch)

            # generate trajectories
            if hasattr(model, 'decode'):
                if self.generate_trajs and trainer.current_epoch % self.generate_trajs == 0:
                    if self.traj_file is None or not os.path.isfile(os.path.abspath(self.traj_file)):
                        print('[Warning] trajectory file missing : %s'%self.traj_file)
                    else:
                        self.generate_trajectories(dataset, model, trainer.log_dir)

