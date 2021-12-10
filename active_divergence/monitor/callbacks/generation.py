import torch, torch.distributions as dist, torchvision as tv, pdb, random, matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus

class ImgReconstructionMonitor(Callback):

    def __init__(self, n_reconstructions=5, n_morphings=5, n_samples=5, n_files=1,
                 temperature_range=None, monitor_epochs=1, reconstruction_epochs=5):
        self.n_reconstructions = n_reconstructions
        self.temperature_range = temperature_range or [0.01, 0.1, 1.0, 3.0, 5.0, 10.0]
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
        out = out.reshape(out.size(0) * out.size(1), *out.shape[2:])
        full_img = tv.utils.make_grid(out, nrow=self.n_samples + 1, value_range=[0, 1]) 
        return full_img

    def reconstruct_file(self, model, files, dataset):
        originals = []; generations = []
        for f in files:
            data, meta = dataset.transform_file(f"{dataset.root_directory}/{f}")
            original, generation = model.reconstruct(data)
            originals.append(dataset.invert_data(original))
            generations.append(dataset.invert_data(generation))
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

    def __init__(self, n_reconstructions=5, n_morphings=5, n_samples=5, n_files=1,
                 temperature_range=None, monitor_epochs=1, reconstruction_epochs=5):
        self.n_reconstructions = n_reconstructions
        self.n_samples = n_samples
        self.temperature_range = temperature_range or [0.01, 0.1, 1.0, 3.0, 5.0, 10.0]
        self.n_files = n_files
        self.n_morphings = n_morphings
        self.monitor_epochs = monitor_epochs
        self.reconstruction_epochs = reconstruction_epochs

    def plot_reconstructions(self, model, loader):
        data = next(loader(batch_size=self.n_reconstructions).__iter__())
        x_original, x_out = model.reconstruct(data)
        x_original.squeeze()
        x_out = [x.squeeze().cpu() for x in x_out]
        fig, ax = plt.subplots(len(x_original), 1)
        for i in range(len(x_original)):
            ax[i].plot(x_original[i], linewidth=1)
            for j in reversed(range(len(x_out))):
                ax[i].plot(x_out[j][i], linewidth=0.6)
        return fig

    def plot_samples(self, model):
        out = model.sample_from_prior(n_samples=self.n_samples, temperature=self.temperature_range)
        if isinstance(out, dist.Distribution):
            out = out.mean
        out = out.cpu()
        fig, ax = plt.subplots(self.n_samples, len(self.temperature_range))
        for i in range(self.n_samples):
            for j in range(len(self.temperature_range)):
                ax[i, j].plot(out[i, j])
        return fig

    def reconstruct_file(self, model, files, dataset):
        originals = []; generations = []
        for f in files:
            data, meta = dataset.transform_file(f"{dataset.root_directory}/{f}")
            original, generation = model.reconstruct(data, only_data=True)
            original = original.cpu()
            generation = generation.cpu()
            originals.append(dataset.invert_data(original))
            generations.append(dataset.invert_data(generation))
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
                trainer.logger.experiment.add_figure('reconstructions/train', train_rec, trainer.current_epoch)
                valid_rec = self.plot_reconstructions(model, trainer.datamodule.val_dataloader)
                trainer.logger.experiment.add_figure('reconstructions/valid', valid_rec, trainer.current_epoch)
                if trainer.datamodule.test_dataset:
                    test_rec = self.plot_reconstructions(model, trainer.datamodule.test_dataloader)
                    trainer.logger.experiment.add_figure('reconstructions/test', test_rec, trainer.current_epoch)
                # perform full audio reconstructions
                if trainer.current_epoch % self.reconstruction_epochs == 0:
                    if dataset is not None:
                        files = random.choices(dataset.files, k=self.n_files)
                        raw_original, raw_generation = self.reconstruct_file(model, files, dataset)
                        trainer.logger.experiment.add_audio('original', raw_original, global_step=trainer.current_epoch, sample_rate=dataset.sr)
                        trainer.logger.experiment.add_audio('generation', raw_generation, global_step=trainer.current_epoch,
                                                            sample_rate=dataset.sr)

            # plot prior sampling
            if hasattr(model, 'sample_from_prior'):
                samples = self.plot_samples(model)
                trainer.logger.experiment.add_figure('samples', samples, trainer.current_epoch)

