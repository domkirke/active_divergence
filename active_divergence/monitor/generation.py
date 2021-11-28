import torch, torch.distributions as dist, torchvision as tv, pdb, random, matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus

def check_mono(sgl, normalize=True):
    sgl = sgl.squeeze()
    if sgl.ndim == 2:
        if sgl.shape[0] > 1:
            sgl = sgl.mean(0).unsqueeze(0)
    if normalize:
        sgl /= sgl.max()
    return sgl


class AudioReconstructionMonitor(Callback):

    def __init__(self, n_reconstructions=5, n_morphings=5, n_samples=5, n_files=5,
                 temperature_range=None, monitor_epochs=1):
        self.n_reconstructions = n_reconstructions
        self.n_samples = n_samples
        self.temperature_range = temperature_range or [0.01, 0.1, 1.0, 3.0, 5.0, 10.0]
        self.n_files = n_files
        self.n_morphings = n_morphings
        self.monitor_epochs = monitor_epochs

    def plot_reconstructions(self, model, loader):
        data = next(loader(batch_size=self.n_reconstructions).__iter__())
        x_original, x_out = model.reconstruct(data)
        x_original.squeeze(); x_out.squeeze()
        fig, ax = plt.subplots(len(x_original), 1)
        for i in range(len(data)):
            ax[i].plot(x_original[i], linewidth=1)
            ax[i].plot(x_out[i], linewidth=0.6)
        return fig

    def plot_samples(self, model):
        out = model.sample_from_prior(n_samples=self.n_samples, temperature=self.temperature_range)
        if isinstance(out, dist.Distribution):
            out = out.mean
        fig, ax = plt.subplots(self.n_samples, len(self.temperature_range))
        for i in range(self.n_samples):
            for j in range(len(self.temperature_range)):
                ax[i, j].plot(out[i, j])
        return fig

    def reconstruct_file(self, model, files, dataset):
        originals = []; generations = []
        for f in files:
            data, meta = dataset.transform_file(f"{dataset.root_directory}/{f}")
            original, generation = model.reconstruct(data)
            originals.append(dataset.invert_data(original))
            generations.append(dataset.invert_data(generation))
        originals, generations = torch.cat(originals, -1).squeeze(), torch.cat(generations, -1).squeeze()
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
                    test_rec = self.plot_reconstructions(model, trainer.datamodule.test_datamodule)
                    trainer.logger.experiment.add_figure('reconstructions/test', test_rec, trainer.current_epoch)
                # perform full audio reconstructions
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

