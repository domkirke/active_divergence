from pytorch_lightning import callbacks   
import sys; sys.path.append('../')
import pdb
from active_divergence.utils import checkdir

class ModelCheckpoint(callbacks.ModelCheckpoint):
    def __init__(self, dirpath=None, filename=None, epoch_period=None, **kwargs):
        dirpath += f"/{filename}"
        super(ModelCheckpoint, self).__init__(dirpath=dirpath, filename=filename, **kwargs)
        self.epoch_period = epoch_period
        if self.epoch_period is not None:
            checkdir(self.dirpath+"/epochs")

    def save_checkpoint(self, trainer):
        super(ModelCheckpoint, self).save_checkpoint(trainer)
        if self.epoch_period is not None:
            if trainer.current_epoch % self.epoch_period == 0:
                filepath = f"{self.dirpath}/epochs/{self.filename}_{self.STARTING_VERSION}_{trainer.current_epoch}{self.FILE_EXTENSION}"
                trainer.save_checkpoint(filepath, self.save_weights_only)

