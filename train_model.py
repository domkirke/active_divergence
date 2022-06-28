import argparse, pdb, os, sys
import logging
import torch, pytorch_lightning as pl, hydra
from omegaconf import OmegaConf, DictConfig
import GPUtil as gpu
from active_divergence import data, models, get_callbacks
from active_divergence.utils import save_trainig_config
logger = logging.getLogger(__name__)

# detect CUDA devices
CUDA = gpu.getAvailable(maxMemory=.05)
VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if VISIBLE_DEVICES:
    use_gpu = int(int(VISIBLE_DEVICES) >= 0)
elif len(CUDA):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
    use_gpu = 1
elif torch.cuda.is_available():
    print("Cuda is available but no fully free GPU found.")
    print("Training may be slower due to concurrent processes.")
    use_gpu = 1
else:
    print("No GPU found.")
    use_gpu = 0

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)
    # import data
    config.data.loader['num_workers'] = config.data.loader.get('num_workers', os.cpu_count())
    data_module = getattr(data, config.data.module)(config.data)
    # import model
    config.model.input_shape = data_module.shape
    model = getattr(models, config.model.type)(config.model)
    # import callbacks
    callbacks = get_callbacks(config.get('callbacks'))
    # setup trainer
    trainer_config = config.get('pl_trainer', {})
    trainer_config['gpus'] = config.get('gpus', use_gpu)
    trainer_config['default_root_dir'] = f"{config.rundir}/{config.name}"
    trainer_config['weights_save_path'] = f"{config.rundir}/{config.name}/models"
    trainer = pl.Trainer(**trainer_config, callbacks=callbacks)
    if bool(config.get('check')):
        pdb.set_trace()
    # train!
    save_trainig_config(config, data_module)
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
