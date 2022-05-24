import argparse, pdb, os, sys
import logging
import torch, pytorch_lightning as pl, hydra
from omegaconf import OmegaConf, DictConfig
import GPUtil as gpu
from active_divergence import data, models, get_callbacks
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

@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)
    # import data
    data_module = getattr(data, config.data.module)(config.data)
    # import model
    config.model.input_size = data_module.shape
    model = getattr(models, config.model.type)(config.model)
    # import callbacks
    callbacks = get_callbacks(config.get('callbacks'))
    # setup trainer
    trainer_config = config.get('pl_trainer', {})
    trainer_config['gpus'] = config.get('gpus', use_gpu)
    trainer = pl.Trainer(**config.get('pl_trainer', {}), callbacks=callbacks)
    if bool(config.get('check')):
        pdb.set_trace()
    # train!
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()


"""
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("configs", type=str, help="YAML configuration file")
pl.Trainer.add_argparse_args(parser)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--check", type=int, default=0)

args = parser.parse_args()
configs = OmegaConf.load(args.configs)
config_save = configs.model.training.save
save_path = f"{config_save.path}/{config_save.name}"
args.default_root_dir = save_path
checkdir(save_path)

# Set up model
data = getattr(data, configs.data.module)(configs.data)
model_type = getattr(model, configs.model.type)
configs.model.input_dim = data.shape

if args.checkpoint is None:
    model = model_type(**dict(configs.model))
    save_config(configs, data, save_path, config_save.name)
else:
    # configs = Config(f"{os.path.dirname(args.checkpoint)}/{os.path.splitext(os.path.basename(args.checkpoint))[0]}.yaml")
    model = model_type.load_from_checkpoint(args.checkpoint)

# Configure callbacks
callbacks = get_callbacks(configs.get('callbacks'))

# Train!
trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
if args.check:
    pdb.set_trace()
trainer.fit(model, datamodule=data)
"""
