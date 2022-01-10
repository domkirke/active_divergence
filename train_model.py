import argparse, pdb, os, sys
sys.path.append('../')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from active_divergence import data, models, get_callbacks, get_plugins
from active_divergence.monitor.callbacks import ModelCheckpoint
from active_divergence.utils import save_config
from omegaconf import OmegaConf
import tensorflow as tf
import tensorboard as tb
from active_divergence.utils.misc import checkdir
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="YAML configuration file")
pl.Trainer.add_argparse_args(parser)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--check", type=int, default=0)

args = parser.parse_args()
config = OmegaConf.load(args.config)
config_save = config.model.training.save
save_path = f"{config_save.path}/{config_save.name}"
args.default_root_dir = save_path
checkdir(save_path)

# Set up model
data = getattr(data, config.data.module)(config.data)
model_type = getattr(models, config.model.type)
config.model.input_dim = data.shape

if args.checkpoint is None:
    model = model_type(**dict(config.model))
    save_config(config, data, save_path, config_save.name)
else:
    # config = Config(f"{os.path.dirname(args.checkpoint)}/{os.path.splitext(os.path.basename(args.checkpoint))[0]}.yaml")
    model = model_type.load_from_checkpoint(args.checkpoint)

# Configure callbacks
callbacks = get_callbacks(config.get('callbacks'))

# Train!
trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
if args.check:
    pdb.set_trace()
trainer.fit(model, datamodule=data)