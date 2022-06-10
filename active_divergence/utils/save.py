import re, os, dill, shutil
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from active_divergence.utils import checkdir

def save_trainig_config(config: OmegaConf, data: LightningDataModule, path: str = None, name: str = None):
    """Saves training configurations and transforms in the training directory."""
    if path is None:
        path = config.rundir
    if name is None:
        name = config.name
    config_path = f"{path}/{name}"
    models_path = f"{path}/{name}/{name}"
    checkdir(models_path)
    current_files = list(filter(lambda x: os.path.splitext(x)[1] == ".ckpt", os.listdir(models_path)))
    versions = list(filter(lambda x: x is not None, [re.match(f"{name}-v(\d+).ckpt", f) for f in current_files]))
    current_version = 1 if len(versions) == 0 else max(map(int, [v[1] for v in versions])) + 1
    name = f'{name}-v{current_version}'
    checkdir(config_path+"/configs")
    with open(f"{config_path}/configs/{name}.yaml", "w+") as f:
        f.write(OmegaConf.to_yaml(config))
    if hasattr(data, "full_transforms"):
        checkdir(config_path+"/transforms")
        with open(f"{config_path}/transforms/transforms_{name}.ct", 'wb') as f:
            dill.dump(data.full_transforms, f)
    # # as pytorch lightning renames the ckpt files if several versions are available,
    # # we have to do the same with tranforms and configs
    # if (os.path.exists(f"{config_path}/configs/{name}.yaml")) and (current_version is not None):
    #     shutil.move(f"{config_path}/configs/{name}.yaml", f"{config_path}/configs/{name}-v1.yaml")
    # if (os.path.exists(f"{config_path}/transforms/transforms_{name}.ct")) and (current_version is not None):
    #     shutil.move(f"{config_path}/transforms/transforms_{name}.ct", f"{config_path}/transforms/transforms_{name}-v1.ct")
    
    