import re, os, dill, pdb
from omegaconf import OmegaConf
from active_divergence.utils import checkdir

def save_config(config, data, path, name):
    current_files = list(filter(lambda x: os.path.splitext(x)[1] == ".ckpt", os.listdir(path)))
    if len(current_files) != 0:
        versions = list(filter(lambda x: x is not None, [re.match(f"{name}-v(\d+).ckpt", f) for f in current_files]))
        current_version = 1 if len(versions) == 0 else max(map(int, [v[1] for v in versions])) + 1
        name = f'{name}-v{current_version}'
    config_path = f"{path}/{name}"
    checkdir(config_path)
    with open(f"{config_path}/{name}.yaml", "w+") as f:
        f.write(OmegaConf.to_yaml(config))
    with open(f"{config_path}/transforms.ct", 'wb') as f:
        dill.dump(data.full_transforms, f)
    