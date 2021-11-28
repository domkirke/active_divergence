import re, os

def save_config(config, path, name):
    current_files = list(filter(lambda x: os.path.splitext(x)[1] == ".ckpt", os.listdir(path)))
    if len(current_files) == 0:
        config_path = f"{path}/{name}.yaml"
    else:
        versions = list(filter(lambda x: x is not None, [re.match(f"{name}-v(\d+).ckpt", f) for f in current_files]))
        current_version = 1 if len(versions) == 0 else max(map(int, [v[1] for v in versions])) + 1
        config_path = f"{path}/{name}-v{current_version}.yaml"
    config.write(config_path)