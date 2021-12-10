from . import utils
from . import data
from . import losses
from . import modules
from . import models
from .monitor import plugins, callbacks

def get_callbacks(config):
    if config is None:
        return []
    cks = []
    for callback in config:
        call_type = callback['type']
        call_args = callback.get("args", {})
        cks.append(getattr(callbacks, call_type)(**call_args))
    return cks

def get_plugins(config):
    if config is None:
        return []
    pgs = []
    for plugin in config:
        plugin_type = plugin['type']
        plugin_args = plugin.get('args', {})
        pgs.append(getattr(plugins, plugin_type)(**plugin_args))
    return pgs
