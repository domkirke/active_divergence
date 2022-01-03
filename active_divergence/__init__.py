import sys
sys.path.append("../")

from active_divergence import utils
from active_divergence import data
from active_divergence import losses
from active_divergence import modules
from active_divergence import models
from active_divergence.monitor import plugins, callbacks

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
    if config.get("plugins") is None:
        return []
    pgs = []
    for plugin in config:
        plugin_type = plugin['type']
        plugin_args = plugin.get('args', {})
        pgs.append(getattr(plugins, plugin_type)(**plugin_args))
    return pgs
