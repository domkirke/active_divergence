from . import utils
from . import data
from . import losses
from . import modules
from . import models
from . import monitor

def get_callbacks(config):
    callbacks = []
    for callback in config:
        call_type = callback['type']
        call_args = callback.get("args", {})
        callbacks.append(getattr(monitor, call_type)(**call_args))
    return callbacks
