from pathlib import Path
import pdb
from typing import Any, Dict, Optional, Union

from pytorch_lightning.plugins import CheckpointIO, SingleDevicePlugin


class ADCheckpointIO(CheckpointIO):
    pass