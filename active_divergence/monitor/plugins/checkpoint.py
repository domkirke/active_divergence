from pathlib import Path
import pdb
from typing import Any, Dict, Optional, Union

from pytorch_lightning.plugins import CheckpointIO, SingleDevicePlugin


class ADCheckpointIO(CheckpointIO):
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        pdb.set_trace()

    def load_checkpoint(self, path: Union[str, Path], storage_options: Optional[Any] = None) -> Dict[str, Any]:
        ...

    def remove_checkpoint(self, path: Union[str, Path]) -> None:
        ...