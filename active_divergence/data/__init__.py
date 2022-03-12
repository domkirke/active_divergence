from . import audio
from . import img
from .audio import AudioDataModule, SingleAudioDataModule
from .video import VideoDataset, VideoDataModule
from .img import MNISTDataModule

modules = [AudioDataModule, VideoDataModule, MNISTDataModule]
