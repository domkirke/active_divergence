import matplotlib.pyplot as plt
import torchaudio
from active_divergence.data.audio.dataset import AudioDataset
from active_divergence.data.audio.transforms import *
from active_divergence.data.audio.augmentations import *

root_directory = "tests/acidsInstruments-test"
dataset = AudioDataset(root_directory, bitrate=16, sr=44100)
dataset.import_data()
dataset.drop_sequences(2**16, -1)
dataset.augmentations = [RandomPhase(prob=1.0), Dequantize(prob=1.0)]

