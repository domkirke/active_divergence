import torch, numpy as np, math, torchaudio, os
import active_divergence as ad, pytest
from active_divergence.data.audio import AudioDataset

TARGET_LENGTH = 3.0
N_FILES_INVERSION = 2
INVERSION_IDS = [0]
OUTDIR = "/tmp"

@pytest.mark.data
def test_raw_import(dataset_path):
    dataset = ad.data.audio.AudioDataset(dataset_path)
    dataset = ad.data.audio.AudioDataset(dataset_path, TARGET_LENGTH=1.0)
    dataset = ad.data.audio.AudioDataset(dataset_path, TARGET_LENGTH=4096)

@pytest.fixture(name="raw_dataset", scope="session")
def raw_dataset(dataset_path):
    dataset = ad.data.audio.AudioDataset(dataset_path, TARGET_LENGTH=TARGET_LENGTH)
    dataset.import_data(flatten=True, scale=False)
    return dataset

@pytest.mark.data
# @pytest.mark.skip(because="lmsqd")
def test_write_transform(dataset_path, dataset_export_transforms):
    name, transform = dataset_export_transforms
    if transform is None:
        name, transform = "none", ad.data.audio.AudioTransform()
    dataset = AudioDataset(dataset_path, target_length=TARGET_LENGTH, transforms=transform)
    dataset.import_data()
    dataset.write_transforms(save_as=name, force=True)
    data_shape = dataset.data.shape
    dataset_loaded = AudioDataset(dataset_path)
    dataset_loaded.import_transform(name)
    assert data_shape == dataset.data.shape
    data = dataset[0]
    if dataset_loaded.transforms.invertible:
        dataset_loaded.transforms.invert(data)


def test_inversion(raw_dataset, transforms_inversion):
    name, transform = transforms_inversion
    raw_dataset.transforms = transform
    raw_dataset.scale_transform(True)
    root_directory = os.path.basename(raw_dataset.root_directory)
    if OUTDIR is None:
        dirname = f"inversion/{root_directory}"
    else:
        dirname = f"{OUTDIR}/inversion/{root_directory}"
    ad.utils.checkdir(dirname)
    for idx in INVERSION_IDS:
        data, metadata = raw_dataset[idx]
        basename = os.path.splitext(os.path.basename(raw_dataset.files[idx]))[0]
        t_data = transform.invert(data)
        if t_data.ndim == 1:
            t_data = t_data[np.newaxis]
        if not os.path.isfile(f"{dirname}/original.wav"):
            original = raw_dataset.data[idx]
            if original.ndim == 1:
                original = original[np.newaxis]
            torchaudio.save(f"{dirname}/{basename}_original.wav", original, sample_rate=int(metadata['sr']))
        torchaudio.save(f"{dirname}/{basename}_{name}.wav", t_data.float(), sample_rate=int(metadata['sr']))

def test_flattening(dataset_path, dataset_export_transforms):
    name, transform = dataset_export_transforms
    if transform is None:
        name, transform = "none", ad.data.audio.AudioTransform()
    dataset = AudioDataset(dataset_path, TARGET_LENGTH=TARGET_LENGTH)
    # dataset.import_data(write_transforms=True)
    # dataset.flatten_data(0)
    transforms = dataset.available_transforms
    for t in transforms:
        dataset.import_transform(t)
        dataset.flatten_data(0)



