import torch, numpy as np, math, torchaudio, os
import active_divergence as ad, pytest
from active_divergence.data.audio import AudioDataset

TARGET_LENGTH = 3.0
N_FILES_INVERSION = 2


@pytest.mark.data
def test_raw_import(dataset_path):
    dataset = ad.data.audio.AudioDataset(dataset_path)
    dataset = ad.data.audio.AudioDataset(dataset_path, TARGET_LENGTH=1.0)
    dataset = ad.data.audio.AudioDataset(dataset_path, TARGET_LENGTH=4096)


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

def test_inversion(dataset_path, transforms_inversion):
    name, transform = transforms_inversion
    dataset = AudioDataset(dataset_path, target_length=TARGET_LENGTH, sr=44100)
    dataset.import_data()
    random_idx = np.random.permutation(len(dataset))[:N_FILES_INVERSION]
    dirname = f"inversion/{dataset_path}/{name}"
    ad.utils.checkdir(dirname)
    for idx in random_idx:
        data = transform(dataset.data[idx])
        basename = os.path.splitext(os.path.basename(dataset.files[idx]))[0]
        t_data = transform.invert(data)
        torchaudio.save(f"{dirname}/{basename}.wav", t_data, sample_rate=44100)

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



