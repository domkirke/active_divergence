import os, sys, pdb
sys.path.append('..')
from active_divergence.data.audio import AudioDataset
from active_divergence.data.audio import transforms
from active_divergence.utils import Config, checklist
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from pytorch_lightning import LightningDataModule

def parse_transforms(transform_list):
    transform_list = checklist(transform_list)
    current_transforms = []
    for t in transform_list:
        transform_tmp = getattr(transforms, t['type'])(**t.get('args', {}))
        current_transforms.append(transform_tmp)
    if len(current_transforms) > 1:
        return transforms.ComposeAudioTransform(current_transforms)
    else:
        return current_transforms[0]

class AudioDataModule(LightningDataModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.dataset = None
        self.dataset_args = config.dataset.dict()
        self.transform_args = config.transforms
        self.loader_args = config.loader.dict()
        self.partition_balance = config.partition_balance or [0.8, 0.2]
        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.import_datasets()

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:
    def load_dataset(self, dataset_args, transform_args, make_partitions=False):
        dataset = AudioDataset(**dataset_args)
        if transform_args:
            name = transform_args.name
            if name in dataset.available_transforms and (not transform_args.force):
                dataset.import_transform(name)
            else:
                assert name is not None
                pre_transforms = parse_transforms(transform_args.pre_transforms) or transforms.AudioTransform()
                dataset.transforms = pre_transforms
                dataset.import_data(write_transforms=True, save_transform_as=name)
        else:
            dataset.import_data()
        current_transforms = parse_transforms(transform_args.transforms or transforms.AudioTransform())
        dataset.transforms = current_transforms
        if dataset_args.get('flatten') is not None:
            dataset.flatten_data(int(dataset_args['flatten']))
        if make_partitions:
            dataset.make_partitions(['train', 'valid'], self.partition_balance)
        if dataset_args.get('sequence'):
            dataset.drop_sequences(dataset_args['sequence'].get('length'),
                                   dataset_args['sequence'].get('mode', "random"),
                                   dataset_args['sequence'].get('idx', -2))
        return dataset

    def import_datasets(self, stage = None):
        # transforms
        self.dataset = self.load_dataset(self.dataset_args, self.transform_args, make_partitions=True)
        self.train_dataset = self.dataset.retrieve('train')
        self.valid_dataset = self.dataset.retrieve('valid')

    @property
    def shape(self):
        return self.dataset.data[0].shape

    # return the dataloader for each split
    def train_dataloader(self, batch_size=None):
        loader_args = self.loader_args
        loader_args['batch_size'] = batch_size or loader_args.get('batch_size', 128)
        loader_train = DataLoader(self.train_dataset, **loader_args)
        return loader_train

    def val_dataloader(self, batch_size=None):
        loader_args = self.loader_args
        loader_args['batch_size'] = batch_size or loader_args.get('batch_size', 128)
        loader_val = DataLoader(self.valid_dataset, **loader_args)
        return loader_val

    def test_dataloader(self, batch_size=None):
        if self.test_dataset is None:
            return None
        loader_args = self.loader_args
        loader_args['batch_size'] = batch_size or loader_args.get('batch_size', 128)
        loader_test = DataLoader(self.test_dataset, **loader_args)
        return loader_test

