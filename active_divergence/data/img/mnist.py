import os, sys, pdb
sys.path.append('..')
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning import LightningDataModule


class MNISTDataModule(LightningDataModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.data_args = config.dataset
        self.polarity = config.dataset.get('polarity', 'unipolar')
        self.loader_args = dict(config.loader)

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    @property
    def shape(self):
        if self.data_args.get('resize'):
            return (1, *self.data_args.resize)
        else:
            return (1, 28, 28)

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage = None):
        # transforms
        transform = []
        if self.data_args.get('resize'):
            transform.append(transforms.Resize(tuple(self.data_args.resize)))
        transform.append(transforms.ToTensor())
        if self.data_args.binary:
            transform.append(transforms.Lambda(lambda x: (x > 0.5).float()))
        if self.polarity == "bipolar":
            transform.append(transforms.Lambda(lambda x: x * 2 - 1))
        transform = transforms.Compose(transform)
        # split dataset
        mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

    # return the dataloader for each split
    def train_dataloader(self, **kwargs):
        loader_args = {**self.loader_args, **kwargs}
        loader_train = DataLoader(self.train_dataset(), **loader_args)
        return loader_train

    def val_dataloader(self, **kwargs):
        loader_args = {**self.loader_args, **kwargs}
        loader_val = DataLoader(self.validation_dataset(), **loader_args)
        return loader_val

    def test_dataloader(self, **kwargs):
        if self.test_dataset is None:
            return None
        loader_args = {**self.loader_args, **kwargs}
        loader_test = DataLoader(self.test_dataset(), **loader_args)
        return loader_test

    # utils callback
    def train_dataset(self, **kwargs):
        return self.mnist_train
    def validation_dataset(self, **kwargs):
        return self.mnist_val
    def test_dataset(self, **kwargs):
        return self.mnist_test
