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
        self.loader_args = dict(config.loader)

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    @property
    def shape(self):
        return (1, 28, 28)

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage = None):
        # transforms
        if self.data_args.binary:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x > 0.5).float())])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        # split dataset
        mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

    # return the dataloader for each split
    def train_dataloader(self, batch_size=None):
        loader_args = self.loader_args
        loader_args['batch_size'] = batch_size or loader_args.get('batch_size', 128)
        loader_train = DataLoader(self.mnist_train, **loader_args)
        return loader_train

    def val_dataloader(self, batch_size=None):
        loader_args = self.loader_args
        loader_args['batch_size'] = batch_size or loader_args.get('batch_size', 128)
        loader_val = DataLoader(self.mnist_val, **loader_args)
        return loader_val

    def test_dataloader(self, batch_size=None):
        if self.test_dataset is None:
            return None
        loader_args = self.loader_args
        loader_args['batch_size'] = batch_size or loader_args.get('batch_size', 128)
        loader_test = DataLoader(self.mnist_test, **loader_args)
        return loader_test

    # utils callback
    def train_dataset(self):
        return self.mnist_train
    def validation_dataset(self):
        return self.mnist_val
    def test_dataset(self):
        return self.mnist_test
