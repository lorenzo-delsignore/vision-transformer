import lightning as L
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir="data/", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # Download
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage):
        self.cifar10_full = datasets.CIFAR10(
            self.data_dir, transform=transforms.ToTensor(), train=True
        )
        self.cifar10_train, self.cifar10_val = random_split(
            self.cifar10_full, [40000, 10000]
        )
        self.cifar10_test = datasets.CIFAR10(
            self.data_dir, transform=transforms.ToTensor(), train=False
        )
        self.cifar10_predict = datasets.CIFAR10(
            self.data_dir, transform=transforms.ToTensor(), train=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar10_val, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar10_test, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

    def predict_dataloader(self):
        return DataLoader(
            self.cifar10_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )
