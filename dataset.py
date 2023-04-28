import lightning as L
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from collections import Counter


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


def main():
    dm = CIFAR10DataModule()
    # Cound label distributions
    dm.setup("")
    train_counter = Counter()
    for images, labels in dm.train_dataloader():
        train_counter.update(labels.tolist())
    print(f"Training label distribution\n{sorted(train_counter.items())}")
    val_counter = Counter()
    for images, labels in dm.val_dataloader():
        val_counter.update(labels.tolist())
    print(f"Validation label distribution\n{sorted(val_counter.items())}")
    test_counter = Counter()
    for images, labels in dm.test_dataloader():
        test_counter.update(labels.tolist())
    print(f"Testing label distribution\n{sorted(test_counter.items())}")


if __name__ == "__main__":
    main()
