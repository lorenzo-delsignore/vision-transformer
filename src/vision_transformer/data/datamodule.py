import logging
from collections import Counter
from functools import cached_property, partial

import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from nn_core.common import PROJECT_ROOT

from .augmentation import GaussianBlur, Solarization

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, class_vocab):
        self.class_vocab = class_vocab

    def save(self, dst_path):
        pylogger.debug(f"Saving Metadata to {dst_path}")
        (dst_path / "class_vocab.tsv").write_text(
            "\n".join(f"{key}\t{value}" for key, value in self.class_vocab.items())
        )

    def load(self, src_path):
        pylogger.debug(f"Loading MetaData from {src_path}")
        lines = (src_path / "class_vocab.tsv").read_text(encoding="utf-8").splitlines()
        class_vocab = {}
        for line in lines:
            key, value = line.strip().split("\t")
            class_vocab[key] = value

        return MetaData(
            class_vocab=class_vocab,
        )


def collate_fn(samples, split, metadata):
    return default_collate(samples)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets,
        num_workers,
        batch_size,
        gpus,
        val_percentage,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.gpus = gpus
        self.pin_memory = gpus is not None and str(gpus) != "0"
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.val_percentage = val_percentage

    @cached_property
    def metadata(self):
        if self.train_dataset is None:
            self.setup(stage="fit")
        return MetaData(class_vocab=self.train_dataset.dataset.class_vocab)

    def setup(self, stage):
        train_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=3),
                transforms.RandomCrop(224, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([Solarization(), GaussianBlur()]),
                transforms.ColorJitter(0.3, 0.3, 0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )

        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_dataset is None):
            cifar10_train = hydra.utils.instantiate(
                self.datasets.train,
                split="train",
                transform=train_transform,
                path=PROJECT_ROOT / "data",
            )
            train_length = int(len(cifar10_train) * (1 - self.val_percentage))
            val_length = len(cifar10_train) - train_length
            self.train_dataset, self.val_dataset = random_split(cifar10_train, [train_length, val_length])
            self.val_dataset.transforms = test_transform

        if stage is None or stage == "test":
            self.test_dataset = hydra.utils.instantiate(
                self.datasets.test,
                split="test",
                path=PROJECT_ROOT / "data",
                transform=test_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="train", metadata=self.metadata),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="val", metadata=self.metadata),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="test", metadata=self.metadata),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg):
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    # Cound label distributions
    _.setup("")
    train_counter = Counter()
    for images, labels in _.train_dataloader():
        train_counter.update(labels.tolist())
    print(f"Training label distribution\n{sorted(train_counter.items())}")
    val_counter = Counter()
    for images, labels in _.val_dataloader():
        val_counter.update(labels.tolist())
    print(f"Validation label distribution\n{sorted(val_counter.items())}")
    test_counter = Counter()
    for images, labels in _.test_dataloader():
        test_counter.update(labels.tolist())
    print(f"Testing label distribution\n{sorted(test_counter.items())}")


if __name__ == "__main__":
    main()
