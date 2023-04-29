import lightning as L
import torch
import torchmetrics
import torch.nn.functional as F
import wandb
from dataset import CIFAR10DataModule
from model import VisionTransformer
from lightning.pytorch.loggers import WandbLogger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model"])
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.warmup_epochs = 5
        self.steps_per_epoch = 52
        self.max_epochs = 60
        self.batch_size = 768

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, labels = batch
        logits = self.model(features)
        loss = F.cross_entropy(logits, labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        features, labels = batch
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=1,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.1,
            num_classes=10,
        )
        features, one_hot_labels = mixup_fn(features, labels)
        logits = self.model(features)
        loss = SoftTargetCrossEntropy()(logits, one_hot_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        self.log("train loss", loss)
        self.train_acc(predicted_labels, labels)
        self.log(
            "train acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch)
        self.log("val loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, labels)
        self.log("val acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, labels)
        self.log("test acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        lr_scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs * self.steps_per_epoch,
                max_epochs=self.max_epochs * self.steps_per_epoch,
            ),
            "monitor": "train loss",
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


def main():
    wandb.init()
    torch.manual_seed(1)
    dm = CIFAR10DataModule(batch_size=768)
    model = VisionTransformer(n_classes=10)
    lightning_model = LightningModel(model=model, learning_rate=0.01)
    trainer = L.Trainer(
        max_epochs=60,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        logger=WandbLogger(),
    )
    trainer.fit(model=lightning_model, datamodule=dm)
    train_acc = trainer.validate(dataloaders=dm.train_dataloader())
    val_acc = trainer.validate(datamodule=dm)
    test_acc = trainer.test(datamodule=dm)
    print(
        f"Train accuracy: {train_acc[0]['val acc'] * 100:.2f} | Validation accuracy: {val_acc[0]['val acc'] * 100:.2f} | Test accuracy: {test_acc[0]['test acc'] * 100:.2f}"
    )


if __name__ == "__main__":
    main()
