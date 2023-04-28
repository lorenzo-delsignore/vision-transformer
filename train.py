import lightning as L
import torch
import torchmetrics
import torch.nn.functional as F
from dataset import CIFAR10DataModule
from model import VisionTransformer
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model"])
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, labels = batch
        logits = self.model(features)
        loss = F.cross_entropy(logits, labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch)
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate * CIFAR10DataModule().batch_size / 512,
            weight_decay=0.05,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=5, max_epochs=60
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train loss",
                "interval": "step",
                "frequency": 1,
            },
        }


def main():
    torch.manual_seed(1)
    dm = CIFAR10DataModule(batch_size=1024)
    model = VisionTransformer(n_classes=10)
    lightning_model = LightningModel(model=model, learning_rate=0.0005)
    trainer = L.Trainer(
        max_epochs=60, accelerator="auto", devices="auto", deterministic=True
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
