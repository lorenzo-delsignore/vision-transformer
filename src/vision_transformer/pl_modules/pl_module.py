import logging

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from vision_transformer.modules.module import VisionTransformer

pylogger = logging.getLogger(__name__)


class LightningModel(pl.LightningModule):
    logger: NNLogger

    def __init__(self, metadata=None, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata
        metric = torchmetrics.Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()
        self.model = VisionTransformer(img_size=224, n_classes=10)

    def forward(self, x):
        return self.model(x)

    def step(self, features, labels, train=None):
        logits = self(features)
        if train:
            loss = SoftTargetCrossEntropy()(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels)
        return {"logits": logits.detach(), "loss": loss}

    def training_step(self, batch, batch_idx):
        features, labels = batch
        # mixup_fn = Mixup(
        #     mixup_alpha=0.8,
        #     cutmix_alpha=1.0,
        #     cutmix_minmax=None,
        #     prob=1,
        #     switch_prob=0.5,
        #     mode="batch",
        #     num_classes=10,
        #     label_smoothing=0.1,
        # )
        # features, smoothing_labels = mixup_fn(features, labels)
        step_out = self.step(features, labels, train=False)

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_accuracy(torch.softmax(step_out["logits"], dim=-1), labels)
        self.log_dict(
            {
                "acc/train": self.train_accuracy,
            },
            on_epoch=True,
        )
        return step_out

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        step_out = self.step(features, labels)
        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_accuracy(torch.softmax(step_out["logits"], dim=-1), labels)
        self.log_dict(
            {
                "acc/val": self.val_accuracy,
            },
            on_epoch=True,
        )
        return step_out

    def test_step(self, batch, batch_idx):
        features, labels = batch
        step_out = self.step(features, labels)

        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach()},
        )

        self.test_accuracy(torch.softmax(step_out["logits"], dim=-1), labels)
        self.log_dict(
            {
                "acc/test": self.test_accuracy,
            },
            on_epoch=True,
        )
        return step_out

    def configure_optimizers_(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.05,
        )
        scheduler = CosineLRScheduler(
            optimizer,
            warmup_t=self.warmup_epochs,
            t_initial=self.max_epochs,
            lr_min=1e-5,
            warmup_lr_init=1e-6,
            t_in_epochs=True,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
