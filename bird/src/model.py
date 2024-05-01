import timm
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics.classification import MultilabelAveragePrecision


@torch.no_grad()
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = lam * x + (1 - lam) * x.flip(dims=(0,))
    y_a, y_b = y, y.flip(dims=(0,))
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class BirdModel(LightningModule):
    def __init__(
        self,
        num_classes: int = 264,
        learning_rate: float = 0.0001,
        lr_step_size: int = 20,
        lr_gamma: float = 0.5,
        model_name: str = "tf_efficientnet_b0_ns",
        pretrained: bool = False,
        drop_rate: float = 0.2,
        use_mixup: bool = True,
        alpha_mixup: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.use_mixup = use_mixup
        self.alpha_mixup = alpha_mixup
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_map = MultilabelAveragePrecision(num_labels=num_classes)
        self.val_map = MultilabelAveragePrecision(num_labels=num_classes)

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            in_chans=1,
        )

    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        images = batch.pop("images")
        labels = batch.pop("labels")
        if self.use_mixup:
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, self.alpha_mixup
            )
            output = self(images)
            loss = mixup_criterion(self.loss_fn, output, labels_a, labels_b, lam)
        else:
            output = self(images)
            loss = self.loss_fn(output, labels)
        self.train_map(output, labels.long())
        self.log_dict(
            {
                "train_loss": loss,
                "train_map": self.train_map,
            },
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            batch_size=labels.size(0),
        )
        return loss

    def configure_optimizers(self):
        tagger_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(tagger_params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        images = batch.pop("images")
        labels = batch.pop("labels")
        output = self(images)
        loss = self.loss_fn(output, labels)
        self.val_map(output, labels.long())
        self.log_dict(
            {
                "val_loss": loss,
                "val_map": self.val_map,
            },
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            batch_size=labels.size(0),
        )
        return
