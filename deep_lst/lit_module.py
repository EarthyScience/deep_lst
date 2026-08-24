import importlib

import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class LitModelModule(LightningModule):
    def __init__(self, model, config, **kwargs):
        super().__init__()

        self.model = model
        self.config = config
        self.learning_rate = config["model"]["learning_rate"]

        self.step_loss = self._get_loss(model_id=config["model"]["name"], loss_name=config["model"]["step_loss"])(
            **kwargs
        )

    def _get_loss(self, model_id: str, loss_name: str):
        module = importlib.import_module(f"deep_lst.{model_id}")
        return getattr(module, loss_name)

    def training_step(self, batch, batch_idx):
        loss = self.step_loss(self, batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step_loss(self, batch, stage="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step_loss(self, batch, stage="test")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            eps=1e-8,
            weight_decay=self.config["model"].get("weight_decay", 1e-4),
        )

        total_steps = self.config["training"]["max_epochs"] * self.config["training"]["limit_train_batches"]
        warmup_steps = self.config["model"].get("warmup_steps", 20000)

        if warmup_steps > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss_step_epoch")
        if val_loss is not None:
            self.log("val_loss_step_epoch_log", torch.log(val_loss))
