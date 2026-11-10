import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_objective(objective_config, **runtime_keys):
    """The training objective from its config node (conf/objective/ + conf/loss/).

    The node carries every configured value; `runtime_keys` are the key lists that
    get_keys_from_config derives from the data option, and `inc_variances` from the statistics.
    """
    objective_factory = instantiate(objective_config, _partial_=True, _convert_="all")
    return objective_factory(**runtime_keys)


class LitModelModule(LightningModule):
    def __init__(self, model, config, **kwargs):
        super().__init__()

        self.model = model
        self.config = config
        self.learning_rate = config["optim"]["learning_rate"]

        self.step_loss = build_objective(config["objective"], **kwargs)

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
        optim_config = self.config["optim"]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            eps=optim_config["eps"],
            weight_decay=optim_config["weight_decay"],
        )

        total_steps = self.config["training"]["max_epochs"] * self.config["training"]["limit_train_batches"]
        warmup_steps = optim_config["warmup_steps"]

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
