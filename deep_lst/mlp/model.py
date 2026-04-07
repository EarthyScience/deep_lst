import importlib

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from deep_lst.utils.io import ExperimentPaths


class MultiModalMLP(nn.Module):
    def __init__(self, in_keys, out_keys, config):
        super().__init__()

        self.in_keys = in_keys
        self.out_keys = out_keys

        self.input_size = len(in_keys)
        self.target_size = len(out_keys)

        self.activation = nn.ReLU() if config["activation"] == "relu" else None
        self.hidden_sizes = config["hidden_sizes"]
        self.dropout = config["dropout"]

        self.net = self.dense()

    def name(self):
        return "mlp"

    def forward(self, td):
        inputs = [td[k] for k in self.in_keys]
        x = torch.cat(inputs, dim=-1)
        out = self.net(x)

        result = td.clone()
        for key, chunk in zip(self.out_keys, out.split(1, dim=-1)):
            result[key] = chunk
        return result

    def dense(self):
        layers = nn.Sequential()
        layers.add_module("input", nn.Linear(self.input_size, self.hidden_sizes[0]))
        layers.add_module("activation0", self.activation)
        layers.add_module("dropout0", nn.Dropout(self.dropout))
        if len(self.hidden_sizes) > 1:
            for i in range(len(self.hidden_sizes) - 1):
                layers.add_module(f"hidden{i}", nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
                layers.add_module(f"activation{i}", self.activation)
                layers.add_module(f"dropout{i}", nn.Dropout(self.dropout))
        layers.add_module("output", nn.Linear(self.hidden_sizes[-1], self.target_size))
        return layers


class LitModelModule(LightningModule):
    def __init__(self, model, config, **kwargs):
        super().__init__()

        self.model = model
        self.config = config
        self.learning_rate = config["model"]["learning_rate"]
        self.max_epochs = config["training"]["max_epochs"]

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
            eps=1e-8,  # Numerical stability
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_train_epoch_end(self):
        """Clear any accumulated state at epoch end"""
        torch.cuda.empty_cache()


def load_model(config: dict, td_keys: dict, experiment: str, version: str, prefer_best: bool = False):
    """Instantiate model and load from checkpoint.

    Args:
        config:     Full experiment config dict.
        td_keys:    TensorDict keys from get_keys_from_config.
        experiment: Experiment name (matches folder under experiments/).
        version:    Version/run name (subfolder containing checkpoints/).

    Returns:
        lit_module: LitModelModule in eval mode, moved to GPU if available.
        device:     The torch.device the model is on.
    """
    model = MultiModalMLP(
        in_keys=td_keys["in_keys"] + td_keys["prognostic_keys"],
        out_keys=td_keys["diagnostic_out_keys"] + td_keys["prognostic_out_keys"],
        config=config["model"],
    )

    paths = ExperimentPaths()
    ckpt_path = (
        paths.best_checkpoint(experiment, version) if prefer_best else paths.latest_checkpoint(experiment, version)
    )

    lit_module = LitModelModule.load_from_checkpoint(
        ckpt_path,
        model=model,
        config=config,
        **td_keys,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_module.eval()
    lit_module.to(device)

    return lit_module, device
