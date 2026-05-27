import importlib

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from deep_lst.utils.io import ExperimentPaths


class MultiModalMLP(nn.Module):
    def __init__(self, in_keys, out_keys, config, in_feature_sizes=None):
        super().__init__()

        self.in_keys = in_keys
        self.out_keys = out_keys
        sizes = in_feature_sizes or {}
        self.input_size = sum(sizes.get(k, 1) for k in in_keys)
        self.target_size = len(out_keys)
        self.hidden_sizes = config["hidden_sizes"]
        self.dropout_p = config["dropout"]
        self.use_residual = config.get("use_residual", False)

        self.pre, self.res_block, self.post, self.out_layer = self._build()

    def name(self):
        return "mlp"

    def _layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )

    def _build(self):
        sizes = [self.input_size] + self.hidden_sizes

        # Find first consecutive same-dimension pair for the residual skip
        res_idx = None
        if self.use_residual:
            for i in range(len(self.hidden_sizes) - 1):
                if self.hidden_sizes[i] == self.hidden_sizes[i + 1]:
                    res_idx = i + 1
                    break

        pre, res, post = [], None, []
        for i in range(len(self.hidden_sizes)):
            layer = self._layer(sizes[i], sizes[i + 1])
            if res_idx is None or i < res_idx:
                pre.append(layer)
            elif i == res_idx:
                res = layer
            else:
                post.append(layer)

        return (
            nn.Sequential(*pre),
            res,
            nn.Sequential(*post) if post else nn.Identity(),
            nn.Linear(self.hidden_sizes[-1], self.target_size),
        )

    def forward(self, td):
        inputs = [td[k] for k in self.in_keys]
        x = torch.cat(inputs, dim=-1)
        x = self.pre(x)
        if self.res_block is not None:
            x = x + self.res_block(x)
        x = self.post(x)
        out = self.out_layer(x)

        result = td.clone()
        for key, chunk in zip(self.out_keys, out.split(1, dim=-1)):
            result[key] = chunk
        return result


class LitModelModule(LightningModule):
    def __init__(self, model, config, **kwargs):
        super().__init__()

        self.model = model
        self.config = config
        self.learning_rate = config["model"]["learning_rate"]

        climatology = kwargs.get("climatology", None)
        if climatology is not None:
            self.add_module("climatology", climatology)  # make available to lightings .to(device) logic

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
        """Clear any accumulated state at epoch end"""
        torch.cuda.empty_cache()


def load_model(
    config: dict,
    td_keys: dict,
    experiment: str,
    version: str,
    prefer_best: bool = False,
    in_feature_sizes: dict | None = None,
):
    """Instantiate model and load from checkpoint.

    Args:
        config:           Full experiment config dict.
        td_keys:          TensorDict keys from get_keys_from_config.
        experiment:       Experiment name (matches folder under experiments/).
        version:          Version/run name (subfolder containing checkpoints/).
        in_feature_sizes: Optional dict mapping context keys to their feature count.

    Returns:
        lit_module: LitModelModule in eval mode, moved to GPU if available.
        device:     The torch.device the model is on.
    """
    lookback = config.get("train_dataset", {}).get("seviri", {}).get("lookback", 0)
    context_vars = config.get("train_dataset", {}).get("seviri", {}).get("context_vars", [])
    if in_feature_sizes is None:
        in_feature_sizes = {}
    if lookback > 0:
        for var in context_vars:
            ctx_key = ("seviri", f"{var}_ctx")
            if ctx_key not in td_keys["in_keys"]:
                td_keys["in_keys"].append(ctx_key)
            in_feature_sizes[ctx_key] = 2 * lookback

    model = MultiModalMLP(
        in_keys=td_keys["in_keys"] + td_keys["prognostic_keys"],
        out_keys=td_keys["diagnostic_out_keys"] + td_keys["prognostic_out_keys"],
        config=config["model"],
        in_feature_sizes=in_feature_sizes,
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
