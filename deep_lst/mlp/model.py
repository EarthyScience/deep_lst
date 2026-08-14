import importlib

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from deep_lst.utils.config import get_packing_keys
from deep_lst.utils.io import ExperimentPaths

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class MultiModalMLP(nn.Module):
    def __init__(self, n_prognostic, n_diagnostic, n_input, n_forcing_context, lookback, config):
        super().__init__()

        self.n_prognostic = n_prognostic
        self.lookback = lookback
        self.input_size = n_input + 2 * lookback * n_prognostic + lookback * n_forcing_context
        self.target_size = n_prognostic + n_diagnostic
        self.hidden_sizes = config["hidden_sizes"]
        self.dropout_p = config["dropout"]
        self.use_residual = config.get("use_residual", False)
        self.use_layer_norm = config.get("use_layer_norm", False)
        self.activation_cls = ACTIVATIONS[config.get("activation", "relu")]

        self.pre, self.res_block, self.post, self.out_layer = self._build()

    def name(self):
        return "mlp"

    def _layer(self, in_dim, out_dim):
        layers = [nn.Linear(in_dim, out_dim)]
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(self.activation_cls())
        layers.append(nn.Dropout(self.dropout_p))
        return nn.Sequential(*layers)

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

    def _flatten_window(self, window, with_mask=True):
        values = torch.nan_to_num(window, nan=0.0)
        if with_mask:
            mask = torch.isfinite(window).to(window.dtype)
            values = torch.cat([values, mask], dim=-1)  # (N, L, H, W, 2C)
        N, L, H, W, twoC = values.shape
        values = values.permute(0, 2, 3, 1, 4)  # (N, H, W, L, 2C)
        return values.reshape(N, H, W, L * twoC)

    def forward(self, state_window, X_dyn_window, step_X_dyn):
        state_flat = self._flatten_window(state_window)  # (N, H, W, 2*L*nP)
        dyn_flat = self._flatten_window(X_dyn_window, with_mask=False)  # (N, H, W, 2*L*nF_ctx)
        x = torch.cat([step_X_dyn, dyn_flat, state_flat], dim=-1)
        x[torch.isnan(x)] = 0.0  # safety check for any remaining NaNs

        x = self.pre(x)
        if self.res_block is not None:
            x = x + self.res_block(x)
        x = self.post(x)
        return self.out_layer(x)  # (N, H, W, nP + nD)


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


def load_model(
    config: dict,
    td_keys: dict,
    experiment: str,
    version: str,
    prefer_best: bool = False,
):
    lookback = config.get("train_dataset", {}).get("seviri", {}).get("lookback", 0)
    packing = get_packing_keys(config["train_dataset"])

    model = MultiModalMLP(
        n_prognostic=packing["n_prognostic"],
        n_diagnostic=packing["n_diagnostic"],
        n_input=len(packing["input_keys"]),
        n_forcing_context=len(packing["forcing_context_keys"]),
        lookback=lookback,
        config=config["model"],
    )

    paths = ExperimentPaths()
    ckpt_path = (
        paths.best_checkpoint(experiment, version) if prefer_best else paths.latest_checkpoint(experiment, version)
    )

    lit_module = LitModelModule.load_from_checkpoint(ckpt_path, model=model, config=config, **td_keys)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_module.eval()
    lit_module.to(device)

    return lit_module, device
