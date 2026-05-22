from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from deep_lst.losses.common import gaussian_crps, sigma_key


class DiagnosticLoss(ABC):
    def __init__(self, impute_inputs=True, **kwargs):
        self.diagnostic_keys = kwargs["diagnostic_keys"]
        self.diagnostic_out_keys = kwargs["diagnostic_out_keys"]
        self.in_keys = kwargs.get("in_keys", [])
        self.impute_inputs = impute_inputs

    def __call__(self, lit_module, batch, stage):
        self._maybe_impute(batch)
        out = lit_module.model(batch)
        target = batch[self.diagnostic_keys[0]]
        valid = self._valid_mask(target, out[self.diagnostic_out_keys[0]])
        count = valid.sum()
        lit_module.log(f"{stage}_count", count, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        if int(count.item()) == 0:
            return self._zero_loss(lit_module, stage, out[self.diagnostic_out_keys[0]])
        loss = self._loss(out, self.diagnostic_out_keys[0], target, valid, batch)
        lit_module.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def _maybe_impute(self, batch):
        if self.impute_inputs and self.in_keys:
            for key in self.in_keys:
                batch[key] = torch.nan_to_num(batch[key], nan=0.0)
        return batch

    def _valid_mask(self, target, y_hat):
        return torch.isfinite(target) & torch.isfinite(y_hat)

    def _zero_loss(self, lit_module, stage, y_hat):
        print("no valid pixels found for loss computation!")
        loss = y_hat.sum() * 0.0
        lit_module.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @abstractmethod
    def _loss(self, out, out_key, target, valid, batch) -> torch.Tensor:
        pass


class DiagnosticMSE(DiagnosticLoss):
    def _loss(self, out, out_key, target, valid, batch):
        return F.mse_loss(out[out_key][valid], target[valid])


class DiagnosticCRPS(DiagnosticLoss):
    def _loss(self, out, out_key, target, valid, batch):
        log_sigma = out[sigma_key(out_key)]
        return gaussian_crps(out[out_key][valid], log_sigma[valid], target[valid]).mean()
