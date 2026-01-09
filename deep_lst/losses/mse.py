import torch
import torch.nn.functional as F

class MaskedMSEStep:
    
    def __init__(self, predictions=("predictions", "LST"),  
                    targets=("seviri", "LST"),           
                    prognostics=("prognostics", "seviri"),
                    inputs_era=("inputs", "era5"),
                    inputs_seviri=("inputs", "seviri"),
    
        ):
        self.predictions = predictions
        self.targets = targets
        self.prognostics = prognostics
        self.inputs_era = inputs_era
        self.inputs_seviri = inputs_seviri

    def __call__(self, lit_module, batch, stage: str):
        """
        Masked mean squared error loss step for PyTorch Lightning modules.
        """

        out = lit_module.model(batch)

        y_hat = out[self.predictions]
        y     = out[self.targets]

        valid = torch.isfinite(y) & torch.isfinite(y_hat)
        
        n_valid = valid.sum().item() 
        lit_module.log(f"{stage}_valid_count", n_valid, prog_bar=True, sync_dist=True)

        if n_valid == 0:
            del out, y_hat, y, valid
            return None

        loss = F.mse_loss(y_hat[valid], y[valid])

        lit_module.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)

        del out, y_hat, y, valid
        return loss


class MaskedMSERollout:
    
    def __init__(self, 
                    predictions=("predictions", "LST"),  
                    targets=("seviri", "LST"),           
                    prognostics=("prognostics", "seviri"),
                    inputs_era=("inputs", "era5"),
                    inputs_seviri=("inputs", "seviri"),
                    detach_rollout=False,  # True = truncated rollout, False = full BPTT through rollout
                    ):

        self.predictions = predictions
        self.targets = targets
        self.prognostics = prognostics
        self.inputs_era = inputs_era
        self.inputs_seviri = inputs_seviri
        self.detach_rollout = detach_rollout

    def __call__(self, lit_module, batch, stage: str):
        """
        Masked mean squared error loss step for PyTorch Lightning modules.
        """
        horizon = lit_module.config["horizon"]

        loss = 0.0

        prog = batch[self.prognostics][:, 0, ...]

        for timestep in range(1, horizon+1):

            in_era = batch[self.inputs_era][:, timestep, ...]
            in_seviri = batch[self.inputs_seviri][:, timestep, ...]

            out = lit_module.model(in_era, in_seviri, prog)

            logits = out[self.predictions]
            y_hat = prog + logits  # predicted absolute state at time t
            y = batch[self.targets][:,timestep, ...]

            valid = torch.isfinite(y) & torch.isfinite(y_hat)
            loss += F.mse_loss(y_hat[valid], y[valid])

            prog = y_hat.detach() if self.detach_rollout else y_hat

        lit_module.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)

        del out, y_hat, y, valid
        return loss
