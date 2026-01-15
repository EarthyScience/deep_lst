import torch
import torch.nn.functional as F
from tensordict import TensorDict

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

        # Verify loss is scalar and has gradient
        print(f"  Loss: {loss.item():.4f}, requires_grad: {loss.requires_grad}")

        lit_module.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

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
                    min_valid_fraction=0.05,  # Minimum fraction of valid pixels to compute loss
                    ):

        self.predictions = predictions
        self.targets = targets
        self.prognostics = prognostics
        self.inputs_era = inputs_era
        self.inputs_seviri = inputs_seviri
        self.detach_rollout = detach_rollout

        self.min_valid_fraction = min_valid_fraction

    def __call__(self, lit_module, batch, stage: str):

        horizon = lit_module.config["horizon"]
        loss = 0.0
        valid_timesteps = 0

        # Get initial prognostic state (t=0)
        prog_raw = batch[self.prognostics][:, 0, ...].clone()  # [B, C, H, W]
        
        # Create mask of initially valid pixels
        initial_valid_mask = torch.isfinite(prog_raw)  # [B, C, H, W]
        num_initially_valid = initial_valid_mask.sum()
        
        if num_initially_valid == 0:
            print(f"  ERROR: No initially valid pixels!")
            return None
        
        # temporarily fill NaNs with mean of valid pixels
        if initial_valid_mask.any():
            valid_mean = prog_raw[initial_valid_mask].mean()
        else:
            valid_mean = 0.0  # Fallback if no valid pixels at all
        prog = torch.where(initial_valid_mask, prog_raw, valid_mean)

        # Get full tensors (keep spatial structure)
        era = batch[self.inputs_era]      # [B, T, C, H, W]
        seviri = batch[self.inputs_seviri]  # [B, T, C, H, W]
        targets = batch[self.targets]      # [B, T, C, H, W]

        for timestep in range(1, horizon + 1):
            # Build input for this timestep - prog now has NO NaNs
            tstep_td = TensorDict(
                {
                    self.inputs_era: era[:, timestep, ...],
                    self.inputs_seviri: seviri[:, timestep, ...],
                    self.prognostics: prog,  # Clean, no NaNs
                },
                batch_size=[prog.shape[0]],
            )

            out = lit_module.model(tstep_td)
            logits = out[self.predictions]  # Should have fewer NaNs now
            
            y_hat = prog + logits

            y_targ = targets[:, timestep, ...]

            # Valid pixels: initially valid AND have valid target at this timestep
            valid_at_this_step = initial_valid_mask & torch.isfinite(y_targ)
            num_valid = valid_at_this_step.sum()
            
            pixels_per_sample = lit_module.config["patch_size"] ** 2 * lit_module.config["batch_size"]
            valid_fraction = num_valid.float() / pixels_per_sample
            
            if valid_fraction >= self.min_valid_fraction:
                timestep_loss = F.mse_loss(
                    y_hat[valid_at_this_step], 
                    y_targ[valid_at_this_step]
                )
                loss += timestep_loss
                valid_timesteps += 1
                
            # Update prognostic: only update initially valid pixels
            prog = prog.clone()
            prog[initial_valid_mask] = y_hat[initial_valid_mask]
            
            if self.detach_rollout:
                prog = prog.detach()
            

        lit_module.log(f"{stage}_loss_step", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        lit_module.log(f"{stage}_valid_timesteps", float(valid_timesteps), sync_dist=True)
        lit_module.log(f"{stage}_initially_valid_pixels", float(num_initially_valid), sync_dist=True)

        del out, logits, y_hat, y_targ, valid_at_this_step

        return loss