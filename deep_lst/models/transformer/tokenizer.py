import torch
import torch.nn as nn


class Tokenizer(nn.Module):
    def __init__(self, n_forcing_context, n_prognostic, n_input, d_model):
        super().__init__()
        self.window_projection = nn.Linear(2 * n_prognostic + n_forcing_context, d_model)  # per-step channel count
        self.step_projection = nn.Linear(n_input, d_model)
        self.d_model = d_model

    def forward(self, state_window, X_dyn_window, step_X_dyn):
        values = torch.nan_to_num(state_window, nan=0.0)
        mask = torch.isfinite(state_window).to(state_window.dtype)
        # forcings must be NaN-free too: one NaN token poisons the whole
        # sequence through attention, and NaN activations poison the weight
        # gradients even for samples the masked loss excludes
        forcing_values = torch.nan_to_num(X_dyn_window, nan=0.0)
        full_window = torch.cat([values, mask, forcing_values], dim=-1)
        N, L, H, W, C = full_window.shape

        window = full_window.permute(0, 2, 3, 1, 4).reshape(-1, L, C)

        window_tokens = self.window_projection(window)
        step_values = torch.nan_to_num(step_X_dyn, nan=0.0)
        step_token = self.step_projection(step_values).reshape(-1, self.d_model).unsqueeze(1)

        tokens = torch.cat([window_tokens, step_token], dim=1)

        return tokens
