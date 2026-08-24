import torch
import torch.nn as nn

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
