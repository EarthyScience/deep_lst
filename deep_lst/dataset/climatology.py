from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr


class ClimatologyLookup(torch.nn.Module):
    def __init__(
        self,
        clim_path: Path,
        spatial_idx_path: Path,  # the zarr's spatial_idx array
        lookback: int,
        horizon: int,
        state_mean: float,
        state_std: float,
        store_time_range: pd.DatetimeIndex,  # STORE_TIME_RANGE
        variable: str = "LST",
    ):
        super().__init__()

        self.lookback = lookback
        self.horizon = horizon

        clim = xr.open_dataset(clim_path)[variable].to_numpy()  # (T, n_pixels), or (T, H, W)
        clim_norm = (clim - state_mean) / state_std  # normalize using training set statistics

        spatial_idx_array = np.load(spatial_idx_path)[
            "mapping"
        ]  # (n_pixels,) mapping from pixel index to spatial index in the climatology
        self.flat_to_rank = {int(v): i for i, v in enumerate(spatial_idx_array)}

        # create Leap-aligned doy
        ts = store_time_range
        natural_doy = ts.dayofyear.values
        is_leap = ts.is_leap_year
        past_feb = ts.month.values >= 3
        leap_doy = natural_doy + np.where(~is_leap & past_feb, 1, 0)
        hour = ts.hour.values
        time_to_clim_row = (leap_doy - 1) * 24 + hour  # (T_store,), int64

        assert time_to_clim_row.min() == 0
        assert time_to_clim_row.max() == 8783

        self.register_buffer("clim", torch.from_numpy(clim_norm))
        self.register_buffer("time_to_clim_row", torch.from_numpy(time_to_clim_row))  # (T_store,)

    def lookup_window(
        self,
        spatial_idx: torch.Tensor,  # (B,)
        time_idx: torch.Tensor,  # (B,)
    ) -> torch.Tensor:  # (B, T_total) where T_total = lookback + horizon + 1
        pixel_ranks = torch.tensor(
            [self.flat_to_rank[int(s)] for s in spatial_idx.cpu()],
            device=self.clim.device,
        )
        offsets = torch.arange(-self.lookback, self.horizon + 1, device=spatial_idx.device)  # (T_total,)
        time_window = time_idx[:, None] + offsets[None, :]  # (B, T_total)
        clim_rows = self.time_to_clim_row[time_window]  # (B, T_total)
        clim_values = self.clim[clim_rows, pixel_ranks[:, None]]  # (B, T_total)
        return clim_values

    def lookup_forward(self, spatial_idx, time_idx, n_steps):  # -> (B, n_steps+1)
        pixel_ranks = torch.tensor([self.flat_to_rank[int(s)] for s in spatial_idx.cpu()], device=self.clim.device)
        offsets = torch.arange(0, n_steps + 1, device=spatial_idx.device)
        time_window = time_idx[:, None] + offsets[None, :]
        clim_rows = self.time_to_clim_row[time_window]
        return self.clim[clim_rows, pixel_ranks[:, None]]
