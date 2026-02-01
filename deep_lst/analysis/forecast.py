from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tensordict import TensorDict
from torch.utils.data import DataLoader
from wedata.grids import T1H

from deep_lst.analysis.spatial import build_local_grid
from deep_lst.dataset.dataset import _load_samples


@dataclass
class ForecastResult:
    """Gridded forecast predictions and observations with coordinates."""

    predictions: np.ndarray  # (N_time, T, lat, lon)
    observations: np.ndarray  # (N_time, T, lat, lon)
    init_times: pd.DatetimeIndex
    lead_times: pd.TimedeltaIndex
    lats: np.ndarray
    lons: np.ndarray

    @classmethod
    def from_model(
        cls,
        model,
        device: torch.device,
        loader: DataLoader,
        td_keys: dict,
        config: dict,
        samples_path: Path,
        predict_fn: None,
    ) -> "ForecastResult":
        """Run inference, remap to grid, and return a complete ForecastResult.

        Args:
            model:         LitModelModule in eval mode, on device.
            device:        torch.device the model lives on.
            loader:        DataLoader over the test dataset.
            td_keys:       TensorDict keys from get_keys_from_config.
            config: The test_dataset section of the config
                           (needs start_time, end_time, seviri.lookback, seviri.horizon).
            samples_path:  Path to the .npz file with spatial indices.
        """
        # --- predict ---
        predict_fn = predict_fn or cls._predict_diagnostic
        preds_flat, obs_flat = predict_fn(model, device, loader, td_keys, **config["seviri"])

        # --- resolve local grid ---
        indices = _load_samples(
            samples_path,
            "seviri",
            config["start_time"],
            config["end_time"],
        )
        unique_spatial_idx = np.unique(indices[:, 0])
        unique_time_idx = np.unique(indices[:, 1])

        # Infer lead-time length from actual predictions
        # (diagnostic: lookback + horizon + 1, rollout: horizon)
        time_horizon = preds_flat.shape[1]

        lats, lons, local_rows, local_cols, grid_shape = build_local_grid(unique_spatial_idx)
        print("Mapping to local grid of shape:", grid_shape)
        print("Lat range:", lats.min(), lats.max())
        print("Lon range:", lons.min(), lons.max())

        # --- remap ---
        predictions, observations = cls._remap_to_grid(
            preds_flat,
            obs_flat,
            len(unique_spatial_idx),
            len(unique_time_idx),
            time_horizon,
            local_rows,
            local_cols,
            grid_shape,
        )

        # --- coordinates ---
        init_times = pd.to_datetime([T1H.index_to_startdate(int(t)) for t in unique_time_idx]).tz_localize(None)
        lead_times = pd.timedelta_range(start="0h", periods=time_horizon, freq="h")

        return cls(
            predictions=predictions,
            observations=observations,
            init_times=init_times,
            lead_times=lead_times,
            lats=lats,
            lons=lons,
        )

    def to_xarray(self) -> xr.Dataset:
        return xr.Dataset(
            {
                "LST_pred": (["init_time", "lead_time", "lat", "lon"], self.predictions),
                "LST_obs": (["init_time", "lead_time", "lat", "lon"], self.observations),
            },
            coords={
                "lat": self.lats,
                "lon": self.lons,
                "init_time": self.init_times,
                "lead_time": self.lead_times,
            },
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_xarray().to_netcdf(path)
        print(f"Saved forecast results to {path}")

    @classmethod
    def load(cls, path: Path) -> "ForecastResult":
        ds = xr.open_dataset(path)
        return cls(
            predictions=ds["LST_pred"].values,
            observations=ds["LST_obs"].values,
            init_times=pd.DatetimeIndex(ds["init_time"].values),
            lead_times=pd.TimedeltaIndex(ds["lead_time"].values),
            lats=ds["lat"].values,
            lons=ds["lon"].values,
        )

    @staticmethod
    def _predict_diagnostic(model, device, loader, td_keys, **kwargs):
        """Single-step: read diagnostic output directly."""
        preds, obs = [], []
        with torch.no_grad():
            for batch in loader:
                print(f"Processing batch of size {batch.batch_size}...")
                batch = batch.to(device)
                out = model.model(batch)
                preds.append(out[td_keys["diagnostic_out_keys"][0]].cpu())
                obs.append(batch[td_keys["diagnostic_keys"][0]].cpu())
        return (
            torch.cat(preds, dim=0).numpy().squeeze((-3, -2, -1)),
            torch.cat(obs, dim=0).numpy().squeeze((-3, -2, -1)),
        )

    @staticmethod
    def _predict_rollout(model, device, loader, td_keys, **kwargs):
        """Autoregressive rollout: predict step-by-step, feeding prognostic state forward."""
        all_preds, all_obs = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                print(f"Processing batch of size {batch.batch_size}...")
                # Initialise prognostic state at t=0
                prog = {}
                for key in td_keys["prognostic_keys"]:
                    prog[key] = batch[key][:, 0, ...]

                step_preds = []
                step_obs = []

                for t in range(kwargs["lookback"], kwargs["horizon"] + 1):
                    # Build single-timestep input
                    tstep_td = TensorDict({}, batch_size=[batch.batch_size[0]])
                    for key in td_keys["in_keys"]:
                        tstep_td[key] = batch[key][:, t, ...]
                    for key in td_keys["prognostic_keys"]:
                        tstep_td[key] = prog[key]

                    out = model.model(tstep_td)

                    # Update prognostic state with predicted increment
                    for key, out_key in zip(td_keys["prognostic_keys"], td_keys["prognostic_out_keys"]):
                        pred_increment = out[out_key]
                        prog[key] = prog[key] + pred_increment

                    # Collect prognostic state predictions
                    for key in td_keys["prognostic_keys"]:
                        step_preds.append(prog[key].cpu())
                        step_obs.append(batch[key][:, t, ...].cpu())

                    # Collect diagnostic predictions (if any)
                    if td_keys.get("diagnostic_out_keys"):
                        for key, out_key in zip(td_keys["diagnostic_keys"], td_keys["diagnostic_out_keys"]):
                            step_preds.append(out[out_key].cpu())
                            step_obs.append(batch[key][:, t, ...].cpu())

                # (horizon, B, ...) → (B, horizon, ...)
                all_preds.append(torch.stack(step_preds, dim=1))
                all_obs.append(torch.stack(step_obs, dim=1))

        all_preds = torch.cat(all_preds, dim=0).numpy().squeeze((-3, -2, -1))
        all_obs = torch.cat(all_obs, dim=0).numpy().squeeze((-3, -2, -1))
        return all_preds, all_obs

    @staticmethod
    def _remap_to_grid(all_preds, all_obs, n_spatial, n_time, t, local_rows, local_cols, grid_shape):
        preds = all_preds.reshape(n_spatial, n_time, t)
        obs = all_obs.reshape(n_spatial, n_time, t)

        spatial_preds = np.full((n_time, t, *grid_shape), np.nan)
        spatial_obs = np.full((n_time, t, *grid_shape), np.nan)

        spatial_preds[:, :, local_rows, local_cols] = preds.transpose(1, 2, 0)
        spatial_obs[:, :, local_rows, local_cols] = obs.transpose(1, 2, 0)

        return spatial_preds, spatial_obs
