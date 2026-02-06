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
from deep_lst.utils import ExperimentPaths, ZScoringNormalizer

PATHS = ExperimentPaths()


@dataclass
class ForecastResult:
    """Gridded forecast predictions and observations with coordinates."""

    predictions: np.ndarray  # (N_time, T, lat, lon)
    observations: np.ndarray  # (N_time, T, lat, lon)
    init_times: pd.DatetimeIndex
    lead_times: pd.TimedeltaIndex
    lats: np.ndarray
    lons: np.ndarray

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


class Forecast:
    """Runs model inference and produces ForecastResult."""

    def __init__(self, model, device: torch.device, loader: DataLoader, td_keys: dict, config: dict, loss: str):
        self.model = model
        self.device = device
        self.loader = loader
        self.td_keys = td_keys
        self.config = config
        self.loss = loss

        # Set up normaliser if requested
        self.normaliser = ZScoringNormalizer(PATHS.statistics_path("seviri", config["seviri"]["statistics"]))

        # Resolve grid once
        indices = _load_samples(
            PATHS.samples_path(config["split"]),
            "seviri",
            config["start_time"],
            config["end_time"],
        )
        self.unique_spatial_idx = np.unique(indices[:, 0])
        self.unique_time_idx = np.unique(indices[:, 1])
        self.lats, self.lons, self.local_rows, self.local_cols, self.grid_shape = build_local_grid(
            self.unique_spatial_idx
        )  # suboptimal

    def predict(self) -> ForecastResult:
        """Run inference and return gridded results."""

        predict_fn = self._predict_diagnostic if (self.loss == "DiagnosticMSE") else self._predict_rollout
        preds_flat, obs_flat = predict_fn(
            self.model,
            self.device,
            self.loader,
            self.td_keys,
            normaliser=self.normaliser,
            **self.config["seviri"],
        )

        time_horizon = preds_flat.shape[1]

        predictions, observations = self._remap_to_grid(
            preds_flat,
            obs_flat,
            len(self.unique_spatial_idx),
            len(self.unique_time_idx),
            time_horizon,
            self.local_rows,
            self.local_cols,
            self.grid_shape,
        )

        init_times = pd.to_datetime([T1H.index_to_startdate(int(t)) for t in self.unique_time_idx]).tz_localize(None)
        lead_times = pd.timedelta_range(start="0h", periods=time_horizon, freq="h")

        return ForecastResult(
            predictions=predictions,
            observations=observations,
            init_times=init_times,
            lead_times=lead_times,
            lats=self.lats,
            lons=self.lons,
        )

    @staticmethod
    def _predict_diagnostic(model, device, loader, td_keys, normaliser=None, **kwargs):
        """Single-step: read diagnostic output directly, denormalise per batch."""
        preds, obs = [], []
        diag_key = td_keys["diagnostic_keys"][0]  # e.g. ("seviri", "LST")
        diag_out_key = td_keys["diagnostic_out_keys"][0]
        var = diag_key[1]  # "LST"

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model.model(batch)

                pred_batch = out[diag_out_key].cpu().numpy().squeeze((-3, -2, -1))
                obs_batch = batch[diag_key].cpu().numpy().squeeze((-3, -2, -1))

                if normaliser:
                    pred_batch = normaliser.denormalise(pred_batch, var)
                    obs_batch = normaliser.denormalise(obs_batch, var)

                preds.append(pred_batch)
                obs.append(obs_batch)

        return np.concatenate(preds, axis=0), np.concatenate(obs, axis=0)

    @staticmethod
    def _predict_rollout(model, device, loader, td_keys, normaliser=None, **kwargs):
        """Autoregressive rollout with per-batch denormalisation."""
        all_preds, all_obs = [], []

        prog_keys = td_keys["prognostic_keys"]
        prog_out_keys = td_keys["prognostic_out_keys"]
        diag_keys = td_keys.get("diagnostic_keys", [])
        diag_out_keys = td_keys.get("diagnostic_out_keys", [])

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)

                prog = {key: batch[key][:, 0, ...] for key in prog_keys}
                step_preds, step_obs = [], []

                for t in range(kwargs["lookback"], kwargs["horizon"] + 1):
                    tstep_td = TensorDict({}, batch_size=[batch.batch_size[0]])
                    for key in td_keys["in_keys"]:
                        tstep_td[key] = batch[key][:, t, ...]
                    for key in prog_keys:
                        tstep_td[key] = prog[key]

                    out = model.model(tstep_td)

                    # Update prognostic state
                    for key, out_key in zip(prog_keys, prog_out_keys):
                        prog[key] = prog[key] + out[out_key]

                    # Collect prognostic predictions
                    for key in prog_keys:
                        var = key[1]
                        pred_t = prog[key].cpu().numpy().squeeze((-3, -2, -1))
                        obs_t = batch[key][:, t, ...].cpu().numpy().squeeze((-3, -2, -1))

                        if normaliser:
                            pred_t = normaliser.denormalise(pred_t, var)
                            obs_t = normaliser.denormalise(obs_t, var)

                        step_preds.append(pred_t)
                        step_obs.append(obs_t)

                    # Collect diagnostic predictions
                    for key, out_key in zip(diag_keys, diag_out_keys):
                        var = key[1]
                        pred_t = out[out_key].cpu().numpy().squeeze((-3, -2, -1))
                        obs_t = batch[key][:, t, ...].cpu().numpy().squeeze((-3, -2, -1))

                        if normaliser:
                            pred_t = normaliser.denormalise(pred_t, var)
                            obs_t = normaliser.denormalise(obs_t, var)

                        step_preds.append(pred_t)
                        step_obs.append(obs_t)

                all_preds.append(np.stack(step_preds, axis=1))
                all_obs.append(np.stack(step_obs, axis=1))

        return np.concatenate(all_preds, axis=0), np.concatenate(all_obs, axis=0)

    @staticmethod
    def _remap_to_grid(all_preds, all_obs, n_spatial, n_time, t, local_rows, local_cols, grid_shape):
        preds = all_preds.reshape(n_spatial, n_time, t)
        obs = all_obs.reshape(n_spatial, n_time, t)

        spatial_preds = np.full((n_time, t, *grid_shape), np.nan)
        spatial_obs = np.full((n_time, t, *grid_shape), np.nan)

        spatial_preds[:, :, local_rows, local_cols] = preds.transpose(1, 2, 0)
        spatial_obs[:, :, local_rows, local_cols] = obs.transpose(1, 2, 0)

        return spatial_preds, spatial_obs
