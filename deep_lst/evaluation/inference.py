from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tensordict import TensorDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_lst.evaluation.spatial import build_local_grid
from deep_lst.utils import ExperimentPaths, ZScoringNormalizer

PATHS = ExperimentPaths()

# Origin of the integer time indices used by TemporalFilter / EOZarr.
# Index 0 = 2016-01-01 00:00 UTC. Must match the start_date passed to TemporalFilter.
TIME_INDEX_ORIGIN = pd.Timestamp("2016-01-01")


@dataclass
class ForecastResult:
    """Gridded forecast predictions and observations with coordinates."""

    predictions: dict[str, np.ndarray]  # {var: (N_time, T, lat, lon)}
    observations: dict[str, np.ndarray]  # {var: (N_time, T, lat, lon)}
    init_times: pd.DatetimeIndex
    lead_times: pd.TimedeltaIndex
    lats: np.ndarray
    lons: np.ndarray

    def to_xarray(self) -> xr.Dataset:
        data_vars = {}
        for var in self.predictions:
            data_vars[f"{var}_pred"] = (["init_time", "lead_time", "lat", "lon"], self.predictions[var])
            data_vars[f"{var}_obs"] = (["init_time", "lead_time", "lat", "lon"], self.observations[var])

        return xr.Dataset(
            data_vars,
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
        pred_vars = [v.replace("_pred", "") for v in ds.data_vars if v.endswith("_pred")]
        predictions = {var: ds[f"{var}_pred"].values for var in pred_vars}
        observations = {var: ds[f"{var}_obs"].values for var in pred_vars}
        return cls(
            predictions=predictions,
            observations=observations,
            init_times=pd.DatetimeIndex(ds["init_time"].values),
            lead_times=pd.TimedeltaIndex(ds["lead_time"].values),
            lats=ds["lat"].values,
            lons=ds["lon"].values,
        )


class Forecast:
    """Runs model inference and produces ForecastResult."""

    def __init__(
        self,
        model,
        device: torch.device,
        loader: DataLoader,
        td_keys: dict,
        config: dict,
        spatial_indices: np.ndarray,
        prediction_mode: str = "diagnostic",
        apply_consistent_mask: bool = True,
    ):
        self.model = model
        self.device = device
        self.loader = loader
        self.td_keys = td_keys
        self.config = config
        self.prediction_mode = prediction_mode
        self.apply_consistent_mask = apply_consistent_mask

        self.normaliser = ZScoringNormalizer(PATHS.statistics_path("seviri", config["seviri"]["statistics"]))

        self.prognostic_vars = [key[1] for key in td_keys.get("prognostic_keys", [])]
        self.diagnostic_vars = [key[1] for key in td_keys.get("diagnostic_keys", [])]

        self.unique_spatial_idx = np.unique(spatial_indices)
        self.lats, self.lons, self.local_rows, self.local_cols, self.grid_shape = build_local_grid(
            self.unique_spatial_idx
        )

    def predict(self) -> ForecastResult:
        """Run inference and return gridded results."""
        predict_dispatch = {
            "DiagnosticMSE": self._predict_diagnostic,
            "RolloutMSE": self._predict_rollout,
        }
        predict_fn = predict_dispatch.get(self.prediction_mode)
        if predict_fn is None:
            raise ValueError(f"Unknown prediction_mode: {self.prediction_mode}. Known: {list(predict_dispatch)}")

        preds_flat, obs_flat = predict_fn(
            self.model,
            self.device,
            self.loader,
            self.td_keys,
            normaliser=self.normaliser,
            **self.config["seviri"],
        )

        if self.apply_consistent_mask and self.prediction_mode != "DiagnosticMSE":
            preds_flat, obs_flat = self._apply_consistent_mask(preds_flat, obs_flat)

        first_var = list(preds_flat.keys())[0]
        time_horizon = preds_flat[first_var].shape[1]

        # Infer number of time steps from data length and number of spatial indices
        n_spatial = len(self.unique_spatial_idx)
        n_time = preds_flat[first_var].shape[0] // n_spatial

        predictions = {}
        observations = {}
        for var in preds_flat:
            pred_grid, obs_grid = self._remap_to_grid(
                preds_flat[var],
                obs_flat[var],
                n_spatial,
                n_time,
                time_horizon,
                self.local_rows,
                self.local_cols,
                self.grid_shape,
            )
            predictions[var] = pred_grid
            observations[var] = obs_grid

        # Reconstruct init times from the dataset's time indices
        # The loader iterates in order; we recover unique time steps from the batch count
        lead_times = pd.timedelta_range(start="1h", periods=time_horizon, freq="h")

        # Use a placeholder index; proper init times require the dataset's time_idx
        init_times = pd.date_range(start="2000-01-01", periods=n_time, freq="h")

        return ForecastResult(
            predictions=predictions,
            observations=observations,
            init_times=init_times,
            lead_times=lead_times,
            lats=self.lats,
            lons=self.lons,
        )

    def predict_with_time_index(self, time_idx: np.ndarray) -> ForecastResult:
        """Run inference and map init_times from explicit T1H time indices."""
        predict_dispatch = {
            "DiagnosticMSE": self._predict_diagnostic,
            "RolloutMSE": self._predict_rollout,
        }
        predict_fn = predict_dispatch.get(self.prediction_mode)
        if predict_fn is None:
            raise ValueError(f"Unknown prediction_mode: {self.prediction_mode}. Known: {list(predict_dispatch)}")

        preds_flat, obs_flat = predict_fn(
            self.model,
            self.device,
            self.loader,
            self.td_keys,
            normaliser=self.normaliser,
            **self.config["seviri"],
        )

        if self.apply_consistent_mask and self.prediction_mode != "DiagnosticMSE":
            preds_flat, obs_flat = self._apply_consistent_mask(preds_flat, obs_flat)

        first_var = list(preds_flat.keys())[0]
        time_horizon = preds_flat[first_var].shape[1]
        n_spatial = len(self.unique_spatial_idx)
        unique_time_idx = np.unique(time_idx)
        n_time = len(unique_time_idx)

        predictions = {}
        observations = {}
        for var in preds_flat:
            pred_grid, obs_grid = self._remap_to_grid(
                preds_flat[var],
                obs_flat[var],
                n_spatial,
                n_time,
                time_horizon,
                self.local_rows,
                self.local_cols,
                self.grid_shape,
            )
            predictions[var] = pred_grid
            observations[var] = obs_grid

        init_times = TIME_INDEX_ORIGIN + pd.to_timedelta(unique_time_idx.astype(np.int64), unit="h")
        lead_times = pd.timedelta_range(start="1h", periods=time_horizon, freq="h")

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
        """Single-step: read all diagnostic outputs, denormalise per batch."""
        diag_keys = td_keys["diagnostic_keys"]
        diag_out_keys = td_keys["diagnostic_out_keys"]

        preds = {key[1]: [] for key in diag_keys}
        obs = {key[1]: [] for key in diag_keys}

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting (diagnostic)"):
                batch = batch.to(device)
                out = model.model(batch)

                for diag_key, diag_out_key in zip(diag_keys, diag_out_keys):
                    var = diag_key[1]
                    pred_batch = out[diag_out_key].cpu().numpy().squeeze((-3, -2, -1))
                    obs_batch = batch[diag_key].cpu().numpy().squeeze((-3, -2, -1))

                    if normaliser:
                        pred_batch = normaliser.denormalise(pred_batch, var)
                        obs_batch = normaliser.denormalise(obs_batch, var)

                    preds[var].append(pred_batch)
                    obs[var].append(obs_batch)

        preds = {var: np.concatenate(preds[var], axis=0) for var in preds}
        obs = {var: np.concatenate(obs[var], axis=0) for var in obs}
        return preds, obs

    @staticmethod
    def _predict_rollout(model, device, loader, td_keys, normaliser=None, **kwargs):
        """Autoregressive rollout with per-batch denormalisation."""
        prog_keys = td_keys["prognostic_keys"]
        prog_out_keys = td_keys["prognostic_out_keys"]
        diag_keys = td_keys.get("diagnostic_keys", [])
        diag_out_keys = td_keys.get("diagnostic_out_keys", [])

        all_vars = [k[1] for k in prog_keys] + [k[1] for k in diag_keys]

        all_preds = {var: [] for var in all_vars}
        all_obs = {var: [] for var in all_vars}

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting (rollout)"):
                batch = batch.to(device)
                prog = {key: batch[key][:, 0, ...] for key in prog_keys}

                batch_preds = {var: [] for var in all_vars}
                batch_obs = {var: [] for var in all_vars}

                for t in range(1, kwargs["horizon"] + 1):
                    tstep_td = TensorDict({}, batch_size=[batch.batch_size[0]])
                    for key in td_keys["in_keys"]:
                        tstep_td[key] = batch[key][:, t, ...]
                    for key in prog_keys:
                        tstep_td[key] = prog[key]

                    out = model.model(tstep_td)

                    for key, out_key in zip(prog_keys, prog_out_keys):
                        var = key[1]
                        prog[key] = prog[key] + out[out_key]
                        pred_t = prog[key].cpu().numpy().squeeze((-3, -2, -1))
                        obs_t = batch[key][:, t, ...].cpu().numpy().squeeze((-3, -2, -1))
                        if normaliser:
                            pred_t = normaliser.denormalise(pred_t, var)
                            obs_t = normaliser.denormalise(obs_t, var)
                        batch_preds[var].append(pred_t)
                        batch_obs[var].append(obs_t)

                    for key, out_key in zip(diag_keys, diag_out_keys):
                        var = key[1]
                        pred_t = out[out_key].cpu().numpy().squeeze((-3, -2, -1))
                        obs_t = batch[key][:, t, ...].cpu().numpy().squeeze((-3, -2, -1))
                        if normaliser:
                            pred_t = normaliser.denormalise(pred_t, var)
                            obs_t = normaliser.denormalise(obs_t, var)
                        batch_preds[var].append(pred_t)
                        batch_obs[var].append(obs_t)

                for var in all_vars:
                    all_preds[var].append(np.stack(batch_preds[var], axis=1))
                    all_obs[var].append(np.stack(batch_obs[var], axis=1))

        preds = {var: np.concatenate(all_preds[var], axis=0) for var in all_vars}
        obs = {var: np.concatenate(all_obs[var], axis=0) for var in all_vars}
        return preds, obs

    @staticmethod
    def _apply_consistent_mask(
        preds: dict[str, np.ndarray], obs: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        preds_masked = {}
        obs_masked = {}
        for var in obs:
            o = obs[var]
            p = preds[var]
            all_valid = np.all(np.isfinite(o), axis=1, keepdims=True)
            preds_masked[var] = np.where(all_valid, p, np.nan)
            obs_masked[var] = np.where(all_valid, o, np.nan)
        return preds_masked, obs_masked

    @staticmethod
    def _remap_to_grid(all_preds, all_obs, n_spatial, n_time, t, local_rows, local_cols, grid_shape):
        preds = all_preds.reshape(n_spatial, n_time, t)
        obs = all_obs.reshape(n_spatial, n_time, t)
        spatial_preds = np.full((n_time, t, *grid_shape), np.nan)
        spatial_obs = np.full((n_time, t, *grid_shape), np.nan)
        spatial_preds[:, :, local_rows, local_cols] = preds.transpose(1, 2, 0)
        spatial_obs[:, :, local_rows, local_cols] = obs.transpose(1, 2, 0)
        return spatial_preds, spatial_obs
