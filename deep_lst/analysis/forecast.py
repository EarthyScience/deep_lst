from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tensordict import TensorDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from wedata.grids import T1H

from deep_lst.analysis.spatial import build_local_grid
from deep_lst.dataset.dataset import _load_samples
from deep_lst.utils import ExperimentPaths, ZScoringNormalizer

PATHS = ExperimentPaths()


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

        # Extract variable names from data vars (e.g. "LST_pred" -> "LST")
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
        prediction_mode: str = "diagnostic",
        apply_consistent_mask: bool = True,  # not used at the moment
    ):
        self.model = model
        self.device = device
        self.loader = loader
        self.td_keys = td_keys
        self.config = config
        self.prediction_mode = prediction_mode
        self.apply_consistent_mask = apply_consistent_mask

        # Set up normaliser
        self.normaliser = ZScoringNormalizer(PATHS.statistics_path("seviri", config["seviri"]["statistics"]))

        # All variables we'll predict (prognostic + diagnostic)
        self.prognostic_vars = [key[1] for key in td_keys.get("prognostic_keys", [])]
        self.diagnostic_vars = [key[1] for key in td_keys.get("diagnostic_keys", [])]
        self.all_vars = self.prognostic_vars + self.diagnostic_vars

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

        # Returns dict[var, np.ndarray] for preds and obs
        preds_flat, obs_flat = predict_fn(
            self.model,
            self.device,
            self.loader,
            self.td_keys,
            normaliser=self.normaliser,
            **self.config["seviri"],
        )
        # Apply consistent cloud mask across all lead times (optional)
        if self.apply_consistent_mask and self.prediction_mode != "DiagnosticMSE":
            preds_flat, obs_flat = self._apply_consistent_mask(preds_flat, obs_flat)

        # Infer time horizon from first variable
        first_var = list(preds_flat.keys())[0]
        time_horizon = preds_flat[first_var].shape[1]

        # Remap each variable to grid
        predictions = {}
        observations = {}
        for var in preds_flat:
            pred_grid, obs_grid = self._remap_to_grid(
                preds_flat[var],
                obs_flat[var],
                len(self.unique_spatial_idx),
                len(self.unique_time_idx),
                time_horizon,
                self.local_rows,
                self.local_cols,
                self.grid_shape,
            )
            predictions[var] = pred_grid
            observations[var] = obs_grid

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
        """Single-step: read all diagnostic outputs, denormalise per batch."""
        diag_keys = td_keys["diagnostic_keys"]
        diag_out_keys = td_keys["diagnostic_out_keys"]

        # Accumulators per variable
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

        # Concatenate batches for each variable
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

        # Accumulators per variable
        all_preds = {var: [] for var in all_vars}
        all_obs = {var: [] for var in all_vars}

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting (rollout)"):
                batch = batch.to(device)

                prog = {key: batch[key][:, 0, ...] for key in prog_keys}

                # Per-timestep accumulators for this batch
                batch_preds = {var: [] for var in all_vars}
                batch_obs = {var: [] for var in all_vars}

                for t in range(kwargs["lookback"], kwargs["horizon"] + 1):
                    tstep_td = TensorDict({}, batch_size=[batch.batch_size[0]])
                    for key in td_keys["in_keys"]:
                        tstep_td[key] = batch[key][:, t, ...]
                    for key in prog_keys:
                        tstep_td[key] = prog[key]

                    out = model.model(tstep_td)

                    # Update and collect prognostic state
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

                    # Collect diagnostic predictions
                    for key, out_key in zip(diag_keys, diag_out_keys):
                        var = key[1]

                        pred_t = out[out_key].cpu().numpy().squeeze((-3, -2, -1))
                        obs_t = batch[key][:, t, ...].cpu().numpy().squeeze((-3, -2, -1))

                        if normaliser:
                            pred_t = normaliser.denormalise(pred_t, var)
                            obs_t = normaliser.denormalise(obs_t, var)

                        batch_preds[var].append(pred_t)
                        batch_obs[var].append(obs_t)

                # Stack timesteps for this batch: list of (B,) -> (B, T)
                for var in all_vars:
                    all_preds[var].append(np.stack(batch_preds[var], axis=1))
                    all_obs[var].append(np.stack(batch_obs[var], axis=1))

        # Concatenate batches for each variable
        preds = {var: np.concatenate(all_preds[var], axis=0) for var in all_vars}
        obs = {var: np.concatenate(all_obs[var], axis=0) for var in all_vars}

        return preds, obs

    @staticmethod
    def _apply_consistent_mask(
        preds: dict[str, np.ndarray], obs: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Mask to pixels valid across all lead times for each variable.
        Args:
            preds: {var: (N_samples, T, ...)} predictions
            obs: {var: (N_samples, T, ...)} observations
        Returns:
            Masked preds and obs with NaN where any timestep was invalid.
        """
        preds_masked = {}
        obs_masked = {}

        for var in obs:
            o = obs[var]
            p = preds[var]

            # Valid if finite at ALL lead times: (N_samples, ...) -> broadcast to (N_samples, T, ...)
            # Assumes shape is (N_samples, T) or (N_samples, T, H, W, ...)
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
