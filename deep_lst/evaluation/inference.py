from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from tensordict import TensorDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_lst.evaluation.spatial import build_local_grid
from deep_lst.utils import ExperimentPaths, ZScoringNormalizer

PATHS = ExperimentPaths()

# Origin of the integer time indices used by TemporalFilter / EOZarr.
# Index 0 = 2016-01-01 00:00 UTC. Must match the start_date passed to TemporalFilter.
TIME_INDEX_ORIGIN = pd.Timestamp("2016-01-01")

# Pixels below this fill fraction of the bounding-box grid use sparse output.
# Avoids allocating a mostly-NaN (H x W) grid for scattered training-site pixels.
_SPARSE_FILL_THRESHOLD = 0.01


@dataclass
class ForecastResult:
    """Forecast predictions and observations with coordinates.

    Dense mode  (sparse=False): predictions shape (N_time, T, lat, lon) — for
        geographically concentrated eval sites where the bounding box is small.
    Sparse mode (sparse=True):  predictions shape (N_pixel, N_time, T) — for
        scattered training-site pixels where the bounding box would be huge.
    """

    predictions: dict[str, np.ndarray]
    observations: dict[str, np.ndarray]
    init_times: pd.DatetimeIndex
    lead_times: pd.TimedeltaIndex
    lats: np.ndarray
    lons: np.ndarray
    sparse: bool = field(default=False)

    def to_xarray(self) -> xr.Dataset:
        data_vars = {}
        if self.sparse:
            for var in self.predictions:
                data_vars[f"{var}_pred"] = (["pixel", "init_time", "lead_time"], self.predictions[var])
                data_vars[f"{var}_obs"] = (["pixel", "init_time", "lead_time"], self.observations[var])
            return xr.Dataset(
                data_vars,
                coords={
                    "lat": ("pixel", self.lats),
                    "lon": ("pixel", self.lons),
                    "init_time": self.init_times,
                    "lead_time": self.lead_times,
                },
            )
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
        ds = self.to_xarray()
        first_var = next(iter(ds.data_vars))
        shape = ds[first_var].shape
        if self.sparse:
            enc = {v: {"chunks": (min(shape[0], 1000), shape[1], shape[2])} for v in ds.data_vars}
        else:
            enc = {v: {"chunks": (1, shape[1], min(shape[2], 256), min(shape[3], 256))} for v in ds.data_vars}
        for c in ds.coords:
            enc[c] = {"chunks": (ds[c].shape[0],)}
        ds.to_zarr(path, encoding=enc, mode="w")
        print(f"Saved forecast results to {path}")

    @classmethod
    def load(cls, path: Path) -> "ForecastResult":
        ds = xr.open_zarr(path)
        pred_vars = [v.replace("_pred", "") for v in ds.data_vars if v.endswith("_pred")]
        predictions = {var: ds[f"{var}_pred"].values for var in pred_vars}
        observations = {var: ds[f"{var}_obs"].values for var in pred_vars}
        sparse = "pixel" in ds.dims
        return cls(
            predictions=predictions,
            observations=observations,
            init_times=pd.DatetimeIndex(ds["init_time"].values),
            lead_times=pd.TimedeltaIndex(ds["lead_time"].values),
            lats=ds["lat"].values,
            lons=ds["lon"].values,
            sparse=sparse,
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

        fill = len(self.unique_spatial_idx) / (self.grid_shape[0] * self.grid_shape[1])
        self.sparse = fill < _SPARSE_FILL_THRESHOLD
        if self.sparse:
            print(f"Sparse output mode (fill={fill:.4f} < {_SPARSE_FILL_THRESHOLD})")

    def predict_with_time_index(self, time_idx: np.ndarray) -> ForecastResult:
        """Run inference and map init_times from explicit T1H time indices."""
        predict_dispatch = {
            "DiagnosticMSE": self._predict_diagnostic,
            "DiagnosticCRPS": self._predict_diagnostic,
            "RolloutMSE": self._predict_rollout,
            "RolloutCRPS": self._predict_rollout,
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

        if self.apply_consistent_mask and self.prediction_mode in ("RolloutMSE", "RolloutCRPS"):
            preds_flat, obs_flat = self._apply_consistent_mask(preds_flat, obs_flat)

        first_var = list(preds_flat.keys())[0]
        time_horizon = preds_flat[first_var].shape[1]
        n_spatial = len(self.unique_spatial_idx)
        unique_time_idx = np.unique(time_idx)
        n_time = preds_flat[first_var].shape[0] // n_spatial
        unique_time_idx = unique_time_idx[:n_time]

        predictions, observations = self._remap(preds_flat, obs_flat, n_spatial, n_time, time_horizon)

        if self.prediction_mode in ("RolloutMSE", "RolloutCRPS"):
            predictions, observations, time_horizon = self._prepend_nan_lead(
                predictions, observations, n_spatial, n_time, time_horizon
            )

        init_times = TIME_INDEX_ORIGIN + pd.to_timedelta(unique_time_idx.astype(np.int64), unit="h")
        lead_times = pd.timedelta_range(start="0h", periods=time_horizon, freq="h")

        pixel_lats, pixel_lons = self._pixel_coords()
        return ForecastResult(
            predictions=predictions,
            observations=observations,
            init_times=init_times,
            lead_times=lead_times,
            lats=pixel_lats,
            lons=pixel_lons,
            sparse=self.sparse,
        )

    def predict_and_save(self, path: Path, time_idx: np.ndarray, time_offset: int | None = None) -> None:
        """Run inference and write directly to zarr without holding the full grid in memory.

        time_offset=None  (default): single-job mode — creates store, fills all steps, consolidates.
        time_offset=int:             distributed fill — writes into an existing store at the given offset.
        """
        predict_dispatch = {
            "DiagnosticMSE": self._predict_diagnostic,
            "DiagnosticCRPS": self._predict_diagnostic,
            "RolloutMSE": self._predict_rollout,
            "RolloutCRPS": self._predict_rollout,
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

        if self.apply_consistent_mask and self.prediction_mode in ("RolloutMSE", "RolloutCRPS"):
            preds_flat, obs_flat = self._apply_consistent_mask(preds_flat, obs_flat)

        first_var = list(preds_flat.keys())[0]
        time_horizon = preds_flat[first_var].shape[1]
        n_spatial = len(self.unique_spatial_idx)
        unique_time_idx = np.unique(time_idx)[: preds_flat[first_var].shape[0] // n_spatial]
        n_time = len(unique_time_idx)

        is_rollout = self.prediction_mode in ("RolloutMSE", "RolloutCRPS")
        t_out = time_horizon + 1 if is_rollout else time_horizon
        init_times = TIME_INDEX_ORIGIN + pd.to_timedelta(unique_time_idx.astype(np.int64), unit="h")
        lead_times = pd.timedelta_range(start="0h", periods=t_out, freq="h")
        pixel_lats, pixel_lons = self._pixel_coords()

        if self.sparse:
            spatial_shape = (len(self.unique_spatial_idx),)
        else:
            H, W = self.grid_shape
            spatial_shape = (int(H), int(W))

        if time_offset is None:
            self._create_zarr_store(
                path,
                list(preds_flat.keys()),
                n_time,
                t_out,
                spatial_shape,
                init_times,
                lead_times,
                pixel_lats,
                pixel_lons,
            )
            self._fill_zarr_store(
                path,
                preds_flat,
                obs_flat,
                n_spatial,
                n_time,
                time_horizon,
                self.local_rows,
                self.local_cols,
                self.grid_shape,
                0,
                is_rollout,
                sparse=self.sparse,
            )
            zarr.consolidate_metadata(str(path))
        else:
            self._fill_zarr_store(
                path,
                preds_flat,
                obs_flat,
                n_spatial,
                n_time,
                time_horizon,
                self.local_rows,
                self.local_cols,
                self.grid_shape,
                time_offset,
                is_rollout,
                sparse=self.sparse,
            )
        print(f"Done. Saved to {path}")

    @staticmethod
    def setup_output(
        path: Path,
        td_keys: dict,
        spatial_indices: np.ndarray,
        time_idx: np.ndarray,
        inference_horizon: int,
        prediction_mode: str,
    ) -> None:
        """Create empty zarr store for distributed inference. No model weights required."""
        unique_spatial_idx = np.unique(spatial_indices)
        lats, lons, local_rows, local_cols, grid_shape = build_local_grid(unique_spatial_idx)
        H, W = int(grid_shape[0]), int(grid_shape[1])

        fill = len(unique_spatial_idx) / (H * W)
        sparse = fill < _SPARSE_FILL_THRESHOLD
        if sparse:
            print(f"Sparse output mode (fill={fill:.4f})")

        all_vars = [key[1] for key in td_keys.get("prognostic_keys", [])]
        all_vars += [key[1] for key in td_keys.get("diagnostic_keys", [])]

        is_rollout = prediction_mode in ("RolloutMSE", "RolloutCRPS")
        t_out = inference_horizon + 1 if is_rollout else inference_horizon

        unique_time_idx = np.sort(np.unique(time_idx))
        n_time = len(unique_time_idx)
        init_times = TIME_INDEX_ORIGIN + pd.to_timedelta(unique_time_idx.astype(np.int64), unit="h")
        lead_times = pd.timedelta_range(start="0h", periods=t_out, freq="h")

        if sparse:
            pixel_lats = lats[local_rows]
            pixel_lons = lons[local_cols]
            spatial_shape = (len(unique_spatial_idx),)
        else:
            pixel_lats, pixel_lons = lats, lons
            spatial_shape = (H, W)

        Forecast._create_zarr_store(
            path, all_vars, n_time, t_out, spatial_shape, init_times, lead_times, pixel_lats, pixel_lons
        )

    @staticmethod
    def _create_zarr_store(
        path, pred_vars, n_time, t_out, spatial_shape, init_times, lead_times, pixel_lats, pixel_lons
    ):
        """Create empty zarr store with coordinates and NaN-filled data arrays.

        spatial_shape=(H, W): dense grid output with lat/lon as 1D coordinate arrays.
        spatial_shape=(N_pixel,): sparse output with per-pixel lat/lon stored on 'pixel' dim.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sparse = len(spatial_shape) == 1

        if sparse:
            n_pixel = spatial_shape[0]
            coord_ds = xr.Dataset(
                coords={
                    "init_time": init_times,
                    "lead_time": lead_times,
                    "lat": xr.DataArray(pixel_lats, dims=["pixel"]),
                    "lon": xr.DataArray(pixel_lons, dims=["pixel"]),
                }
            )
            enc = {
                "init_time": {"chunks": (n_time,)},
                "lead_time": {"chunks": (t_out,)},
                "lat": {"chunks": (n_pixel,)},
                "lon": {"chunks": (n_pixel,)},
            }
            coord_ds.to_zarr(path, encoding=enc, mode="w")

            store = zarr.open_group(str(path), mode="r+")
            chunk_shape = (1, t_out, n_pixel)
            dim_names = ["init_time", "lead_time", "pixel"]
            array_shape = (n_time, t_out, n_pixel)
        else:
            H, W = spatial_shape
            chunk_h, chunk_w = min(H, 256), min(W, 256)
            coord_ds = xr.Dataset(
                coords={
                    "init_time": init_times,
                    "lead_time": lead_times,
                    "lat": pixel_lats,
                    "lon": pixel_lons,
                }
            )
            enc = {c: {"chunks": (int(coord_ds[c].shape[0]),)} for c in coord_ds.coords}
            coord_ds.to_zarr(path, encoding=enc, mode="w")

            store = zarr.open_group(str(path), mode="r+")
            chunk_shape = (1, t_out, chunk_h, chunk_w)
            dim_names = ["init_time", "lead_time", "lat", "lon"]
            array_shape = (n_time, t_out, H, W)

        for var in pred_vars:
            store.require_array(
                f"{var}_pred",
                shape=array_shape,
                dtype="float32",
                chunks=chunk_shape,
                fill_value=float("nan"),
                dimension_names=dim_names,
            )
            store.require_array(
                f"{var}_obs",
                shape=array_shape,
                dtype="float32",
                chunks=chunk_shape,
                fill_value=float("nan"),
                dimension_names=dim_names,
            )
        zarr.consolidate_metadata(str(path))
        print(f"Created zarr store at {path} ({n_time} init times, {len(pred_vars)} variable(s))")

    @staticmethod
    def _fill_zarr_store(
        path,
        preds_flat,
        obs_flat,
        n_spatial,
        n_time,
        time_horizon,
        local_rows,
        local_cols,
        grid_shape,
        time_offset,
        prepend_nan,
        sparse=False,
    ):
        """Write n_time steps of predictions into existing zarr store starting at time_offset."""
        lead_offset = 1 if prepend_nan else 0

        store = zarr.open_group(str(path), mode="r+")
        first_var = next(iter(preds_flat))
        t_out = store[f"{first_var}_pred"].shape[1]
        zarr_arrays = {var: (store[f"{var}_pred"], store[f"{var}_obs"]) for var in preds_flat}

        p_by_pixel = {var: preds_flat[var].reshape(n_spatial, n_time, time_horizon) for var in preds_flat}
        o_by_pixel = {var: obs_flat[var].reshape(n_spatial, n_time, time_horizon) for var in obs_flat}

        if sparse:
            # Shape: (n_time, t_out, N_pixel) — write per-pixel data directly, no scatter.
            buf_p = np.full((1, t_out, n_spatial), np.nan, dtype=np.float32)
            buf_o = np.full((1, t_out, n_spatial), np.nan, dtype=np.float32)
            for ti in range(n_time):
                abs_ti = time_offset + ti
                for var, (pa, oa) in zarr_arrays.items():
                    buf_p[:] = np.nan
                    # p_by_pixel[:, ti, :] is (N_pixel, T); transpose to (T, N_pixel) for lead/pixel dims
                    buf_p[0, lead_offset:, :] = p_by_pixel[var][:, ti, :].T
                    pa[abs_ti : abs_ti + 1] = buf_p
                    buf_o[:] = np.nan
                    buf_o[0, lead_offset:, :] = o_by_pixel[var][:, ti, :].T
                    oa[abs_ti : abs_ti + 1] = buf_o
        else:
            H, W = grid_shape
            dense_p = np.full((1, t_out, H, W), np.nan, dtype=np.float32)
            dense_o = np.full((1, t_out, H, W), np.nan, dtype=np.float32)
            for ti in range(n_time):
                abs_ti = time_offset + ti
                for var, (pa, oa) in zarr_arrays.items():
                    dense_p[:] = np.nan
                    dense_p[0, lead_offset:, local_rows, local_cols] = p_by_pixel[var][:, ti, :]
                    pa[abs_ti : abs_ti + 1] = dense_p
                    dense_o[:] = np.nan
                    dense_o[0, lead_offset:, local_rows, local_cols] = o_by_pixel[var][:, ti, :]
                    oa[abs_ti : abs_ti + 1] = dense_o

    def _remap(self, preds_flat, obs_flat, n_spatial, n_time, time_horizon):
        predictions = {}
        observations = {}
        for var in preds_flat:
            if self.sparse:
                p, o = self._remap_to_sparse(preds_flat[var], obs_flat[var], n_spatial, n_time, time_horizon)
            else:
                p, o = self._remap_to_grid(
                    preds_flat[var],
                    obs_flat[var],
                    n_spatial,
                    n_time,
                    time_horizon,
                    self.local_rows,
                    self.local_cols,
                    self.grid_shape,
                )
            predictions[var] = p
            observations[var] = o
        return predictions, observations

    def _prepend_nan_lead(self, predictions, observations, n_spatial, n_time, time_horizon):
        """Prepend NaN at lead_time=0 so rollout output aligns with diagnostic shape."""
        if self.sparse:
            nan_slice = np.full((n_spatial, n_time, 1), np.nan, dtype=np.float32)
            axis = 2
        else:
            nan_slice = np.full((n_time, 1, *self.grid_shape), np.nan, dtype=np.float32)
            axis = 1
        for var in predictions:
            predictions[var] = np.concatenate([nan_slice, predictions[var]], axis=axis)
            observations[var] = np.concatenate([nan_slice.copy(), observations[var]], axis=axis)
        return predictions, observations, time_horizon + 1

    def _pixel_coords(self):
        """Per-pixel lat/lon for sparse output or bbox lat/lon arrays for dense."""
        if self.sparse:
            return self.lats[self.local_rows], self.lons[self.local_cols]
        return self.lats, self.lons

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
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model.model(batch)

                for diag_key, diag_out_key in zip(diag_keys, diag_out_keys):
                    var = diag_key[1]
                    pred_batch = out[diag_out_key].float().cpu().numpy().squeeze((-3, -2, -1))
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

        lookback = kwargs.get("lookback", 0)
        context_vars = kwargs.get("context_vars", [])

        all_vars = [k[1] for k in prog_keys] + [k[1] for k in diag_keys]

        all_preds = {var: [] for var in all_vars}
        all_obs = {var: [] for var in all_vars}

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting (rollout)"):
                batch = batch.to(device)

                # Initial prognostic state is at batch index = lookback (t0)
                prog = {key: batch[key][:, lookback, ...] for key in prog_keys}

                # Seed the rolling context history from the batch lookback window.
                # history[var] is a list of lookback tensors (oldest first), each [B, H, W, C].
                # Batch indices 1..lookback correspond to times t0-lookback+1 .. t0.
                history = {}
                if lookback > 0:
                    for var in context_vars:
                        src_key = ("seviri", var)
                        history[var] = [batch[src_key][:, i, ...] for i in range(1, lookback + 1)]

                batch_preds = {var: [] for var in all_vars}
                batch_obs = {var: [] for var in all_vars}

                for t in range(1, kwargs["horizon"] + 1):
                    tstep_td = TensorDict({}, batch_size=[batch.batch_size[0]])
                    for key in td_keys["in_keys"]:
                        if key[1].endswith("_ctx"):
                            continue  # context keys injected below
                        mod_lb = lookback if key[0] == "seviri" else 0
                        tstep_td[key] = batch[key][:, mod_lb + t, ...]
                    for key in prog_keys:
                        tstep_td[key] = prog[key]

                    # Inject rolling context with Option B: values + validity mask.
                    # At t=1 the history is all teacher-forced; beyond that it mixes
                    # predictions as the window slides past the observed lookback.
                    if lookback > 0:
                        for var in context_vars:
                            ctx_key = ("seviri", f"{var}_ctx")
                            # [B, lookback, H, W, C] -> [B, H, W, 2*lookback*C]
                            ctx = torch.stack(history[var], dim=1)
                            B, lb, H, W, C = ctx.shape
                            valid_mask = torch.isfinite(ctx).all(dim=-1, keepdim=True)
                            ctx_vals = torch.nan_to_num(ctx, nan=0.0)
                            ctx_hw = ctx_vals.permute(0, 2, 3, 1, 4).reshape(B, H, W, lb * C)
                            valid_hw = valid_mask.float().permute(0, 2, 3, 1, 4).reshape(B, H, W, lb * C)
                            tstep_td[ctx_key] = torch.cat([ctx_hw, valid_hw], dim=-1)

                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        out = model.model(tstep_td)

                    for key, out_key in zip(prog_keys, prog_out_keys):
                        var = key[1]
                        prog[key] = prog[key] + out[out_key].float()
                        batch_preds[var].append(prog[key].squeeze((-3, -2, -1)))
                        batch_obs[var].append(batch[key][:, lookback + t, ...].squeeze((-3, -2, -1)))

                        # Advance history: drop oldest, append current prediction
                        if lookback > 0 and var in history:
                            history[var].pop(0)
                            history[var].append(prog[key])

                    for key, out_key in zip(diag_keys, diag_out_keys):
                        var = key[1]
                        batch_preds[var].append(out[out_key].float().squeeze((-3, -2, -1)))
                        batch_obs[var].append(batch[key][:, lookback + t, ...].squeeze((-3, -2, -1)))

                for var in all_vars:
                    p = torch.stack(batch_preds[var], dim=1).cpu().numpy()
                    o = torch.stack(batch_obs[var], dim=1).cpu().numpy()
                    if normaliser:
                        p = normaliser.denormalise(p, var)
                        o = normaliser.denormalise(o, var)
                    all_preds[var].append(p)
                    all_obs[var].append(o)

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
            # Require both obs and pred to be finite at ALL lead times.
            # Checking obs alone misses rollout pixels where a NaN init state
            # (cloud at t=0) propagates through the autoregressive loop,
            # producing NaN predictions even when obs clears later.
            all_valid = np.all(np.isfinite(o), axis=1, keepdims=True) & np.all(np.isfinite(p), axis=1, keepdims=True)
            preds_masked[var] = np.where(all_valid, p, np.nan)
            obs_masked[var] = np.where(all_valid, o, np.nan)
        return preds_masked, obs_masked

    @staticmethod
    def _remap_to_sparse(all_preds, all_obs, n_spatial, n_time, t):
        """Reshape flat predictions to (N_pixel, N_time, T) — no grid allocation."""
        preds = all_preds.reshape(n_spatial, n_time, t)
        obs = all_obs.reshape(n_spatial, n_time, t)
        return preds.astype(np.float32), obs.astype(np.float32)

    @staticmethod
    def _remap_to_grid(all_preds, all_obs, n_spatial, n_time, t, local_rows, local_cols, grid_shape):
        preds = all_preds.reshape(n_spatial, n_time, t)
        obs = all_obs.reshape(n_spatial, n_time, t)
        spatial_preds = np.full((n_time, t, *grid_shape), np.nan, dtype=np.float32)
        spatial_obs = np.full((n_time, t, *grid_shape), np.nan, dtype=np.float32)
        spatial_preds[:, :, local_rows, local_cols] = preds.transpose(1, 2, 0)
        spatial_obs[:, :, local_rows, local_cols] = obs.transpose(1, 2, 0)
        return spatial_preds, spatial_obs
