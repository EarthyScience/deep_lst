from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from eodata.grids import T1H
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_lst.utils import ExperimentPaths, StatsZarrNormalizer
from deep_lst.utils.io import stats_spatial
from deep_lst.utils.spatial import pixel_grid

PATHS = ExperimentPaths()

# Pixels below this fill fraction of the bounding-box grid use sparse output.
# Avoids allocating a mostly-NaN (H x W) grid for scattered training-site pixels.
_SPARSE_FILL_THRESHOLD = 1.1


class Forecast:
    """Runs model inference and writes predictions to a forecast.zarr store."""

    def __init__(
        self,
        model,
        device: torch.device,
        loader: DataLoader,
        td_keys: dict,
        config: dict,
        spatial_indices: np.ndarray,
        pixel_lonlat: np.ndarray,
        non_land: np.ndarray,
        prediction_mode: str = "diagnostic",
        apply_lw_mask: bool = True,
    ):
        self.model = model
        self.device = device
        self.loader = loader
        self.td_keys = td_keys
        self.config = config
        self.prediction_mode = prediction_mode
        self.apply_lw_mask = apply_lw_mask

        self.normaliser = StatsZarrNormalizer(PATHS.stats_zarr_path("seviri", stats_spatial(self.config)))

        self.prognostic_vars = [key[1] for key in td_keys.get("prognostic_keys", [])]
        self.diagnostic_vars = [key[1] for key in td_keys.get("diagnostic_keys", [])]

        # Per-pixel coords come from the preloaded cell_lonlat (no sampled store); the dense-grid
        # geometry is reconstructed by pure math from those coords.
        self.unique_spatial_idx = np.unique(spatial_indices)
        pixel_lons = pixel_lonlat[:, 0]
        pixel_lats = pixel_lonlat[:, 1]
        self.lats, self.lons, self.local_rows, self.local_cols, self.grid_shape = pixel_grid(pixel_lons, pixel_lats)

        if self.apply_lw_mask:
            self.non_land = non_land

        fill = len(self.unique_spatial_idx) / (self.grid_shape[0] * self.grid_shape[1])
        self.sparse = fill < _SPARSE_FILL_THRESHOLD
        if self.sparse:
            print(f"Sparse output mode (fill={fill:.4f} < {_SPARSE_FILL_THRESHOLD})")

    def predict_and_save(self, path: Path, time_idx: np.ndarray) -> None:
        """Run inference, streaming predictions into the zarr store one init_time at a time.

        Relies on the loader iterating time-major (all pixels of init_time 0, then 1, ...)
        so each flush writes whole init_time chunks. Memory stays bounded at ~one batch
        plus a sub-init_time carry buffer, independent of the dataset size.
        """
        predict_dispatch = {
            "DiagnosticMSE": self._predict_diagnostic,
            "DiagnosticCRPS": self._predict_diagnostic,
            "RolloutLoss": self._predict_rollout,
        }
        predict_fn = predict_dispatch.get(self.prediction_mode)
        if predict_fn is None:
            raise ValueError(f"Unknown prediction_mode: {self.prediction_mode}. Known: {list(predict_dispatch)}")
        is_rollout = self.prediction_mode == "RolloutLoss"

        gen = predict_fn(
            self.model,
            self.device,
            self.loader,
            self.td_keys,
            normaliser=self.normaliser,
            **self.config["seviri"],
        )

        # Peek the first batch to size the lead axis from actual output shapes.
        try:
            first_batch = next(gen)
        except StopIteration:
            raise ValueError("Loader produced no batches; nothing to save.") from None

        first_var = next(iter(first_batch[0]))
        time_horizon = first_batch[0][first_var].shape[1]
        t_out = time_horizon + 1 if is_rollout else time_horizon
        lead_offset = 1 if is_rollout else 0

        n_spatial = len(self.unique_spatial_idx)
        unique_time_idx = np.unique(time_idx)
        n_time = len(unique_time_idx)
        # index_to_startdate returns tz-aware UTC; drop the tz so the coords are naive datetime64
        # (xarray's zarr encoder rejects tz-aware, and the evaluator compares against naive dates).
        init_times = pd.to_datetime([T1H.index_to_startdate(int(t)) for t in unique_time_idx]).tz_localize(None)
        lead_times = pd.timedelta_range(start="0h", periods=t_out, freq="h")
        pixel_lats, pixel_lons = self._pixel_coords()

        if self.sparse:
            spatial_shape = (n_spatial,)
        else:
            H, W = self.grid_shape
            spatial_shape = (int(H), int(W))

        all_vars = list(first_batch[0].keys())
        self._create_zarr_store(
            path, all_vars, n_time, t_out, spatial_shape, init_times, lead_times, pixel_lats, pixel_lons
        )
        store = zarr.open_group(str(path), mode="r+")
        zarr_arrays = {var: (store[f"{var}_pred"], store[f"{var}_obs"]) for var in all_vars}

        carry_p = {var: [] for var in all_vars}
        carry_o = {var: [] for var in all_vars}
        carry_count = 0
        ti_cursor = 0

        for preds_b, obs_b in chain([first_batch], gen):
            for var in all_vars:
                carry_p[var].append(preds_b[var])
                carry_o[var].append(obs_b[var])
            carry_count += preds_b[first_var].shape[0]

            n_complete = carry_count // n_spatial
            if n_complete == 0:
                continue
            cut = n_complete * n_spatial
            block_p = {var: np.concatenate(carry_p[var], axis=0) for var in all_vars}
            block_o = {var: np.concatenate(carry_o[var], axis=0) for var in all_vars}
            if self.apply_lw_mask:
                # The first `cut` rows are n_complete whole init-times of n_spatial pixels each,
                # in unique_spatial_idx order — so tiling non_land to length cut aligns row-for-row.
                non_land_rows = np.tile(self.non_land, n_complete)
                for var in all_vars:
                    block_p[var][:cut][non_land_rows] = np.nan
                    block_o[var][:cut][non_land_rows] = np.nan
            self._write_init_times(
                zarr_arrays,
                {var: a[:cut] for var, a in block_p.items()},
                {var: a[:cut] for var, a in block_o.items()},
                ti_cursor,
                n_spatial,
                lead_offset,
                self.local_rows,
                self.local_cols,
                self.grid_shape,
                self.sparse,
            )
            ti_cursor += n_complete
            carry_p = {var: [a[cut:]] for var, a in block_p.items()}
            carry_o = {var: [a[cut:]] for var, a in block_o.items()}
            carry_count -= cut

        if carry_count != 0 or ti_cursor != n_time:
            raise RuntimeError(
                f"Stream ended at init_time {ti_cursor}/{n_time} with {carry_count} leftover samples — "
                "dataset size does not match n_spatial × n_time; forecast.zarr is incomplete."
            )

        zarr.consolidate_metadata(str(path))
        print(f"Done. Saved to {path}")

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
    def rechunk_store(path, target_bytes: int = 256 * 1024**2) -> None:
        """Rewrite a finished forecast store with init_time chunked into large blocks."""
        import shutil

        path = Path(path)
        ds = xr.open_zarr(path)
        sample = ds[next(iter(ds.data_vars))]
        other = int(np.prod([sample.sizes[d] for d in sample.dims if d != "init_time"]))
        init_chunk = int(max(1, min(sample.sizes["init_time"], target_bytes // (other * sample.dtype.itemsize))))

        enc = {}
        for v in ds.data_vars:
            da = ds[v]
            chunks = tuple(init_chunk if d == "init_time" else da.sizes[d] for d in da.dims)
            ev = {"chunks": chunks}
            for key in ("compressors", "filters", "serializer"):
                if key in da.encoding:
                    ev[key] = da.encoding[key]
            enc[v] = ev
        for c in ds.coords:
            enc[c] = {"chunks": tuple(ds[c].sizes[d] for d in ds[c].dims)}

        ds = ds.chunk({"init_time": init_chunk})
        tmp = path.parent / (path.name + ".rechunk.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        ds.to_zarr(tmp, encoding=enc, mode="w")
        ds.close()
        zarr.consolidate_metadata(str(tmp))
        shutil.rmtree(path)
        tmp.rename(path)
        print(
            f"Rechunked {path.name}: init_time chunk={init_chunk} (~{other * init_chunk * sample.dtype.itemsize / 1e6:.0f} MB/chunk)"
        )

    @staticmethod
    def _write_init_times(
        zarr_arrays,
        preds,
        obs,
        ti_start,
        n_spatial,
        lead_offset,
        local_rows,
        local_cols,
        grid_shape,
        sparse,
    ):
        """Scatter (k * n_spatial, t) prediction blocks into k whole init_time slots.

        Rows must be time-major: rows [i*n_spatial : (i+1)*n_spatial] hold all pixels of
        init_time ti_start + i, in the pixel order of local_rows/local_cols.
        """
        first_var = next(iter(preds))
        k = preds[first_var].shape[0] // n_spatial
        t_out = zarr_arrays[first_var][0].shape[1]

        if sparse:
            buf_p = np.full((1, t_out, n_spatial), np.nan, dtype=np.float32)
            buf_o = np.full((1, t_out, n_spatial), np.nan, dtype=np.float32)
            for i in range(k):
                rows = slice(i * n_spatial, (i + 1) * n_spatial)
                ti = ti_start + i
                for var, (pa, oa) in zarr_arrays.items():
                    buf_p[:] = np.nan
                    # preds[var][rows] is (n_spatial, t); transpose to (t, n_spatial) for lead/pixel dims
                    buf_p[0, lead_offset:, :] = preds[var][rows].T
                    pa[ti : ti + 1] = buf_p
                    buf_o[:] = np.nan
                    buf_o[0, lead_offset:, :] = obs[var][rows].T
                    oa[ti : ti + 1] = buf_o
        else:
            H, W = grid_shape
            dense_p = np.full((1, t_out, H, W), np.nan, dtype=np.float32)
            dense_o = np.full((1, t_out, H, W), np.nan, dtype=np.float32)
            for i in range(k):
                rows = slice(i * n_spatial, (i + 1) * n_spatial)
                ti = ti_start + i
                for var, (pa, oa) in zarr_arrays.items():
                    dense_p[:] = np.nan
                    dense_p[0, lead_offset:, local_rows, local_cols] = preds[var][rows]
                    pa[ti : ti + 1] = dense_p
                    dense_o[:] = np.nan
                    dense_o[0, lead_offset:, local_rows, local_cols] = obs[var][rows]
                    oa[ti : ti + 1] = dense_o

    def _pixel_coords(self):
        """Per-pixel lat/lon for sparse output or bbox lat/lon arrays for dense."""
        if self.sparse:
            return self.lats[self.local_rows], self.lons[self.local_cols]
        return self.lats, self.lons

    @staticmethod
    def _predict_diagnostic(model, device, loader, td_keys, normaliser=None, **kwargs):
        """Single-step generator: yields per-batch (preds, obs) dicts, denormalised."""
        diag_keys = td_keys["diagnostic_keys"]

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting (diagnostic)"):
                batch = batch.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model.model(batch)

                preds_b = {}
                obs_b = {}
                # Pair by construction, not by zip: diagnostic_out_keys may carry extra
                # {var}_log_sigma_pred entries (sigma_diagnostics) that have no target key.
                for diag_key in diag_keys:
                    modality, var = diag_key
                    diag_out_key = (modality, f"{var}_pred")
                    pred_batch = out[diag_out_key].float().cpu().numpy().squeeze((-3, -2, -1))
                    obs_batch = batch[diag_key].cpu().numpy().squeeze((-3, -2, -1))

                    if normaliser:
                        pred_batch = normaliser.denormalise(pred_batch, var)
                        obs_batch = normaliser.denormalise(obs_batch, var)

                    preds_b[var] = pred_batch
                    obs_b[var] = obs_batch
                yield preds_b, obs_b

    @staticmethod
    def _predict_rollout(model, device, loader, td_keys, normaliser=None, **kwargs):
        """Autoregressive rollout generator over the packed tensors. Mirrors the training
        rollout loop: carry the predicted state back into the window. Yields per-batch
        (preds, obs) dicts of shape (B, horizon), denormalised, one entry per variable."""
        prog_keys = td_keys["prognostic_keys"]
        diag_keys = td_keys.get("diagnostic_keys", [])
        nP = len(prog_keys)
        prog_vars = [key[1] for key in prog_keys]
        diag_vars = [key[1] for key in diag_keys]
        all_vars = prog_vars + diag_vars

        lookback = kwargs.get("lookback", 0)
        horizon = kwargs.get("horizon", 0)  # reads the modality-overwritten horizon, not infer.yaml

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting (rollout)"):
                batch = batch.to(device)
                y_prog = batch["y_prog"]
                X_dyn = batch["X_dyn"]
                X_dyn_ctx = batch["X_dyn_ctx"]

                state_rollout = y_prog.clone()
                batch_preds = {var: [] for var in all_vars}
                batch_obs = {var: [] for var in all_vars}

                for t in range(1, horizon + 1):
                    state_window = state_rollout[:, t : lookback + t, ..., :nP]
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        out = model.model(state_window, X_dyn_ctx[:, t : lookback + t], X_dyn[:, lookback + t])

                    inc = out[..., :nP].float()
                    diag = out[..., nP:].float()
                    prior = state_rollout[:, lookback + t - 1, ..., :nP]
                    state_t = prior + inc

                    target = y_prog[:, lookback + t]
                    for c, var in enumerate(prog_vars):
                        batch_preds[var].append(state_t[..., c].squeeze((-2, -1)))
                        batch_obs[var].append(target[..., c].squeeze((-2, -1)))
                    for c, var in enumerate(diag_vars):
                        batch_preds[var].append(diag[..., c].squeeze((-2, -1)))
                        batch_obs[var].append(target[..., nP + c].squeeze((-2, -1)))

                    if t < horizon:
                        state_rollout[:, lookback + t, ..., :nP] = state_t

                preds_b = {}
                obs_b = {}
                for var in all_vars:
                    p = torch.stack(batch_preds[var], dim=1).cpu().numpy()
                    o = torch.stack(batch_obs[var], dim=1).cpu().numpy()
                    if normaliser:
                        p = normaliser.denormalise(p, var)
                        o = normaliser.denormalise(o, var)
                    preds_b[var] = p
                    obs_b[var] = o
                yield preds_b, obs_b
