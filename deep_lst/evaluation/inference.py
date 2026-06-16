from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from tensordict import TensorDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_lst.evaluation.spatial import lookup_pixel_coords
from deep_lst.utils import ExperimentPaths, ZScoringNormalizer
from deep_lst.utils.config import get_context_var_pairs

PATHS = ExperimentPaths()

# Origin of the integer time indices used by TemporalFilter / EOZarr.
# Index 0 = 2016-01-01 00:00 UTC. Must match the start_date passed to TemporalFilter.
TIME_INDEX_ORIGIN = pd.Timestamp("2016-01-01")

# Pixels below this fill fraction of the bounding-box grid use sparse output.
# Avoids allocating a mostly-NaN (H x W) grid for scattered training-site pixels.
_SPARSE_FILL_THRESHOLD = 0.01


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
        seviri_store_path = PATHS.sampled_store_path("seviri", self.config["spatial"])
        self.lats, self.lons, self.local_rows, self.local_cols, self.grid_shape = lookup_pixel_coords(
            seviri_store_path, self.unique_spatial_idx
        )

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
            "RolloutMSE": self._predict_rollout,
            "RolloutSmoothL1": self._predict_rollout,
            "RolloutCRPS": self._predict_rollout,
            "RolloutIncrementLoss": self._predict_rollout,
        }
        predict_fn = predict_dispatch.get(self.prediction_mode)
        if predict_fn is None:
            raise ValueError(f"Unknown prediction_mode: {self.prediction_mode}. Known: {list(predict_dispatch)}")
        is_rollout = self.prediction_mode in ("RolloutMSE", "RolloutSmoothL1", "RolloutCRPS", "RolloutIncrementLoss")

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
        init_times = TIME_INDEX_ORIGIN + pd.to_timedelta(unique_time_idx.astype(np.int64), unit="h")
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
            if self.apply_consistent_mask and is_rollout:
                preds_b, obs_b = self._apply_consistent_mask(preds_b, obs_b)
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
        diag_out_keys = td_keys["diagnostic_out_keys"]

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting (diagnostic)"):
                batch = batch.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model.model(batch)

                preds_b = {}
                obs_b = {}
                for diag_key, diag_out_key in zip(diag_keys, diag_out_keys):
                    var = diag_key[1]
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
        """Autoregressive rollout generator: yields per-batch (preds, obs) dicts of shape (B, horizon)."""
        prog_keys = td_keys["prognostic_keys"]
        prog_out_keys = td_keys["prognostic_out_keys"]
        diag_keys = td_keys.get("diagnostic_keys", [])
        diag_out_keys = td_keys.get("diagnostic_out_keys", [])

        lookback = kwargs.get("lookback", 0)
        inference_horizon = kwargs.get(
            "horizon", 0
        )  # currently reads from the modalty overwritten horizon, not from infer.yaml!
        context_pairs = get_context_var_pairs(model.config["train_dataset"])

        all_vars = [k[1] for k in prog_keys] + [k[1] for k in diag_keys]

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting (rollout)"):
                batch = batch.to(device)

                # Initial prognostic state is at batch index = lookback (t0)
                prog = {key: batch[key][:, lookback, ...] for key in prog_keys}

                history = {}
                for modality, var in context_pairs:
                    history[(modality, var)] = [batch[(modality, var)][:, i, ...] for i in range(1, lookback + 1)]

                batch_preds = {var: [] for var in all_vars}
                batch_obs = {var: [] for var in all_vars}

                predict_anomaly = model.config["model"].get("predict_anomaly", False)
                clim_fwd = (
                    model.climatology.lookup_forward(batch["spatial_idx"], batch["time_idx"], inference_horizon)
                    if predict_anomaly
                    else None
                )

                for t in range(1, inference_horizon + 1):
                    tstep_td = TensorDict({}, batch_size=[batch.batch_size[0]])
                    for key in td_keys["in_keys"]:
                        if key[1].endswith("_ctx"):
                            continue  # context keys injected below
                        tstep_td[key] = batch[key][:, lookback + t, ...]
                    for key in prog_keys:
                        tstep_td[key] = prog[key]

                    if lookback > 0:
                        for modality, var in context_pairs:
                            ctx_key = (modality, f"{var}_ctx")
                            ctx = torch.stack(history[(modality, var)], dim=1)
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
                        model_increment = out[out_key].float()

                        if clim_fwd is not None and var == "LST":
                            pad = (1,) * (prog[key].ndim - 1)
                            clim_t = clim_fwd[:, t].view(-1, *pad)
                            clim_tprev = clim_fwd[:, t - 1].view(-1, *pad)
                            clim_increment = torch.nan_to_num(clim_t - clim_tprev, nan=0.0)
                        else:
                            clim_increment = 0

                        state_increment = model_increment + clim_increment
                        prog[key] = prog[key] + state_increment

                        batch_preds[var].append(prog[key].squeeze((-3, -2, -1)))
                        batch_obs[var].append(batch[key][:, lookback + t, ...].squeeze((-3, -2, -1)))

                    for key, out_key in zip(diag_keys, diag_out_keys):
                        var = key[1]
                        batch_preds[var].append(out[out_key].float().squeeze((-3, -2, -1)))
                        batch_obs[var].append(batch[key][:, lookback + t, ...].squeeze((-3, -2, -1)))

                    if lookback > 0:
                        for modality, var in context_pairs:
                            history_key = (modality, var)
                            history[history_key].pop(0)
                            if history_key in prog_keys:
                                history[history_key].append(prog[history_key])
                            else:
                                history[history_key].append(batch[history_key][:, lookback + t, ...])

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

    @staticmethod
    def _apply_consistent_mask(
        preds: dict[str, np.ndarray], obs: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        preds_masked = {}
        obs_masked = {}
        for var in obs:
            o = obs[var]
            p = preds[var]

            all_valid = np.all(np.isfinite(o), axis=1, keepdims=True) & np.all(np.isfinite(p), axis=1, keepdims=True)
            preds_masked[var] = np.where(all_valid, p, np.nan)
            obs_masked[var] = np.where(all_valid, o, np.nan)
        return preds_masked, obs_masked
