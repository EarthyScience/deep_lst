from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from eodata.grids import build_seviri_grid

from deep_lst.evaluation.metrics import compute_grouped_scores, compute_scores, to_init_time, to_valid_time
from deep_lst.evaluation.spatial import build_local_grid
from deep_lst.utils import ExperimentPaths

_BASELINE_REGISTRY: dict[str, tuple[callable, str]] = {
    "era5": (lambda p, n: p.baseline_path(n), "skt"),
    "clim_hourly": (lambda p, n: p.climatology_path(n, "hourly"), "LST"),
    "clim_smooth": (lambda p, n: p.climatology_path(n, "smoothed_hourly"), "LST"),
    "persistence": (lambda p, n: p.persistence_path(n), "LST"),
}


class Evaluator(ABC):
    """Shared: load, filter, baseline loading, saving."""

    def __init__(
        self,
        experiment,
        version,
        eval_name,
        var,
        metrics,
        baseline,
        temporal,
        aggregation,
        consistent_mask,
        overwrite,
        reduction_order: tuple = ("space", "time"),
    ):
        self.experiment = experiment
        self.version = version
        self.eval_name = eval_name
        self.var = var
        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.baseline = baseline
        self.overwrite = overwrite
        self.temporal = temporal
        self.aggregation = aggregation
        self.reduction_order = reduction_order
        self.consistent_mask = consistent_mask
        self.paths = ExperimentPaths()

    def run(self, forecast_ds=None):
        if forecast_ds is not None:
            self.forecast_ds = forecast_ds
        else:
            self.forecast_ds = self._load_forecast()
            self.forecast_ds = self._filter(self.forecast_ds)

        self.forecast_ds = self._attach_class_coords(self.forecast_ds)

        self.pred = self.forecast_ds[f"{self.var}_pred"]
        self.obs = self.forecast_ds[f"{self.var}_obs"]

        self.baseline_ds = self._load_baseline()  # raw, unaligned

        if self.baseline_ds is not None:
            self.baseline_ds = self._subset_pixels(self.baseline_ds, self.pred.isel(lead_time=0))

        self._compute()  # subclass sets self.metrics

    @abstractmethod
    def _compute(self):
        pass

    # Shared helpers
    def _load_forecast(self) -> xr.Dataset:
        path = self.paths.forecast_path(self.experiment, self.version, self.eval_name)
        if not path.exists():
            raise FileNotFoundError(f"Forecast not found: {path}")
        return xr.open_zarr(path, chunks={"init_time": 500})

    def _filter(self, ds: xr.Dataset) -> xr.Dataset:
        filtered = ds
        if self.temporal is not None:
            origin = pd.Timestamp("2016-01-01")
            time_idx = np.concatenate(
                [np.load(self.paths.temporal_indices_path(p))["time_indices"].astype(np.int64) for p in self.temporal]
            )
            keep_times = origin + pd.to_timedelta(np.unique(time_idx), unit="h")
            filtered = ds.sel(init_time=ds["init_time"].isin(keep_times))

        if self.consistent_mask:
            pred_raw = filtered[f"{self.var}_pred"]
            obs_raw = filtered[f"{self.var}_obs"]
            all_finite = pred_raw.notnull().all("lead_time") & obs_raw.notnull().all("lead_time")
            filtered = filtered.assign(
                {
                    f"{self.var}_pred": pred_raw.where(all_finite),
                    f"{self.var}_obs": obs_raw.where(all_finite),
                }
            )

        return filtered

    def _attach_class_coords(self, ds: xr.Dataset) -> xr.Dataset:
        path = self.paths.eval_indices_path("seviri", self.eval_name)
        if not path.exists():
            raise FileNotFoundError(f"Eval indices not found for {self.eval_name}: {path}")
        indices_ds = np.load(path)
        class_keys = ["kg_class", "kg_major"]

        coords = {}
        if "pixel" in ds.dims:
            seviri_grid = build_seviri_grid()
            points = shapely.points(ds["lon"].values, ds["lat"].values)
            forecast_src = np.asarray(seviri_grid.centers_tree.nearest(geometry=points))
            npz_src = indices_ds["src_indices"]
            npz_pos = {int(s): i for i, s in enumerate(npz_src)}
            lookup = np.array([npz_pos[int(s)] for s in forecast_src], dtype=np.int64)
            for key in class_keys:
                if key in indices_ds.files:
                    coords[key] = (("pixel",), indices_ds[key][lookup])
        else:
            src_indices = indices_ds["src_indices"]
            _, _, local_rows, local_cols, grid_shape = build_local_grid(src_indices)
            if grid_shape != (ds.sizes["lat"], ds.sizes["lon"]):
                raise ValueError(
                    f"Grid shape mismatch: expected {grid_shape}, got {(ds.sizes['lat'], ds.sizes['lon'])}"
                )
            for key in class_keys:
                if key in indices_ds.files:
                    class_values = indices_ds[key]
                    grid = np.full(grid_shape, -1, dtype=class_values.dtype)
                    grid[local_rows, local_cols] = class_values
                    coords[key] = (("lat", "lon"), grid)
        ds = ds.assign_coords(coords)
        return ds

    def _load_baseline(self) -> xr.DataArray | None:
        if self.baseline is None:
            return None
        if self.baseline not in _BASELINE_REGISTRY:
            raise ValueError(f"Unknown baseline: {self.baseline!r}. Available: {list(_BASELINE_REGISTRY)}")

        path_fn, var_name = _BASELINE_REGISTRY[self.baseline]
        path = path_fn(self.paths, self.eval_name)
        if not path.exists():
            raise FileNotFoundError(f"Baseline '{self.baseline}' not found: {path}")

        return xr.open_dataset(path)[var_name]

    def _align_baseline(self, valid_times: np.ndarray) -> xr.DataArray:
        return self.baseline_ds.reindex(valid_time=valid_times)

    def _subset_pixels(self, baseline_ds: xr.DataArray, forecast_ds: xr.DataArray) -> xr.DataArray:
        """Subset da's pixel dim to match ref's lat/lon coordinates.

        No-op when pixel counts already match or lat coords are absent.
        Used when a baseline covers all store pixels but the forecast uses a subset.
        """
        if (
            "pixel" not in baseline_ds.dims
            or "pixel" not in forecast_ds.dims
            or baseline_ds.sizes["pixel"] == forecast_ds.sizes["pixel"]
            or "lat" not in baseline_ds.coords
            or "lat" not in forecast_ds.coords
        ):
            return baseline_ds
        bl_lats = baseline_ds["lat"].values.astype(np.float32)
        bl_lons = baseline_ds["lon"].values.astype(np.float32)
        lookup = {(float(la), float(lo)): i for i, (la, lo) in enumerate(zip(bl_lats, bl_lons))}
        p_lats = forecast_ds["lat"].values.astype(np.float32)
        p_lons = forecast_ds["lon"].values.astype(np.float32)
        idx = [lookup[(float(la), float(lo))] for la, lo in zip(p_lats, p_lons)]
        return baseline_ds.isel(pixel=idx)

    def summarise(self, source: xr.Dataset, quantiles=(0.05, 0.50, 0.95)) -> xr.Dataset:
        keep_dim = next((d for d in source.dims if d in ("lead_time", "hour", "month")), None)
        first_metric = next(iter(source.data_vars.values()))
        spatial_dims = [d for d in first_metric.dims if d != keep_dim]
        if not spatial_dims:
            return source
        q_labels = [f"q{int(q * 100):02d}" for q in quantiles]
        return xr.Dataset(
            {
                f"{metric}_{ql}": source[metric].quantile(q, dim=spatial_dims).drop_vars("quantile")
                for metric in source.data_vars
                for q, ql in zip(quantiles, q_labels)
            }
        )


class EvaluatorDiagnostics(Evaluator):
    """Fixed lead time → valid_time space. Standard metrics, diurnal, spatial."""

    def __init__(
        self,
        experiment,
        version,
        eval_name,
        var,
        metrics,
        baseline,
        temporal,
        aggregation,
        groupby,
        consistent_mask,
        overwrite,
        lead_time,
        reduction_order: tuple = ("space", "time"),
    ):
        super().__init__(
            experiment,
            version,
            eval_name,
            var,
            metrics,
            baseline,
            temporal,
            aggregation,
            consistent_mask,
            overwrite,
            reduction_order=reduction_order,
        )
        self.lead_time = lead_time
        self.groupby = groupby

    def _compute(self):
        self.pred = to_valid_time(self.pred.sel(lead_time=self.lead_time), self.lead_time)
        self.obs = to_valid_time(self.obs.sel(lead_time=self.lead_time), self.lead_time)

        self.scores = compute_scores(self.pred, self.obs, self.metrics, self.aggregation, self.reduction_order)
        self.grouped = compute_grouped_scores(
            self.pred, self.obs, self.metrics, self.aggregation, self.groupby, self.reduction_order
        )

        if self.baseline_ds is not None:
            aligned = self._align_baseline(self.obs["valid_time"].values)
            aligned_masked = aligned.where(np.isfinite(self.pred))
            self.baseline_aligned = aligned_masked

            self.baseline_scores = compute_scores(
                aligned_masked, self.obs, self.metrics, self.aggregation, self.reduction_order
            )
            self.baseline_grouped = compute_grouped_scores(
                aligned_masked, self.obs, self.metrics, self.aggregation, self.groupby, self.reduction_order
            )
        else:
            self.baseline_aligned = None
            self.baseline_scores = None
            self.baseline_grouped = None


class EvaluatorVerification(Evaluator):
    """Full (init_time, lead_time) space. Error growth, skill scores, climate metrics."""

    _CLIMATE_BASELINES = ("clim_smooth", "clim_hourly")
    _ANOMALY_METRICS = ["activity", "acc"]

    def _compute(self):
        lead_times = self.forecast_ds["lead_time"].values

        self.scores = compute_scores(self.pred, self.obs, self.metrics, self.aggregation)

        baseline_slices = []
        if self.baseline_ds is not None:
            for lead_time in lead_times:
                pred_lt = self.pred.sel(lead_time=lead_time)
                baseline_slices.append(self._process_baseline(pred_lt, lead_time))

            lt_index = pd.Index(lead_times, name="lead_time")
            baseline_full = xr.concat(baseline_slices, dim=lt_index)
            self.baseline_scores = compute_scores(baseline_full, self.obs, self.metrics, self.aggregation)

            if self.baseline in self._CLIMATE_BASELINES:
                pred_anom = self.pred - baseline_full
                obs_anom = self.obs - baseline_full
                self.climate_scores = compute_scores(pred_anom, obs_anom, self._ANOMALY_METRICS, self.aggregation)

            self.skill_scores = self.scores / self.baseline_scores
        else:
            self.baseline_scores = None
            self.climate_scores = None
            self.skill_scores = None

    def _process_baseline(self, pred_lt, lead_time) -> xr.DataArray:
        lead_time_hour = int(pd.Timedelta(lead_time).total_seconds() // 3600)
        lt_str = f"{lead_time_hour}h"
        valid_times = pred_lt["init_time"].values + np.timedelta64(lead_time_hour, "h")
        aligned = self._align_baseline(valid_times)
        aligned = to_init_time(aligned, lt_str)
        return aligned.where(np.isfinite(pred_lt))


def run_evaluators_across_sites(
    ev_kwargs: dict,
    eval_names: list[str],
    baseline: str | None = None,
) -> dict[str, EvaluatorDiagnostics]:
    """Run EvaluatorDiagnostics for each eval region, skipping missing forecasts."""
    results = {}
    for eval_region in eval_names:
        try:
            evaluator = EvaluatorDiagnostics(**ev_kwargs, eval_name=eval_region, baseline=baseline)
            evaluator.run()
            results[eval_region] = evaluator
        except FileNotFoundError:
            print(f"  [skip] {eval_region}: forecast not found")
    return results
