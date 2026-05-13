from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.metrics import compute_grouped_scores, compute_scores, to_valid_time
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
        self, experiment, version, eval_name, var, metrics, baseline, temporal, aggregation, consistent_mask, overwrite
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
        self.consistent_mask = consistent_mask
        self.paths = ExperimentPaths()

    def run(self):
        self.forecast_ds = self._load_forecast()
        self.forecast_ds = self._filter(self.forecast_ds)
        self.baseline_ds = self._load_baseline()  # raw, unaligned
        self._compute()  # subclass sets self.metrics

    @abstractmethod
    def _compute(self):
        pass

    # Shared helpers
    def _load_forecast(self) -> xr.Dataset:
        path = self.paths.forecast_path(self.experiment, self.version, self.eval_name)
        if not path.exists():
            raise FileNotFoundError(f"Forecast not found: {path}")
        return xr.open_zarr(path, chunks={})

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

    def _save_metric(self, ds, path):
        pass

    def _to_valid_time(self, lead_time) -> tuple[xr.DataArray, xr.DataArray]:
        return to_valid_time(self.forecast_ds, self.var, lead_time)

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
    ):
        super().__init__(
            experiment, version, eval_name, var, metrics, baseline, temporal, aggregation, consistent_mask, overwrite
        )
        self.lead_time = lead_time
        self.groupby = groupby

    def _compute(self):
        self.pred, self.obs = self._to_valid_time(self.lead_time)
        self._compute_scores_forecast()
        self._compute_scores_baseline()

    def _compute_scores_forecast(self):
        self.scores = compute_scores(self.pred, self.obs, self.metrics, self.aggregation)
        self.grouped = compute_grouped_scores(self.pred, self.obs, self.metrics, self.aggregation, self.groupby)

    def _compute_scores_baseline(self):
        if self.baseline_ds is None:
            self.baseline_scores = self.baseline_grouped = None
            return

        bl = self._align_baseline(self.obs["valid_time"].values)
        bl = self._subset_pixels(bl, self.pred)
        bl = bl.where(np.isfinite(self.pred))
        self.baseline_aligned = bl

        self.baseline_scores = compute_scores(bl, self.obs, self.metrics, self.aggregation)

        self.baseline_grouped = compute_grouped_scores(bl, self.obs, self.metrics, self.aggregation, self.groupby)

    def summarise(self, source: xr.Dataset, quantiles=(0.05, 0.50, 0.95)) -> xr.Dataset:
        group_dim = next(d for d in source.dims if d in ("hour", "month"))
        spatial_dims = [d for d in source.dims if d != group_dim]
        if not spatial_dims:
            return source
        q_labels = [f"q{int(q * 100):02d}" for q in quantiles]
        return xr.Dataset(
            {
                f"{m}_{ql}": source[m].quantile(q, dim=spatial_dims).drop_vars("quantile")
                for m in source.data_vars
                if m != "n_valid"
                for q, ql in zip(quantiles, q_labels)
            }
        )


class EvaluatorVerification(Evaluator):
    """Full (init_time, lead_time) space. Error growth, skill scores."""

    def _compute(self):
        self._compute_scores_forecast()
        self._compute_scores_baseline()
        self._compute_skill_scores()

    def _align_baseline(self, valid_times) -> xr.DataArray:
        aligned = super()._align_baseline(valid_times)
        return aligned.assign_coords(valid_time=self.forecast_ds["init_time"].values).rename(
            {"valid_time": "init_time"}
        )

    def _compute_scores_forecast(self):
        lead_times = self.forecast_ds["lead_time"].values
        slices = []
        for lead_time in lead_times:
            pred_lt = self.forecast_ds[f"{self.var}_pred"].sel(lead_time=lead_time)
            obs_lt = self.forecast_ds[f"{self.var}_obs"].sel(lead_time=lead_time)
            slices.append(compute_scores(pred_lt, obs_lt, self.metrics, aggregation=self.aggregation))
        self.scores = xr.concat(slices, dim=pd.Index(lead_times, name="lead_time"))

    def _compute_scores_baseline(self):
        if self.baseline_ds is None:
            self.baseline_scores = None
            return

        lead_times = self.forecast_ds["lead_time"].values
        slices = []
        for lead_time in lead_times:
            pred_lt = self.forecast_ds[f"{self.var}_pred"].sel(lead_time=lead_time)
            obs_lt = self.forecast_ds[f"{self.var}_obs"].sel(lead_time=lead_time)

            lead_time_hour = int(pd.Timedelta(lead_time).total_seconds() // 3600)
            _, obs_valid_time = self._to_valid_time(f"{lead_time_hour}h")
            aligned = self._align_baseline(obs_valid_time["valid_time"].values)
            aligned = self._subset_pixels(aligned, pred_lt)

            # mask to valid times for fair comparison.
            masked = aligned.where(np.isfinite(pred_lt))

            slices.append(compute_scores(masked, obs_lt, self.metrics, aggregation=self.aggregation))
        self.baseline_scores = xr.concat(slices, dim=pd.Index(lead_times, name="lead_time"))

    def _compute_skill_scores(self):
        if self.baseline_scores is not None:
            self.skill_scores = self.scores / self.baseline_scores

    def summarise(
        self,
        source: xr.Dataset,  # self.scores, self.baseline_scores[label], or self.skill_scores[label]
        quantiles: tuple[float, ...] = (0.05, 0.50, 0.95),
        period=None,
    ) -> xr.Dataset:
        ds = source if period is None else source.sel(init_time=period)
        q_labels = [f"q{int(q * 100):02d}" for q in quantiles]
        return xr.Dataset(
            {
                f"{m}_{ql}": ds[m].quantile(q, dim="init_time").drop_vars("quantile")
                for m in ds.data_vars
                if m != "n_valid"
                for q, ql in zip(quantiles, q_labels)
            },
            coords={"lead_time": self.forecast_ds["lead_time"].values},
        )
