import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.evaluator import (
    ANOMALY_METRICS,
    RAW_METRICS,
)
from deep_lst.evaluation.plots.cdf import day_night_summary, lst_cdf_plot, rmse_cdf_plot
from deep_lst.evaluation.plots.diurnal import diurnal_delta_comparison, diurnal_line, diurnal_mean_plot
from deep_lst.evaluation.plots.error_growth import (
    acc_day_night_plot,
    error_growth_by_init_plot,
    error_growth_daily_plot,
    error_growth_day_night_plot,
    error_growth_plot,
    skill_day_night_plot,
    skill_plot,
)
from deep_lst.evaluation.plots.init_time import (
    diurnal_lead_heatmap,
    diurnal_lead_lines,
    plot_init_vs_lead,
    score_over_init_plot,
)
from deep_lst.evaluation.plots.maps import acc_map, improvement_map_multi, lead_bucket_map_grid, lead_bucket_rmse_grid
from deep_lst.evaluation.plots.scores import (
    category_score_violinplot,
    error_overview_plot,
    monthly_daily_violin,
    monthly_daily_violin_buckets,
)
from deep_lst.evaluation.plots.spectra import compute_temporal_power_spectrum, temporal_power_spectrum_plot
from deep_lst.evaluation.plots.timeseries import timeseries_plot
from deep_lst.evaluation.plots.verification import activity_plot, srmse_by_latitude

_MONTH_LABELS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def _grid_vmax(tod_values, labels, percentile, use_abs):
    """Shared colour-scale max pooled across several tod dicts {bucket: {label: DataArray}}, so the
    day and night map figures are comparable. abs for diverging maps (improvement/skill), raw for
    the sequential error maps."""
    pool = []
    for values in tod_values:
        for bucket in values:
            for label in labels:
                da = values[bucket].get(label)
                if da is None:
                    continue
                arr = da.values[np.isfinite(da.values)].ravel()
                pool.append(np.abs(arr) if use_abs else arr)
    if not pool:
        return None
    pooled = np.concatenate(pool)
    return float(np.nanpercentile(pooled, percentile)) if pooled.size else None


def _with_delta(ds: xr.Dataset) -> xr.Dataset:
    """Add pred_delta / obs_delta (circular roll) to a banded grouped dataset."""
    group_dim = next(d for d in ds.dims if d in ("hour", "month"))
    pm_key = "pred_mean_q50" if "pred_mean_q50" in ds else "pred_mean"
    om_key = "obs_mean_q50" if "obs_mean_q50" in ds else "obs_mean"
    ds = ds.copy()
    ds["pred_delta"] = ds[pm_key].roll({group_dim: -1}) - ds[pm_key]
    ds["obs_delta"] = ds[om_key].roll({group_dim: -1}) - ds[om_key]
    return ds


def _quantile_bands(source, dims, quantiles=(0.05, 0.50, 0.95)):
    """Render-side q05/q50/q95 bands over `dims` for every variable (legacy summarise schema).

    NaN-safe: fully-NaN variables (e.g. empty hour bins) yield NaN bands instead of
    tripping xarray's quantile on all-NaN input.
    """
    q_labels = [f"q{int(q * 100):02d}" for q in quantiles]
    data_vars = {}
    for name in source.data_vars:
        var = source[name]
        reduce_dims = [d for d in dims if d in var.dims]
        if not reduce_dims:
            for ql in q_labels:
                data_vars[f"{name}_{ql}"] = var
            continue
        has_data = bool(var.notnull().any())
        for q, ql in zip(quantiles, q_labels):
            if has_data:
                data_vars[f"{name}_{ql}"] = var.quantile(q, dim=reduce_dims).drop_vars("quantile")
            else:
                data_vars[f"{name}_{ql}"] = var.isel({d: 0 for d in reduce_dims}, drop=True) * np.nan
    return xr.Dataset(data_vars)


def _requires(*products):
    """Mark a plot method with the evaluator products it consumes."""

    def mark(fn):
        fn.requires = products
        return fn

    return mark


def required_products(runner_cls, plots: list | None = None) -> tuple:
    """Union of the `requires` declarations of the requested plot_* methods."""
    requested = set(plots) if plots else None
    needed = []
    for name in sorted(dir(runner_cls)):
        if not name.startswith("plot_"):
            continue
        if requested is not None and name[5:] not in requested:
            continue
        needed.extend(getattr(getattr(runner_cls, name), "requires", ()))
    return tuple(dict.fromkeys(needed))


class PlotRunnerDiagnostics:
    """Renders single-lead diagnostic plots from evaluator diag products (evaluation/products.py).

    Score plots declare their products via `@_requires` (checked against ev.diag_products);
    field plots (timeseries, temporal spectrum) consume the raw forecast by declared
    exception. `baseline_labels` maps registry baseline names to display labels.
    """

    CLASS_COORDS = ("kg_class", "kg_major", "kg_classes", "biome_classes", "modis_lc_classes", "ecoregion_classes")

    def __init__(self, evaluator, label, plots_dir, baseline_labels=None, clim_baseline=None, lead_time="1h"):
        self.ev = evaluator
        self.label = label
        self.plots_dir = Path(plots_dir)
        self.baseline_labels = baseline_labels or {}
        self.clim_baseline = clim_baseline
        self.lead_time = lead_time

    def run(self, plots: list[str] | None = None):
        requested = set(plots) if plots else None
        for name in sorted(dir(self)):
            if name.startswith("plot_"):
                if requested is None or name[5:] in requested:
                    getattr(self, name)()

    def _product(self, name: str):
        return getattr(self.ev, "diag_products", {}).get(name)

    def _space_dims(self, ds) -> list:
        return [d for d in ds.dims if d in ("lat", "lon", "pixel")]

    def _entity_datasets(self, product, measures) -> dict:
        """{display_label: Dataset of `measures`} for the model and every baseline column present."""
        entities = {self.label: xr.Dataset({m: product[m] for m in measures if m in product})}
        for name, display in self.baseline_labels.items():
            columns = {m: product[f"{m}_{name}"] for m in measures if f"{m}_{name}" in product}
            if columns:
                entities[display] = xr.Dataset(columns)
        return entities

    def _class_labels(self) -> dict[str, dict]:
        if getattr(self, "_class_labels_cache", None) is not None:
            return self._class_labels_cache
        meta_path = self.ev.paths.eval_meta_path()
        try:
            meta = json.loads(meta_path.read_text())
            labels = {coord: meta.get(f"{coord}_labels", {}) for coord in self.CLASS_COORDS}
        except FileNotFoundError:
            labels = {coord: {} for coord in self.CLASS_COORDS}
        self._class_labels_cache = labels
        return labels

    # --- field plots ----------------------------------------------------------

    def plot_timeseries(self):
        timeseries_plot(
            self.ev.forecast_ds,
            var=self.ev.var,
            save_path=self.plots_dir / f"{self.ev.var}_timeseries.pdf",
            eval_region=self.ev.eval_name,
        )

    def plot_temporal_power_spectrum(self):
        from deep_lst.evaluation.metrics import to_valid_time

        pred = to_valid_time(self.ev.pred.sel(lead_time=self.lead_time), self.lead_time)
        obs = to_valid_time(self.ev.obs.sel(lead_time=self.lead_time), self.lead_time)
        diffs = np.diff(pred["valid_time"].values).astype("timedelta64[h]").astype(float)
        dt_hours = float(np.median(diffs[diffs > 0])) if len(diffs) > 0 else 1.0
        spectra = {
            self.label: compute_temporal_power_spectrum(pred.transpose("valid_time", ...).values, dt_hours=dt_hours)
        }
        obs_spectrum = compute_temporal_power_spectrum(obs.transpose("valid_time", ...).values, dt_hours=dt_hours)
        temporal_power_spectrum_plot(
            pred_spectra_by_case=spectra,
            obs_spectrum=obs_spectrum,
            var=self.ev.var,
            save_path=self.plots_dir / f"{self.ev.var}_temporal_power_spectrum.pdf",
            eval_region=self.ev.eval_name,
        )

    # --- grouped (diurnal) ------------------------------------------------------

    @_requires("diag_by_hour")
    def plot_grouped_mean(self):
        product = self._product("diag_by_hour")
        if product is None:
            return
        bands = _quantile_bands(product, self._space_dims(product))
        bl_means = {}
        for name, display in self.baseline_labels.items():
            key = f"pred_mean_{name}_q50"
            if key in bands:
                bl_means[display] = bands[key].values
        diurnal_mean_plot(
            bands,
            var=self.ev.var,
            baselines_mean=bl_means or None,
            save_path=self.plots_dir / f"{self.ev.var}_grouped_mean.pdf",
            eval_region=self.ev.eval_name,
            case_label=self.label,
        )

    @_requires("diag_by_hour", "diag_anomaly_by_hour")
    def plot_grouped_scores(self):
        product = self._product("diag_by_hour")
        if product is None:
            return
        bands = _quantile_bands(product, self._space_dims(product))
        for m in RAW_METRICS:
            if m not in product:
                continue
            diurnal_line(
                {self.label: bands},
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_grouped_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} (diurnal)",
            )
        anomaly = self._product("diag_anomaly_by_hour")
        if anomaly is None:
            return
        anomaly_bands = _quantile_bands(anomaly, self._space_dims(anomaly))
        for m in ANOMALY_METRICS:
            if m not in anomaly:
                continue
            diurnal_line(
                {self.label: anomaly_bands},
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_grouped_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} (diurnal)",
            )

    @_requires("diag_by_hour")
    def plot_grouped_delta(self):
        product = self._product("diag_by_hour")
        if product is None:
            return
        bands = _quantile_bands(product, self._space_dims(product))
        cases = {self.label: _with_delta(bands)}
        for name, display in self.baseline_labels.items():
            pm_key = f"pred_mean_{name}_q50"
            if pm_key not in bands:
                continue
            entity = xr.Dataset({"pred_mean_q50": bands[pm_key], "obs_mean_q50": bands["obs_mean_q50"]})
            cases[display] = _with_delta(entity)
        diurnal_delta_comparison(
            cases,
            var=self.ev.var,
            save_path=self.plots_dir / f"{self.ev.var}_grouped_delta.pdf",
            eval_region=self.ev.eval_name,
        )

    @_requires("diag_by_hour", "diag_anomaly_by_hour")
    def plot_grouped_comparison(self):
        product = self._product("diag_by_hour")
        if product is None or not self.baseline_labels:
            return
        space = self._space_dims(product)
        cases = {self.label: _quantile_bands(product[[m for m in RAW_METRICS if m in product]], space)}
        for name, display in self.baseline_labels.items():
            columns = {m: product[f"{m}_{name}"] for m in RAW_METRICS if f"{m}_{name}" in product}
            if columns:
                cases[display] = _quantile_bands(xr.Dataset(columns), space)
        for m in RAW_METRICS:
            if m not in product:
                continue
            diurnal_line(
                cases,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_grouped_comparison_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} comparison (diurnal)",
            )
        anomaly = self._product("diag_anomaly_by_hour")
        if anomaly is None:
            return
        anomaly_bands = _quantile_bands(anomaly, self._space_dims(anomaly))
        for m in ANOMALY_METRICS:
            if m not in anomaly:
                continue
            diurnal_line(
                {self.label: anomaly_bands},
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_grouped_comparison_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} comparison (diurnal)",
            )

    # --- monthly / class violins ------------------------------------------------

    @_requires("diag_by_day", "diag_anomaly_by_day")
    def plot_monthly_score_violin(self):
        for product_name, metrics in (("diag_by_day", RAW_METRICS), ("diag_anomaly_by_day", ANOMALY_METRICS)):
            product = self._product(product_name)
            if product is None:
                continue
            for m in metrics:
                if m not in product:
                    continue
                monthly_daily_violin(
                    product[m],
                    float(product[f"{m}_total"]),
                    metric=m,
                    case_label=self.label,
                    month_labels=_MONTH_LABELS,
                    title=f"{self.ev.eval_name} — {m.upper()} by month (lead_time={self.lead_time})",
                    save_path=self.plots_dir / f"{self.ev.var}_monthly_score_violin_{m}.pdf",
                )

    @_requires("diag_by_class", "diag_anomaly_by_class")
    def plot_class_score_violin(self):
        labels_by_coord = self._class_labels()
        for prefix, metrics in (("diag_by_class", RAW_METRICS), ("diag_anomaly_by_class", ANOMALY_METRICS)):
            for coord in self.CLASS_COORDS:
                product = self._product(f"{prefix}__{coord}")
                if product is None:
                    continue
                labels = labels_by_coord.get(coord) or None
                for m in metrics:
                    if m not in product:
                        continue
                    category_score_violinplot(
                        product,
                        metric=m,
                        category_dim=coord,
                        case_label=self.label,
                        category_labels=labels,
                        xlabel=coord,
                        title=f"{self.ev.eval_name} — {m.upper()} by {coord} (lead_time={self.lead_time})",
                        save_path=self.plots_dir / f"{self.ev.var}_class_score_violin_{coord}_{m}.pdf",
                    )

    # --- benchmark ----------------------------------------------------------------

    @_requires("diag_scores_by_time", "diag_anomaly_by_time")
    def plot_error_overview(self):
        product = self._product("diag_scores_by_time")
        if product is None or not self.baseline_labels:
            return
        raw_cases = self._entity_datasets(product, RAW_METRICS)
        for m in RAW_METRICS:
            if m not in product:
                continue
            error_overview_plot(
                raw_cases,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_error_overview_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} (lead_time={self.lead_time})",
            )
        anomaly = self._product("diag_anomaly_by_time")
        if anomaly is None:
            return
        anom_cases = {self.label: anomaly}
        for m in ANOMALY_METRICS:
            if m not in anomaly:
                continue
            error_overview_plot(
                anom_cases,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_error_overview_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} (lead_time={self.lead_time})",
            )

    _SCORE_MAP_METRICS = {"acc": "ACC", "nse": "NSE"}

    @_requires("diag_map", "diag_anomaly_map")
    def plot_score_map(self):
        if "pixel" in self.ev.forecast_ds.dims:
            return
        for m, label in self._SCORE_MAP_METRICS.items():
            product = self._product("diag_anomaly_map" if m in ANOMALY_METRICS else "diag_map")
            if product is None or m not in product:
                continue
            acc_map(
                acc=product[m],
                cbar_label=label,
                save_path=self.plots_dir / f"{self.ev.var}_{m}_map.pdf",
                eval_region=self.ev.eval_name,
            )

    @_requires("diag_map")
    def plot_improvement_map(self):
        product = self._product("diag_map")
        if product is None or "pixel" in self.ev.forecast_ds.dims or not self.baseline_labels:
            return
        rmse_by_baseline = {}
        for name, display in self.baseline_labels.items():
            if f"rmse_{name}" in product:
                rmse_by_baseline[display] = product[f"rmse_{name}"]
        if not rmse_by_baseline:
            return
        improvement_map_multi(
            rmse_model=product["rmse"],
            rmse_by_baseline=rmse_by_baseline,
            save_path=self.plots_dir / f"{self.ev.var}_improvement_map.pdf",
            model_label=self.label,
            eval_region=self.ev.eval_name,
        )


class PlotRunnerVerification:
    """Renders verification plots from evaluator products (see evaluation/products.py).

    Score plots declare the products they consume via `@_requires`; `required_products`
    derives what the evaluator must compute for a --plots selection. A few field plots
    (cdf, error growth by hour/init, srmse-by-latitude) consume raw pred/obs by
    declared exception. `baseline_labels` maps registry names to display labels.
    """

    def __init__(self, evaluator, label, plots_dir, baseline_labels=None, clim_baseline=None):
        self.ev = evaluator
        self.label = label
        self.plots_dir = Path(plots_dir)
        self.baseline_labels = baseline_labels or {}
        self.clim_baseline = clim_baseline

    def run(self, plots: list[str] | None = None):
        requested = set(plots) if plots else None
        for name in sorted(dir(self)):
            if name.startswith("plot_"):
                if requested is None or name[5:] in requested:
                    getattr(self, name)()

    def _product(self, name: str):
        return getattr(self.ev, "products", {}).get(name)

    def _per_baseline(self, product, template: str) -> dict:
        """{display_label: product column} for every baseline column present in `product`."""
        columns = {}
        for name, display in self.baseline_labels.items():
            key = template.format(name=name)
            if key in product:
                columns[display] = product[key]
        return columns

    # --- init-time views of the per-(init, lead) scores -----------------------

    # Phase-aligned leads (all ≡ 6 h mod 24) so the init-hour and valid-hour
    # diurnal phases coincide across the overlaid curves.
    _INIT_LEADS = ("6h", "30h", "78h")

    def _init_score_series(self, scores, metric: str, group: str) -> dict[str, dict]:
        """Per-lead quantile curves of ``metric`` over the init-time axis.

        ``scores`` is any (init_time, lead_time) dataset — raw scores or a skill
        cube. ``group="date"`` resamples init_time to daily bins, so the q05–q95
        band is the within-day (diurnal) spread; ``group="hour"`` groups by init
        hour-of-day, so the band is the across-days (seasonal) spread.
        """
        if metric not in scores:
            return {}
        qs = [0.05, 0.50, 0.95]
        available = pd.TimedeltaIndex(scores["lead_time"].values)
        series = {}
        for lead in self._INIT_LEADS:
            td = pd.Timedelta(lead)
            if td not in available:
                continue
            s = scores[metric].sel(lead_time=td)
            if group == "hour":
                g = s.groupby(s["init_time"].dt.hour).quantile(qs, skipna=True)
                x = g["hour"].values
            else:
                g = s.resample(init_time="1D").quantile(qs, skipna=True)
                x = g["init_time"].values
            series[lead] = {
                "x": x,
                "q05": g.sel(quantile=0.05).values,
                "q50": g.sel(quantile=0.50).values,
                "q95": g.sel(quantile=0.95).values,
            }
        return series

    def _diurnal_lead_matrix(self, scores, metric: str, max_lead_h: int = 24):
        """Median ``metric`` over days, per (init hour-of-day, lead time) up to
        ``max_lead_h``. Returns a DataArray with dims (hour, lead_time)."""
        if metric not in scores:
            return None
        leads = pd.TimedeltaIndex(scores["lead_time"].values)
        keep = scores["lead_time"].values[leads <= pd.Timedelta(hours=max_lead_h)]
        sub = scores[metric].sel(lead_time=keep)
        return sub.groupby(sub["init_time"].dt.hour).median()

    @_requires("scores_by_init_lead")
    def plot_error_by_init_date(self):
        product = self._product("scores_by_init_lead")
        if product is None:
            return
        series = self._init_score_series(product, "rmse", group="date")
        if not series:
            return
        score_over_init_plot(
            series,
            xlabel="Initialisation date",
            ylabel="RMSE (K)",
            x_kind="date",
            legend_title="Lead time",
            title=f"{self.ev.eval_name} — RMSE by init date",
            save_path=self.plots_dir / f"{self.ev.var}_rmse_by_init_date.pdf",
        )

    @_requires("scores_by_init_lead")
    def plot_error_by_init_hour(self):
        product = self._product("scores_by_init_lead")
        if product is None:
            return
        series = self._init_score_series(product, "rmse", group="hour")
        if not series:
            return
        shift = int(pd.Timedelta(self._INIT_LEADS[0]).total_seconds() // 3600) % 24
        score_over_init_plot(
            series,
            xlabel="Initialisation hour of day",
            ylabel="RMSE (K)",
            x_kind="hour",
            legend_title="Lead time",
            valid_shift_h=shift,
            title=f"{self.ev.eval_name} — RMSE by init hour",
            save_path=self.plots_dir / f"{self.ev.var}_rmse_by_init_hour.pdf",
        )

    @_requires("scores_by_init_lead")
    def plot_error_by_init_hour_leads(self):
        product = self._product("scores_by_init_lead")
        if product is None:
            return
        m = self._diurnal_lead_matrix(product, "rmse")
        if m is None:
            return
        diurnal_lead_lines(
            m,
            xlabel="Initialisation hour of day",
            ylabel="RMSE (K)",
            title=f"{self.ev.eval_name} — RMSE by init hour and lead (0–24 h)",
            save_path=self.plots_dir / f"{self.ev.var}_rmse_by_init_hour_leads.pdf",
        )

    @_requires("scores_by_init_lead")
    def plot_diurnal_lead_heatmap(self):
        product = self._product("scores_by_init_lead")
        if product is None:
            return
        m = self._diurnal_lead_matrix(product, "rmse")
        if m is None:
            return
        diurnal_lead_heatmap(
            m,
            metric_label="RMSE (K)",
            cmap="RdPu",
            vmin=0,
            vmax=3,
            title=f"{self.ev.eval_name} — RMSE by init hour and lead",
            save_path=self.plots_dir / f"{self.ev.var}_rmse_diurnal_lead_heatmap.pdf",
        )

    @_requires("scores_by_init_lead")
    def plot_skill_by_init_hour(self):
        product = self._product("scores_by_init_lead")
        if product is None:
            return
        shift = int(pd.Timedelta(self._INIT_LEADS[0]).total_seconds() // 3600) % 24
        for display, skill in self._per_baseline(product, "rmse_skill_{name}").items():
            series = self._init_score_series(xr.Dataset({"rmse": skill}), "rmse", group="hour")
            if not series:
                continue
            safe = display.replace(" ", "_")
            score_over_init_plot(
                series,
                xlabel="Initialisation hour of day",
                ylabel="RMSE skill score",
                x_kind="hour",
                legend_title="Lead time",
                valid_shift_h=shift,
                hline=0.0,
                title=f"{self.ev.eval_name} — RMSE skill vs {display} by init hour",
                save_path=self.plots_dir / f"{self.ev.var}_skill_{safe}_by_init_hour.pdf",
            )

    @_requires("scores_by_init_lead")
    def plot_skill_diurnal_lead_heatmap(self):
        product = self._product("scores_by_init_lead")
        if product is None:
            return
        for display, skill in self._per_baseline(product, "rmse_skill_{name}").items():
            m = self._diurnal_lead_matrix(xr.Dataset({"rmse": skill}), "rmse")
            if m is None:
                continue
            safe = display.replace(" ", "_")
            diurnal_lead_heatmap(
                m,
                metric_label="RMSE skill score",
                cmap="RdBu",
                vmin=-1,
                vmax=1,
                title=f"{self.ev.eval_name} — RMSE skill vs {display} by init hour and lead",
                save_path=self.plots_dir / f"{self.ev.var}_skill_{safe}_diurnal_lead_heatmap.pdf",
            )

    # --- pooled curves --------------------------------------------------------

    @_requires("scores_by_lead")
    def plot_error_growth(self):
        product = self._product("scores_by_lead")
        if product is None:
            return
        for m in RAW_METRICS:
            if m not in product:
                continue
            error_growth_plot(
                product,
                case_label=self.label,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_error_growth_{m}.pdf",
                eval_region=self.ev.eval_name,
            )

    @_requires("scores_by_forecast_day")
    def plot_error_growth_daily(self):
        product = self._product("scores_by_forecast_day")
        if product is None:
            return
        for m in RAW_METRICS:
            if m not in product:
                continue
            error_growth_daily_plot(
                product,
                case_label=self.label,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_error_growth_daily_{m}.pdf",
                eval_region=self.ev.eval_name,
            )

    @_requires("scores_by_forecast_day")
    def plot_skill_scores(self):
        product = self._product("scores_by_forecast_day")
        if product is None:
            return
        for m in RAW_METRICS:
            references = {}
            for display, skill in self._per_baseline(product, f"{m}_skill_{{name}}").items():
                references[display] = xr.Dataset({m: skill})
            if not references:
                continue
            skill_plot(
                references,
                case_label=self.label,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_skill_{m}.pdf",
                eval_region=self.ev.eval_name,
            )

    @_requires("rmse_by_forecast_day_daytime", "rmse_by_forecast_day_nighttime")
    def plot_skill_day_night(self):
        day = self._product("rmse_by_forecast_day_daytime")
        night = self._product("rmse_by_forecast_day_nighttime")
        if day is None or night is None:
            return
        day_skill = self._per_baseline(day, "rmse_skill_{name}")
        night_skill = self._per_baseline(night, "rmse_skill_{name}")
        if not day_skill:
            return
        skill_day_night_plot(
            day_skill,
            night_skill,
            self.ev.var,
            self.plots_dir / f"{self.ev.var}_skill_day_night.pdf",
            self.ev.eval_name,
        )

    @_requires("anomaly_by_forecast_day_daytime", "anomaly_by_forecast_day_nighttime")
    def plot_climate_acc(self):
        day = self._product("anomaly_by_forecast_day_daytime")
        night = self._product("anomaly_by_forecast_day_nighttime")
        if day is None or night is None:
            return
        acc_day_night_plot(
            day["acc"],
            night["acc"],
            self.ev.var,
            self.plots_dir / f"{self.ev.var}_climate_acc.pdf",
            self.ev.eval_name,
        )

    @_requires("anomaly_by_forecast_day")
    def plot_climate_activity(self):
        product = self._product("anomaly_by_forecast_day")
        if product is None or self.clim_baseline is None:
            return
        display = self.baseline_labels.get(self.clim_baseline, self.clim_baseline)
        activity_plot(
            {display: product},
            save_path=self.plots_dir / f"{self.ev.var}_climate_activity.pdf",
            eval_region=self.ev.eval_name,
        )

    @_requires("anomaly_by_init_lead")
    def plot_init_lead_heatmap(self):
        product = self._product("anomaly_by_init_lead")
        if product is None:
            return
        plot_init_vs_lead(
            product,
            metric="acc",
            save_path=self.plots_dir / f"{self.ev.var}_init_lead_heatmap_acc.pdf",
        )

    @_requires("rmse_by_valid_day_lead1_24", "rmse_by_valid_day_lead217_240")
    def plot_monthly_daily_violin(self):
        buckets = {}
        for bucket_label, product_name in (
            ("Day 1", "rmse_by_valid_day_lead1_24"),
            ("Day 10", "rmse_by_valid_day_lead217_240"),
        ):
            product = self._product(product_name)
            if product is None or product.sizes.get("day", 0) == 0:
                continue
            buckets[bucket_label] = (product["rmse"], float(product["rmse_total"]))
        if not buckets:
            return
        monthly_daily_violin_buckets(
            buckets,
            metric="rmse",
            case_label=self.label,
            month_labels=_MONTH_LABELS,
            title=f"{self.ev.eval_name} — RMSE by month, forecast-day buckets",
            save_path=self.plots_dir / f"{self.ev.var}_monthly_daily_violin.pdf",
        )

    @_requires("rmse_map_by_bucket_tod")
    def plot_improvement_skill_maps(self):
        product = self._product("rmse_map_by_bucket_tod")
        if product is None or "pixel" in self.ev.forecast_ds.dims or not self.baseline_labels:
            return
        bucket_days = [d for d in (1, 5, 10) if d in product["day"].values]
        if not bucket_days:
            return
        bucket_labels = [f"≤ Day {d}" for d in bucket_days]
        baseline_displays = list(self.baseline_labels.values())
        tods = (("day", "daytime 10–15 h local"), ("night", "nighttime 22–03 h local"))

        error_by_tod, improvement_by_tod, skill_by_tod = {}, {}, {}
        for tod, _ in tods:
            error = {b: {} for b in bucket_labels}
            improvement = {b: {} for b in bucket_labels}
            skill = {b: {} for b in bucket_labels}
            for bucket_label, day in zip(bucket_labels, bucket_days):
                cell = product.sel(tod=tod, day=day)
                error[bucket_label][self.label] = cell["rmse"]
                for name, display in self.baseline_labels.items():
                    if f"rmse_{name}" not in cell:
                        continue
                    error[bucket_label][display] = cell[f"rmse_{name}"]
                    improvement[bucket_label][display] = cell[f"rmse_{name}"] - cell["rmse"]
                    skill[bucket_label][display] = cell[f"rmse_skill_{name}"]
            error_by_tod[tod] = error
            improvement_by_tod[tod] = improvement
            skill_by_tod[tod] = skill

        error_vmax = _grid_vmax(error_by_tod.values(), [self.label, *baseline_displays], 98, use_abs=False)
        improvement_vmax = _grid_vmax(improvement_by_tod.values(), baseline_displays, 95, use_abs=True)
        skill_vmax = _grid_vmax(skill_by_tod.values(), baseline_displays, 95, use_abs=True)

        for tod, tod_label in tods:
            lead_bucket_rmse_grid(
                error_by_tod[tod],
                bucket_labels,
                [self.label, *baseline_displays],
                self.plots_dir / f"{self.ev.var}_error_map_{tod}.pdf",
                tod_label=tod_label,
                eval_region=self.ev.eval_name,
                vmax=error_vmax,
            )
            lead_bucket_map_grid(
                improvement_by_tod[tod],
                bucket_labels,
                baseline_displays,
                self.plots_dir / f"{self.ev.var}_improvement_map_{tod}.pdf",
                kind="improvement",
                tod_label=tod_label,
                model_label=self.label,
                eval_region=self.ev.eval_name,
                vmax=improvement_vmax,
            )
            lead_bucket_map_grid(
                skill_by_tod[tod],
                bucket_labels,
                baseline_displays,
                self.plots_dir / f"{self.ev.var}_skill_map_{tod}.pdf",
                kind="skill",
                tod_label=tod_label,
                model_label=self.label,
                eval_region=self.ev.eval_name,
                vmax=skill_vmax,
            )

    # --- field plots: consume raw pred/obs by declared exception --------------

    def plot_error_growth_day_night(self):
        # RMSE vs lead at fixed local hour. Field plot: needs the raw cube.
        if "lead_time" not in self.ev.pred.dims:
            return
        error_growth_day_night_plot(
            self.ev.pred,
            self.ev.obs,
            self.ev.var,
            self.plots_dir / f"{self.ev.var}_error_growth_day_night.pdf",
            self.ev.eval_name,
        )

    def plot_error_growth_by_init(self):
        # RMSE vs lead, one curve per initialisation hour. Field plot.
        if "lead_time" not in self.ev.pred.dims:
            return
        error_growth_by_init_plot(
            self.ev.pred,
            self.ev.obs,
            self.ev.var,
            self.plots_dir / f"{self.ev.var}_error_growth_by_init.pdf",
            self.ev.eval_name,
        )

    def plot_cdf(self):
        # Day/night value- & per-pixel-RMSE CDFs, lead 1→24 gradient. Field plot.
        if "lead_time" not in self.ev.pred.dims:
            return
        summary = day_night_summary(self.ev.pred, self.ev.obs)
        lst_cdf_plot(
            self.ev.pred,
            self.ev.obs,
            self.ev.var,
            self.plots_dir / f"{self.ev.var}_value_cdf.pdf",
            self.ev.eval_name,
            summary=summary,
        )
        rmse_cdf_plot(
            self.ev.pred,
            self.ev.obs,
            self.ev.var,
            self.plots_dir / f"{self.ev.var}_rmse_cdf.pdf",
            self.ev.eval_name,
            summary=summary,
        )

    def plot_srmse_by_latitude(self):
        # Only for the temporal-extrapolation eval over the spread-out training sites.
        if self.ev.eval_name != "train_mtg" or "lead_time" not in self.ev.pred.dims or "lat" not in self.ev.pred.coords:
            return
        srmse_by_latitude(
            self.ev.pred,
            self.ev.obs,
            self.ev.var,
            save_path=self.plots_dir / f"{self.ev.var}_srmse_by_latitude.pdf",
            eval_region=self.ev.eval_name,
        )
