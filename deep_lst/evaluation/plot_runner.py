import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.evaluator import (
    ANOMALY_METRICS,
    CLIMATE_BASELINES,
    RAW_METRICS,
    EvaluatorDiagnostics,
)
from deep_lst.evaluation.metrics import compute_class_scores, compute_scores
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


class PlotRunnerDiagnostics:
    def __init__(
        self,
        evaluator: EvaluatorDiagnostics,
        label: str,
        plots_dir: Path,
        baseline_evaluators: dict[str, EvaluatorDiagnostics] | None = None,
        class_coords: tuple[str, ...] = (
            "kg_class",
            "kg_major",
            "kg_classes",
            "biome_classes",
            "modis_lc_classes",
            "ecoregion_classes",
        ),
    ):
        self.ev = evaluator
        self.label = label
        self.plots_dir = Path(plots_dir)
        self.baselines = baseline_evaluators or {}  # {display_label: EvaluatorDiagnostics}
        self.class_coords = class_coords

        # The primary evaluator runs with baseline=None and cannot compute
        # anomalies on its own. Copy them across from the climate-baseline
        # evaluator (same model pred/obs, with clim_smooth subtracted) so
        # self.ev exposes climate_scores/climate_grouped/pred_anom/obs_anom
        # for all downstream anomaly plots.
        clim_ev = next(
            (bl for bl in self.baselines.values() if bl.baseline in CLIMATE_BASELINES),
            None,
        )
        if clim_ev is not None and clim_ev.climate_scores is not None:
            self.ev.pred_anom = clim_ev.pred_anom
            self.ev.obs_anom = clim_ev.obs_anom
            self.ev.climate_scores = clim_ev.climate_scores
            self.ev.climate_grouped = clim_ev.climate_grouped

    def run(self, plots: list[str] | None = None):
        requested = set(plots) if plots else None
        for name in sorted(dir(self)):
            if name.startswith("plot_"):
                if requested is None or name[5:] in requested:
                    getattr(self, name)()

    # --- fast eval ---

    def plot_timeseries(self):
        timeseries_plot(
            self.ev.forecast_ds,
            var=self.ev.var,
            save_path=self.plots_dir / f"{self.ev.var}_timeseries.pdf",
            eval_region=self.ev.eval_name,
        )

    def plot_temporal_power_spectrum(self):
        diffs = np.diff(self.ev.pred["valid_time"].values).astype("timedelta64[h]").astype(float)
        dt_hours = float(np.median(diffs[diffs > 0])) if len(diffs) > 0 else 1.0
        spectra = {
            self.label: compute_temporal_power_spectrum(
                self.ev.pred.transpose("valid_time", ...).values, dt_hours=dt_hours
            )
        }
        obs_spectrum = compute_temporal_power_spectrum(
            self.ev.obs.transpose("valid_time", ...).values, dt_hours=dt_hours
        )
        temporal_power_spectrum_plot(
            pred_spectra_by_case=spectra,
            obs_spectrum=obs_spectrum,
            var=self.ev.var,
            save_path=self.plots_dir / f"{self.ev.var}_temporal_power_spectrum.pdf",
            eval_region=self.ev.eval_name,
        )

    # --- grouped (diurnal / monthly) ---

    def plot_grouped_mean(self):
        summarised = self.ev.summarise(self.ev.grouped)
        bl_means = {
            label: bl_ev.summarise(bl_ev.baseline_grouped)["pred_mean_q50"].values
            for label, bl_ev in self.baselines.items()
            if getattr(bl_ev, "baseline_grouped", None) is not None
        }
        diurnal_mean_plot(
            summarised,
            var=self.ev.var,
            baselines_mean=bl_means or None,
            save_path=self.plots_dir / f"{self.ev.var}_grouped_mean.pdf",
            eval_region=self.ev.eval_name,
            case_label=self.label,
        )

    def plot_grouped_scores(self):
        # Raw-field metrics: from self.ev.grouped (rmse, mae, bias, nse).
        summarised = self.ev.summarise(self.ev.grouped)
        for m in RAW_METRICS:
            if m not in self.ev.grouped:
                continue
            diurnal_line(
                {self.label: summarised},
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_grouped_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} ({self.ev.groupby})",
            )
        # Anomaly metrics: from self.ev.climate_grouped (acc, activity).
        if self.ev.climate_grouped is None:
            return
        summarised_anom = self.ev.summarise(self.ev.climate_grouped)
        for m in ANOMALY_METRICS:
            if m not in self.ev.climate_grouped:
                continue
            diurnal_line(
                {self.label: summarised_anom},
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_grouped_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} ({self.ev.groupby})",
            )

    def _class_labels(self) -> dict[str, dict]:
        if getattr(self, "_class_labels_cache", None) is not None:
            return self._class_labels_cache
        meta_path = self.ev.paths.eval_meta_path("seviri")
        try:
            meta = json.loads(meta_path.read_text())
            labels = {coord: meta.get(f"{coord}_labels", {}) for coord in self.class_coords}
        except FileNotFoundError:
            labels = {coord: {} for coord in self.class_coords}
        self._class_labels_cache = labels
        return labels

    def plot_monthly_score_violin(self):
        # Per month, the distribution of daily pooled scores (horizontal violins) with the total
        # pooled score as a vertical reference line. Computed on the evaluator (self.ev.daily_pooled).
        daily_pooled = dict(self.ev.daily_pooled)
        if self.ev.daily_pooled_anom is not None:
            daily_pooled.update(self.ev.daily_pooled_anom)
        for m, (daily_scores, total) in daily_pooled.items():
            monthly_daily_violin(
                daily_scores,
                total,
                metric=m,
                case_label=self.label,
                month_labels=_MONTH_LABELS,
                title=f"{self.ev.eval_name} — {m.upper()} by month (lead_time={self.ev.lead_time})",
                save_path=self.plots_dir / f"{self.ev.var}_monthly_score_violin_{m}.pdf",
            )

    def plot_class_score_violin(self):
        available = [c for c in self.class_coords if c in self.ev.pred.coords]
        labels_by_coord = self._class_labels()
        for class_coord in available:
            labels = labels_by_coord.get(class_coord) or None
            # Raw-field metrics
            raw_scores = compute_class_scores(
                self.ev.pred, self.ev.obs, class_coord=class_coord, metrics=RAW_METRICS, aggregation="region_time"
            )
            for m in RAW_METRICS:
                if m not in raw_scores:
                    continue
                category_score_violinplot(
                    raw_scores,
                    metric=m,
                    category_dim=class_coord,
                    case_label=self.label,
                    category_labels=labels,
                    xlabel=class_coord,
                    title=f"{self.ev.eval_name} — {m.upper()} by {class_coord} (lead_time={self.ev.lead_time})",
                    save_path=self.plots_dir / f"{self.ev.var}_class_score_violin_{class_coord}_{m}.pdf",
                )
            # Anomaly metrics
            if self.ev.pred_anom is None:
                continue
            anom_scores = compute_class_scores(
                self.ev.pred_anom,
                self.ev.obs_anom,
                class_coord=class_coord,
                metrics=ANOMALY_METRICS,
                aggregation="region_time",
            )
            for m in ANOMALY_METRICS:
                if m not in anom_scores:
                    continue
                category_score_violinplot(
                    anom_scores,
                    metric=m,
                    category_dim=class_coord,
                    case_label=self.label,
                    category_labels=labels,
                    xlabel=class_coord,
                    title=f"{self.ev.eval_name} — {m.upper()} by {class_coord} (lead_time={self.ev.lead_time})",
                    save_path=self.plots_dir / f"{self.ev.var}_class_score_violin_{class_coord}_{m}.pdf",
                )

    # --- benchmark (silently skip if no baselines) ---

    def plot_error_overview(self):
        if not self.baselines:
            return
        # Raw-field metrics: model + every baseline as side-by-side boxes.
        raw_cases = {self.label: compute_scores(self.ev.pred, self.ev.obs, RAW_METRICS, "region_time")}
        for label, bl_ev in self.baselines.items():
            bl = getattr(bl_ev, "baseline_aligned", None)
            if bl is not None:
                raw_cases[label] = compute_scores(bl, self.ev.obs, RAW_METRICS, "region_time")
        for m in RAW_METRICS:
            if m not in raw_cases[self.label]:
                continue
            error_overview_plot(
                raw_cases,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_error_overview_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} (lead_time={self.ev.lead_time})",
            )
        # Anomaly metrics: only the model (no meaningful per-baseline anomaly).
        if self.ev.pred_anom is None:
            return
        anom_cases = {self.label: compute_scores(self.ev.pred_anom, self.ev.obs_anom, ANOMALY_METRICS, "region_time")}
        for m in ANOMALY_METRICS:
            if m not in anom_cases[self.label]:
                continue
            error_overview_plot(
                anom_cases,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_error_overview_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} (lead_time={self.ev.lead_time})",
            )

    def _with_delta(self, ds: xr.Dataset) -> xr.Dataset:
        """Add pred_delta / obs_delta (circular roll) to a grouped dataset."""
        group_dim = next(d for d in ds.dims if d in ("hour", "month"))
        pm_key = "pred_mean_q50" if "pred_mean_q50" in ds else "pred_mean"
        om_key = "obs_mean_q50" if "obs_mean_q50" in ds else "obs_mean"
        ds = ds.copy()
        ds["pred_delta"] = ds[pm_key].roll({group_dim: -1}) - ds[pm_key]
        ds["obs_delta"] = ds[om_key].roll({group_dim: -1}) - ds[om_key]
        return ds

    def plot_grouped_delta(self):
        cases = {self.label: self._with_delta(self.ev.summarise(self.ev.grouped))}
        cases.update(
            {
                label: self._with_delta(bl_ev.summarise(bl_ev.baseline_grouped))
                for label, bl_ev in self.baselines.items()
                if getattr(bl_ev, "baseline_grouped", None) is not None
            }
        )
        diurnal_delta_comparison(
            cases,
            var=self.ev.var,
            save_path=self.plots_dir / f"{self.ev.var}_grouped_delta.pdf",
            eval_region=self.ev.eval_name,
        )

    _SCORE_MAP_METRICS = {
        "acc": "ACC",
        "nse": "NSE",
    }

    def plot_score_map(self):
        if "pixel" in self.ev.forecast_ds.dims:
            return
        for m, label in self._SCORE_MAP_METRICS.items():
            if m in ANOMALY_METRICS:
                if self.ev.pred_anom is None:
                    continue
                pred_in, obs_in = self.ev.pred_anom, self.ev.obs_anom
            else:
                pred_in, obs_in = self.ev.pred, self.ev.obs
            score = compute_scores(pred_in, obs_in, [m], "region_space")[m]
            acc_map(
                acc=score,
                cbar_label=label,
                save_path=self.plots_dir / f"{self.ev.var}_{m}_map.pdf",
                eval_region=self.ev.eval_name,
            )

    def plot_improvement_map(self):
        if "pixel" in self.ev.forecast_ds.dims or not self.baselines:
            return
        rmse_model = compute_scores(self.ev.pred, self.ev.obs, ["rmse"], "region_space")["rmse"]
        rmse_by_baseline = {
            label: compute_scores(bl_ev.baseline_aligned, self.ev.obs, ["rmse"], "region_space")["rmse"]
            for label, bl_ev in self.baselines.items()
            if getattr(bl_ev, "baseline_aligned", None) is not None
        }
        if not rmse_by_baseline:
            return
        improvement_map_multi(
            rmse_model=rmse_model,
            rmse_by_baseline=rmse_by_baseline,
            save_path=self.plots_dir / f"{self.ev.var}_improvement_map.pdf",
            model_label=self.label,
            eval_region=self.ev.eval_name,
        )

    def plot_grouped_comparison(self):
        if not self.baselines:
            return
        # Raw-field metrics: model + every baseline.
        cases = {self.label: self.ev.summarise(self.ev.grouped)}
        cases.update(
            {
                label: bl_ev.summarise(bl_ev.baseline_grouped)
                for label, bl_ev in self.baselines.items()
                if getattr(bl_ev, "baseline_grouped", None) is not None
            }
        )
        for m in RAW_METRICS:
            if m not in self.ev.grouped:
                continue
            diurnal_line(
                cases,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_grouped_comparison_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} comparison ({self.ev.groupby})",
            )
        # Anomaly metrics: only the model line.
        if self.ev.climate_grouped is None:
            return
        anom_cases = {self.label: self.ev.summarise(self.ev.climate_grouped)}
        for m in ANOMALY_METRICS:
            if m not in self.ev.climate_grouped:
                continue
            diurnal_line(
                anom_cases,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_grouped_comparison_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} comparison ({self.ev.groupby})",
            )


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
