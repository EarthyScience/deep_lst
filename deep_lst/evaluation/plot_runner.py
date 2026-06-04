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
    EvaluatorVerification,
)
from deep_lst.evaluation.metrics import compute_class_scores, compute_grouped_scores, compute_scores
from deep_lst.evaluation.plots.diurnal import diurnal_delta_comparison, diurnal_line, diurnal_mean_plot
from deep_lst.evaluation.plots.error_growth import error_growth_daily_plot, error_growth_plot, skill_plot
from deep_lst.evaluation.plots.fingerprint import fingerprint_plot, improvement_da
from deep_lst.evaluation.plots.init_time import (
    diurnal_lead_heatmap,
    diurnal_lead_lines,
    plot_init_vs_lead,
    score_over_init_plot,
)
from deep_lst.evaluation.plots.maps import acc_map, improvement_map_multi
from deep_lst.evaluation.plots.scores import category_score_violinplot, error_overview_plot
from deep_lst.evaluation.plots.spectra import (
    _grid_spacing_km,
    compute_spatial_power_spectrum,
    compute_temporal_power_spectrum,
    spatial_power_spectrum_plot,
    temporal_power_spectrum_plot,
)
from deep_lst.evaluation.plots.timeseries import timeseries_plot
from deep_lst.evaluation.plots.verification import activity_plot

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


class PlotRunnerDiagnostics:
    def __init__(
        self,
        evaluator: EvaluatorDiagnostics,
        label: str,
        plots_dir: Path,
        baseline_evaluators: dict[str, EvaluatorDiagnostics] | None = None,
        class_coords: tuple[str, ...] = ("kg_class", "kg_major"),
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
        bl = {
            label: bl_ev.baseline_aligned
            for label, bl_ev in self.baselines.items()
            if getattr(bl_ev, "baseline_aligned", None) is not None
        }
        timeseries_plot(
            self.ev.forecast_ds,
            var=self.ev.var,
            lead_time=self.ev.lead_time,
            baselines=bl or None,
            save_path=self.plots_dir / f"{self.ev.var}_timeseries.pdf",
            eval_region=self.ev.eval_name,
            case_label=self.label,
        )

    def plot_fingerprint(self):
        fp_data = {f"{self.label} pred": self.ev.pred, "obs": self.ev.obs}
        fp_data.update(
            {
                label: bl_ev.baseline_aligned
                for label, bl_ev in self.baselines.items()
                if getattr(bl_ev, "baseline_aligned", None) is not None
            }
        )
        fingerprint_plot(
            fp_data,
            save_path=self.plots_dir / f"{self.ev.var}_fingerprint.pdf",
            title=f"{self.ev.eval_name} — fingerprint (lead_time={self.ev.lead_time})",
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
        # Raw-field metrics
        raw_scores = compute_grouped_scores(
            self.ev.pred, self.ev.obs, RAW_METRICS, aggregation="region_time", groupby="monthly"
        )
        for m in RAW_METRICS:
            if m not in raw_scores:
                continue
            category_score_violinplot(
                raw_scores,
                metric=m,
                category_dim="month",
                case_label=self.label,
                category_labels=_MONTH_LABELS,
                xlabel="Month",
                title=f"{self.ev.eval_name} — {m.upper()} by month (lead_time={self.ev.lead_time})",
                save_path=self.plots_dir / f"{self.ev.var}_monthly_score_violin_{m}.pdf",
            )
        # Anomaly metrics
        if self.ev.pred_anom is None:
            return
        anom_scores = compute_grouped_scores(
            self.ev.pred_anom, self.ev.obs_anom, ANOMALY_METRICS, aggregation="region_time", groupby="monthly"
        )
        for m in ANOMALY_METRICS:
            if m not in anom_scores:
                continue
            category_score_violinplot(
                anom_scores,
                metric=m,
                category_dim="month",
                case_label=self.label,
                category_labels=_MONTH_LABELS,
                xlabel="Month",
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

    def plot_improvement_fingerprint(self):
        if not self.baselines:
            return
        for label, bl_ev in self.baselines.items():
            bl = getattr(bl_ev, "baseline_aligned", None)
            if bl is None:
                continue
            safe = label.replace(" ", "_")
            fingerprint_plot(
                {f"{self.label} vs {label}": improvement_da(self.ev.pred, self.ev.obs, bl)},
                vcenter=0.0,
                cbar_label="ΔMAE (K)",
                title=f"{self.ev.eval_name} — improvement vs {label}",
                save_path=self.plots_dir / f"{self.ev.var}_improvement_{safe}.pdf",
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


class PlotRunnerVerification:
    def __init__(
        self,
        evaluator: EvaluatorVerification,
        label: str,
        plots_dir: Path,
        baseline_evaluators: dict[str, EvaluatorVerification] | None = None,
    ):
        self.ev = evaluator
        self.label = label
        self.plots_dir = Path(plots_dir)
        self.baselines = baseline_evaluators or {}

    def run(self, plots: list[str] | None = None):
        self.scores_summary = self.ev.summarise(self.ev.scores)
        self.skill_references = {}
        self.climate_references = {}
        self.climate_scores_raw = {}
        for label, bl_ev in self.baselines.items():
            skill = getattr(bl_ev, "skill_scores", None)
            if skill is not None:
                self.skill_references[label] = self.ev.summarise(1 - skill)
            cs = getattr(bl_ev, "climate_scores", None)
            if cs is not None:
                cs_computed = cs.compute()
                self.climate_scores_raw[label] = cs_computed
                self.climate_references[label] = self.ev.summarise(cs_computed)
        requested = set(plots) if plots else None
        for name in sorted(dir(self)):
            if name.startswith("plot_"):
                if requested is None or name[5:] in requested:
                    getattr(self, name)()

    # Phase-aligned leads (all ≡ 6 h mod 24) so the init-hour and valid-hour
    # diurnal phases coincide across the overlaid curves.
    _INIT_LEADS = ("6h", "30h", "78h")

    def _init_score_series(self, metric: str, group: str) -> dict[str, dict]:
        """Per-lead quantile curves of ``metric`` over the init-time axis.

        ``group="date"`` resamples init_time to daily bins, so the q05–q95 band
        is the within-day (diurnal) spread; ``group="hour"`` groups by init
        hour-of-day, so the band is the across-days (seasonal) spread. The line
        is q50 in both, matching the (0.05, 0.50, 0.95) quantiles of summarise.
        """
        scores = self.ev.scores
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

    def plot_error_by_init_date(self):
        series = self._init_score_series("rmse", group="date")
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

    def plot_error_by_init_hour(self):
        series = self._init_score_series("rmse", group="hour")
        if not series:
            return
        # All _INIT_LEADS are phase-aligned (≡ same offset mod 24), so one
        # valid-hour shift annotates every curve.
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

    def _diurnal_lead_matrix(self, metric: str, max_lead_h: int = 24):
        """Median ``metric`` over days, per (init hour-of-day, lead time) up to
        ``max_lead_h``. Returns a DataArray with dims (hour, lead_time)."""
        scores = self.ev.scores
        if metric not in scores:
            return None
        leads = pd.TimedeltaIndex(scores["lead_time"].values)
        keep = scores["lead_time"].values[leads <= pd.Timedelta(hours=max_lead_h)]
        sub = scores[metric].sel(lead_time=keep)
        return sub.groupby(sub["init_time"].dt.hour).median()

    def plot_error_by_init_hour_leads(self):
        m = self._diurnal_lead_matrix("rmse")
        if m is None:
            return
        diurnal_lead_lines(
            m,
            xlabel="Initialisation hour of day",
            ylabel="RMSE (K)",
            title=f"{self.ev.eval_name} — RMSE by init hour and lead (0–24 h)",
            save_path=self.plots_dir / f"{self.ev.var}_rmse_by_init_hour_leads.pdf",
        )

    def plot_diurnal_lead_heatmap(self):
        m = self._diurnal_lead_matrix("rmse")
        if m is None:
            return
        diurnal_lead_heatmap(
            m,
            metric_label="RMSE (K)",
            title=f"{self.ev.eval_name} — RMSE by init hour and lead",
            save_path=self.plots_dir / f"{self.ev.var}_rmse_diurnal_lead_heatmap.pdf",
        )

    def plot_error_growth(self):
        for m in RAW_METRICS:
            if m not in self.ev.scores:
                continue
            error_growth_plot(
                self.scores_summary,
                case_label=self.label,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_error_growth_{m}.pdf",
                eval_region=self.ev.eval_name,
            )

    def plot_error_growth_daily(self):
        for m in RAW_METRICS:
            if m not in self.ev.scores:
                continue
            error_growth_daily_plot(
                self.scores_summary,
                case_label=self.label,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_error_growth_daily_{m}.pdf",
                eval_region=self.ev.eval_name,
            )

    def plot_skill_scores(self):
        if not self.skill_references:
            return
        for m in RAW_METRICS:
            if m not in self.ev.scores:
                continue
            skill_plot(
                self.skill_references,
                case_label=self.label,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_skill_{m}.pdf",
                eval_region=self.ev.eval_name,
            )

    def plot_climate_acc(self):
        if not self.climate_references:
            return
        skill_plot(
            self.climate_references,
            case_label=self.label,
            metric="acc",
            save_path=self.plots_dir / f"{self.ev.var}_climate_acc.pdf",
            eval_region=self.ev.eval_name,
        )

    def plot_climate_activity(self):
        if not self.climate_references:
            return
        activity_plot(
            self.climate_references,
            case_label=self.label,
            save_path=self.plots_dir / f"{self.ev.var}_climate_activity.pdf",
            eval_region=self.ev.eval_name,
        )

    def plot_spatial_power_spectrum(self):
        if "pixel" in self.ev.forecast_ds.dims:
            return
        if self.ev.forecast_ds.sizes.get("lead_time", 1) > 1:
            return
        from deep_lst.evaluation.metrics import to_valid_time

        pred = to_valid_time(
            self.ev.forecast_ds[f"{self.ev.forecast_ds.var}_pred"].sel(lead_time=self.ev.lead_time),
            lead_time=self.ev.lead_time,
        )
        obs = to_valid_time(
            self.ev.forecast_ds[f"{self.ev.forecast_ds.var}_obs"].sel(lead_time=self.ev.lead_time),
            lead_time=self.ev.lead_time,
        )

        lats = self.ev.forecast_ds["lat"].values
        lons = self.ev.forecast_ds["lon"].values
        dx_km = _grid_spacing_km(lats, lons)
        spectra = {self.label: compute_spatial_power_spectrum(pred.transpose("valid_time", ...).values, dx_km=dx_km)}
        obs_spectrum = compute_spatial_power_spectrum(obs.transpose("valid_time", ...).values, dx_km=dx_km)
        spatial_power_spectrum_plot(
            pred_spectra_by_case=spectra,
            obs_spectrum=obs_spectrum,
            var=self.ev.var,
            save_path=self.plots_dir / f"{self.ev.var}_spatial_power_spectrum.pdf",
            eval_region=self.ev.eval_name,
        )

    def plot_init_lead_heatmap(self):
        if not self.climate_scores_raw:
            return
        for label, scores in self.climate_scores_raw.items():
            plot_init_vs_lead(
                scores,
                metric="acc",
                save_path=self.plots_dir / f"{self.ev.var}_init_lead_heatmap_acc.pdf",
            )
