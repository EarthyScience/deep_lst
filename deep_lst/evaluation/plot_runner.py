from pathlib import Path

import numpy as np
import xarray as xr

from deep_lst.evaluation.evaluator import EvaluatorDiagnostics, EvaluatorVerification
from deep_lst.evaluation.metrics import compute_scores
from deep_lst.evaluation.plots.diurnal import diurnal_delta_comparison, diurnal_line, diurnal_mean_plot
from deep_lst.evaluation.plots.error_growth import error_growth_daily_plot, error_growth_plot, skill_plot
from deep_lst.evaluation.plots.fingerprint import fingerprint_plot, improvement_da
from deep_lst.evaluation.plots.maps import acc_map, improvement_map_multi
from deep_lst.evaluation.plots.scores import error_overview_plot
from deep_lst.evaluation.plots.spectra import (
    _grid_spacing_km,
    compute_spatial_power_spectrum,
    compute_temporal_power_spectrum,
    spatial_power_spectrum_plot,
    temporal_power_spectrum_plot,
)
from deep_lst.evaluation.plots.timeseries import timeseries_plot


class PlotRunnerDiagnostics:
    def __init__(
        self,
        evaluator: EvaluatorDiagnostics,
        label: str,
        plots_dir: Path,
        baseline_evaluators: dict[str, EvaluatorDiagnostics] | None = None,
    ):
        self.ev = evaluator
        self.label = label
        self.plots_dir = Path(plots_dir)
        self.baselines = baseline_evaluators or {}  # {display_label: EvaluatorDiagnostics}

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
        summarised = self.ev.summarise(self.ev.grouped)
        for m in self.ev.metrics:
            diurnal_line(
                {self.label: summarised},
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_grouped_{m}.pdf",
                title=f"{self.ev.eval_name} — {m.upper()} ({self.ev.groupby})",
            )

    # --- benchmark (silently skip if no baselines) ---

    def plot_error_overview(self):
        if not self.baselines:
            return
        region_time = {self.label: compute_scores(self.ev.pred, self.ev.obs, self.ev.metrics, "region_time")}
        for label, bl_ev in self.baselines.items():
            bl = getattr(bl_ev, "baseline_aligned", None)
            if bl is not None:
                region_time[label] = compute_scores(bl, self.ev.obs, self.ev.metrics, "region_time")
        for m in self.ev.metrics:
            error_overview_plot(
                region_time,
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
        "anomaly_correlation": "ACC",
        "nash_sufcliffe_efficiency": "NSE",
    }

    def plot_score_map(self):
        if "pixel" in self.ev.forecast_ds.dims:
            return
        for m, label in self._SCORE_MAP_METRICS.items():
            score = compute_scores(self.ev.pred, self.ev.obs, [m], "region_space")[m]
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
        cases = {self.label: self.ev.summarise(self.ev.grouped)}
        cases.update(
            {
                label: bl_ev.summarise(bl_ev.baseline_grouped)
                for label, bl_ev in self.baselines.items()
                if getattr(bl_ev, "baseline_grouped", None) is not None
            }
        )
        for m in self.ev.metrics:
            diurnal_line(
                cases,
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
        # Eagerly compute scores once so that plot_error_growth / plot_skill_scores
        # don't each re-read the full zarr (201 lead_times × spatial grid is multi-GB).
        self.ev.scores = self.ev.scores.compute()
        requested = set(plots) if plots else None
        for name in sorted(dir(self)):
            if name.startswith("plot_"):
                if requested is None or name[5:] in requested:
                    getattr(self, name)()

    def plot_error_growth(self):
        ds = self.ev.summarise(self.ev.scores)
        for m in self.ev.metrics:
            error_growth_plot(
                ds,
                case_label=self.label,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_error_growth_{m}.pdf",
                eval_region=self.ev.eval_name,
            )

    def plot_error_growth_daily(self):
        ds = self.ev.summarise(self.ev.scores)
        for m in self.ev.metrics:
            error_growth_daily_plot(
                ds,
                case_label=self.label,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_error_growth_daily_{m}.pdf",
                eval_region=self.ev.eval_name,
            )

    def plot_skill_scores(self):
        references = {}
        for label, bl_ev in self.baselines.items():
            bl_scores = getattr(bl_ev, "baseline_scores", None)
            if bl_scores is None:
                continue
            # Force-compute baseline scores once per baseline (reads baseline zarr),
            # then divide against the already in-memory self.ev.scores.
            skill = self.ev.scores / bl_scores.compute()
            references[label] = self.ev.summarise(skill)
        if not references:
            return
        for m in self.ev.metrics:
            skill_plot(
                references,
                case_label=self.label,
                metric=m,
                save_path=self.plots_dir / f"{self.ev.var}_skill_{m}.pdf",
                eval_region=self.ev.eval_name,
            )

    def plot_spatial_power_spectrum(self):
        if "pixel" in self.ev.forecast_ds.dims:
            return
        if self.ev.forecast_ds.sizes.get("lead_time", 1) > 1:
            return
        from deep_lst.evaluation.metrics import to_valid_time

        pred, obs = to_valid_time(self.ev.forecast_ds, self.ev.var)
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
