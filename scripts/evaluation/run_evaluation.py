"""Per-experiment evaluation: compute metrics and generate all plots.

Always generates all plots. Benchmark plots are skipped automatically when
no --baselines are provided.

Two tiers of plots
------------------
Fast eval  — self-contained per experiment, per selected lead time:
    timeseries, diurnal_mean,
    forecast_gif, fingerprint,
    error_growth, error_growth_daily

Benchmarking — model compared against --baselines:
    error_overview, error_fingerprint, improvement_fingerprint,
    diurnal, diurnal_delta, diurnal_delta_error,
    power_spectrum, bias_maps, improvement_map

Comparison  — cross-experiment (handled by run_comparison.py)

Always saved
------------
  metrics_total.nc, metrics_region_time.nc, metrics_region_space.nc,
  metrics_diurnal.nc, metrics_diurnal_mean.nc, metrics_by_lead_time.nc

Usage
-----
# Single experiment, all baselines
uv run scripts/evaluation/run_evaluation.py \\
    --experiment mlp_rollout_L_KG_v1 \\
    --version version_1 \\
    --eval dahra \\
    --lead_time 3h \\
    --baselines era5 clim_hourly clim_smooth

# Skip benchmark plots (no baselines)
uv run scripts/evaluation/run_evaluation.py ... --baselines

# Specific temporal period
uv run scripts/evaluation/run_evaluation.py ... \\
    --temporal test_period_00 test_period_01

Available --baselines:
    era5          ERA5 skin temperature (assets/era5/baselines/)
    clim_hourly   SEVIRI hourly climatology (assets/climatology/hourly/)
    clim_smooth   SEVIRI smoothed climatology (assets/climatology/smoothed_hourly/)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.metrics import (
    compute_diurnal_mean,
    compute_diurnal_scores,
    compute_scores,
    compute_scores_by_lead_time,
    compute_validity_report,
    to_valid_time,
)
from deep_lst.evaluation.plots._colors import CASE_COLORS as _CASE_COLORS
from deep_lst.evaluation.plots._colors import DEFAULT_COLOR as _DEFAULT_COLOR
from deep_lst.evaluation.plots.animation import forecast_gif
from deep_lst.evaluation.plots.diurnal import (
    diurnal_delta_comparison,
    diurnal_delta_error_plot,
    diurnal_line,
    diurnal_mean_plot,
)
from deep_lst.evaluation.plots.fingerprint import abs_error_da, fingerprint_plot, improvement_da
from deep_lst.evaluation.plots.maps import bias_maps, improvement_map_multi
from deep_lst.evaluation.plots.scores import error_overview_plot
from deep_lst.evaluation.plots.spectra import (
    _grid_spacing_km,
    compute_spatial_power_spectrum,
    spatial_power_spectrum_plot,
)
from deep_lst.evaluation.plots.timeseries import timeseries_plot
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

# ---------------------------------------------------------------------------
# Plot catalogue
# ---------------------------------------------------------------------------

FAST_EVAL_PLOTS = [
    "timeseries",  # region-mean time series with ±1σ shading
    "diurnal_mean",  # mean LST by hour of day (pred vs obs)
    "forecast_gif",  # animated map: pred vs obs over full horizon
    "fingerprint",  # (DOY × hour) heatmap, spatially averaged
    "error_growth",  # RMSE vs lead_time line plot (hourly)
    "error_growth_daily",  # RMSE per day bucket (day 1 / 2 / 3)
]

BENCHMARK_PLOTS = [
    "error_overview",  # boxplot: model + each baseline (region_time RMSE)
    "error_fingerprint",  # fingerprint of |pred-obs| and |baseline-obs|
    "improvement_fingerprint",  # MAE improvement over each baseline
    "diurnal",  # diurnal RMSE/bias lines, model + all baselines
    "diurnal_delta",  # ΔT/h by hour: model + all baselines in one panel
    "diurnal_delta_error",  # ΔT/h error by hour: model + all baselines in one panel
    "power_spectrum",  # spatial power spectrum: model + obs + all baselines
    "bias_maps",  # spatial bias maps (model + baselines)
    "improvement_map",  # RMSE improvement: one multi-panel figure per model
]

ALL_PLOTS = FAST_EVAL_PLOTS + BENCHMARK_PLOTS

# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------

# name → (display_label, path_fn(eval_name), variable_name_in_file)
BASELINE_REGISTRY: dict[str, tuple[str, callable, str]] = {
    "era5": (
        "ERA5",
        lambda n: PATHS.baseline_path(n),
        "skt",
    ),
    "clim_hourly": (
        "hourly clim",
        lambda n: PATHS.climatology_path(n, "hourly"),
        "LST",
    ),
    "clim_smooth": (
        "smoothed clim",
        lambda n: PATHS.climatology_path(n, "smoothed_hourly"),
        "LST",
    ),
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment", required=True, help="Experiment name, e.g. mlp_rollout_L_KG_v1")
    p.add_argument("--version", required=True, help="Version folder, e.g. version_0")
    p.add_argument("--eval", required=True, help="Eval region name, e.g. dahra")
    p.add_argument("--variable", default="LST", help="Variable to evaluate (default: LST)")
    p.add_argument(
        "--lead_time",
        default="1h",
        help="Lead time for valid_time conversion and lead-time-specific plots (default: 1h)",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["rmse"],
        help="Metrics to compute: rmse bias mae r2 (default: rmse)",
    )
    p.add_argument(
        "--baselines",
        nargs="*",
        default=["era5"],
        metavar="NAME",
        help=(
            "Baselines to load for benchmark plots. "
            f"Available: {list(BASELINE_REGISTRY)}. "
            "Pass --baselines with no arguments to skip all benchmark plots. "
            "(default: era5)"
        ),
    )
    p.add_argument(
        "--temporal",
        nargs="+",
        default=None,
        metavar="PERIOD",
        help="Filter to specific temporal period(s), e.g. --temporal test_period_00 test_period_01",
    )
    p.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing metric files")
    p.add_argument(
        "--gif_init_time",
        default=None,
        help="init_time for forecast_gif, e.g. '2024-01-07T06:00'. Default: first init_time.",
    )
    p.add_argument(
        "--gif_lead_times",
        nargs="+",
        default=None,
        metavar="LT",
        help="Lead times for forecast_gif, e.g. --gif_lead_times 1h 2h 3h. Default: all.",
    )
    p.add_argument("--gif_fps", type=int, default=2, help="FPS for forecast_gif (default: 2)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_forecast(experiment: str, version: str, eval_name: str) -> xr.Dataset:
    path = PATHS.forecast_path(experiment, version, eval_name)
    if not path.exists():
        raise FileNotFoundError(f"Forecast not found: {path}\nRun scripts/inference/run_inference.py first.")
    return xr.open_dataset(path)


def _load_baselines(eval_name: str, baseline_names: list[str]) -> dict[str, xr.DataArray]:
    """Load baseline DataArrays (full valid_time range, not aligned to model).

    Returns
    -------
    dict[str, xr.DataArray]
        Keys are display labels (e.g. "ERA5"). Values are unaligned DataArrays
        with dim ``valid_time``. Align to model valid_time with
        ``.reindex(valid_time=pred["valid_time"])`` where needed.
    """
    result = {}
    if not baseline_names:
        return result
    for name in baseline_names:
        if name not in BASELINE_REGISTRY:
            print(f"  [warn] Unknown baseline: {name!r}. Available: {list(BASELINE_REGISTRY)}")
            continue
        label, path_fn, var_name = BASELINE_REGISTRY[name]
        path = path_fn(eval_name)
        if not path.exists():
            print(f"  [skip] {label}: not found at {path}")
            continue
        ds = xr.open_dataset(path)
        if var_name not in ds:
            print(f"  [skip] {label}: variable {var_name!r} not in {path.name}")
            continue
        da = ds[var_name]
        n_t = da.sizes.get("valid_time", 0)
        print(f"  [ok]   {label}: {n_t} timesteps  ({path.name})")
        result[label] = da
    return result


def _detect_case_label(experiment: str) -> str:
    if "rollout" in experiment.lower():
        return "MLP-Rollout"
    if "diag" in experiment.lower():
        return "MLP-Diag"
    return experiment


def _save_metrics(ds_metrics: xr.Dataset, path, overwrite: bool, tag: str = "") -> None:
    if path.exists() and not overwrite:
        label = f" ({tag})" if tag else ""
        print(f"  Metrics{label} already exist: {path}  (--overwrite to recompute)")
        return
    ds_metrics.to_netcdf(path)
    label = f" ({tag})" if tag else ""
    print(f"  Saved metrics{label}: {path}")


def _align_baseline(da: xr.DataArray, valid_times) -> xr.DataArray:
    """Reindex a baseline DataArray to match model valid_time (NaN for missing)."""
    return da.reindex(valid_time=valid_times)


# ---------------------------------------------------------------------------
# Error-growth plot helpers (inline — use pre-computed metrics_by_lead_time)
# ---------------------------------------------------------------------------

_PLOT_RC = {
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
}


def _plot_error_growth(
    ds_by_lead: xr.Dataset,
    case_label: str,
    metric: str,
    save_path,
    eval_region: str,
    baseline_rmse: dict[str, float] | None = None,
) -> None:
    """RMSE vs lead time — model line + flat baseline reference lines.

    Baselines are scored against the *same* observations the model sees at
    ``--lead_time`` (one RMSE value, independent of lead time). The model's
    RMSE grows because at longer lead times the forecast was launched from an
    earlier init_time and is further from the valid_time — the baseline has no
    such dependence, so it appears as a flat horizontal reference.
    """
    plt.rcParams.update(_PLOT_RC)
    lead_td = pd.TimedeltaIndex(ds_by_lead["lead_time"].values)
    lead_hours = [int(lt.total_seconds() / 3600) for lt in lead_td]
    vals = ds_by_lead[metric].values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    model_color = _CASE_COLORS.get(case_label, _DEFAULT_COLOR)
    ax.plot(lead_hours, vals, color=model_color, linewidth=2, marker="o", markersize=5, label=case_label)

    if baseline_rmse:
        for bl_label, bl_val in baseline_rmse.items():
            if bl_val is not None and np.isfinite(bl_val):
                c = _CASE_COLORS.get(bl_label, _DEFAULT_COLOR)
                ax.axhline(bl_val, color=c, linewidth=1.5, linestyle="--", label=f"{bl_label}  ({bl_val:.2f} K)")

    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"{metric.upper()} (K)")
    # Show at most ~12 tick labels to avoid overlap
    step = max(1, len(lead_hours) // 12)
    ax.set_xticks(lead_hours[::step])
    ax.set_title(f"{eval_region}  —  RMSE vs. lead time  (mean over test period)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def _plot_error_growth_daily(
    ds_by_lead: xr.Dataset,
    case_label: str,
    metric: str,
    save_path,
    eval_region: str,
    baseline_rmse: dict[str, float] | None = None,
) -> None:
    """Mean RMSE per day bucket (day 1: 1–24 h, day 2: 25–48 h, day 3: 49–72 h).

    Model values are averaged over the lead times in each bucket.
    Baselines are flat horizontal references (lead-time independent).
    """
    plt.rcParams.update(_PLOT_RC)
    lead_td = pd.TimedeltaIndex(ds_by_lead["lead_time"].values)
    lead_hours = np.array([int(lt.total_seconds() / 3600) for lt in lead_td])
    vals = ds_by_lead[metric].values

    DAY_BUCKETS = [(1, 24), (25, 48), (49, 72)]
    day_labels, day_means = [], []

    for day_idx, (lo, hi) in enumerate(DAY_BUCKETS, start=1):
        mask = (lead_hours >= lo) & (lead_hours <= hi)
        if not mask.any():
            continue
        day_labels.append(f"Day {day_idx}\n({lo}–{hi} h)")
        day_means.append(float(np.nanmean(vals[mask])))

    if not day_means:
        return

    x = np.arange(len(day_means))
    fig, ax = plt.subplots(figsize=(max(5, len(day_means) * 1.8), 4.5))

    model_color = _CASE_COLORS.get(case_label, _DEFAULT_COLOR)
    ax.plot(x, day_means, color=model_color, linewidth=2, marker="o", markersize=8, label=case_label)

    if baseline_rmse:
        for bl_label, bl_val in baseline_rmse.items():
            if bl_val is not None and np.isfinite(bl_val):
                c = _CASE_COLORS.get(bl_label, _DEFAULT_COLOR)
                ax.axhline(bl_val, color=c, linewidth=1.5, linestyle="--", label=f"{bl_label}  ({bl_val:.2f} K)")

    ax.set_xticks(x)
    ax.set_xticklabels(day_labels)
    ax.set_ylabel(f"{metric.upper()} (K)")
    ax.set_title(f"{eval_region}  —  RMSE vs. forecast day  (mean over test period)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_evaluation():
    args = parse_args()
    var = args.variable
    case_label = _detect_case_label(args.experiment)
    period_tag = "_".join(args.temporal) if args.temporal else "all"
    results_dir = PATHS.results_dir(args.experiment, args.version, args.eval)
    plots_dir = PATHS.results_plots_dir(args.experiment, args.version, args.eval) / period_tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment : {args.experiment} / {args.version}")
    print(f"Eval region: {args.eval}")
    print(f"Variable   : {var}    Lead time: {args.lead_time}")
    print(f"Case       : {case_label}")
    print(f"Baselines  : {args.baselines or '(none)'}")

    # -------------------------------------------------------------------------
    # Load forecast
    # -------------------------------------------------------------------------
    ds = _load_forecast(args.experiment, args.version, args.eval)

    # Filter to requested temporal period(s)
    if args.temporal is not None:
        origin = pd.Timestamp("2016-01-01")
        time_idx = np.concatenate(
            [np.load(PATHS.temporal_indices_path(p))["time_indices"].astype(np.int64) for p in args.temporal]
        )
        keep_times = origin + pd.to_timedelta(np.unique(time_idx), unit="h")
        ds = ds.sel(init_time=ds["init_time"].isin(keep_times))
        n_retained = ds.sizes["init_time"]
        print(f"Filtered to periods {args.temporal}: {n_retained} init_times retained")
        if n_retained == 0:
            print("No init_times matched the requested period(s). Check --temporal or re-run inference.")
            return

    print(
        f"\nForecast dims: init_time={ds.sizes['init_time']}, "
        f"lead_time={ds.sizes['lead_time']}, "
        f"lat={ds.sizes['lat']}, lon={ds.sizes['lon']}"
    )

    # -------------------------------------------------------------------------
    # Convert to valid_time at the selected lead time
    # -------------------------------------------------------------------------
    pred, obs = to_valid_time(ds, var, lead_time=args.lead_time)

    # Validity report
    report = compute_validity_report(pred, obs)
    print(f"\nValidity report ({var} @ lead_time={args.lead_time}):")
    for k, v in report.items():
        pct = 100 * v / report["n_total"] if report["n_total"] else 0
        print(f"  {k:<22}: {v:>8,}  ({pct:.1f}%)")

    # -------------------------------------------------------------------------
    # Load forecast into memory (avoids repeated disk reads inside lead-time loops)
    # -------------------------------------------------------------------------
    print("  Loading forecast into memory …")
    ds = ds.compute()

    # -------------------------------------------------------------------------
    # Compute and save metrics (always, all aggregations)
    # -------------------------------------------------------------------------
    print("\nComputing metrics...")

    for agg in ["total", "region_time", "region_space"]:
        scores = compute_scores(pred, obs, metrics=args.metrics, aggregation=agg)
        _save_metrics(scores, results_dir / f"metrics_{agg}.nc", args.overwrite, agg)
        if agg == "total":
            print("  Total scores:")
            for m in args.metrics:
                print(f"    {m.upper():<8}: {float(scores[m].values):.4f} K")

    # Diurnal scores and mean
    diurnal_ds = compute_diurnal_scores(pred, obs, metrics=args.metrics)
    _save_metrics(diurnal_ds, results_dir / "metrics_diurnal.nc", args.overwrite, "diurnal")

    diurnal_mean_ds = compute_diurnal_mean(pred, obs)
    _save_metrics(diurnal_mean_ds, results_dir / "metrics_diurnal_mean.nc", args.overwrite, "diurnal_mean")

    # metrics_by_lead_time (always computed if multi-lead forecast)
    ds_by_lead = None
    if ds.sizes.get("lead_time", 1) > 1:
        ds_by_lead = compute_scores_by_lead_time(ds, var=var, metrics=args.metrics)
        _save_metrics(ds_by_lead, results_dir / "metrics_by_lead_time.nc", args.overwrite, "by_lead_time")

    # -------------------------------------------------------------------------
    # Load baselines
    # -------------------------------------------------------------------------
    print("\nLoading baselines...")
    baselines: dict[str, xr.DataArray] = _load_baselines(args.eval, args.baselines or [])

    # Aligned versions (matched to model's valid_time) used for metrics comparisons
    baselines_aligned: dict[str, xr.DataArray] = {
        label: _align_baseline(da, pred["valid_time"]) for label, da in baselines.items()
    }

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    requested = set(ALL_PLOTS)
    print(f"\nGenerating plots → {plots_dir}")

    # =========================================================================
    # FAST EVAL
    # =========================================================================

    # a) Time series
    if "timeseries" in requested:
        timeseries_plot(
            ds,
            var=var,
            lead_time=args.lead_time,
            baselines=baselines_aligned or None,
            save_path=plots_dir / f"{var}_timeseries_lt{args.lead_time}.png",
            eval_region=args.eval,
            show_std=True,
            case_label=case_label,
        )

    # b) Diurnal mean
    if "diurnal_mean" in requested:
        # Use raw (unaligned) baselines for full 24 h coverage — aligned versions
        # are restricted to a single lead-time slice and may miss some hours.
        bl_hourly_means = {}
        for label, da in baselines.items():
            hours_da = da["valid_time"].dt.hour.values
            bl_hourly_means[label] = np.array(
                [float(np.nanmean(da.values[hours_da == h])) if (hours_da == h).any() else np.nan for h in range(24)]
            )
        diurnal_mean_plot(
            diurnal_mean_ds,
            var=var,
            baselines_mean=bl_hourly_means if bl_hourly_means else None,
            save_path=plots_dir / f"{var}_diurnal_mean_lt{args.lead_time}.png",
            eval_region=args.eval,
            case_label=case_label,
        )

    # c) Forecast GIF
    if "forecast_gif" in requested:
        init_tag = pd.Timestamp(args.gif_init_time).strftime("%Y%m%dT%H%M") if args.gif_init_time else "first"
        forecast_gif(
            ds,
            var=var,
            init_time=args.gif_init_time,
            lead_times=args.gif_lead_times,
            fps=args.gif_fps,
            save_path=plots_dir / f"{var}_forecast_{init_tag}.gif",
            eval_region=args.eval,
        )

    # f) Fingerprint — model pred + obs (+ baselines in same figure if available)
    if "fingerprint" in requested:
        fp_data = {f"{case_label} pred": pred, "SEVIRI obs": obs}
        fp_data.update({label: da for label, da in baselines.items()})
        fingerprint_plot(
            fp_data,
            title=f"{args.eval} — LST fingerprint  (lead_time={args.lead_time})",
            save_path=plots_dir / f"{var}_fingerprint_lt{args.lead_time}.png",
        )

    # g) Error growth plots — compute one baseline RMSE value per baseline.
    # Baselines live in valid_time space (no lead-time dependence), so a single
    # overall RMSE score serves as a flat horizontal reference on both plots.
    _need_bl_rmse = (
        ds_by_lead is not None
        and baselines_aligned
        and ("error_growth" in requested or "error_growth_daily" in requested)
    )
    if _need_bl_rmse:
        bl_rmse: dict[str, float] = {}
        for label, da in baselines_aligned.items():
            s = compute_scores(da, obs, metrics=["rmse"], aggregation="total")
            val = float(s["rmse"].values)
            bl_rmse[label] = val if np.isfinite(val) else None
    else:
        bl_rmse = {}

    if "error_growth" in requested and ds_by_lead is not None:
        _plot_error_growth(
            ds_by_lead,
            case_label=case_label,
            metric="rmse",
            save_path=plots_dir / f"{var}_error_growth.png",
            eval_region=args.eval,
            baseline_rmse=bl_rmse or None,
        )

    if "error_growth_daily" in requested and ds_by_lead is not None:
        _plot_error_growth_daily(
            ds_by_lead,
            case_label=case_label,
            metric="rmse",
            save_path=plots_dir / f"{var}_error_growth_daily.png",
            eval_region=args.eval,
            baseline_rmse=bl_rmse or None,
        )

    # =========================================================================
    # BENCHMARK (only when baselines are loaded)
    # =========================================================================

    if not baselines_aligned:
        if any(p in requested for p in BENCHMARK_PLOTS):
            print("\nNo baselines loaded — skipping benchmark plots.")
    else:
        # Scores for each baseline (region_time, for boxplot + cross-site eval)
        region_time_scores: dict[str, xr.Dataset] = {
            case_label: compute_scores(pred, obs, metrics=args.metrics, aggregation="region_time"),
        }
        for label, da in baselines_aligned.items():
            region_time_scores[label] = compute_scores(da, obs, metrics=args.metrics, aggregation="region_time")

        # Save baseline region_time metrics so run_across_sites.py can load them
        for label, ds_scores in region_time_scores.items():
            if label == case_label:
                continue  # already saved above
            safe_label = label.replace(" ", "_")
            _save_metrics(
                ds_scores,
                results_dir / f"metrics_region_time_{safe_label}.nc",
                args.overwrite,
                f"region_time/{label}",
            )

        # A) Error overview boxplot
        if "error_overview" in requested:
            for m in args.metrics:
                error_overview_plot(
                    region_time_scores,
                    metric=m,
                    save_path=plots_dir / f"{var}_error_overview_{m}_lt{args.lead_time}.png",
                    title=f"{args.eval} — {m.upper()} overview  (lead_time={args.lead_time})",
                    ylabel=f"{m.upper()} (K)",
                )

        # B) Error fingerprints — |pred-obs| and |baseline-obs|
        if "error_fingerprint" in requested:
            fp_err = {f"{case_label}  |error|": abs_error_da(pred, obs)}
            for label, da in baselines_aligned.items():
                fp_err[f"{label}  |error|"] = abs_error_da(da, obs)
            fingerprint_plot(
                fp_err,
                cmap="Reds",
                cbar_label="MAE (K)",
                title=f"{args.eval} — error fingerprint  (lead_time={args.lead_time})",
                save_path=plots_dir / f"{var}_error_fingerprint_lt{args.lead_time}.png",
            )

        # C) Improvement fingerprint — one panel per baseline
        if "improvement_fingerprint" in requested:
            for label, da in baselines_aligned.items():
                safe_label = label.replace("\n", "_").replace(" ", "_")
                fp_imp = {f"{case_label} vs {label}": improvement_da(pred, obs, da)}
                fingerprint_plot(
                    fp_imp,
                    cmap="RdBu_r",
                    vcenter=0.0,
                    cbar_label="ΔMAE (K, neg = model better)",
                    title=f"{args.eval} — improvement vs {label}  (lead_time={args.lead_time})",
                    save_path=plots_dir / f"{var}_improvement_fingerprint_{safe_label}_lt{args.lead_time}.png",
                )

        # D) Diurnal multi-case
        if "diurnal" in requested:
            diurnal_by_case = {case_label: diurnal_ds}
            for label, da in baselines_aligned.items():
                diurnal_by_case[label] = compute_diurnal_scores(da, obs, metrics=args.metrics)
            for m in args.metrics:
                diurnal_line(
                    diurnal_by_case,
                    metric=m,
                    save_path=plots_dir / f"{var}_diurnal_{m}_lt{args.lead_time}.png",
                    title=f"{args.eval} — {var} diurnal {m.upper()}  (lead_time={args.lead_time})",
                )

        # D2) Diurnal delta — ΔT/h by hour: model + all baselines
        if "diurnal_delta" in requested:
            bl_diurnal_mean_for_delta = {
                label: compute_diurnal_mean(da, obs) for label, da in baselines_aligned.items()
            }
            diurnal_delta_comparison(
                {case_label: diurnal_mean_ds, **bl_diurnal_mean_for_delta},
                var=var,
                save_path=plots_dir / f"{var}_diurnal_delta_lt{args.lead_time}.png",
                eval_region=args.eval,
            )

        # D3) Diurnal delta error — ΔT/h error for model + all baselines in one panel
        if "diurnal_delta_error" in requested:
            bl_diurnal_mean = {label: compute_diurnal_mean(da, obs) for label, da in baselines_aligned.items()}
            diurnal_delta_error_plot(
                model_diurnal_mean=diurnal_mean_ds,
                baselines_diurnal_mean=bl_diurnal_mean,
                var=var,
                save_path=plots_dir / f"{var}_diurnal_delta_error_lt{args.lead_time}.png",
                eval_region=args.eval,
                case_label=case_label,
            )

        # D4) Spatial power spectrum — model + obs + all baselines
        if "power_spectrum" in requested:
            lats = ds["lat"].values
            lons = ds["lon"].values
            dx_km = _grid_spacing_km(lats, lons)

            wav_pred, pwr_pred = compute_spatial_power_spectrum(pred.values, dx_km=dx_km)
            wav_obs, pwr_obs = compute_spatial_power_spectrum(obs.values, dx_km=dx_km)

            pred_spectra = {case_label: (wav_pred, pwr_pred)}
            for label, da in baselines_aligned.items():
                wav_bl, pwr_bl = compute_spatial_power_spectrum(da.values, dx_km=dx_km)
                pred_spectra[label] = (wav_bl, pwr_bl)

            spatial_power_spectrum_plot(
                pred_spectra_by_case=pred_spectra,
                obs_spectrum=(wav_obs, pwr_obs),
                var=var,
                save_path=plots_dir / f"{var}_power_spectrum_lt{args.lead_time}.png",
                eval_region=args.eval,
            )

        # E) Bias maps — model + each baseline
        if "bias_maps" in requested:
            bias_by_case = {case_label: compute_scores(pred, obs, metrics=["bias"], aggregation="region_space")["bias"]}
            for label, da in baselines_aligned.items():
                bias_by_case[label] = compute_scores(da, obs, metrics=["bias"], aggregation="region_space")["bias"]
            bias_maps(
                bias_by_case=bias_by_case,
                save_path=plots_dir / f"{var}_bias_maps_lt{args.lead_time}.png",
                eval_region=args.eval,
            )

        # F) Improvement map — all baselines in one multi-panel figure
        if "improvement_map" in requested:
            rmse_model = compute_scores(pred, obs, metrics=["rmse"], aggregation="region_space")["rmse"]
            rmse_by_baseline = {
                label: compute_scores(da, obs, metrics=["rmse"], aggregation="region_space")["rmse"]
                for label, da in baselines_aligned.items()
            }
            improvement_map_multi(
                rmse_model=rmse_model,
                rmse_by_baseline=rmse_by_baseline,
                save_path=plots_dir / f"{var}_improvement_map_lt{args.lead_time}.png",
                model_label=case_label,
                eval_region=args.eval,
            )

    ds.close()
    print("\nDone.")


if __name__ == "__main__":
    run_evaluation()
