"""Per-experiment evaluation: compute metrics and generate all plots.

Always generates all plots. Benchmark plots are skipped automatically when
no --baselines are provided.

Two tiers of plots
------------------
Fast eval  — self-contained per experiment, per selected lead time:
    timeseries, diurnal_mean, fingerprint,
    error_growth, error_growth_daily

Benchmarking — model compared against --baselines:
    error_overview, error_fingerprint, improvement_fingerprint,
    diurnal, diurnal_delta, diurnal_delta_error,
    power_spectrum, acc_map, improvement_map

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
    persistence   SEVIRI obs at init_time (assets/seviri/baselines/); precompute with
                  scripts/benchmarks/extract_seviri_persistence.py
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.metrics import (
    compute_diurnal_mean,
    compute_diurnal_scores,
    compute_scores,
    compute_scores_by_lead_time,
    compute_skill_scores_by_lead_time,
    compute_validity_report,
    to_valid_time,
)
from deep_lst.evaluation.plots.diurnal import (
    diurnal_delta_comparison,
    diurnal_delta_error_plot,
    diurnal_line,
    diurnal_mean_plot,
)
from deep_lst.evaluation.plots.error_growth import (
    error_growth_daily_plot,
    error_growth_plot,
    error_growth_single_valid_time,
    skill_plot,
)
from deep_lst.evaluation.plots.fingerprint import abs_error_da, fingerprint_plot, improvement_da
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
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

# ---------------------------------------------------------------------------
# Plot catalogue
# ---------------------------------------------------------------------------

FAST_EVAL_PLOTS = [
    "timeseries",  # region-mean time series with ±1σ shading
    "diurnal_mean",  # mean LST by hour of day (pred vs obs)
    "fingerprint",  # (DOY × hour) heatmap, spatially averaged
    "error_growth",  # RMSE vs lead_time line plot (hourly)
    "error_growth_daily",  # RMSE per day bucket (day 1 / 2 / 3)
    "temporal_power_spectrum",  # PSD vs period (hours); works on sparse and grid
    "skill_plot",  # skill score vs lead time relative to persistence
]

BENCHMARK_PLOTS = [
    "error_overview",  # boxplot: model + each baseline (region_time RMSE)
    "error_fingerprint",  # fingerprint of |pred-obs| and |baseline-obs|
    "improvement_fingerprint",  # MAE improvement over each baseline
    "diurnal",  # diurnal RMSE/bias lines, model + all baselines
    "diurnal_delta",  # ΔT/h by hour: model + all baselines in one panel
    "diurnal_delta_error",  # ΔT/h error by hour: model + all baselines in one panel
    "power_spectrum",  # spatial power spectrum: model + obs + all baselines
    "acc_map",  # spatial ACC map (anomaly correlation coefficient)
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
    "persistence": (
        "persistence",
        lambda n: PATHS.persistence_path(n),
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
        default=None,
        metavar="NAME",
        help=(
            "Baselines to load for benchmark plots. "
            f"Available: {list(BASELINE_REGISTRY) + ['persistence']}. "
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
        "--plots",
        nargs="+",
        default=None,
        metavar="PLOT",
        help=(f"Subset of plots to generate. Available: {ALL_PLOTS}. Default: all plots."),
    )
    p.add_argument(
        "--target_valid_time",
        default=None,
        help="Valid time for error_growth_single_valid_time plot, e.g. '2024-01-10T12:00'. Default: None (skip this plot).",
    )
    p.add_argument(
        "--consistent_mask",
        action="store_true",
        help=(
            "Require obs and pred to be finite at ALL lead times for a given (init_time, pixel). "
            "Evaluates 'perfect conditions' only. Default: per-lead-time masking (all valid trajectories kept)."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_forecast(experiment: str, version: str, eval_name: str) -> xr.Dataset:
    path = PATHS.forecast_path(experiment, version, eval_name)
    if not path.exists():
        raise FileNotFoundError(f"Forecast not found: {path}\nRun scripts/inference/run_inference.py first.")
    return xr.open_zarr(path)


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


def _subset_pixels(da: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Subset da's pixel dim to match ref's lat/lon coordinates.

    No-op when pixel counts already match or lat coords are absent.
    Used when a baseline covers all store pixels but the forecast uses a subset.
    """
    if (
        "pixel" not in da.dims
        or "pixel" not in ref.dims
        or da.sizes["pixel"] == ref.sizes["pixel"]
        or "lat" not in da.coords
        or "lat" not in ref.coords
    ):
        return da
    bl_lats = da["lat"].values.astype(np.float32)
    bl_lons = da["lon"].values.astype(np.float32)
    lookup = {(float(la), float(lo)): i for i, (la, lo) in enumerate(zip(bl_lats, bl_lons))}
    p_lats = ref["lat"].values.astype(np.float32)
    p_lons = ref["lon"].values.astype(np.float32)
    idx = [lookup[(float(la), float(lo))] for la, lo in zip(p_lats, p_lons)]
    return da.isel(pixel=idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_evaluation():
    args = parse_args()
    if args.target_valid_time is not None:
        args.target_valid_time = pd.Timestamp(args.target_valid_time)
    var = args.variable
    case_label = _detect_case_label(args.experiment)
    period_tag = "_".join(args.temporal) if args.temporal else "all"
    condition_tag = "consistent_mask" if args.consistent_mask else ""
    results_dir = PATHS.results_dir(args.experiment, args.version, args.eval)
    if condition_tag:
        results_dir = results_dir / condition_tag
        results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PATHS.results_plots_dir(args.experiment, args.version, args.eval) / period_tag
    if condition_tag:
        plots_dir = plots_dir / condition_tag
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

    sparse = "pixel" in ds.dims
    spatial_str = f"pixel={ds.sizes['pixel']}" if sparse else f"lat={ds.sizes['lat']}, lon={ds.sizes['lon']}"
    print(f"\nForecast dims: init_time={ds.sizes['init_time']}, lead_time={ds.sizes['lead_time']}, {spatial_str}")

    # -------------------------------------------------------------------------
    # Optional consistent mask: keep only (init_time, pixel) where obs and pred
    # are finite at every lead time — replicates "perfect conditions" setting.
    # -------------------------------------------------------------------------
    if args.consistent_mask:
        pred_raw = ds[f"{var}_pred"]
        obs_raw = ds[f"{var}_obs"]
        all_finite = pred_raw.notnull().all("lead_time") & obs_raw.notnull().all("lead_time")
        ds = ds.assign(
            {
                f"{var}_pred": pred_raw.where(all_finite),
                f"{var}_obs": obs_raw.where(all_finite),
            }
        )
        print("  Consistent mask applied (perfect conditions only)")

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

    # Aligned versions (matched to model's valid_time) used for metrics comparisons.
    # Persistence needs a time-shift: obs(init_time) = obs(valid_time - lead_time),
    # so we reindex on shifted times then relabel to pred valid_times.
    lt_ns = np.timedelta64(int(pd.Timedelta(args.lead_time).total_seconds() * 1e9), "ns")
    pred_mask = np.isfinite(pred)
    baselines_aligned: dict[str, xr.DataArray] = {}
    for label, da in baselines.items():
        if label == "persistence":
            init_times = pred["valid_time"].values - lt_ns
            pers = da.reindex(valid_time=init_times)
            aligned = pers.assign_coords(valid_time=pred["valid_time"].values)
        else:
            aligned = _align_baseline(da, pred["valid_time"])
        aligned = _subset_pixels(aligned, pred)
        baselines_aligned[label] = aligned.where(pred_mask)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    requested = set(args.plots) if args.plots else set(ALL_PLOTS)
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
            # Use aligned version for persistence so the 6h shift is reflected in
            # the diurnal mean (obs(t-lead_time) averaged by hour(t)).  For all
            # other baselines the aligned and raw versions cover the same hours.
            src = baselines_aligned.get(label, da)
            hours_da = src["valid_time"].dt.hour.values
            bl_hourly_means[label] = np.array(
                [float(np.nanmean(src.values[hours_da == h])) if (hours_da == h).any() else np.nan for h in range(24)]
            )
        diurnal_mean_plot(
            diurnal_mean_ds,
            var=var,
            baselines_mean=bl_hourly_means if bl_hourly_means else None,
            save_path=plots_dir / f"{var}_diurnal_mean_lt{args.lead_time}.png",
            eval_region=args.eval,
            case_label=case_label,
        )

    # f) Fingerprint — model pred + obs (+ baselines in same figure if available)
    if "fingerprint" in requested:
        fp_data = {f"{case_label} pred": pred, "SEVIRI obs": obs}
        fp_data.update({label: da for label, da in baselines_aligned.items()})
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

    if "error_growth" in requested and args.target_valid_time is not None:
        bl_rmse: dict[str, float] = {}
        for label, da in baselines_aligned.items():
            s = compute_scores(da, obs, metrics=["rmse"], aggregation="region_time")
            # then select your specific valid time:
            bl_rmse[label] = float(s["rmse"].sel(valid_time=args.target_valid_time).values)
        error_growth_single_valid_time(
            ds_by_lead,
            case_label=case_label,
            metric="rmse",
            save_path=plots_dir / f"{var}_error_growth_valid_time.pdf",
            valid_time=args.target_valid_time,
            climatological_rmse=bl_rmse,
        )

    if "error_growth" in requested and ds_by_lead is not None:
        error_growth_plot(
            ds_by_lead,
            case_label=case_label,
            metric="rmse",
            save_path=plots_dir / f"{var}_error_growth.pdf",
            eval_region=args.eval,
        )

    if "error_growth_daily" in requested and ds_by_lead is not None:
        error_growth_daily_plot(
            ds_by_lead,
            case_label=case_label,
            metric="rmse",
            save_path=plots_dir / f"{var}_error_growth_daily.pdf",
            eval_region=args.eval,
        )

    # i) Skill score vs lead time — all baselines in one plot
    if "skill_plot" in requested and ds_by_lead is not None and baselines_aligned:
        references = {}
        for bl_label in baselines_aligned:
            ref_type = "persistence" if bl_label == "persistence" else "climatology"
            references[bl_label] = compute_skill_scores_by_lead_time(
                ds, _subset_pixels(baselines[bl_label], pred), ref_type=ref_type, var=var, metrics=["rmse"]
            )
        skill_plot(
            references,
            case_label=case_label,
            metric="rmse",
            save_path=plots_dir / f"{var}_skill.pdf",
            eval_region=args.eval,
        )

    # h) Temporal power spectrum — PSD of raw pred vs obs only.
    # Baselines are excluded: ERA5 has no cloud NaNs, persistence samples
    # cloud-free init_times, climatologies have full coverage — all produce
    # artificially inflated spectra compared to SEVIRI obs (which has cloud gaps).
    # A fair spectral comparison requires the same valid/NaN mask.
    if "temporal_power_spectrum" in requested:
        diffs = np.diff(pred["valid_time"].values).astype("timedelta64[h]").astype(float)
        dt_hours = float(np.median(diffs[diffs > 0])) if len(diffs) > 0 else 1.0

        spectra_t = {
            case_label: compute_temporal_power_spectrum(pred.transpose("valid_time", ...).values, dt_hours=dt_hours)
        }
        obs_spectrum_t = compute_temporal_power_spectrum(obs.transpose("valid_time", ...).values, dt_hours=dt_hours)

        temporal_power_spectrum_plot(
            pred_spectra_by_case=spectra_t,
            obs_spectrum=obs_spectrum_t,
            var=var,
            save_path=plots_dir / f"{var}_temporal_power_spectrum.pdf",
            eval_region=args.eval,
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

        # Save spatial ACC for cross-site evaluation (needs hourly climatology)
        if "hourly clim" in baselines_aligned:
            clim = baselines_aligned["hourly clim"]
            acc_ds = compute_scores(pred - clim, obs - clim, metrics=["correlation"], aggregation="region_space")
            _save_metrics(acc_ds, results_dir / "metrics_region_space_acc.nc", args.overwrite, "region_space_acc")

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
                save_path=plots_dir / f"{var}_error_fingerprint_lt{args.lead_time}.pdf",
            )

        # C) Improvement fingerprint — one panel per baseline
        if "improvement_fingerprint" in requested:
            for label, da in baselines_aligned.items():
                safe_label = label.replace("\n", "_").replace(" ", "_")
                fp_imp = {f"{case_label} vs {label}": improvement_da(pred, obs, da)}
                fingerprint_plot(
                    fp_imp,
                    vcenter=0.0,
                    cbar_label="ΔMAE (K)  [low = model better, high = baseline better]",
                    title=f"{args.eval} — improvement vs {label}  (lead_time={args.lead_time})",
                    save_path=plots_dir / f"{var}_improvement_fingerprint_{safe_label}_lt{args.lead_time}.pdf",
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
                    save_path=plots_dir / f"{var}_diurnal_{m}_lt{args.lead_time}.pdf",
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
                save_path=plots_dir / f"{var}_diurnal_delta_lt{args.lead_time}.pdf",
                eval_region=args.eval,
            )

        # D3) Diurnal delta error — ΔT/h error for model + all baselines in one panel
        if "diurnal_delta_error" in requested:
            bl_diurnal_mean = {label: compute_diurnal_mean(da, obs) for label, da in baselines_aligned.items()}
            diurnal_delta_error_plot(
                model_diurnal_mean=diurnal_mean_ds,
                baselines_diurnal_mean=bl_diurnal_mean,
                var=var,
                save_path=plots_dir / f"{var}_diurnal_delta_error_lt{args.lead_time}.pdf",
                eval_region=args.eval,
                case_label=case_label,
                lead_time=args.lead_time,
            )

        # D4) Spatial power spectrum — PSD of raw pred vs obs only.
        # Same reasoning as temporal: baselines have different NaN structures
        # (ERA5: none; persistence: cloud-free init_times; clim: full coverage)
        # making spectral comparison against SEVIRI obs unfair.
        if "power_spectrum" in requested and not sparse:
            lats = ds["lat"].values
            lons = ds["lon"].values
            dx_km = _grid_spacing_km(lats, lons)

            spectra_s = {
                case_label: compute_spatial_power_spectrum(pred.transpose("valid_time", ...).values, dx_km=dx_km)
            }
            obs_spectrum_s = compute_spatial_power_spectrum(obs.transpose("valid_time", ...).values, dx_km=dx_km)

            spatial_power_spectrum_plot(
                pred_spectra_by_case=spectra_s,
                obs_spectrum=obs_spectrum_s,
                var=var,
                save_path=plots_dir / f"{var}_power_spectrum_lt{args.lead_time}.pdf",
                eval_region=args.eval,
            )

        # E) ACC map — anomaly correlation coefficient per pixel
        if "acc_map" in requested and not sparse:
            clim_path = PATHS.climatology_path(args.eval, "hourly")
            if clim_path.exists():
                clim_da = xr.open_dataset(clim_path)[var]
                clim_aligned = _align_baseline(clim_da, pred["valid_time"])
                acc = compute_scores(
                    pred - clim_aligned,
                    obs - clim_aligned,
                    metrics=["correlation"],
                    aggregation="region_space",
                )["correlation"]
                acc_map(
                    acc=acc,
                    save_path=plots_dir / f"{var}_acc_map_lt{args.lead_time}.pdf",
                    eval_region=args.eval,
                )
            else:
                print(f"  [skip] acc_map: hourly climatology not found at {clim_path}")

        # F) Improvement map — all baselines in one multi-panel figure
        if "improvement_map" in requested and not sparse:
            rmse_model = compute_scores(pred, obs, metrics=["rmse"], aggregation="region_space")["rmse"]
            rmse_by_baseline = {
                label: compute_scores(da, obs, metrics=["rmse"], aggregation="region_space")["rmse"]
                for label, da in baselines_aligned.items()
            }
            improvement_map_multi(
                rmse_model=rmse_model,
                rmse_by_baseline=rmse_by_baseline,
                save_path=plots_dir / f"{var}_improvement_map_lt{args.lead_time}.pdf",
                model_label=case_label,
                eval_region=args.eval,
            )

    ds.close()
    print("\nDone.")


if __name__ == "__main__":
    run_evaluation()
