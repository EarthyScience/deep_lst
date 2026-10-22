"""Evaluation entry point: derives all products from the saved verif/*.nc files and
renders every figure. No climpred, no slurm -- run directly after the three verify
jobs, or any time a derivation convention or plot changes.

Run: python -m deep_lst.evaluation_climpred.evaluate --experiment <exp> --eval_name <name>
"""

import re

import xarray as xr

from deep_lst.utils.io import ExperimentPaths

from .common import build_parser, verification_dir
from .cubes import pixel_attributes
from .derive import derive_verif_products, diurnal_scores
from .gifs import month_fields, monthly_verif_path, render_standard_gifs
from .masks import register_month_windows
from .plots import (
    COLORS,
    DIURNAL_EFFICIENCY_ROWS,
    DIURNAL_ERROR_ROWS,
    REGION_STYLES,
    apply_print_style,
    plot_acc_r2,
    plot_acc_r2_regions,
    plot_amplitude_error,
    plot_diurnal_cycles,
    plot_error_and_skill,
    plot_error_skill_anomaly,
    plot_forecast_limit_map,
    plot_fvc_decile_curves,
    plot_fvc_pixel_scatter,
    plot_pixel_bias_planes,
    plot_pixel_maps,
    plot_reference_scatter,
    plot_regime_summary,
    plot_score_bars,
    plot_skill_bars,
    plot_skill_limit_maps,
    plot_skill_summary,
)
from .tables import _row_groups, write_verif_tables
from .timeseries import plot_example_forecasts, plot_regime_forecasts, plot_site_forecasts

# The dense case studies share one colour range per map kind so their maps compare
# across regions (set by hand: 98th percentiles of |bias| span 0.7-2.5 K and of
# |skill| 0.35-0.8 across the three at stride 7). The continental evaluation keeps
# its data-driven range. Stride suffixes (_s7) name reruns of the same region.
CASE_STUDIES = ("sahel_niger", "yangambi", "zambia_drought")
CASE_STUDY_LIMITS = {"bias": 2.0, "skill": 0.8}
# Windows of the hourly error/skill curves: the total and the two solar peaks. Drawn for
# every verification (the plain windows), unlike the regime summaries.
LEAD_WINDOWS = ("all", "day", "night")
# Per-pixel efficiency maps next to the RMSE, skill and bias maps, drawn where the verif
# saved the per-pixel sums (pixel_daily <variable>_model).
EFFICIENCY_MAPS = ("nse", "acc", "nrmse")


def main():
    apply_print_style()
    parser = build_parser()
    parser.add_argument(
        "--skill_windows",
        nargs="+",
        default=["all", "kg_arid", "kg_tropical", "kg_temperate"],
        help="Windows for the skill/ACC curves; reduced to the verified ones.",
    )
    parser.add_argument(
        "--map_windows",
        nargs="+",
        default=["all", "day", "night"],
        help="Windows for the pixel maps; reduced to the verified ones.",
    )
    parser.add_argument(
        "--bias_plane_days",
        nargs=2,
        type=int,
        default=[1, 10],
        metavar=("FIRST", "SECOND"),
        help="The two forecast days overlaid in the per-pixel bias planes (need the day and night windows).",
    )
    args = parser.parse_args()
    month_windows = register_month_windows(args.months)
    if month_windows:
        args.skill_windows = args.skill_windows + [month_windows[0]]
        args.map_windows = args.map_windows + month_windows
    args.paths = ExperimentPaths()

    out, pooled, fc_limit, pixel_daily = derive_verif_products(args)
    plots_dir = verification_dir(args, "plots")

    windows = [str(window) for window in out["window"].values]
    skill_windows = [w for w in args.skill_windows if w in windows]
    map_windows = [w for w in args.map_windows if w in windows]
    skipped = sorted(set(args.skill_windows + args.map_windows) - set(windows))
    if skipped:
        print(f"Skipping unverified windows: {skipped}")

    for window in skill_windows:
        plot_acc_r2(
            out["lead"],
            out["acc_centered"].sel(window=window),
            out["r2_anomaly"].sel(window=window),
            plots_dir,
            f"_{window}",
        )
    for window in [w for w in LEAD_WINDOWS if w in windows]:
        plot_error_and_skill(
            out["lead"],
            out["rmse_model"].sel(window=window),
            out["rmse_baseline"].sel(window=window),
            out["rmse_ss"].sel(window=window),
            plots_dir,
            f"_{window}",
            dispersion=out["rmse_ss_dispersion"].sel(window=window),
            rmse_dispersion_model=out["rmse_model_dispersion"].sel(window=window),
            rmse_dispersion_baseline=out["rmse_baseline_dispersion"].sel(window=window),
        )
    regions = [w for w in REGION_STYLES if w in windows]
    if regions:
        plot_acc_r2_regions(out, plots_dir, regions)
    if regions and "acc" in pooled and "all" in windows:
        plot_error_skill_anomaly(pooled, "all", regions, plots_dir)

    horizon = int(pooled["day"].values[-1])
    if horizon >= 2:  # a single forecast day leaves no gradient to map
        for window in map_windows:
            plot_forecast_limit_map(
                fc_limit.sel(window=window, baseline="clim_smooth"), horizon, plots_dir, f"_{window}_clim_smooth"
            )

    days = [1, horizon]  # nowcast and medium range
    region = re.sub(r"_s\d+$", "", args.eval_name)
    limits = CASE_STUDY_LIMITS if region in CASE_STUDIES else {}
    for window in map_windows:
        per_day = window == "all"  # one map per forecast day for the total skill and bias only
        plot_pixel_maps(pixel_daily["rmse_model"].sel(window=window), days, plots_dir, "rmse", f"_{window}")
        plot_pixel_maps(
            pixel_daily["rmse_ss"].sel(window=window, baseline="clim_smooth"),
            days,
            plots_dir,
            "skill",
            f"_{window}_clim_smooth",
            label="Skill improvement (RMSE)",
            per_day=per_day,
            limit=limits.get("skill"),
        )
        plot_pixel_maps(
            pixel_daily["bias_model"].sel(window=window),
            days,
            plots_dir,
            "bias",
            f"_{window}",
            per_day=per_day,
            limit=limits.get("bias"),
        )
        for variable in EFFICIENCY_MAPS:
            if f"{variable}_model" in pixel_daily:
                field = pixel_daily[f"{variable}_model"].sel(window=window)
                plot_pixel_maps(field, days, plots_dir, variable, f"_{window}")
        if horizon >= 2:
            plot_skill_limit_maps(
                pixel_daily["rmse_ss"].sel(window=window, baseline="clim_smooth"),
                fc_limit.sel(window=window, baseline="clim_smooth"),
                days,
                horizon,
                plots_dir,
                f"_{window}_clim_smooth",
                limit=limits.get("skill"),
            )

    if "day" in map_windows and "night" in map_windows:
        by_pixel = xr.open_dataset(verification_dir(args, "verif") / "by_pixel.nc", decode_timedelta=False)
        if max(args.bias_plane_days) > horizon:
            print(f"Skipping the bias planes: days {args.bias_plane_days} exceed the {horizon}-day horizon")
        else:
            plot_pixel_bias_planes(pixel_daily, by_pixel["n"], args.bias_plane_days, plots_dir)
    else:
        print("Skipping the pixel bias planes: both the 'day' and 'night' windows must be verified and mapped")

    # Reference scatter: the climatology's per-pixel RMSE, bias and NSE against the model's
    # (needs the per-pixel sums); the month column needs the monthly verif on top.
    if "nse_baseline" in pixel_daily:
        month_set = month_fields(args) if monthly_verif_path(args).exists() else None
        plot_reference_scatter(pixel_daily, month_set, days, plots_dir)

    # Diurnal figures: the error cycle fanned by forecast day and the amplitude error, per
    # regime where verified; need verif/diurnal.nc (verif_diurnal, no climpred).
    diurnal_path = verification_dir(args, "verif") / "diurnal.nc"
    if diurnal_path.exists():
        diurnal = xr.open_dataset(diurnal_path, decode_timedelta=False)
        scores = diurnal_scores(diurnal, ["all"] + regions)
        scores.to_netcdf(verification_dir(args, "scores") / "diurnal.nc")
        plot_diurnal_cycles(scores, DIURNAL_ERROR_ROWS, plots_dir, "diurnal_cycle_error.pdf")
        plot_diurnal_cycles(scores, DIURNAL_EFFICIENCY_ROWS, plots_dir, "diurnal_cycle_efficiency.pdf")
        plot_amplitude_error(scores, plots_dir)
    else:
        print(f"Skipping the diurnal figures: no {diurnal_path} (verif_diurnal)")

    write_verif_tables(pooled, days, verification_dir(args, "tables"), list(COLORS))
    groups = _row_groups(pooled)
    # Bars and summaries draw the regime groups where verified, else the "All" group.
    plot_score_bars(pooled, days, groups, plots_dir)
    plot_skill_bars(pooled, days, groups, plots_dir)
    plot_regime_summary(out, pooled, groups, plots_dir, days=days)
    if "day" in windows and "night" in windows:
        plot_skill_summary(out, pooled, groups, plots_dir, days=days)

    # FVC conditioning: the decile curves need the fvc_d<k> windows (holdout-sized runs only);
    # the pixel scatter needs just the day/night windows plus the per-pixel FVC from the npz.
    if any(window.startswith("fvc_d") for window in windows):
        plot_fvc_decile_curves(pooled, days, plots_dir)
    attributes = pixel_attributes(args.paths.spatial_indices_path(args.paths.baseline_spatial(args.eval_name)))
    if attributes is not None and "day" in windows and "night" in windows:
        plot_fvc_pixel_scatter(pixel_daily, attributes, days, plots_dir, variable="rmse_ss")
        plot_fvc_pixel_scatter(pixel_daily, attributes, days, plots_dir, variable="rmse_model")
        if "lst_validity" in attributes:  # same scatter against the observational validity
            plot_fvc_pixel_scatter(
                pixel_daily, attributes, days, plots_dir, variable="rmse_ss", x_attribute="lst_validity"
            )
    plot_site_forecasts(args, plots_dir)
    plot_example_forecasts(args, plots_dir)  # one good and one bad pixel of this eval set, any selection
    if regions:  # one example pixel per KG regime: needs the regime windows (train domain)
        plot_regime_forecasts(args, plots_dir)
    # Animated maps: the bias by forecast day always; the month pairs where verif_pixel
    # --monthly has run (gifs.py).
    render_standard_gifs(args, pixel_daily, plots_dir)


if __name__ == "__main__":
    main()
