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
from .derive import derive_verif_products
from .gifs import render_standard_gifs
from .masks import register_month_windows
from .plots import (
    COLORS,
    REGION_STYLES,
    apply_print_style,
    plot_acc_r2,
    plot_acc_r2_regions,
    plot_bias_maps,
    plot_error_and_skill,
    plot_error_skill_anomaly,
    plot_forecast_limit_map,
    plot_fvc_decile_curves,
    plot_fvc_pixel_scatter,
    plot_pixel_bias_planes,
    plot_regime_summary,
    plot_rmse_maps,
    plot_score_bars,
    plot_skill_bars,
    plot_skill_limit_maps,
    plot_skill_maps,
    plot_skill_summary,
)
from .tables import _row_groups, write_verif_tables
from .timeseries import plot_regime_forecasts, plot_site_forecasts

# The dense case studies share one colour range per map kind so their maps compare
# across regions (set by hand: 98th percentiles of |bias| span 0.7-2.5 K and of
# |skill| 0.35-0.8 across the three at stride 7). The continental evaluation keeps
# its data-driven range. Stride suffixes (_s7) name reruns of the same region.
CASE_STUDIES = ("sahel_niger", "yangambi", "zambia_drought")
CASE_STUDY_LIMITS = {"bias": 2.0, "skill": 0.8}


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
    parser.add_argument(
        "--gifs",
        action="store_true",
        help="Also render the animated maps (gifs.py standard set; training sites only, the month set needs verif_pixel --monthly).",
    )
    args = parser.parse_args()
    month_windows = register_month_windows(args.months)
    if month_windows:
        args.skill_windows = args.skill_windows + [month_windows[0]]
        args.map_windows = args.map_windows + month_windows
    args.paths = ExperimentPaths()

    out, pooled, bootstrap_hourly, bootstrap_daily, fc_limit, pixel_daily = derive_verif_products(args)
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
        plot_error_and_skill(
            pooled["day"],
            pooled["rmse_model"].sel(window=window),
            pooled["rmse_baseline"].sel(window=window),
            pooled["rmse_ss"].sel(window=window),
            plots_dir,
            f"_daily_ci_{window}",
            xlabel="Lead time [day]",
            ci=bootstrap_daily.sel(window=window),
            dispersion=pooled["rmse_ss_dispersion"].sel(window=window),
            rmse_dispersion_model=pooled["rmse_model_dispersion"].sel(window=window),
            rmse_dispersion_baseline=pooled["rmse_baseline_dispersion"].sel(window=window),
        )
    regions = [w for w in REGION_STYLES if w in windows]
    if regions:
        plot_acc_r2_regions(out, plots_dir, regions)
    if regions and "acc" in pooled and "all" in windows:
        plot_error_skill_anomaly(pooled, "all", regions, bootstrap_daily, plots_dir)

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
        plot_rmse_maps(pixel_daily["rmse_model"].sel(window=window), days, plots_dir, f"_{window}")
        plot_skill_maps(
            pixel_daily["rmse_ss"].sel(window=window, baseline="clim_smooth"),
            days,
            plots_dir,
            f"_{window}_clim_smooth",
            "Skill improvement (RMSE)",
            per_day=per_day,
            limit=limits.get("skill"),
        )
        plot_bias_maps(
            pixel_daily["bias_model"].sel(window=window),
            days,
            plots_dir,
            f"_{window}",
            per_day=per_day,
            limit=limits.get("bias"),
        )
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

    write_verif_tables(pooled, days, verification_dir(args, "tables"), list(COLORS))
    groups = _row_groups(pooled)
    plot_score_bars(pooled, days, groups, plots_dir)
    plot_skill_bars(pooled, days, groups, plots_dir)
    if regions:
        plot_regime_summary(out, pooled, groups, plots_dir, days=days)
    if regions and "day" in windows and "night" in windows:
        plot_skill_summary(out, pooled, bootstrap_hourly, bootstrap_daily, groups, plots_dir, days=days)

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
    if regions:  # one example pixel per KG regime: needs the regime windows (train domain)
        plot_regime_forecasts(args, plots_dir)
    if args.gifs:
        render_standard_gifs(args, pixel_daily, plots_dir)


if __name__ == "__main__":
    main()
