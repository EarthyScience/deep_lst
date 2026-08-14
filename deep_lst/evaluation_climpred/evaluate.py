"""Evaluation entry point: derives all products from the saved verif/*.nc files and
renders every figure. No climpred, no slurm -- run directly after the three verify
jobs, or any time a derivation convention or plot changes.

Run: python -m deep_lst.evaluation_climpred.evaluate --experiment <exp> --eval_name <name>
"""

from deep_lst.utils.io import ExperimentPaths

from .common import build_parser, nearest_pixel, verification_dir
from .derive import derive_verif_products
from .masks import register_month_windows
from .plots import (
    COLORS,
    REGION_STYLES,
    SITE_STYLES,
    SPATIAL_BASELINE,
    plot_acc_r2,
    plot_acc_r2_regions,
    plot_bias_maps,
    plot_forecast_limit_map,
    plot_rmse_maps,
    plot_skill_maps,
    plot_skill_score,
)
from .tables import write_verif_tables
from .timeseries import plot_site_forecasts


def main():
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

    site_pixels = {name: nearest_pixel(pixel_daily, name) for name in SITE_STYLES}
    for window in skill_windows:
        plot_acc_r2(
            out["lead"],
            out["acc_centered"].sel(window=window),
            out["r2_anomaly"].sel(window=window),
            plots_dir,
            f"_{window}",
        )
        site_skill = pixel_daily["rmse_ss"].sel(window=window, baseline=SPATIAL_BASELINE)
        plot_skill_score(
            pooled["day"],
            pooled["rmse_ss"].sel(window=window),
            plots_dir,
            f"_daily_ci_{window}",
            xlabel="forecast day",
            ci=bootstrap_daily.sel(window=window),
            dispersion=pooled["rmse_ss_dispersion"].sel(window=window),
            sites={name: site_skill.isel(pixel=pixel) for name, pixel in site_pixels.items()},
        )
    regions = [w for w in REGION_STYLES if w in windows]
    if regions:
        plot_acc_r2_regions(out, plots_dir, regions)

    horizon = int(pooled["day"].values[-1])
    if horizon >= 2:  # a single forecast day leaves no gradient to map
        for window in map_windows:
            plot_forecast_limit_map(
                fc_limit.sel(window=window, baseline="clim_smooth"), horizon, plots_dir, f"_{window}_clim_smooth"
            )

    days = [1, horizon]  # nowcast and medium range
    for window in map_windows:
        plot_rmse_maps(pixel_daily["rmse_model"].sel(window=window), days, plots_dir, f"_{window}")
        plot_skill_maps(
            pixel_daily["rmse_ss"].sel(window=window, baseline="clim_smooth"),
            days,
            plots_dir,
            f"_{window}_clim_smooth",
            "RMSE-SS vs Climatology",
        )
        plot_bias_maps(pixel_daily["bias_model"].sel(window=window), days, plots_dir, f"_{window}")

    write_verif_tables(pooled, days, verification_dir(args, "tables"), list(COLORS))
    plot_site_forecasts(args, plots_dir)


if __name__ == "__main__":
    main()
