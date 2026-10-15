"""Animated per-pixel maps for presentations.

Two layouts (--layout): `single` is one map stepping through the forecast days
(--over day) or through the months at one forecast day (--over month, --day); `pair`
is the day-1 and day-10 maps side by side stepping through the months. Both show the
model RMSE or its RMSE skill score against climatology or persistence (--variable,
--baseline). Meant for the training sites: the holdouts are too sparse to read as maps.

The day frames come from scores/pixel_daily.nc (window 'all', written by evaluate).
The month frames need verif/by_pixel_monthly.nc from

    python -m deep_lst.evaluation_climpred.verif_pixel --experiment <exp> --eval_name <name> --windows all --monthly

(thirteen stacked windows: ~450G on train_mtg_new; run_climpred.sh submits it for the
training sites), pooled into forecast days here exactly as derive.py pools by_pixel.nc.
The month is the valid-time month, as for the JFMA windows, so the January frame also
holds the first January days of the following year from the late-December inits.

One colour scale per gif over every frame (98th percentile, as the static maps), so
the colours stay comparable while the animation runs.

`evaluate --gifs` renders the standard set (render_standard_gifs). This entry point
renders one gif of any combination:
    python -m deep_lst.evaluation_climpred.gifs --experiment <exp> --layout single --over month --variable rmse --day 10
"""

import argparse
import io

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize
from PIL import Image

from deep_lst.utils.io import ExperimentPaths

from .derive import pooled_mean
from .masks import MONTHLY_WINDOWS, WINDOW_LABELS
from .plots import BASELINE_LABELS, _diverging_scale, _map_ax, _map_extent

PAIR_DAYS = (1, 10)  # nowcast and medium range, as the static maps
# The standard set: (variable, baseline) each as a single map over the days and, with
# the month windows verified, as the pair and the day-1 single map over the months.
STANDARD_VARIABLES = (("rmse", None), ("skill", "clim_smooth"), ("skill", "persistence"))
STANDARD_MONTH_DAY = 1
# Presentation sizes: smaller than PRINT_RC so the panels get the space on a slide.
GIF_RC = {
    "font.size": 11,
    "axes.titlesize": 14,
    "figure.titlesize": 16,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
PANEL_INCHES = 7.0
DPI = 110


def build_parser():
    parser = argparse.ArgumentParser(description="Animated per-pixel RMSE / skill maps (gif).")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name (results/<experiment>/...).")
    parser.add_argument("--version", type=str, default="version_0", help="Experiment version.")
    parser.add_argument("--eval_name", type=str, default="train_mtg_new", help="Eval name (results subfolder).")
    parser.add_argument(
        "--layout", choices=["single", "pair"], default="pair", help="One map, or day 1 next to day 10."
    )
    parser.add_argument(
        "--over",
        choices=["month", "day"],
        default="month",
        help="What the frames step through (single only; pair always steps through the months).",
    )
    parser.add_argument(
        "--variable",
        choices=["rmse", "skill"],
        default="skill",
        help="RMSE of the model, or its RMSE skill score against --baseline.",
    )
    parser.add_argument(
        "--baseline",
        choices=["clim_smooth", "persistence"],
        default="clim_smooth",
        help="Reference of the skill score (ignored for rmse).",
    )
    parser.add_argument("--day", type=int, default=1, help="Forecast day of a single map stepping through the months.")
    parser.add_argument("--seconds", type=float, default=1.0, help="Seconds per frame.")
    return parser


def day_fields(args):
    """Per-pixel scores by forecast day for the total window: scores/pixel_daily.nc."""
    scores_dir = args.paths.verification_dir(args.experiment, args.version, args.eval_name, "scores")
    pixel_daily = xr.open_dataset(scores_dir / "pixel_daily.nc", decode_timedelta=False)
    return pixel_daily.sel(window="all")


def monthly_verif_path(args):
    verif_dir = args.paths.verification_dir(args.experiment, args.version, args.eval_name, "verif")
    return verif_dir / "by_pixel_monthly.nc"


def month_fields(args):
    """Per-pixel scores by forecast day and month: by_pixel_monthly.nc pooled over the
    24 leads of a day as derive.py does (count-weighted in the mse domain, sqrt after)."""
    by_pixel = xr.open_dataset(monthly_verif_path(args), decode_timedelta=False)
    by_pixel = by_pixel.sel(window=MONTHLY_WINDOWS)
    day = xr.DataArray((by_pixel["lead"].values - 1) // 24 + 1, dims="lead", name="day")
    rmse_model = np.sqrt(pooled_mean(by_pixel["mse_model"], by_pixel["n"], dim="lead", group=day))
    rmse_baseline = np.sqrt(pooled_mean(by_pixel["mse_baseline"], by_pixel["n"], dim="lead", group=day))
    return xr.Dataset({"rmse_model": rmse_model, "rmse_ss": 1.0 - rmse_model / rmse_baseline})


def init_year(args):
    """Year of the forecast init times, for the month titles."""
    store = xr.open_zarr(args.paths.forecast_path(args.experiment, args.version, args.eval_name))
    return int(store["init_time"].dt.year.min())


def select_field(fields, variable, baseline):
    """The drawn variable and its colorbar label."""
    if variable == "rmse":
        return fields["rmse_model"], "RMSE [K]"
    return fields["rmse_ss"].sel(baseline=baseline), f"Skill improvement (RMSE) vs {BASELINE_LABELS[baseline]}"


def animated_field(field, layout, over, day):
    """The part of the field the frames draw; the colour scale is computed over it."""
    if layout == "pair":
        return field.sel(day=list(PAIR_DAYS))
    if over == "month":
        return field.sel(day=day)
    return field


def colour_scale(field, variable):
    """One scale for every frame: viridis from zero for the RMSE, the diverging skill
    scale of the static maps; both capped at the 98th percentile over all frames."""
    limit = float(abs(field).quantile(0.98))
    if variable == "rmse":
        return plt.get_cmap("viridis"), Normalize(0.0, limit), "max"
    cmap, norm = _diverging_scale(limit)
    return cmap, norm, "both"


def frame_specs(field, layout, over, day, year):
    """One (figure title, [(panel title, pixel values), ...]) per frame."""
    frames = []
    if layout == "pair":
        for window in MONTHLY_WINDOWS:
            panels = [(f"Day {pair_day}", field.sel(window=window, day=pair_day)) for pair_day in PAIR_DAYS]
            frames.append((f"{WINDOW_LABELS[window]} {year}", panels))
    elif over == "day":
        for frame_day in field["day"].values:
            frames.append((f"Forecast day {int(frame_day)}", [(None, field.sel(day=frame_day))]))
    else:
        for window in MONTHLY_WINDOWS:
            title = f"{WINDOW_LABELS[window]} {year}, forecast day {day}"
            frames.append((title, [(None, field.sel(window=window))]))
    return frames


def render_frame(title, panels, extent, cmap, norm, extend, label):
    """One frame as an image. Fixed margins, no tight bbox: every frame must have the
    same pixel size and panel placement. Latitude labels on the first panel only."""
    n_panels = len(panels)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(PANEL_INCHES * n_panels, PANEL_INCHES + 0.8),
        dpi=DPI,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes_list = list(np.atleast_1d(axes))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.08, wspace=0.04)
    for index, (ax, (panel_title, values)) in enumerate(zip(axes_list, panels)):
        label_sides = ("left", "bottom") if index == 0 else ("bottom",)
        _map_ax(ax, extent, label_sides=label_sides)
        sc = ax.scatter(values["lon"], values["lat"], c=values, s=8, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        if panel_title:
            ax.set_title(panel_title)
    fig.suptitle(title)
    # A thin bar (fraction) close to the maps (pad): both are taken from the panel
    # height, and the maps are height-limited at this aspect.
    fig.colorbar(
        sc,
        ax=axes_list,
        orientation="horizontal",
        fraction=0.04,
        shrink=0.5,
        pad=0.06,
        aspect=40,
        extend=extend,
        format="%.2f",
        label=label,
    )
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def write_gif(frames, path, seconds):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * seconds), loop=0)


def gif_path(plots_dir, layout, over, variable, baseline, day):
    baseline_tag = f"_{baseline}" if variable == "skill" else ""
    day_tag = f"_day{day:02d}" if day is not None else ""
    return plots_dir / f"gif_{layout}_{variable}{baseline_tag}_{over}{day_tag}.gif"


def render_gif(fields, layout, over, variable, baseline, day, year, path, seconds=1.0):
    """One gif of `fields` (day_fields or month_fields output) to `path`. `day` is the
    fixed forecast day of a single map over the months, None otherwise."""
    field, label = select_field(fields, variable, baseline)
    field = animated_field(field, layout, over, day)
    cmap, norm, extend = colour_scale(field, variable)
    extent = _map_extent(field["lon"], field["lat"])
    with plt.rc_context(GIF_RC):
        frames = []
        for title, panels in frame_specs(field, layout, over, day, year):
            frames.append(render_frame(title, panels, extent, cmap, norm, extend, label))
    write_gif(frames, path, seconds)
    print(f"Wrote {len(frames)} frames to {path}")


def render_standard_gifs(args, pixel_daily, plots_dir):
    """The evaluate set (--gifs): STANDARD_VARIABLES each as a single map over the
    forecast days and, once verif_pixel --monthly has run, as the day-1/day-10 pair and
    the day-1 single map over the months."""
    day_set = pixel_daily.sel(window="all")
    for variable, baseline in STANDARD_VARIABLES:
        path = gif_path(plots_dir, "single", "day", variable, baseline, None)
        render_gif(day_set, "single", "day", variable, baseline, None, None, path)

    if not monthly_verif_path(args).exists():
        print(f"Skipping the month gifs: no {monthly_verif_path(args)} (verif_pixel --monthly)")
        return
    month_set = month_fields(args)
    year = init_year(args)
    for variable, baseline in STANDARD_VARIABLES:
        path = gif_path(plots_dir, "pair", "month", variable, baseline, None)
        render_gif(month_set, "pair", "month", variable, baseline, None, year, path)
        path = gif_path(plots_dir, "single", "month", variable, baseline, STANDARD_MONTH_DAY)
        render_gif(month_set, "single", "month", variable, baseline, STANDARD_MONTH_DAY, year, path)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.layout == "pair" and args.over == "day":
        parser.error("--layout pair steps through the months; --over day needs --layout single")
    args.paths = ExperimentPaths()
    monthly = args.layout == "pair" or args.over == "month"
    over = "month" if monthly else "day"
    day = args.day if args.layout == "single" and monthly else None

    fields = month_fields(args) if monthly else day_fields(args)
    year = init_year(args) if monthly else None
    plots_dir = args.paths.verification_dir(args.experiment, args.version, args.eval_name, "plots")
    path = gif_path(plots_dir, args.layout, over, args.variable, args.baseline, day)
    render_gif(fields, args.layout, over, args.variable, args.baseline, day, year, path, args.seconds)


if __name__ == "__main__":
    main()
