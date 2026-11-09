"""Animated per-pixel maps for presentations.

Two layouts (--layout): `single` is one map stepping through the forecast days
(--over day) or through the months at one forecast day (--over month, --day); `pair`
is the day-1 and day-10 maps side by side stepping through the months. Both show the
model RMSE, its normalised RMSE, its bias, its Nash-Sutcliffe efficiency, its anomaly
correlation, or its RMSE skill score against climatology or persistence (--variable,
--baseline). Meant for the training sites and the dense case studies: the holdouts are
too sparse to read as maps.

The day frames come from scores/pixel_daily.nc (window 'all', written by evaluate).
The month frames need verif/by_pixel_monthly.nc from

    python -m deep_lst.evaluation_climpred.verif_pixel --experiment <exp> --eval_name <name> --windows all --monthly

(thirteen stacked windows: ~450G on train_mtg_new; run_climpred.sh submits it for the
training sites), pooled into forecast days exactly as derive.py pools by_pixel.nc
(derive.pixel_day_scores). The month is the valid-time month, as for the JFMA windows,
so the January frame also holds the first January days of the following year from the
late-December inits.

The frames are drawn by plots.draw_pixel_panels, the renderer of the static pixel maps,
so a frame and the matching pixel_map pdf look alike. One colour scale per gif over
every frame (plots.map_scale, as the static maps), so the colours stay comparable while
the animation runs.

evaluate renders the standard set (render_standard_gifs): the bias as a single map over
the forecast days, and the pair over the months per variable where the month windows
were verified. This entry point renders one gif of any combination:
    python -m deep_lst.evaluation_climpred.gifs --experiment <exp> --layout single --over month --variable rmse --day 10
"""

import argparse
import io

import matplotlib.pyplot as plt
import xarray as xr
from PIL import Image

from deep_lst.utils.io import ExperimentPaths

from .derive import pixel_day_scores
from .masks import MONTHLY_WINDOWS, WINDOW_LABELS
from .plots import BASELINE_LABELS, MAP_LABELS, MAP_RC, _map_extent, draw_pixel_panels, map_scale

PAIR_DAYS = (1, 10)  # nowcast and medium range, as the static maps
# The standard set (render_standard_gifs): (variable, baseline) each as a single map over
# the forecast days (scores/pixel_daily.nc, every verification) and, once verif_pixel
# --monthly has run, as the day-1/day-10 pair over the months.
STANDARD_DAY_VARIABLES = (("bias", None),)
STANDARD_MONTH_VARIABLES = (
    ("rmse", None),
    ("nrmse", None),
    ("skill", "clim_smooth"),
    ("skill", "persistence"),
    ("nse", None),
    ("acc", None),
)
VARIABLES = ("rmse", "nrmse", "bias", "skill", "nse", "acc")
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
        choices=list(VARIABLES),
        default="skill",
        help="RMSE of the model, its normalised RMSE, bias, NSE or ACC, or its RMSE skill score against --baseline.",
    )
    parser.add_argument(
        "--baseline",
        choices=["clim_smooth", "persistence"],
        default="clim_smooth",
        help="Reference of the skill score (ignored for the other variables).",
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
    24 leads of a day as derive.py pools by_pixel.nc (the file's 'all' window supplies
    the annual nRMSE span, then only the twelve month windows are kept)."""
    by_pixel = xr.open_dataset(monthly_verif_path(args), decode_timedelta=False)
    fields = pixel_day_scores(by_pixel)
    return fields.sel(window=MONTHLY_WINDOWS)


def init_year(args):
    """Year of the forecast init times, for the month titles."""
    store = xr.open_zarr(args.paths.forecast_path(args.experiment, args.version, args.eval_name))
    return int(store["init_time"].dt.year.min())


def select_field(fields, variable, baseline):
    """The drawn variable and its colorbar label."""
    if variable == "skill":
        return fields["rmse_ss"].sel(baseline=baseline), f"Skill improvement (RMSE) vs {BASELINE_LABELS[baseline]}"
    return fields[f"{variable}_model"], MAP_LABELS[variable]


def animated_field(field, layout, over, day):
    """The part of the field the frames draw; the colour scale is computed over it."""
    if layout == "pair":
        return field.sel(day=list(PAIR_DAYS))
    if over == "month":
        return field.sel(day=day)
    return field


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
    """One frame as an image: the static map layout (plots.draw_pixel_panels, fixed
    margins) with the frame title above, so every frame has the same pixel size."""
    fig = draw_pixel_panels(panels, extent, cmap, norm, extend, label, title=title)
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
    cmap, norm, extend = map_scale(variable, field)
    extent = _map_extent(field["lon"], field["lat"])
    with plt.rc_context(MAP_RC):
        frames = []
        for title, panels in frame_specs(field, layout, over, day, year):
            frames.append(render_frame(title, panels, extent, cmap, norm, extend, label))
    write_gif(frames, path, seconds)
    print(f"Wrote {len(frames)} frames to {path}")


def render_standard_gifs(args, pixel_daily, plots_dir):
    """The evaluate set: STANDARD_DAY_VARIABLES each as a single map over the forecast
    days (from `pixel_daily`, window 'all') and, once verif_pixel --monthly has run,
    STANDARD_MONTH_VARIABLES each as the day-1/day-10 pair over the months. Variables
    a monthly verif predating the per-pixel sums lacks (nrmse, nse, acc) are skipped."""
    day_set = pixel_daily.sel(window="all")
    for variable, baseline in STANDARD_DAY_VARIABLES:
        path = gif_path(plots_dir, "single", "day", variable, baseline, None)
        render_gif(day_set, "single", "day", variable, baseline, None, None, path)

    if not monthly_verif_path(args).exists():
        print(f"Skipping the month gifs: no {monthly_verif_path(args)} (verif_pixel --monthly)")
        return
    month_set = month_fields(args)
    year = init_year(args)
    for variable, baseline in STANDARD_MONTH_VARIABLES:
        if variable != "skill" and f"{variable}_model" not in month_set:
            print(f"Skipping the {variable} month gif: {monthly_verif_path(args)} predates the per-pixel sums")
            continue
        path = gif_path(plots_dir, "pair", "month", variable, baseline, None)
        render_gif(month_set, "pair", "month", variable, baseline, None, year, path)


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
