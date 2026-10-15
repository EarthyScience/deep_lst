"""Cross-model curves for the hydrostates ladder: per-site skill binned along a site
attribute, one line per data condition, seeds as a band, no background points.

Reads scores/pixel_daily.nc of every (condition, seed) run for one eval, takes the per-site
skill against climatology at a forecast day and solar window, and draws per bin the median
site (top row) and the median paired difference to the `full` run of the same seed (bottom
row). Bins are the shared FVC classes on the FVC axis and ten equal-count bins on the
validity axis. Lines are the seed mean of the bin medians, the band spans the seeds.

    python scripts/evaluation/hydrostates_ladder.py --eval_name eval_mtg
    python scripts/evaluation/hydrostates_ladder.py --eval_name train_mtg_new --day 10 --x lst_validity
    python scripts/evaluation/hydrostates_ladder.py --x greenness --conditions full full_noswvl

`--conditions` restricts the figure to a subset of the ladder; the file name then carries the
non-reference conditions (e.g. `ladder_eval_mtg_greenness_noswvl_day01.pdf`).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from hydrostates_runs import CONDITIONS, OUT_DIR, REFERENCE, SEEDS, WINDOWS, run_name  # noqa: E402

from deep_lst.evaluation_climpred.cubes import pixel_attributes  # noqa: E402
from deep_lst.evaluation_climpred.plots import LC_STYLES, SCATTER_AXES, _lc_group, _scatter_bins  # noqa: E402
from deep_lst.utils.io import ExperimentPaths  # noqa: E402

PATHS = ExperimentPaths()
BASELINE = "clim_smooth"
GREEN_CLASSES = range(1, 11)  # masks.GREEN_BINS: relative greenness at init in 0.1-wide bins
GREEN_POSITIONS = np.array([(k - 0.5) / 10 for k in GREEN_CLASSES])


def condition_suffix(conditions):
    """File-name suffix naming the non-reference conditions when a subset of the ladder is drawn."""
    if list(conditions) == list(CONDITIONS):
        return ""
    return "_" + "_".join(data.removeprefix("full_") for data in conditions if data != REFERENCE)


def load_skill(eval_name, day, window, conditions):
    """{condition: {seed: (n_pixels,) skill}} for the runs whose pixel_daily.nc exists."""
    skill = {}
    for data in conditions:
        for seed in SEEDS:
            path = PATHS.results_dir(run_name(data, seed), "version_0", eval_name) / "scores" / "pixel_daily.nc"
            if not path.exists():
                print(f"missing: {path}")
                continue
            pixel_daily = xr.open_dataset(path, decode_timedelta=False)
            values = pixel_daily["rmse_ss"].sel(baseline=BASELINE, day=day, window=window).values
            skill.setdefault(data, {})[seed] = np.asarray(values, dtype=float)
    return skill


def load_greenness_skill(eval_name, day, window, conditions):
    """{condition: {seed: (10, n_pixels)}} per-site skill inside each greenness-at-init window
    (green_d<k>_<window>), for the runs verified with those windows."""
    skill = {}
    for data in conditions:
        for seed in SEEDS:
            path = PATHS.results_dir(run_name(data, seed), "version_0", eval_name) / "scores" / "pixel_daily.nc"
            if not path.exists():
                print(f"missing: {path}")
                continue
            pixel_daily = xr.open_dataset(path, decode_timedelta=False)
            windows = [f"green_d{k}_{window}" for k in GREEN_CLASSES]
            if not all(name in pixel_daily["window"].values for name in windows):
                print(f"no greenness windows in {path}")
                continue
            values = pixel_daily["rmse_ss"].sel(baseline=BASELINE, day=day, window=windows).values
            skill.setdefault(data, {})[seed] = np.asarray(values, dtype=float)
    return skill


def class_medians(per_class_values):
    """Median site per greenness class of a (10, n_pixels) array; NaN under 5 sites."""
    medians = []
    for values in per_class_values:
        finite = values[np.isfinite(values)]
        medians.append(np.median(finite) if finite.size >= 5 else np.nan)
    return np.array(medians)


def site_mask(eval_name, lc_group):
    """(n_pixels,) True for the sites of one landcover group (plots.LC_STYLES keys), all if None."""
    attributes = pixel_attributes(PATHS.spatial_indices_path(PATHS.baseline_spatial(eval_name)))
    if lc_group is None:
        return np.ones(len(attributes["lc_class"]), dtype=bool)
    groups = np.array([_lc_group(int(lc_class)) for lc_class in attributes["lc_class"]])
    return groups == lc_group


def plot_greenness_ladder(eval_name, day, out_dir, lc_group=None, conditions=tuple(CONDITIONS)):
    """Ladder over the vegetation state at init: x = relative greenness bin, y = median site
    skill inside that bin's window; paired difference to `full` below. `lc_group` restricts
    the sites to one landcover group (e.g. "lc_savanna"); the file name carries it."""
    mask = site_mask(eval_name, lc_group)
    fig, axes = plt.subplots(2, len(WINDOWS), figsize=(5.2 * len(WINDOWS), 6.4), sharex=True, sharey="row")
    for column, window in enumerate(WINDOWS):
        skill = load_greenness_skill(eval_name, day, window, conditions)
        skill = {data: {seed: values[:, mask] for seed, values in per_seed.items()} for data, per_seed in skill.items()}
        reference = skill.get(REFERENCE, {})
        for data in conditions:
            label, color = CONDITIONS[data]
            if data not in skill:
                continue
            curves = np.array([class_medians(values) for values in skill[data].values()])
            draw_condition(
                axes[0, column],
                GREEN_POSITIONS,
                np.nanmean(curves, 0),
                np.nanmin(curves, 0),
                np.nanmax(curves, 0),
                label,
                color,
            )
            if data == REFERENCE:
                continue
            paired = [class_medians(skill[data][seed] - reference[seed]) for seed in skill[data] if seed in reference]
            if paired:
                paired = np.array(paired)
                draw_condition(
                    axes[1, column],
                    GREEN_POSITIONS,
                    np.nanmean(paired, 0),
                    np.nanmin(paired, 0),
                    np.nanmax(paired, 0),
                    label,
                    color,
                )
        axes[0, column].set_title(f"{window} window, day {day}", fontsize=12)
        axes[1, column].axhline(0.0, color="#898781", lw=0.8)
        axes[1, column].set_xlabel("Relative greenness at init (fraction of the site's annual FVC range)")
        for ax in axes[:, column]:
            ax.grid(axis="y", color="#e1e0d9", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Skill improvement (RMSE) vs climatology\nmedian site per state bin")
    axes[1, 0].set_ylabel("Paired difference to `full`\nsame seed, median site per state bin")
    axes[0, 0].legend(frameon=False, fontsize=8)
    group_label = "" if lc_group is None else f", {LC_STYLES[lc_group][0]} sites only ({int(mask.sum())})"
    fig.suptitle(f"{eval_name}{group_label}: seed mean of 3 seeds, band = seed range", fontsize=10, color="#52514e")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = condition_suffix(conditions) + ("" if lc_group is None else f"_{lc_group.removeprefix('lc_')}")
    path = out_dir / f"ladder_{eval_name}_greenness{suffix}_day{day:02d}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def bin_medians(values, bins, x_values):
    """Median of `values` and of `x_values` per bin id 1..10 (NaN where fewer than 5 sites)."""
    medians, positions = [], []
    for bin_id in range(1, 11):
        members = (bins == bin_id) & np.isfinite(values) & np.isfinite(x_values)
        medians.append(np.median(values[members]) if members.sum() >= 5 else np.nan)
        positions.append(np.median(x_values[members]) if members.sum() >= 5 else np.nan)
    return np.array(positions), np.array(medians)


def seed_curves(per_seed, bins, x_values):
    """Seed mean and min/max of the per-bin medians over the seeds of one condition."""
    curves = np.array([bin_medians(values, bins, x_values)[1] for values in per_seed.values()])
    positions = bin_medians(next(iter(per_seed.values())), bins, x_values)[0]
    return positions, np.nanmean(curves, axis=0), np.nanmin(curves, axis=0), np.nanmax(curves, axis=0)


def draw_condition(ax, positions, mean, low, high, label, color):
    ax.fill_between(positions, low, high, color=color, alpha=0.18, linewidth=0)
    ax.plot(positions, mean, color=color, lw=2, marker="o", ms=4, label=label)


def plot_ladder(eval_name, day, x_attribute, out_dir, conditions=tuple(CONDITIONS)):
    attributes = pixel_attributes(PATHS.spatial_indices_path(PATHS.baseline_spatial(eval_name)))
    x_values = np.asarray(attributes[x_attribute], dtype=float)
    bins = _scatter_bins(attributes, x_attribute, x_values)
    xlabel, prefix = SCATTER_AXES[x_attribute]

    fig, axes = plt.subplots(2, len(WINDOWS), figsize=(5.2 * len(WINDOWS), 6.4), sharex=True, sharey="row")
    for column, window in enumerate(WINDOWS):
        skill = load_skill(eval_name, day, window, conditions)
        reference = skill.get(REFERENCE, {})
        for data in conditions:
            label, color = CONDITIONS[data]
            if data not in skill:
                continue
            positions, mean, low, high = seed_curves(skill[data], bins, x_values)
            draw_condition(axes[0, column], positions, mean, low, high, label, color)
            if data == REFERENCE:
                continue
            paired = {seed: skill[data][seed] - reference[seed] for seed in skill[data] if seed in reference}
            if paired:
                positions, mean, low, high = seed_curves(paired, bins, x_values)
                draw_condition(axes[1, column], positions, mean, low, high, label, color)
        axes[0, column].set_title(f"{window} window, day {day}", fontsize=12)
        axes[1, column].axhline(0.0, color="#898781", lw=0.8)
        axes[1, column].set_xlabel(xlabel)
        for ax in axes[:, column]:
            ax.grid(axis="y", color="#e1e0d9", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Skill improvement (RMSE) vs climatology\nmedian site per bin")
    axes[1, 0].set_ylabel("Paired difference to `full`\nsame seed, median site per bin")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{eval_name}: seed mean of 3 seeds, band = seed range", fontsize=10, color="#52514e")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ladder_{eval_name}_{prefix}{condition_suffix(conditions)}_day{day:02d}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_name", default="eval_mtg")
    parser.add_argument("--day", type=int, nargs="+", default=[1, 10])
    parser.add_argument(
        "--x",
        dest="x_attributes",
        nargs="+",
        default=["fvc_mean", "lst_validity"],
        choices=list(SCATTER_AXES) + ["greenness"],
    )
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        choices=list(CONDITIONS),
        help="subset of the ladder to draw, in ladder order; `full` is always the paired reference",
    )
    parser.add_argument(
        "--lc_groups",
        nargs="*",
        default=[],
        choices=[g for g in LC_STYLES if g != "other"],
        help="greenness ladders restricted to these landcover groups, one figure each (in addition to the pooled one)",
    )
    args = parser.parse_args()
    conditions = [data for data in CONDITIONS if data in args.conditions]  # ladder order
    for day in args.day:
        for x_attribute in args.x_attributes:
            if x_attribute == "greenness":
                print("wrote", plot_greenness_ladder(args.eval_name, day, args.out_dir, conditions=conditions))
                for lc_group in args.lc_groups:
                    print("wrote", plot_greenness_ladder(args.eval_name, day, args.out_dir, lc_group, conditions))
            else:
                print("wrote", plot_ladder(args.eval_name, day, x_attribute, args.out_dir, conditions))


if __name__ == "__main__":
    main()
