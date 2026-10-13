"""Spatial generalisation of the hydrostates ladder: eval-site RMSE against training-site RMSE
per bin, the no-GloFAS model next to each GloFAS model.

For every run the per-site RMSE (scores/pixel_daily.nc, pooled with the sample counts of
verif/by_pixel.nc) is pooled per bin on the 3000 training sites and on the 400 holdout
sites. One figure per GloFAS condition: x = training-site RMSE of a bin, y = holdout RMSE
of the same bin, the `full` model in black and the condition in colour, seed mean with the
seed range as error bars, the 1:1 line as the no-gap reference. A summary figure shows the
paired change of the gap (eval - train) relative to `full`, same seed, per bin.

Bins are shared between the two site sets: the FVC classes of the selection npz, or fixed
10 %-wide LST validity bins. Holdout sites that are also training sites are excluded.

    python scripts/evaluation/hydrostates_generalisation.py
    python scripts/evaluation/hydrostates_generalisation.py --x lst_validity --day 10
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from deep_lst.utils.io import ExperimentPaths

PATHS = ExperimentPaths()
OUT_DIR = Path("results/independent/hydrostates")
TRAIN_EVAL = "train_mtg_new"
HOLDOUT_EVAL = "eval_mtg"
REFERENCE = "full"
CONDITIONS = {
    "full": ("full (no GloFAS)", "#0b0b0b"),
    "full_rzsm": ("+ RZSM", "#2a78d6"),
    "full_rzsm_dis": ("+ RZSM + discharge", "#eb6834"),
    "full_glofas": ("+ RZSM + discharge + runoff", "#1baf7a"),
}
SEEDS = (42, 7, 23)
WINDOWS = ("day", "night")
VALIDITY_EDGES = np.arange(10, 100, 10)  # 10 %-wide bins, shared by both site sets
MIN_SITES = 5


def run_name(condition, seed):
    tag = "hydrostates" if seed == 42 else f"hydrostates_s{seed}"
    return f"mlp_smoothl1_h24_{condition}_v16_{tag}"


def site_bins(eval_name, x_attribute):
    """Per-site bin id (0 = unbinned) in forecast pixel order, plus a keep mask."""
    npz = np.load(PATHS.spatial_indices_path(PATHS.baseline_spatial(eval_name)))
    order = np.argsort(npz["src_indices"], kind="stable")
    if x_attribute == "fvc_mean":
        bins = npz["fvc_classes"][order].astype(int)
    else:
        validity = npz["lst_validity_annual_mean"][order]
        bins = 1 + np.searchsorted(VALIDITY_EDGES, validity, side="right")
        bins[~np.isfinite(validity)] = 0
    keep = np.ones(len(bins), dtype=bool)
    if eval_name == HOLDOUT_EVAL:  # unseen sites only
        training = np.load(PATHS.spatial_indices_path(PATHS.baseline_spatial(TRAIN_EVAL)))["src_indices"].astype(
            np.int64
        )
        keep = ~np.isin(npz["src_indices"][order].astype(np.int64), training)
    return bins, keep


def bin_rmse(eval_name, condition, seed, day, window, bins, keep):
    """Pooled RMSE per bin id (NaN where missing or under MIN_SITES sites), or None without a verification."""
    results_dir = PATHS.results_dir(run_name(condition, seed), "version_0", eval_name)
    scores_path = results_dir / "scores" / "pixel_daily.nc"
    if not scores_path.exists():
        return None
    rmse = xr.open_dataset(scores_path, decode_timedelta=False)["rmse_model"].sel(day=day, window=window).values
    counts = xr.open_dataset(results_dir / "verif" / "by_pixel.nc", decode_timedelta=False)["n"].sel(window=window)
    day_of_lead = (counts["lead"] - 1) // 24 + 1
    n = counts.where(day_of_lead == day).sum("lead").values
    pooled = np.full(bins.max() + 1, np.nan)
    for bin_id in range(1, bins.max() + 1):
        members = keep & (bins == bin_id) & np.isfinite(rmse) & (n > 0)
        if members.sum() >= MIN_SITES:
            pooled[bin_id] = np.sqrt(np.sum(n[members] * rmse[members] ** 2) / np.sum(n[members]))
    return pooled


def collect(day, window, x_attribute):
    """{condition: {seed: {"train": per-bin RMSE, "eval": per-bin RMSE}}} for the verified runs."""
    train_bins, train_keep = site_bins(TRAIN_EVAL, x_attribute)
    eval_bins, eval_keep = site_bins(HOLDOUT_EVAL, x_attribute)
    n_bins = max(train_bins.max(), eval_bins.max()) + 1
    out = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            train = bin_rmse(TRAIN_EVAL, condition, seed, day, window, train_bins, train_keep)
            held = bin_rmse(HOLDOUT_EVAL, condition, seed, day, window, eval_bins, eval_keep)
            if train is None or held is None:
                print(f"missing verification: {condition} seed {seed} ({'train' if train is None else 'eval'})")
                continue
            padded = {
                name: np.pad(values, (0, n_bins - len(values)), constant_values=np.nan)
                for name, values in (("train", train), ("eval", held))
            }
            out.setdefault(condition, {})[seed] = padded
    return out, n_bins


def seed_stats(per_seed, key):
    stack = np.array([values[key] for values in per_seed.values()])
    return np.nanmean(stack, axis=0), np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)


def bin_labels(x_attribute, n_bins):
    if x_attribute == "fvc_mean":
        return {1: "bare", **{k: f"c{k}" for k in range(2, n_bins)}}
    return {
        k: f"{VALIDITY_EDGES[k - 2] if k > 1 else 0}-{VALIDITY_EDGES[k - 1] if k - 1 < len(VALIDITY_EDGES) else 100}%"
        for k in range(1, n_bins)
    }


def draw_pairs(ax, data, condition, n_bins, labels):
    """Training vs holdout RMSE per bin for `full` and one condition, with the shift between them."""
    ref_train, ref_lo_t, ref_hi_t = seed_stats(data[REFERENCE], "train")
    ref_eval, ref_lo_e, ref_hi_e = seed_stats(data[REFERENCE], "eval")
    con_train, con_lo_t, con_hi_t = seed_stats(data[condition], "train")
    con_eval, con_lo_e, con_hi_e = seed_stats(data[condition], "eval")
    label, color = CONDITIONS[condition]
    for bin_id in range(1, n_bins):
        if not (np.isfinite(ref_train[bin_id]) and np.isfinite(con_train[bin_id])):
            continue
        ax.plot(
            [ref_train[bin_id], con_train[bin_id]],
            [ref_eval[bin_id], con_eval[bin_id]],
            color="#c3c2b7",
            lw=0.8,
            zorder=1,
        )
        ax.annotate(
            labels.get(bin_id, str(bin_id)),
            (ref_train[bin_id], ref_eval[bin_id]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#52514e",
        )
    ax.errorbar(
        ref_train[1:],
        ref_eval[1:],
        xerr=[ref_train[1:] - ref_lo_t[1:], ref_hi_t[1:] - ref_train[1:]],
        yerr=[ref_eval[1:] - ref_lo_e[1:], ref_hi_e[1:] - ref_eval[1:]],
        fmt="o",
        ms=5,
        color=CONDITIONS[REFERENCE][1],
        ecolor="#898781",
        elinewidth=0.8,
        label=CONDITIONS[REFERENCE][0],
        zorder=3,
    )
    ax.errorbar(
        con_train[1:],
        con_eval[1:],
        xerr=[con_train[1:] - con_lo_t[1:], con_hi_t[1:] - con_train[1:]],
        yerr=[con_eval[1:] - con_lo_e[1:], con_hi_e[1:] - con_eval[1:]],
        fmt="s",
        ms=5,
        color=color,
        ecolor=color,
        elinewidth=0.8,
        label=label,
        zorder=3,
    )
    finite = np.concatenate([ref_train[1:], con_train[1:], ref_eval[1:], con_eval[1:]])
    finite = finite[np.isfinite(finite)]
    low, high = finite.min() * 0.95, finite.max() * 1.05
    ax.plot([low, high], [low, high], color="#898781", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")


def plot_condition(condition, x_attribute, days, out_dir):
    fig, axes = plt.subplots(len(days), len(WINDOWS), figsize=(5.2 * len(WINDOWS), 5.0 * len(days)))
    axes = np.atleast_2d(axes)
    for row, day in enumerate(days):
        for column, window in enumerate(WINDOWS):
            data, n_bins = collect(day, window, x_attribute)
            ax = axes[row, column]
            if REFERENCE in data and condition in data:
                draw_pairs(ax, data, condition, n_bins, bin_labels(x_attribute, n_bins))
            ax.set_title(f"{window} window, day {day}", fontsize=12)
            ax.set_xlabel("RMSE on training sites [K], pooled per bin")
            ax.set_ylabel("RMSE on unseen holdout sites [K], pooled per bin")
            ax.grid(color="#e1e0d9", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    prefix = "fvc" if x_attribute == "fvc_mean" else "validity"
    fig.suptitle(
        f"{CONDITIONS[condition][0]}: bins = {prefix} classes, markers = seed mean, bars = seed range, grey line = shift from `full`",
        fontsize=9,
        color="#52514e",
    )
    fig.tight_layout()
    path = out_dir / f"generalisation_{condition}_{prefix}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _draw_band(ax, positions, curves, label, color):
    ax.fill_between(
        positions, np.nanmin(curves, axis=0), np.nanmax(curves, axis=0), color=color, alpha=0.18, linewidth=0
    )
    ax.plot(positions, np.nanmean(curves, axis=0), color=color, lw=2, marker="o", ms=4, label=label)


def plot_gap_ladder(x_attribute, day, out_dir):
    """Ladder layout of the generalisation gap: top row the gap (holdout − training RMSE)
    per bin for every condition, bottom row its paired change relative to `full`, same seed;
    day and night as columns, seed mean with the seed range as a band."""
    fig, axes = plt.subplots(2, len(WINDOWS), figsize=(5.2 * len(WINDOWS), 6.4), sharex=True, sharey="row")
    for column, window in enumerate(WINDOWS):
        data, n_bins = collect(day, window, x_attribute)
        positions = np.arange(1, n_bins)
        gaps = {}
        for condition, (label, color) in CONDITIONS.items():
            if condition not in data:
                continue
            gaps[condition] = {seed: (values["eval"] - values["train"])[1:] for seed, values in data[condition].items()}
            _draw_band(axes[0, column], positions, np.array(list(gaps[condition].values())), label, color)
        for condition, (label, color) in CONDITIONS.items():
            if condition == REFERENCE or condition not in gaps or REFERENCE not in gaps:
                continue
            changes = [
                gaps[condition][seed] - gaps[REFERENCE][seed] for seed in gaps[condition] if seed in gaps[REFERENCE]
            ]
            if changes:
                _draw_band(axes[1, column], positions, np.array(changes), label, color)
        axes[0, column].set_title(f"{window} window, day {day}", fontsize=12)
        axes[0, column].axhline(0.0, color="#898781", lw=0.8)
        axes[1, column].axhline(0.0, color="#898781", lw=0.8)
        axes[1, column].set_xticks(positions)
        axes[1, column].set_xticklabels(
            [bin_labels(x_attribute, n_bins).get(k, str(k)) for k in positions], fontsize=8, rotation=45
        )
        for ax in axes[:, column]:
            ax.grid(axis="y", color="#e1e0d9", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Generalisation gap [K]\nholdout − training RMSE, pooled per bin")
    axes[1, 0].set_ylabel("Change of the gap relative to `full` [K]\nsame seed (negative = narrower)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    prefix = "fvc" if x_attribute == "fvc_mean" else "validity"
    fig.suptitle("seed mean of 3 seeds, band = seed range; unseen holdout sites only", fontsize=9, color="#52514e")
    fig.tight_layout()
    path = out_dir / f"generalisation_gap_{prefix}_day{day:02d}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_gap_change(x_attribute, days, out_dir):
    """Kept for the first rendering pass; the ladder layout above supersedes it."""
    return [plot_gap_ladder(x_attribute, day, out_dir) for day in days]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--x",
        dest="x_attributes",
        nargs="+",
        default=["fvc_mean", "lst_validity"],
        choices=["fvc_mean", "lst_validity"],
    )
    parser.add_argument("--day", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--scatter", action="store_true", help="also the per-condition train-vs-holdout scatter figures"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for x_attribute in args.x_attributes:
        for day in args.day:
            print("wrote", plot_gap_ladder(x_attribute, day, args.out_dir))
        if args.scatter:
            for condition in CONDITIONS:
                if condition != REFERENCE:
                    print("wrote", plot_condition(condition, x_attribute, args.day, args.out_dir))


if __name__ == "__main__":
    main()
