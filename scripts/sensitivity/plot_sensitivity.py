"""Relevance of a model's stored inputs across runs: a ranked dot plot and a lead-time heatmap.

Relevance of one input = the pooled RMSE increase, in percent of the unperturbed RMSE, when
that input's days are permuted (deep_lst.sensitivity; summary.csv of every run). A group is a
set of runs, e.g. the seeds of one data condition. The dot plot draws every run as a small
marker and the group mean as a large one, so the seed spread and the differences between
conditions sit on the same axis; panels are block x forecast day. The heatmap shows the
group-mean relevance per forecast day. Variables are ranked by their mean relevance over all
groups in the first block and first day.

    python scripts/sensitivity/plot_sensitivity.py --eval_name eval_mtg_hydrostates_val \\
        --group "no swvl=mlp_smoothl1_h24_full_noswvl_v16.2_hydrostates,mlp_smoothl1_h24_full_noswvl_v16.2_hydrostates_s7" \\
        --group "full=mlp_smoothl1_h24_full_v16.2_hydrostates" --colors "#7f7f7f" "#1f77b4" \\
        --blocks month all --days 1 10 --out results/independent/sensitivity/v03

Writes <out>/relevance_<eval>.pdf and <out>/relevance_leads_<eval>.pdf and prints the ranked
group means. Missing summary files are reported and skipped.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter

from deep_lst.sensitivity.forecast import sensitivity_dir
from deep_lst.utils.io import ExperimentPaths

MODALITY_LABELS = {"era5": "ERA5", "era5_land": "ERA5-Land", "seviri": "SEVIRI", "glofas_v4": "GloFAS"}
BLOCK_LABELS = {"month": "days permuted within the calendar month", "all": "days permuted across the whole record"}
LINEAR_THRESHOLD = 0.1  # percent; the symlog axis is linear below it
TICKS = [0, 0.1, 1, 10, 100]
DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def parse_groups(specs):
    """["label=run1,run2", ...] -> {label: [run1, run2]}, in command-line order."""
    groups = {}
    for spec in specs:
        label, _, runs = spec.partition("=")
        groups[label.strip()] = [run.strip() for run in runs.split(",") if run.strip()]
    return groups


def load_relevance(paths, groups, eval_name, version, blocks, window):
    """Long table (group, run, block, day, variable, relevance, share_pixels_worse)."""
    frames = []
    for block in blocks:
        for label, runs in groups.items():
            for run in runs:
                path = sensitivity_dir(paths, run, version, eval_name, "permute_days", block) / "summary.csv"
                if not path.exists():
                    print(f"missing: {path}")
                    continue
                table = pd.read_csv(path)
                table = table[table["window"] == window].copy()
                table["relevance"] = 100.0 * table["delta_rmse"] / table["rmse_reference"]
                table["group"] = label
                table["run"] = run
                table["block"] = block
                frames.append(table[["group", "run", "block", "day", "variable", "relevance", "share_pixels_worse"]])
    if not frames:
        raise SystemExit("no summary.csv found for the requested runs")
    return pd.concat(frames, ignore_index=True)


def variable_label(name):
    modality, _, variable = name.partition(".")
    return f"{variable}  ({MODALITY_LABELS.get(modality, modality)})"


def rank_variables(table, block, day):
    """Variables ordered by their mean relevance over all groups in one block and day, largest first."""
    subset = table[(table["block"] == block) & (table["day"] == day)]
    means = subset.groupby("variable")["relevance"].mean().sort_values(ascending=False)
    return list(means.index)


def percent_tick(value, position):
    if value == 0:
        return "0"
    if value < 1:
        return f"{value:g}"
    return f"{value:.0f}"


def draw_panel(ax, subset, order, colors, offsets):
    positions = {variable: index for index, variable in enumerate(order)}
    for offset, (label, group) in zip(offsets, subset.groupby("group", sort=False)):
        color = colors[label]
        y_runs = group["variable"].map(positions) + offset
        ax.scatter(group["relevance"], y_runs, s=14, color=color, alpha=0.45, linewidths=0, zorder=2)
        means = group.groupby("variable")["relevance"].mean()
        y_means = means.index.map(positions) + offset
        ax.scatter(means.values, y_means, s=46, color=color, edgecolor="white", linewidth=0.6, zorder=3, label=label)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([variable_label(name) for name in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xscale("symlog", linthresh=LINEAR_THRESHOLD, linscale=0.6)
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(percent_tick))
    ax.set_xlim(min(-0.05, 1.2 * subset["relevance"].min()), 2.0 * max(subset["relevance"].max(), 1.0))
    ax.grid(axis="x", color="0.9", zorder=0)
    ax.axvline(0, color="0.6", linewidth=0.8, zorder=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_relevance(table, order, groups, colors, blocks, days, out_path):
    n_rows, n_cols = len(blocks), len(days)
    panel_height = 0.36 * len(order) + 1.0
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols + 1.8, panel_height * n_rows), squeeze=False, layout="constrained"
    )
    offsets = np.linspace(-0.28, 0.28, len(groups)) if len(groups) > 1 else np.zeros(1)
    for row, block in enumerate(blocks):
        for col, day in enumerate(days):
            ax = axes[row, col]
            subset = table[(table["block"] == block) & (table["day"] == day)]
            draw_panel(ax, subset, order, colors, offsets)
            ax.set_title(f"{BLOCK_LABELS.get(block, block)}, forecast day {day}", fontsize=9.5, loc="left")
            if col > 0:
                ax.tick_params(labelleft=False)
    for ax in axes[-1]:
        ax.set_xlabel("RMSE increase when the input is permuted (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(groups), frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


def plot_leads(table, order, groups, block, out_path):
    subset = table[table["block"] == block]
    days = sorted(subset["day"].unique())
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(0.5 * len(days) * len(groups) + 3.0, 0.34 * len(order) + 1.6),
        squeeze=False,
        sharey=True,
        layout="constrained",
    )
    vmax = max(subset["relevance"].max(), 1.0)
    norm = SymLogNorm(linthresh=LINEAR_THRESHOLD, vmin=0, vmax=vmax)
    image = None
    for ax, label in zip(axes[0], groups):
        means = subset[subset["group"] == label].groupby(["variable", "day"])["relevance"].mean().unstack("day")
        means = means.reindex(index=order, columns=days)
        image = ax.imshow(means.values, aspect="auto", cmap="YlOrRd", norm=norm)
        for row in range(len(order)):
            for col in range(len(days)):
                value = means.values[row, col]
                if not np.isfinite(value):
                    continue
                text = f"{value:.1f}" if abs(value) >= 1 else f"{value:.2f}".replace("-0.00", "0.00")
                ax.text(
                    col,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if norm(value) > 0.65 else "black",
                )
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels(days)
        ax.set_xlabel("forecast day")
        ax.set_title(label, loc="left", fontsize=10)
    axes[0, 0].set_yticks(range(len(order)))
    axes[0, 0].set_yticklabels([variable_label(name) for name in order])
    fig.colorbar(image, ax=axes[0], label="RMSE increase when the input is permuted (%)", shrink=0.85)
    fig.suptitle(BLOCK_LABELS.get(block, block), fontsize=10, x=0.02, ha="left")
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_name", required=True)
    parser.add_argument("--version", default="version_0")
    parser.add_argument("--group", action="append", required=True, help='"label=run1,run2,..." (repeat per group)')
    parser.add_argument("--colors", nargs="*", default=None, help="one colour per group")
    parser.add_argument("--blocks", nargs="+", default=["month", "all"])
    parser.add_argument("--days", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--window", default="all", help="solar window of the summary rows: all | day | night")
    parser.add_argument("--heat_block", default=None, help="block of the lead heatmap (default: first of --blocks)")
    parser.add_argument("--out", type=Path, default=Path("results/independent/sensitivity"))
    args = parser.parse_args()

    groups = parse_groups(args.group)
    colors = dict(zip(groups, args.colors if args.colors else DEFAULT_COLORS))
    table = load_relevance(ExperimentPaths(), groups, args.eval_name, args.version, args.blocks, args.window)
    order = rank_variables(table, args.blocks[0], args.days[0])

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.eval_name}" if args.window == "all" else f"{args.eval_name}_{args.window}"
    plot_relevance(table, order, groups, colors, args.blocks, args.days, args.out / f"relevance_{suffix}.pdf")
    plot_leads(table, order, groups, args.heat_block or args.blocks[0], args.out / f"relevance_leads_{suffix}.pdf")

    first = table[(table["block"] == args.blocks[0]) & (table["day"] == args.days[0])]
    ranked = first.groupby(["variable", "group"], sort=False)["relevance"].mean().unstack("group").reindex(order)
    print(f"relevance (%), block {args.blocks[0]}, day {args.days[0]}, window {args.window}:")
    print(ranked.round(2).to_string())
    print(f"wrote {args.out / f'relevance_{suffix}.pdf'} and {args.out / f'relevance_leads_{suffix}.pdf'}")


if __name__ == "__main__":
    main()
