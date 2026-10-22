"""Figures of the input sensitivity of a trained run.

Input: results/<experiment>/<version>/<eval>/sensitivity/permute_days_<block>/summary.csv,
written by deep_lst.sensitivity.scores: the pooled RMSE of the run's own forecast and of the
forecast with one input's days permuted, per variable, solar window and forecast day. The
relevance of an input is that RMSE increase in percent of the unperturbed RMSE.

Several experiments (e.g. the seeds of one model) are drawn as small markers with their mean
as a large one; a single experiment is drawn as the large marker alone.

    relevance_<eval>.pdf        inputs ranked by relevance; panels = permutation block x forecast day
    relevance_leads_<eval>.pdf  relevance per forecast day as an annotated heatmap (one block)

CLI: scripts/sensitivity/plot_sensitivity.py, launcher scripts/sensitivity/plot_sensitivity.sh.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter

from .forecast import sensitivity_dir

MODALITY_LABELS = {"era5": "ERA5", "era5_land": "ERA5-Land", "seviri": "SEVIRI", "glofas_v4": "GloFAS"}
BLOCK_LABELS = {"month": "days permuted within the calendar month", "all": "days permuted across the whole record"}
LINEAR_THRESHOLD = 0.1  # percent; the symlog axis is linear below it
TICKS = [0, 0.1, 1, 10, 100]
COLOR = "#0b0b0b"


def load_relevance(paths, experiments, eval_name, version, blocks, window="all"):
    """Long table (experiment, block, day, variable, relevance) from the summary.csv files."""
    frames = []
    for block in blocks:
        for experiment in experiments:
            path = sensitivity_dir(paths, experiment, version, eval_name, "permute_days", block) / "summary.csv"
            if not path.exists():
                print(f"missing: {path}")
                continue
            summary = pd.read_csv(path)
            summary = summary[summary["window"] == window].copy()
            summary["relevance"] = 100.0 * summary["delta_rmse"] / summary["rmse_reference"]
            summary["experiment"] = experiment
            summary["block"] = block
            frames.append(summary[["experiment", "block", "day", "variable", "relevance"]])
    if not frames:
        raise FileNotFoundError("no summary.csv found for the requested experiments")
    return pd.concat(frames, ignore_index=True)


def rank_variables(table, block, day):
    """Variables ordered by their mean relevance over the experiments, largest first."""
    subset = table[(table["block"] == block) & (table["day"] == day)]
    means = subset.groupby("variable")["relevance"].mean()
    return list(means.sort_values(ascending=False).index)


def relevance_table(table, block, day, order):
    """Variables x experiments (plus their mean) of the relevance in one block and day, ranked."""
    subset = table[(table["block"] == block) & (table["day"] == day)]
    wide = subset.pivot(index="variable", columns="experiment", values="relevance").reindex(order)
    wide["mean"] = wide.mean(axis=1)
    return wide


def variable_label(name):
    modality, _, variable = name.partition(".")
    return f"{variable}  ({MODALITY_LABELS.get(modality, modality)})"


def percent_tick(value, position):
    if value == 0:
        return "0"
    if value < 1:
        return f"{value:g}"
    return f"{value:.0f}"


def draw_ranking(ax, subset, order):
    """One panel: the relevance of every variable; experiments as small markers, their mean large."""
    positions = {variable: index for index, variable in enumerate(order)}
    if subset["experiment"].nunique() > 1:
        y_runs = subset["variable"].map(positions)
        ax.scatter(subset["relevance"], y_runs, s=14, color=COLOR, alpha=0.4, linewidths=0, zorder=2)
    means = subset.groupby("variable")["relevance"].mean()
    y_means = means.index.map(positions)
    ax.scatter(means.values, y_means, s=46, color=COLOR, edgecolor="white", linewidth=0.6, zorder=3)
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


def plot_relevance(table, order, blocks, days, out_path):
    """Ranked relevance; panels = block (rows) x forecast day (columns)."""
    panel_height = 0.36 * len(order) + 1.0
    figsize = (4.0 * len(days) + 1.8, panel_height * len(blocks))
    fig, axes = plt.subplots(len(blocks), len(days), figsize=figsize, squeeze=False, layout="constrained")
    for row, block in enumerate(blocks):
        for col, day in enumerate(days):
            ax = axes[row, col]
            subset = table[(table["block"] == block) & (table["day"] == day)]
            draw_ranking(ax, subset, order)
            ax.set_title(f"{BLOCK_LABELS.get(block, block)}, forecast day {day}", fontsize=9.5, loc="left")
            if col > 0:
                ax.tick_params(labelleft=False)
    for ax in axes[-1]:
        ax.set_xlabel("RMSE increase when the input is permuted (%)")
    n_experiments = table["experiment"].nunique()
    if n_experiments > 1:
        fig.suptitle(f"small markers: {n_experiments} runs, large: their mean", fontsize=9, x=0.02, ha="left")
    fig.savefig(out_path)
    plt.close(fig)


def plot_leads(table, order, block, out_path):
    """Mean relevance per variable and forecast day as an annotated heatmap."""
    subset = table[table["block"] == block]
    days = sorted(subset["day"].unique())
    means = subset.groupby(["variable", "day"])["relevance"].mean().unstack("day")
    means = means.reindex(index=order, columns=days)
    fig, ax = plt.subplots(figsize=(0.5 * len(days) + 3.0, 0.34 * len(order) + 1.6), layout="constrained")
    vmax = max(float(np.nanmax(means.values)), 1.0)
    norm = SymLogNorm(linthresh=LINEAR_THRESHOLD, vmin=0, vmax=vmax)
    image = ax.imshow(means.values, aspect="auto", cmap="YlOrRd", norm=norm)
    for row in range(len(order)):
        for col in range(len(days)):
            value = means.values[row, col]
            if not np.isfinite(value):
                continue
            text = f"{value:.1f}" if abs(value) >= 1 else f"{value:.2f}".replace("-0.00", "0.00")
            color = "white" if norm(value) > 0.65 else "black"
            ax.text(col, row, text, ha="center", va="center", fontsize=6.5, color=color)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days)
    ax.set_xlabel("forecast day")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([variable_label(name) for name in order])
    ax.set_title(BLOCK_LABELS.get(block, block), fontsize=10, loc="left")
    fig.colorbar(image, ax=ax, label="RMSE increase when the input is permuted (%)", shrink=0.85)
    fig.savefig(out_path)
    plt.close(fig)
