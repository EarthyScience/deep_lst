"""Figures of the input sensitivity of a trained run.

Input: results/<experiment>/<version>/<eval>/sensitivity/permute_days_<block>/summary.csv,
written by deep_lst.sensitivity.scores: the pooled RMSE of the run's own forecast and of the
forecast with one input's days permuted, per variable, window and forecast day. The figures
show the increase of that RMSE, in Kelvin (metric "kelvin", the default) or in percent of the
unperturbed RMSE (metric "percent"); in the window "amplitude" the RMSE is the RMS diurnal
amplitude error. Several experiments (e.g. seeds) are averaged.

    relevance_<block>_<eval>.pdf        inputs ranked by the day-1 RMSE increase; columns = RMSE
                                        increase and amplitude error increase; filled marker at
                                        the first day, arrow to a hollow marker at the second
                                        (red: the effect grows with lead, blue: it shrinks), the
                                        two values at the right margin
    relevance_leads_<block>_<eval>.pdf  RMSE increase per forecast day as a heatmap, every row
                                        coloured on its own range

CLI: scripts/sensitivity/plot_sensitivity.py, launcher scripts/sensitivity/plot_sensitivity.sh.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

from .forecast import sensitivity_dir
from .scores import AMPLITUDE_WINDOW

MODALITY_LABELS = {"era5": "ERA5", "era5_land": "ERA5-Land", "seviri": "SEVIRI", "glofas_v4": "GloFAS"}
BLOCK_LABELS = {"month": "days permuted within the calendar month", "all": "days permuted across the whole record"}
METRICS = {  # column of the table, unit, symlog threshold (linear below it) and axis ticks
    "kelvin": {"column": "increase_k", "unit": "K", "linthresh": 0.01, "ticks": [0, 0.01, 0.1, 1]},
    "percent": {"column": "increase_pct", "unit": "%", "linthresh": 0.1, "ticks": [0, 0.1, 1, 10, 100]},
}
GROWS, SHRINKS, FIRST = "#d6281e", "#2a6fd6", "#0b0b0b"


def load_relevance(paths, experiments, eval_name, version, blocks):
    """Long table (experiment, block, window, day, variable, increase_k, increase_pct)."""
    frames = []
    for block in blocks:
        for experiment in experiments:
            path = sensitivity_dir(paths, experiment, version, eval_name, "permute_days", block) / "summary.csv"
            if not path.exists():
                print(f"missing: {path}")
                continue
            summary = pd.read_csv(path)
            summary["increase_k"] = summary["delta_rmse"]
            summary["increase_pct"] = 100.0 * summary["delta_rmse"] / summary["rmse_reference"]
            summary["experiment"] = experiment
            summary["block"] = block
            frames.append(summary[["experiment", "block", "window", "day", "variable", "increase_k", "increase_pct"]])
    if not frames:
        raise FileNotFoundError("no summary.csv found for the requested experiments")
    return pd.concat(frames, ignore_index=True)


def mean_increase(table, block, window, metric):
    """variables x forecast days, averaged over the experiments; empty when the window is absent."""
    subset = table[(table["block"] == block) & (table["window"] == window)]
    if subset.empty:
        return pd.DataFrame()
    return subset.groupby(["variable", "day"])[METRICS[metric]["column"]].mean().unstack("day")


def rank_variables(table, block, day, window="all", metric="kelvin"):
    """Variables ordered by their increase in one block, window and day, largest first."""
    means = mean_increase(table, block, window, metric)
    return list(means[day].sort_values(ascending=False).index)


def increase_table(table, block, order, days, window="all", metric="kelvin"):
    """Ranked variables x the requested days, for printing."""
    means = mean_increase(table, block, window, metric)
    if means.empty:
        return means
    return means.reindex(index=order, columns=days)


def column_label(window, metric):
    quantity = "amplitude error increase" if window == AMPLITUDE_WINDOW else "RMSE increase"
    return f"{quantity} ({METRICS[metric]['unit']})"


def variable_label(name):
    modality, _, variable = name.partition(".")
    return f"{variable}  ({MODALITY_LABELS.get(modality, modality)})"


def axis_tick(value, position):
    return "0" if value == 0 else f"{value:g}"


def value_text(value, metric):
    """Two or three decimals for Kelvin, one or two for percent; a value that rounds to zero
    prints without a sign, every other negative value keeps it."""
    if not np.isfinite(value):
        return "-"
    if metric == "kelvin":
        decimals = 2 if abs(value) >= 0.1 else 3
    else:
        decimals = 1 if abs(value) >= 1 else 2
    rounded = round(value, decimals)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"


def draw_lead_change(ax, means, order, days, metric):
    """One column: per variable a filled marker at days[0], an arrow to a hollow marker at
    days[1] coloured by the direction of change, and both values at the right margin."""
    settings = METRICS[metric]
    for position, variable in enumerate(order):
        if variable not in means.index:
            continue
        first, last = means.loc[variable, days[0]], means.loc[variable, days[1]]
        if not (np.isfinite(first) and np.isfinite(last)):
            continue
        color = GROWS if last > first else SHRINKS
        ax.annotate(
            "",
            xy=(last, position),
            xytext=(first, position),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.9, "shrinkA": 4, "shrinkB": 4},
            zorder=2,
        )
        ax.scatter([first], [position], s=38, color=FIRST, zorder=3)
        ax.scatter([last], [position], s=38, facecolor="white", edgecolor=color, linewidth=1.2, zorder=4)
        ax.text(
            1.02,
            position,
            f"{value_text(first, metric)} → {value_text(last, metric)}",
            transform=ax.get_yaxis_transform(),
            fontsize=7,
            va="center",
            ha="left",
            color="0.3",
        )
    ax.set_yticks(range(len(order)))
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xscale("symlog", linthresh=settings["linthresh"], linscale=0.6)
    ax.xaxis.set_major_locator(FixedLocator(settings["ticks"]))
    ax.xaxis.set_major_formatter(FuncFormatter(axis_tick))
    finite = means[list(days)].to_numpy()
    finite = finite[np.isfinite(finite)]
    low = min(-0.5 * settings["linthresh"], 1.2 * finite.min()) if finite.size else -settings["linthresh"]
    high = 2.0 * max(finite.max(), settings["ticks"][-1] / 10) if finite.size else settings["ticks"][-1]
    ax.set_xlim(low, high)
    ax.grid(axis="x", color="0.9", zorder=0)
    ax.axvline(0, color="0.6", linewidth=0.8, zorder=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_relevance(table, order, block, days, out_path, window="all", metric="kelvin"):
    """Ranked increase of one block: RMSE increase and amplitude error increase side by side."""
    columns = [window, AMPLITUDE_WINDOW]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.36 * len(order) + 1.4), sharey=True, layout="constrained")
    for ax, column in zip(axes, columns):
        means = mean_increase(table, block, column, metric)
        if means.empty:
            ax.text(0.5, 0.5, f"window {column!r} not scored", transform=ax.transAxes, ha="center", color="0.5")
        else:
            draw_lead_change(ax, means, order, days, metric)
        ax.set_xlabel(column_label(column, metric))
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([variable_label(name) for name in order])
    fig.suptitle(
        f"{BLOCK_LABELS.get(block, block)}; filled: forecast day {days[0]}, hollow: day {days[1]} "
        f"(red grows, blue shrinks)",
        fontsize=9.5,
        x=0.02,
        ha="left",
    )
    fig.savefig(out_path)
    plt.close(fig)


def plot_leads(table, order, block, out_path, window="all", metric="kelvin"):
    """Increase per variable and forecast day; every row coloured on its own range, the
    absolute values printed."""
    means = mean_increase(table, block, window, metric)
    days = sorted(means.columns)
    means = means.reindex(index=order, columns=days)
    row_range = (means.max(axis=1) - means.min(axis=1)).replace(0, np.nan)
    scaled = means.sub(means.min(axis=1), axis=0).div(row_range, axis=0)
    fig, ax = plt.subplots(figsize=(0.5 * len(days) + 3.0, 0.34 * len(order) + 1.6), layout="constrained")
    ax.imshow(scaled.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    for row in range(len(order)):
        for col in range(len(days)):
            value = means.values[row, col]
            if not np.isfinite(value):
                continue
            color = "white" if scaled.values[row, col] > 0.65 else "black"
            ax.text(col, row, value_text(value, metric), ha="center", va="center", fontsize=6.5, color=color)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days)
    ax.set_xlabel("forecast day")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([variable_label(name) for name in order])
    ax.set_title(
        f"{column_label(window, metric)}, {BLOCK_LABELS.get(block, block)}\ncolour: each row scaled to its own range",
        fontsize=9.5,
        loc="left",
    )
    fig.savefig(out_path)
    plt.close(fig)
