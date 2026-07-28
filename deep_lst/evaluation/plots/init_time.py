from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from deep_lst.evaluation.plots._design import DEFAULT_COLOR, RC

_DEFAULT_COLOR = DEFAULT_COLOR


def plot_init_vs_lead(
    scores,
    metric,
    save_path: Path = None,
):
    """Heat plot with init time on x-axis and lead time on y-axis."""
    plt.rcParams.update(RC)

    m_ds = scores[metric].transpose("lead_time", "init_time").values

    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.cm.get_cmap("gray_r", 10)
    im = ax.imshow(m_ds, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1)

    # Set axis labels and ticks
    ax.set_xlabel("Initilisation time")
    ax.set_ylabel("Lead time")
    timestamps = pd.DatetimeIndex([pd.Timestamp(t) for t in scores.init_time.values])
    n_months = len({(ts.year, ts.month) for ts in timestamps})
    if n_months > 1:
        seen, tick_indices, tick_labels = set(), [], []
        for i, ts in enumerate(timestamps):
            key = (ts.year, ts.month)
            if key not in seen:
                seen.add(key)
                tick_indices.append(i)
                tick_labels.append(ts.strftime("%Y-%m"))
    else:
        seen, tick_indices, tick_labels = set(), [], []
        for i, ts in enumerate(timestamps):
            key = (ts.year, ts.month, ts.day)
            if key not in seen:
                seen.add(key)
                tick_indices.append(i)
                tick_labels.append(ts.strftime("%m-%d"))
    ax.set_xticks(tick_indices, labels=tick_labels)

    lead_hours = [int(pd.Timedelta(lt).total_seconds() // 3600) for lt in scores.lead_time.values]
    y_indices = [i for i, h in enumerate(lead_hours) if h % 24 == 0]
    ax.set_yticks(y_indices, labels=[f"{lead_hours[i]}h" for i in y_indices])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric.upper())

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def score_over_init_plot(
    series: dict[str, dict],
    *,
    xlabel: str,
    ylabel: str,
    x_kind: str = "date",
    legend_title: str | None = None,
    title: str | None = None,
    colors: dict[str, str] | None = None,
    show_band: bool = True,
    valid_shift_h: int | None = None,
    hline: float | None = None,
    save_path: Path = None,
) -> None:
    """Median line (+ optional q05–q95 band) of a score over the init-time axis.

    Series-agnostic: each entry in ``series`` is one labelled curve
    ``{label: {"x", "q05", "q50", "q95"}}``. Used to overlay lead times now;
    the same helper takes classes or regions later without change.

    ``x_kind="date"`` renders a datetime x-axis with monthly ticks;
    ``"hour"`` renders an integer 0–23 hour-of-day axis. Set ``show_band=False``
    to drop the bands and keep only the median lines. When ``x_kind="hour"`` and
    ``valid_shift_h`` is given, a secondary top axis labels the valid hour-of-day
    (init hour + shift, mod 24) — only meaningful when all overlaid curves share
    the same shift (i.e. phase-aligned leads).
    """
    plt.rcParams.update(RC)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [_DEFAULT_COLOR])
    figsize = (11, 4.5) if x_kind == "date" else (8, 4.5)
    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, d) in enumerate(series.items()):
        c = (colors or {}).get(label) or cycle[i % len(cycle)]
        if show_band:
            ax.fill_between(d["x"], d["q05"], d["q95"], color=c, alpha=0.2)
        ax.plot(d["x"], d["q50"], color=c, linewidth=2, label=label)

    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    if x_kind == "hour":
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(-0.5, 23.5)
        if valid_shift_h:
            secax = ax.secondary_xaxis("top")
            ticks = list(range(0, 24, 3))
            secax.set_xticks(ticks)
            secax.set_xticklabels([str((t + valid_shift_h) % 24) for t in ticks])
            secax.set_xlabel("Valid hour of day (UTC)")
    elif x_kind == "date":
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        if valid_shift_h:  # leave room for the secondary top axis label
            ax.set_title(title, pad=28)
        else:
            ax.set_title(title)
    ax.legend(title=legend_title, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def diurnal_lead_lines(
    da,
    *,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    cmap: str = "Blues",
    save_path: Path = None,
) -> None:
    """One median-score curve per lead time over init hour-of-day.

    ``da`` has dims ``(hour, lead_time)``. Curves are coloured by a lead-time
    gradient with a colourbar instead of a legend, and no bands — so the
    short-lead-to-long-lead evolution of the diurnal error reads at a glance.
    """
    plt.rcParams.update(RC)
    hours = da["hour"].values
    lead_h = pd.TimedeltaIndex(da["lead_time"].values).total_seconds().to_numpy() / 3600.0

    # Truncate the low (near-white) end so 0 h stays visible; same cmap+norm
    # feed both the lines and the colourbar so they match exactly.
    base = plt.get_cmap(cmap)
    cmap_obj = LinearSegmentedColormap.from_list(f"{cmap}_trunc", base(np.linspace(0.25, 1.0, 256)))
    norm = plt.Normalize(vmin=float(lead_h.min()), vmax=float(lead_h.max()))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows = da.transpose("lead_time", "hour").values
    for lt, row in zip(lead_h, rows):
        ax.plot(hours, row, color=cmap_obj(norm(lt)), linewidth=1.5)

    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label("Lead time (h)")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def diurnal_lead_heatmap(
    da,
    *,
    metric_label: str = "RMSE (K)",
    title: str | None = None,
    cmap: str = "RdPu",
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: Path = None,
) -> None:
    """Heatmap of a score over init hour-of-day (x) and lead time (y).

    ``da`` has dims ``(hour, lead_time)``. Colour encodes the score. ``vmin`` /
    ``vmax`` fix the colour scale so heatmaps stay comparable side by side
    across regions.
    """
    plt.rcParams.update(RC)
    hours = da["hour"].values
    lead_h = pd.TimedeltaIndex(da["lead_time"].values).total_seconds().to_numpy() / 3600.0
    m = da.transpose("lead_time", "hour").values

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        m,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[hours.min() - 0.5, hours.max() + 0.5, lead_h.min() - 0.5, lead_h.max() + 0.5],
    )
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Initialisation hour of day (UTC)")
    ax.set_ylabel("Lead time (h)")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax).set_label(metric_label)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
