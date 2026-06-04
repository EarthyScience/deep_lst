from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

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
    save_path: Path = None,
) -> None:
    """Median line (+ optional q05–q95 band) of a score over the init-time axis.

    Series-agnostic: each entry in ``series`` is one labelled curve
    ``{label: {"x", "q05", "q50", "q95"}}``. Used to overlay lead times now;
    the same helper takes classes or regions later without change.

    ``x_kind="date"`` renders a datetime x-axis with monthly ticks;
    ``"hour"`` renders an integer 0–23 hour-of-day axis. Set ``show_band=False``
    to drop the bands and keep only the median lines.
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

    if x_kind == "hour":
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(-0.5, 23.5)
    elif x_kind == "date":
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
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
