from pathlib import Path

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
