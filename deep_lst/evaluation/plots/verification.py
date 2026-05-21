from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.plots._design import _LABEL_ALIASES, RC
from deep_lst.evaluation.plots._design import CASE_COLORS as _CASE_COLORS
from deep_lst.evaluation.plots._design import DEFAULT_COLOR as _DEFAULT_COLOR


def regression_parity(
    ds: xr.Dataset,
    variable: str,
    save_dir: Path,
    init_time: int = 0,
    n_lead_times: int = None,
) -> None:
    """Parity scatter plot coloured by lead time.

    Args:
        ds: ForecastResult Dataset with dims (init_time, lead_time, lat, lon).
        variable: Variable name, e.g. "LST".
        save_dir: Directory in which to save the PNG.
        init_time: Integer index into init_time dimension.
        n_lead_times: Number of lead times to include (None = all).
    """
    plt.rcParams.update(RC)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lead_times = ds.lead_time if n_lead_times is None else ds.lead_time[:n_lead_times]
    colors = plt.cm.viridis(np.linspace(0, 1, len(lead_times)))

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, lt in enumerate(lead_times):
        ax.scatter(
            ds[f"{variable}_obs"].isel(init_time=init_time, lead_time=i).values.ravel(),
            ds[f"{variable}_pred"].isel(init_time=init_time, lead_time=i).values.ravel(),
            c=[colors[i]],
            label=f"LT {i}",
            alpha=0.4,
            s=10,
        )

    lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("SEVIRI")
    ax.set_ylabel("Forecast")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f"{variable}_regression_parity.png", dpi=150, bbox_inches="tight")
    plt.close()


def activity_plot(
    references: dict[str, xr.Dataset],
    case_label: str,
    save_path,
    eval_region: str,
) -> None:
    """Anomaly activity (pred std) by forecast-day bucket, one line per reference.

    Unlike skill_plot there is no break-even line and no hard upper y-limit,
    because activity is an absolute quantity in Kelvin, not a bounded skill score.
    """
    plt.rcParams.update(RC)
    first = next(iter(references.values()))
    lead_td = pd.TimedeltaIndex(first["lead_time"].values)
    lead_hours = np.array([int(lt.total_seconds() / 3600) for lt in lead_td])

    DAY_BUCKETS = [(1, 24), (25, 48), (49, 72), (73, 96), (97, 120), (121, 144), (145, 168)]
    x, day_labels = [], []
    for day_idx, (lo, hi) in enumerate(DAY_BUCKETS, start=1):
        if ((lead_hours >= lo) & (lead_hours <= hi)).any():
            day_labels.append(f"Day {day_idx}")
            x.append(day_idx - 1)

    if not x:
        return
    x = np.array(x)

    fig, ax = plt.subplots(figsize=(max(5, len(x) * 1.2), 4.5))

    for ref_label, ds_ref in references.items():
        vals = ds_ref["activity_q50"].values
        q_lo = ds_ref["activity_q05"].values
        q_hi = ds_ref["activity_q95"].values

        day_median, day_q_lo, day_q_hi = [], [], []
        for lo, hi in DAY_BUCKETS:
            mask = (lead_hours >= lo) & (lead_hours <= hi)
            if not mask.any():
                continue
            day_median.append(float(np.nanmedian(vals[mask])))
            day_q_lo.append(float(np.nanmean(q_lo[mask])))
            day_q_hi.append(float(np.nanmean(q_hi[mask])))

        c = _CASE_COLORS.get(ref_label, _DEFAULT_COLOR)
        display_label = _LABEL_ALIASES.get(ref_label, ref_label)
        ax.plot(x, day_median, color=c, linewidth=2, marker="o", markersize=8, label=display_label)
        ax.fill_between(x, day_q_lo, day_q_hi, color=c, alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, fontweight="bold")
    ax.set_ylabel("Anomaly Std (K)", fontweight="bold")
    ax.legend(title="Reference", loc="upper right", borderaxespad=0, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"  Saved: {save_path}")
    plt.close(fig)
