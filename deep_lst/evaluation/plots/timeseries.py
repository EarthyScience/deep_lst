from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.metrics import to_valid_time
from deep_lst.evaluation.plots._design import OBS_COLOR, RC


def timeseries_plot(
    ds: xr.Dataset,
    var: str = "LST",
    leads_h: tuple[int, ...] = (1, 3, 24),
    save_path: Path = None,
    title: str = None,
    eval_region: str = "",
) -> plt.Figure:
    """Region-mean obs by valid_time, with region-mean forecasts at several lead times overlaid.

    Each forecast lead is plotted at its own valid time (valid = init + lead), so the same valid
    time is reached from earlier inits at longer lead; the spread between the lead series shows
    error growth in the time domain. Obs is the region-mean SEVIRI series (taken at the first
    available lead). Leads not present in the horizon are skipped.
    """
    plt.rcParams.update(RC)
    pred_da = ds[f"{var}_pred"]
    obs_da = ds[f"{var}_obs"]

    lead_hours = (pred_da["lead_time"] / np.timedelta64(1, "h")).astype(int).values
    leads_h = [h for h in leads_h if h in lead_hours]
    if not leads_h:
        return None

    obs_lead = pd.Timedelta(hours=leads_h[0])  # obs by valid_time at the first available lead (lead 0 is a placeholder)
    obs = to_valid_time(obs_da.sel(lead_time=obs_lead), lead_time=obs_lead)
    obs_mean = obs.mean(dim=[d for d in obs.dims if d != "valid_time"], skipna=True)
    obs_times = pd.DatetimeIndex(obs_mean["valid_time"].values)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        obs_times,
        obs_mean.values,
        color=OBS_COLOR,
        linewidth=1.0,
        marker=".",
        markersize=3,
        label="SEVIRI obs",
        zorder=3,
    )

    cmap = plt.get_cmap("viridis")
    for i, h in enumerate(leads_h):
        lead = pd.Timedelta(hours=h)
        pred = to_valid_time(pred_da.sel(lead_time=lead), lead_time=lead)
        pred_mean = pred.mean(dim=[d for d in pred.dims if d != "valid_time"], skipna=True)
        times = pd.DatetimeIndex(pred_mean["valid_time"].values)
        color = cmap(i / max(1, len(leads_h) - 1))
        ax.plot(
            times,
            pred_mean.values,
            linestyle="none",
            marker=".",
            markersize=5,
            color=color,
            label=f"forecast, lead={h} h",
            zorder=4,
        )

    ax.set_xlabel("valid_time (UTC)")
    ax.set_ylabel(f"{var} (K)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title is None:
        title = f"{eval_region} — region-mean {var} by valid time"
    ax.set_title(title)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
