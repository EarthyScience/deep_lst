"""Animated GIF plots for forecast evaluation.

forecast_gif(ds, var, init_time, lead_times, save_path)
    Two-panel animation: predicted LST (left) and observed LST (right)
    stepping through lead times for a single init_time.
    Each frame title shows init_time + lead_time = valid_time.
"""

import copy
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.plots._design import RC_ANIMATION as RC


def forecast_gif(
    ds: xr.Dataset,
    var: str = "LST",
    init_time: str | pd.Timestamp | None = None,
    lead_times: list[str] | None = None,
    save_path: Path = None,
    fps: int = 2,
    eval_region: str = "",
    cmap: str = "RdYlBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
) -> animation.FuncAnimation:
    """Two-panel animated GIF: forecast vs observed LST over the horizon.

    Parameters
    ----------
    ds : xr.Dataset
        ForecastResult with dims (init_time, lead_time, lat, lon).
        Must contain {var}_pred and {var}_obs.
    var : str
        Variable to animate, e.g. "LST".
    init_time : str or Timestamp, optional
        Which init_time to animate. If None, the first init_time is used.
        Accepts any string pandas can parse, e.g. "2024-01-07T06:00".
    lead_times : list of str, optional
        Subset of lead times to include, e.g. ["1h", "2h", "3h"].
        Default: all lead_times present in ds (excluding 0h NaN row).
    save_path : Path
        Output path for the .gif file. Parent dir is created if needed.
    fps : int
        Frames per second in the output GIF.
    eval_region : str
        Used in figure title.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Colorscale limits. Default: symmetric range derived from obs across
        all selected frames (robust 2nd–98th percentile).
    """
    plt.rcParams.update(RC)

    pred_all = ds[f"{var}_pred"]
    obs_all = ds[f"{var}_obs"]

    # --- Select init_time ---
    if init_time is None:
        # Find the first init_time that has at least some valid (non-NaN) predictions
        it = ds["init_time"].values[0]  # default fallback
        for _it in ds["init_time"].values:
            if np.isfinite(pred_all.sel(init_time=_it).values).any():
                it = _it
                break
        else:
            print(
                "  [warn] forecast_gif: no init_time found with valid predictions; "
                "using the first one — the GIF may appear empty."
            )
    else:
        it = pd.Timestamp(init_time).to_datetime64()
    pred_all = pred_all.sel(init_time=it)
    obs_all = obs_all.sel(init_time=it)

    # --- Select lead_times (skip 0h which is NaN for rollout) ---
    if lead_times is not None:
        lts = [pd.Timedelta(lt) for lt in lead_times]
        pred_all = pred_all.sel(lead_time=lts)
        obs_all = obs_all.sel(lead_time=lts)
    else:
        # Drop lead_time=0h if all-NaN (rollout padding)
        nonzero = [lt for lt in ds["lead_time"].values if lt != np.timedelta64(0, "ns")]
        if len(nonzero) < len(ds["lead_time"]):
            pred_all = pred_all.sel(lead_time=nonzero)
            obs_all = obs_all.sel(lead_time=nonzero)

    n_frames = pred_all.sizes["lead_time"]
    lats = ds["lat"].values
    lons = ds["lon"].values

    # --- Colorscale ---
    if vmin is None or vmax is None:
        # Try obs first, fall back to pred, then hard-coded defaults
        finite_obs = obs_all.values[np.isfinite(obs_all.values)]
        finite_pred = pred_all.values[np.isfinite(pred_all.values)]
        ref = finite_obs if len(finite_obs) > 0 else finite_pred
        if len(ref) > 0:
            _vmin = float(np.nanpercentile(ref, 2))
            _vmax = float(np.nanpercentile(ref, 98))
        else:
            _vmin, _vmax = 270.0, 330.0
        vmin = vmin if vmin is not None else _vmin
        vmax = vmax if vmax is not None else _vmax

    init_str = pd.Timestamp(it).strftime("%Y-%m-%d %H:%M UTC")

    # --- Colormap — render NaN/masked pixels as light gray ---
    cmap_obj = copy.copy(plt.get_cmap(cmap))
    cmap_obj.set_bad(color="lightgray")

    # --- Build figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    suptitle = fig.suptitle("", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, f"init: {init_str}", ha="center", va="bottom", fontsize=10, color="#555555")

    im_pred = axes[0].pcolormesh(
        lons, lats, pred_all.isel(lead_time=0).values, cmap=cmap_obj, vmin=vmin, vmax=vmax, shading="auto"
    )
    im_obs = axes[1].pcolormesh(
        lons, lats, obs_all.isel(lead_time=0).values, cmap=cmap_obj, vmin=vmin, vmax=vmax, shading="auto"
    )

    for ax, label in zip(axes, [f"{var} predicted", f"{var} observed"]):
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(label)
        ax.set_aspect("equal")

    plt.colorbar(im_obs, ax=axes, orientation="vertical", fraction=0.03, pad=0.04, label=f"{var} (K)")

    lead_td = pred_all["lead_time"].values

    def _update(frame_idx):
        lt = lead_td[frame_idx]
        valid_time = pd.Timestamp(it) + pd.Timedelta(lt)
        lt_h = int(pd.Timedelta(lt).total_seconds() / 3600)

        pred_frame = pred_all.isel(lead_time=frame_idx).values
        obs_frame = obs_all.isel(lead_time=frame_idx).values

        im_pred.set_array(pred_frame.ravel())
        im_obs.set_array(obs_frame.ravel())

        suptitle.set_text(f"{eval_region}  |  {valid_time.strftime('%Y-%m-%d %H:%M UTC')}  (+{lt_h}h)")
        return im_pred, im_obs, suptitle

    anim = animation.FuncAnimation(fig, _update, frames=n_frames, interval=int(1000 / fps), blit=False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Saved: {save_path}")

    plt.close(fig)
    return anim
