"""Distribution (CDF) diagnostics for forecast verification, split by day/night valid hour.

These are pure forecast-array routines — they take the (init_time, lead_time, spatial) pred/obs
arrays the evaluator already exposes; no extra scores/stats are needed.

lst_cdf_plot   ECDF of LST *values* (pred per lead + observed reference), x = LST [K].
rmse_cdf_plot  ECDF across pixels of per-pixel RMSE per lead,            x = RMSE [K].

Both: two panels (daytime 10–15 h, nighttime 22–03 h LOCAL solar valid hour, from
`conditioning.time_of_day_mask`), with leads 1→24 drawn as a light→dark-blue gradient.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from deep_lst.evaluation.conditioning import time_of_day_mask
from deep_lst.evaluation.plots._design import RC

_PANELS = (
    ("Daytime (10–15 h local)", "day"),
    ("Nighttime (22–03 h local)", "night"),
)


def _ecdf(a: np.ndarray):
    a = np.sort(a[np.isfinite(a)])
    return a, (np.linspace(0, 1, a.size) if a.size else a)


def _lead_colorbar(fig, axes, leads):
    sm = cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(leads[0], leads[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Lead time [h]", fraction=0.046, pad=0.04)


def day_night_summary(pred, obs, leads=range(1, 25)):
    """Pooled RMSE [K] and valid-obs fraction [%] for the local-solar daytime (10–15 h) and
    nighttime (22–03 h) windows, over `leads`, across the holdout region. Returns
    {panel_title: (rmse, valid_pct)}. Computed once (from the evaluator's pred/obs) and passed
    to the plots as annotations."""
    leads = list(leads)
    pred = pred.isel(lead_time=leads).load()
    obs = obs.isel(lead_time=leads).load()
    out = {}
    for title, kind in _PANELS:
        window = time_of_day_mask(pred, kind)
        err = (pred - obs).where(window)
        rmse = float(np.sqrt((err**2).mean(skipna=True)))
        n_window = int(window.broadcast_like(obs).sum())
        n_valid = int(obs.where(window).notnull().sum())
        valid_pct = 100.0 * n_valid / n_window if n_window else float("nan")
        out[title] = (rmse, valid_pct)
    return out


def _annotate(ax, title, summary):
    if not summary or title not in summary:
        return
    rmse, vpct = summary[title]
    ax.text(
        0.03,
        0.70,
        f"RMSE: {rmse:.2f} K\nValid obs: {vpct:.0f}%",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
    )


def lst_cdf_plot(pred, obs, var, save_path, eval_region, leads=range(1, 25), summary=None):
    """ECDF of LST values: observed reference (black dashed) vs forecast at each lead (gradient)."""
    plt.rcParams.update(RC)
    leads = list(leads)
    # Load only the requested leads into memory once — per-lead lazy-dask .values is pathologically slow.
    pred = pred.isel(lead_time=leads).load()
    obs = obs.isel(lead_time=leads).load()
    colors = cm.Blues(np.linspace(0.3, 1.0, len(leads)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (title, kind) in zip(axes, _PANELS):
        window = time_of_day_mask(pred, kind)  # (init, lead[, lon]); broadcasts over the grid
        obs_pool = []
        for i, color in enumerate(colors):
            window_i = window.isel(lead_time=i)
            if not bool(window_i.any()):
                continue
            xs, ys = _ecdf(pred.isel(lead_time=i).where(window_i).values.ravel())
            ax.plot(xs, ys, color=color, lw=1.0)
            obs_pool.append(obs.isel(lead_time=i).where(window_i).values.ravel())
        if obs_pool:
            xs, ys = _ecdf(np.concatenate(obs_pool))
            ax.plot(xs, ys, "k--", lw=2.0, label="Observed", zorder=5)
        ax.set_title(title)
        ax.set_xlabel(f"{var} [K]")
        ax.grid(alpha=0.3)
        _annotate(ax, title, summary)
    axes[0].set_ylabel("Cumulative fraction")
    axes[0].legend(loc="upper left")
    _lead_colorbar(fig, axes, leads)
    fig.suptitle(f"{eval_region} — {var} value CDF (day/night), lead {leads[0]}–{leads[-1]} h")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def rmse_cdf_plot(pred, obs, var, save_path, eval_region, leads=range(1, 25), summary=None):
    """ECDF across pixels of per-pixel RMSE (RMSE over init times) at each lead (gradient)."""
    plt.rcParams.update(RC)
    leads = list(leads)
    err = (pred - obs).isel(lead_time=leads).load()  # load the lead subset once (lazy per-lead is slow)
    colors = cm.Blues(np.linspace(0.3, 1.0, len(leads)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (title, kind) in zip(axes, _PANELS):
        window = time_of_day_mask(err, kind)
        for i, color in enumerate(colors):
            window_i = window.isel(lead_time=i)
            if not bool(window_i.any()):
                continue
            e = err.isel(lead_time=i).where(window_i)
            with np.errstate(invalid="ignore"):
                rmse_pix = np.sqrt((e**2).mean("init_time", skipna=True))  # reduce over init → per pixel
            xs, ys = _ecdf(rmse_pix.values.ravel())
            ax.plot(xs, ys, color=color, lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(f"{var} RMSE [K]")
        ax.grid(alpha=0.3)
        _annotate(ax, title, summary)
    axes[0].set_ylabel("Cumulative fraction (pixels)")
    _lead_colorbar(fig, axes, leads)
    fig.suptitle(f"{eval_region} — per-pixel {var} RMSE CDF (day/night), lead {leads[0]}–{leads[-1]} h")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
