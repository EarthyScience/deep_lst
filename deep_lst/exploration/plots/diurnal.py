"""Diurnal LST cycle plot for exploration.

plot_diurnal_cycle(diurnal, period_labels, save_path, nan_frac, descriptors, solar_offsets)
    2×4 grid of diurnal LST panels, one per eval site, shared y-axis and colorbar.
    Background shading: mean ± 2 std across all period-means at each hour.
    Lines: per-period mean, Blues gradient light → dark = start → end of year.
    Best period (lowest NaN fraction per site) drawn in purple.
    Vertical gray lines: T_max (solid) and T_min (dashed) in UTC.
    Red lines: warming-rate window (solid) and cooling-rate window (dashed) fit to best period.
"""

import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _month_period_ticks(period_names, period_labels):
    """First period index and month abbreviation for each calendar month."""
    seen = {}
    for i, name in enumerate(period_names):
        month = int(period_labels[name].split("–")[0].strip().split("-")[1])
        if month not in seen:
            seen[month] = i
    return list(seen.values()), [calendar.month_abbr[m] for m in seen]


def plot_diurnal_cycle(
    diurnal, period_labels, save_path, nan_frac=None, descriptors=None, solar_offsets=None, pop_slopes=None
):
    eval_sites = list(diurnal.keys())
    period_names = list(next(iter(diurnal.values())).keys())
    n = len(period_names)
    hours = np.arange(24)
    cmap = plt.cm.Blues
    shades = np.linspace(0.35, 0.9, n)

    tick_pos, tick_labels = _month_period_ticks(period_names, period_labels)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True, sharex=True)

    for ax, site in zip(axes.flat, eval_sites):
        site_means = np.array([np.nanmean(diurnal[site][name], axis=0) for name in period_names])
        env_mean = np.nanmean(site_means, axis=0)
        env_std = np.nanstd(site_means, axis=0)

        best_idx = None
        if nan_frac is not None:
            best_period = min(nan_frac[site], key=lambda n: nan_frac[site][n].mean())
            best_idx = period_names.index(best_period)

        ax.fill_between(hours, env_mean - 2 * env_std, env_mean + 2 * env_std, color=cmap(0.2), alpha=0.4, linewidth=0)

        # Descriptor overlays drawn before period lines so they sit in the background
        if descriptors is not None and solar_offsets is not None and site in descriptors:
            desc = descriptors[site]
            offset = solar_offsets[site]

            t_max_utc = (desc["t_max_solar"][0] - offset) % 24
            t_min_utc = (desc["t_min_solar"][0] - offset) % 24
            ax.axvline(t_max_utc, color="gray", linewidth=0.8, alpha=0.6, zorder=1)
            ax.axvline(t_min_utc, color="gray", linewidth=0.8, alpha=0.6, linestyle="--", zorder=1)

            if pop_slopes is not None:
                warm_idx = (np.arange(7, 14) - round(offset)).astype(int) % 24
                cool_idx = (np.arange(14, 22) - round(offset)).astype(int) % 24
                warm_sort = np.sort(warm_idx)
                cool_sort = np.sort(cool_idx)
                m_w = pop_slopes["warming"][0]
                m_c = pop_slopes["cooling"][0]
                # intercept anchored to the annual-mean curve at the window centre
                b_w = np.mean(env_mean[warm_idx]) - m_w * np.mean(warm_idx)
                b_c = np.mean(env_mean[cool_idx]) - m_c * np.mean(cool_idx)
                ax.plot(warm_sort, m_w * warm_sort + b_w, color="red", linewidth=1.5, zorder=4)
                ax.plot(cool_sort, m_c * cool_sort + b_c, color="red", linewidth=1.5, linestyle="--", zorder=4)

        for i, name in enumerate(period_names):
            color = "#8e44ad" if i == best_idx else cmap(shades[i])
            lw = 1.5 if i == best_idx else 1.0
            zorder = 3 if i == best_idx else 2
            ax.plot(hours, site_means[i], color=color, linewidth=lw, alpha=0.9, zorder=zorder)

        ax.set_title(site, fontsize=10)
        ax.set_xticks(np.arange(0, 24, 6))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Mean LST (K)")
    for ax in axes[1, :]:
        ax.set_xlabel("Hour of day (UTC)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.invert_yaxis()  # Jan at top, Dec at bottom

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig
