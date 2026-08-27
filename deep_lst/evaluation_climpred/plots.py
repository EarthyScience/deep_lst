import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D

COLORS = {"clim_smooth": "#256abf", "persistence": "#d95926", "era5land": "#199e70"}

# The two example pixels carried through the results: Sahel (arid) and Yangambi (wet
# tropics) bracket the spatial skill distribution -- p76 and p07 of the per-pixel
# day-10 skill -- so they anchor the aggregate curves to the two case studies.
SITE_STYLES = {"Sahel": "#0b0b0b", "Yangambi": "#52514e"}

# Entities in the site time series. Climatology keeps the colour it carries
# everywhere else; the model takes a hue no baseline uses.
MODEL_COLOR = "#e34948"
CLIMATOLOGY_COLOR = "#256abf"
OBS_COLOR = "#333333"

# The spatial spread and the named sites are drawn for one baseline only -- three
# overlapping bands compound into colours that read as extra series. Climatology is
# the reference the forecast-limit maps use.
SPATIAL_BASELINE = "clim_smooth"

BASELINE_LABELS = {"clim_smooth": "Climatology", "persistence": "Persistence", "era5land": "Era5Land"}

REGION_STYLES = {
    "kg_tropical": ("Tropical", "#1b9e77", "#0c5c46"),
    "kg_arid": ("Arid", "#d95f02", "#8c3d00"),
    "kg_temperate": ("Temperate", "#7570b3", "#3d3a75"),
}


def _map_extent(lon, lat, margin=0.3):
    """Extent hugging the drawn pixels: continent-wide for the training selections,
    the tile itself for a dense box region."""
    return [float(lon.min()) - margin, float(lon.max()) + margin, float(lat.min()) - margin, float(lat.max()) + margin]


def _map_ax(ax, extent):
    ax.coastlines(lw=0.5)
    ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="grey")
    ax.set_extent(extent, crs=ccrs.PlateCarree())


_SKILL_NEUTRAL = 0.05  # |RMSE-SS| below this is drawn as "no better than the baseline"
_BIAS_NEUTRAL = 0.25  # |bias| in K below this is drawn as "unbiased"


def _diverging_scale(limit, neutral, red_high=False, steps=4):
    """Discrete diverging scale with an explicit grey neutral band.

    Discrete bins survive being printed small far better than a continuous ramp,
    and giving the neutral value a band with a width -- rather than a midpoint that
    washes out against the page -- makes "no different" readable at a glance.

    red_high flips which pole is red. Skill wants red low (worse than the baseline);
    a temperature bias wants red high, because red=warm/blue=cold is the convention
    every reader already has. The arms are reversed as well as swapped so that
    saturation keeps tracking magnitude in both directions."""
    arm = np.linspace(neutral, limit, steps + 1)
    edges = np.concatenate([-arm[::-1], arm])
    reds = plt.get_cmap("RdBu")(np.linspace(0.02, 0.32, steps))
    blues = plt.get_cmap("RdBu")(np.linspace(0.68, 0.98, steps))
    low, high = (blues[::-1], reds[::-1]) if red_high else (reds, blues)
    cmap = ListedColormap(list(low) + ["#c9c8c1"] + list(high))
    cmap.set_under(low[0])
    cmap.set_over(high[-1])
    return cmap, BoundaryNorm(edges, cmap.N)


def _limit(field, days):
    """Shared symmetric colour limit across the drawn days."""
    return float(abs(field.sel(day=days)).quantile(0.98))


def plot_rmse_maps(field, days, out_dir, suffix):
    norm = Normalize(0.0, float(field.sel(day=days).quantile(0.98)))
    _plot_pixel_day_maps(field, days, out_dir, f"_rmse{suffix}", "RMSE [K]", "viridis", norm, "max")


def plot_skill_maps(field, days, out_dir, suffix, label):
    cmap, norm = _diverging_scale(_limit(field, days), _SKILL_NEUTRAL)
    _plot_pixel_day_maps(field, days, out_dir, f"_skill{suffix}", label, cmap, norm, "both")


def plot_bias_maps(field, days, out_dir, suffix):
    cmap, norm = _diverging_scale(_limit(field, days), _BIAS_NEUTRAL, red_high=True)
    _plot_pixel_day_maps(field, days, out_dir, f"_bias{suffix}", "Bias [K]  (forecast − observed)", cmap, norm, "both")


def _plot_pixel_day_maps(field, days, out_dir, suffix, label, cmap, norm, extend="neither"):
    """One per-pixel map per forecast day, on a shared colour scale.

    Day 1 and the last day are the nowcast and medium-range windows."""
    axes_list = []
    fig, axes = plt.subplots(1, len(days), figsize=(6.5 * len(days), 6), subplot_kw={"projection": ccrs.PlateCarree()})
    extent = _map_extent(field["lon"], field["lat"])
    for ax, day in zip(np.atleast_1d(axes), days):
        _map_ax(ax, extent)
        values = field.sel(day=day)
        sc = ax.scatter(values["lon"], values["lat"], c=values, s=8, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        ax.set_title(f"Forecast day {int(day)} (leads {24 * (int(day) - 1) + 1}-{24 * int(day)} h)")
        axes_list.append(ax)
    fig.colorbar(sc, ax=axes_list, label=label, shrink=0.7, extend=extend, format="%.2f")
    fig.savefig(out_dir / f"pixel_map{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_limit_map(limit_days, horizon, out_dir, suffix):
    """Sites that never lose skill (censored, limit > horizon) are drawn as white
    circles with a grey edge; the colour gradient covers the resolved days, the
    horizon day included."""
    values = limit_days.values
    lon = limit_days["lon"].values
    lat = limit_days["lat"].values
    finite = np.isfinite(values)
    censored = finite & (values > horizon)
    resolved = finite & (values <= horizon)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": ccrs.PlateCarree()})
    _map_ax(ax, _map_extent(lon, lat))
    ax.scatter(
        lon[censored],
        lat[censored],
        s=8,
        facecolor="white",
        edgecolor="grey",
        linewidths=0.3,
        transform=ccrs.PlateCarree(),
        label=f"$>${horizon} days",
    )
    cmap = plt.get_cmap("viridis", horizon)
    norm = BoundaryNorm(np.arange(0.5, horizon + 1.5), cmap.N)
    sc = ax.scatter(
        lon[resolved], lat[resolved], c=values[resolved], s=8, cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
    )
    fig.colorbar(sc, ticks=np.arange(1, horizon + 1), shrink=0.7, label="forecast limit [days]")
    legend = ax.legend(loc="lower left", frameon=False)
    for handle in legend.legend_handles:
        handle.set_sizes([40.0])
    fig.savefig(out_dir / f"forecast_limit_map{suffix}.png", dpi=150, bbox_inches="tight")


def plot_skill_score(x, skill, out_dir, suffix, xlabel="Lead [h]", ci=None, dispersion=None, sites=None):
    """Pooled skill per baseline with its sampling CI and its 5-95% spread across
    pixels, plus the named site pixels.

    The spread is 5-95 rather than the interquartile range because failing pixels are
    a minority at every lead (14% below zero at day 10), so an IQR band never reaches
    them and the figure reads as a model that never fails.

    Band and median carry one legend entry each rather than one per baseline -- the
    fill colour already says which baseline it belongs to."""
    fig, ax = plt.subplots(figsize=(7, 4))
    lines, spread = [], []
    for kind, color in COLORS.items():
        (line,) = ax.plot(x, skill.sel(baseline=kind), color=color, lw=1.4, label=BASELINE_LABELS[kind])
        lines.append(line)
        if ci is not None:
            # 0.013-0.030 wide against a unit axis -- thinner than the line drawn on
            # top of it. Kept at full strength so it resolves where it can; that it
            # stays invisible IS the result (sampling noise is an order of magnitude
            # below the spatial spread).
            low = ci.sel(baseline=kind, quantile=0.025)
            high = ci.sel(baseline=kind, quantile=0.975)
            ax.fill_between(x, low, high, color=color, alpha=0.45, lw=0)
        if dispersion is not None:
            low_px = dispersion.sel(baseline=kind, quantile=0.05)
            high_px = dispersion.sel(baseline=kind, quantile=0.95)
            band = ax.fill_between(x, low_px, high_px, color=color, alpha=0.08, lw=0)
            median_px = dispersion.sel(baseline=kind, quantile=0.5)
            (median,) = ax.plot(x, median_px, color=color, ls="--", lw=0.9, alpha=0.6)
            if kind == SPATIAL_BASELINE:
                band.set_label("5–95 % pixel spread")
                median.set_label("pixel median")
                spread = [band, median]
    site_lines = []
    if sites is not None:
        for name, values in sites.items():
            (line,) = ax.plot(x, values, color=SITE_STYLES[name], lw=1.2, label=name)
            site_lines.append(line)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_ylim(top=1.0)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("RMSE-SS", fontweight="bold")
    # Two columns: baselines on the left, the pixel-level references on the right.
    # Matplotlib splits the handles evenly, so the shorter column is padded with
    # blanks to keep the grouping.
    left, right = lines, spread + site_lines
    rows = max(len(left), len(right))
    blank = Line2D([], [], linestyle="none", label=" ")
    left = left + [blank] * (rows - len(left))
    right = right + [blank] * (rows - len(right))
    ax.legend(handles=left + right, frameon=False, ncol=2, loc="upper right")
    fig.savefig(out_dir / f"skill_score_by_lead{suffix}.png", dpi=150, bbox_inches="tight")


def _share_limits(axes, symmetric=False):
    """One y range across a family of panels, so the seasons stay comparable."""
    low = min(ax.get_ylim()[0] for ax in axes)
    high = max(ax.get_ylim()[1] for ax in axes)
    if symmetric:
        high = max(abs(low), abs(high))
        low = -high
    for ax in axes:
        ax.set_ylim(low, high)


def plot_site(site, chosen, pred, observed, climatology, lookback_obs, lead, lookback, init_times, out_dir):
    """One panel per season, with the lookback window shaded ahead of lead 0, and a
    half-height anomaly subpanel beneath each.

    Observations are drawn as points joined by a hairline: the line makes the cloud
    gaps obvious instead of letting the eye interpolate across them. The subpanel
    repeats the same forecast against the same observations as departures from the
    climatology, where climatology is zero by definition -- the state panel is
    dominated by the diurnal cycle, which both the model and the climatology get
    right, so only the anomalies show whether the departures are predicted."""
    days = lead / 24
    lookback_days = np.arange(-lookback, 0) / 24
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 3, 1.5]})
    state_axes, anomaly_axes = [], []
    for position, (season, pick) in enumerate(chosen.items()):
        row, column = 2 * (position // 2), position % 2
        ax, anomaly_ax = axes[row, column], axes[row + 1, column]
        state_axes.append(ax)
        anomaly_axes.append(anomaly_ax)
        index = pick["index"]

        ax.axvspan(-lookback / 24, 0.0, color="#c9c8c1", alpha=0.35, lw=0)
        ax.plot(lookback_days, lookback_obs[index], color=OBS_COLOR, lw=0.6, alpha=0.8)
        ax.scatter(lookback_days, lookback_obs[index], color=OBS_COLOR, s=3, zorder=3)
        ax.plot(days, climatology[index], color=CLIMATOLOGY_COLOR, ls="--", lw=1.4, label="Climatology")
        ax.plot(days, pred[index], color=MODEL_COLOR, lw=1.6, label="Forecast")
        ax.plot(days, observed[index], color=OBS_COLOR, lw=0.6, alpha=0.8, zorder=3)
        ax.scatter(days, observed[index], color=OBS_COLOR, s=3, zorder=3, label="Observed")
        ax.axvline(0.0, color="#52514e", lw=0.8)
        ax.set_title(f"{season} — init {str(init_times[index])[:13]}", fontsize=10)
        ax.text(
            0.02,
            0.04,
            f"clear {100 * pick['clear']:.0f}%   day-10 RMSE {pick['rmse_model']:.2f} K "
            f"vs clim {pick['rmse_clim']:.2f} K   (p{100 * pick['percentile']:.0f}, "
            f"n={pick['scored']})",
            transform=ax.transAxes,
            fontsize=7,
            color="#52514e",
        )

        anomaly_ax.axvspan(-lookback / 24, 0.0, color="#c9c8c1", alpha=0.35, lw=0)
        anomaly_ax.axhline(0.0, color=CLIMATOLOGY_COLOR, ls="--", lw=1.2)
        anomaly_ax.plot(days, pred[index] - climatology[index], color=MODEL_COLOR, lw=1.4)
        anomaly_ax.plot(days, observed[index] - climatology[index], color=OBS_COLOR, lw=0.6, alpha=0.8, zorder=3)
        anomaly_ax.scatter(days, observed[index] - climatology[index], color=OBS_COLOR, s=3, zorder=3)
        anomaly_ax.axvline(0.0, color="#52514e", lw=0.8)

    # The two families cannot share a y axis -- state is ~300 K, anomaly ~0 K -- so
    # link them within each family instead, keeping the seasons comparable.
    _share_limits(state_axes)
    _share_limits(anomaly_axes, symmetric=True)
    for ax in axes[-1]:
        ax.set_xlabel("forecast day", fontweight="bold")
    for ax in axes[::2, 0]:
        ax.set_ylabel("LST [K]", fontweight="bold")
    for ax in axes[1::2, 0]:
        ax.set_ylabel("anomaly [K]", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(site, fontweight="bold")
    fig.savefig(out_dir / f"site_forecast_{site.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acc_r2(x, acc, r2, out_dir, suffix):
    """ACC, its square (potential skill) and anomaly R2; the shaded gap between
    ACC^2 and R2 is the bias + amplitude penalty (Murphy decomposition)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, acc, color="grey", ls="--", lw=1.0, label="ACC")
    ax.plot(x, acc**2, color="black", label="ACC$^2$")
    ax.plot(x, r2, color="tab:red", label="Anomaly R$^2$")
    ax.fill_between(x, r2, acc**2, color="tab:red", alpha=0.15, label="Calibration penalty")
    ax.set_xlabel("Lead [h]", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.legend(frameon=False)
    fig.savefig(out_dir / f"acc_r2_by_lead{suffix}.png", dpi=150, bbox_inches="tight")


def plot_acc_r2_regions(out, out_dir, regions):
    """ACC^2, R2 and ACC for the verified class-group windows in one figure.
    Color = region; linestyle: ACC^2 solid, R2 dashed, ACC dotted (darker shade);
    the shaded gap between ACC^2 and R2 is the calibration penalty. ACC lines are
    drawn last so they stay visible on top of the shades."""
    fig, ax = plt.subplots(figsize=(7, 4))
    region_handles = {}
    for window in regions:
        label, color, dark = REGION_STYLES[window]
        acc = out["acc_centered"].sel(window=window)
        r2 = out["r2_anomaly"].sel(window=window)
        ax.fill_between(out["lead"], r2, acc**2, color=color, alpha=0.15)
        (line_acc2,) = ax.plot(out["lead"], acc**2, color=color, ls="-")
        (line_r2,) = ax.plot(out["lead"], r2, color=color, ls="--")
        region_handles[label] = [line_acc2, line_r2]
    for window in regions:
        label, color, dark = REGION_STYLES[window]
        acc = out["acc_centered"].sel(window=window)
        (line_acc,) = ax.plot(out["lead"], acc, color=dark, ls=":")
        region_handles[label].append(line_acc)
    ax.set_xlabel("Lead [h]", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_ylim(bottom=0.0, top=1.0)
    labels = ["ACC$^2$", "R$^2$", "ACC"]
    anchors = [(1.02, 1.0 - 0.32 * position) for position in range(len(regions))]
    legends = []
    for (region, handles), anchor in zip(region_handles.items(), anchors):
        legend = ax.legend(
            handles, labels, title=region, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=anchor
        )
        legend.get_title().set_fontweight("bold")
        ax.add_artist(legend)
        legends.append(legend)
    fig.savefig(out_dir / "acc_r2_by_lead_regions.png", dpi=150, bbox_inches="tight", bbox_extra_artists=legends)
