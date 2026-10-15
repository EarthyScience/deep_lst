import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .masks import KG_GROUPS, LC_GROUPS

# Baseline hues, muted so the regime hues of the summary figures lead. Checked
# 2026-09-16: contrast on white 6.9 / 4.1 / 4.4; min pairwise dE 36 (normal vision),
# 19 (deutan), 13 (protan: the blue-teal pair is the weak one; greener teals collide
# with the brown instead); >= 44 from every regime hue. The earlier strong palette
# (#256abf / #d95926 / #199e70) had protan 28 but competed with the regime hues.
COLORS = {"clim_smooth": "#3b5b8c", "persistence": "#3a8a80", "era5land": "#8a7460"}

MODEL_LABEL = "DeepLST"
# Model line next to the three baseline hues: the model is the one neutral series,
# which also keeps it clear of the muted baselines.
MODEL_LINE = "#0b0b0b"

# The two example pixels carried through the results: Sahel (arid) and Yangambi (wet
# tropics) bracket the spatial skill distribution -- p76 and p07 of the per-pixel
# day-10 skill -- so they anchor the aggregate curves to the two case studies.
SITE_STYLES = {"Sahel": "#0b0b0b", "Yangambi": "#52514e"}

# Entities in the site time series. Climatology keeps the colour it carries
# everywhere else; the model takes a hue no baseline uses.
MODEL_COLOR = "#e34948"
CLIMATOLOGY_COLOR = COLORS["clim_smooth"]
OBS_COLOR = "#333333"

# The spatial spread and the named sites are drawn for one baseline only -- three
# overlapping bands compound into colours that read as extra series. Climatology is
# the reference the forecast-limit maps use.
SPATIAL_BASELINE = "clim_smooth"

BASELINE_LABELS = {"clim_smooth": "Climatology", "persistence": "Persistence", "era5land": "Era5Land"}

# Region hues (label, colour, darker shade for the ACC lines). Deliberately far from the
# baseline hues above -- regions and baselines show different things and must not be
# confused across figures. Validated 2026-09-08: within the triple normal-vision dE 21.6,
# CVD 11.1; nearest baseline hue dE >= 44 with the muted palette (the old green/orange
# region hues sat at dE 1.0 from the earlier Era5Land/Persistence hues).
REGION_STYLES = {
    "kg_tropical": ("Tropical", "#e7298a", "#8f1955"),
    "kg_arid": ("Arid", "#b8860b", "#725306"),
    "kg_temperate": ("Temperate", "#7b3294", "#4c1f5b"),
}
REGION_COLORS = {label: color for label, color, dark in REGION_STYLES.values()}
# Column order of the per-regime example forecasts.
REGIME_COLUMNS = ("kg_arid", "kg_tropical", "kg_temperate")

# Print text sizes of every figure: axis labels, titles and legends 20, tick labels
# and other text 18. evaluate applies them to the rc (apply_print_style); the
# summary figures and the regime example set the same values in their own rc
# context, so they render alike from any entry point.
PRINT_RC = {
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "figure.labelsize": 20,
    "legend.fontsize": 20,
    "legend.title_fontsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
}


def apply_print_style():
    plt.rcParams.update(PRINT_RC)


def _map_extent(lon, lat, margin_fraction=0.005):
    """Extent hugging the drawn pixels: continent-wide for the training selections,
    the tile itself for a dense box region. The margin is a fraction of the span,
    so the dense tiles get under a pixel (no empty band with borders showing
    around the data) while the continental maps keep their outermost dots whole."""
    lon_low, lon_high = float(lon.min()), float(lon.max())
    lat_low, lat_high = float(lat.min()), float(lat.max())
    margin = margin_fraction * max(lon_high - lon_low, lat_high - lat_low)
    return [lon_low - margin, lon_high + margin, lat_low - margin, lat_high + margin]


def _map_ax(ax, extent, label_sides=("left", "bottom")):
    """Coastlines and borders, no frame; lon/lat labels on `label_sides` only."""
    ax.coastlines(lw=0.5)
    ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="grey")
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.spines["geo"].set_visible(False)
    gridlines = ax.gridlines(draw_labels=list(label_sides))
    gridlines.xlines = False
    gridlines.ylines = False
    gridlines.xlabel_style = {"size": plt.rcParams["xtick.labelsize"], "color": "#52514e"}
    gridlines.ylabel_style = {"size": plt.rcParams["ytick.labelsize"], "color": "#52514e"}


def _diverging_scale(limit, red_high=False):
    """Continuous symmetric diverging scale, white at zero.

    red_high flips which pole is red. Skill wants red low (worse than the baseline);
    a temperature bias wants red high, because red=warm/blue=cold is the convention
    every reader already has."""
    cmap = plt.get_cmap("RdBu_r" if red_high else "RdBu")
    return cmap, Normalize(-limit, limit)


def _limit(field, days):
    """Shared symmetric colour limit across the drawn days."""
    return float(abs(field.sel(day=days)).quantile(0.98))


def plot_rmse_maps(field, days, out_dir, suffix):
    norm = Normalize(0.0, float(field.sel(day=days).quantile(0.98)))
    _plot_pixel_day_maps(field, days, out_dir, f"_rmse{suffix}", "RMSE [K]", "viridis", norm, "max")


def plot_skill_maps(field, days, out_dir, suffix, label, per_day=False, limit=None):
    """`limit` fixes the colour range (shared across case studies); None = data-driven."""
    cmap, norm = _diverging_scale(limit or _limit(field, days))
    _plot_pixel_day_maps(field, days, out_dir, f"_skill{suffix}", label, cmap, norm, "both", per_day)


def plot_bias_maps(field, days, out_dir, suffix, per_day=False, limit=None):
    """`limit` fixes the colour range in K (shared across case studies); None = data-driven."""
    cmap, norm = _diverging_scale(limit or _limit(field, days), red_high=True)
    label = "Bias [K]  (forecast − observed)"
    _plot_pixel_day_maps(field, days, out_dir, f"_bias{suffix}", label, cmap, norm, "both", per_day)


def _draw_pixel_day(ax, field, day, extent, cmap, norm):
    """One per-pixel map of `field` at forecast `day`; returns the scatter for the colorbar."""
    _map_ax(ax, extent)
    values = field.sel(day=day)
    sc = ax.scatter(values["lon"], values["lat"], c=values, s=8, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    ax.set_title(f"Forecast day {int(day)} (leads {24 * (int(day) - 1) + 1}-{24 * int(day)} h)")
    return sc


def _plot_pixel_day_maps(field, days, out_dir, suffix, label, cmap, norm, extend="neither", per_day=False):
    """The `days` panels side by side on one colour scale (day 1 and the last day are
    the nowcast and medium-range windows). With `per_day`, also one figure per
    forecast day on that same scale in out_dir/days/ (2.6 s each -- requested for
    the total-window skill and bias only)."""
    extent = _map_extent(field["lon"], field["lat"])
    fig, axes = plt.subplots(1, len(days), figsize=(8.5 * len(days), 8), subplot_kw={"projection": ccrs.PlateCarree()})
    axes_list = list(np.atleast_1d(axes))
    for ax, day in zip(axes_list, days):
        sc = _draw_pixel_day(ax, field, day, extent, cmap, norm)
    fig.colorbar(sc, ax=axes_list, label=label, shrink=0.7, extend=extend, format="%.2f")
    fig.savefig(out_dir / f"pixel_map{suffix}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not per_day:
        return
    day_dir = out_dir / "days"
    day_dir.mkdir(exist_ok=True)
    for day in field["day"].values:
        fig, ax = plt.subplots(figsize=(8.5, 8), subplot_kw={"projection": ccrs.PlateCarree()})
        sc = _draw_pixel_day(ax, field, day, extent, cmap, norm)
        fig.colorbar(sc, ax=ax, label=label, shrink=0.7, extend=extend, format="%.2f")
        fig.savefig(day_dir / f"pixel_map{suffix}_day{int(day):02d}.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _draw_forecast_limit(ax, limit_days, horizon, extent):
    """Forecast-limit map on `ax`. Sites that never lose skill (censored, limit >
    horizon) are drawn as white circles with a grey edge; the colour gradient covers
    the resolved days, the horizon day included. Returns the gradient scatter for
    the colorbar."""
    values = limit_days.values
    lon = limit_days["lon"].values
    lat = limit_days["lat"].values
    finite = np.isfinite(values)
    censored = finite & (values > horizon)
    resolved = finite & (values <= horizon)

    _map_ax(ax, extent)
    # Smaller and paler than the resolved pixels: they are the background the
    # coloured limits stand out from, not a fourth class.
    ax.scatter(
        lon[censored],
        lat[censored],
        s=5,
        facecolor="white",
        edgecolor="#bdbdbd",
        linewidths=0.3,
        transform=ccrs.PlateCarree(),
        label=f"$>${horizon} days",
    )
    cmap = plt.get_cmap("viridis", horizon)
    norm = BoundaryNorm(np.arange(0.5, horizon + 1.5), cmap.N)
    sc = ax.scatter(
        lon[resolved], lat[resolved], c=values[resolved], s=8, cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
    )
    legend = ax.legend(loc="lower left", frameon=False)
    for handle in legend.legend_handles:
        handle.set_sizes([40.0])
    return sc


def plot_forecast_limit_map(limit_days, horizon, out_dir, suffix):
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": ccrs.PlateCarree()})
    sc = _draw_forecast_limit(ax, limit_days, horizon, _map_extent(limit_days["lon"], limit_days["lat"]))
    fig.colorbar(sc, ticks=np.arange(1, horizon + 1), shrink=0.7, label="forecast limit [days]")
    fig.savefig(out_dir / f"forecast_limit_map{suffix}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_skill_limit_maps(skill_field, limit_days, days, horizon, out_dir, suffix, limit=None):
    """Skill maps at the two `days` on one colour scale (colorbar beneath both) next
    to the forecast-limit map with its own colorbar beneath it. The skill scale is
    the one plot_skill_maps uses (same `limit`), so the two figures agree."""
    extent = _map_extent(skill_field["lon"], skill_field["lat"])
    cmap, norm = _diverging_scale(limit or _limit(skill_field, days))
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, day in zip(axes[:2], days):
        sc_skill = _draw_pixel_day(ax, skill_field, day, extent, cmap, norm)
        ax.set_title(f"Day {int(day)}")  # short titles here; the lead range stays on the single maps
    sc_limit = _draw_forecast_limit(axes[2], limit_days, horizon, extent)
    axes[2].set_title("Forecast Limit")
    fig.colorbar(
        sc_skill,
        ax=list(axes[:2]),
        orientation="horizontal",
        shrink=0.5,
        pad=0.08,
        extend="both",
        format="%.2f",
        label="Skill improvement",
    )
    fig.colorbar(
        sc_limit,
        ax=axes[2],
        orientation="horizontal",
        shrink=0.9,
        pad=0.08,
        ticks=np.arange(1, horizon + 1),
        label="Forecast limit [days]",
    )
    fig.savefig(out_dir / f"skill_limit_maps{suffix}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_and_skill(
    x,
    rmse_model,
    rmse_baseline,
    skill,
    out_dir,
    suffix,
    xlabel="Lead [h]",
    ci=None,
    dispersion=None,
    rmse_dispersion_model=None,
    rmse_dispersion_baseline=None,
):
    """Left: RMSE of the model and of every baseline, each with its 95 % spread across
    pixels. Right: pooled skill per baseline with its sampling CI and the 95 % spread
    across pixels (2.5-97.5).

    The baselines are valid-time series, so their RMSE is flat over the lead up to
    the lead-dependent init population (year-end truncation under maximize).

    The spread is a wide quantile band rather than the interquartile range because
    failing pixels are a minority at every lead (14% below zero at day 10), so an
    IQR band never reaches them and the figure reads as a model that never fails.
    Band and median carry one legend entry each rather than one per baseline -- the
    fill colour already says which baseline it belongs to."""
    fig, (ax_rmse, ax_skill) = plt.subplots(1, 2, figsize=(14, 6))
    _draw_error(ax_rmse, x, rmse_model, rmse_baseline, xlabel, rmse_dispersion_model, rmse_dispersion_baseline)
    _draw_skill(ax_skill, x, skill, xlabel, ci, dispersion)
    fig.savefig(out_dir / f"error_skill_by_lead{suffix}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_error(ax, x, rmse_model, rmse_baseline, xlabel, rmse_dispersion_model=None, rmse_dispersion_baseline=None):
    """RMSE of the model and of every baseline, each with its 95 % spread across pixels."""
    ax.plot(x, rmse_model, color=MODEL_LINE, lw=1.6, label=MODEL_LABEL)
    top = max(float(rmse_model.max()), float(rmse_baseline.max()))
    if rmse_dispersion_model is not None:
        low_px = rmse_dispersion_model.sel(quantile=0.025)
        high_px = rmse_dispersion_model.sel(quantile=0.975)
        ax.fill_between(x, low_px, high_px, color=MODEL_LINE, alpha=0.08, lw=0, label="95 % Pixel spread")
        top = max(top, float(high_px.max()))
    for kind, color in COLORS.items():
        ax.plot(x, rmse_baseline.sel(baseline=kind), color=color, lw=1.4, label=BASELINE_LABELS[kind])
        if rmse_dispersion_baseline is not None:
            low_px = rmse_dispersion_baseline.sel(baseline=kind, quantile=0.025)
            high_px = rmse_dispersion_baseline.sel(baseline=kind, quantile=0.975)
            ax.fill_between(x, low_px, high_px, color=color, alpha=0.08, lw=0)
            top = max(top, float(high_px.max()))
    # Headroom above the highest band so the legend never sits on a line.
    ax.set_ylim(0.0, 1.3 * top)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE [K]")
    ax.legend(frameon=False, loc="upper left")


def _draw_skill(ax, x, skill, xlabel, ci=None, dispersion=None):
    """Pooled skill per baseline with its sampling CI and the 95 % spread across pixels."""
    spread = []
    for kind, color in COLORS.items():
        ax.plot(x, skill.sel(baseline=kind), color=color, lw=1.4, label=BASELINE_LABELS[kind])
        if ci is not None:
            # 0.013-0.030 wide against a unit axis -- thinner than the line drawn on
            # top of it. Kept at full strength so it resolves where it can; that it
            # stays invisible IS the result (sampling noise is an order of magnitude
            # below the spatial spread).
            low = ci.sel(baseline=kind, quantile=0.025)
            high = ci.sel(baseline=kind, quantile=0.975)
            ax.fill_between(x, low, high, color=color, alpha=0.45, lw=0)
        if dispersion is not None:
            low_px = dispersion.sel(baseline=kind, quantile=0.025)
            high_px = dispersion.sel(baseline=kind, quantile=0.975)
            band = ax.fill_between(x, low_px, high_px, color=color, alpha=0.08, lw=0)
            median_px = dispersion.sel(baseline=kind, quantile=0.5)
            (median,) = ax.plot(x, median_px, color=color, ls="--", lw=0.9, alpha=0.6)
            if kind == SPATIAL_BASELINE:
                band.set_label("95 % Pixel spread")
                median.set_label("Pixel median")
                spread = [band, median]
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_ylim(top=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Skill improvement")
    ax.legend(handles=spread, frameon=False, loc="upper right")


def plot_error_skill_anomaly(pooled, window, regions, bootstrap_daily, out_dir, xlabel="Lead time [day]"):
    """The daily error and skill panels of `window` next to the daily anomaly scores
    per region: ACC^2 solid, R^2 dashed, ACC dotted, the shaded gap = calibration
    penalty. All three read skill_daily.nc, so the x axis is the forecast day
    throughout and the anomaly scores are the exact pooled ones (moment sums)."""
    fig, (ax_rmse, ax_skill, ax_anomaly) = plt.subplots(1, 3, figsize=(19, 5.5))
    x = pooled["day"]
    _draw_error(
        ax_rmse,
        x,
        pooled["rmse_model"].sel(window=window),
        pooled["rmse_baseline"].sel(window=window),
        xlabel,
        pooled["rmse_model_dispersion"].sel(window=window),
        pooled["rmse_baseline_dispersion"].sel(window=window),
    )
    _draw_skill(
        ax_skill,
        x,
        pooled["rmse_ss"].sel(window=window),
        xlabel,
        bootstrap_daily.sel(window=window),
        pooled["rmse_ss_dispersion"].sel(window=window),
    )
    _draw_anomaly_regions(ax_anomaly, x, pooled["acc"], pooled["r2"], regions, xlabel)
    # Regions by colour outside the panel (right), scores by line style inside, in
    # the empty lower left corner (no curve starts below 0.4).
    region_handles = [Patch(color=REGION_STYLES[w][1], label=REGION_STYLES[w][0]) for w in regions]
    region_legend = ax_anomaly.legend(
        handles=region_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0)
    )
    ax_anomaly.add_artist(region_legend)
    ax_anomaly.legend(handles=_anomaly_style_handles(), frameon=False, loc="lower left")
    fig.savefig(out_dir / f"error_skill_anomaly_by_day_{window}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _anomaly_style_handles():
    """Legend handles for the line styles of the per-region anomaly panel."""
    return [
        Line2D([], [], color=INK_SECONDARY, ls="-", label="ACC$^2$"),
        Line2D([], [], color=INK_SECONDARY, ls="--", label="R$^2$"),
        Line2D([], [], color=INK_SECONDARY, ls=":", label="ACC"),
    ]


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
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 3, 1.5]})
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
        ax.set_title(f"{season} — init {str(init_times[index])[:13]}")
        ax.text(
            0.02,
            0.04,
            f"clear {100 * pick['clear']:.0f}%   day-10 RMSE {pick['rmse_model']:.2f} K "
            f"vs clim {pick['rmse_clim']:.2f} K   (p{100 * pick['percentile']:.0f}, "
            f"n={pick['scored']})",
            transform=ax.transAxes,
            fontsize="x-small",
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
        ax.set_xlabel("Forecast day", fontweight="bold")
    for ax in axes[::2, 0]:
        ax.set_ylabel("LST [K]", fontweight="bold")
    for ax in axes[1::2, 0]:
        ax.set_ylabel("Anomaly [K]")
    # Below the panels: inside a state panel the legend covers the diurnal peaks.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(site, fontweight="bold")
    fig.savefig(out_dir / f"site_forecast_{site.lower()}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regime_days(picks, days, out_dir):
    """One example forecast per climate regime (columns) at the drawn forecast
    `days` (rows): forecast, observations, climatology and the diurnal persistence
    over the valid hours 00-24 of that day. Each column is one pixel and one init at
    00 UTC. All panels share the y range, so the diurnal amplitude differs visibly
    between regimes. The baselines keep the hues of the lead figures; the forecast is
    the neutral DeepLST line (a red model line sat too close to the persistence
    orange), and the observations are white-filled points so the line stays visible
    where it runs through them.

    `picks` maps the regime window to a dict with lead (hours), valid_times, pred,
    observed, climatology and persistence (all along lead), plus lon and lat."""
    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(
            len(days), len(picks), figsize=(5.6 * len(picks), 4.0 * len(days)), sharey=True, squeeze=False
        )
        for column, (regime, pick) in enumerate(picks.items()):
            for row, day in enumerate(days):
                ax = axes[row, column]
                positions = _day_positions(pick["lead"], day)
                x = pick["valid_times"][positions]
                ax.plot(
                    x, pick["climatology"][positions], color=COLORS["clim_smooth"], ls="--", lw=2.2, label="Climatology"
                )
                ax.plot(
                    x, pick["persistence"][positions], color=COLORS["persistence"], ls=":", lw=2.4, label="Persistence"
                )
                ax.plot(x, pick["pred"][positions], color=MODEL_LINE, lw=2.2, label="Forecast")
                ax.scatter(
                    x,
                    pick["observed"][positions],
                    s=36,
                    facecolor="white",
                    edgecolor=OBS_COLOR,
                    linewidths=1.3,
                    zorder=3,
                    label="Observed",
                )
                # Date and (lat, lon) in the upper corners: the diurnal peak sits in
                # the middle of a 00-24 panel, so the corners stay free.
                ax.text(0.03, 0.93, str(x[0])[:10], transform=ax.transAxes, ha="left", va="top")
                location = f"({pick['lat']:.1f}, {pick['lon']:.1f})"
                ax.text(0.97, 0.93, location, transform=ax.transAxes, ha="right", va="top")
                # Valid hours 00-24: the first drawn hour is init + 1 h.
                ax.set_xlim(x[0] - np.timedelta64(1, "h"), x[-1])
                ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 6, 12, 18)))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            axes[0, column].set_title(REGION_STYLES[regime][0])
        for ax in axes[-1]:
            ax.set_xlabel("Valid hour [UTC]")
        for row, day in enumerate(days):
            ax = axes[row, 0]
            ax.set_ylabel("LST [K]")
            # Row label just left of the y label, positioned relative to it so the
            # tick-label width does not matter.
            ax.annotate(
                f"Day {day}",
                xy=(0, 0.5),
                xycoords=ax.yaxis.label,
                xytext=(-10, 0),
                textcoords="offset points",
                rotation=90,
                ha="right",
                va="center",
            )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout()
        name = "_".join(f"day{day:02d}" for day in days)
        fig.savefig(out_dir / f"regime_forecast_{name}.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_acc_r2(x, acc, r2, out_dir, suffix):
    """ACC, its square (potential skill) and anomaly R2; the shaded gap between
    ACC^2 and R2 is the bias + amplitude penalty (Murphy decomposition)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, acc, color="grey", ls="--", lw=1.0, label="ACC")
    ax.plot(x, acc**2, color="black", label="ACC$^2$")
    ax.plot(x, r2, color="tab:red", label="Anomaly R$^2$")
    ax.fill_between(x, r2, acc**2, color="tab:red", alpha=0.15, label="Calibration penalty")
    ax.set_xlabel("Lead [h]", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(out_dir / f"acc_r2_by_lead{suffix}.pdf", dpi=150, bbox_inches="tight")


def plot_acc_r2_regions(out, out_dir, regions):
    """ACC^2, R2 and ACC for the verified class-group windows in one figure.
    Color = region; linestyle: ACC^2 solid, R2 dashed, ACC dotted (darker shade);
    the shaded gap between ACC^2 and R2 is the calibration penalty. ACC lines are
    drawn last so they stay visible on top of the shades."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    region_handles = _draw_anomaly_regions(ax, out["lead"], out["acc_centered"], out["r2_anomaly"], regions, "Lead [h]")
    labels = ["ACC$^2$", "R$^2$", "ACC"]
    anchors = [(1.02, 1.0 - 0.32 * position) for position in range(len(regions))]
    legends = []
    for (region, handles), anchor in zip(region_handles.items(), anchors):
        legend = ax.legend(handles, labels, title=region, frameon=False, loc="upper left", bbox_to_anchor=anchor)
        legend.get_title().set_fontweight("bold")
        ax.add_artist(legend)
        legends.append(legend)
    fig.savefig(out_dir / "acc_r2_by_lead_regions.pdf", dpi=150, bbox_inches="tight", bbox_extra_artists=legends)
    plt.close(fig)


def _draw_anomaly_regions(ax, x, acc, r2, regions, xlabel):
    """ACC^2 (solid), R^2 (dashed) and ACC (dotted, darker shade) per region window
    of `acc`/`r2` (dims: x, window), the shaded gap = calibration penalty. ACC lines
    are drawn last so they stay visible on top of the shades. Returns the line
    handles per region label, in the order ACC^2, R^2, ACC."""
    region_handles = {}
    for window in regions:
        label, color, dark = REGION_STYLES[window]
        acc_window = acc.sel(window=window)
        r2_window = r2.sel(window=window)
        ax.fill_between(x, r2_window, acc_window**2, color=color, alpha=0.15)
        (line_acc2,) = ax.plot(x, acc_window**2, color=color, ls="-")
        (line_r2,) = ax.plot(x, r2_window, color=color, ls="--")
        region_handles[label] = [line_acc2, line_r2]
    for window in regions:
        label, color, dark = REGION_STYLES[window]
        (line_acc,) = ax.plot(x, acc.sel(window=window), color=dark, ls=":")
        region_handles[label].append(line_acc)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.set_ylim(bottom=0.0, top=1.0)
    return region_handles


# Score rows of the bar figure: (label, pooled variable, unit, value format).
BAR_SCORES = [
    ("RMSE", "rmse_model", "[K]", "%.2f"),
    ("Bias", "bias_model", "[K]", "%+.2f"),
    ("ACC", "acc", "", "%.2f"),
    ("R$^2$", "r2", "", "%.2f"),
]
# Keys into the tables' row groups (lowercase); the tick labels are capitalised.
BAR_MEMBERS = ("day", "night")
# Regions as a grey ramp in the bar figures (dark -> light in group_order); the region
# hues stay reserved for the ACC/R2 lead figure.
BAR_REGION_GREYS = {"Arid": "#404040", "Temperate": "#8c8c8c", "Tropical": "#c8c8c8"}


FVC_DECILES = range(1, 11)
# Total / day / night series of the decile curves: the model's neutral line plus one
# warm and one cool hue, so the two solar windows read as a pair against the total.
FVC_WINDOW_STYLES = {"total": ("All hours", MODEL_LINE), "day": ("Day", "#eda100"), "night": ("Night", "#2a78d6")}
# Landcover groups of the pixel scatter (masks.LC_GROUPS membership): three categorical
# hues that pass the all-pairs separation check, the sparse shrub/barren group in grey,
# everything else in a lighter grey. Marker shapes repeat the identity for print.
LC_STYLES = {
    "lc_forest": ("Forest", "#2a78d6", "o"),
    "lc_savanna": ("Savanna", "#eb6834", "s"),
    "lc_grass_crop": ("Grass/Crop", "#1baf7a", "^"),
    "lc_shrub_barren": ("Shrub/Barren", "#898781", "D"),
    "other": ("Other", "#c3c2b7", "x"),
}


def plot_fvc_decile_curves(pooled, days, out_dir):
    """Pooled scores along the FVC deciles: RMSE and the skill against climatology
    (rows) at the nowcast and medium-range day (columns), one line per solar window.
    Reads the fvc_d<k>(_day|_night) windows; deciles that were not verified leave gaps."""
    available = {str(window) for window in pooled["window"].values}
    rows = [
        ("RMSE [K]", pooled["rmse_model"], "%.2f"),
        ("Skill improvement (RMSE) vs climatology", pooled["rmse_ss"].sel(baseline="clim_smooth"), "%.2f"),
    ]
    fig, axes = plt.subplots(
        len(rows), len(days), figsize=(5.0 * len(days), 3.0 * len(rows)), sharex=True, sharey="row"
    )
    axes = np.atleast_2d(axes)
    for row_index, (ylabel, values, fmt) in enumerate(rows):
        for column, day in enumerate(days):
            ax = axes[row_index, column]
            for suffix, (label, color) in FVC_WINDOW_STYLES.items():
                series = []
                for decile in FVC_DECILES:
                    window = f"fvc_d{decile}" if suffix == "total" else f"fvc_d{decile}_{suffix}"
                    series.append(float(values.sel(window=window, day=day)) if window in available else np.nan)
                ax.plot(list(FVC_DECILES), series, color=color, lw=2, marker="o", ms=5, label=label)
            ax.axhline(0.0, color="#c3c2b7", lw=0.6)
            ax.set_xticks(list(FVC_DECILES))
            ax.grid(axis="y", color="#e1e0d9", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(f"Day {day}", fontsize=12)
            if row_index == len(rows) - 1:
                ax.set_xlabel("Annual-mean FVC class (1 = bare, 2-10 = vegetated bins)")
            if column == 0:
                ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fvc_decile_curves.pdf")
    plt.close(fig)


def _lc_group(lc_class):
    for group_name, class_ids in LC_GROUPS.items():
        if lc_class in class_ids:
            return group_name
    return "other"


# x axes of the pixel scatter: attribute key in cubes.pixel_attributes -> (axis label, file prefix).
SCATTER_AXES = {
    "fvc_mean": ("Annual-mean FVC", "fvc"),
    "lst_validity": ("Annual-mean LST observational validity [%]", "validity"),
}


def _scatter_bins(attributes, x_attribute, x_values):
    """Per-pixel bin id for the median line: the shared FVC classes on the FVC axis, ten
    equal-count bins of the finite values on any other axis (0 = unbinned)."""
    if x_attribute == "fvc_mean":
        return np.asarray(attributes["fvc_class"])
    finite = np.isfinite(x_values)
    edges = np.quantile(x_values[finite], np.linspace(0, 1, 11)[1:-1])
    bins = 1 + np.searchsorted(edges, x_values, side="right")
    bins[~finite] = 0
    return bins


def plot_fvc_pixel_scatter(
    pixel_daily, attributes, days, out_dir, variable="rmse_ss", baseline="clim_smooth", x_attribute="fvc_mean"
):
    """One point per pixel: score at the nowcast / medium-range day (columns) in the
    day / night windows (rows) against a per-pixel attribute (annual-mean FVC by default,
    or the LST observational validity), coloured by landcover group, with the per-bin
    median as a line: the shared FVC classes on the FVC axis, ten equal-count bins
    otherwise. `attributes` is cubes.pixel_attributes output. variable: rmse_ss (needs a
    baseline) or rmse_model."""
    xlabel, prefix = SCATTER_AXES[x_attribute]
    if variable == "rmse_ss":
        field = pixel_daily["rmse_ss"].sel(baseline=baseline)
        ylabel = f"Skill improvement (RMSE) vs {BASELINE_LABELS[baseline].lower()}"
        name = f"{prefix}_pixel_scatter_skill_{baseline}.pdf"
    else:
        field = pixel_daily[variable]
        ylabel = "RMSE [K]"
        name = f"{prefix}_pixel_scatter_{variable}.pdf"
    x_values = np.asarray(attributes[x_attribute], dtype=float)
    bins = _scatter_bins(attributes, x_attribute, x_values)
    groups = np.array([_lc_group(int(lc_class)) for lc_class in attributes["lc_class"]])
    windows = ["day", "night"]

    # One y range for the whole figure, from the central 99% of all panels: a handful of
    # pixels with runaway scores would otherwise flatten the cloud (seen: a night skill of -27).
    panels = {
        (window, day): np.asarray(field.sel(window=window, day=day).values, dtype=float)
        for window in windows
        for day in days
    }
    pooled_values = np.concatenate([values[np.isfinite(values)] for values in panels.values()])
    low, high = np.quantile(pooled_values, [0.005, 0.995])
    margin = 0.05 * (high - low)

    fig, axes = plt.subplots(
        len(windows), len(days), figsize=(5.0 * len(days), 3.4 * len(windows)), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    for row_index, window in enumerate(windows):
        for column, day in enumerate(days):
            ax = axes[row_index, column]
            values = panels[(window, day)]
            shown = np.isfinite(values) & np.isfinite(x_values)
            for group_name, (label, color, marker) in LC_STYLES.items():
                members = shown & (groups == group_name)
                if not members.any():
                    continue
                ax.scatter(
                    x_values[members],
                    values[members],
                    s=9,
                    color=color,
                    marker=marker,
                    alpha=0.55,
                    linewidths=0,
                    label=f"{label} ({int(members.sum())})",
                )
            decile_x, decile_y = [], []
            for bin_id in range(1, 11):
                members = shown & (bins == bin_id)
                if members.sum() < 5:
                    continue
                decile_x.append(np.median(x_values[members]))
                decile_y.append(np.median(values[members]))
            ax.plot(decile_x, decile_y, color=MODEL_LINE, lw=2, marker="o", ms=5, label="Class median")
            ax.axhline(0.0, color="#c3c2b7", lw=0.6)
            ax.set_ylim(low - margin, high + margin)
            ax.grid(axis="y", color="#e1e0d9", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(f"Day {day}", fontsize=12)
            if row_index == len(windows) - 1:
                ax.set_xlabel(xlabel)
            if column == 0:
                ax.set_ylabel(f"{ylabel}\n{window} window")
    axes[0, 0].legend(frameon=False, fontsize=8, markerscale=1.8)
    fig.tight_layout()
    fig.savefig(out_dir / name)
    plt.close(fig)


def plot_score_bars(pooled, days, groups, out_dir, group_order=("Arid", "Temperate", "Tropical")):
    """The score tables as bars: one row per score (RMSE, bias, ACC, R2). Rows whose
    variable is absent from `pooled` (verifs without the anomaly moments) are skipped."""
    rows = []
    for label, variable, unit, fmt in BAR_SCORES:
        if variable in pooled:
            rows.append((label, pooled[variable], unit, fmt))
    _plot_region_bars(rows, days, groups, out_dir / "score_bars.pdf", group_order)


def plot_skill_bars(pooled, days, groups, out_dir, group_order=("Arid", "Temperate", "Tropical")):
    """The skill table as bars: one row per baseline (RMSE-SS against it)."""
    rows = []
    for kind, label in BASELINE_LABELS.items():
        rows.append((label, pooled["rmse_ss"].sel(baseline=kind), "", "%.2f"))
    _plot_region_bars(
        rows, days, groups, out_dir / "skillscore_bars.pdf", group_order, shared_ylabel="Skill improvement"
    )


def _plot_region_bars(rows, days, groups, path, group_order, shared_ylabel=None):
    """One panel row per score, columns = nowcast day, medium-range day, their
    absolute difference; day/night groups on the x axis with one bar per region in
    grey shades. Bars take two thirds of their slot, so the regions of a group stay
    separated. `rows` are (label, values (day, window), unit, format); `groups` is
    tables._row_groups output (region label -> member windows), so bars and tables
    read the same cells. `shared_ylabel` adds an outer y label spanning all rows,
    with the row labels as the inner level."""
    nowcast, medium = days
    regions = [region for region in group_order if region in groups]
    if not regions or not rows:
        return
    available = {str(window) for window in rows[0][1]["window"].values}

    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 3.0 * len(rows)), sharex=True)
    axes = np.atleast_2d(axes)
    slot = 0.8 / len(regions)
    width = slot * 2 / 3
    positions = np.arange(len(BAR_MEMBERS))
    column_titles = [f"Day {nowcast}", f"Day {medium}", f"Day {medium} − Day {nowcast}"]
    # Value labels per panel and region, for the collision check after the layout.
    value_labels = {(row, column): [] for row in range(len(rows)) for column in range(3)}
    for row_index, (label, values, unit, fmt) in enumerate(rows):
        for region_index, region in enumerate(regions):
            offsets = positions + (region_index - (len(regions) - 1) / 2) * slot
            at_nowcast = _member_values(values, groups[region], nowcast, available)
            at_medium = _member_values(values, groups[region], medium, available)
            columns = [at_nowcast, at_medium, at_medium - at_nowcast]
            for column, column_values in enumerate(columns):
                ax = axes[row_index, column]
                bars = ax.bar(offsets, column_values, width, color=BAR_REGION_GREYS[region], label=region)
                labels = ax.bar_label(
                    bars,
                    fmt=fmt if column < 2 else "%+.2f",
                    fontsize=SUMMARY_VALUE_LABEL_SIZE,
                    color="#52514e",
                    padding=2,
                )
                value_labels[(row_index, column)].append(labels)
        for column in range(3):
            ax = axes[row_index, column]
            ax.axhline(0.0, color="grey", lw=0.6)
            ax.set_xticks(positions)
            ax.set_xticklabels([member.capitalize() for member in BAR_MEMBERS])
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(column_titles[column])
        axes[row_index, 0].set_ylabel(f"{label} {unit}".strip())
        _share_ylim(axes[row_index, :2])
        _share_ylim(axes[row_index, 2:])
    handles = [Patch(color=BAR_REGION_GREYS[region], label=region) for region in regions]
    # Above the panels: inside the first panel it collides with the bar labels.
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=len(regions), bbox_to_anchor=(0.5, 1.0))
    if shared_ylabel is not None:
        fig.supylabel(shared_ylabel)
    fig.tight_layout()
    for labels in value_labels.values():
        _stagger_colliding_labels(fig, labels, lift=1.1 * SUMMARY_VALUE_LABEL_SIZE)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _member_values(values, members, day, available):
    """`values` (day, window) at forecast `day` for the day/night member windows of
    one region group (BAR_MEMBERS order); NaN where the member was not verified."""
    picked = []
    for member in BAR_MEMBERS:
        window = members[member]
        if window not in available:
            picked.append(np.nan)
        else:
            picked.append(float(values.sel(window=window, day=day)))
    return np.array(picked)


def _share_ylim(axes):
    """One y range across the panels, with room for the value labels above the
    tallest bar and below the most negative one."""
    low = min(ax.get_ylim()[0] for ax in axes)
    high = max(ax.get_ylim()[1] for ax in axes)
    span = high - low
    for ax in axes:
        ax.set_ylim(low - 0.08 * span, high + 0.08 * span)


# Regime summary figure: region order of the legend and the bars.
SUMMARY_REGIONS = ("Arid", "Temperate", "Tropical")


def _region_windows(labels):
    """REGION_STYLES window names for the display labels, in that order."""
    by_label = {label: window for window, (label, color, dark) in REGION_STYLES.items()}
    return [by_label[label] for label in labels]


def _draw_regime_error(ax, x, rmse_model, rmse_clim, regions, xlabel):
    """Model RMSE (solid) and the climatology RMSE (dashed) per regime window in the
    region hue: the error growth reads against the ceiling each regime approaches."""
    for window in regions:
        label, color, dark = REGION_STYLES[window]
        ax.plot(x, rmse_model.sel(window=window), color=color, lw=SUMMARY_LINE_WIDTH)
        ax.plot(x, rmse_clim.sel(window=window), color=color, ls="--", lw=0.8 * SUMMARY_LINE_WIDTH)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE [K]")
    style_handles = [
        Line2D([], [], color=INK_SECONDARY, ls="-", label=MODEL_LABEL),
        Line2D([], [], color=INK_SECONDARY, ls="--", label=BASELINE_LABELS[SPATIAL_BASELINE]),
    ]
    ax.legend(handles=style_handles, frameon=False, loc="lower right")


def _draw_difference_bars(ax, values, groups, regions, days, ylabel, label_size):
    """Day `days[1]` minus day `days[0]` of `values` (day, window) per region and
    day/night member: one bar per region in the region hue, value labels on the bars.
    Same cells as the difference column of the bar figures. Returns the value
    labels per region (for _stagger_colliding_labels)."""
    first, second = days
    available = {str(window) for window in values["window"].values}
    # Slots fill 0.9 of the group unit (0.8 in the bar figures): the 12 pt value
    # labels are wider than a slot of the narrower panels otherwise.
    slot = 0.9 / len(regions)
    width = slot * SUMMARY_BAR_FILL
    positions = np.arange(len(BAR_MEMBERS))
    labels = []
    for region_index, region in enumerate(regions):
        offsets = positions + (region_index - (len(regions) - 1) / 2) * slot
        at_first = _member_values(values, groups[region], first, available)
        at_second = _member_values(values, groups[region], second, available)
        bars = ax.bar(offsets, at_second - at_first, width, color=REGION_COLORS[region], label=region)
        labels.append(ax.bar_label(bars, fmt="%+.2f", fontsize=label_size, color=INK_SECONDARY, padding=2))
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([member.capitalize() for member in BAR_MEMBERS])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(ylabel)
    _share_ylim([ax])
    return labels


def _stagger_colliding_labels(fig, labels, lift):
    """`labels`: per region, the bar value labels at each group position. Where the
    labels of neighbouring regions overlap horizontally (narrow panels), every
    second region's labels move `lift` points further from the bar, out of line.
    Run after the layout is final: the overlap depends on the axes width."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    n_regions = len(labels)
    for position in range(len(labels[0])):
        boxes = [labels[region][position].get_window_extent(renderer) for region in range(n_regions)]
        overlapping = any(boxes[i].x1 > boxes[i + 1].x0 for i in range(n_regions - 1))
        if not overlapping:
            continue
        for region in range(1, n_regions, 2):
            text = labels[region][position]
            dx, dy = text.xyann
            text.xyann = (dx, dy + np.sign(dy) * lift)


# Print settings shared by the summary figures: base text (ticks, in-panel text),
# axis labels + titles + legends + row letters, bar value labels, top-row lines; row
# gap in font-size units; figure size, top:bottom row ratio, bar width as a fraction
# of its slot.
SUMMARY_FONT_SIZE = 18
SUMMARY_AXIS_FONT_SIZE = 20
SUMMARY_VALUE_LABEL_SIZE = 12
SUMMARY_LINE_WIDTH = 2.2
SUMMARY_LEGEND_LINE_WIDTH = 4.0  # colour swatches of the baseline legend
SUMMARY_ROW_GAP = 3.0
SUMMARY_ROW_LETTERS = ("A.", "B.")
SUMMARY_FIGSIZE = (17, 10)
SUMMARY_HEIGHT_RATIOS = (2.0, 1.0)
SUMMARY_BAR_FILL = 0.55


def _summary_rc(font_size, axis_font_size):
    """rc overrides of the summary figures: base text, axis labels / titles /
    legends, and the line width (which also reaches the shared panels drawing with
    the rc default, without touching the other figures)."""
    rc = dict(PRINT_RC)
    rc["lines.linewidth"] = SUMMARY_LINE_WIDTH
    if font_size is not None:
        rc.update({"font.size": font_size, "xtick.labelsize": font_size, "ytick.labelsize": font_size})
    rc.update({"axes.labelsize": axis_font_size, "axes.titlesize": axis_font_size, "legend.fontsize": axis_font_size})
    return rc


def _row_letters(fig, first_axes, size):
    """Bold SUMMARY_ROW_LETTERS at the left edge of each row's first y label, just
    above the axes. Label extents are only known after a draw."""
    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()
    for letter, ax in zip(SUMMARY_ROW_LETTERS, first_axes):
        label_box = ax.yaxis.label.get_window_extent().transformed(to_figure)
        # A y label taller than a low panel sticks out above it: go above both.
        top = max(ax.get_position().y1, label_box.y1) + 0.005
        fig.text(label_box.x0, top, letter, fontsize=size, fontweight="bold", ha="left", va="bottom")


def plot_regime_summary(
    scores,
    pooled,
    groups,
    out_dir,
    resolution="hourly",
    days=(1, 10),
    font_size=SUMMARY_FONT_SIZE,
    axis_font_size=SUMMARY_AXIS_FONT_SIZE,
    value_label_size=SUMMARY_VALUE_LABEL_SIZE,
    figsize=SUMMARY_FIGSIZE,
):
    """Top row: RMSE growth per regime with the regime's climatology ceiling (left)
    and the per-regime anomaly scores ACC^2 / R^2 / ACC with the shaded calibration
    gap (right), both by lead. Bottom row: the day `days[1]` minus day `days[0]`
    change of RMSE, bias, ACC and R2 per regime and time of day -- the difference
    column of the bar figures. Regions keep their hues throughout, one legend,
    ordered as SUMMARY_REGIONS.

    `resolution` "hourly" takes the lead curves from `scores` (skill.nc), "daily"
    from `pooled` (skill_daily.nc); the bars always come from `pooled`, the cells
    the tables print. `font_size` is the base text size (ticks, in-panel text),
    `axis_font_size` the axis labels, legends and row letters, `value_label_size`
    the bar value labels; font_size None keeps the rc default for the base text."""
    regions = [label for label in SUMMARY_REGIONS if label in groups]
    windows = _region_windows(regions)
    if resolution == "hourly":
        x = scores["lead"]
        rmse_model = scores["rmse_model"]
        rmse_clim = scores["rmse_baseline"].sel(baseline=SPATIAL_BASELINE)
        acc = scores["acc_centered"]
        r2 = scores["r2_anomaly"]
        xlabel = "Lead [h]"
    else:
        x = pooled["day"]
        rmse_model = pooled["rmse_model"]
        rmse_clim = pooled["rmse_baseline"].sel(baseline=SPATIAL_BASELINE)
        acc = pooled["acc"]
        r2 = pooled["r2"]
        xlabel = "Lead time [day]"
    rows = [(label, variable, unit) for label, variable, unit, fmt in BAR_SCORES if variable in pooled]

    # The line width applies to the shared anomaly panel too (it draws with the rc
    # default), without touching the other figures that use it.
    with plt.rc_context(_summary_rc(font_size, axis_font_size)):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(2, len(rows), height_ratios=SUMMARY_HEIGHT_RATIOS)
        half = len(rows) // 2
        ax_error = fig.add_subplot(grid[0, :half])
        ax_anomaly = fig.add_subplot(grid[0, half:])
        _draw_regime_error(ax_error, x, rmse_model, rmse_clim, windows, xlabel)
        _draw_anomaly_regions(ax_anomaly, x, acc, r2, windows, xlabel)
        ax_anomaly.legend(handles=_anomaly_style_handles(), frameon=False, loc="lower left")
        bar_axes, bar_labels = [], []
        for column, (label, variable, unit) in enumerate(rows):
            ax = fig.add_subplot(grid[1, column])
            ylabel = f"Δ{label} {unit}".strip()
            labels = _draw_difference_bars(ax, pooled[variable], groups, regions, days, ylabel, value_label_size)
            bar_axes.append(ax)
            bar_labels.append(labels)
        region_handles = [Patch(color=REGION_COLORS[region], label=region) for region in regions]
        fig.legend(
            handles=region_handles, frameon=False, loc="lower center", ncol=len(regions), bbox_to_anchor=(0.5, 1.0)
        )
        # Tight between panels (pads in font-size units, well below the 1.08
        # default); the row gap alone stays wide, it also holds the second row letter.
        fig.tight_layout(pad=0.4, h_pad=SUMMARY_ROW_GAP, w_pad=0.6)
        for labels in bar_labels:
            _stagger_colliding_labels(fig, labels, lift=1.1 * value_label_size)
        _row_letters(fig, (ax_error, bar_axes[0]), axis_font_size)
        fig.savefig(out_dir / f"regime_summary_{resolution}.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_skill_summary(
    scores,
    pooled,
    bootstrap_hourly,
    bootstrap_daily,
    groups,
    out_dir,
    resolution="hourly",
    days=(1, 10),
    font_size=SUMMARY_FONT_SIZE,
    axis_font_size=SUMMARY_AXIS_FONT_SIZE,
    value_label_size=SUMMARY_VALUE_LABEL_SIZE,
    figsize=SUMMARY_FIGSIZE,
):
    """Top row: skill improvement against the three baselines by lead for the
    daytime peak (left) and the nighttime peak (right), each with its bootstrap CI
    and the pixel spread of the climatology skill (the skill panel of the lead
    figures). Bottom row: the day `days[1]` minus day `days[0]` change of the skill
    against each baseline per region and time of day, regions in their hues, one y
    range across the three panels; the grid is the regime summary's, the panels take
    its first three quarters and the region legend the fourth, beside them. Baseline
    legend above the figure.

    `resolution` "hourly" reads the curves from `scores` (skill.nc) and
    `bootstrap_hourly`, "daily" from `pooled` (skill_daily.nc) and `bootstrap_daily`;
    the bars always come from `pooled`. Text sizes as in plot_regime_summary."""
    regions = [label for label in SUMMARY_REGIONS if label in groups]
    if resolution == "hourly":
        x = scores["lead"]
        skill = scores["rmse_ss"]
        ci = bootstrap_hourly
        dispersion = scores["rmse_ss_dispersion"]
        xlabel = "Lead [h]"
    else:
        x = pooled["day"]
        skill = pooled["rmse_ss"]
        ci = bootstrap_daily
        dispersion = pooled["rmse_ss_dispersion"]
        xlabel = "Lead time [day]"

    with plt.rc_context(_summary_rc(font_size, axis_font_size)):
        fig = plt.figure(figsize=figsize)
        n_baselines = len(BASELINE_LABELS)
        grid = fig.add_gridspec(2, n_baselines + 1, height_ratios=SUMMARY_HEIGHT_RATIOS)
        half = (n_baselines + 1) // 2
        ax_day = fig.add_subplot(grid[0, :half])
        ax_night = fig.add_subplot(grid[0, half:])
        for ax, window, title in ((ax_day, "day", "Daytime peak"), (ax_night, "night", "Nighttime peak")):
            _draw_skill(ax, x, skill.sel(window=window), xlabel, ci.sel(window=window), dispersion.sel(window=window))
            ax.set_title(title)
        ax_night.get_legend().remove()  # the spread legend once is enough

        bar_axes, bar_labels = [], []
        for column, (kind, label) in enumerate(BASELINE_LABELS.items()):
            ax = fig.add_subplot(grid[1, column])
            ylabel = "ΔSkill\nimprovement" if column == 0 else ""
            skill_change = pooled["rmse_ss"].sel(baseline=kind)
            labels = _draw_difference_bars(ax, skill_change, groups, regions, days, ylabel, value_label_size)
            ax.set_title(label)
            bar_axes.append(ax)
            bar_labels.append(labels)
        _share_ylim(bar_axes)
        # The region legend takes the fourth cell of the bottom row, beside the bars:
        # an invisible axes keeps the cell in the layout.
        ax_legend = fig.add_subplot(grid[1, n_baselines])
        ax_legend.axis("off")
        region_handles = [Patch(color=REGION_COLORS[region], label=region) for region in regions]
        ax_legend.legend(handles=region_handles, frameon=False, loc="center left", borderaxespad=0.0)

        # Thick, long swatches: the muted hues need more colour to be recognised.
        baseline_handles = [
            Line2D([], [], color=color, lw=SUMMARY_LEGEND_LINE_WIDTH, label=BASELINE_LABELS[kind])
            for kind, color in COLORS.items()
        ]
        fig.legend(
            handles=baseline_handles,
            frameon=False,
            loc="lower center",
            ncol=n_baselines,
            bbox_to_anchor=(0.5, 1.0),
            handlelength=3.0,
        )
        fig.tight_layout(pad=0.4, h_pad=SUMMARY_ROW_GAP, w_pad=0.6)
        for labels in bar_labels:
            _stagger_colliding_labels(fig, labels, lift=1.1 * value_label_size)
        _row_letters(fig, (ax_day, bar_axes[0]), axis_font_size)
        fig.savefig(out_dir / f"skill_summary_{resolution}.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)


# Per-pixel bias planes: two forecast days overlaid as slate-blue clouds (the nowcast
# day darker and on top, the medium-range day lighter behind), the regime medians in
# the region hues with an arrow from the first day to the second. The unreachable
# wedge RMSE < |bias| is the light neutral grey of the censored forecast-limit dots.
CLOUD_COLORS = {"first": "#2f4b7c", "second": "#9db4d3"}
BIAS_LEGEND_MARKER_SIZE = 14  # the day markers in the legend, larger than the cloud dots
WEDGE_COLOR = "#bdbdbd"
INK_SECONDARY = "#52514e"


def _day_leads(day):
    """Lead hours pooled into forecast `day` (1-based): 24 per day."""
    return slice(24 * (day - 1) + 1, 24 * day)


def _day_positions(lead, day):
    """Positions along the `lead` coordinate (hours) of the leads pooled into forecast `day`."""
    leads = _day_leads(day)
    inside = (lead >= leads.start) & (lead <= leads.stop)
    return np.where(inside)[0]


def _regime_groups(kg_class):
    """Pixel masks per regime group (KG_GROUPS order); pixels outside all groups are
    left out of the figure."""
    groups = {}
    for name, ids in KG_GROUPS.items():
        groups[name] = np.isin(kg_class, ids)
    return groups


def plot_pixel_bias_planes(pixel_daily, counts, days, out_dir, n_min=10):
    """Per-pixel bias planes at the two forecast `days`, day and night windows.

    Panels A: (bias, RMSE) per pixel for the daytime and the nighttime peak, with the
    unreachable wedge RMSE < |bias| and iso-curves of the debiased RMSE
    sqrt(RMSE^2 - bias^2). Panel B: level = (bias_day + bias_night)/2 against
    amplitude = (bias_day - bias_night)/2, one point per pixel. Both days are drawn
    as clouds (first day dark, second light); the regime medians of each day are the
    coloured markers, joined by an arrow from the first day to the second.

    `pixel_daily` (day, window, pixel) carries bias_model/rmse_model and kg_class;
    `counts` is by_pixel n (window, lead, pixel). Pixels with fewer than `n_min`
    samples in either window at either day are dropped, and the losses per group are
    printed; pixels outside the three regime groups are not drawn."""
    first, second = days
    kg_class = pixel_daily["kg_class"].values
    groups = _regime_groups(kg_class)
    grouped = np.any(list(groups.values()), axis=0)

    keep = grouped.copy()
    values = {}
    for day in days:
        n_day = counts.sel(lead=_day_leads(day)).sum("lead")
        keep &= (n_day.sel(window="day").values >= n_min) & (n_day.sel(window="night").values >= n_min)
        for window in ("day", "night"):
            bias = pixel_daily["bias_model"].sel(day=day, window=window).values
            rmse = pixel_daily["rmse_model"].sel(day=day, window=window).values
            keep &= np.isfinite(bias) & np.isfinite(rmse)
            values[(day, window)] = (bias, rmse)
    for name, members in groups.items():
        lost = int((members & ~keep).sum())
        print(
            f"bias planes days {days}: {REGION_STYLES[name][0]} loses {lost} of {int(members.sum())} pixels (n >= {n_min})"
        )
    print(f"bias planes days {days}: {int((~grouped).sum())} pixels outside the regime groups are not drawn")

    fig, (ax_day, ax_night, ax_level) = plt.subplots(1, 3, figsize=(19, 6))

    # Panels A: shared limits over both windows and both days.
    all_bias = np.concatenate([values[key][0][keep] for key in values])
    all_rmse = np.concatenate([values[key][1][keep] for key in values])
    x_low, x_high = np.percentile(all_bias, [0.5, 99.5])
    x_pad = 0.1 * (x_high - x_low)
    plane_limits = (x_low - x_pad, x_high + x_pad, np.percentile(all_rmse, 99.5) * 1.1)
    for ax, window, title in ((ax_day, "day", "Daytime peak"), (ax_night, "night", "Nighttime peak")):
        _draw_bias_scaffold(ax, plane_limits)
        _draw_cloud(ax, values[(second, window)], keep, CLOUD_COLORS["second"])
        _draw_cloud(ax, values[(first, window)], keep, CLOUD_COLORS["first"])
        _draw_median_arrows(ax, values[(first, window)], values[(second, window)], keep, groups)
        ax.set_xlim(plane_limits[0], plane_limits[1])
        ax.set_ylim(0.0, plane_limits[2])
        ax.set_xlabel("Bias [K]")
        ax.set_ylabel("RMSE [K]")
        ax.set_title(title)

    # Panel B: level-amplitude plane, symmetric limits over both days.
    planes = {}
    for day in days:
        bias_day, _ = values[(day, "day")]
        bias_night, _ = values[(day, "night")]
        planes[day] = (0.5 * (bias_day + bias_night), 0.5 * (bias_day - bias_night))
    spread = np.concatenate([np.concatenate([planes[day][0][keep], planes[day][1][keep]]) for day in days])
    limit = float(np.max(np.abs(np.percentile(spread, [1, 99]))))
    ax_level.axhline(0.0, color=INK_SECONDARY, lw=0.6, zorder=1)
    ax_level.axvline(0.0, color=INK_SECONDARY, lw=0.6, zorder=1)
    _draw_cloud(ax_level, planes[second], keep, CLOUD_COLORS["second"])
    _draw_cloud(ax_level, planes[first], keep, CLOUD_COLORS["first"])
    _draw_median_arrows(ax_level, planes[first], planes[second], keep, groups)
    ax_level.set_xlim(-limit, limit)
    ax_level.set_ylim(-limit, limit)
    ax_level.set_xlabel("Level bias [K]")
    ax_level.set_ylabel("Amplitude bias [K]")

    handles = [
        Patch(color=REGION_STYLES[name][1], label=f"{REGION_STYLES[name][0]} (n={int((keep & members).sum())})")
        for name, members in groups.items()
    ]
    # The day markers mirror the medians: hollow with a dark edge for the first day,
    # filled light for the second.
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markersize=BIAS_LEGEND_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=CLOUD_COLORS["first"],
            markeredgewidth=1.5,
            label=f"Day {first}",
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markersize=BIAS_LEGEND_MARKER_SIZE,
            color=CLOUD_COLORS["second"],
            label=f"Day {second}",
        )
    )
    # Above the panels: inside or beside the level panel it squeezes the planes.
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()

    fig.savefig(out_dir / f"pixel_bias_plane_day{first:02d}_day{second:02d}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_bias_scaffold(ax, limits):
    """The unreachable wedge RMSE < |bias|, iso-curves of the debiased RMSE
    sqrt(bias^2 + s^2) labelled at the left edge, and a zero line."""
    x_low, x_high, _ = limits
    b = np.linspace(x_low, x_high, 400)
    ax.fill_between(b, 0.0, np.abs(b), color=WEDGE_COLOR, alpha=0.25, lw=0, zorder=0)
    for s in (1, 2, 3):
        ax.plot(b, np.sqrt(b**2 + s**2), ls="--", lw=1.0, color=WEDGE_COLOR, zorder=1)
        ax.text(
            x_low,
            np.sqrt(x_low**2 + s**2),
            f"{s} K",
            fontsize="small",
            style="italic",
            color=INK_SECONDARY,
            ha="left",
            va="bottom",
        )
    ax.axvline(0.0, color=INK_SECONDARY, lw=0.6, zorder=1)


def _draw_cloud(ax, xy, keep, color):
    x, y = xy
    ax.scatter(x[keep], y[keep], s=6, color=color, alpha=0.3, lw=0, zorder=2)


def _draw_median_arrows(ax, xy_first, xy_second, keep, groups):
    """Per regime: the component-wise median of the first day (hollow) and of the
    second day (filled), joined by an arrow in the region hue."""
    for name, members in groups.items():
        shown = keep & members
        color = REGION_STYLES[name][1]
        start = (np.median(xy_first[0][shown]), np.median(xy_first[1][shown]))
        end = (np.median(xy_second[0][shown]), np.median(xy_second[1][shown]))
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.6, "shrinkA": 4, "shrinkB": 4},
            zorder=5,
        )
        ax.scatter(*start, s=60, facecolor="white", edgecolor=color, linewidths=1.5, zorder=6)
        ax.scatter(*end, s=70, color=color, edgecolor="white", linewidths=1.2, zorder=6)
