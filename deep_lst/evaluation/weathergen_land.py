from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, LogNorm

from deep_lst.evaluation.metrics import compute_pooled_by_day, pool_metric, to_lead_hours
from deep_lst.evaluation.plots._design import CASE_COLORS, OBS_COLOR, RC
from deep_lst.evaluation.plots.error_growth import _day_bucket_masks, skill_day_night_plot
from deep_lst.evaluation.plots.scores import score_boxplots

WGL = "/Net/Groups/BGI/people/mwessel/deep_lst/assets/baselines/WGL/v0/weathergen_landv0.nc"
SEV = "/Net/Groups/BGI/tscratch/mwessel/wegen-stores/sampled_stores/seviri_sampled_eval_eval_mtg.zarr"
CLIM = "/Net/Groups/BGI/people/mwessel/deep_lst/assets/baselines/climatology_smoothed_hourly/eval_eval_mtg.nc"
PERS = "/Net/Groups/BGI/people/mwessel/deep_lst/assets/baselines/persistence/eval_eval_mtg.nc"
PLOTDIR = Path("/Net/Groups/BGI/people/mwessel/deep_lst/plots/wgl")
TABLEDIR = Path("/Net/Groups/BGI/people/mwessel/deep_lst/tables/wgl")

WGL_NAME = "WeatherGen Land"
CLIM_NAME = "Climatology (smooth)"
PERS_NAME = "Persistence"
WGL_COLOR = "#d62728"
CLIM_COLOR = CASE_COLORS["smoothed clim"]
PERS_COLOR = CASE_COLORS["persistence"]
CASE_COLORS[WGL_NAME] = WGL_COLOR  # so score_boxplots / skill plot colour the WeatherGen Land boxes
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


sev = xr.open_zarr(SEV)
obs_src = sev["LST"].rename(spatial_idx="pixel")  # dim 'pixel' indexed by global MajorTom index


def load_baseline(path):
    """Baseline netCDF (valid_time, pixel) -> 'pixel' dim indexed by global index (SEVIRI-store order)."""
    field = xr.open_dataset(path)["LST"]
    return field.assign_coords(pixel=("pixel", sev["spatial_idx"].values))


def align(src, time_dim, ref, pixel_ids):
    """Reindex a (pixel, <time_dim>) source onto ref's (init_time, lead_time, pixel) grid."""
    valid = ref.init_time + ref.lead_time  # (init_time, lead_time) valid times
    sub = src.sel(pixel=pixel_ids).sel({time_dim: valid}).transpose("init_time", "lead_time", ...)
    return xr.DataArray(sub.values, dims=ref.dims, coords=ref.coords)


# WeatherGenerator forecast + obs and baselines on its (init_time, lead_time, pixel) grid, full test year
wgl = xr.open_dataset(WGL)
gidx = wgl["global_idx"].values
pred_w = wgl["lst"].load()
obs_w = align(obs_src, "time_idx", pred_w, gidx)
clim_w = align(load_baseline(CLIM), "valid_time", pred_w, gidx)
pers_w = align(load_baseline(PERS), "valid_time", pred_w, gidx)

# per-pixel LST annual amplitude (Tmax - Tmin over all 2024 SEVIRI obs), in gidx pixel order
_obs_2024 = obs_src.sel(pixel=gidx).values[:, pd.DatetimeIndex(sev["time_idx"].values).year == 2024]
pixel_amplitude = np.nanmax(_obs_2024, axis=1) - np.nanmin(_obs_2024, axis=1)  # (n_pixel,)

MODELS = {WGL_NAME: (pred_w, WGL_COLOR), CLIM_NAME: (clim_w, CLIM_COLOR), PERS_NAME: (pers_w, PERS_COLOR)}
METRICS = [("RMSE", "rmse"), ("MAE", "mae"), ("R2", "nse")]  # ACC shown separately (WeatherGen Land only)
SPACE = [d for d in ("lat", "lon", "pixel") if d in pred_w.dims]
REDUCE = ["init_time", "lead_time", *SPACE]

valid_hour = (pred_w.init_time.dt.hour + (pred_w.lead_time / np.timedelta64(1, "h")).astype(int)) % 24
WINDOWS = {
    "total": None,
    "daytime": (valid_hour >= 10) & (valid_hour <= 15),
    "nighttime": (valid_hour >= 22) | (valid_hour <= 3),
}


def metric_table(tod):
    """Per-forecast-day RMSE/MAE/R2 for each model, pooled within the time-of-day window `tod`."""
    day_labels, lead_masks = _day_bucket_masks(pred_w)
    bucket_masks = lead_masks + [pred_w.lead_time > np.timedelta64(0, "ns")]  # + Total (all leads >= 1 h)
    row_names = day_labels + ["Total"]
    columns = {}
    for model, (pred, _) in MODELS.items():
        for metric_label, metric in METRICS:
            values = []
            for bucket in bucket_masks:
                mask = bucket if tod is None else (tod & bucket)
                values.append(float(pool_metric(pred.where(mask), obs_w.where(mask), metric, REDUCE)))
            columns[(model, metric_label)] = values
    table = pd.DataFrame(columns, index=row_names)
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


def error_growth_daily(save_path):
    """RMSE per forecast-day bucket (24 lead-h each), one line per model."""
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (pred, color) in MODELS.items():
        by_day = compute_pooled_by_day(pred, obs_w, ["rmse"])
        x = np.arange(by_day.sizes["day"])
        ax.plot(x, by_day["rmse"].values, color=color, linewidth=2, marker="o", markersize=6, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(by_day["day_label"].values, rotation=45)
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("RMSE (K)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def diurnal(field):
    """Mean value per valid hour-of-day, pooled over pixels, inits and leads."""
    panel = field.mean("pixel", skipna=True).values
    hour = valid_hour.values
    return np.array([np.nanmean(panel[hour == h]) for h in range(24)])


def diurnal_plot(save_path):
    """Diurnal cycle of truth, WeatherGenerator and climatology."""
    plt.rcParams.update(RC)
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hours, diurnal(obs_w), color=OBS_COLOR, linewidth=2, marker="o", markersize=4, label="SEVIRI obs")
    ax.plot(hours, diurnal(pred_w), color=WGL_COLOR, linewidth=2, marker="o", markersize=4, label="WeatherGen Land")
    ax.plot(
        hours, diurnal(clim_w), color=CLIM_COLOR, linewidth=2, marker="o", markersize=4, label="Climatology (smooth)"
    )
    ax.plot(hours, diurnal(pers_w), color=PERS_COLOR, linewidth=2, marker="o", markersize=4, label="Persistence")
    ax.set_xlabel("Valid hour of day (UTC)")
    ax.set_ylabel("LST (K)")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def diurnal_bias_plot(save_path):
    """Diurnal cycle of the forecast bias (model - obs) per valid hour-of-day."""
    plt.rcParams.update(RC)
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, field, color in [
        ("WeatherGen Land", pred_w, WGL_COLOR),
        ("Climatology (smooth)", clim_w, CLIM_COLOR),
        ("Persistence", pers_w, PERS_COLOR),
    ]:
        ax.plot(hours, diurnal(field - obs_w), color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax.axhline(0, color=OBS_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Valid hour of day (UTC)")
    ax.set_ylabel("LST bias (K)")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def mean_lst_plot(save_path):
    """Raw mean LST vs lead hour (pooled over inits and pixels) for obs and every model."""
    plt.rcParams.update(RC)
    series = [
        ("SEVIRI obs", obs_w, OBS_COLOR),
        ("WeatherGen Land", pred_w, WGL_COLOR),
        ("Climatology (smooth)", clim_w, CLIM_COLOR),
        ("Persistence", pers_w, PERS_COLOR),
    ]
    lead = to_lead_hours(pred_w)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, field, color in series:
        ax.plot(lead, field.mean(["init_time", "pixel"], skipna=True).values, color=color, linewidth=1.5, label=label)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Mean LST (K)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def hourly_rmse_plot(save_path):
    """Per-pixel RMSE (panel a) and debiased RMSE (panel b) by valid hour of day, boxed per model.

    Each box is the distribution across the eval pixels of that pixel's pooled (over init + lead)
    metric at that valid hour, so it shows the spatial spread of error, not just the pooled mean.
    Debiased RMSE (centred RMSE) = sqrt(MSE - bias^2) removes the systematic offset.
    """
    plt.rcParams.update(RC)
    models = list(MODELS.items())
    offsets = np.linspace(-0.28, 0.28, len(models))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for i, (model, (pred, color)) in enumerate(models):
        err = pred - obs_w
        rmse_boxes, debiased_boxes = [], []
        for hour in range(24):
            e = err.where(valid_hour == hour)
            mse = (e**2).mean(["init_time", "lead_time"], skipna=True).values  # (pixel,)
            bias = e.mean(["init_time", "lead_time"], skipna=True).values  # (pixel,)
            rmse_px = np.sqrt(mse)
            debiased_px = np.sqrt(np.clip(mse - bias**2, 0, None))
            rmse_boxes.append(rmse_px[np.isfinite(rmse_px)])
            debiased_boxes.append(debiased_px[np.isfinite(debiased_px)])
        positions = np.arange(24) + offsets[i]
        for ax, boxes in ((axes[0], rmse_boxes), (axes[1], debiased_boxes)):
            ax.boxplot(
                boxes,
                positions=positions,
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.6, linewidth=0.6),
                medianprops=dict(color="black", linewidth=1),
                whiskerprops=dict(color=color, linewidth=0.8),
                capprops=dict(color=color, linewidth=0.8),
                manage_ticks=False,
            )
    handles = [mpatches.Patch(facecolor=color, alpha=0.6, label=model) for model, (_, color) in MODELS.items()]
    for ax, title in zip(axes, ("RMSE", "Debiased RMSE")):
        ax.set_xlabel("Valid hour of day (UTC)")
        ax.set_title(title)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels(range(0, 24, 3))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("K")
    axes[0].legend(handles=handles, loc="upper left")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _month_hour_suffstats(pred, obs):
    """Per-(cell, pixel) sufficient stats over the common valid mask; cell = (month-1)*24 + hour.

    Mirrors table_runner._month_hour_suffstats but also accumulates the obs on the SAME (pred & obs
    finite) mask, so sigma_obs' is computed over exactly the samples that enter the error stats.
    Returns S1=Sigma e, S2=Sigma e^2, n=count (error e = pred - obs) and O1=Sigma obs, O2=Sigma obs^2,
    each (288, n_pixel).
    """
    error = pred - obs
    space = [d for d in error.dims if d not in ("init_time", "lead_time")]
    e = error.transpose("init_time", "lead_time", *space).values
    o = obs.transpose("init_time", "lead_time", *space).values
    n_pixel = int(np.prod(e.shape[2:]))
    e = e.reshape(-1, n_pixel)
    o = o.reshape(-1, n_pixel)

    init = pd.DatetimeIndex(pred["init_time"].values).values
    lead = pd.TimedeltaIndex(pred["lead_time"].values).values
    valid = pd.DatetimeIndex((init[:, None] + lead[None, :]).ravel())
    bucket = (valid.month.values - 1) * 24 + valid.hour.values  # 0..287 per (init, lead) cell

    finite = np.isfinite(e) & np.isfinite(o)
    e_filled = np.where(finite, e, 0.0)
    o_filled = np.where(finite, o, 0.0)
    S1 = np.zeros((288, n_pixel))
    S2 = np.zeros((288, n_pixel))
    n = np.zeros((288, n_pixel))
    O1 = np.zeros((288, n_pixel))
    O2 = np.zeros((288, n_pixel))
    for cell in range(288):
        rows = bucket == cell
        if not rows.any():
            continue
        S1[cell] = e_filled[rows].sum(axis=0)
        S2[cell] = (e_filled[rows] ** 2).sum(axis=0)
        n[cell] = finite[rows].sum(axis=0)
        O1[cell] = o_filled[rows].sum(axis=0)
        O2[cell] = (o_filled[rows] ** 2).sum(axis=0)
    return S1, S2, n, O1, O2


def _month_hour_acc(pred, obs, clim):
    """Forecast ACC (uncentered anomaly correlation vs climatology) on the month x hour grid (288,).

    Anomalies are pred - clim and obs - clim; per (month, hour) cell the uncentered ACC is pooled over
    all (pixel, init, lead) samples: r = Sum ap*ao / sqrt(Sum ap^2 * Sum ao^2). This is the STRICT
    forecast ACC -- skill of the predicted anomaly (departure from climatology) against the observed
    anomaly, so a forecast that just reverts to climatology scores ACC -> 0.
    """
    ap = (pred - clim).transpose("init_time", "lead_time", ...).values
    ao = (obs - clim).transpose("init_time", "lead_time", ...).values
    n_pixel = int(np.prod(ap.shape[2:]))
    ap = ap.reshape(-1, n_pixel)
    ao = ao.reshape(-1, n_pixel)

    init = pd.DatetimeIndex(pred["init_time"].values).values
    lead = pd.TimedeltaIndex(pred["lead_time"].values).values
    valid = pd.DatetimeIndex((init[:, None] + lead[None, :]).ravel())
    bucket = (valid.month.values - 1) * 24 + valid.hour.values

    finite = np.isfinite(ap) & np.isfinite(ao)
    apf = np.where(finite, ap, 0.0)
    aof = np.where(finite, ao, 0.0)
    acc = np.full(288, np.nan)
    for cell in range(288):
        rows = bucket == cell
        if not rows.any():
            continue
        num = float((apf[rows] * aof[rows]).sum())
        denom = np.sqrt(float((apf[rows] ** 2).sum()) * float((aof[rows] ** 2).sum()))
        if denom > 0:
            acc[cell] = num / denom
    return acc


def month_hour_heatmaps(save_path, sigma_floor=0.5):
    """RMSE (a), per-pixel-debiased RMSE (b), debiased sRMSE (c): month (y) x valid hour-of-day (x).

    Each cell pools every (init, lead, pixel) sample whose valid time falls in that (month, hour).
    Debiasing is PER PIXEL: each pixel's own temporal mean error within the cell is removed, so
    spatially-varying biases do not cancel (a single global offset would):
        raw_MSE      = Sigma_p S2_p / Sigma_p n_p
        debiased_MSE = Sigma_p (S2_p - S1_p^2/n_p) / Sigma_p n_p
        bias         = sqrt(raw_MSE - debiased_MSE)     # quadrature: raw^2 = debiased^2 + bias^2
    Panel c normalises the debiased RMSE by the per-pixel-centered obs std sigma_obs' over the same
    samples -> dimensionless (~ sqrt(1-NSE) in anomaly space), comparable across latitude/season.
    Cells with sigma_obs' < `sigma_floor` K are masked (the ratio is unstable there).
    """
    plt.rcParams.update(RC)
    S1, S2, n, O1, O2 = _month_hour_suffstats(pred_w, obs_w)
    total = n.sum(axis=1)  # (288,) Sigma_p n_p
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_mse = np.divide(S2.sum(axis=1), total, out=np.full(288, np.nan), where=total > 0)
        per_pixel_ss = np.maximum(S2 - np.divide(S1**2, n, out=np.zeros_like(S1), where=n > 0), 0.0)
        debiased_mse = np.divide(per_pixel_ss.sum(axis=1), total, out=np.full(288, np.nan), where=total > 0)
    raw = np.sqrt(np.clip(raw_mse, 0, None))
    debiased = np.sqrt(np.clip(debiased_mse, 0, None))
    bias = np.sqrt(np.clip(raw_mse - debiased_mse, 0, None))
    # nRMSE: divide each pixel's squared error by that pixel's OWN annual amplitude^2 BEFORE pooling,
    # so pixels across the wide latitudinal range are comparable (error as a fraction of the local range).
    amp2_safe = np.where(np.isfinite(pixel_amplitude) & (pixel_amplitude > 0), pixel_amplitude, np.nan) ** 2
    valid_amp = np.isfinite(amp2_safe)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_cnt = np.where(valid_amp[None, :], n, 0.0).sum(axis=1)
        srmse = np.sqrt(
            np.divide(
                np.where(valid_amp[None, :], S2 / amp2_safe[None, :], 0.0).sum(axis=1),
                norm_cnt,
                out=np.full(288, np.nan),
                where=norm_cnt > 0,
            )
        )

    # --- verification ---
    cells = np.isfinite(raw) & np.isfinite(debiased)
    max_violation = float(np.nanmax((debiased - raw)[cells])) if cells.any() else 0.0
    recon = float(np.nanmax(np.abs(raw**2 - (debiased**2 + bias**2))[cells])) if cells.any() else 0.0
    print(f"  check: max(debiased-raw)={max_violation:.2e} (should be <=~0); max|raw^2-(deb^2+bias^2)|={recon:.2e}")
    print(
        f"  pixel LST annual amplitude {np.nanmin(pixel_amplitude):.1f}..{np.nanmax(pixel_amplitude):.1f} K "
        f"(nRMSE = RMSE / per-pixel amplitude)"
    )
    if max_violation > 1e-6:
        print(f"  WARNING: {int((debiased - raw > 1e-6).sum())} cells with debiased > raw")

    panels = [
        # ("RMSE (K)", raw.reshape(12, 24), float(np.nanmax(raw)), "K"),
        # ("Debiased RMSE (K)", debiased.reshape(12, 24), float(np.nanmax(raw)), "K"),
        ("nRMSE (–)", srmse.reshape(12, 24), None, "–"),
        # ("Debiased nRMSE (–)", srmse_debiased.reshape(12, 24), None, "–"),
        ("ACC (–)", _month_hour_acc(pred_w, obs_w, clim_w).reshape(12, 24), None, "–"),
    ]
    # first panel: full day-of-year x hour fingerprint of mean observed LST over all pixels.
    # Built from ALL 2024 SEVIRI obs (independent of forecast sampling, so no forecast-coverage gaps);
    # nan-agnostic mean = sum of clear-sky obs / count of clear-sky obs per (doy, hour).
    obs_times = pd.DatetimeIndex(sev["time_idx"].values)
    in_2024 = obs_times.year == 2024
    obs_year = obs_src.sel(pixel=gidx).values[:, in_2024]  # (n_pixel, n_time_2024)
    finite_o = np.isfinite(obs_year)
    bucket = (obs_times[in_2024].dayofyear.values - 1) * 24 + obs_times[in_2024].hour.values
    grid_sum = np.zeros(366 * 24)
    grid_cnt = np.zeros(366 * 24)
    np.add.at(grid_sum, bucket, np.where(finite_o, obs_year, 0.0).sum(axis=0))
    np.add.at(grid_cnt, bucket, finite_o.sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        fingerprint = np.divide(grid_sum, grid_cnt, out=np.full(366 * 24, np.nan), where=grid_cnt > 0).reshape(366, 24)
    month_start = [pd.Timestamp(2024, m, 1).dayofyear for m in range(1, 13)]

    # validity: months whose forecast covers < 75% of their calendar days (scores from fewer days)
    fc_dates = pd.DatetimeIndex((pred_w.init_time + pred_w.lead_time).values.ravel()).normalize().unique()
    days_present = pd.Series(1, index=fc_dates).groupby(fc_dates.month).count().reindex(range(1, 13)).fillna(0)
    days_in_month = pd.Series({m: pd.Timestamp(2024, m, 1).days_in_month for m in range(1, 13)})
    coverage = days_present / days_in_month
    undercovered = [m for m in range(1, 13) if coverage[m] < 0.75]
    print(
        f"  under-covered months (<75% of days): "
        f"{[(MONTH_LABELS[m - 1], round(float(coverage[m]), 2)) for m in undercovered]}"
    )

    axes = np.atleast_1d(plt.subplots(1, 1 + len(panels), figsize=(5 * (1 + len(panels)), 5))[1])
    fig = axes[0].figure
    # fingerprint: day-of-year y-axis (month labels), Jan at bottom -> Dec at top
    im0 = axes[0].imshow(fingerprint, aspect="auto", cmap="inferno", origin="lower", extent=[-0.5, 23.5, 0.5, 366.5])
    axes[0].set_xlabel("Valid hour of day (UTC)")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].set_yticks(month_start)
    axes[0].set_yticklabels(MONTH_LABELS)
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="mean LST (K)")
    # metric panels: month y-axis, Jan at bottom -> Dec at top
    for ax, (title, grid, vmax, unit) in zip(axes[1:], panels):
        if title.startswith("ACC"):
            cmap = plt.get_cmap("winter", 10)  # 10 discrete classes 0-0.1 .. 0.9-1.0
            boundaries = np.linspace(0, 1, 11)
            im = ax.imshow(
                grid,
                aspect="auto",
                cmap=cmap,
                origin="lower",
                extent=[-0.5, 23.5, 0.5, 12.5],
                norm=BoundaryNorm(boundaries, cmap.N),
            )
            fig.colorbar(im, ax=ax, shrink=0.85, label=title, ticks=boundaries)
        elif "nRMSE" in title:
            # nRMSE = RMSE / per-pixel annual amplitude -> sequential (low = small error vs local LST range)
            im = ax.imshow(
                grid,
                aspect="auto",
                cmap="viridis",
                origin="lower",
                extent=[-0.5, 23.5, 0.5, 12.5],
                vmin=0,
                vmax=float(np.nanmax(grid)),
            )
            fig.colorbar(im, ax=ax, shrink=0.85, label=title)
        else:
            im = ax.imshow(
                grid, aspect="auto", cmap="winter", origin="lower", extent=[-0.5, 23.5, 0.5, 12.5], vmin=0, vmax=vmax
            )
            fig.colorbar(im, ax=ax, shrink=0.85, label=title)
        for m in undercovered:  # hatch under-sampled month rows
            ax.axhspan(m - 0.5, m + 0.5, facecolor="none", edgecolor="0.25", hatch="//", linewidth=0.0)
        ax.set_xlabel("Valid hour of day (UTC)")
        ax.set_xticks(range(0, 24, 3))
        ax.set_yticks(range(1, 13))
        ax.set_yticklabels(MONTH_LABELS)
    if undercovered:
        hatch_proxy = mpatches.Patch(facecolor="none", edgecolor="0.25", hatch="//", label="< 75% coverage")
        fig.legend(handles=[hatch_proxy], loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def per_pixel_metric(pred, metric):
    """Pooled metric per pixel (reduce over init + lead only)."""
    return pool_metric(pred, obs_w, metric, ["init_time", "lead_time"]).values


def class_boxplots(metric_label, metric, classes, names, xlabel, save_path, rotate, min_count=1):
    """Reuse score_boxplots (1:3 aspect, thin boxes): per-pixel metric boxed per class, one box per model.

    n above each group is the number of eval pixels in that class (the points making up each box).
    Classes with fewer than `min_count` pixels are dropped.
    """
    per_model = {model: per_pixel_metric(pred, metric) for model, (pred, _) in MODELS.items()}
    scores_by_group = {}
    group_counts = {}
    for value in sorted(np.unique(classes)):
        members = classes == value
        if members.sum() < min_count:
            continue
        label = names.get(int(value), str(int(value)))
        group_counts[label] = int(members.sum())
        scores_by_group[label] = {
            model: xr.Dataset({metric: ("pixel", vals[members])}) for model, vals in per_model.items()
        }
    score_boxplots(
        scores_by_group,
        metric=metric,
        xlabel=xlabel,
        ylabel=metric_label,
        save_path=save_path,
        figsize=(12, 4),
        box_width=0.4,
        rotate_xticks=rotate,
        group_counts=group_counts,
        legend_loc="upper left",
        legend_bbox=(1.0, 1.0),
    )


def pred_obs_hexbin_by_kg(save_path):
    """WeatherGen Land predicted-vs-observed density scatter, one hexbin panel per Köppen-Geiger class.

    Each panel pools every (init_time, lead_time, pixel) sample of the pixels in that class. A hexagon
    is shaded by the log number of samples whose (observed, predicted) LST falls in that bin (shared
    colour scale across panels), so it is a 2-D density of the forecast against the truth. The dashed
    line is 1:1 (perfect forecast): mass above it = over-prediction (warm bias), below = under-prediction,
    and the vertical spread at a given observed LST is the conditional error. A tilt flatter than 1:1
    (over-predicting cold scenes, under-predicting hot scenes) would indicate regression-to-the-mean.
    """
    plt.rcParams.update(RC)
    pred = pred_w.values
    obs = obs_w.values

    classes = sorted(np.unique(kg_class))
    ncol = 4
    nrow = int(np.ceil(len(classes) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.6 * ncol, 3.4 * nrow),  # independent per-panel axes
        gridspec_kw={"wspace": 0.45, "hspace": 0.3},
    )
    axes = axes.ravel()
    hexbins = []
    for ax, value in zip(axes, classes):
        members = kg_class == value
        p = pred[:, :, members].ravel()
        o = obs[:, :, members].ravel()
        valid = np.isfinite(p) & np.isfinite(o)
        lo = float(np.floor(min(np.percentile(o[valid], 0.5), np.percentile(p[valid], 0.5))))
        hi = float(np.ceil(max(np.percentile(o[valid], 99.5), np.percentile(p[valid], 99.5))))
        hb = ax.hexbin(o[valid], p[valid], gridsize=30, cmap="viridis", extent=(lo, hi, lo, hi), mincnt=1)
        hexbins.append(hb)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle="--")
        ax.set_title(KG_NAMES.get(int(value), value), fontsize=14)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    norm = LogNorm(vmin=1, vmax=max(hb.get_array().max() for hb in hexbins))
    for hb in hexbins:
        hb.set_norm(norm)
    for ax in axes[len(classes) :]:
        ax.set_visible(False)
    fig.supxlabel("Observed LST (K)")
    fig.supylabel("Predicted LST (K)")
    fig.colorbar(hexbins[0], ax=list(axes), shrink=0.6, label="samples per bin (log)")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def climatology_two_weeks(save_path, start_doy=183, n_days=14, window_days=7):
    """Two weeks of the smoothed hourly climatology at the eval pixels: smoothed MEAN vs smoothed MEDIAN.

    Both are built in-script from the SEVIRI history (years < 2024): for each (day-of-year, hour) the
    statistic is taken over every obs within a +/- window_days day-of-year window at that hour, per
    pixel, then averaged over pixels. The mean reproduces the precomputed climatology baseline; the
    median is the robust alternative (less pulled by cloud-contaminated / skewed LST samples).
    """
    plt.rcParams.update(RC)
    history = obs_src.sel(pixel=gidx)
    times = pd.DatetimeIndex(history["time_idx"].values)
    lst = history.values  # (n_pixel, n_time)
    train = times.year.values < 2024
    doy = times.dayofyear.values
    hour = times.hour.values

    mean_series, median_series, day_ticks = [], [], []
    for i in range(n_days):
        d = ((start_doy - 1 + i) % 366) + 1
        day_ticks.append((i * 24, d))
        window = ((d - 1 + np.arange(-window_days, window_days + 1)) % 366) + 1
        in_window = train & np.isin(doy, window)
        for h in range(24):
            block = lst[:, in_window & (hour == h)]  # (n_pixel, n_samples)
            mean_series.append(np.nanmean(np.nanmean(block, axis=1)))
            median_series.append(np.nanmean(np.nanmedian(block, axis=1)))

    x = np.arange(len(mean_series))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(x, mean_series, color=CLIM_COLOR, linewidth=2, label="Smoothed mean")
    ax.plot(x, median_series, color="#2ca02c", linewidth=2, label="Smoothed median")
    ax.set_xlabel("Day of year (ticks) / hour of day")
    ax.set_ylabel("Climatological LST (K)")
    ax.set_xticks([t for t, _ in day_ticks])
    ax.set_xticklabels([str(d) for _, d in day_ticks])
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


skill_day_night_plot(
    pred_w,
    obs_w,
    {CLIM_NAME: clim_w, PERS_NAME: pers_w},
    var="LST",
    save_path=PLOTDIR / "skill_day_night.png",
    eval_region="WGL eval",
    clim=clim_w,
)
climatology_two_weeks(PLOTDIR / "climatology_two_weeks.png")
diurnal_plot(PLOTDIR / "diurnal.png")
diurnal_bias_plot(PLOTDIR / "diurnal_bias.png")
mean_lst_plot(PLOTDIR / "mean_lst.png")
error_growth_daily(PLOTDIR / "error_growth_daily.png")
hourly_rmse_plot(PLOTDIR / "hourly_rmse.png")
month_hour_heatmaps(PLOTDIR / "month_hour_rmse.png")


def amplitude_map(save_path):
    """Scatter the eval pixels on a map of Africa, coloured by their 2024 LST annual amplitude (Tmax - Tmin)."""
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-20, 52, -36, 40], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    sc = ax.scatter(
        pred_w["lon"].values,
        pred_w["lat"].values,
        c=pixel_amplitude,
        s=90,
        cmap="magma",
        edgecolor="black",
        linewidth=0.3,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("LST annual amplitude  Tmax − Tmin (K)")
    ax.set_title(f"Eval pixels (n={len(pixel_amplitude)}) — LST annual amplitude")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


amplitude_map(PLOTDIR / "amplitude_map.png")

TABLEDIR.mkdir(parents=True, exist_ok=True)
for window, tod in WINDOWS.items():
    table = metric_table(tod).round(3)
    print(f"\n=== {window} (full test year 2024) ===")
    print(table.to_string())
    table.to_csv(TABLEDIR / f"metrics_{window}.csv")
    table.to_latex(
        TABLEDIR / f"metrics_{window}.tex",
        float_format="%.3f",
        multicolumn=True,
        multicolumn_format="c",
        caption=f"WeatherGen Land vs baselines, {window} valid hours, full test year 2024.",
    )
    print(f"  Saved: {TABLEDIR / f'metrics_{window}.csv'} (+ .tex)")


# --- pooled-metric boxplots across Koeppen-Geiger and land-cover classes ---
KG_NAMES = {
    1: "Af",
    2: "Am",
    3: "Aw",
    4: "BWh",
    5: "BWk",
    6: "BSh",
    7: "BSk",
    8: "Csa",
    9: "Csb",
    10: "Csc",
    11: "Cwa",
    12: "Cwb",
    13: "Cwc",
    14: "Cfa",
    15: "Cfb",
    16: "Cfc",
    17: "Dsa",
    18: "Dsb",
    19: "Dsc",
    20: "Dsd",
    21: "Dwa",
    22: "Dwb",
    23: "Dwc",
    24: "Dwd",
    25: "Dfa",
    26: "Dfb",
    27: "Dfc",
    28: "Dfd",
    29: "ET",
    30: "EF",
}
LC_NAMES = {
    1: "Evergreen Needleleaf",
    2: "Evergreen Broadleaf",
    3: "Decid. Needleleaf",
    4: "Decid. Broadleaf",
    5: "Mixed Forest",
    6: "Closed Shrubland",
    7: "Open Shrubland",
    8: "Woody Savanna",
    9: "Savanna",
    10: "Grassland",
    11: "Wetland",
    12: "Cropland",
    13: "Urban",
    14: "Cropland/Natural",
    15: "Snow/Ice",
    16: "Barren",
    17: "Water",
}

INDICES = "/Net/Groups/BGI/people/mwessel/deep_lst/assets/seviri/spatial/indices_eval_eval_mtg.npz"
classes_npz = np.load(INDICES)
global_to_kg = dict(zip(classes_npz["mapping"].tolist(), classes_npz["kg_classes"].tolist()))
global_to_lc = dict(zip(classes_npz["mapping"].tolist(), classes_npz["modis_lc_classes"].tolist()))
kg_class = np.array([global_to_kg[int(g)] for g in gidx])
lc_class = np.array([global_to_lc[int(g)] for g in gidx])


BOXDIR = PLOTDIR / "boxplots"
for metric_label, metric in METRICS:
    class_boxplots(metric_label, metric, kg_class, KG_NAMES, "Köppen-Geiger class", BOXDIR / f"{metric}_kg.png", 0)
    class_boxplots(
        metric_label,
        metric,
        lc_class,
        LC_NAMES,
        "Land-cover class",
        BOXDIR / f"{metric}_landcover.png",
        45,
        min_count=3,
    )


pred_obs_hexbin_by_kg(PLOTDIR / "pred_obs_hexbin_kg.png")
