"""Error-growth and persistence-skill plots for multi-lead-time forecasts.

All three functions consume a pre-computed metrics_by_lead_time dataset
(xr.Dataset with a 'lead_time' dim and metric variables such as 'rmse').

error_growth_plot              RMSE vs lead time — hourly resolution line plot
error_growth_daily_plot        Mean RMSE bucketed into forecast days (day 1/2/3 …)
skill_plot                     Skill score SS = model_RMSE / reference_RMSE, daily buckets
error_growth_single_valid_time RMSE vs lead time for a fixed valid time, with clim. baseline
"""

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.metrics import LEAD_DAY_BUCKETS as DAY_BUCKETS
from deep_lst.evaluation.metrics import _acc
from deep_lst.evaluation.plots._design import _LABEL_ALIASES, RC
from deep_lst.evaluation.plots._design import CASE_COLORS as _CASE_COLORS
from deep_lst.evaluation.plots._design import DEFAULT_COLOR as _DEFAULT_COLOR


def error_growth_plot(
    scores_by_lead: xr.Dataset,
    case_label: str,
    metric: str,
    save_path,
    eval_region: str,
) -> None:
    """Pooled metric vs lead time — single model line, no band.

    ``scores_by_lead`` is a pooled-per-lead score dataset (init + space pooled per lead in the
    squared-error domain); see metrics.compute_pooled_scores.
    """
    plt.rcParams.update(RC)
    lead_td = pd.TimedeltaIndex(scores_by_lead["lead_time"].values)
    lead_hours = [int(lt.total_seconds() / 3600) for lt in lead_td]
    vals = scores_by_lead[metric].values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    model_color = _CASE_COLORS.get(case_label, _DEFAULT_COLOR)
    ax.plot(lead_hours, vals, color=model_color, linewidth=2, marker="o", markersize=5, label="Deep LST")

    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"{metric.upper()} (K)")
    step = max(1, len(lead_hours) // 12)
    ax.set_xticks(lead_hours[::step])
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def error_growth_daily_plot(
    scores_by_day: xr.Dataset,
    case_label: str,
    metric: str,
    save_path,
    eval_region: str,
) -> None:
    """Pooled metric per forecast-day bucket — single model line, no band.

    ``scores_by_day`` is a pooled-per-day score dataset (init + the bucket's leads + space pooled
    per forecast day); see metrics.compute_pooled_by_day.
    """
    if scores_by_day.sizes.get("day", 0) == 0:
        return
    plt.rcParams.update(RC)
    day_labels = scores_by_day["day_label"].values
    day_means = scores_by_day[metric].values
    x = np.arange(len(day_means))

    fig, ax = plt.subplots(figsize=(max(5, len(day_means) * 1.0), 4.5))
    model_color = _CASE_COLORS.get(case_label, _DEFAULT_COLOR)
    ax.plot(x, day_means, color=model_color, linewidth=2, marker="o", markersize=8, label="Deep LST")

    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45)
    ax.set_ylabel(f"{metric.upper()} (K)")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def error_growth_single_valid_time(
    ds_by_lead: xr.Dataset,
    case_label: str,
    metric: str,
    save_path,
    valid_time: pd.Timestamp,
    climatological_rmse: float,
) -> None:
    """RMSE vs lead time for a single fixed valid_time.

    The valid time was reached from different init times, so each point on the
    x-axis corresponds to a different initial condition / lead time.
    A horizontal line marks the climatological RMSE at that valid time.
    """
    plt.rcParams.update(RC)
    lead_td = pd.TimedeltaIndex(ds_by_lead["lead_time"].values)
    lead_hours = [int(lt.total_seconds() / 3600) for lt in lead_td]
    vals = ds_by_lead[metric].values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    model_color = _CASE_COLORS.get(case_label, _DEFAULT_COLOR)
    ax.plot(lead_hours, vals, color=model_color, linewidth=2, marker="o", markersize=5, label="Deep LST")

    for bl_label, bl_val in climatological_rmse.items():
        if bl_val is not None and np.isfinite(bl_val):
            c = _CASE_COLORS.get(bl_label, _DEFAULT_COLOR)
            ax.axhline(bl_val, color=c, linewidth=1.5, linestyle="--", label=f"{bl_label}  ({bl_val:.2f} K)")

    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"{metric.upper()} (K)")
    step = max(1, len(lead_hours) // 12)
    ax.set_xticks(lead_hours[::step])
    ax.set_title(f"Valid time: {valid_time.strftime('%Y-%m-%d %H:%M')}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def skill_plot(
    references: dict[str, xr.Dataset],
    case_label: str,
    metric: str,
    save_path,
    eval_region: str,
) -> None:
    """Skill score vs forecast day, one line per reference, no band.

    Each value in ``references`` is a pooled-per-day skill dataset (1 - pooled_model / pooled_ref;
    see metrics.compute_pooled_by_day). 0 is marked as the break-even line.
    """
    plt.rcParams.update(RC)
    first = next(iter(references.values()))
    if first.sizes.get("day", 0) == 0:
        return
    day_labels = first["day_label"].values
    x = np.arange(len(day_labels))

    fig, ax = plt.subplots(figsize=(max(5, len(x) * 1.0), 4.5))
    for ref_label, ds_ref in references.items():
        display_label = _LABEL_ALIASES.get(ref_label, ref_label)
        c = _CASE_COLORS.get(ref_label, _DEFAULT_COLOR)
        ax.plot(x, ds_ref[metric].values, color=c, linewidth=2, marker="o", markersize=8, label=display_label)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(None, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45)
    ax.set_ylabel(f"Skill Score ({metric.upper()})", fontweight="bold")
    ax.legend(title="Reference", loc="upper right", borderaxespad=0, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def _day_bucket_masks(pred):
    """Per forecast-day lead-time boolean masks (24 h each) that fit the horizon, with labels.

    Bucketing over a full day averages out the 6-hourly init/valid-hour aliasing that makes the
    per-lead day/night view bumpy (a 6 h-wide valid-hour window samples one init group per lead).
    """
    lead_h = (pred["lead_time"] / np.timedelta64(1, "h")).values
    day_labels, day_lead_masks = [], []
    for day_idx, (lo, hi) in enumerate(DAY_BUCKETS, start=1):
        lead_mask = (lead_h >= lo) & (lead_h <= hi)
        if not lead_mask.any():
            continue
        day_labels.append(f"Day {day_idx}")
        day_lead_masks.append(xr.DataArray(lead_mask, dims="lead_time", coords={"lead_time": pred["lead_time"]}))
    return day_labels, day_lead_masks


def skill_day_night_plot(pred, obs, baselines, var, save_path, eval_region, clim=None):
    """RMSE skill vs forecast day, one curve per baseline, split into daytime / nighttime panels.

    Shows only the average tendency: per forecast-day bucket (24 lead-hours) and valid-hour
    window the SSE is pooled over init_time, the bucket's lead times AND pixels into a pooled
    RMSE, and the skill 1 - RMSE_model / RMSE_baseline is drawn as a single line per baseline.
    Daily bucketing removes the 6-hourly aliasing of the per-lead view. No uncertainty band.

    Day/night follow the established valid-hour windows (daytime 10–15 h, nighttime 22–03 h);
    valid hour = init hour + lead hour (mod 24).

    If ``clim`` (the climatological reference on the same grid as ``pred``) is given, the anomaly
    correlation (ACC) of the model vs that climatology is added on a right-hand y-axis as a dashed
    dark-grey line, so skill and ACC share the panel.
    """
    plt.rcParams.update(RC)
    pred = pred.isel(lead_time=slice(1, None))  # drop lead-0 placeholder
    obs = obs.isel(lead_time=slice(1, None))
    space = [d for d in ("lat", "lon", "pixel") if d in pred.dims]
    valid_hour = (pred["init_time"].dt.hour + (pred["lead_time"] / np.timedelta64(1, "h")).astype(int)) % 24

    err2_model = ((pred - obs) ** 2).load()
    err2_baselines = {
        label: ((field.isel(lead_time=slice(1, None)) - obs) ** 2).load() for label, field in baselines.items()
    }
    if clim is not None:
        clim = clim.isel(lead_time=slice(1, None))
        anom_pred = (pred - clim).load()
        anom_obs = (obs - clim).load()

    day_labels, day_lead_masks = _day_bucket_masks(pred)
    x = np.arange(len(day_labels))

    panels = (
        ("Daytime (10–15 h)", (valid_hour >= 10) & (valid_hour <= 15)),
        ("Nighttime (22–03 h)", (valid_hour >= 22) | (valid_hour <= 3)),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    acc_axes = []
    for ax, (panel_title, tod_mask) in zip(axes, panels):
        for label, err2_bl in err2_baselines.items():
            skill = []
            for lead_mask in day_lead_masks:
                bucket_mask = tod_mask & lead_mask
                rmse_model = np.sqrt(
                    err2_model.where(bucket_mask).mean(["init_time", "lead_time", *space], skipna=True)
                )
                rmse_bl = np.sqrt(err2_bl.where(bucket_mask).mean(["init_time", "lead_time", *space], skipna=True))
                skill.append(float(1 - rmse_model / rmse_bl))
            display_label = _LABEL_ALIASES.get(label, label)
            color = _CASE_COLORS.get(label, _DEFAULT_COLOR)
            ax.plot(x, skill, color=color, linewidth=2, marker="o", markersize=6, label=display_label)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylim(None, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(day_labels, rotation=45)
        ax.set_xlabel("Forecast day")
        ax.set_title(panel_title)
        ax.spines["top"].set_visible(False)
        if clim is None:
            ax.spines["right"].set_visible(False)
        else:
            acc_ax = ax.twinx()
            acc_axes.append(acc_ax)
            acc = []
            for lead_mask in day_lead_masks:
                bucket_mask = tod_mask & lead_mask
                acc.append(
                    float(
                        _acc(
                            anom_pred.where(bucket_mask),
                            anom_obs.where(bucket_mask),
                            dim=["init_time", "lead_time", *space],
                        )
                    )
                )
            acc_ax.plot(x, acc, color="0.3", linestyle="--", linewidth=2, marker="s", markersize=4, label="ACC")
            acc_ax.set_ylim(0, 1)
            acc_ax.spines["top"].set_visible(False)

    axes[0].set_ylabel("Skill score (RMSE)")
    if clim is not None:
        acc_axes[0].set_yticklabels([])  # only the right-most panel shows the ACC axis
        acc_axes[1].set_ylabel("ACC")
        handles, labels_ = axes[1].get_legend_handles_labels()
        acc_handles, acc_labels = acc_axes[1].get_legend_handles_labels()
        acc_axes[1].legend(
            handles + acc_handles, labels_ + acc_labels, title="Reference", loc="upper right", framealpha=0.9
        )
    else:
        axes[0].legend(title="Reference", loc="upper right", framealpha=0.9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def acc_day_night_plot(pred, obs, clim, var, save_path, eval_region):
    """Anomaly correlation (ACC) vs forecast day, daytime and nighttime as two curves in one panel.

    Anomalies are formed against the climatological reference `clim` (pred - clim, obs - clim).
    Per forecast-day bucket (24 lead-hours) and valid-hour window the uncentred ACC is pooled over
    init_time, the bucket's lead times AND pixels (daytime 10–15 h, nighttime 22–03 h; valid hour
    = init hour + lead hour). Daily bucketing removes the 6-hourly aliasing of the per-lead view.
    Single panel, two curves, no uncertainty band.
    """
    plt.rcParams.update(RC)
    pred = pred.isel(lead_time=slice(1, None))  # drop lead-0 placeholder
    obs = obs.isel(lead_time=slice(1, None))
    clim = clim.isel(lead_time=slice(1, None))
    space = [d for d in ("lat", "lon", "pixel") if d in pred.dims]
    valid_hour = (pred["init_time"].dt.hour + (pred["lead_time"] / np.timedelta64(1, "h")).astype(int)) % 24

    anom_pred = (pred - clim).load()
    anom_obs = (obs - clim).load()

    day_labels, day_lead_masks = _day_bucket_masks(pred)
    x = np.arange(len(day_labels))

    cmap = plt.get_cmap("twilight_shifted")
    panels = (
        ("Daytime (10–15 h)", (valid_hour >= 10) & (valid_hour <= 15), cmap(12 / 23.0)),
        ("Nighttime (22–03 h)", (valid_hour >= 22) | (valid_hour <= 3), cmap(0 / 23.0)),
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for panel_title, tod_mask, color in panels:
        acc = []
        for lead_mask in day_lead_masks:
            bucket_mask = tod_mask & lead_mask
            acc_value = _acc(
                anom_pred.where(bucket_mask), anom_obs.where(bucket_mask), dim=["init_time", "lead_time", *space]
            )
            acc.append(float(acc_value))
        ax.plot(x, acc, color=color, linewidth=2, marker="o", markersize=6, label=panel_title)

    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45)
    ax.set_xlabel("Forecast day")
    ax.set_ylabel(f"{var} ACC")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def error_growth_day_night_plot(pred, obs, var, save_path, eval_region):
    """RMSE vs lead for all 24 valid hours-of-day (colour gradient + colorbar), with local midday
    (12 h) and midnight (00 h) emphasised on top.

    Fixing the valid hour removes the diurnal-aliasing wiggle (no within-window hour variation);
    with 6-hourly inits each hour recurs every 6 lead-hours, so the curves are 6-hourly. Computed
    from raw pred/obs (valid hour = init + lead). No uncertainty band.
    """
    plt.rcParams.update(RC)
    pred = pred.isel(lead_time=slice(1, None))  # drop lead-0 placeholder
    obs = obs.isel(lead_time=slice(1, None))
    space = [d for d in ("lat", "lon", "pixel") if d in pred.dims]
    valid_hour = (pred["init_time"].dt.hour + (pred["lead_time"] / np.timedelta64(1, "h")).astype(int)) % 24
    err2 = ((pred - obs) ** 2).load()  # in-memory once; the 24 hourly reductions then run fast
    lead_h = (pred["lead_time"] / np.timedelta64(1, "h")).values
    cmap = plt.get_cmap("twilight_shifted")

    rmse_by_hour = {
        hour: np.sqrt(err2.where(valid_hour == hour).mean(["init_time", *space], skipna=True)).values
        for hour in range(24)
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for hour in range(24):
        keep = np.isfinite(rmse_by_hour[hour])
        ax.plot(lead_h[keep], rmse_by_hour[hour][keep], color=cmap(hour / 23.0), lw=1.0, zorder=2)
    for hour, label in ((12, "Midday (12 h)"), (0, "Midnight (00 h)")):
        keep = np.isfinite(rmse_by_hour[hour])
        ax.plot(
            lead_h[keep],
            rmse_by_hour[hour][keep],
            color=cmap(hour / 23.0),
            lw=2.6,
            marker="o",
            ms=3,
            label=label,
            zorder=3,
            path_effects=[pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()],
        )
    ax.set_xlabel("Lead time [h]")
    ax.set_ylabel(f"{var} RMSE [K]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 23))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Valid hour of day", ticks=[0, 6, 12, 18, 23])
    ax.set_title(f"{eval_region} — {var} RMSE growth by valid hour")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def error_growth_by_init_plot(pred, obs, var, save_path, eval_region):
    """RMSE per forecast day, one curve per initialisation hour (no init-hour mixing).

    Each curve fixes the init hour; per forecast-day bucket the SSE is pooled over the inits with
    that hour, the bucket's lead times AND pixels into a pooled RMSE. Bucketing averages out the
    within-day diurnal oscillation of the valid time, leaving the initialisation-time skill
    dependence (vertical separation between curves) and its growth across forecast days.
    """
    plt.rcParams.update(RC)
    pred = pred.isel(lead_time=slice(1, None))  # drop lead-0 placeholder
    obs = obs.isel(lead_time=slice(1, None))
    space = [d for d in ("lat", "lon", "pixel") if d in pred.dims]
    init_hour = pred["init_time"].dt.hour
    err2 = ((pred - obs) ** 2).load()
    day_labels, day_lead_masks = _day_bucket_masks(pred)
    x = np.arange(len(day_labels))
    cmap = plt.get_cmap("twilight_shifted")

    fig, ax = plt.subplots(figsize=(9, 5))
    for hour in sorted(set(int(h) for h in init_hour.values)):
        err2_hour = err2.where(init_hour == hour)
        rmse = []
        for lead_mask in day_lead_masks:
            rmse.append(
                float(np.sqrt(err2_hour.where(lead_mask).mean(["init_time", "lead_time", *space], skipna=True)))
            )
        ax.plot(x, rmse, color=cmap(hour / 23.0), lw=1.6, marker="o", markersize=5, label=f"init {hour:02d}h")
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45)
    ax.set_xlabel("Forecast day")
    ax.set_ylabel(f"{var} RMSE [K]")
    ax.grid(alpha=0.3)
    ax.legend(title="Initialisation hour", loc="upper left")
    ax.set_title(f"{eval_region} — {var} RMSE growth by initialisation hour")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
