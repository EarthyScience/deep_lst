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

from deep_lst.evaluation.plots._design import _LABEL_ALIASES, RC
from deep_lst.evaluation.plots._design import CASE_COLORS as _CASE_COLORS
from deep_lst.evaluation.plots._design import DEFAULT_COLOR as _DEFAULT_COLOR

DAY_BUCKETS = [
    (1, 24),
    (25, 48),
    (49, 72),
    (73, 96),
    (97, 120),
    (121, 144),
    (145, 168),
    (169, 192),
    (193, 217),
    (218, 240),
]


def error_growth_plot(
    ds_by_lead: xr.Dataset,
    case_label: str,
    metric: str,
    save_path,
    eval_region: str,
    # baseline_rmse: dict[str, float] | None = None,
) -> None:
    """RMSE vs lead time — model line + flat baseline reference lines.

    Baselines are scored against the same observations the model sees at
    ``--lead_time`` (one RMSE value, independent of lead time). The model's
    RMSE grows because at longer lead times the forecast was launched from an
    earlier init_time; the baseline has no such dependence, so it appears as
    a flat horizontal reference.
    """
    plt.rcParams.update(RC)
    lead_td = pd.TimedeltaIndex(ds_by_lead["lead_time"].values)
    lead_hours = [int(lt.total_seconds() / 3600) for lt in lead_td]
    vals = ds_by_lead[f"{metric}_q50"].values
    lo = ds_by_lead[f"{metric}_q05"].values
    hi = ds_by_lead[f"{metric}_q95"].values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    model_color = _CASE_COLORS.get(case_label, _DEFAULT_COLOR)
    ax.fill_between(lead_hours, lo, hi, color=model_color, alpha=0.5)
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
    ds_by_lead: xr.Dataset,
    case_label: str,
    metric: str,
    save_path,
    eval_region: str,
    # baseline_rmse: dict[str, float] | None = None,
) -> None:
    """Mean RMSE per day bucket (day 1: 1–24 h, day 2: 25–48 h, day 3: 49–72 h).

    Model values are averaged over the lead times in each bucket.
    Baselines are flat horizontal references (lead-time independent).
    """
    plt.rcParams.update(RC)
    lead_td = pd.TimedeltaIndex(ds_by_lead["lead_time"].values)
    lead_hours = np.array([int(lt.total_seconds() / 3600) for lt in lead_td])
    vals = ds_by_lead[f"{metric}_q50"].values
    q_lo = ds_by_lead[f"{metric}_q05"].values
    q_hi = ds_by_lead[f"{metric}_q95"].values

    day_labels, day_means, day_los, day_his = [], [], [], []

    for day_idx, (blo, bhi) in enumerate(DAY_BUCKETS, start=1):
        mask = (lead_hours >= blo) & (lead_hours <= bhi)
        if not mask.any():
            continue
        day_labels.append(f"Day {day_idx}\n({blo}–{bhi} h)")
        day_means.append(float(np.nanmean(vals[mask])))
        day_los.append(float(np.nanmean(q_lo[mask])))
        day_his.append(float(np.nanmean(q_hi[mask])))

    if not day_means:
        return

    x = np.arange(len(day_means))
    day_means = np.array(day_means)
    day_los = np.array(day_los)
    day_his = np.array(day_his)

    fig, ax = plt.subplots(figsize=(max(5, len(day_means) * 1.5), 4.5))

    model_color = _CASE_COLORS.get(case_label, _DEFAULT_COLOR)
    ax.plot(x, day_means, color=model_color, linewidth=2, marker="o", markersize=8, label="Deep LST")
    ax.fill_between(x, day_los, day_his, color=model_color, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(day_labels)
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
    """Skill score vs lead time, daily buckets.

    Values in ``references`` are plotted directly — apply any transformation
    (e.g. ``1 - ratio``) before passing. 0 is marked as the break-even line.
    """
    plt.rcParams.update(RC)
    first = next(iter(references.values()))
    lead_td = pd.TimedeltaIndex(first["lead_time"].values)
    lead_hours = np.array([int(lt.total_seconds() / 3600) for lt in lead_td])

    x, day_labels = [], []
    for day_idx, (lo, hi) in enumerate(DAY_BUCKETS, start=1):
        mask = (lead_hours >= lo) & (lead_hours <= hi)
        if not mask.any():
            continue
        day_labels.append(f"Day {day_idx}")
        x.append(day_idx - 1)

    if not x:
        return
    x = np.array(x)

    fig, ax = plt.subplots(figsize=(max(5, len(x) * 1.0), 4.5))

    for ref_label, ds_ref in references.items():
        skill = ds_ref[f"{metric}_q50"].values
        skill_q_lo = ds_ref[f"{metric}_q05"].values
        skill_q_hi = ds_ref[f"{metric}_q95"].values

        day_median, day_q_lo, day_q_hi = [], [], []
        for lo, hi in DAY_BUCKETS:
            mask = (lead_hours >= lo) & (lead_hours <= hi)
            if not mask.any():
                continue
            day_median.append(float(np.nanmedian(skill[mask])))
            day_q_lo.append(float(np.nanmean(skill_q_lo[mask])))
            day_q_hi.append(float(np.nanmean(skill_q_hi[mask])))

        day_median = np.array(day_median)
        day_q_lo = np.array(day_q_lo)
        day_q_hi = np.array(day_q_hi)
        display_label = _LABEL_ALIASES.get(ref_label, ref_label)
        c = _CASE_COLORS.get(ref_label, _DEFAULT_COLOR)
        ax.plot(x, day_median, color=c, linewidth=2, marker="o", markersize=8, label=display_label)
        ax.fill_between(x, day_q_lo, day_q_hi, color=c, alpha=0.3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(None, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, fontweight="bold")
    ax.set_ylabel(f"Skill Score ({metric.upper()})", fontweight="bold")
    ax.legend(title="Reference", loc="lower left", borderaxespad=0, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
        print(f"  Saved: {save_path}")
    plt.close(fig)


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
    """RMSE vs lead, one curve per initialisation hour (no init-hour mixing).

    Each curve fixes the init hour, so its oscillation is the diurnal cycle of the *valid* time
    (valid hour = init + lead); the vertical separation between curves is the initialisation-time
    skill dependence. The companion to the by-valid-hour plot, which instead fixes the valid hour.
    """
    plt.rcParams.update(RC)
    pred = pred.isel(lead_time=slice(1, None))  # drop lead-0 placeholder
    obs = obs.isel(lead_time=slice(1, None))
    space_axes = (0,) + tuple(i for i, d in enumerate(pred.dims) if d not in ("init_time", "lead_time"))
    init_hour = (pred["init_time"].dt.hour.values) % 24
    err2 = ((pred - obs) ** 2).transpose("init_time", "lead_time", ...).values  # (init, lead, *space)
    lead_h = (pred["lead_time"] / np.timedelta64(1, "h")).values
    cmap = plt.get_cmap("twilight_shifted")

    fig, ax = plt.subplots(figsize=(9, 5))
    for hour in sorted(set(init_hour.tolist())):
        with np.errstate(invalid="ignore"):
            rmse = np.sqrt(np.nanmean(err2[init_hour == hour], axis=space_axes))  # reduce init + space → (lead,)
        keep = np.isfinite(rmse)
        ax.plot(lead_h[keep], rmse[keep], color=cmap(hour / 23.0), lw=1.6, label=f"init {hour:02d}h")
    ax.set_xlabel("Lead time [h]")
    ax.set_ylabel(f"{var} RMSE [K]")
    ax.grid(alpha=0.3)
    ax.legend(title="Initialisation hour", loc="upper left")
    ax.set_title(f"{eval_region} — {var} RMSE growth by initialisation hour")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
