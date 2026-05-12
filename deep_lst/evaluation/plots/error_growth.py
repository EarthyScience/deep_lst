"""Error-growth and persistence-skill plots for multi-lead-time forecasts.

All three functions consume a pre-computed metrics_by_lead_time dataset
(xr.Dataset with a 'lead_time' dim and metric variables such as 'rmse').

error_growth_plot              RMSE vs lead time — hourly resolution line plot
error_growth_daily_plot        Mean RMSE bucketed into forecast days (day 1/2/3 …)
skill_plot                     Skill score SS = model_RMSE / reference_RMSE, daily buckets
error_growth_single_valid_time RMSE vs lead time for a fixed valid time, with clim. baseline
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.plots._design import _LABEL_ALIASES, RC
from deep_lst.evaluation.plots._design import CASE_COLORS as _CASE_COLORS
from deep_lst.evaluation.plots._design import DEFAULT_COLOR as _DEFAULT_COLOR


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
    vals = ds_by_lead[metric].values
    stds = ds_by_lead[f"{metric}_std"].values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    model_color = _CASE_COLORS.get(case_label, _DEFAULT_COLOR)
    ax.fill_between(lead_hours, vals + 2 * stds, vals - 2 * stds, color=model_color, alpha=0.5)
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
    vals = ds_by_lead[metric].values
    stds = ds_by_lead[f"{metric}_std"].values

    DAY_BUCKETS = [(1, 24), (25, 48), (49, 72), (73, 96), (97, 120), (121, 144), (145, 168)]
    day_labels, day_means, day_vars = [], [], []

    for day_idx, (lo, hi) in enumerate(DAY_BUCKETS, start=1):
        mask = (lead_hours >= lo) & (lead_hours <= hi)
        if not mask.any():
            continue
        day_labels.append(f"Day {day_idx}\n({lo}–{hi} h)")
        day_means.append(float(np.nanmean(vals[mask])))
        day_vars.append(float(np.nanmean(stds[mask])))

    if not day_means:
        return

    x = np.arange(len(day_means))
    day_means = np.array(day_means)
    day_vars = np.array(day_vars)

    fig, ax = plt.subplots(figsize=(max(5, len(day_means) * 1.5), 4.5))

    model_color = _CASE_COLORS.get(case_label, _DEFAULT_COLOR)
    ax.plot(x, day_means, color=model_color, linewidth=2, marker="o", markersize=8, label="Deep LST")
    ax.fill_between(x, day_means + 2 * day_vars, day_means - 2 * day_vars, color=model_color, alpha=0.5)

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
    """Skill score (median per-time model/ref RMSE ratio) vs lead time, daily buckets.

    Each dataset in ``references`` must come from compute_skill_scores_by_lead_time
    and contain {metric}, {metric}_q05, {metric}_q95.
    SS < 1 means the model beats the reference; SS = 1 is the break-even line.
    """
    plt.rcParams.update(RC)
    first = next(iter(references.values()))
    lead_td = pd.TimedeltaIndex(first["lead_time"].values)
    lead_hours = np.array([int(lt.total_seconds() / 3600) for lt in lead_td])

    DAY_BUCKETS = [(1, 24), (25, 48), (49, 72), (73, 96), (97, 120), (121, 144), (145, 168)]
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

    fig, ax = plt.subplots(figsize=(max(5, len(x) * 1.2), 4.5))

    for ref_label, ds_ref in references.items():
        skill = ds_ref[metric].values
        q_lo_key = next(k for k in ds_ref.data_vars if k.startswith(f"{metric}_q") and not k.endswith("95"))
        q_hi_key = next(k for k in ds_ref.data_vars if k.startswith(f"{metric}_q") and not k.endswith("05"))
        skill_q_lo = ds_ref[q_lo_key].values
        skill_q_hi = ds_ref[q_hi_key].values

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

    ax.set_ylim(0, None)
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, fontweight="bold")
    ax.set_ylabel(f"Skill ({metric.upper()})", fontweight="bold")
    ax.legend(title="Reference", loc="upper right", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
        print(f"  Saved: {save_path}")
    plt.close(fig)
