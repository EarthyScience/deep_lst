from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from deep_lst.evaluation.plots._design import _LABEL_ALIASES, RC
from deep_lst.evaluation.plots._design import CASE_COLORS as _CASE_COLORS
from deep_lst.evaluation.plots._design import DEFAULT_COLOR as _DEFAULT_COLOR


def srmse_by_latitude(
    pred: xr.DataArray,
    obs: xr.DataArray,
    var: str,
    save_path,
    eval_region: str,
    leads_h: tuple[int, ...] = (24, 72, 120),
    lat_bin_width: float = 3.0,
    n_boot: int = 300,
) -> None:
    """Standardised RMSE (RMSE / sigma_LST) profiled by latitude, one curve per lead.

    Per latitude bin everything is POOLED in the squared-error domain over {init_time, the bin's
    pixels}, consistent with the tables and the by-lead plots: the RMSE pools the bin's squared
    errors (not a mean of per-pixel RMSEs); sigma is the pooled obs std over the bin; validity is the
    pooled finite fraction. sigma is a fixed, lead-independent normaliser taken from the obs at the
    first lead, so the curves are comparable across leads. The bootstrap CI resamples the bin's
    pixels and re-pools. A twin x-axis overlays the observation validity (% clear-sky) per bin.
    Meant for the temporal-extrapolation eval over the spread-out training sites. Leads absent from
    the horizon are replaced by three evenly spaced available leads.
    """
    plt.rcParams.update(RC)
    lead_int = (pred["lead_time"] / np.timedelta64(1, "h")).astype(int).values
    leads_h = [h for h in leads_h if h in lead_int]
    if not leads_h:
        available = sorted(int(h) for h in lead_int if h >= 1)
        if not available:
            return
        leads_h = sorted(
            {available[len(available) // 4], available[len(available) // 2], available[3 * len(available) // 4]}
        )

    # Lead-independent context from obs at the first lead (lead 0 is a placeholder): per-pixel
    # additive sufficient statistics, so a bin's pooled obs std and validity are exact re-pools.
    obs_ref = obs.isel(lead_time=1)
    n_init = obs_ref.sizes["init_time"]
    obs_sum = obs_ref.sum("init_time").values.ravel()
    obs_sqsum = (obs_ref**2).sum("init_time").values.ravel()
    obs_count = obs_ref.notnull().sum("init_time").values.ravel()
    lat_flat = pred["lat"].broadcast_like(obs_ref.notnull().sum("init_time")).values.ravel()

    lat_bins = np.arange(np.floor(np.nanmin(lat_flat)), np.ceil(np.nanmax(lat_flat)) + lat_bin_width, lat_bin_width)
    centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    idx = np.digitize(lat_flat, lat_bins)
    rng = np.random.default_rng(0)

    sigma_bin, val_bin = [], []
    for i in range(1, len(lat_bins)):
        in_bin = np.where(idx == i)[0]
        count = obs_count[in_bin].sum()
        if in_bin.size < 3 or count == 0:
            sigma_bin.append(np.nan)
            val_bin.append(np.nan)
            continue
        mean = obs_sum[in_bin].sum() / count
        variance = obs_sqsum[in_bin].sum() / count - mean**2
        sigma_bin.append(np.sqrt(variance) if variance > 0 else np.nan)
        val_bin.append(100.0 * count / (n_init * in_bin.size))
    sigma_bin = np.array(sigma_bin)
    val_bin = np.array(val_bin)

    fig, ax = plt.subplots(figsize=(7.5, 11))
    for h in leads_h:
        lead_idx = int(np.where(lead_int == h)[0][0])
        err2 = (pred.isel(lead_time=lead_idx) - obs.isel(lead_time=lead_idx)) ** 2
        se_pix = err2.sum("init_time").values.ravel()
        count_pix = err2.notnull().sum("init_time").values.ravel()
        med, lo, hi = [], [], []
        for i in range(1, len(lat_bins)):
            in_bin = np.where((idx == i) & (count_pix > 0))[0]
            sigma = sigma_bin[i - 1]
            if in_bin.size < 3 or not np.isfinite(sigma):
                med.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
                continue
            se_bin = se_pix[in_bin]
            count_bin = count_pix[in_bin]
            med.append(np.sqrt(se_bin.sum() / count_bin.sum()) / sigma)
            boot = []
            for _ in range(n_boot):
                sample = rng.integers(0, in_bin.size, in_bin.size)
                boot.append(np.sqrt(se_bin[sample].sum() / count_bin[sample].sum()) / sigma)
            lo.append(np.percentile(boot, 2.5))
            hi.append(np.percentile(boot, 97.5))
        (line,) = ax.plot(med, centers, "-o", ms=4, label=f"Lead time {h} h")
        ax.fill_betweenx(centers, lo, hi, color=line.get_color(), alpha=0.2)

    ax.set_xlabel(r"RMSE / $\sigma_{\mathrm{LST}}$  [-]")
    ax.set_ylabel("Latitude [deg]")
    ax.legend(loc="lower right")

    axv = ax.twiny()
    axv.plot(val_bin, centers, "k--", alpha=0.6)
    axv.set_xlabel("Observation validity [% clear-sky pixels]")

    ax.set_title(f"{eval_region} — {var} sRMSE by latitude")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def activity_plot(
    references: dict[str, xr.Dataset],
    save_path,
    eval_region: str,
) -> None:
    """Anomaly activity (pooled pred std) by forecast-day bucket, one line per reference, no band.

    Each value in ``references`` is a pooled-per-day score dataset (see metrics.compute_pooled_by_day).
    Unlike skill_plot there is no break-even line and no hard upper y-limit, because activity is an
    absolute quantity in Kelvin, not a bounded skill score.
    """
    plt.rcParams.update(RC)
    first = next(iter(references.values()))
    if first.sizes.get("day", 0) == 0:
        return
    day_labels = first["day_label"].values
    x = np.arange(len(day_labels))

    fig, ax = plt.subplots(figsize=(max(5, len(x) * 1.2), 4.5))

    for ref_label, ds_ref in references.items():
        c = _CASE_COLORS.get(ref_label, _DEFAULT_COLOR)
        display_label = _LABEL_ALIASES.get(ref_label, ref_label)
        ax.plot(x, ds_ref["activity"].values, color=c, linewidth=2, marker="o", markersize=8, label=display_label)

    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45)
    ax.set_ylabel("Anomaly Std (K)", fontweight="bold")
    ax.legend(title="Reference", loc="upper right", borderaxespad=0, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"  Saved: {save_path}")
    plt.close(fig)
