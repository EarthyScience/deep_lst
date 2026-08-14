"""Retrieval-noise temporal check at the nine eval-region centres (CRPS pre-work).

No-truth check of the SEVIRI LST retrieval error against the product's standard_error
(SE), one centre pixel per eval region, hourly 2016-2024. No station data involved;
the one assumption is that SE is the total 1-sigma error of each retrieval (A1).

  white floor   Nighttime LST evolves smoothly while retrieval noise does not. The
                semivariogram gamma(k) = 0.5 * mean((LST(t+k) - LST(t))^2) over
                clear-sky within-night pairs, fitted as gamma(k) = c0 + c2 * k^2
                (the quadratic absorbs the smooth cooling signal), has the white
                error variance as its intercept: sigma_w^2 = c0.
  f estimate    With SE^2 the total error variance, the static (hour-to-hour
                correlated) fraction is f_hat = 1 - sigma_w^2 / median(SE)^2.
                A LOWER bound: rough sub-hourly surface signal inflates sigma_w.
  tracking      Pearson r between per-night 0.5*mean(diff^2) and per-night median
                SE. If SE varies but the white floor does not follow, SE variation
                is static-component variation (independent support for a high f).
  persistence   Lag-1..4 autocorrelation of quadratically detrended within-night
                residuals, compared against a WHITE-NOISE NULL simulated on the
                same cloud-gap pattern and pushed through the identical detrend
                (fitting a quadratic to ~6-point segments induces an artificial
                oscillatory ACF, so the raw values are not interpretable alone).
                Observed ~ null justifies calling the nugget "white".

Night-only by construction (22-04 local solar): daytime solar-driven variability
makes the signal rough at 1 h, so the daytime white error is not estimable this way.
SE Gaussianity is assumed throughout, not tested.

Run:  .venv/bin/python -m deep_lst.exploration.autoregression
Out:  results/autoregression/{table.csv, variograms.png, night_acf.png}
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eodata.grids import T1H

from deep_lst.dataset.dataset import DATA_TYPES, _preload_columns
from deep_lst.evaluation.plots._design import OBS_COLOR, RC
from deep_lst.exploration.full_store import SEVIRI_STORE, open_store_reader
from deep_lst.exploration.plots.kg_sampling import EVAL_CENTERS
from deep_lst.utils.spatial import store_grid

OUT_DIR = Path("/Net/Groups/BGI/people/mwessel/deep_lst/results/autoregression")
SEVIRI_LOAD = {"data": ["LST"], "metadata": ["standard_error"]}
DAY_RANGE = (T1H.date_to_index("2016-01-01") // 24, T1H.date_to_index("2025-01-01") // 24)

NIGHT_START, NIGHT_END = 22.0, 4.0  # local solar hours; the within-night window
LAGS = (1, 2, 3, 4)  # variogram / ACF lags [h]; > 4 leaves the 6 h window
MIN_NIGHT_HOURS = 5  # clear hours a night needs for the quadratic detrend
MIN_PAIRS = 300  # lag-1 pairs below this -> flag low_n in the table
SEASONS = {
    "all": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

FIT_COLOR = "#1f77b4"  # quadratic variogram fit
SE_COLOR = "#d62728"  # the SE^2 "all-white" reference level


def center_pixels():
    """Nearest seviri store row per eval centre: {display_name: (row, lon, lat)}."""
    centers = store_grid("seviri").centers  # (n_rows, 2) lon/lat
    pixels = {}
    for name, (lon, lat) in EVAL_CENTERS.items():
        display_name = name.replace("\n", "")
        distance2 = (centers[:, 0] - lon) ** 2 + (centers[:, 1] - lat) ** 2
        row = int(np.argmin(distance2))
        pixels[display_name] = (row, float(centers[row, 0]), float(centers[row, 1]))
    return pixels


def load_center_columns(rows, num_workers=4):
    """Raw hourly LST + SE for explicit store rows (full_store.load_store_columns
    minus the anchor mapping). Returns (lst, se): (n_rows, n_hours), Kelvin, NaN-gapped."""
    reader = open_store_reader("seviri", SEVIRI_STORE, SEVIRI_LOAD)
    select = {
        data_type: [list(getattr(reader, DATA_TYPES[data_type]).dtype.fields).index(var) for var in names]
        for data_type, names in SEVIRI_LOAD.items()
    }
    columns = _preload_columns(reader, None, rows, select, num_workers=num_workers, day_range=DAY_RANGE)
    lst = columns["data"].numpy()[:, :, 0]
    se = columns["metadata"].numpy()[:, :, 0]
    return lst, se


def night_axes(n_hours, lon):
    """Night mask + night id for one centre. Local solar hour = UTC + lon/15; night_id
    groups the hours of one 22-04 window (22:00 through 03:00) under one integer."""
    utc_hour = np.arange(n_hours, dtype=float) % 24
    local_hour = (utc_hour + lon / 15.0) % 24
    night = (local_hour >= NIGHT_START) | (local_hour < NIGHT_END)
    absolute_local_hour = np.arange(n_hours, dtype=float) + lon / 15.0
    night_id = np.floor((absolute_local_hour - NIGHT_START) / 24.0).astype(int)
    return night, night_id


def night_variogram(lst_pixel, night, in_season):
    """gamma(k) = 0.5 * mean(diff^2) and pair counts over clear within-night pairs."""
    gamma = {}
    n_pairs = {}
    for lag in LAGS:
        left = lst_pixel[:-lag]
        right = lst_pixel[lag:]
        ok = np.isfinite(left) & np.isfinite(right)
        ok &= night[:-lag] & night[lag:]
        ok &= in_season[:-lag] & in_season[lag:]
        diffs = right[ok] - left[ok]
        gamma[lag] = 0.5 * float(np.mean(diffs**2)) if diffs.size else np.nan
        n_pairs[lag] = int(diffs.size)
    return gamma, n_pairs


def fit_nugget(gamma):
    """Least-squares fit gamma(k) = c0 + c2 * k^2; the intercept c0 is sigma_w^2."""
    ks = np.array([lag for lag in LAGS if np.isfinite(gamma[lag])], dtype=float)
    if ks.size < 3:
        return np.nan, np.nan
    values = np.array([gamma[int(k)] for k in ks])
    design = np.column_stack([np.ones_like(ks), ks**2])
    (c0, c2), *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(c0), float(c2)


def night_detrended_residuals(lst_pixel, night, night_id):
    """Residuals from a per-night quadratic fit (nights with >= MIN_NIGHT_HOURS clear
    hours), NaN elsewhere. The quadratic removes the smooth cooling curve."""
    residuals = np.full(lst_pixel.shape, np.nan)
    usable = night & np.isfinite(lst_pixel)
    for nid in np.unique(night_id[usable]):
        index = np.where(usable & (night_id == nid))[0]
        if index.size < MIN_NIGHT_HOURS:
            continue
        hours = index - index.mean()
        coefficients = np.polyfit(hours, lst_pixel[index], deg=2)
        residuals[index] = lst_pixel[index] - np.polyval(coefficients, hours)
    return residuals


def residual_acf(residuals, night_id):
    """Autocorrelation of the detrended night residuals, same-night pairs only."""
    acf = {}
    for lag in LAGS:
        left = residuals[:-lag]
        right = residuals[lag:]
        ok = np.isfinite(left) & np.isfinite(right) & (night_id[:-lag] == night_id[lag:])
        if ok.sum() < 30:
            acf[lag] = np.nan
            continue
        acf[lag] = float(np.corrcoef(left[ok], right[ok])[0, 1])
    return acf


def white_noise_null_acf(lst_pixel, night, night_id, rng, n_draws=3):
    """ACF the detrend machinery produces for PURE white noise with the same gaps.

    The per-night quadratic fit on ~6-point segments makes even white-noise
    residuals oscillate (negative at short lags, positive at longer ones), so the
    observed ACF is only interpretable relative to this null."""
    per_lag_draws = {lag: [] for lag in LAGS}
    for _ in range(n_draws):
        simulated = np.where(np.isfinite(lst_pixel), rng.standard_normal(lst_pixel.size), np.nan)
        residuals = night_detrended_residuals(simulated, night, night_id)
        acf = residual_acf(residuals, night_id)
        for lag in LAGS:
            per_lag_draws[lag].append(acf[lag])
    return {lag: float(np.nanmean(per_lag_draws[lag])) for lag in LAGS}


def tracking_correlation(lst_pixel, se_pixel, night, night_id):
    """Pearson r between per-night 0.5*mean(diff^2) and per-night median SE."""
    left = lst_pixel[:-1]
    right = lst_pixel[1:]
    ok = np.isfinite(left) & np.isfinite(right) & night[:-1] & night[1:]
    diffs = pd.DataFrame({"night": night_id[:-1][ok], "sq": (right[ok] - left[ok]) ** 2})
    per_night = diffs.groupby("night")["sq"].agg(["mean", "size"])
    white_floor = 0.5 * per_night.loc[per_night["size"] >= 3, "mean"]

    se_ok = night & np.isfinite(se_pixel)
    se_frame = pd.DataFrame({"night": night_id[se_ok], "se": se_pixel[se_ok]})
    se_night = se_frame.groupby("night")["se"].median()

    joined = pd.concat([white_floor.rename("white"), se_night.rename("se")], axis=1).dropna()
    if len(joined) < 30:
        return np.nan, len(joined)
    correlation = float(np.corrcoef(joined["white"], joined["se"])[0, 1])
    return correlation, len(joined)


def analyse_center(name, lst_pixel, se_pixel, lon, months, rng):
    """Season rows for the table + the pooled pieces the variogram panel needs."""
    night, night_id = night_axes(lst_pixel.size, lon)
    rows = []
    panel = None
    for season, season_months in SEASONS.items():
        in_season = np.isin(months, season_months)
        gamma, n_pairs = night_variogram(lst_pixel, night, in_season)
        c0, c2 = fit_nugget(gamma)
        night_se = se_pixel[night & in_season & np.isfinite(se_pixel)]
        med_se = float(np.median(night_se)) if night_se.size else np.nan
        sigma_w2 = max(c0, 0.0) if np.isfinite(c0) else np.nan
        f_hat = np.nan
        if np.isfinite(sigma_w2) and np.isfinite(med_se) and med_se > 0:
            f_hat = float(np.clip(1.0 - sigma_w2 / med_se**2, 0.0, 1.0))
        rows.append(
            {
                "center": name,
                "season": season,
                "n_pairs_lag1": n_pairs[1],
                "low_n": n_pairs[1] < MIN_PAIRS,
                "med_night_se": med_se,
                "sigma_w": float(np.sqrt(sigma_w2)) if np.isfinite(sigma_w2) else np.nan,
                "sigma_w_naive": float(np.sqrt(gamma[1])) if np.isfinite(gamma[1]) else np.nan,
                "f_hat": f_hat,
            }
        )
        if season == "all":
            panel = {"name": name, "gamma": gamma, "n_pairs": n_pairs, "c0": c0, "c2": c2, "med_se": med_se}

    residuals = night_detrended_residuals(lst_pixel, night, night_id)
    acf = residual_acf(residuals, night_id)
    acf_null = white_noise_null_acf(lst_pixel, night, night_id, rng)
    track_r, track_n = tracking_correlation(lst_pixel, se_pixel, night, night_id)
    rows[0]["valid_frac"] = float(np.isfinite(lst_pixel).mean())
    rows[0]["track_r"] = track_r
    rows[0]["track_n_nights"] = track_n
    for lag in LAGS:
        rows[0][f"acf_lag{lag}"] = acf[lag]
        rows[0][f"acf_lag{lag}_null"] = acf_null[lag]
    panel["acf"] = acf
    panel["acf_null"] = acf_null
    panel["f_hat"] = rows[0]["f_hat"]
    panel["sigma_w"] = rows[0]["sigma_w"]
    return rows, panel


def plot_variograms(panels, out_path):
    """3x3 panels: pooled night variogram, quadratic fit, nugget, SE^2 reference."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
    ks = np.array(LAGS, dtype=float)
    for ax, panel in zip(axes.ravel(), panels):
        gamma_values = [panel["gamma"][lag] for lag in LAGS]
        ax.scatter(ks, gamma_values, color=OBS_COLOR, zorder=3, label="night variogram $\\gamma(k)$")
        if np.isfinite(panel["c0"]):
            k_grid = np.linspace(0.0, LAGS[-1] + 0.3, 60)
            fit_curve = panel["c0"] + panel["c2"] * k_grid**2
            ax.plot(k_grid, fit_curve, color=FIT_COLOR, label="fit $c_0 + c_2 k^2$")
            nugget = max(panel["c0"], 0.0)
            ax.scatter([0.0], [nugget], marker="D", color=FIT_COLOR, zorder=4, label="nugget $\\sigma_w^2$")
        if np.isfinite(panel["med_se"]):
            ax.axhline(panel["med_se"] ** 2, color=SE_COLOR, linestyle="--", label="SE$^2$ (all-white level)")
        ax.set_title(panel["name"])
        summary = (
            f"$\\sigma_w$={panel['sigma_w']:.2f} K   $\\hat{{f}}$={panel['f_hat']:.2f}\nn$_1$={panel['n_pairs'][1]}"
        )
        ax.annotate(summary, xy=(0.03, 0.96), xycoords="axes fraction", va="top", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("lag k [h]")
    for ax in axes[:, 0]:
        ax.set_ylabel("$\\gamma(k)$ [K$^2$]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4)
    fig.suptitle("Within-night LST variogram vs retrieval standard_error, 2016-2024 (pooled)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_night_acf(panels, out_path):
    """3x3 panels: detrended within-night residual ACF vs the white-noise null.

    The null (open diamonds) is what the detrend machinery alone produces for
    white noise with the same cloud gaps; observed bars matching it means the
    retrieval noise carries no extra hour-to-hour persistence."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    for ax, panel in zip(axes.ravel(), panels):
        acf_values = [panel["acf"][lag] for lag in LAGS]
        null_values = [panel["acf_null"][lag] for lag in LAGS]
        ax.bar(LAGS, acf_values, width=0.6, color=OBS_COLOR, label="observed residual ACF")
        ax.scatter(
            LAGS,
            null_values,
            marker="D",
            facecolors="white",
            edgecolors=FIT_COLOR,
            zorder=3,
            label="white-noise null (same gaps + detrend)",
        )
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_title(panel["name"])
        ax.set_ylim(-0.75, 0.75)
        ax.grid(alpha=0.3, axis="y")
    for ax in axes[-1, :]:
        ax.set_xlabel("lag [h]")
        ax.set_xticks(list(LAGS))
    for ax in axes[:, 0]:
        ax.set_ylabel("autocorrelation")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2)
    fig.suptitle("Detrended within-night residual ACF vs white-noise null")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    matplotlib.use("Agg")  # headless runs only; importing this module must not switch the backend
    plt.rcParams.update(RC)
    pixels = center_pixels()
    names = list(pixels)
    store_rows = np.array([pixels[name][0] for name in names])
    print(f"loading {len(names)} centre pixels from {SEVIRI_STORE} (days {DAY_RANGE[0]}..{DAY_RANGE[1]})...")
    lst, se = load_center_columns(store_rows)
    n_hours = lst.shape[1]
    store_time = pd.date_range("2016-01-01", periods=n_hours, freq="h")
    months = store_time.month.values

    rng = np.random.default_rng(0)
    table_rows = []
    panels = []
    for i, name in enumerate(names):
        lon = pixels[name][1]
        rows, panel = analyse_center(name, lst[i], se[i], lon, months, rng)
        table_rows.extend(rows)
        panels.append(panel)
        all_row = rows[0]
        print(
            f"  {name:14s} valid={all_row['valid_frac']:.2f} n1={all_row['n_pairs_lag1']:6d} "
            f"medSE={all_row['med_night_se']:.2f}K sigma_w={all_row['sigma_w']:.2f}K "
            f"f_hat={all_row['f_hat']:.2f} acf1={all_row['acf_lag1']:.2f} track_r={all_row['track_r']:.2f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(table_rows)
    table.to_csv(OUT_DIR / "table.csv", index=False)
    plot_variograms(panels, OUT_DIR / "variograms.png")
    plot_night_acf(panels, OUT_DIR / "night_acf.png")
    print(f"saved {OUT_DIR}/table.csv, variograms.png, night_acf.png")


if __name__ == "__main__":
    pixels = center_pixels()
    names = list(pixels)
    store_rows = np.array([pixels[name][0] for name in names])
    print(f"loading {len(names)} centre pixels from {SEVIRI_STORE} (days {DAY_RANGE[0]}..{DAY_RANGE[1]})...")
