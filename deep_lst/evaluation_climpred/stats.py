import numpy as np
import pandas as pd
import xarray as xr

from .cubes import obs_cube
from .masks import MASKS

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
MEDIUM_RANGE = 24  # last N leads: the window the reported site skill is scored on


def select_init(pred, observed, climatology, init_times):
    """Per season, the clearest init whose medium-range skill sits inside the IQR.

    Scored on the last MEDIUM_RANGE leads, not the whole horizon: the early leads are
    easy for everything and would flatter every init. Clarity is measured over the
    full window instead, because that is what the drawn trajectory looks like.

    Selecting on clarity alone is not neutral: cloudy spells are convective, so they
    are both harder to observe and harder to forecast, and the clearest init is
    therefore also the least typical one. Restricting to the interquartile range
    first picks a representative forecast rather than the most flattering one --
    the rule is outcome-aware only in targeting typicality."""
    valid = np.isfinite(pred) & np.isfinite(observed) & np.isfinite(climatology)
    clear = valid.mean(axis=1)

    scored = valid[:, -MEDIUM_RANGE:]
    with np.errstate(invalid="ignore"):
        rmse_model = np.sqrt(
            np.nanmean(np.where(scored, (pred[:, -MEDIUM_RANGE:] - observed[:, -MEDIUM_RANGE:]) ** 2, np.nan), axis=1)
        )
        rmse_clim = np.sqrt(
            np.nanmean(
                np.where(scored, (climatology[:, -MEDIUM_RANGE:] - observed[:, -MEDIUM_RANGE:]) ** 2, np.nan), axis=1
            )
        )
    skill = 1.0 - rmse_model / rmse_clim

    month = pd.DatetimeIndex(init_times).month
    chosen = {}
    for season, months in SEASONS.items():
        # Inits whose day-10 window is fully cloudy have no skill; they must not
        # count in the rank, or the reported percentile is biased low.
        pool = np.where(np.isin(month, months))[0]
        pool = pool[np.isfinite(skill[pool])]
        low, high = np.quantile(skill[pool], [0.25, 0.75])
        band = pool[(skill[pool] >= low) & (skill[pool] <= high)]
        index = band[np.argmax(clear[band])]
        chosen[season] = {
            "index": index,
            "clear": clear[index],
            "skill": skill[index],
            "rmse_model": rmse_model[index],
            "rmse_clim": rmse_clim[index],
            "scored": int(scored[index].sum()),
            "percentile": float(np.mean(skill[pool] < skill[index])),
        }
    return chosen


def anomaly_obs_variance(initilised, anomaly_obs, windows, masks=None):
    """Variance of the obs anomalies over the jointly valid samples per (window, lead).

    Denominator of the anomaly R2 = 1 - mse / var. The anomaly error equals the raw
    error, so the numerator is the already-verified model mse -- no extra verify pass."""
    anomaly = obs_cube(anomaly_obs, initilised)
    valid = initilised["LST"].notnull() & anomaly.notnull()
    variances = []
    masks = MASKS if masks is None else masks
    for window in windows:
        mask_fn = masks[window]
        window_valid = valid if mask_fn is None else valid & mask_fn(initilised)
        x = anomaly.where(window_valid)
        n = window_valid.sum(["init", "pixel"])
        mean = x.sum(["init", "pixel"], skipna=True) / n
        mean_sq = (x**2).sum(["init", "pixel"], skipna=True) / n
        variances.append(mean_sq - mean**2)
    return xr.concat(variances, dim="window").assign_coords(window=windows)


def anomaly_moments(initilised, anomaly_forecast, anomaly_obs, windows, masks=None):
    """Sufficient statistics of the (forecast, obs) anomaly pairs per (window, lead):
    count and the sums of f, o, f^2, o^2 and f*o over the jointly valid samples.
    Any anomaly correlation, variance or mse pooled over any set of leads follows
    from these -- unlike a per-lead correlation, which cannot be re-pooled.

    The cubes stay float32 (the dense evals are 16k pixels); only the accumulation
    is float64."""
    obs_anomaly = obs_cube(anomaly_obs, initilised)
    valid = anomaly_forecast.notnull() & obs_anomaly.notnull()
    masks = MASKS if masks is None else masks
    per_window = []
    for window in windows:
        mask_fn = masks[window]
        window_valid = valid if mask_fn is None else valid & mask_fn(initilised)
        f = anomaly_forecast.where(window_valid)
        o = obs_anomaly.where(window_valid)
        moments = xr.Dataset(
            {
                "n": window_valid.sum(["init", "pixel"]),
                "sum_f": f.sum(["init", "pixel"], dtype="float64"),
                "sum_o": o.sum(["init", "pixel"], dtype="float64"),
                "sum_ff": (f * f).sum(["init", "pixel"], dtype="float64"),
                "sum_oo": (o * o).sum(["init", "pixel"], dtype="float64"),
                "sum_fo": (f * o).sum(["init", "pixel"], dtype="float64"),
            }
        )
        per_window.append(moments)
    return xr.concat(per_window, dim="window").assign_coords(window=windows)


def per_init_counts(initilised, obs, windows, masks=None):
    """Jointly valid samples per (window, init, lead) -- weights for the paired bootstrap."""
    valid = initilised["LST"].notnull() & obs_cube(obs, initilised).notnull()
    counts = []
    masks = MASKS if masks is None else masks
    for window in windows:
        mask_fn = masks[window]
        window_valid = valid if mask_fn is None else valid & mask_fn(initilised)
        counts.append(window_valid.sum("pixel"))
    return xr.concat(counts, dim="window").assign_coords(window=windows)


def window_counts(initilised, obs, windows, masks=None):
    """Jointly valid samples per (window, lead) -- the weights for day-bucket pooling."""
    return per_init_counts(initilised, obs, windows, masks).sum("init")


def per_pixel_counts(initilised, obs, windows, masks=None):
    """Jointly valid samples per (window, lead, pixel) -- weights for per-pixel day pooling."""
    valid = initilised["LST"].notnull() & obs_cube(obs, initilised).notnull()
    counts = []
    masks = MASKS if masks is None else masks
    for window in windows:
        mask_fn = masks[window]
        window_valid = valid if mask_fn is None else valid & mask_fn(initilised)
        counts.append(window_valid.sum("init"))
    return xr.concat(counts, dim="window").assign_coords(window=windows)


def bootstrap_skill_score(score_model, score_baseline, counts, iterations=500, seed=0, block_days=1):
    """Paired block bootstrap: draws whole init-days (blocks of 4 6-hourly inits),
    preserving the serial correlation of errors within each block.
    Assumes inits are chronological with exactly 4 per day (1460 = 365 x 4).

    Returns (hourly, daily) RMSE-SS quantiles. The daily buckets are pooled INSIDE
    each draw before the sqrt/ratio, so their intervals inherit all correlations
    (across leads and between model and baseline) without propagation formulas."""
    rng = np.random.default_rng(seed)
    weighted_model = score_model.fillna(0.0) * counts
    weighted_baseline = score_baseline.fillna(0.0) * counts
    day = xr.DataArray((counts["lead"].values - 1) // 24 + 1, dims="lead", name="day")
    inits_per_block = 4 * block_days
    n_blocks = counts.sizes["init"] // inits_per_block
    hourly_samples, daily_samples = [], []
    for _ in range(iterations):
        blocks = rng.integers(0, n_blocks, n_blocks)
        idx = (blocks[:, None] * inits_per_block + np.arange(inits_per_block)).ravel()
        pooled_n = counts.isel(init=idx).sum("init")
        pooled_m = weighted_model.isel(init=idx).sum("init")
        pooled_b = weighted_baseline.isel(init=idx).sum("init")
        hourly_samples.append(1.0 - np.sqrt(pooled_m / pooled_n) / np.sqrt(pooled_b / pooled_n))
        day_n = pooled_n.groupby(day).sum("lead")
        day_m = pooled_m.groupby(day).sum("lead")
        day_b = pooled_b.groupby(day).sum("lead")
        daily_samples.append(1.0 - np.sqrt(day_m / day_n) / np.sqrt(day_b / day_n))
    quantiles = [0.025, 0.5, 0.975]
    hourly = xr.concat(hourly_samples, dim="iteration").quantile(quantiles, dim="iteration")
    daily = xr.concat(daily_samples, dim="iteration").quantile(quantiles, dim="iteration")
    return hourly, daily


def forecast_limit_map(rmse_ss_px_daily):
    """First forecast day with skill < 0 per pixel (fc_limits convention: strict < 0,
    NaN never a crossing). Never-crossing pixels are censored one day BEYOND the
    horizon, so a crossing on the horizon day stays distinguishable from no crossing
    at all; pixels with no valid skill at all become NaN."""
    below = rmse_ss_px_daily < 0
    crossed = below.any("day")
    first_day = rmse_ss_px_daily["day"].isel(day=below.argmax("day"))
    horizon = int(rmse_ss_px_daily["day"].values[-1])
    limit_days = xr.where(crossed, first_day, horizon + 1)
    has_skill = rmse_ss_px_daily.notnull().any("day")
    return limit_days.where(has_skill).rename("forecast_limit_days")
