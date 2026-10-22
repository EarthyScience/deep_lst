import numpy as np
import pandas as pd
import xarray as xr

from .cubes import obs_cube
from .masks import KG_GROUPS, MASKS

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
MEDIUM_RANGE = 24  # last N leads: the window the reported site skill is scored on
MIN_OBSERVED_HOURS = 12  # a drawn forecast day needs at least half its hours observed


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


def select_regime_pixels(pixel_daily, day, quantile=0.75, window="all", baseline="clim_smooth"):
    """Per KG regime, the pixel at `quantile` of the day-`day` skill against the
    baseline: a good example by construction (above the class median), not the
    most flattering one. Returns regime window -> pixel index."""
    skill = pixel_daily["rmse_ss"].sel(window=window, baseline=baseline, day=day).values
    kg_class = pixel_daily["kg_class"].values
    pixels = {}
    for regime, class_ids in KG_GROUPS.items():
        members = np.where(np.isin(kg_class, class_ids) & np.isfinite(skill))[0]
        ranked = members[np.argsort(skill[members])]
        position = int(quantile * (len(ranked) - 1))
        pixels[regime] = int(ranked[position])
    return pixels


def select_example_pixel(pixel_daily, day, quantile=0.9, window="all", baseline="clim_smooth"):
    """The pixel at `quantile` of the day-`day` skill against the baseline over the
    whole eval set: a very good example by construction, not the most flattering one
    (the maximum is where the artefacts live). Returns (pixel index, its skill)."""
    skill = pixel_daily["rmse_ss"].sel(window=window, baseline=baseline, day=day).values
    members = np.where(np.isfinite(skill))[0]
    ranked = members[np.argsort(skill[members])]
    position = int(quantile * (len(ranked) - 1))
    pixel = int(ranked[position])
    return pixel, float(skill[pixel])


def select_clear_lookback_init(coverage, pred, observed, day_positions, eligible=None):
    """The init with the fullest lookback window at one pixel, among the inits whose
    drawn forecast days are completely forecast and each observed for at least
    MIN_OBSERVED_HOURS. `eligible` (init,) narrows the pool further, e.g. to one init
    hour so every drawn day spans the same valid hours. Ties in coverage (common
    under clear skies) go to the init with more observed hours.

    `coverage` is the (init,) lookback clear fraction, `pred` and `observed` the
    (init, lead) forecast and obs of the pixel, `day_positions` the lead positions of
    each drawn day."""
    usable = np.isfinite(coverage)
    if eligible is not None:
        usable &= eligible
    n_observed = np.zeros(len(coverage), dtype=int)
    for positions in day_positions:
        forecast_complete = np.isfinite(pred[:, positions]).all(axis=1)
        n_day = np.isfinite(observed[:, positions]).sum(axis=1)
        usable &= forecast_complete & (n_day >= MIN_OBSERVED_HOURS)
        n_observed += n_day
    pool = np.where(usable)[0]
    if len(pool) == 0:
        raise ValueError("no init with a complete forecast and enough observed hours in the drawn days")
    order = np.lexsort((n_observed[pool], coverage[pool]))  # coverage first, then observed hours
    index = pool[order[-1]]
    return int(index)


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


def anomaly_moments(initilised, anomaly_forecast, anomaly_obs, windows, masks=None, dims=("init", "pixel")):
    """Sufficient statistics of the (forecast, obs) anomaly pairs summed over `dims`:
    count and the sums of f, o, f^2, o^2 and f*o over the jointly valid samples, per
    (window, lead) for the domain-wide scores (init and pixel summed out) or per
    (window, lead, pixel) for the per-pixel scores (init alone). Any anomaly
    correlation, variance or mse pooled over any set of leads follows from these --
    unlike a per-lead correlation, which cannot be re-pooled.

    The cubes stay float32 (the dense evals are 16k pixels); only the accumulation
    is float64."""
    dims = list(dims)
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
                "n": window_valid.sum(dims),
                "sum_f": f.sum(dims, dtype="float64"),
                "sum_o": o.sum(dims, dtype="float64"),
                "sum_ff": (f * f).sum(dims, dtype="float64"),
                "sum_oo": (o * o).sum(dims, dtype="float64"),
                "sum_fo": (f * o).sum(dims, dtype="float64"),
            }
        )
        per_window.append(moments)
    return xr.concat(per_window, dim="window").assign_coords(window=windows)


def per_init_counts(initilised, obs, windows, masks=None):
    """Jointly valid samples per (window, init, lead) -- weights for pooling init out of by_init."""
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


OBS_RANGE_QUANTILES = [0.05, 0.95]  # robust span of a site's observed LST: the nRMSE normaliser


def per_pixel_obs_moments(initilised, obs, windows, masks=None):
    """Obs statistics of the model's samples per pixel: the jointly valid count per
    (window, lead, pixel) -- the weight for per-pixel day pooling -- with the sums of o
    and o^2 over those samples (the NSE denominator, exact under any lead pooling), and
    per (window, pixel) the 5-95 % range of the obs over every lead of the window: the
    span the normalised RMSE divides by, one fixed value per site and window."""
    observed = obs_cube(obs, initilised)
    valid = initilised["LST"].notnull() & observed.notnull()
    masks = MASKS if masks is None else masks
    per_window = []
    for window in windows:
        mask_fn = masks[window]
        window_valid = valid if mask_fn is None else valid & mask_fn(initilised)
        o = observed.where(window_valid)
        quantiles = o.quantile(OBS_RANGE_QUANTILES, dim=["init", "lead"])
        # The difference of the two slices carries no quantile coord any more.
        obs_range = quantiles.sel(quantile=OBS_RANGE_QUANTILES[1]) - quantiles.sel(quantile=OBS_RANGE_QUANTILES[0])
        moments = xr.Dataset(
            {
                "n": window_valid.sum("init"),
                "sum_o": o.sum("init", dtype="float64"),
                "sum_oo": (o * o).sum("init", dtype="float64"),
                "obs_range": obs_range,
            }
        )
        per_window.append(moments)
    return xr.concat(per_window, dim="window").assign_coords(window=windows)


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


# Sums of the diurnal verification per (solar hour, forecast day, pixel) over the samples
# where model, obs and climatology are all valid: the count, the model's squared and
# signed errors, the obs sums (NSE denominator) and the anomaly sums (ACC, named as in
# by_pixel.nc).
MODEL_SUMS = [
    "n",
    "se_model",
    "e_model",
    "sum_o",
    "sum_oo",
    "anom_sum_f",
    "anom_sum_o",
    "anom_sum_ff",
    "anom_sum_oo",
    "anom_sum_fo",
]
# The same for every reference forecast on the model's samples where the reference is
# itself valid (persistence has gaps), stored as <stat>_<reference>.
REFERENCE_SUMS = ["n", "se", "e", "sum_o", "sum_oo"]
# Sums of the diurnal amplitude errors per (forecast day, pixel): the model's, and per
# reference (<stat>_<reference>) on the (init, day) pairs with enough hours for it.
AMPLITUDE_MODEL_SUMS = ["n", "e_model", "se_model", "sum_obs"]
AMPLITUDE_REFERENCE_SUMS = ["n", "e", "se"]


def local_solar_hour(init_hour, lead, lon):
    """Integer local solar hour (0-23) of the (init, pixel) samples at one lead: the hour
    masks.solar_hour computes, floored to the hour bin."""
    valid_hour = (init_hour + lead) % 24
    solar = (valid_hour[:, None] + lon[None, :] / 15.0) % 24
    return np.floor(solar).astype(int)


def _bin_hourly(sums, weights, bins, day, n_px):
    """Add the per-sample `weights` (None = count) binned by `bins` = hour * n_px + pixel
    into sums[name][:, day]."""
    for name, values in weights.items():
        binned = np.bincount(bins, weights=values, minlength=24 * n_px)
        sums[name][:, day] += binned.reshape(24, n_px)


def diurnal_sums(initilised, observed, references):
    """MODEL_SUMS per (local solar hour, forecast day, pixel) over the samples where model,
    obs and the climatology are all valid, and REFERENCE_SUMS of every reference in
    `references` (name -> (init, lead, pixel) cube; 'clim' is required, it defines the
    anomalies) on those samples where the reference is itself valid. Lead by lead, so
    nothing larger than one (init, pixel) slab is allocated besides the cubes."""
    forecast = initilised["LST"].values
    n_init, n_lead, n_px = forecast.shape
    leads = initilised["lead"].values
    n_days = int((leads[-1] - 1) // 24 + 1)
    init_hour = initilised["init"].dt.hour.values
    lon = initilised["lon"].values
    pixel = np.broadcast_to(np.arange(n_px)[None, :], (n_init, n_px))
    sums = {}
    for name in MODEL_SUMS:
        sums[name] = np.zeros((24, n_days, n_px))
    for reference in references:
        for stat in REFERENCE_SUMS:
            sums[f"{stat}_{reference}"] = np.zeros((24, n_days, n_px))
    for position, lead in enumerate(leads):
        day = int((lead - 1) // 24)
        bins = local_solar_hour(init_hour, int(lead), lon) * n_px + pixel
        f = forecast[:, position].astype(np.float64)
        o = observed[:, position].astype(np.float64)
        c = references["clim"][:, position].astype(np.float64)
        valid = np.isfinite(f) & np.isfinite(o) & np.isfinite(c)
        error = (f - o)[valid]
        anomaly_f = (f - c)[valid]
        anomaly_o = (o - c)[valid]
        weights = {
            "n": None,
            "se_model": error**2,
            "e_model": error,
            "sum_o": o[valid],
            "sum_oo": (o * o)[valid],
            "anom_sum_f": anomaly_f,
            "anom_sum_o": anomaly_o,
            "anom_sum_ff": anomaly_f * anomaly_f,
            "anom_sum_oo": anomaly_o * anomaly_o,
            "anom_sum_fo": anomaly_f * anomaly_o,
        }
        _bin_hourly(sums, weights, bins[valid], day, n_px)
        for reference, cube in references.items():
            r = cube[:, position].astype(np.float64)
            valid_r = valid & np.isfinite(r)
            error_r = (r - o)[valid_r]
            weights_r = {
                f"n_{reference}": None,
                f"se_{reference}": error_r**2,
                f"e_{reference}": error_r,
                f"sum_o_{reference}": o[valid_r],
                f"sum_oo_{reference}": (o * o)[valid_r],
            }
            _bin_hourly(sums, weights_r, bins[valid_r], day, n_px)
    coords = {"hour": np.arange(24), "day": np.arange(1, n_days + 1), "pixel": initilised["pixel"].values}
    data = {}
    for name, values in sums.items():
        data[name] = (("hour", "day", "pixel"), values)
    return xr.Dataset(data, coords=coords)


def _amplitude(values, valid):
    """Max - min over the valid hours (axis 1) of (init, hour, pixel) values. Invalid hours
    sit at -inf for the max and +inf for the min, so they never win; the caller drops the
    (init, pixel) days without enough valid hours."""
    high = np.where(valid, values, -np.inf).max(axis=1)
    low = np.where(valid, values, np.inf).min(axis=1)
    return (high - low).astype(np.float64)


def amplitude_sums(initilised, observed, references, min_hours=MIN_OBSERVED_HOURS):
    """AMPLITUDE_MODEL_SUMS per (forecast day, pixel): for every (init, pixel, day) with at
    least min_hours hours valid for model, obs and climatology, the diurnal amplitude
    (max - min over those hours) of the model and of the obs; summed over init the count,
    the signed and squared amplitude errors and the obs amplitude. AMPLITUDE_REFERENCE_SUMS
    likewise for every reference in `references`, over the hours where the reference is
    valid as well (the obs amplitude taken over the same hours). Days with fewer hours
    are skipped, their amplitude being truncated."""
    forecast = initilised["LST"].values
    n_init, n_lead, n_px = forecast.shape
    n_days = n_lead // 24
    pixel = np.broadcast_to(np.arange(n_px)[None, :], (n_init, n_px))
    sums = {}
    for name in AMPLITUDE_MODEL_SUMS:
        sums[name] = np.zeros((n_days, n_px))
    for reference in references:
        for stat in AMPLITUDE_REFERENCE_SUMS:
            sums[f"{stat}_{reference}"] = np.zeros((n_days, n_px))
    for day in range(n_days):
        hours = slice(24 * day, 24 * day + 24)
        f = forecast[:, hours]
        o = observed[:, hours]
        valid = np.isfinite(f) & np.isfinite(o) & np.isfinite(references["clim"][:, hours])
        enough = valid.sum(axis=1) >= min_hours
        amplitude_o = _amplitude(o, valid)
        error = (_amplitude(f, valid) - amplitude_o)[enough]
        weights = {"n": None, "e_model": error, "se_model": error**2, "sum_obs": amplitude_o[enough]}
        for name, values in weights.items():
            sums[name][day] += np.bincount(pixel[enough], weights=values, minlength=n_px)
        for reference, cube in references.items():
            valid_r = valid & np.isfinite(cube[:, hours])
            enough_r = valid_r.sum(axis=1) >= min_hours
            error_r = (_amplitude(cube[:, hours], valid_r) - _amplitude(o, valid_r))[enough_r]
            weights_r = {f"n_{reference}": None, f"e_{reference}": error_r, f"se_{reference}": error_r**2}
            for name, values in weights_r.items():
                sums[name][day] += np.bincount(pixel[enough_r], weights=values, minlength=n_px)
    coords = {"day": np.arange(1, n_days + 1), "pixel": initilised["pixel"].values}
    data = {}
    for name, values in sums.items():
        data[name] = (("day", "pixel"), values)
    return xr.Dataset(data, coords=coords)


def amplitude_cells(forecast, observed, references, peak, trough, min_hours=MIN_OBSERVED_HOURS):
    """Diurnal amplitude error per (init, forecast day, pixel), paired on the jointly observed
    hours. A forecast day counts when at least min_hours of its 24 hours are valid for model
    and obs and, among those, one lies in the peak window and one in the trough window
    (`peak`, `trough`: boolean (init, lead, pixel) masks of local solar hours). Its error is the
    amplitude (max - min over exactly those hours) of the model minus that of the obs, so the
    truncation by cloud gaps hits both alike. Returns a Dataset in the by_cell layout (init,
    day, pixel) with the variables of the other windows -- mse_model (squared error),
    bias_model (signed error), n (joint hours), mse_<reference> (squared error of each
    reference over the hours where it is valid as well) -- plus amplitude_obs (the paired obs
    amplitude), amplitude_model_full (the model's amplitude over all 24 hours; its difference to
    the paired amplitude is the truncation) and the two coverage flags. Days failing the rule
    are NaN with n = 0. The 24-lead forecast day is a solar day for 00 UTC inits only; later
    inits straddle two diurnal cycles."""
    model = np.asarray(forecast)
    obs = np.asarray(observed)
    peak = np.asarray(peak)
    trough = np.asarray(trough)
    n_init, n_lead, n_px = model.shape
    n_days = n_lead // 24
    shape = (n_init, n_days, n_px)
    out = {
        name: np.full(shape, np.nan) for name in ("mse_model", "bias_model", "amplitude_obs", "amplitude_model_full")
    }
    for reference in references:
        out[f"mse_{reference}"] = np.full(shape, np.nan)
    out["n"] = np.zeros(shape, dtype=np.int32)
    out["peak_observed"] = np.zeros(shape, dtype=bool)
    out["trough_observed"] = np.zeros(shape, dtype=bool)

    def covered(valid, hours):
        """Rule per (init, pixel): enough joint hours, one in the peak and one in the trough window."""
        enough = valid.sum(axis=1) >= min_hours
        peak_seen = (valid & peak[:, hours]).any(axis=1)
        trough_seen = (valid & trough[:, hours]).any(axis=1)
        return enough & peak_seen & trough_seen, peak_seen, trough_seen

    for day in range(n_days):
        hours = slice(24 * day, 24 * day + 24)
        model_day = model[:, hours]
        obs_day = obs[:, hours]
        valid = np.isfinite(model_day) & np.isfinite(obs_day)
        keep, peak_seen, trough_seen = covered(valid, hours)
        amplitude_obs = _amplitude(obs_day, valid)
        error = _amplitude(model_day, valid) - amplitude_obs
        complete = np.isfinite(model_day).all(axis=1)
        with np.errstate(invalid="ignore"):
            full = np.where(complete, model_day.max(axis=1) - model_day.min(axis=1), np.nan)
        out["bias_model"][:, day] = np.where(keep, error, np.nan)
        out["mse_model"][:, day] = np.where(keep, error**2, np.nan)
        out["amplitude_obs"][:, day] = np.where(keep, amplitude_obs, np.nan)
        out["amplitude_model_full"][:, day] = np.where(keep, full, np.nan)
        out["n"][:, day] = np.where(keep, valid.sum(axis=1), 0)
        out["peak_observed"][:, day] = peak_seen
        out["trough_observed"][:, day] = trough_seen
        for reference, cube in references.items():
            reference_day = np.asarray(cube)[:, hours]
            valid_reference = valid & np.isfinite(reference_day)
            keep_reference, _, _ = covered(valid_reference, hours)
            error_reference = _amplitude(reference_day, valid_reference) - _amplitude(obs_day, valid_reference)
            out[f"mse_{reference}"][:, day] = np.where(keep_reference, error_reference**2, np.nan)

    coords = {"init": forecast["init"].values, "day": np.arange(1, n_days + 1), "pixel": forecast["pixel"].values}
    data = {name: (("init", "day", "pixel"), values) for name, values in out.items()}
    return xr.Dataset(data, coords=coords)
