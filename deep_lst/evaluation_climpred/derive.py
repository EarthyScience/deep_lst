import numpy as np
import xarray as xr

from .common import verification_dir
from .masks import KG_GROUPS, MONTHLY_WINDOWS
from .stats import forecast_limit_map


def pooled_mean(values, counts, dim, group=None):
    """Count-weighted mean of per-cell averages over `dim`, optionally within `group`.

    Cells carry different sample counts, so averaging the averages is wrong: undo the
    averaging (x count), sum, divide by the summed counts. For an RMSE, pool in the
    mse domain and take the sqrt at the call site -- never before."""
    weighted = values.fillna(0.0) * counts
    if group is None:
        return weighted.sum(dim) / counts.sum(dim)
    return weighted.groupby(group).sum(dim) / counts.groupby(group).sum(dim)


MOMENT_NAMES = ["n", "sum_f", "sum_o", "sum_ff", "sum_oo", "sum_fo"]


def anomaly_scores(moments, dim=None, group=None):
    """Centered ACC and anomaly R2 from summed moments. `dim`/`group` pool the sums
    first (leads into days), so the scores are exact for the pooled sample -- mse,
    variance and covariance all come from the same five sums."""
    if dim is not None:
        moments = moments.sum(dim) if group is None else moments.groupby(group).sum(dim)
    n = moments["n"]
    mean_f = moments["sum_f"] / n
    mean_o = moments["sum_o"] / n
    var_f = moments["sum_ff"] / n - mean_f**2
    var_o = moments["sum_oo"] / n - mean_o**2
    cov = moments["sum_fo"] / n - mean_f * mean_o
    acc = cov / np.sqrt(var_f * var_o)
    mse = (moments["sum_ff"] - 2.0 * moments["sum_fo"] + moments["sum_oo"]) / n
    r2 = 1.0 - mse / var_o
    return acc, r2


# by_pixel.nc stores the per-pixel anomaly sums under these names (verif_pixel).
ANOMALY_PIXEL_NAMES = {f"anom_{name}": name for name in MOMENT_NAMES}
PIXEL_QUANTILES = [0.025, 0.5, 0.975]  # the per-pixel spread of a score: lower, median, upper


def day_labels(lead):
    """Forecast day of each lead hour (1-24 h -> day 1), the groupby label on the lead dim."""
    return xr.DataArray((lead.values - 1) // 24 + 1, dims="lead", name="day")


def nrmse_normaliser(obs_range):
    """Per-(window, pixel) span the normalised RMSE divides by: the window's own 5-95 %
    obs range, except for the single-month windows, which share the all-hours annual
    span so the month-to-month change of the map is the error's, not the normaliser's."""
    normaliser = obs_range.copy()
    for window in MONTHLY_WINDOWS:
        if window in obs_range["window"].values:
            normaliser.loc[{"window": window}] = obs_range.sel(window="all")
    return normaliser


def pixel_day_scores(by_pixel):
    """Per-pixel scores by forecast day from a by_pixel verification file: the 24 leads
    of a day pooled count-weighted in the mse domain (sqrt after) and the bias linearly;
    where the verif saved the obs and anomaly sums, also the Nash-Sutcliffe efficiency
    of the model and the baselines (obs variance of the same pooled samples), the
    normalised RMSE (percent of the site's obs span), the centered ACC and the anomaly
    R2 -- all exact for the pooled day, like the domain-wide anomaly scores."""
    day = day_labels(by_pixel["lead"])
    counts = by_pixel["n"]
    bias_model = pooled_mean(by_pixel["bias_model"], counts, dim="lead", group=day)
    mse_model = pooled_mean(by_pixel["mse_model"], counts, dim="lead", group=day)
    mse_baseline = pooled_mean(by_pixel["mse_baseline"], counts, dim="lead", group=day)
    rmse_model = np.sqrt(mse_model)
    rmse_baseline = np.sqrt(mse_baseline)
    fields = {
        "bias_model": bias_model,
        "rmse_model": rmse_model,
        "rmse_baseline": rmse_baseline,
        "rmse_ss": 1.0 - rmse_model / rmse_baseline,
    }
    if "bias_baseline" in by_pixel:
        fields["bias_baseline"] = pooled_mean(by_pixel["bias_baseline"], counts, dim="lead", group=day)
    if "sum_oo" in by_pixel:
        day_counts = counts.groupby(day).sum("lead")
        mean_o = by_pixel["sum_o"].groupby(day).sum("lead") / day_counts
        mean_oo = by_pixel["sum_oo"].groupby(day).sum("lead") / day_counts
        obs_variance = mean_oo - mean_o**2
        fields["nse_model"] = 1.0 - mse_model / obs_variance
        fields["nse_baseline"] = 1.0 - mse_baseline / obs_variance
        fields["nrmse_model"] = 100.0 * rmse_model / nrmse_normaliser(by_pixel["obs_range"])
    if all(name in by_pixel for name in ANOMALY_PIXEL_NAMES):
        moments = by_pixel[list(ANOMALY_PIXEL_NAMES)].rename(ANOMALY_PIXEL_NAMES)
        acc, r2 = anomaly_scores(moments, dim="lead", group=day)
        fields["acc_model"] = acc
        fields["r2_model"] = r2
    return xr.Dataset(fields)


MIN_CELL_SAMPLES = 30  # a per-pixel efficiency at one solar hour and day needs this many samples
DIURNAL_REFERENCES = ("clim", "persistence", "era5land")  # scored where the verif saved them


def _reference_sum(cells, stat, reference):
    """The reference's own sum (<stat>_<reference>) where the verif saved one, else the
    model's <stat> (older files scored the climatology on the model's sample)."""
    name = f"{stat}_{reference}"
    return cells[name] if name in cells else cells[stat]


def diurnal_scores(diurnal, groups):
    """Diurnal products of verif/diurnal.nc per pixel group ('all' or a masks.KG_GROUPS
    name, by the kg_class coord): per (hour, day) the RMSE and bias of the model and of
    every reference the verif holds (DIURNAL_REFERENCES) pooled over the group's pixels,
    and the per-pixel NSE (model and references) and ACC (model) as the median over the
    group's pixels, cells with fewer than MIN_CELL_SAMPLES samples left out; per day the
    amplitude RMSE and bias of all of them pooled over the pixels, with the mean observed
    amplitude. Returns a (group, ...) set."""
    references = [name for name in DIURNAL_REFERENCES if f"se_{name}" in diurnal]
    per_group = []
    for group in groups:
        if group == "all":
            members = np.ones(diurnal.sizes["pixel"], dtype=bool)
        else:
            members = diurnal["kg_class"].isin(list(KG_GROUPS[group])).values
        cells = diurnal.isel(pixel=np.where(members)[0])
        n = cells["n"].sum("pixel")
        scores = {
            "rmse_model": np.sqrt(cells["se_model"].sum("pixel") / n),
            "bias_model": cells["e_model"].sum("pixel") / n,
            "n": n,
        }
        enough = cells["n"] >= MIN_CELL_SAMPLES
        mean_o = cells["sum_o"] / cells["n"]
        obs_variance = cells["sum_oo"] / cells["n"] - mean_o**2
        nse_model = 1.0 - (cells["se_model"] / cells["n"]) / obs_variance
        moments = xr.Dataset({name: cells[f"anom_{name}"] for name in MOMENT_NAMES if name != "n"})
        moments["n"] = cells["n"]
        acc_model, _ = anomaly_scores(moments)
        scores["nse_model"] = nse_model.where(enough).median("pixel")
        scores["acc_model"] = acc_model.where(enough).median("pixel")
        for reference in references:
            n_reference = _reference_sum(cells, "n", reference)
            scores[f"rmse_{reference}"] = np.sqrt(cells[f"se_{reference}"].sum("pixel") / n_reference.sum("pixel"))
            scores[f"bias_{reference}"] = cells[f"e_{reference}"].sum("pixel") / n_reference.sum("pixel")
            mean_o = _reference_sum(cells, "sum_o", reference) / n_reference
            variance = _reference_sum(cells, "sum_oo", reference) / n_reference - mean_o**2
            nse = 1.0 - (cells[f"se_{reference}"] / n_reference) / variance
            scores[f"nse_{reference}"] = nse.where(n_reference >= MIN_CELL_SAMPLES).median("pixel")

        n_amplitude = cells["amplitude_n"].sum("pixel")
        scores["amplitude_rmse_model"] = np.sqrt(cells["amplitude_se_model"].sum("pixel") / n_amplitude)
        scores["amplitude_bias_model"] = cells["amplitude_e_model"].sum("pixel") / n_amplitude
        scores["amplitude_obs"] = cells["amplitude_sum_obs"].sum("pixel") / n_amplitude
        scores["amplitude_n"] = n_amplitude
        for reference in references:
            n_reference = _reference_sum(cells, "amplitude_n", reference).sum("pixel")
            squared = cells[f"amplitude_se_{reference}"].sum("pixel")
            scores[f"amplitude_rmse_{reference}"] = np.sqrt(squared / n_reference)
            scores[f"amplitude_bias_{reference}"] = cells[f"amplitude_e_{reference}"].sum("pixel") / n_reference
        per_group.append(xr.Dataset(scores))
    return xr.concat(per_group, dim="group").assign_coords(group=groups)


def derive_verif_products(args):
    verif_dir = verification_dir(args, "verif")
    scores_dir = verification_dir(args, "scores")

    by_init = xr.open_dataset(verif_dir / "by_init.nc", decode_timedelta=False)
    by_pixel = xr.open_dataset(verif_dir / "by_pixel.nc", decode_timedelta=False)
    anom = xr.open_dataset(verif_dir / "anom.nc", decode_timedelta=False)

    # Scores by lead: pool init out of by_init, pixel out of by_pixel.
    # -> skill.nc (lead, window[, baseline][, quantile])

    init_counts = by_init["n"].sum("init")
    rmse_model = np.sqrt(pooled_mean(by_init["mse_model"], counts=by_init["n"], dim="init"))
    rmse_baseline = np.sqrt(pooled_mean(by_init["mse_baseline"], counts=by_init["n"], dim="init"))

    rmse_model_px = np.sqrt(by_pixel["mse_model"])
    rmse_baseline_px = np.sqrt(by_pixel["mse_baseline"])
    rmse_ss_px = 1.0 - rmse_model_px / rmse_baseline_px
    dispersion = rmse_ss_px.quantile([0.025, 0.5, 0.975], dim="pixel")

    out = xr.Dataset(
        {
            "rmse_model": rmse_model,
            "rmse_baseline": rmse_baseline,
            "rmse_ss": 1.0 - rmse_model / rmse_baseline,
            "acc_centered": anom["acc_centered"],
            "r2_anomaly": 1.0 - rmse_model**2 / anom["anomaly_obs_variance"],
            "rmse_ss_dispersion": dispersion,
            "rmse_model_dispersion": rmse_model_px.quantile([0.025, 0.5, 0.975], dim="pixel"),
            "rmse_baseline_dispersion": rmse_baseline_px.quantile([0.025, 0.5, 0.975], dim="pixel"),
            "n": init_counts,
        }
    )
    out.to_netcdf(scores_dir / "skill.nc")

    # Per-pixel scores by forecast day: pool the 24 leads of a day, keep pixel.
    # -> pixel_daily.nc (day, window, pixel[, baseline])  = the maps
    # -> forecast_limit_map.nc (window, pixel, baseline)

    day = day_labels(by_pixel["lead"])

    # Day-level ACC and anomaly R2 from the moment sums (exact pooling); absent on
    # verifs that predate the sums, in which case the anomaly table is skipped.
    has_moments = all(name in anom for name in MOMENT_NAMES)
    if has_moments:
        moments = anom[MOMENT_NAMES]
        acc_hourly, r2_hourly = anomaly_scores(moments)
        # Same statistic as climpred's pearson_r over (init, pixel); a mismatch beyond
        # float32 noise means the joint-validity masks disagree.
        acc_mismatch = float(abs(acc_hourly - anom["acc_centered"]).max())
        print(f"ACC from moments vs climpred: max |diff| = {acc_mismatch:.2e}")
        acc_daily, r2_daily = anomaly_scores(moments, dim="lead", group=day)

    pixel_daily = pixel_day_scores(by_pixel)
    rmse_ss_daily = pixel_daily["rmse_ss"]
    dispersion_daily = rmse_ss_daily.quantile([0.025, 0.5, 0.975], dim="pixel")
    rmse_model_dispersion = pixel_daily["rmse_model"].quantile([0.025, 0.5, 0.975], dim="pixel")
    rmse_baseline_dispersion = pixel_daily["rmse_baseline"].quantile([0.025, 0.5, 0.975], dim="pixel")

    fc_limit = forecast_limit_map(rmse_ss_daily)
    fc_limit.to_netcdf(scores_dir / "forecast_limit_map.nc")
    pixel_daily.to_netcdf(scores_dir / "pixel_daily.nc")

    # Domain-wide scores by forecast day: pool init and the 24 leads of a day, no
    # pixel dim left. All three come from by_init so the table rows share one source.
    # -> skill_daily.nc (day, window[, baseline][, quantile])  = the curves + tables

    # No sqrt on the bias: it is a signed mean error, not a squared one.
    day_bias_model = pooled_mean(by_init["bias_model"], by_init["n"], dim=["init", "lead"], group=day)
    day_rmse_model = np.sqrt(pooled_mean(by_init["mse_model"], by_init["n"], dim=["init", "lead"], group=day))
    day_rmse_baseline = np.sqrt(pooled_mean(by_init["mse_baseline"], by_init["n"], dim=["init", "lead"], group=day))

    pooled = xr.Dataset(
        {
            "bias_model": day_bias_model,
            "rmse_model": day_rmse_model,
            "rmse_baseline": day_rmse_baseline,
            "rmse_ss": 1.0 - day_rmse_model / day_rmse_baseline,
            "rmse_ss_dispersion": dispersion_daily,
            "rmse_model_dispersion": rmse_model_dispersion,
            "rmse_baseline_dispersion": rmse_baseline_dispersion,
        }
    )
    if has_moments:
        pooled["acc"] = acc_daily
        pooled["r2"] = r2_daily
    # The per-pixel efficiencies of pixel_daily summarised over the pixels of each window
    # (lower, median, upper); the median is the efficiency table's entry. Absent on verifs
    # that predate the per-pixel sums, in which case that table is skipped.
    for name in ("nse", "acc", "r2"):
        if f"{name}_model" in pixel_daily:
            pooled[f"{name}_pixel"] = pixel_daily[f"{name}_model"].quantile(PIXEL_QUANTILES, dim="pixel")
    pooled.to_netcdf(scores_dir / "skill_daily.nc")

    return out, pooled, fc_limit, pixel_daily


def _by_valid_hour(field, solar_offset):
    """(init_hour, lead[, ...]) -> (valid_hour, lead[, ...]). The UTC hour of the valid time
    is init hour + lead mod 24, so every valid hour collects the (init hour, lead) pairs
    that reach it; NaN at the leads no init hour reaches. The `solar_hour` coordinate adds
    the region's longitude offset, for colouring."""
    init_hours = field["init_hour"].values
    leads = field["lead"].values
    valid_hours = np.arange(24)
    needed_init = (valid_hours[:, None] - leads[None, :]) % 24
    reachable = np.isin(needed_init, init_hours)
    init_position = np.searchsorted(init_hours, np.where(reachable, needed_init, init_hours[0]))
    lead_position = np.broadcast_to(np.arange(leads.size), needed_init.shape)
    other_dims = [dim for dim in field.dims if dim not in ("init_hour", "lead")]
    values = field.transpose("init_hour", "lead", *other_dims).values[init_position, lead_position]
    values = np.where(reachable.reshape(reachable.shape + (1,) * len(other_dims)), values, np.nan)
    coords = {dim: field[dim].values for dim in other_dims}
    coords["valid_hour"] = valid_hours
    coords["lead"] = leads
    coords["solar_hour"] = ("valid_hour", (valid_hours + solar_offset) % 24)
    return xr.DataArray(values, dims=["valid_hour", "lead", *other_dims], coords=coords)


def solar_hour_scores(by_init, solar_offset):
    """Error growth and skill by lead, one series per hour of the valid time, pooled over
    the inits: the hourly curve of skill.nc without the diurnal cycle folded into it. Each
    valid hour is reached by one init hour per day, so on the 6-hourly init grid a series
    has a point every sixth lead, and the pooled curve is the count-weighted mix of the
    four series reaching a lead.

    `by_init` is the by_init.nc product of one window, `solar_offset` the region's mean
    longitude / 15 in hours. Meaningful for regions compact in longitude only, where the
    pixels share one offset (the continental sets mix solar hours within a UTC hour).
    -> (valid_hour, lead[, baseline]) with the coordinate solar_hour, plus the pooled
    curves over all hours."""
    init_hour = by_init["init"].dt.hour.rename("init_hour")
    counts = by_init["n"]
    mse_model = pooled_mean(by_init["mse_model"], counts, dim="init", group=init_hour)
    mse_baseline = pooled_mean(by_init["mse_baseline"], counts, dim="init", group=init_hour)
    rmse_model = np.sqrt(_by_valid_hour(mse_model, solar_offset))
    rmse_baseline = np.sqrt(_by_valid_hour(mse_baseline, solar_offset))
    rmse_model_pooled = np.sqrt(pooled_mean(by_init["mse_model"], counts, dim="init"))
    rmse_baseline_pooled = np.sqrt(pooled_mean(by_init["mse_baseline"], counts, dim="init"))
    return xr.Dataset(
        {
            "rmse_model": rmse_model,
            "rmse_ss": 1.0 - rmse_model / rmse_baseline,
            "rmse_model_pooled": rmse_model_pooled,
            "rmse_ss_pooled": 1.0 - rmse_model_pooled / rmse_baseline_pooled,
        }
    )


def solar_hour_month_scores(by_init, solar_offset):
    """solar_hour_scores per month of the init -- not of the valid time, so every series
    rests on one fixed set of forecasts along the lead. -> extra dim month (1-12)."""
    init_month = by_init["init"].dt.month.values
    months = np.unique(init_month)
    per_month = [solar_hour_scores(by_init.isel(init=init_month == month), solar_offset) for month in months]
    return xr.concat(per_month, dim="month").assign_coords(month=months)
