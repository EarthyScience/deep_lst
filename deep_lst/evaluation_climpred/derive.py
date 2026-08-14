import numpy as np
import xarray as xr

from .common import verification_dir
from .stats import bootstrap_skill_score, forecast_limit_map


def pooled_mean(values, counts, dim, group=None):
    """Count-weighted mean of per-cell averages over `dim`, optionally within `group`.

    Cells carry different sample counts, so averaging the averages is wrong: undo the
    averaging (x count), sum, divide by the summed counts. For an RMSE, pool in the
    mse domain and take the sqrt at the call site -- never before."""
    weighted = values.fillna(0.0) * counts
    if group is None:
        return weighted.sum(dim) / counts.sum(dim)
    return weighted.groupby(group).sum(dim) / counts.groupby(group).sum(dim)


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

    rmse_ss_px = 1.0 - np.sqrt(by_pixel["mse_model"]) / np.sqrt(by_pixel["mse_baseline"])
    dispersion = rmse_ss_px.quantile([0.05, 0.5, 0.95], dim="pixel")

    out = xr.Dataset(
        {
            "rmse_model": rmse_model,
            "rmse_baseline": rmse_baseline,
            "rmse_ss": 1.0 - rmse_model / rmse_baseline,
            "acc_centered": anom["acc_centered"],
            "r2_anomaly": 1.0 - rmse_model**2 / anom["anomaly_obs_variance"],
            "rmse_ss_dispersion": dispersion,
            "n": init_counts,
        }
    )
    out.to_netcdf(scores_dir / "skill.nc")

    # Per-pixel scores by forecast day: pool the 24 leads of a day, keep pixel.
    # -> pixel_daily.nc (day, window, pixel[, baseline])  = the maps
    # -> forecast_limit_map.nc (window, pixel, baseline)

    day = xr.DataArray((by_pixel["lead"].values - 1) // 24 + 1, dims="lead", name="day")  # create day labels

    day_model_bias = pooled_mean(by_pixel["bias_model"], by_pixel["n"], dim="lead", group=day)
    day_model_mse = pooled_mean(by_pixel["mse_model"], by_pixel["n"], dim="lead", group=day)
    day_baseline_mse = pooled_mean(by_pixel["mse_baseline"], by_pixel["n"], dim="lead", group=day)

    day_model_rmse = np.sqrt(day_model_mse)
    day_baseline_rmse = np.sqrt(day_baseline_mse)

    rmse_ss_daily = 1.0 - day_model_rmse / day_baseline_rmse
    dispersion_daily = rmse_ss_daily.quantile([0.05, 0.5, 0.95], dim="pixel")

    fc_limit = forecast_limit_map(rmse_ss_daily)
    fc_limit.to_netcdf(scores_dir / "forecast_limit_map.nc")

    fields = {
        "bias_model": day_model_bias,
        "rmse_model": day_model_rmse,
        "rmse_baseline": day_baseline_rmse,
        "rmse_ss": rmse_ss_daily,
    }

    pixel_daily = xr.Dataset(fields)
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
        }
    )
    pooled.to_netcdf(scores_dir / "skill_daily.nc")

    # Paired block bootstrap over init days -- sampling CI on the pooled skill.
    # -> bootstrap.nc (quantile, lead, window, baseline)
    # -> bootstrap_daily.nc (quantile, day, window, baseline)
    bootstrap_hourly, bootstrap_daily = bootstrap_skill_score(
        by_init["mse_model"], by_init["mse_baseline"], by_init["n"], iterations=500, seed=0, block_days=5
    )
    bootstrap_hourly.rename("rmse_ss_ci").to_netcdf(scores_dir / "bootstrap.nc")
    bootstrap_daily.rename("rmse_ss_ci").to_netcdf(scores_dir / "bootstrap_daily.nc")

    return out, pooled, bootstrap_hourly, bootstrap_daily, fc_limit, pixel_daily
