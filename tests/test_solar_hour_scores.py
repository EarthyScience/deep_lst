"""derive.solar_hour_scores / solar_hour_month_scores: the hour-of-day decomposition of the
hourly curves from the per-init product, on a synthetic by_init with 6-hourly inits."""

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation_climpred.derive import pooled_mean, solar_hour_month_scores, solar_hour_scores

LEADS = np.arange(1, 25)
INITS = pd.date_range("2024-01-01", "2024-02-29T18", freq="6h")
SOLAR_OFFSET = 0.5
BASELINE_MSE = 25.0


def synthetic_by_init():
    """Model MSE = (valid UTC hour + 1)^2, so every hour series is flat; the baseline MSE is
    constant; the counts vary by init and lead so the pooling is genuinely weighted."""
    init_hour = INITS.hour.values
    valid_hour = (init_hour[:, None] + LEADS[None, :]) % 24
    mse_model = ((valid_hour + 1) ** 2).astype(float)
    counts = np.random.default_rng(0).integers(1, 10, size=valid_hour.shape)
    return xr.Dataset(
        {
            "mse_model": (("lead", "init"), mse_model.T),
            "mse_baseline": (("lead", "baseline", "init"), np.full((LEADS.size, 1, INITS.size), BASELINE_MSE)),
            "n": (("init", "lead"), counts),
        },
        coords={"init": INITS, "lead": LEADS, "baseline": ["clim_smooth"]},
    )


def test_each_valid_hour_is_reached_at_one_lead_in_six():
    scores = solar_hour_scores(synthetic_by_init(), SOLAR_OFFSET)
    reached = np.isfinite(scores["rmse_model"].values)  # (valid_hour, lead)
    assert reached.sum(axis=1).tolist() == [4] * 24
    for valid_hour in range(24):
        expected = (valid_hour - LEADS) % 24 % 6 == 0
        assert np.array_equal(reached[valid_hour], expected)


def test_hour_series_are_flat_when_the_error_depends_on_the_valid_hour_only():
    scores = solar_hour_scores(synthetic_by_init(), SOLAR_OFFSET)
    for valid_hour in range(24):
        rmse = scores["rmse_model"].sel(valid_hour=valid_hour).dropna("lead")
        skill = scores["rmse_ss"].sel(valid_hour=valid_hour, baseline="clim_smooth").dropna("lead")
        assert np.allclose(rmse, valid_hour + 1)
        assert np.allclose(skill, 1.0 - (valid_hour + 1) / np.sqrt(BASELINE_MSE))
    assert np.allclose(scores["solar_hour"], (np.arange(24) + SOLAR_OFFSET) % 24)


def test_pooled_curve_is_the_count_weighted_pool_over_all_inits():
    by_init = synthetic_by_init()
    scores = solar_hour_scores(by_init, SOLAR_OFFSET)
    expected = np.sqrt(pooled_mean(by_init["mse_model"], by_init["n"], dim="init"))
    assert np.allclose(scores["rmse_model_pooled"], expected)


def test_month_scores_split_the_inits_by_init_month():
    by_init = synthetic_by_init()
    monthly = solar_hour_month_scores(by_init, SOLAR_OFFSET)
    assert monthly["month"].values.tolist() == [1, 2]
    january = by_init.isel(init=by_init["init"].dt.month.values == 1)
    xr.testing.assert_allclose(monthly.sel(month=1).drop_vars("month"), solar_hour_scores(january, SOLAR_OFFSET))
