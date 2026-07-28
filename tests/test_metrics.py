import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.metrics import to_init_time, to_valid_time


def make_fc_array(n_init=10, n_lead=4, lat=5, lon=5):
    arr = xr.DataArray(
        data=np.random.rand(n_init, n_lead, lat, lon).astype(np.float32),
        dims=["init_time", "lead_time", "lat", "lon"],
        coords={
            "init_time": pd.date_range("2023-01-01", periods=n_init, freq="h"),
            "lead_time": np.arange(n_lead),
            "lat": np.linspace(-10, 10, lat),
            "lon": np.linspace(20, 40, lon),
        },
        name="LST_pred",
    )
    return arr


def test_to_valid_time():
    lead_time = 1
    preds = make_fc_array().sel(lead_time=lead_time)

    init_times = preds.init_time.values
    valid_times = to_valid_time(preds, lead_time=f"{lead_time}h").valid_time.values

    assert init_times.shape == valid_times.shape
    assert valid_times[0] == init_times[0] + np.timedelta64(lead_time, "h")


def test_to_init_time():
    lead_time = 1
    preds = make_fc_array().sel(lead_time=lead_time)
    pred_valid = to_valid_time(preds, lead_time=f"{lead_time}h")
    valid_times = pred_valid.valid_time.values
    init_times = to_init_time(pred_valid, lead_time=f"{lead_time}h").init_time.values

    assert valid_times.shape == init_times.shape
    assert init_times[0] == valid_times[0] - np.timedelta64(lead_time, "h")


def test_baseline_alignment():
    lead_time = 1
    preds = make_fc_array().sel(lead_time=lead_time)
    valid_times = to_valid_time(preds, lead_time=f"{lead_time}h").valid_time.values

    clim_on_valid = make_fc_array().sel(lead_time=lead_time).rename({"init_time": "valid_time"}).rename("LST_clim")

    clim = (
        clim_on_valid.reindex(valid_time=valid_times)  # reindexes existing valid_time dim
    )
    clim = to_init_time(clim, lead_time=f"{lead_time}h")  # shifts valid_time → init_time

    assert clim.init_time.values.shape == clim_on_valid.valid_time.values.shape
    assert clim_on_valid.valid_time.values[0] == clim.init_time.values[0]


def test_skill_score():
    from deep_lst.evaluation.metrics import skill_score

    assert skill_score(1.0, 2.0) == 0.5  # model halves the baseline error
    assert skill_score(2.0, 2.0) == 0.0  # no improvement
    assert skill_score(3.0, 2.0) == -0.5  # model worse than baseline

    model = xr.DataArray([1.0, 2.0], dims="lead_time")
    baseline = xr.DataArray([2.0, 2.0], dims="lead_time")
    result = skill_score(model, baseline)
    np.testing.assert_allclose(result.values, [0.5, 0.0])


# --- the sufficient-statistics chain (mirrors notebooks/suffstats_chain.ipynb) ---


def _synthetic_cube():
    """Small (init, lead, lat, lon) forecast cube with NaNs on both sides, seeded."""
    rng = np.random.default_rng(7)
    init_times = pd.date_range("2024-01-01", periods=8, freq="6h")
    lead_times = pd.timedelta_range("1h", periods=48, freq="1h")
    shape = (8, 48, 3, 2)
    pred_values = rng.normal(300.0, 5.0, size=shape)
    obs_values = pred_values + rng.normal(0.0, 2.0, size=shape)
    obs_values[rng.random(shape) < 0.15] = np.nan  # clouds
    pred_values[rng.random(shape) < 0.03] = np.nan
    coords = {"init_time": init_times, "lead_time": lead_times, "lat": [10.0, 10.05, 10.1], "lon": [0.0, 30.0]}
    dims = ("init_time", "lead_time", "lat", "lon")
    pred = xr.DataArray(pred_values, dims=dims, coords=coords)
    obs = xr.DataArray(obs_values, dims=dims, coords=coords)
    return pred, obs


def test_sufficient_stats_hand_table():
    from deep_lst.evaluation.metrics import measures_from_stats, sufficient_stats

    pred = xr.DataArray([1.0, 2.0, 3.0, 4.0, np.nan], dims="time")
    obs = xr.DataArray([1.0, 1.0, np.nan, 2.0, 2.0], dims="time")
    stats = sufficient_stats(pred, obs, dim="time")

    # joint mask keeps samples 0, 1, 3; sample 2 (pred finite, obs NaN) must NOT count
    assert int(stats["n"]) == 3
    assert float(stats["e_sum"]) == 3.0
    assert float(stats["se_sum"]) == 5.0
    assert float(stats["ae_sum"]) == 3.0

    finished = measures_from_stats(stats, ("rmse", "bias", "mae"))
    assert np.isclose(float(finished["rmse"]), np.sqrt(5.0 / 3.0))
    assert np.isclose(float(finished["bias"]), 1.0)
    assert np.isclose(float(finished["mae"]), 1.0)


def test_pool_stats_plain_dim_and_1d_labels():
    from deep_lst.evaluation.metrics import pool_stats

    s = xr.Dataset({"se_sum": ("t", [1.0, 2.0, 3.0, 4.0]), "n": ("t", [1, 1, 1, 1])})
    labels = xr.DataArray([0, 0, 1, 1], dims="t", name="group")
    pooled = pool_stats(s, by=labels)
    assert list(pooled["se_sum"].values) == [3.0, 7.0]
    assert list(pooled["n"].values) == [2, 2]

    total = pool_stats(s, dim="t")
    assert float(total["se_sum"]) == 10.0


def test_pool_stats_multidim_labels():
    from deep_lst.evaluation.metrics import pool_stats

    s = xr.Dataset({"se_sum": (("a", "b"), [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])})
    labels = xr.DataArray([["A", "A", "B"], ["A", "B", "B"]], dims=("a", "b"), name="grp")
    pooled = pool_stats(s, by=labels)
    assert list(pooled["se_sum"].values) == [13.0, 53.0]  # A = 1+2+10, B = 3+20+30


def test_chain_reproduces_masked_pooling():
    from deep_lst.evaluation.conditioning import time_of_day_mask
    from deep_lst.evaluation.metrics import measures_from_stats, sufficient_stats

    pred, obs = _synthetic_cube()
    day_mask = time_of_day_mask(pred, "day")
    pred_day = pred.where(day_mask)
    obs_day = obs.where(day_mask)

    reference = float(np.sqrt(((obs_day - pred_day) ** 2).mean(skipna=True)))
    stats = sufficient_stats(pred_day, obs_day, dim=list(pred.dims))
    new = float(measures_from_stats(stats, ("rmse",))["rmse"])
    assert np.isclose(new, reference, rtol=1e-6)


def test_nse_joint_masking_hand_table():
    from deep_lst.evaluation.metrics import measures_from_stats, sufficient_stats

    pred = xr.DataArray([1.0, 2.0, 3.0, 4.0, np.nan], dims="time")
    obs = xr.DataArray([1.0, 1.0, np.nan, 2.0, 2.0], dims="time")
    # joint samples 0, 1, 3: o = [1, 1, 2], o_mean = 4/3, ss_res = 5, ss_tot = 2/3 -> nse = -6.5
    stats = sufficient_stats(pred, obs, dim="time")
    chain_nse = float(measures_from_stats(stats, ("nse",))["nse"])
    assert np.isclose(chain_nse, -6.5)


def test_nse_chain_matches_inline_reference_on_cube():
    from deep_lst.evaluation.metrics import measures_from_stats, sufficient_stats

    pred, obs = _synthetic_cube()
    valid = pred.notnull() & obs.notnull()
    p, o = pred.where(valid), obs.where(valid)
    ss_res = float(((o - p) ** 2).sum(skipna=True))
    ss_tot = float(((o - o.mean(skipna=True)) ** 2).sum(skipna=True))
    reference_nse = 1.0 - ss_res / ss_tot
    stats = sufficient_stats(pred, obs, dim=list(pred.dims))
    chain_nse = float(measures_from_stats(stats, ("nse",))["nse"])
    assert np.isclose(chain_nse, reference_nse, rtol=1e-4)  # float32 reference vs float64 chain sums
