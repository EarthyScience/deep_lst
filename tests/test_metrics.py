import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.metrics import compute_grouped_scores, compute_scores, to_init_time, to_valid_time


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


def test_aggregation():
    lead_time = 1
    preds = make_fc_array().sel(lead_time=lead_time)
    targets = make_fc_array().sel(lead_time=lead_time).rename("LST_obs")
    preds = to_valid_time(preds, lead_time=f"{lead_time}h")
    targets = to_valid_time(targets, lead_time=f"{lead_time}h")
    valid_times = preds.valid_time.values

    clim = make_fc_array().sel(lead_time=lead_time).rename({"init_time": "valid_time"}).reindex(valid_time=valid_times)
    clim = to_init_time(clim, lead_time=f"{lead_time}h")

    metrics = compute_scores(preds, targets, metrics=["rmse"], aggregation="total")
    assert np.isscalar(metrics.rmse.item())
    assert metrics.rmse.shape == ()

    metrics = compute_scores(preds, targets, metrics=["rmse"], aggregation="region_time")
    assert metrics.rmse.shape == (preds.valid_time.size,)

    metrics = compute_scores(preds, targets, metrics=["rmse"], aggregation="region_space")
    assert metrics.rmse.shape == (preds.lat.size, preds.lon.size)


def test_compute_metrics():
    preds = make_fc_array().sel(lead_time=1)
    targets = make_fc_array().sel(lead_time=1)

    metrics = compute_scores(preds, targets, metrics=["rmse", "acc", "nse", "c_acc"], aggregation="total")

    assert "rmse" in metrics.data_vars
    assert "acc" in metrics.data_vars
    assert "nse" in metrics.data_vars
    assert "c_acc" in metrics.data_vars

    metrics = compute_scores(preds, targets, metrics=["rmse", "acc", "nse", "c_acc"], aggregation="region_time")

    assert "rmse" in metrics.data_vars
    assert "acc" in metrics.data_vars
    assert "nse" in metrics.data_vars
    assert "c_acc" in metrics.data_vars

    metrics = compute_scores(preds, targets, metrics=["rmse", "acc", "nse", "c_acc"], aggregation="region_space")

    assert "rmse" in metrics.data_vars
    assert "acc" in metrics.data_vars
    assert "nse" in metrics.data_vars
    assert "c_acc" in metrics.data_vars


def test_compute_grouped_scores():
    lead_time = 1
    preds = to_valid_time(make_fc_array().sel(lead_time=lead_time), lead_time=f"{lead_time}h")
    targets = to_valid_time(make_fc_array().sel(lead_time=lead_time), lead_time=f"{lead_time}h")

    # Case 1: total aggregation, diurnal → one scalar per hour
    ds = compute_grouped_scores(preds, targets, metrics=["rmse"], aggregation="total", groupby="diurnal")
    assert "hour" in ds.dims
    assert ds.dims["hour"] == 24
    assert ds["rmse"].shape == (24,)
    assert "pred_mean" in ds and "obs_mean" in ds
    # hours with data are finite; empty hours (e.g. 0) are NaN — not an error
    assert np.isfinite(ds["rmse"].sel(hour=1)).item()
    assert np.isnan(ds["rmse"].sel(hour=0)).item()

    # CAse 2: total aggregation, monthly
    ds = compute_grouped_scores(preds, targets, metrics=["rmse"], aggregation="total", groupby="monthly")
    assert "month" in ds.dims
    assert ds.dims["month"] == 12
    assert ds["rmse"].shape == (12,)
    assert "pred_mean" in ds and "obs_mean" in ds
    # months with data are finite; empty months (e.g. 2-12) are NaN — not an error
    assert np.isfinite(ds["rmse"].sel(month=1)).item()
    assert np.isnan(ds["rmse"].sel(month=2)).item()

    # CAse 3: region_time aggregation, diurnal → one value per hour per valid_time
    ds = compute_grouped_scores(preds, targets, metrics=["rmse"], aggregation="region_time", groupby="diurnal")
    assert "hour" in ds.dims and "valid_time" in ds.dims
    assert ds.dims["hour"] == 24
    assert ds.dims["valid_time"] == preds.valid_time.size
    print(ds["rmse"].dims)
    assert ds["rmse"].shape == (24, preds.valid_time.size)
    assert "pred_mean" in ds and "obs_mean" in ds
    # hours with data are finite; empty hours (e.g. 0) are NaN — not an error
    assert np.isnan(ds["rmse"].sel(hour=0).values).all()  # hour 0: all NaN
    assert np.isfinite(ds["rmse"].sel(hour=1).values).any()

    # CAse 4: region_space aggregation, diurnal → one value per hour per lat/lon
    ds = compute_grouped_scores(preds, targets, metrics=["rmse"], aggregation="region_space", groupby="diurnal")
    assert "hour" in ds.dims and "lat" in ds.dims and "lon" in ds.dims
    assert ds.dims["hour"] == 24
    assert ds.dims["lat"] == preds.lat.size
    assert ds.dims["lon"] == preds.lon.size
    assert ds["rmse"].shape == (24, preds.lat.size, preds.lon.size)
    assert ds["rmse"].dims == ("hour", "lat", "lon")
    assert "pred_mean" in ds and "obs_mean" in ds
