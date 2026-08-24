import numpy as np
import xarray as xr

from .masks import stack_windows


def payload(dataset):
    """Payload variable, named by the file's 'variable' attr; falls back to the first
    data var for in-memory series without attrs (LST for obs and climatology)."""
    variable = dataset.attrs.get("variable", list(dataset.data_vars)[0])
    return dataset[variable]


def hourly_gather(times, values, valid_times):
    """Valid-time gather from an hourly (time, ...) series; NaN outside the series span."""
    index = ((valid_times - times[0]) / np.timedelta64(1, "h")).astype(int)
    inside = (index >= 0) & (index < len(times))
    out = np.full(valid_times.shape + values.shape[1:], np.nan, dtype=np.float32)
    out[inside] = values[index[inside]]
    return out


def series_at_pixel(dataset, valid_times, pixel):
    """Valid-time gather of a (time, pixel) series into an (init, lead) matrix.

    Same indexing as baseline_cube, taken at a single pixel so the full-store cube is
    never built for a two-site figure."""
    times = dataset["valid_time"].values if "valid_time" in dataset.coords else dataset["time"].values
    values = payload(dataset).values[:, pixel]
    return hourly_gather(times, values, valid_times)


def prepare_initialised(fc_data):
    """LST_pred (init_time, lead_time, pixel) -> climpred-ready dataset (notebook cell)."""
    initilised = fc_data[["LST_pred"]].rename({"LST_pred": "LST"})
    initilised = initilised.isel(lead_time=slice(1, None)).rename({"init_time": "init", "lead_time": "lead"})
    lead_hours = (initilised["lead"] / np.timedelta64(1, "h")).astype(int)
    initilised = initilised.assign_coords(lead=lead_hours)
    initilised["lead"].attrs["units"] = "hours"
    return initilised


def obs_series(fc_data):
    block = fc_data["LST_obs"].isel(lead_time=slice(1, 7)).values
    tail = fc_data["LST_obs"].isel(init_time=-1, lead_time=slice(7, 30)).values
    values = np.concatenate([block.reshape(-1, block.shape[2]), tail], axis=0)
    start = fc_data["init_time"].values[0] + np.timedelta64(1, "h")
    hours = start + np.arange(values.shape[0]) * np.timedelta64(1, "h")
    coords = {"time": hours, "lat": fc_data["lat"], "lon": fc_data["lon"]}
    return xr.DataArray(values, dims=("time", "pixel"), coords=coords, name="LST").to_dataset()


def baseline_cube(path, initilised):
    series = xr.open_dataset(path)
    if "lat" in series.coords and not np.allclose(series["lat"].values, initilised["lat"].values):
        raise ValueError(f"{path} pixels do not match the forecast store")

    valid = initilised["init"].values[:, None] + initilised["lead"].values[None, :] * np.timedelta64(1, "h")
    series_values = payload(series).values.astype(np.float32)  # LST for clim/persistence, skt for era5land
    return hourly_gather(series["valid_time"].values, series_values, valid)


def obs_cube(obs, initilised):
    """Obs series -> (init, lead, pixel) cube, same indexing as baseline_cube."""
    valid = initilised["init"].values[:, None] + initilised["lead"].values[None, :] * np.timedelta64(1, "h")
    cube = hourly_gather(obs["time"].values, obs["LST"].values, valid)
    return xr.DataArray(cube, dims=("init", "lead", "pixel"), coords=initilised.coords)


def anomaly_ensemble(initilised, obs, clim_path, windows):
    """Anomalies w.r.t. the smoothed climatology on both sides, for the centered ACC."""
    clim = baseline_cube(clim_path, initilised)
    anomaly_init = initilised.copy()
    anomaly_init["LST"] = initilised["LST"] - clim
    clim_series = xr.open_dataset(clim_path)
    clim_obs = payload(clim_series).rename({"valid_time": "time"})
    anomaly_obs = (obs["LST"] - clim_obs).to_dataset(name="LST")
    return stack_windows(anomaly_init, windows), anomaly_obs


def attach_classes(initilised, npz_path):
    """Per-pixel KG and landcover classes from the selection npz (resolved by io.py),
    reordered to the forecast pixel order (sorted src_indices)."""
    npz = np.load(npz_path)
    order = np.argsort(npz["src_indices"], kind="stable")
    initilised = initilised.assign_coords(kg_class=("pixel", npz["kg_classes"][order]))
    initilised = initilised.assign_coords(lc_class=("pixel", npz["modis_lc_classes"][order]))
    return initilised


def attach_coverage(initilised, obs, lookback):
    """Fraction of valid obs hours in the lookback window before each init, as an
    (init, pixel) coord. Inits whose window starts before the obs series are set to
    NaN so they drop out of every coverage window (but stay in all/day/night)."""
    init_times = initilised["init"].values
    lookback_times = init_times[:, None] + np.arange(-lookback, 0)[None, :] * np.timedelta64(1, "h")

    observed = payload(obs).values
    times = obs["valid_time"].values if "valid_time" in obs.coords else obs["time"].values

    gathered = hourly_gather(times, observed, lookback_times)

    coverage = np.isfinite(gathered).mean(axis=1)
    truncated = init_times < times[0] + lookback * np.timedelta64(1, "h")
    coverage[truncated] = np.nan

    initilised = initilised.assign_coords(lookback_coverage=(("init", "pixel"), coverage))

    return initilised
