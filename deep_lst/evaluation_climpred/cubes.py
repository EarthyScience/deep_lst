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
    """Continuous hourly obs series scattered from the store's own LST_obs.

    Every (init, lead) pair writes its obs at valid time init+lead; overlapping writes
    are identical by construction (obs depend on the valid time only), so for evenly
    spaced inits this reproduces the previously tiled series exactly. Hours no forecast
    covers (e.g. the gaps between midmonth validation blocks) stay NaN -- the axis
    remains dense hourly, which hourly_gather's positional indexing requires."""
    init_times = fc_data["init_time"].values
    obs = fc_data["LST_obs"].isel(lead_time=slice(1, None)).values  # (init, lead 1..H, pixel)
    n_leads = obs.shape[1]
    n_hours = int((init_times[-1] - init_times[0]) / np.timedelta64(1, "h")) + n_leads
    values = np.full((n_hours, obs.shape[2]), np.nan, dtype=np.float32)
    offsets = ((init_times - init_times[0]) / np.timedelta64(1, "h")).astype(int)
    for i, offset in enumerate(offsets):
        values[offset : offset + n_leads] = obs[i]
    start = init_times[0] + np.timedelta64(1, "h")
    hours = start + np.arange(n_hours) * np.timedelta64(1, "h")
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


def anomaly_pair(initilised, obs, clim_path):
    """Forecast and obs anomalies w.r.t. the smoothed climatology: the forecast as an
    (init, lead, pixel) cube, the obs as a continuous (time, pixel) series."""
    clim = baseline_cube(clim_path, initilised)
    anomaly_forecast = initilised["LST"] - clim
    clim_series = xr.open_dataset(clim_path)
    clim_obs = payload(clim_series).rename({"valid_time": "time"})
    anomaly_obs = (obs["LST"] - clim_obs).to_dataset(name="LST")
    return anomaly_forecast, anomaly_obs


def anomaly_ensemble(initilised, obs, clim_path, windows):
    """Anomalies w.r.t. the smoothed climatology on both sides, for the centered ACC."""
    anomaly_forecast, anomaly_obs = anomaly_pair(initilised, obs, clim_path)
    anomaly_init = initilised.copy()
    anomaly_init["LST"] = anomaly_forecast
    return stack_windows(anomaly_init, windows), anomaly_obs


def attach_classes(initilised, npz_path):
    """Per-pixel KG and landcover classes from the selection npz (resolved by io.py),
    reordered to the forecast pixel order (sorted src_indices)."""
    npz = np.load(npz_path)
    order = np.argsort(npz["src_indices"], kind="stable")
    initilised = initilised.assign_coords(kg_class=("pixel", npz["kg_classes"][order]))
    initilised = initilised.assign_coords(lc_class=("pixel", npz["modis_lc_classes"][order]))
    if "fvc_classes" in npz.files:  # annual-mean FVC decile, written by scripts/assets/fvc_site_classes.py
        initilised = initilised.assign_coords(fvc_class=("pixel", npz["fvc_classes"][order]))
    return initilised


def pixel_attributes(npz_path):
    """Per-pixel FVC (annual mean + decile) and landcover class in forecast pixel order, for
    the FVC scatter; None when the selection npz predates scripts/assets/fvc_site_classes.py."""
    npz = np.load(npz_path)
    if "fvc_classes" not in npz.files:
        return None
    order = np.argsort(npz["src_indices"], kind="stable")
    attributes = {
        "fvc_mean": npz["fvc_annual_mean"][order],
        "fvc_class": npz["fvc_classes"][order],
        "lc_class": npz["modis_lc_classes"][order],
    }
    if "lst_validity_annual_mean" in npz.files:
        attributes["lst_validity"] = npz["lst_validity_annual_mean"][order]
    return attributes


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


def attach_init_state(initilised, state_path, variables=("relative_greenness",)):
    """States of each pixel on its init day from a (site, time) asset file
    (assets/static/fvc_init_state_<selection>.nc from scripts/assets/fvc_init_state.py, or
    hydro_init_state_<selection>.nc from hydro_init_state.py) as (init, pixel) coords named
    like the variables; NaN where the state is unknown, so the init drops out of every state
    window but stays in all/day/night. variables=None attaches every data variable of the file."""
    state = xr.open_dataset(state_path)
    # Positional alignment, like attach_classes: the forecast pixels and the state file's sites
    # both follow the sorted src_indices order (the forecast's pixel coordinate is a plain
    # 0..N-1 position, not the MTG ids, so the values themselves cannot be compared).
    if state.sizes["site"] != initilised.sizes["pixel"]:
        raise ValueError(f"{state_path}: {state.sizes['site']} sites but {initilised.sizes['pixel']} forecast pixels")
    init_days = initilised["init"].values.astype("datetime64[D]")
    if variables is None:
        variables = list(state.data_vars)
    for variable in variables:
        values = state[variable].reindex(time=init_days).values  # (pixel, init), NaN outside the file's year
        initilised = initilised.assign_coords({variable: (("init", "pixel"), values.T)})
    return initilised
