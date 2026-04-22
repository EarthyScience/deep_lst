import xarray as xr

from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()


def load_forecast(
    experiment: str,
    version: str,
    eval_name: str,
    suffix: str = "",
    paths: ExperimentPaths = None,
) -> xr.Dataset:
    """Load a saved forecast zarr store for the given experiment/version/eval."""
    p = (paths or PATHS).forecast_path(experiment, version, eval_name, suffix=suffix)
    return xr.open_zarr(p)


def load_metrics(
    experiment: str,
    version: str,
    eval_name: str,
    paths: ExperimentPaths = None,
) -> xr.Dataset:
    """Load saved metrics NetCDF for the given experiment/version/eval."""
    p = (paths or PATHS).metrics_path(experiment, version, eval_name)
    return xr.open_dataset(p)
