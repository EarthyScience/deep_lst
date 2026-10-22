import argparse

import climpred
import numpy as np
import xarray as xr
import yaml

from deep_lst.utils.io import ExperimentPaths

from .cubes import (
    anomaly_ensemble,
    attach_classes,
    attach_coverage,
    attach_init_state,
    baseline_cube,
    obs_series,
    prepare_initialised,
)
from .masks import MASKS, MONTHLY_WINDOWS, register_month_windows, stack_windows

VERIFY_KWARGS = {"metric": "mse", "comparison": "e2o", "alignment": "maximize", "skipna": True}

EVAL_CENTERS = {
    "Dahra": (-15.43, 15.35),
    "West\n Africa": (-3.00, 8.00),
    "Sahel": (9.50, 15.50),
    "Gobabeb": (15.04, -23.56),
    "Johannesburg": (28.05, -26.20),
    "Antananarivo": (46.50, -19.50),
    "Passage": (37.79, 3.36),
    "Yangambi": (24.5, 0.81),
    "Bouamir": (12.82, 3.19),
}


def nearest_pixel(field, name, max_offset_deg=None):
    """Index of the pixel closest to a named evaluation centre; None if the centre
    lies farther than max_offset_deg from every pixel (centre outside the region)."""
    lon, lat = EVAL_CENTERS[name]
    distance = (field["lon"].values - lon) ** 2 + (field["lat"].values - lat) ** 2
    index = int(np.argmin(distance))
    if max_offset_deg is not None and distance[index] > max_offset_deg**2:
        return None
    return index


def lookback_hours(paths, experiment, version):
    """Longest input window in the experiment's snapshot config, in hours.

    Read rather than hard-coded: the shaded region in the site figure claims "this is
    what the model saw", so a stale constant would be a silent lie in the figure."""
    config = yaml.safe_load((paths.experiment_dir(experiment) / version / "config.yml").read_text())
    found = []
    stack = [config]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "lookback" in node:
                found.append(int(node["lookback"]))
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return max(found)


def build_parser(extra_windows=()):
    """extra_windows: window names a verifier computes itself, beyond the masks (verif_cell's "amplitude")."""
    parser = argparse.ArgumentParser(description="Verify model and baselines with climpred.")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name (results/<experiment>/...).")
    parser.add_argument("--version", type=str, default="version_0", help="Experiment version.")
    parser.add_argument(
        "--eval_name",
        type=str,
        required=True,
        help="Eval name (results subfolder holding forecast.zarr); the spatial selection is derived from it.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        choices=list(MASKS) + list(extra_windows),
        default=["all", "day", "night"],
        help="Solar windows to stack along the 'window' dim; one shared verify pass.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=list(range(1, 13)),
        choices=range(1, 13),
        metavar="M",
        help="Months to filter by (1-12); if given, the 'month' mask is applied.",
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Add the twelve single-month windows m01-m12 (month-by-month maps); verif_pixel then writes by_pixel_monthly.nc.",
    )
    return parser


def baseline_files(paths, spatial):
    return {
        "clim_smooth": paths.climatology_path(spatial, "smoothed_hourly"),
        "persistence": paths.persistence_path(spatial),
        "era5land": paths.modality_baseline_path("era5_land", spatial),
    }


def setup_verif(extra_windows=()):
    args = build_parser(extra_windows).parse_args()
    args.windows = args.windows + register_month_windows(args.months)
    if args.monthly:
        args.windows = args.windows + MONTHLY_WINDOWS
    args.paths = ExperimentPaths()
    args.spatial = args.paths.baseline_spatial(args.eval_name)

    forecast = xr.open_dataset(args.paths.forecast_path(args.experiment, args.version, args.eval_name))
    initilised = prepare_initialised(forecast)
    initilised = attach_classes(initilised, args.paths.spatial_indices_path(args.spatial))
    obs = obs_series(forecast)
    lookback = lookback_hours(args.paths, args.experiment, args.version)
    initilised = attach_coverage(initilised, obs, lookback)
    state_path = args.paths.assets_root / "static" / f"fvc_init_state_{args.spatial}.nc"
    if state_path.exists():  # only the selections with a state file get the greenness windows
        initilised = attach_init_state(initilised, state_path)
    hydro_path = args.paths.assets_root / "static" / f"hydro_init_state_{args.spatial}.nc"
    if hydro_path.exists():  # hydrological states at init (wetness, ...) as coords for the per-cell product
        initilised = attach_init_state(initilised, hydro_path, variables=None)
    return args, initilised, obs


def verification_dir(args, kind):
    return args.paths.verification_dir(args.experiment, args.version, args.eval_name, kind)


def build_ensembles(args, initilised, obs):
    files = baseline_files(args.paths, args.spatial)
    cubes = np.stack([baseline_cube(path, initilised) for path in files.values()])
    # Score the baselines on the model's samples only. The rollout issues no forecast from a
    # cloudy init hour (NaN seed state), so without this mask climpred averages each baseline
    # over every observed hour -- a superset the model is never scored on -- and the skill
    # score compares two different sample sets. Climatology and persistence are valid wherever
    # the model is, so their sample set becomes identical to the model's and the joint counts
    # (per_init_counts / per_pixel_counts) remain the exact pooling weights.
    cubes = np.where(np.isfinite(initilised["LST"].values), cubes, np.nan)
    baseline_init = xr.Dataset(
        {"LST": (("baseline", "init", "lead", "pixel"), cubes)},
        coords=dict(initilised.coords) | {"baseline": list(files)},
    )
    baseline_init["lead"].attrs["units"] = "hours"

    hindcast = climpred.HindcastEnsemble(stack_windows(initilised, args.windows)).add_observations(obs)
    baselines = climpred.HindcastEnsemble(stack_windows(baseline_init, args.windows)).add_observations(obs)
    return hindcast, baselines


def build_anomaly_ensemble(args, initilised, obs):
    clim_path = baseline_files(args.paths, args.spatial)["clim_smooth"]
    anomaly_init, anomaly_obs = anomaly_ensemble(initilised, obs, clim_path, args.windows)
    acc_hindcast = climpred.HindcastEnsemble(anomaly_init).add_observations(anomaly_obs)
    return acc_hindcast, anomaly_obs
