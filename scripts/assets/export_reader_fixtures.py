"""Golden fixtures of the eodata readers, for the migration to deep_lst's own readers.

Writes tests/fixtures/readers_v1/<modality>.npz and meta.json: what the preload of the old
pipeline (eodata v0.3_DeepLST readers on the lowres_old stores, zarr 3.2) returns for ten
training anchors and four windows of four days, raw and normalised. The new readers cannot
share an environment with the old ones (they need zarr >= 3.4.2), so these files are the only
reference the new readers can be compared with. Never regenerate them from the new readers.

Arrays in <modality>.npz (P pixels, W windows, H = N_DAYS * 24 hours, C variables):

    anchor_ids       (P,)     MajorTom ids, in the order of PIXELS
    anchor_lonlat    (P, 2)   lon, lat of the anchor
    store_rows       (P,)     native cell of the anchor in the old store (not deduplicated)
    cell_lonlat      (P, 2)   lon, lat of that cell
    start_days       (W,)     first day of each window, in days since 1970-01-01
    raw__<data type>         values as read, NaN where missing
    normalized__<data type>  the same after the z-score (and transform) of the training preload

Hourly and daily data types have the shape (W, P, H, C); position h of window w is the hour
start_days[w] * 24 + h since 1970-01-01T00, and daily values are repeated over their 24 hours.
Static data types have the shape (P, C). The variable names of every data type are in meta.json.

statistics.json holds the statistics that normalised these values (mean, std, min, max per
variable), copied from the statistics files of the selection mtg_new, so that a normaliser can be
tested against the normalised arrays without the statistics files.

Run from the repository root, in the environment that has eodata:

    .venv/bin/python scripts/assets/export_reader_fixtures.py
"""

import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import zarr
from eodata.normalization import ModalityNormalizer
from omegaconf import OmegaConf

from deep_lst.dataset.dataset import DATA_TYPES, PATHS, _preload_columns
from deep_lst.dataset.transforms import with_transform
from deep_lst.utils.full_store import open_store_reader
from deep_lst.utils.spatial import anchor_grid, map_to_store_rows, store_grid

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "tests" / "fixtures" / "readers_v1"
SOURCES_FILE = REPO_ROOT / "conf" / "sources" / "eozarr.yaml"

MODALITIES = ("era5", "era5_land", "seviri", "glofas_v4")
STATS_SPATIAL = "mtg_new"  # selection whose statistics normalise the training preload
N_DAYS = 4  # one shard of the new hourly stores is 96 hours
STATISTICS_GROUPS = ("data", "data_daily", "data_static")  # the groups of stored variables

# Anchors of the selection mtg_new. The first six have LST in the old store and data in every
# new product store. The last four sit on a cell that is masked (not on the land grid) in at
# least one new store, as checked on 2026-11-23; the masked stores are named.
PIXELS = {
    "niamey_sahel": 12616679,
    "yangambi_rainforest": 10324148,
    "gobabeb_namib": 6097282,
    "johannesburg_highveld": 5654254,
    "madagascar_disk_edge": 7510778,  # largest angle from the sub-satellite point (52 deg)
    "tunisia_north": 16348980,  # northernmost anchor with LST (37.1 N)
    "lake_victoria": 9859505,  # masked: seviri, era5, era5_land, glofas_v4; no SEVIRI data in the old store
    "south_coast": 4483159,  # masked: seviri, era5
    "dakar_coast": 12794938,  # masked: era5_land; no SEVIRI and no ERA5-Land data in the old store
    "durban_coast": 5088792,  # masked: seviri, era5_land, glofas_v4
}

# First day of each window.
WINDOWS = {
    "record_start": "2016-01-01",
    "lst_gap_9h": "2021-10-10",  # the new LST store has no data from 2021-10-11T22 to 10-12T06
    "year_boundary": "2023-12-30",  # validation year into test year
    "leap_day": "2024-02-27",
}


def days_since_1970(date):
    return int((np.datetime64(date, "D") - np.datetime64("1970-01-01", "D")).astype(np.int64))


def channel_selection(reader, load):
    """Position of every configured variable within its data type, as in the training preload."""
    select = {}
    for data_type, names in load.items():
        store_variables = list(getattr(reader, DATA_TYPES[data_type]).dtype.fields)
        select[data_type] = [store_variables.index(name) for name in names]
    return select


def open_normalizer(modality, reader, transform):
    """The normaliser of the training preload: z-score from the statistics, after the transform."""
    stats_zarr = PATHS.stats_zarr_path(modality, STATS_SPATIAL)
    normalizer = ModalityNormalizer.from_spec(f"zscore:{stats_zarr}", reader.schema)
    return with_transform(normalizer, transform)


def read_windows(reader, normalizer, store_rows, select, start_days):
    """Every data type over all windows: temporal -> (W, P, H, C), static -> (P, C)."""
    per_window = []
    for start_day in start_days:
        day_range = (int(start_day), int(start_day) + N_DAYS)
        columns = _preload_columns(reader, normalizer, store_rows, select, day_range=day_range)
        per_window.append(columns)

    arrays = {}
    for data_type in select:
        first = per_window[0][data_type].numpy()
        if first.ndim == 2:  # static: the same in every window
            arrays[data_type] = first
        else:
            stacked = [columns[data_type].numpy() for columns in per_window]
            arrays[data_type] = np.stack(stacked)
    return arrays


def export_modality(modality, source, anchor_ids, start_days):
    """Arrays of one modality and the part of meta.json that describes them."""
    mtg = anchor_grid("mtg")
    native = store_grid(modality)
    store_rows = map_to_store_rows(mtg, anchor_ids, native)

    load = OmegaConf.to_container(source["load"])
    transform = source.get("transform")
    reader = open_store_reader(modality, source["store"], load)
    select = channel_selection(reader, load)
    normalizer = open_normalizer(modality, reader, transform)

    arrays = {
        "anchor_ids": anchor_ids,
        "anchor_lonlat": np.asarray(mtg.centers[anchor_ids], dtype=np.float64),
        "store_rows": store_rows,
        "cell_lonlat": np.asarray(native.centers[store_rows], dtype=np.float64),
        "start_days": start_days,
    }
    raw = read_windows(reader, None, store_rows, select, start_days)
    normalized = read_windows(reader, normalizer, store_rows, select, start_days)
    for data_type in select:
        arrays[f"raw__{data_type}"] = raw[data_type]
        arrays[f"normalized__{data_type}"] = normalized[data_type]

    store_root = PATHS.store_roots.get(modality)
    description = {
        "store": str(PATHS.eozarr_store_path(source["store"], store_root)),
        "statistics": str(PATHS.stats_zarr_path(modality, STATS_SPATIAL)),
        "transform": transform,
        "native_grid": native.name,
        "variables": load,
        "shapes": {name: list(array.shape) for name, array in arrays.items()},
        "dtypes": {name: str(array.dtype) for name, array in arrays.items()},
    }
    return arrays, description


def git_commit(repository):
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def export_statistics():
    """Write statistics.json: per modality the transform and, per group, the variables and their statistics."""
    sources = OmegaConf.load(SOURCES_FILE)
    statistics = {}
    for modality in MODALITIES:
        root = zarr.open_group(str(PATHS.stats_zarr_path(modality, STATS_SPATIAL)), mode="r")
        groups = {}
        for group_name in STATISTICS_GROUPS:
            if group_name not in root:
                continue
            group = root[group_name]
            groups[group_name] = {
                "variables": list(group.attrs["variables"]),
                "mean": group["mean"][:].tolist(),
                "std": group["std"][:].tolist(),
                "min": group["min"][:].tolist(),
                "max": group["max"][:].tolist(),
            }
        statistics[modality] = {"transform": sources[modality].get("transform"), "groups": groups}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")


def main():
    sources = OmegaConf.load(SOURCES_FILE)
    anchor_ids = np.array(list(PIXELS.values()), dtype=np.int64)
    start_days = np.array([days_since_1970(date) for date in WINDOWS.values()], dtype=np.int64)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modalities = {}
    for modality in MODALITIES:
        arrays, description = export_modality(modality, sources[modality], anchor_ids, start_days)
        np.savez_compressed(OUTPUT_DIR / f"{modality}.npz", **arrays)
        modalities[modality] = description
        print(f"{modality}: {description['shapes']}")

    meta = {
        "written": str(np.datetime64("today")),
        "deep_lst_commit": git_commit(REPO_ROOT),
        "eodata_commit": git_commit(REPO_ROOT.parent / "eodata_slim"),
        "versions": {name: version(name) for name in ("eodata", "zarr", "numpy", "torch")},
        "stats_spatial": STATS_SPATIAL,
        "n_days": N_DAYS,
        "pixels": {label: int(anchor_id) for label, anchor_id in PIXELS.items()},
        "windows": {label: {"first_day": date, "start_day": days_since_1970(date)} for label, date in WINDOWS.items()},
        "modalities": modalities,
    }
    (OUTPUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    export_statistics()


if __name__ == "__main__":
    main()
