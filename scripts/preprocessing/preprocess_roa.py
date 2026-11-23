"""Preprocess the Rain over Africa (RoA) raw mirror into roa_land_expected_hourly_acc.zarr.

The store has the layout of the SEVIRI product stores (seviri_product_hourly_lst.zarr): hourly
time axis since 1970, 0.05 degree grid of 3201 x 3201 cells, values on the cells of the SEVIRI
land grid only, one shard file per 96 hours.

The work is split into three calls:

  setup     1. Create the empty store from the eodata template of the SEVIRI 0.05 degree
               products and compare its layout with the reference LST store.
            2. Build the regrid mapping of every source grid in the raw mirror.
            Both are skipped when their output is already there.
  shards    3. Compute and write the hourly values of a range of shards. A finished shard
               leaves a file in written_shards/ and is skipped by a later call, so shard
               ranges can run in parallel jobs and an interrupted range can be repeated.
  indices   4. Register the hours of all finished shards in the _indices array of the store.
               Run once, after all shards are written; repeating it is harmless.

The raw mirror comes from scripts/downloads/download_roa.sh. Run this script with the
environment of the eodata_main worktree (eodata v0.6.1, zarr 3.4):

    bash scripts/preprocessing/preprocess_roa.sh setup
    bash scripts/preprocessing/preprocess_roa.sh shards --first_shard 4200 --last_shard 4209
    bash scripts/preprocessing/preprocess_roa.sh indices
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zarr
from eodata import grids
from eodata.eo_zarr import EOZarrStore
from eodata.registry import get_spatial_grid, register_zarr_schema
from eodata.zarr_schema import seviri_product_hourly as seviri_template
from eodata.zarr_schema.base import ZarrDataVariable, ZarrSchema

from deep_lst.preprocessing.rain_over_africa.aggregation import (
    RAIN_RATE_SCALE,
    STORE_DTYPE,
    STORE_FILL_VALUE,
)
from deep_lst.preprocessing.rain_over_africa.mapping import (
    RegridMapping,
    build_mapping,
    grid_fingerprint,
)
from deep_lst.preprocessing.rain_over_africa.writer import (
    build_catalogue,
    mapping_path,
    open_raw_years,
    shard_values,
    time_index_of_hours,
)

RAW_DIR = Path("/Net/Groups/BGI/tscratch/mwessel/data/rain_over_africa")
OUTPUT_DIR = Path("/Net/Groups/BGI/tscratch/mwessel/data/rain_over_africa_product")
REFERENCE_STORE = Path("/Net/Groups/BGI/tscratch/weathergenerator/lowres_model/seviri_product_hourly_lst.zarr")
MODALITY_ID = "roa_land_expected_hourly_acc"
LAND_GRID_NAME = "SEVIRIProductDiskLandGrid_Equiangular"
YEARS = range(2016, 2026)

# Codec chain of the existing product stores. eodata v0.6.1 would default to crc32c first and
# blosc without shuffle.
REFERENCE_COMPRESSOR_CONFIG = (
    {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "shuffle"}},
    {"name": "crc32c"},
)

DESCRIPTION = (
    "Rain over Africa (RoA) expected rain rate. rain_rate: mean of the 15-minute scans that end "
    f"in the hour before the time stamp, stored integer * {RAIN_RATE_SCALE} = mm/h = mm of rain "
    "in that hour. n_scans: number of scans in the mean. Regridded from the geostationary grid "
    "to the SEVIRI 0.05 degree land grid. "
    "Uses a structured dtype to store multiple bands in a single array."
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["setup", "shards", "indices"])
    parser.add_argument("--first_shard", type=int, help="first shard to write (step shards)")
    parser.add_argument("--last_shard", type=int, help="last shard to write, inclusive (step shards)")
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR, help="folder with roa_<year>.zarr")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="folder for the store and the mappings")
    parser.add_argument("--reference_store", type=Path, default=REFERENCE_STORE, help="store to compare the layout to")
    return parser.parse_args()


# --- step 1: empty store ------------------------------------------------------------------


def roa_schema():
    """Schema of the RoA store: the SEVIRI 0.05 degree hourly template with the RoA fields."""
    spatial_chunk = seviri_template.SEVIRI_PRODUCT_5KM_chunk_spatial_size
    temporal_chunk = seviri_template.SEVIRI_PRODUCT_5KM_chunk_temporal_size
    data_variable = ZarrDataVariable.temporal_array(
        time_grid=grids.T1H,
        name="data",
        dimensions=seviri_template.SEVIRI_PRODUCT_5KM_extra_dims,
        coords=seviri_template.SEVIRI_PRODUCT_5KM_extra_coords,
        chunks=(temporal_chunk, spatial_chunk, spatial_chunk),
        shards=seviri_template.SEVIRI_PRODUCT_5KM_shard_size,
        dtype=STORE_DTYPE,
        fill_value=STORE_FILL_VALUE,
        compressor_config=REFERENCE_COMPRESSOR_CONFIG,
        short_name="data",
        long_name=f"Rain over Africa {STORE_DTYPE.names} hourly product",
        description=DESCRIPTION,
        metadata=seviri_template.SEVIRI_PRODUCT_5KM_PROFILE_WKT,
    )
    return ZarrSchema(data_variable, spatial_grid=get_spatial_grid(LAND_GRID_NAME), time_grid=grids.T1H)


def create_empty_store(store_path):
    """Create the store with its arrays, coordinates and root attributes; no data is written."""
    register_zarr_schema(MODALITY_ID, roa_schema())
    EOZarrStore(str(store_path), mode="w-", modality_id=MODALITY_ID)


def read_metadata(array_path):
    with open(Path(array_path) / "zarr.json") as handle:
        return json.load(handle)


def without_typesize(codecs):
    """The codec chain without the blosc typesize, which is the item size of the dtype."""
    text = json.dumps(codecs)
    cleaned = json.loads(text)
    for codec in cleaned:
        inner_codecs = codec.get("configuration", {}).get("codecs", [])
        for inner_codec in inner_codecs:
            inner_codec.get("configuration", {}).pop("typesize", None)
    return cleaned


def layout_differences(store_path, reference_path):
    """Compare the layout of a store with a reference store of the same time and spatial grid.

    Returns:
        A list of differences as text; empty when the layouts agree. The fields of the data
        array (dtype, fill value, attributes) are expected to differ and are not compared.
    """
    differences = []

    data = read_metadata(store_path / "data")
    reference_data = read_metadata(reference_path / "data")
    for key in ("shape", "chunk_grid", "chunk_key_encoding", "dimension_names", "zarr_format", "node_type"):
        if data[key] != reference_data[key]:
            differences.append(f"data {key}: {data[key]} != {reference_data[key]}")
    if without_typesize(data["codecs"]) != without_typesize(reference_data["codecs"]):
        differences.append(f"data codecs: {data['codecs']} != {reference_data['codecs']}")

    for name in ("lat", "lon", "time_idx"):
        if read_metadata(store_path / name) != read_metadata(reference_path / name):
            differences.append(f"{name}: metadata differs")
        values = zarr.open_array(store_path / name, mode="r")[:]
        reference_values = zarr.open_array(reference_path / name, mode="r")[:]
        if not np.array_equal(values, reference_values):
            differences.append(f"{name}: values differ")

    indices = read_metadata(store_path / "_indices")
    reference_indices = read_metadata(reference_path / "_indices")
    indices.pop("shape")
    reference_indices.pop("shape")
    if indices != reference_indices:
        differences.append("_indices: metadata differs")

    attributes = read_metadata(store_path)["attributes"]
    reference_attributes = read_metadata(reference_path)["attributes"]
    if attributes["EOZarr_version"] != reference_attributes["EOZarr_version"]:
        differences.append("EOZarr_version differs")
    for key in ("spatial_grid", "time_grid"):
        if attributes["zarr_schema"][key] != reference_attributes["zarr_schema"][key]:
            differences.append(f"zarr_schema {key} differs")
    return differences


def step_create_store(store_path, reference_path):
    if store_path.exists():
        print(f"step 1: {store_path} exists, not created again")
    else:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        create_empty_store(store_path)
        print(f"step 1: created empty store {store_path}")

    if not reference_path.exists():
        print(f"step 1: reference store {reference_path} not found, layout not compared")
        return
    differences = layout_differences(store_path, reference_path)
    if differences:
        raise RuntimeError("layout differs from the reference store:\n" + "\n".join(differences))
    print(f"step 1: layout equals {reference_path.name} (apart from the fields of the data array)")


# --- step 2: regrid mappings --------------------------------------------------------------


def read_source_coordinates(raw_dir, year):
    """Latitude and longitude of the source pixels of one yearly raw store."""
    store_path = raw_dir / f"roa_{year}.zarr"
    latitude = zarr.open_array(store_path / "latitude", mode="r")[:]
    longitude = zarr.open_array(store_path / "longitude", mode="r")[:]
    return latitude, longitude


def step_build_mappings(raw_dir, mapping_dir):
    mapping_dir.mkdir(parents=True, exist_ok=True)
    land_cells = None

    for year in YEARS:
        latitude, longitude = read_source_coordinates(raw_dir, year)
        fingerprint = grid_fingerprint(latitude, longitude)
        path = mapping_path(mapping_dir, fingerprint)

        if path.exists():
            mapping = RegridMapping.load(path)
        else:
            if land_cells is None:
                land_cells = np.asarray(get_spatial_grid(LAND_GRID_NAME).base_indices)
            mapping = build_mapping(latitude, longitude, land_cells)
            mapping.save(path)
            print(f"step 2: built {path.name}")

        n_cells = mapping.target_cells.size
        n_fallback = int(mapping.cell_is_fallback.sum())
        print(
            f"step 2: {year} source grid {fingerprint}: {n_cells} land cells with a value "
            f"({n_fallback} from the nearest pixel), {mapping.pair_pixel.size} pairs"
        )


def print_shard_range(store_path):
    """Tell which shards hold the hours of YEARS (the scans of a year reach into the next hour)."""
    hours_per_shard = zarr.open_array(store_path / "data", mode="r").shards[0]
    first_hour = np.datetime64(f"{YEARS[0]}-01-01T01", "h")
    last_hour = np.datetime64(f"{YEARS[-1] + 1}-01-01T00", "h")
    first_shard = int(time_index_of_hours(first_hour)) // hours_per_shard
    last_shard = int(time_index_of_hours(last_hour)) // hours_per_shard
    n_shards = last_shard - first_shard + 1
    print(f"hours {first_hour} to {last_hour} are in the shards {first_shard} to {last_shard} ({n_shards} shards)")


# --- step 3: shards -----------------------------------------------------------------------


def shard_marker_path(written_dir, shard):
    return written_dir / f"shard_{shard:05d}.npy"


def step_write_shards(store_path, raw_dir, mapping_dir, written_dir, first_shard, last_shard):
    written_dir.mkdir(parents=True, exist_ok=True)
    data = zarr.open_array(store_path / "data", mode="r+")
    hours_per_shard = data.shards[0]

    raw_years = open_raw_years(raw_dir, mapping_dir, YEARS)
    catalogue = build_catalogue(raw_years)

    for shard in range(first_shard, last_shard + 1):
        marker_path = shard_marker_path(written_dir, shard)
        if marker_path.exists():
            print(f"step 3: shard {shard} is already written, skipped")
            continue

        first_time_index = shard * hours_per_shard
        start_time = time.perf_counter()
        values, written_time_indices = shard_values(raw_years, catalogue, first_time_index, hours_per_shard)
        compute_seconds = time.perf_counter() - start_time

        start_time = time.perf_counter()
        if written_time_indices.size > 0:
            data[first_time_index : first_time_index + hours_per_shard] = values
        write_seconds = time.perf_counter() - start_time
        np.save(marker_path, written_time_indices)

        first_hour = np.datetime64(first_time_index, "h")
        print(
            f"step 3: shard {shard} (from {first_hour}): {written_time_indices.size} of {hours_per_shard} hours "
            f"with scans, computed in {compute_seconds:.0f} s, written in {write_seconds:.0f} s"
        )


# --- step 4: indices ----------------------------------------------------------------------


def step_register_indices(store_path, written_dir):
    """Register the hours of all finished shards as (0, time index) rows, as the product stores do."""
    marker_paths = sorted(written_dir.glob("shard_*.npy"))
    time_indices = np.concatenate([np.load(marker_path) for marker_path in marker_paths])
    rows = np.zeros((time_indices.size, 2), dtype=np.int64)
    rows[:, 1] = time_indices

    register_zarr_schema(MODALITY_ID, roa_schema())
    store = EOZarrStore(str(store_path), mode="r+", modality_id=MODALITY_ID)
    store.register_indices(rows)
    registered = store.consolidate_indices()
    print(f"step 4: {len(marker_paths)} finished shards, {registered.shape[0]} hours registered in _indices")


def main():
    args = parse_arguments()
    store_path = args.output_dir / f"{MODALITY_ID}.zarr"
    mapping_dir = args.output_dir / "mappings"
    written_dir = args.output_dir / "written_shards"

    if args.step == "setup":
        step_create_store(store_path, args.reference_store)
        step_build_mappings(args.raw_dir, mapping_dir)
        print_shard_range(store_path)
    elif args.step == "shards":
        if args.first_shard is None or args.last_shard is None:
            raise SystemExit("step shards needs --first_shard and --last_shard")
        step_write_shards(store_path, args.raw_dir, mapping_dir, written_dir, args.first_shard, args.last_shard)
    elif args.step == "indices":
        step_register_indices(store_path, written_dir)


if __name__ == "__main__":
    main()
