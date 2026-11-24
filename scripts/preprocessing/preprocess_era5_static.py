"""Copy the static ERA5 fields of the old ERA5 store into era5_static_product.zarr.

The store has the layout of the static product stores (one time step, data[0, lat, lon]) on the
grid of era5_product.zarr; deep_lst/preprocessing/era5_static/store.py describes its content.

The old store can only be read with eodata v0.3_DeepLST (zarr below 3.3), the product layout
can only be written with zarr 3.4.2 or newer. The work is therefore split into two calls that
run in different environments:

  export   Read the static arrays of the old store and save them to a .npz file.
           Needs eodata v0.3_DeepLST: the environment of eodata_slim, or the deep_lst
           environment as it was before eodata was removed from it.
  write    Check the grid, write the store from the .npz file and read it back.
           Needs zarr >= 3.4.2: the deep_lst environment.

    <python with eodata> scripts/preprocessing/preprocess_era5_static.py export
    <python of deep_lst> scripts/preprocessing/preprocess_era5_static.py write
"""

import argparse
from pathlib import Path

import numpy as np

from deep_lst.preprocessing.era5_static.store import (
    CHUNK_CELLS,
    CLASS_FIELDS,
    CONTINUOUS_FIELDS,
    GRID_NAME,
    MODALITY_ID,
    STORE_FILL_VALUE,
    cells_to_grid,
    check_grid,
    flatten_fields,
)

OLD_STORE = Path("/Net/Groups/BGI/work_5/scratch/WeatherGenerator/data/lowres_old/era5_2014_2025.eozarr.zip")
EXPORT_FILE = Path("/Net/Groups/BGI/tscratch/mwessel/data/era5_static_product/era5_static_old_store.npz")
OUTPUT_DIR = Path("/Net/Groups/BGI/work_5/scratch/WeatherGenerator/data/deep_lst")
GRID_STORE = "era5_product.zarr"  # store in the output folder whose lat and lon the new store takes


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["export", "write"])
    parser.add_argument("--old_store", type=Path, default=OLD_STORE, help="the old ERA5 store (step export)")
    parser.add_argument("--export_file", type=Path, default=EXPORT_FILE, help=".npz written by export, read by write")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="folder of the product stores (step write)")
    return parser.parse_args()


def step_export(old_store, export_file):
    """Save the static fields, the cell indices and the cell centres of the old store."""
    import eodata
    import zarr

    reader = eodata.get_data_reader(
        "era5", store=str(old_store), average_hourly=False, get_data_static=True, get_data_static_categorical=True
    )
    columns = flatten_fields(reader.data_static_categorical_arr[:])
    columns.update(flatten_fields(reader.data_static_arr[:]))
    missing = [name for name in CLASS_FIELDS + CONTINUOUS_FIELDS if name not in columns]
    if missing:
        raise ValueError(f"The old store has no static field {missing}")

    old_root = zarr.open_group(reader.data_static_arr.store, mode="r")
    export_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(export_file, spatial_idx=old_root["spatial_idx"][:], centers=old_root["centers"][:], **columns)
    n_cells = columns[CLASS_FIELDS[0]].size
    print(f"exported {len(columns)} fields of {n_cells} cells from {old_store} to {export_file}")


def step_write(export_file, output_dir):
    """Write era5_static_product.zarr from the exported fields and compare it with them."""
    import zarr

    from deep_lst.stores.writer import write_product_store

    exported = np.load(export_file)
    grid_store = output_dir / GRID_STORE
    lat = zarr.open_array(str(grid_store / "lat"), mode="r")[:]
    lon = zarr.open_array(str(grid_store / "lon"), mode="r")[:]
    check_grid(exported["spatial_idx"], exported["centers"], lat, lon)

    grid_values = cells_to_grid(exported, lat.size, lon.size)
    store_path = output_dir / f"{MODALITY_ID}.zarr"
    write_product_store(
        store_path,
        MODALITY_ID,
        time_axis=None,
        lat=lat,
        lon=lon,
        steps_per_shard=1,
        chunk_cells=CHUNK_CELLS,
        written_steps=[0],
        values=grid_values[None],
        fill_value=STORE_FILL_VALUE,
        spatial_grid_name=GRID_NAME,
    )

    stored = zarr.open_array(str(store_path / "data"), mode="r")[0]
    for name in stored.dtype.names:
        equal_nan = stored[name].dtype.kind == "f"
        if not np.array_equal(stored[name], grid_values[name], equal_nan=equal_nan):
            raise ValueError(f"The field {name} read back from {store_path} differs from the export")
    n_with_values = int((stored[CLASS_FIELDS[0]] != STORE_FILL_VALUE[0]).sum())
    print(f"wrote {store_path}: {len(stored.dtype.names)} fields, {n_with_values} of {stored.size} cells with values")


def main():
    args = parse_arguments()
    if args.step == "export":
        step_export(args.old_store, args.export_file)
    else:
        step_write(args.export_file, args.output_dir)


if __name__ == "__main__":
    main()
