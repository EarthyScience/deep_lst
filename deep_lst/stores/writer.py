"""Writer of a store in the layout of the product stores.

A product store is a zarr v3 group with five arrays:

    data      (time, lat, lon)  struct dtype, one field per variable; one shard file holds
                                steps_per_shard time steps of the whole grid
    lat       (n_lat,)          float64, north to south
    lon       (n_lon,)          float64, west to east
    time_idx  (time,)           datetime64[ns], start of every step of the time axis
    _indices  (n_written, 2)    int64, one row (0, time index) per written time step

The time dimension spans the whole time axis (1970 to 2040); only the shards that were written
exist as files. A static store has one time step. Array metadata and codecs are those of the
stores in data/deep_lst as read on 2026-11-23; the root attributes follow EOZarr_v2.1.

Writing sharded struct arrays needs zarr >= 3.4.2.
"""

import numpy as np
import zarr
from zarr.codecs import BloscCodec, Crc32cCodec

STATIC_TIME = np.array(["1970-01-01T00"], dtype="datetime64[ns]")  # the one time step of a static store
KEY_ENCODING = {"name": "default", "separator": "."}  # shard files are named c.<time>.<lat>.<lon>


def shard_size(n_cells, chunk_cells):
    """Smallest multiple of the inner chunk size that covers the grid along one dimension."""
    n_chunks = -(-n_cells // chunk_cells)  # rounded up
    return n_chunks * chunk_cells


def write_product_store(
    path,
    modality_id,
    time_axis,
    lat,
    lon,
    steps_per_shard,
    chunk_cells,
    written_steps,
    values,
    fill_value,
    spatial_grid_name,
):
    """Write a store in the layout of the product stores; an existing store is replaced.

    Args:
        path: Folder of the store (<name>.zarr).
        modality_id: Name of the store's content, written to the root attributes.
        time_axis: TimeAxis of the store; None writes a static store with one time step.
        lat: Latitudes of the cell centres, north to south.
        lon: Longitudes of the cell centres, west to east.
        steps_per_shard: Time steps in one shard file.
        chunk_cells: (lat, lon) size of the inner chunks of a shard.
        written_steps: Time indices that receive values.
        values: Struct array (len(written_steps), n_lat, n_lon).
        fill_value: One fill value per field.
        spatial_grid_name: Name of the grid, written to the root attributes.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    written_steps = np.asarray(written_steps, dtype=np.int64)
    if time_axis is None:
        time_values = STATIC_TIME
        time_grid_name = "TSTATIC"
    else:
        all_steps = np.arange(len(time_axis))
        time_values = time_axis.index_to_startdate(all_steps).tz_localize(None).to_numpy()
        time_grid_name = time_axis.name
    n_steps = len(time_values)

    attributes = {
        "EOZarr_version": "EOZarr_v2.1",
        "modality_id": modality_id,
        "spatial_grid": spatial_grid_name,
        "time_grid": time_grid_name,
    }
    root = zarr.open_group(str(path), mode="w", zarr_format=3, attributes=attributes)

    shard_lat = shard_size(lat.size, chunk_cells[0])
    shard_lon = shard_size(lon.size, chunk_cells[1])
    data = root.create_array(
        "data",
        shape=(n_steps, lat.size, lon.size),
        dtype=values.dtype,
        shards=(steps_per_shard, shard_lat, shard_lon),
        chunks=(steps_per_shard, chunk_cells[0], chunk_cells[1]),
        fill_value=np.array(fill_value, dtype=values.dtype)[()],
        compressors=[BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"), Crc32cCodec()],
        dimension_names=["time_idx", "lat", "lon"],
        chunk_key_encoding=KEY_ENCODING,
    )
    for position, step in enumerate(written_steps):
        data[int(step)] = values[position]

    for name, coordinate in (("lat", lat), ("lon", lon)):
        array = root.create_array(
            name,
            shape=coordinate.shape,
            dtype="float64",
            chunks=coordinate.shape,
            fill_value=np.nan,
            dimension_names=[name],
            chunk_key_encoding=KEY_ENCODING,
        )
        array[:] = coordinate

    time_idx = root.create_array(
        "time_idx",
        shape=(n_steps,),
        dtype="datetime64[ns]",
        chunks=(n_steps,),
        dimension_names=["time_idx"],
        chunk_key_encoding=KEY_ENCODING,
    )
    time_idx[:] = time_values

    index_rows = np.zeros((len(written_steps), 2), dtype=np.int64)
    index_rows[:, 1] = written_steps
    indices = root.create_array(
        "_indices",
        shape=index_rows.shape,
        dtype="int64",
        chunks=(100000, 2),
        fill_value=-1,
        compressors=None,
        dimension_names=["sample", "axis"],
        chunk_key_encoding=KEY_ENCODING,
    )
    indices[:] = index_rows
