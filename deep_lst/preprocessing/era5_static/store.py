"""The static ERA5 fields as a product store (era5_static_product.zarr).

The ERA5 product store (era5_product.zarr) has no static fields. The store built here holds the
static fields of the older ERA5 store era5_2014_2025.eozarr, cell by cell and without any change,
on the grid of era5_product.zarr (0.25 degree, 721 x 1440). Runs that were trained with the
ERA5 static fields (tvh, tvl, lsm, slt) therefore read the values they were trained with.

Two things carry over from the old store:

- The fields exist for 394,899 of the 1,038,240 cells; the other cells hold the fill value.
  This is the coverage of the old store, not the land grid of the product stores.
- lsm is the land-sea mask truncated to an integer: 1 where the cell is entirely land, 0
  elsewhere, also for cells that are mostly land.

Class fields are int8 with the fill value -1, the other fields float32 with NaN.
"""

import numpy as np

MODALITY_ID = "era5_static_product"
GRID_NAME = "EquiAng_0.25deg"  # name of the grid in the old store
CHUNK_CELLS = (103, 144)  # (lat, lon) size of the inner chunks, as in era5_product.zarr

CLASS_FIELDS = ("tvh", "tvl", "lsm", "slt")
CONTINUOUS_FIELDS = ("cl", "dl", "cvl", "cvh", "z", "sdfor", "sdor", "isor", "anor", "slor")
CLASS_FILL = -1

STORE_DTYPE = np.dtype([(name, "i1") for name in CLASS_FIELDS] + [(name, "<f4") for name in CONTINUOUS_FIELDS])
STORE_FILL_VALUE = (CLASS_FILL,) * len(CLASS_FIELDS) + (np.nan,) * len(CONTINUOUS_FIELDS)


def flatten_fields(old_values):
    """Fields of a static array of the old store as plain columns.

    Args:
        old_values: Struct array (n_cells,) whose fields have the shape (1, 1), as the old
            store holds them.

    Returns:
        {field name: array (n_cells,)}
    """
    columns = {}
    for name in old_values.dtype.names:
        columns[name] = old_values[name].reshape(old_values.shape[0])
    return columns


def check_grid(spatial_idx, centers, lat, lon):
    """Raise unless the cells of the old store are the cells of the target grid in row order.

    Args:
        spatial_idx: Cell index of every entry of the old store. (n_cells,)
        centers: Lon, lat of the cell centres in the old store. (n_cells, 2)
        lat: Latitudes of the target grid, north to south.
        lon: Longitudes of the target grid, west to east.
    """
    n_cells = lat.size * lon.size
    if not np.array_equal(spatial_idx, np.arange(n_cells)):
        raise ValueError("The old store does not hold the cells 0 .. n_cells - 1 in order")
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    same_lon = np.array_equal(centers[:, 0], lon_grid.ravel())
    same_lat = np.array_equal(centers[:, 1], lat_grid.ravel())
    if not (same_lon and same_lat):
        raise ValueError("The cell centres of the old store are not those of the target grid")


def cells_to_grid(columns, n_lat, n_lon):
    """Columns per cell, in row order -> struct array (n_lat, n_lon) of the store dtype."""
    grid_values = np.empty((n_lat, n_lon), dtype=STORE_DTYPE)
    for name in STORE_DTYPE.names:
        grid_values[name] = columns[name].reshape(n_lat, n_lon)
    return grid_values
