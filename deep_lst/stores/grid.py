"""Regular latitude-longitude grid of a product store.

A product store holds its values as data[time, lat, lon], with the coordinate arrays lat (north
to south) and lon (west to east). deep_lst addresses a cell by its row, the position in the
flattened grid:

    row = lat_index * n_lon + lon_index

This is the cell index of the eodata equiangular grids that LatLonGrid replaces, so the store
rows in existing index files and preload caches keep their meaning.
"""

from functools import cached_property

import numpy as np


def regular_step(values, axis_name):
    """Step of a coordinate array; negative for a descending array.

    Raises:
        ValueError: The array is not evenly spaced.
    """
    step = (values[-1] - values[0]) / (values.size - 1)
    differences = np.diff(values)
    tolerance = abs(step) * 1e-6
    if not np.allclose(differences, step, rtol=0.0, atol=tolerance):
        raise ValueError(f"The {axis_name} axis is not evenly spaced")
    return float(step)


class LatLonGrid:
    """Grid given by the lat and lon arrays of a store.

    Attributes:
        lat: Latitudes of the cell centres, north to south. (n_lat,)
        lon: Longitudes of the cell centres, west to east. (n_lon,)
        name: Name of the grid; two grids with the same name are the same grid.
        lat_step: Spacing of lat in degrees (negative).
        lon_step: Spacing of lon in degrees.
    """

    def __init__(self, lat, lon, name):
        self.lat = np.asarray(lat, dtype=np.float64)
        self.lon = np.asarray(lon, dtype=np.float64)
        self.name = name
        self.lat_step = regular_step(self.lat, "lat")
        self.lon_step = regular_step(self.lon, "lon")

    @property
    def shape(self):
        """(n_lat, n_lon)."""
        return (self.lat.size, self.lon.size)

    def __len__(self):
        return self.lat.size * self.lon.size

    def rows_to_lat_lon_index(self, rows):
        """Rows -> (lat_index, lon_index), the positions along the lat and lon dimensions."""
        rows = np.asarray(rows, dtype=np.int64)
        lat_index, lon_index = np.divmod(rows, self.lon.size)
        return lat_index, lon_index

    def lat_lon_index_to_rows(self, lat_index, lon_index):
        """(lat_index, lon_index) -> rows."""
        lat_index = np.asarray(lat_index, dtype=np.int64)
        lon_index = np.asarray(lon_index, dtype=np.int64)
        return lat_index * self.lon.size + lon_index

    @cached_property
    def centers(self):
        """Lon, lat of every cell centre in row order. (n_cells, 2)

        Built on first use and kept: 16 bytes per cell, 164 MB for the SEVIRI grid.
        """
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        return np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)

    def nearest_rows(self, lon, lat):
        """Row of the cell whose centre is nearest to each point.

        A point outside the grid gets the nearest cell on the edge. Longitudes are not wrapped
        around the date line.

        Args:
            lon: Longitudes of the points in degrees.
            lat: Latitudes of the points in degrees, same shape as lon.

        Returns:
            The rows, same shape as lon (int64).
        """
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        lat_position = (lat - self.lat[0]) / self.lat_step
        lon_position = (lon - self.lon[0]) / self.lon_step
        lat_index = np.clip(np.rint(lat_position), 0, self.lat.size - 1)
        lon_index = np.clip(np.rint(lon_position), 0, self.lon.size - 1)
        return self.lat_lon_index_to_rows(lat_index, lon_index)
