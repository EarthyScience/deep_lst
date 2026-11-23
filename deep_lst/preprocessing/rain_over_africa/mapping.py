"""Mapping from the Rain over Africa (RoA) grid to the SEVIRI 0.05 degree product grid.

RoA is delivered on the geostationary SEVIRI grid (about 3 km at the sub-satellite point, 2-D
latitude / longitude). The product stores of deep_lst are on a regular 0.05 degree grid and
hold values only on the cells of the SEVIRI land grid. The mapping is a list of
(target cell, source pixel) pairs:

1. Every source pixel whose centre falls into a land cell is paired with that cell. The cell
   value is the mean of its pixels (2.3 pixels per cell on average).
2. A land cell without a pixel centre takes the nearest source pixel, if that pixel is not
   farther away than MAX_FALLBACK_SPACINGS times the local pixel spacing. This fills the cells
   where the source pixels are larger than a cell (towards the edge of the disk) and leaves
   the cells outside the RoA domain empty.

The source grid is not the same in all years (2016-2017 and 2018-2025 differ by 1.5 km), so a
mapping belongs to one source grid, identified by grid_fingerprint.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0
MAX_FALLBACK_SPACINGS = 0.71  # about half the diagonal of a square pixel
SEARCH_LIMIT_KM = 50.0  # cells farther than this from every pixel are never filled


@dataclass(frozen=True)
class TargetGrid:
    """Regular latitude / longitude grid, cells numbered row by row from the north-west."""

    lat_north: float  # latitude of the first row (cell centres)
    lon_west: float  # longitude of the first column (cell centres)
    resolution: float  # degrees
    n_lat: int
    n_lon: int

    @property
    def shape(self):
        return (self.n_lat, self.n_lon)

    @property
    def n_cells(self):
        return self.n_lat * self.n_lon

    def cell_of(self, latitude, longitude):
        """Number of the cell that contains each point, -1 for points outside the grid."""
        rows = np.rint((self.lat_north - latitude) / self.resolution).astype(np.int64)
        columns = np.rint((longitude - self.lon_west) / self.resolution).astype(np.int64)
        inside = (rows >= 0) & (rows < self.n_lat) & (columns >= 0) & (columns < self.n_lon)
        cells = rows * self.n_lon + columns
        cells[~inside] = -1
        return cells

    def cell_centres(self, cells):
        """Latitude and longitude of the centre of each cell."""
        rows = np.asarray(cells) // self.n_lon
        columns = np.asarray(cells) % self.n_lon
        latitude = self.lat_north - rows * self.resolution
        longitude = self.lon_west + columns * self.resolution
        return latitude, longitude


SEVIRI_PRODUCT_GRID = TargetGrid(lat_north=80.0, lon_west=-80.0, resolution=0.05, n_lat=3201, n_lon=3201)


@dataclass(frozen=True)
class RegridMapping:
    """Pairs of (target cell, source pixel) for one source grid.

    Attributes:
        source_shape: Shape (n_y, n_x) of the source grid.
        source_fingerprint: grid_fingerprint of the source latitude / longitude.
        target_shape: Shape (n_lat, n_lon) of the target grid.
        target_cells: Numbers of the target cells that receive a value, sorted. (n_cells,)
        cell_is_fallback: True for the cells filled from the nearest pixel. (n_cells,)
        pair_cell: Position in target_cells of every pair. (n_pairs,)
        pair_pixel: Flat index into the source grid of every pair. (n_pairs,)
    """

    source_shape: tuple
    source_fingerprint: str
    target_shape: tuple
    target_cells: np.ndarray
    cell_is_fallback: np.ndarray
    pair_cell: np.ndarray
    pair_pixel: np.ndarray

    def save(self, path):
        np.savez(
            path,
            source_shape=np.asarray(self.source_shape),
            source_fingerprint=np.asarray(self.source_fingerprint),
            target_shape=np.asarray(self.target_shape),
            target_cells=self.target_cells,
            cell_is_fallback=self.cell_is_fallback,
            pair_cell=self.pair_cell,
            pair_pixel=self.pair_pixel,
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as stored:
            return cls(
                source_shape=tuple(int(size) for size in stored["source_shape"]),
                source_fingerprint=str(stored["source_fingerprint"]),
                target_shape=tuple(int(size) for size in stored["target_shape"]),
                target_cells=stored["target_cells"],
                cell_is_fallback=stored["cell_is_fallback"],
                pair_cell=stored["pair_cell"],
                pair_pixel=stored["pair_pixel"],
            )


def grid_fingerprint(latitude, longitude):
    """Short hash of the source coordinates; two years share a mapping if it is equal."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(latitude, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(longitude, dtype=np.float32).tobytes())
    return digest.hexdigest()[:16]


def unit_vectors(latitude, longitude):
    """Points on the unit sphere for latitudes / longitudes in degrees. Shape (..., 3)."""
    latitude_rad = np.radians(latitude)
    longitude_rad = np.radians(longitude)
    x = np.cos(latitude_rad) * np.cos(longitude_rad)
    y = np.cos(latitude_rad) * np.sin(longitude_rad)
    z = np.sin(latitude_rad)
    return np.stack([x, y, z], axis=-1)


def chord_to_km(chord):
    """Great-circle distance in km for a straight-line distance between unit vectors."""
    half_chord = np.clip(np.asarray(chord) / 2.0, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(half_chord)


def km_to_chord(distance_km):
    """Straight-line distance between unit vectors for a great-circle distance in km."""
    return 2.0 * np.sin(distance_km / EARTH_RADIUS_KM / 2.0)


def pixel_spacing_km(latitude, longitude):
    """Distance from every source pixel to its neighbour, the larger of the x and y direction.

    The last column / row uses the distance to the previous one. NaN where a pixel or all its
    neighbours have no coordinates.
    """
    vectors = unit_vectors(latitude, longitude)

    spacing_x = np.full(vectors.shape[:2], np.nan)
    step_x = chord_to_km(np.linalg.norm(vectors[:, 1:] - vectors[:, :-1], axis=-1))
    spacing_x[:, :-1] = step_x
    spacing_x[:, -1] = step_x[:, -1]

    spacing_y = np.full(vectors.shape[:2], np.nan)
    step_y = chord_to_km(np.linalg.norm(vectors[1:, :] - vectors[:-1, :], axis=-1))
    spacing_y[:-1, :] = step_y
    spacing_y[-1, :] = step_y[-1, :]

    return np.fmax(spacing_x, spacing_y)


def build_mapping(latitude, longitude, land_cells, target_grid=SEVIRI_PRODUCT_GRID):
    """Build the (target cell, source pixel) pairs for one source grid.

    Args:
        latitude: Latitude of every source pixel in degrees, NaN off the disk. (n_y, n_x)
        longitude: Longitude of every source pixel in degrees, same shape.
        land_cells: Numbers of the target cells that may receive a value (the land grid).
        target_grid: The regular grid the cell numbers refer to.

    Returns:
        The RegridMapping of this source grid.
    """
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    land_cells = np.asarray(land_cells, dtype=np.int64)

    has_coordinates = np.isfinite(latitude) & np.isfinite(longitude)
    pixel_index = np.flatnonzero(has_coordinates.ravel())
    pixel_latitude = latitude.ravel()[pixel_index]
    pixel_longitude = longitude.ravel()[pixel_index]

    # 1. pixels whose centre falls into a land cell
    is_land = np.zeros(target_grid.n_cells, dtype=bool)
    is_land[land_cells] = True
    pixel_cell = target_grid.cell_of(pixel_latitude, pixel_longitude)
    in_grid = pixel_cell >= 0
    on_land = np.zeros(pixel_cell.shape, dtype=bool)
    on_land[in_grid] = is_land[pixel_cell[in_grid]]
    direct_cell = pixel_cell[on_land]
    direct_pixel = pixel_index[on_land]

    # 2. land cells without a pixel centre take the nearest pixel, if it is close enough
    has_pixel = np.zeros(target_grid.n_cells, dtype=bool)
    has_pixel[direct_cell] = True
    empty_cells = land_cells[~has_pixel[land_cells]]
    empty_latitude, empty_longitude = target_grid.cell_centres(empty_cells)

    tree = cKDTree(unit_vectors(pixel_latitude, pixel_longitude))
    chord, nearest = tree.query(
        unit_vectors(empty_latitude, empty_longitude),
        distance_upper_bound=km_to_chord(SEARCH_LIMIT_KM),
    )
    found = np.isfinite(chord)  # the query returns inf beyond the search limit
    nearest_pixel = pixel_index[nearest[found]]
    distance = chord_to_km(chord[found])
    spacing = pixel_spacing_km(latitude, longitude).ravel()[nearest_pixel]
    close_enough = distance <= MAX_FALLBACK_SPACINGS * spacing
    fallback_cell = empty_cells[found][close_enough]
    fallback_pixel = nearest_pixel[close_enough]

    # 3. number the cells that receive a value
    all_cells = np.concatenate([direct_cell, fallback_cell])
    all_pixels = np.concatenate([direct_pixel, fallback_pixel])
    target_cells, pair_cell = np.unique(all_cells, return_inverse=True)
    cell_is_fallback = np.isin(target_cells, fallback_cell)

    return RegridMapping(
        source_shape=tuple(latitude.shape),
        source_fingerprint=grid_fingerprint(latitude, longitude),
        target_shape=target_grid.shape,
        target_cells=target_cells,
        cell_is_fallback=cell_is_fallback,
        pair_cell=pair_cell.astype(np.int64),
        pair_pixel=all_pixels.astype(np.int64),
    )


def regrid_scan(clean_rates, valid, mapping):
    """Average one scan on the source grid to the target cells of a mapping.

    Args:
        clean_rates: Rain rates on the source grid, invalid values already set to 0. (n_y, n_x)
        valid: True where the source pixel has a valid rate, same shape.
        mapping: The RegridMapping of this source grid.

    Returns:
        cell_rates: Mean of the valid pixels of every target cell, 0 where none is valid.
            float32, shape (n_cells,), in the order of mapping.target_cells.
        cell_valid: True where at least one pixel of the cell is valid, same shape.
    """
    if tuple(clean_rates.shape) != mapping.source_shape:
        raise ValueError(f"scan has shape {clean_rates.shape}, the mapping expects {mapping.source_shape}")

    pair_valid = np.asarray(valid).ravel()[mapping.pair_pixel]
    pair_rates = np.asarray(clean_rates).ravel()[mapping.pair_pixel]
    pair_rates = np.where(pair_valid, pair_rates, 0.0)

    n_cells = mapping.target_cells.size
    rate_sum = np.bincount(mapping.pair_cell, weights=pair_rates, minlength=n_cells)
    valid_count = np.bincount(mapping.pair_cell, weights=pair_valid, minlength=n_cells)

    cell_valid = valid_count > 0
    cell_rates = np.zeros(n_cells, dtype=np.float32)
    cell_rates[cell_valid] = rate_sum[cell_valid] / valid_count[cell_valid]
    return cell_rates, cell_valid


def cells_to_grid(cell_values, mapping, fill_value):
    """Place the values of the target cells on the full target grid.

    Args:
        cell_values: One value per target cell, in the order of mapping.target_cells.
        mapping: The RegridMapping the values belong to.
        fill_value: Value of all other cells.

    Returns:
        Array of shape mapping.target_shape with the dtype of cell_values.
    """
    cell_values = np.asarray(cell_values)
    grid = np.full(mapping.target_shape[0] * mapping.target_shape[1], fill_value, dtype=cell_values.dtype)
    grid[mapping.target_cells] = cell_values
    return grid.reshape(mapping.target_shape)
