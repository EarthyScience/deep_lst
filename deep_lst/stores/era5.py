"""Readers of the ERA5 stores (0.25 degree, values in the units of ERA5)."""

from deep_lst.stores.cube import CubeReader
from deep_lst.stores.time_axis import T1H


class Era5Reader(CubeReader):
    """Hourly fields."""

    MODALITY_ID = "era5_product"
    TIME_GRID = T1H


class Era5StaticReader(CubeReader):
    """Static fields, copied from the older ERA5 store (deep_lst/preprocessing/era5_static)."""

    MODALITY_ID = "era5_static_product"
