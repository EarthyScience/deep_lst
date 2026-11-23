"""Reader of the ERA5 store (0.25 degree, hourly, values in the units of ERA5)."""

from deep_lst.stores.cube import CubeReader
from deep_lst.stores.time_axis import T1H


class Era5Reader(CubeReader):
    MODALITY_ID = "era5_product"
    TIME_GRID = T1H
