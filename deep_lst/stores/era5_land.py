"""Readers of the ERA5-Land stores (0.1 degree, values in the units of ERA5-Land)."""

from deep_lst.stores.cube import CubeReader
from deep_lst.stores.time_axis import T1H


class Era5LandReader(CubeReader):
    """Hourly fields."""

    MODALITY_ID = "era5_land_product"
    TIME_GRID = T1H


class Era5LandStaticReader(CubeReader):
    """Static fields: vegetation and soil types, land-sea mask, lake cover."""

    MODALITY_ID = "era5_land_static_product"
