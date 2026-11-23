"""Readers of the SEVIRI stores (LSA SAF products on the 0.05 degree grid of the SEVIRI disk).

The stores hold the integers of the source files. Scale factors are those of the LSA SAF
products, as listed in the README next to the stores.
"""

from deep_lst.stores.cube import CubeReader
from deep_lst.stores.time_axis import T1D, T1H


class SeviriLSTReader(CubeReader):
    """Hourly land surface temperature. LST is stored in 0.01 degree Celsius and returned in Kelvin."""

    MODALITY_ID = "seviri_product_hourly_lst"
    TIME_GRID = T1H
    SCALE = {"LST": 0.01, "standard_error": 0.01}
    OFFSET = {"LST": 273.15}


class SeviriFVCReader(CubeReader):
    """Daily fraction of vegetation cover (0 to 1)."""

    MODALITY_ID = "seviri_product_daily_mdfvc"
    TIME_GRID = T1D
    SCALE = {"FVC": 0.0001, "standard_error": 0.0001}


class SeviriDEMReader(CubeReader):
    """Elevation in metres."""

    MODALITY_ID = "seviri_product_static_dem"


class SeviriLWMASKReader(CubeReader):
    """Land/water mask classes."""

    MODALITY_ID = "seviri_product_static_lwmask"


class SeviriLANDCOVReader(CubeReader):
    """Land cover classes."""

    MODALITY_ID = "seviri_product_static_landcov"
