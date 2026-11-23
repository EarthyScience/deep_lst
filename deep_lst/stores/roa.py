"""Reader of the Rain over Africa store (hourly rain on the SEVIRI grid).

The store is written by deep_lst/preprocessing/rain_over_africa. The value at an hour is the
rain in mm that fell in the hour before it.
"""

from deep_lst.preprocessing.rain_over_africa.aggregation import RAIN_RATE_SCALE
from deep_lst.stores.cube import CubeReader
from deep_lst.stores.time_axis import T1H


class RoaReader(CubeReader):
    MODALITY_ID = "roa_land_expected_hourly_acc"
    TIME_GRID = T1H
    SCALE = {"rain_rate": RAIN_RATE_SCALE}
