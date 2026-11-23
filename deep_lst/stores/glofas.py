"""Reader of the GloFAS v4 store (0.05 degree, daily)."""

from deep_lst.stores.cube import CubeReader
from deep_lst.stores.time_axis import T1D


class GlofasReader(CubeReader):
    MODALITY_ID = "glofas_v4_product"
    TIME_GRID = T1D
