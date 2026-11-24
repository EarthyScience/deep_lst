"""Reader of the daily ASCAT soil moisture store (0.25 degree, the ERA5 grid).

The store holds the EUMETSAT ASCAT Level 2 surface soil moisture, merged per day and pass:
every variable exists once for the ascending pass (suffix _asc, about 21:30 local time) and once
for the descending pass (_des, about 09:30 local time). sm is the degree of saturation in
percent (0 to 100).

The store is right labeled: the value at day D was observed on day D - 1.
"""

from deep_lst.stores.cube import CubeReader
from deep_lst.stores.time_axis import T1D


class AscatReader(CubeReader):
    MODALITY_ID = "ascat_smo_product"
    TIME_GRID = T1D
