"""Every reader of deep_lst.stores opens and reads its real store.

The stores are looked up under `product_root` of conf/paths.yaml; a reader is skipped where
its store is not there. Values are compared with the old pipeline in test_stores_parity.py;
this file covers what every store must satisfy, including the stores the old pipeline never
read (ASCAT, Rain over Africa, the ERA5-Land static fields).
"""

import numpy as np
import pytest

from deep_lst.stores import (
    AscatReader,
    Era5LandReader,
    Era5LandStaticReader,
    Era5Reader,
    GlofasReader,
    RoaReader,
    SeviriDEMReader,
    SeviriFVCReader,
    SeviriLANDCOVReader,
    SeviriLSTReader,
    SeviriLWMASKReader,
)
from deep_lst.utils.io import ExperimentPaths

pytest.importorskip("zarr", minversion="3.4.2")

READERS = [
    AscatReader,
    Era5LandReader,
    Era5LandStaticReader,
    Era5Reader,
    GlofasReader,
    RoaReader,
    SeviriDEMReader,
    SeviriFVCReader,
    SeviriLANDCOVReader,
    SeviriLSTReader,
    SeviriLWMASKReader,
]

# lon, lat of a cell that is land in every store (near Niamey)
LAND_POINT = (2.1, 13.5)


def open_reader(reader_class):
    store = ExperimentPaths().product_store_path(reader_class.MODALITY_ID)
    if not (store / "data" / "zarr.json").exists():
        pytest.skip(f"{store} is not available")
    return reader_class(store)


@pytest.mark.parametrize("reader_class", READERS, ids=[reader_class.__name__ for reader_class in READERS])
def test_reader_opens_its_store_and_reads_a_land_cell(reader_class):
    reader = open_reader(reader_class)

    n_steps, n_lat, n_lon = reader.data_arr.shape
    assert reader.spatial_grid.shape == (n_lat, n_lon)
    expected_steps = 1 if reader.is_static else len(reader.time_grid)
    assert n_steps == expected_steps

    written_steps = reader.available_indices[:, 1]
    assert written_steps.size > 0
    assert written_steps.min() >= 0
    assert written_steps.max() < n_steps

    row = reader.spatial_grid.nearest_rows(*LAND_POINT)
    last_written = int(written_steps.max())
    first_step = max(last_written - 9, 0)
    series = reader.get_data_series([row], first_step, last_written + 1)
    assert series.shape == (1, last_written + 1 - first_step, len(reader.variables))
    assert not np.isnan(series).all(), "no value at a land cell in the last written steps"
