"""The new readers return what the eodata readers returned.

Compares the readers of deep_lst.stores on the real product stores with the golden fixtures
of tests/fixtures/readers_v1, which hold what the old pipeline read for ten anchors and four
windows of four days (scripts/assets/export_reader_fixtures.py).

The stores are looked up under `product_root` of conf/paths.yaml; a case is skipped where its
store is not there. Sharded stores need zarr >= 3.4.2.

What is compared: wherever the old pipeline had a value, the new reader has the same value.
The exception are the fixture pixels that sit on a cell outside the land grid of a new store
(MASKED_PIXELS): there the new store holds nothing. The new store may hold values where the old
one had none (the old SEVIRI store has no data on 2023-12-30 and 2023-12-31).

The ERA5 static fields are compared with era5_static_product, which was copied from the old
ERA5 store and is not masked to a land grid. Not compared, because the new stores do not hold
them: SEVIRI MLE and MH.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from deep_lst.stores import (
    Era5LandReader,
    Era5Reader,
    Era5StaticReader,
    GlofasReader,
    SeviriDEMReader,
    SeviriFVCReader,
    SeviriLANDCOVReader,
    SeviriLSTReader,
    SeviriLWMASKReader,
)
from deep_lst.utils.full_store import load_store_columns
from deep_lst.utils.io import ExperimentPaths

pytest.importorskip("zarr", minversion="3.4.2")

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "readers_v1"
META = json.loads((FIXTURE_DIR / "meta.json").read_text())
PIXELS = list(META["pixels"])
N_DAYS = META["n_days"]

# Fixture pixels whose cell is not on the land grid of the stores of a modality.
MASKED_PIXELS = {
    "seviri": ["lake_victoria", "south_coast", "durban_coast"],
    "era5": ["lake_victoria", "south_coast"],
    "era5_land": ["lake_victoria", "dakar_coast", "durban_coast"],
    "glofas_v4": ["lake_victoria", "durban_coast"],
}

# Stores that are not masked to a land grid: they hold values at the pixels above as well.
UNMASKED_STORES = ["era5_static_product"]

# The old pipeline marked a missing class with -1 instead of NaN.
OLD_MISSING_CLASS = -1

# Largest accepted difference of a variable, as a share of its largest value in the fixture.
# The ERA5 fields of the old and the new store differ in the last digits of float32 (for
# example 0.25 J/m2 in ssrd at 2.9e6 J/m2); everything else is equal to float32 rounding.
RELATIVE_TOLERANCE = 1e-6

# (modality of the fixture, data type of the fixture, reader of the new store)
CASES = [
    ("seviri", "data", SeviriLSTReader),
    ("seviri", "metadata", SeviriLSTReader),
    ("seviri", "data_daily", SeviriFVCReader),
    ("seviri", "data_static", SeviriDEMReader),
    ("seviri", "data_static_categorical", SeviriLWMASKReader),
    ("seviri", "data_static_categorical", SeviriLANDCOVReader),
    ("era5", "data", Era5Reader),
    ("era5", "data_static_categorical", Era5StaticReader),
    ("era5_land", "data", Era5LandReader),
    ("glofas_v4", "data", GlofasReader),
]
CASE_IDS = [f"{modality}.{data_type}-{reader_class.__name__}" for modality, data_type, reader_class in CASES]


def open_reader(reader_class):
    store = ExperimentPaths().product_store_path(reader_class.MODALITY_ID)
    if not (store / "data" / "zarr.json").exists():
        pytest.skip(f"{store} is not available")
    return reader_class(store)


def read_windows(reader, store_rows, start_days):
    """What the reader has for the fixture pixels, in the layout of the fixture.

    Returns:
        Static reader: (P, C). Otherwise (W, P, H, C) with H = N_DAYS * 24 hours; the values of
        a daily store are repeated over the 24 hours of their day.
    """
    if reader.is_static:
        return reader.get_data_series(store_rows, 0, 1)[:, 0]

    steps_per_day = 24 if reader.time_grid.name == "T1H" else 1
    n_steps = N_DAYS * steps_per_day
    windows = []
    for start_day in start_days:
        first_step = int(start_day) * steps_per_day
        values = reader.get_data_series(store_rows, first_step, first_step + n_steps)
        if steps_per_day == 1:
            values = np.repeat(values, 24, axis=1)
        windows.append(values)
    return np.stack(windows)


def pixels_first(values):
    """(P,) or (W, P, H) -> (P, n) with one row per pixel."""
    pixel_axis = 0 if values.ndim == 1 else 1
    return np.moveaxis(values, pixel_axis, 0).reshape(len(PIXELS), -1)


@pytest.mark.parametrize(("modality", "data_type", "reader_class"), CASES, ids=CASE_IDS)
def test_reader_returns_the_values_of_the_old_pipeline(modality, data_type, reader_class):
    reader = open_reader(reader_class)
    fixture = np.load(FIXTURE_DIR / f"{modality}.npz")
    old_names = META["modalities"][modality]["variables"][data_type]
    old_values = fixture[f"raw__{data_type}"].astype(np.float64)
    new_values = read_windows(reader, fixture["store_rows"], fixture["start_days"]).astype(np.float64)

    masked_pixels = [] if reader.modality_id in UNMASKED_STORES else MASKED_PIXELS[modality]
    masked = np.isin(PIXELS, masked_pixels)
    shared_names = [name for name in old_names if name in reader.variables]
    assert shared_names, f"{reader_class.__name__} shares no variable with {modality}.{data_type}"

    for name in shared_names:
        old = pixels_first(old_values[..., old_names.index(name)])
        new = pixels_first(new_values[..., reader.variables.index(name)])
        if data_type == "data_static_categorical":
            old = np.where(old == OLD_MISSING_CLASS, np.nan, old)
        old_valid = ~np.isnan(old)
        new_valid = ~np.isnan(new)

        assert not new_valid[masked].any(), f"{name}: values on cells outside the land grid"

        lost = old_valid[~masked] & ~new_valid[~masked]
        assert not lost.any(), f"{name}: {lost.sum()} values of the old pipeline are missing"

        both = old_valid & new_valid
        assert both.any(), f"{name}: nothing to compare"
        largest_difference = np.abs(new[both] - old[both]).max()
        tolerance = RELATIVE_TOLERANCE * np.abs(old[both]).max()
        assert largest_difference <= tolerance, f"{name}: values differ by up to {largest_difference}"


def test_load_store_columns_returns_the_values_of_the_old_pipeline(tmp_path, monkeypatch):
    """The analysis entry point (deep_lst/utils/full_store.py): raw LST and DEM of the fixture
    anchors, selected through an index file as the exploration scripts do."""
    open_reader(SeviriLSTReader)  # skips where the stores are not available
    fixture = np.load(FIXTURE_DIR / "seviri.npz")
    monkeypatch.chdir(tmp_path)  # the index file is looked up under ./assets
    index_path = ExperimentPaths().spatial_indices_path("fixture_pixels")
    index_path.parent.mkdir(parents=True)
    np.savez(index_path, src_indices=fixture["anchor_ids"])

    window = 3
    start_day = int(fixture["start_days"][window])
    load = {"data": ["LST"], "data_static": ["DEM"]}
    columns, unique_rows, inverse, anchor_ids = load_store_columns(
        "seviri", "fixture_pixels", load, day_range=(start_day, start_day + N_DAYS), num_workers=2
    )

    order = np.argsort(fixture["anchor_ids"])  # the loader returns the anchors sorted
    assert np.array_equal(anchor_ids, fixture["anchor_ids"][order])
    assert np.array_equal(unique_rows[inverse], fixture["store_rows"][order])
    masked = np.isin(np.array(PIXELS)[order], MASKED_PIXELS["seviri"])
    new_lst = columns["data"].numpy()[inverse][:, :, 0].astype(np.float64)
    old_lst = fixture["raw__data"][window][order][:, :, 0].astype(np.float64)
    new_dem = columns["data_static"].numpy()[inverse][:, 0].astype(np.float64)
    old_dem = fixture["raw__data_static"][order][:, 0].astype(np.float64)
    for new, old in ((new_lst, old_lst), (new_dem, old_dem)):
        old_valid = ~np.isnan(old)
        assert not (old_valid[~masked] & np.isnan(new[~masked])).any()
        both = old_valid & ~np.isnan(new)
        assert both.any()
        assert np.abs(new[both] - old[both]).max() <= RELATIVE_TOLERANCE * np.abs(old[both]).max()
