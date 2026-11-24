"""deep_lst's own readers for the product stores (no eodata)."""

from deep_lst.stores.anchors import anchor_lonlat, map_to_store_rows
from deep_lst.stores.ascat import AscatReader
from deep_lst.stores.base import StoreReader
from deep_lst.stores.cube import CubeReader
from deep_lst.stores.era5 import Era5Reader
from deep_lst.stores.era5_land import Era5LandReader, Era5LandStaticReader
from deep_lst.stores.glofas import GlofasReader
from deep_lst.stores.grid import LatLonGrid
from deep_lst.stores.majortom import majortom_centers
from deep_lst.stores.roa import RoaReader
from deep_lst.stores.seviri import (
    SeviriDEMReader,
    SeviriFVCReader,
    SeviriLANDCOVReader,
    SeviriLSTReader,
    SeviriLWMASKReader,
)
from deep_lst.stores.time_axis import T1D, T1H, TimeAxis

# store name (the modality_id of the store, and its file name without .zarr) -> reader class
READERS = {
    reader_class.MODALITY_ID: reader_class
    for reader_class in (
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
}


def get_data_reader(store_name, store):
    """The reader of a product store.

    Args:
        store_name: Name of the store, e.g. "seviri_product_hourly_lst".
        store: Folder of the store.
    """
    if store_name not in READERS:
        raise ValueError(f"No reader for the store {store_name!r}, expected one of {sorted(READERS)}")
    return READERS[store_name](store)


__all__ = [
    "AscatReader",
    "CubeReader",
    "Era5LandReader",
    "Era5LandStaticReader",
    "Era5Reader",
    "GlofasReader",
    "LatLonGrid",
    "READERS",
    "RoaReader",
    "SeviriDEMReader",
    "SeviriFVCReader",
    "SeviriLANDCOVReader",
    "SeviriLSTReader",
    "SeviriLWMASKReader",
    "StoreReader",
    "T1D",
    "T1H",
    "TimeAxis",
    "anchor_lonlat",
    "get_data_reader",
    "majortom_centers",
    "map_to_store_rows",
]
