"""ERA5-Land reference baseline: an ERA5-Land variable (e.g. skt) at the eval pixels for the test year."""

from deep_lst.baselines.base import ModalityBaselineExtractor


class Era5LandExtractor(ModalityBaselineExtractor):
    modality = "era5_land"
