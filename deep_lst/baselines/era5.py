"""ERA5 reference baseline: an ERA5 variable at the eval pixels for the test year."""

from deep_lst.baselines.base import ModalityBaselineExtractor


class Era5Extractor(ModalityBaselineExtractor):
    modality = "era5"
