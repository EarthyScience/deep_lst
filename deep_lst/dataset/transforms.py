"""Per-modality value transforms applied before z-scoring.

A `transform:` key in a modality's dataset config names one of these. The statistics script
applies the same transform before accumulating moments, so the stored mean/std describe the
transformed values and the z-score that follows is well-conditioned.

signed_log1p: sign(x) * log1p(|x|). Compresses heavy-tailed, zero-anchored hydrology variables
(GloFAS discharge, snow depth) while keeping zero at zero and the sign of negative values.
"""

import torch
from eodata.normalization import ModalityNormalizer, ZScoreNormalizer


def signed_log1p(values):
    return torch.sign(values) * torch.log1p(torch.abs(values))


def signed_expm1(values):
    return torch.sign(values) * torch.expm1(torch.abs(values))


class SignedLog1pZScore(ZScoreNormalizer):
    """sign(x) * log1p(|x|), then z-score with statistics computed on the transformed values."""

    def normalize_torch(self, values):
        transformed = signed_log1p(values)
        return super().normalize_torch(transformed)

    def denormalize_torch(self, values):
        transformed = super().denormalize_torch(values)
        return signed_expm1(transformed)


TRANSFORMS = {"signed_log1p": signed_log1p}
NORMALIZERS = {"signed_log1p": SignedLog1pZScore}


def transform_function(transform):
    """The value transform named by a config `transform:` key, or None when the key is unset."""
    if transform is None:
        return None
    if transform not in TRANSFORMS:
        raise ValueError(f"Unknown transform {transform!r}, expected one of {sorted(TRANSFORMS)}")
    return TRANSFORMS[transform]


def with_transform(normalizer, transform):
    """Swap the `data` leaf's z-score for the transform-then-z-score variant, keeping its mean/std.

    eodata's `ModalityNormalizer.from_statistics` only accepts its own normalizer classes, so the
    leaf is replaced after construction instead of built through it.
    """
    if transform is None:
        return normalizer
    if transform not in NORMALIZERS:
        raise ValueError(f"Unknown transform {transform!r}, expected one of {sorted(NORMALIZERS)}")
    normalizers = normalizer.normalizers
    zscore = normalizers["data"]
    normalizers["data"] = NORMALIZERS[transform](zscore.mean, zscore.std)
    return ModalityNormalizer(normalizers)
