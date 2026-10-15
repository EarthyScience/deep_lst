"""Input perturbations for the sensitivity analysis.

They act on the preloaded columns every model consumes, {modality: {leaf: (cells, hours, C)}},
so windows, collator, model and forecast writer stay untouched and any stored input of any
model can be perturbed. Computed features (solar zenith, hour/day-of-year encodings) are built
per sample and cannot be perturbed here; static leaves (DEM, land cover) have no time axis.

permute_days shuffles whole days of one variable independently per cell: every cell keeps its
marginal distribution and diurnal cycle, but the day-to-day alignment with the other inputs
and with the target is broken. block="month" keeps the shuffle inside calendar months, so the
seasonal cycle survives and only the day-to-day anomalies are destroyed; block="all" shuffles
across the whole preload axis.
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from eodata.grids import T1H

from deep_lst.dataset.dataset import PRELOAD_END_DAY, PRELOAD_START_DAY

HOURS_PER_DAY = 24
STATIC_LEAVES = ("data_static", "data_static_categorical")


def preload_dates():
    """Calendar date of every day on the preload axis (naive timestamps; the grid epoch is UTC)."""
    start = pd.Timestamp(T1H.start)
    if start.tzinfo is not None:
        start = start.tz_localize(None)
    start = start + pd.Timedelta(days=PRELOAD_START_DAY)
    return pd.date_range(start, periods=PRELOAD_END_DAY - PRELOAD_START_DAY, freq="D")


def day_blocks(block):
    """Non-decreasing block id per preload day: one block for "all", one per calendar month."""
    dates = preload_dates()
    if block == "all":
        return np.zeros(len(dates), dtype=int)
    if block == "month":
        return np.asarray((dates.year - dates.year[0]) * 12 + dates.month - 1)
    raise ValueError(f"unknown block {block!r}; use 'all' or 'month'")


def permute_days(series, blocks, rng):
    """Shuffle whole days within blocks, independently per cell.

    series: (cells, hours) starting at a day boundary; blocks: (days,) non-decreasing ids, so
    sorting by block plus noise keeps every block in place and shuffles inside it.
    """
    cells, hours = series.shape
    days = hours // HOURS_PER_DAY
    daily = series[:, : days * HOURS_PER_DAY].reshape(cells, days, HOURS_PER_DAY).numpy()
    order = np.argsort(blocks[None, :] + rng.random((cells, days)), axis=1)
    shuffled = np.take_along_axis(daily, order[:, :, None], axis=1)
    out = series.clone()
    out[:, : days * HOURS_PER_DAY] = torch.from_numpy(shuffled.reshape(cells, days * HOURS_PER_DAY))
    return out


METHODS = {"permute_days": permute_days}


def locate(var_names, modality, variable):
    """(leaf, channel) of a stored variable in one modality's preload; KeyError for computed ones."""
    for leaf, names in var_names[modality].items():
        if variable in names:
            return leaf, names.index(variable)
    raise KeyError(f"{modality}.{variable} is not a stored variable; stored: {var_names[modality]}")


@contextmanager
def perturbed(shared, modality, variable, method, block, seed):
    """Replace one variable's series in the shared preload in place while the block runs, then
    restore it. In-RAM only; the preload cache on disk is never touched."""
    leaf, channel = locate(shared["var_names"], modality, variable)
    if leaf in STATIC_LEAVES:
        raise ValueError(f"{modality}.{variable} is static (no time axis); nothing to permute")
    column = shared["columns"][modality][leaf]
    original = column[:, :, channel].clone()
    column[:, :, channel] = METHODS[method](original, day_blocks(block), np.random.default_rng(seed))
    try:
        yield
    finally:
        column[:, :, channel] = original
