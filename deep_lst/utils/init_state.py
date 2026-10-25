"""Lookback statistics of daily site states (scripts/assets/fvc_init_state.py, hydro_init_state.py).

Every function takes a (cells, days) array with a contiguous day axis and returns an array of
the same shape; the window of a day is the lookback_days days ending at that day, inclusive.
"""

import numpy as np
import pandas as pd


def lookback_mean(values, lookback_days, min_days=None):
    """Mean over the window, NaN ignored; NaN where fewer than min_days values are present
    (default: the whole window must be present)."""
    min_days = lookback_days if min_days is None else min_days
    frame = pd.DataFrame(np.asarray(values).T)
    rolling = frame.rolling(lookback_days, min_periods=min_days).mean()
    return rolling.to_numpy().T.astype(np.float32)


def lookback_tendency(values, lookback_days):
    """Value of the day minus the value lookback_days days earlier; NaN where either is missing."""
    values = np.asarray(values)
    tendency = np.full(values.shape, np.nan, dtype=np.float32)
    tendency[:, lookback_days:] = values[:, lookback_days:] - values[:, :-lookback_days]
    return tendency


def lookback_count(flags, lookback_days):
    """Number of True flags in the window."""
    frame = pd.DataFrame(np.asarray(flags).T.astype(np.int16))
    rolling = frame.rolling(lookback_days, min_periods=1).sum()
    return rolling.to_numpy().T.astype(np.int16)
