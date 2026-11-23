"""Time axes of the product stores.

Every product store has one time axis that starts at 1970-01-01T00 UTC and runs in fixed steps
up to 2040-01-01: 613,608 steps for an hourly store (T1H), 25,567 for a daily store (T1D). The
position of a step on that axis is its time index. It addresses the first dimension of a store's
data array, and it is the hour index used throughout deep_lst (time index files, forecasts).

TimeAxis replaces the eodata time grids of the same names and keeps their method names.
"""

import numpy as np
import pandas as pd

AXIS_START = pd.Timestamp("1970-01-01", tz="UTC").as_unit("ns")
AXIS_END = pd.Timestamp("2040-01-01", tz="UTC").as_unit("ns")  # exclusive


class TimeAxis:
    """Fixed-step time axis from AXIS_START to AXIS_END.

    Attributes:
        name: Name of the axis, as in the root attributes of a store ("T1H", "T1D").
        step: Length of one step (pd.Timedelta).
        start: Start of the first step (pd.Timestamp, UTC).
        end: End of the last step, exclusive (pd.Timestamp, UTC).
    """

    def __init__(self, name, step):
        self.name = name
        self.step = pd.Timedelta(step)
        self.start = AXIS_START
        self.end = AXIS_END

    def __len__(self):
        return int((self.end - self.start) // self.step)

    def date_to_index(self, date):
        """Index of the step that contains a date.

        Args:
            date: A string or timestamp. Without a time zone it is read as UTC.

        Returns:
            The time index (int). A date inside a step gives the index of that step.

        Raises:
            KeyError: The date is before the start or after the end of the axis.
        """
        timestamp = pd.Timestamp(date)
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize("UTC")
        index = int((timestamp - self.start) // self.step)
        if index < 0 or index >= len(self):
            raise KeyError(f"{date} is not on the time axis {self.name}")
        return index

    def index_to_startdate(self, index):
        """Start of one step or of several steps.

        Args:
            index: One time index, or an array of them.

        Returns:
            A pd.Timestamp for one index, a pd.DatetimeIndex for an array; both in UTC.
        """
        if np.ndim(index) == 0:
            return self.start + int(index) * self.step
        steps = np.asarray(index, dtype=np.int64)
        offsets = pd.to_timedelta(steps * self.step.value, unit="ns")
        return self.start + offsets


T1H = TimeAxis("T1H", "1h")
T1D = TimeAxis("T1D", "1D")
