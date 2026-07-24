"""Persistence baseline: the forecast at valid_time t is the last clear-sky observation from the
same time of day on the previous day (forward-filled with no limit)."""

from pathlib import Path

import numpy as np
import pandas as pd

from deep_lst.baselines.base import _ANCHOR_MODALITY, BaselineExtractor


class PersistenceExtractor(BaselineExtractor):
    def _acquire(self, shared: dict, config: dict) -> None:
        self.spatial_idx, self.timestamps, self.var, self.pixel_lonlat = self._preload_series(
            shared, _ANCHOR_MODALITY, self.var_name
        )

    def _compute(self):
        time_of_day = self.timestamps.time  # datetime.time per timestep
        unique_times = np.unique(time_of_day)

        hour_wise_persistence = []
        for hour in unique_times:
            hourly_mask = time_of_day == hour
            hour_timestamps = self.timestamps[hourly_mask]
            test_mask = hour_timestamps.year == self.test_year

            hour_lst = self.var[:, hourly_mask]  # (n_pixels, n_time_at_this_hour)

            df = pd.DataFrame(hour_lst.T, index=hour_timestamps)  # (n_time_at_this_hour, n_pixels)
            persistence = df.shift(1).ffill()  # shift one day, then carry the last observed value forward

            hour_wise_persistence.append(persistence.loc[test_mask])

        results_df = pd.concat(hour_wise_persistence).sort_index()
        persistence = results_df.values.astype(np.float32)  # valid_time × pixel
        test_timestamps = pd.DatetimeIndex(results_df.index)

        attrs = dict(
            test_year=str(self.test_year),
            variable=self.var_name,
            description=(
                "Persistence forecast for test year. "
                "Persistence at valid_time t uses the last clear-sky obs from (t - 1 day) backward with no limit. "
                "run_evaluation.py applies the lead_time shift during alignment."
            ),
        )
        self.result = self._make_dataset(persistence, test_timestamps, attrs)

    def output_path(self) -> Path:
        return self.paths.persistence_path(self.store_key)
