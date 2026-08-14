import numpy as np
from eodata.selection import TemporalFilter


def zambia_drought_seasonal():
    sampler = TemporalFilter(start_date="2016-01-01", end_date="2025-01-01")
    init_dates = ["2024-01-01", "2024-02-01", "2024-03-01"]
    indices = np.array([sampler.T1H.date_to_index(date) for date in init_dates], dtype=np.int64)

    kept = sampler.exclude_bad_dates(indices, "assets/temporal/bad_dates.txt")
    assert len(kept) == len(indices), "an init date sits on a bad-dates day"

    TemporalFilter.save("assets/temporal", "test_zambia_seasonal", indices)


if __name__ == "__main__":
    zambia_drought_seasonal()
