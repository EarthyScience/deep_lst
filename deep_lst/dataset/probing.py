"""Data-loading throughput probe for the new eodata .eozarr stores.

Question it answers: is random-access reading from the full SEVIRI .eozarr fast
enough to train on directly, or do we still need the in-memory sampled extraction?

It simulates one "epoch" of random (pixel, window) reads at a realistic batch size
and worker count, and reports samples/s. Compare that to your model's step rate.
"""

import random
import time
from datetime import date

import eodata
import torch
from torch.utils.data import DataLoader, Dataset

STORE = "/Net/Groups/BGI/tscratch/weathergenerator/seviri_2016_2024.eozarr"
LOOKBACK = 24
HORIZON = 24

NUM_WORKERS = 6
BATCH = 128
N_BATCHES = 100  # ~25k samples — enough to average out; raise for a steadier number

# Reader config — same selection as adapter.py. get_data_multi reads ALL of these from
# disk per call, so the measured I/O cost is representative even though the inline adapter
# below only windows "data".
reader_kwargs = dict(
    get_data_static=True,
    get_data_static_categorical=True,
    data_variables=["LST", "MLE"],
    data_daily_variables=["FVC"],
    data_static_variables=["DEM"],
    data_static_categorical_variables=["LANDCOV"],
    average_hourly=False,
)

# --- valid hour range -------------------------------------------------------
# The store's day axis is the full T1D grid (day 0 = 1970-01-01, 25567 days to ~2040),
# but only 2016-2024 is populated. global_hour = days_since_1970 * 24, so we must sample
# hours inside the populated days or we'd read empty (fill) chunks and flatter the number.
EPOCH = date(1970, 1, 1)
DAY_MIN = (date(2016, 1, 1) - EPOCH).days
DAY_MAX = (date(2024, 12, 31) - EPOCH).days
HOUR_MIN = DAY_MIN * 24 + LOOKBACK
HOUR_MAX = DAY_MAX * 24 - HORIZON

# --- working set of pixels --------------------------------------------------
# A fixed random subset of the 10.2M land pixels. This is a CONSERVATIVE (worst-case)
# choice for chunk locality: scattered pixels get almost no chunk reuse. Your real
# training pixels (stratified sites) may cluster differently — swap in that subset to refine.
N_TOTAL_PIXELS = 10_246_401
N_PIXELS = 5000
random.seed(0)
PIXELS = random.sample(range(N_TOTAL_PIXELS), N_PIXELS)


class Adapter:
    """Minimal, data-only windowing adapter (see dataset/adapter.py for the full one)."""

    def __init__(self, reader, lookback, horizon):
        self.reader = reader
        self.lookback = lookback
        self.horizon = horizon

    def get_window(self, spatial_idx, time_idx):
        start_hour = time_idx - self.lookback
        end_hour = time_idx + self.horizon
        start_day = start_hour // 24
        end_day = end_hour // 24

        indices = [(spatial_idx, day) for day in range(start_day, end_day + 1)]
        sample = self.reader.get_data_multi(indices)  # reads all categories from disk

        data = sample["data"].permute(0, 2, 1, 3, 4)  # [days, 24, vars, 1, 1]
        n_vars = data.shape[2]
        data = data.reshape(-1, n_vars, data.shape[3], data.shape[4])  # [days*24, vars, 1, 1]
        local_start = start_hour - start_day * 24
        length = self.lookback + self.horizon + 1
        return {"data": data[local_start : local_start + length]}  # [L+H+1, vars, 1, 1]


class ProbeDataset(Dataset):
    def __init__(self, n):
        self.n = n
        self.adapter = None  # open lazily, per worker (NOT in __init__)

    def _lazy(self):
        if self.adapter is None:
            reader = eodata.get_data_reader("seviri", store=STORE, **reader_kwargs)
            self.adapter = Adapter(reader, LOOKBACK, HORIZON)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self._lazy()
        pixel = PIXELS[random.randrange(len(PIXELS))]
        hour = random.randrange(HOUR_MIN, HOUR_MAX)
        return self.adapter.get_window(pixel, hour)["data"]  # triggers the read


def main():
    dl = DataLoader(ProbeDataset(BATCH * N_BATCHES), batch_size=BATCH, num_workers=NUM_WORKERS, collate_fn=torch.stack)

    print(
        f"config: BATCH={BATCH} NUM_WORKERS={NUM_WORKERS} N_BATCHES={N_BATCHES} "
        f"N_PIXELS={N_PIXELS} hours=[{HOUR_MIN},{HOUR_MAX}] (days {DAY_MIN}-{DAY_MAX})",
        flush=True,
    )

    t0 = time.time()
    seen = 0
    first_batch_time = None
    for b in dl:
        if first_batch_time is None:
            first_batch_time = time.time() - t0  # cold: includes worker startup + first reads
        seen += b.shape[0]
    elapsed = time.time() - t0

    print(f"sample shape (last batch): {tuple(b.shape)}", flush=True)
    print(f"samples:            {seen}", flush=True)
    print(f"elapsed:            {elapsed:.1f}s", flush=True)
    print(f"first batch (cold): {first_batch_time:.1f}s", flush=True)
    print(f"throughput:         {seen / elapsed:.0f} samples/s", flush=True)
    print(f"batches/s:          {N_BATCHES / elapsed:.2f}", flush=True)
    # warm rate excluding the cold first batch:
    warm = elapsed - first_batch_time
    print(f"warm throughput:    {(seen - BATCH) / warm:.0f} samples/s (excl. cold first batch)", flush=True)


if __name__ == "__main__":
    main()
