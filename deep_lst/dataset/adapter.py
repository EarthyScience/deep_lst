import eodata
from eodata.normalization import ModalityNormalizer

# verify imports, read one window through 15-line example adapter on era5, era5land fix.
# data loading throughput smoke test,
# simulate epochs access pattern, random reads at my real batch size -> need extraction?
# windowing adapter : get_window(pixel, hour) equals a hand-checked raw-zarr slice, anchor lands at position lookback.
# Wire into dataset.py, move role-packing into __getitem__, reduce collator. Verify: one batch has correct shapes (B, L+H+1, H, W, nS/nF/nFc).
# Statistics recompute for the new variable sets → store in the format your normalizer reads. Verify: a normalized batch has ~0 mean / 1 std per channel.
# Train-step smoke
# Extraction — only if step 2 said the full store is too slow: migrate the sampled-extraction pipeline to the day×24 h schema + pixel-friendly rechunk; re-run step 2. Verify: throughput now saturates the model.
# ERA5-Land as a second modality once its modality-id is fixed upstream. Verify: multi-store batch.


class Adapter:
    def __init__(self, reader, lookback, horizon):
        self.reader = reader
        self.lookback = lookback
        self.horizon = horizon

    def get_window(self, spatial_idx, time_idx):
        # time_idx here is the a global hour index.

        start_hour = time_idx - self.lookback
        end_hour = time_idx + self.horizon
        start_day = start_hour // 24
        end_day = end_hour // 24

        sample_indices = [(spatial_idx, day) for day in range(start_day, end_day + 1)]
        sample = self.reader.get_data_multi(sample_indices)

        data = sample["data"]
        data_daily = sample["data_daily"]
        data_static = sample["data_static"][0]
        data_static_categorical = sample["data_static_categorical"][0]
        metadata = sample["metadata"]

        # window dynamic
        data = data.permute(0, 2, 1, 3, 4)
        n_vars = data.shape[2]
        data = data.reshape(-1, n_vars, data.shape[3], data.shape[4])  # [days*24, variables, 1, 1]  continuous hours
        local_start = start_hour - start_day * 24
        length = self.lookback + self.horizon + 1
        data_window = data[local_start : local_start + length]  # [L+H+1, C, 1, 1]

        # window metadata
        metadata = metadata.permute(0, 2, 1, 3, 4)
        n_vars = metadata.shape[2]
        metadata = metadata.reshape(
            -1, n_vars, metadata.shape[3], metadata.shape[4]
        )  # [days*24, variables, 1, 1]  continuous hours
        local_start = start_hour - start_day * 24
        length = self.lookback + self.horizon + 1
        metadata_window = metadata[local_start : local_start + length]  # [L+H+1, C, 1, 1]

        # window dynamic daily
        data_daily = data_daily.unsqueeze(2).expand(-1, -1, 24, -1, -1)  # [days, C, 24, 1, 1]
        data_daily = data_daily.permute(0, 2, 1, 3, 4)
        data_daily = data_daily.reshape(-1, data_daily.shape[2], data_daily.shape[3], data_daily.shape[4])
        data_daily_window = data_daily[local_start : local_start + length]  # [L+H+1, C, 1, 1]

        # broadcast daily and static variables to the window shape
        static_window = data_static.expand(length, -1, -1, -1)
        data_static_categorical_window = data_static_categorical.expand(length, -1, -1, -1)

        return {
            "data": data_window,
            "data_daily": data_daily_window,
            "data_static": static_window,
            "data_static_categorical": data_static_categorical_window,
            "metadata": metadata_window,
        }


if __name__ == "__main__":
    reader = eodata.get_data_reader(
        "seviri",
        store="/Net/Groups/BGI/tscratch/weathergenerator/seviri_2016_2024.eozarr",
        get_data_static=True,
        get_data_static_categorical=True,
        average_hourly=False,
    )  # load ALL variables (matches the full stats zarr); select roles downstream in packing

    reader.post_processor = ModalityNormalizer.from_spec(
        "zscore:/Net/Groups/BGI/people/mwessel/deep_lst/assets/statistics/seviri_stats.zarr",
        reader.schema,
    ).normalize

    sample_indices = eodata.indices.SampleIndices([(100, 100), (100, 101), (100, 102)])

    adapter = Adapter(reader, lookback=24, horizon=24)
    adapter.get_window(100, 101)
