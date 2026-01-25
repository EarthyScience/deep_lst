from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFilter:
    time_index: pd.DatetimeIndex

    @classmethod
    def from_range(
        cls,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        freq: str = "h",
    ) -> "TemporalFilter":
        ti = pd.date_range(
            start=start,
            end=end,
            freq=freq,
        )
        return cls(pd.DatetimeIndex(ti))

    def filter_by_date(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        inclusive_left: bool = True,
        inclusive_right: bool = False,
    ) -> np.ndarray:
        """
        Return indices i where start_date <= t < end_date by default.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        left_ok = self.time_index >= start if inclusive_left else self.time_index > start
        right_ok = self.time_index <= end if inclusive_right else self.time_index < end
        mask = left_ok & right_ok

        idx = np.flatnonzero(mask).astype(np.int64)
        return idx

    def stratified_sample(self, indices: np.ndarray, frac: float, by: str = "month", seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)

        indices = np.asarray(indices, dtype=np.int64)
        t = self.time_index[indices]

        if by == "month":
            labels = t.month.to_numpy()
        elif by == "hour":
            labels = t.hour.to_numpy()
        elif by == "month_hour":
            labels = t.month.to_numpy() * 100 + t.hour.to_numpy()  # 1..12 and 0..23 -> unique key
        else:
            raise ValueError("by must be 'month', 'hour', or 'month_hour'")

        total = max(1, int(round(len(indices) * frac)))

        # group using stable sort + split
        order = np.argsort(labels, kind="stable")
        labels_sorted = labels[order]
        idx_sorted = indices[order]

        _, start = np.unique(labels_sorted, return_index=True)
        groups = np.split(idx_sorted, start[1:])
        sizes = np.array([len(g) for g in groups], dtype=int)

        # proportional allocation (largest remainder)
        raw = sizes / sizes.sum() * total
        alloc = np.floor(raw).astype(int)
        rem = total - alloc.sum()
        if rem > 0:
            alloc[np.argsort(raw - alloc)[::-1][:rem]] += 1

        out = [rng.choice(g, size=k, replace=False) for g, k in zip(groups, alloc) if k > 0]
        out = np.concatenate(out).astype(np.int64)
        rng.shuffle(out)
        return out


def get_temporal_indices(config, split: str, stratify: bool = False) -> np.ndarray:
    tf = TemporalFilter.from_range("2017-01-01 00:00", "2017-12-31 23:00")
    temporal_indices = tf.filter_by_date(config[f"{split}_start"], config[f"{split}_end"])
    if stratify:
        temporal_indices = tf.stratified_sample(temporal_indices, frac=0.05, by="month_hour", seed=42)
    return temporal_indices
