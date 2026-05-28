import numpy as np
import pandas as pd
import torch
import xarray as xr

from deep_lst.dataset.climatology import ClimatologyLookup


def _write_synthetic_clim(tmp_path, values: np.ndarray, flat_ids: list[int]):
    """values: (8784, N), flat_ids: list of len N global flat ints."""
    clim_path = tmp_path / "clim.nc"
    idx_path = tmp_path / "indices.npz"
    ds = xr.Dataset(
        {"LST": (("valid_time", "pixel"), values.astype(np.float32))},
        coords={"valid_time": pd.date_range("2024-01-01", periods=8784, freq="h")},
    )
    ds.to_netcdf(clim_path)
    np.savez(idx_path, mapping=np.array(flat_ids, dtype=np.int64))
    return clim_path, idx_path


def test_lookup_window_round_trip(tmp_path):
    flat_ids = [10, 20, 30]
    values = np.zeros((8784, 3), dtype=np.float32)
    values[:, :] = np.arange(8784)[:, None] * 1000 + np.array(flat_ids)[None, :]  # values[t, p] = 1000*t + flat_id

    clim_path, idx_path = _write_synthetic_clim(tmp_path, values, flat_ids)
    ts = pd.date_range("2024-01-01", periods=8784, freq="h")  # one leap year — natural doy already leap-aligned

    lookup = ClimatologyLookup(
        clim_path=clim_path,
        spatial_idx_path=idx_path,
        lookback=2,
        horizon=3,
        state_mean=0.0,
        state_std=1.0,
        store_time_range=ts,
    )

    spatial_idx = torch.tensor([20, 30])
    time_idx = torch.tensor([100, 500])
    out = lookup.lookup_window(spatial_idx, time_idx)  # (2, 6)

    assert out.shape == (2, 6)
    # Window is [-2, +3] inclusive → 6 entries
    for b, (t0, fid) in enumerate(zip([100, 500], [20, 30])):
        for k, dt in enumerate(range(-2, 4)):
            expected = 1000 * (t0 + dt) + fid
            assert out[b, k].item() == expected


def test_lookup_forward_round_trip(tmp_path):
    flat_ids = [10, 20, 30]
    values = np.zeros((8784, 3), dtype=np.float32)
    values[:, :] = np.arange(8784)[:, None] * 1000 + np.array(flat_ids)[None, :]  # values[t, p] = 1000*t + flat_id

    clim_path, idx_path = _write_synthetic_clim(tmp_path, values, flat_ids)
    ts = pd.date_range("2024-01-01", periods=8784, freq="h")  # one leap year — natural doy already leap-aligned

    horizon = 3
    lookback = 2
    lookup = ClimatologyLookup(
        clim_path=clim_path,
        spatial_idx_path=idx_path,
        lookback=lookback,
        horizon=horizon,
        state_mean=0.0,
        state_std=1.0,
        store_time_range=ts,
    )

    spatial_idx = torch.tensor([20, 30])
    time_idx = torch.tensor([100, 500])
    nsteps = 5
    out = lookup.lookup_forward(spatial_idx, time_idx, nsteps)
    assert out.shape == (2, nsteps + 1)

    out = lookup.lookup_forward(spatial_idx, time_idx, horizon)
    out_lw = lookup.lookup_window(spatial_idx, time_idx)[:, lookback : lookback + horizon + 1]

    assert torch.equal(out, out_lw), "forward lookup must match window lookup over [0..horizon]"


def test_leap_year_alignment(tmp_path):
    # Values encode the (doy, hour) row position so the lookup result lets us reverse-engineer the row used.
    values = np.zeros((8784, 1), dtype=np.float32)
    values[:, 0] = np.arange(8784)  # row index recoverable from the value
    clim_path, idx_path = _write_synthetic_clim(tmp_path, values, [42])

    ts = pd.date_range("2015-01-01", "2016-12-31 23:00", freq="h")  # spans non-leap + leap

    lookup = ClimatologyLookup(
        clim_path=clim_path,
        spatial_idx_path=idx_path,
        lookback=0,
        horizon=0,
        state_mean=0.0,
        state_std=1.0,
        store_time_range=ts,
    )

    cases = [
        (pd.Timestamp("2015-02-28 12:00"), (59 - 1) * 24 + 12),  # 1404
        (pd.Timestamp("2015-03-01 12:00"), (61 - 1) * 24 + 12),  # 1452  ← the leap-shift case
        (pd.Timestamp("2016-02-29 12:00"), (60 - 1) * 24 + 12),  # 1428
        (pd.Timestamp("2016-03-01 12:00"), (61 - 1) * 24 + 12),  # 1452
    ]
    for date, expected_row in cases:
        t = ts.get_loc(date)
        out = lookup.lookup_window(torch.tensor([42]), torch.tensor([t]))
        assert out.item() == expected_row, f"{date}: got row {int(out.item())}, expected {expected_row}"


def test_normalization(tmp_path):
    state_mean, state_std = 300.0, 10.0
    # Constant clim equal to state_mean → normalized clim should be 0 everywhere
    values = np.full((8784, 1), state_mean, dtype=np.float32)
    clim_path, idx_path = _write_synthetic_clim(tmp_path, values, [42])

    ts = pd.date_range("2024-01-01", periods=8784, freq="h")
    lookup = ClimatologyLookup(
        clim_path=clim_path,
        spatial_idx_path=idx_path,
        lookback=0,
        horizon=0,
        state_mean=state_mean,
        state_std=state_std,
        store_time_range=ts,
    )
    out = lookup.lookup_window(torch.tensor([42]), torch.tensor([5000]))
    assert torch.isclose(out, torch.tensor([[0.0]]))
