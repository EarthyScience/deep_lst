import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.products import align_baseline


def _make_baseline_full_inputs():
    """pred (init, lead, pixel) with one masked cell, plus a baseline on valid_time whose value
    encodes the valid hour (hour * 10 + pixel) so the alignment is checkable by hand."""
    init_times = pd.date_range("2016-01-01", periods=3, freq="h")
    lead_times = np.array([1, 2], dtype="timedelta64[h]").astype("timedelta64[ns]")
    pixels = [0, 1]

    pred_data = np.ones((3, 2, 2), dtype=np.float32)
    pred_data[0, 0, 0] = np.nan  # mask init=0, lead=1h, pixel=0
    pred = xr.DataArray(
        pred_data,
        dims=["init_time", "lead_time", "pixel"],
        coords={"init_time": init_times, "lead_time": lead_times, "pixel": pixels},
        name="LST_pred",
    )

    valid_times = pd.date_range("2016-01-01", periods=6, freq="h")
    hours = valid_times.hour.values.astype(np.float32)
    base = np.stack([hours * 10 + p for p in pixels], axis=1)  # (valid_time, pixel)
    baseline_ds = xr.DataArray(
        base,
        dims=["valid_time", "pixel"],
        coords={"valid_time": valid_times, "pixel": pixels},
        name="LST",
    )
    return pred, baseline_ds


def test_align_baseline_aligns_and_masks():
    # Ported from the deleted EvaluatorVerification._baseline_full; align_baseline is its successor.
    pred, baseline_ds = _make_baseline_full_inputs()

    baseline_full = align_baseline(baseline_ds, pred)

    lead1, lead2 = pred["lead_time"].values
    init0, _init1, init2 = pred["init_time"].values

    assert set(baseline_full.dims) == {"lead_time", "init_time", "pixel"}
    assert dict(baseline_full.sizes) == {"lead_time": 2, "init_time": 3, "pixel": 2}

    # value = baseline at valid = init + lead = (hour(init) + lead_hours) * 10 + pixel
    assert float(baseline_full.sel(lead_time=lead1, init_time=init0, pixel=1)) == 11.0
    assert float(baseline_full.sel(lead_time=lead1, init_time=init2, pixel=0)) == 30.0
    assert float(baseline_full.sel(lead_time=lead2, init_time=init2, pixel=0)) == 40.0

    # masked pred cell -> NaN in the aligned baseline
    masked = baseline_full.sel(lead_time=lead1, init_time=init0, pixel=0)
    assert bool(np.isnan(masked))
