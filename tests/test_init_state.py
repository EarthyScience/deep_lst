"""utils.init_state: lookback mean, tendency and count on a small daily series."""

import numpy as np

from deep_lst.utils.init_state import lookback_count, lookback_mean, lookback_tendency


def test_lookback_statistics():
    values = np.array([[1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0]])
    full = lookback_mean(values, 3)
    np.testing.assert_allclose(full[0, :3], [np.nan, np.nan, 2.0])
    assert np.isnan(full[0, 3:6]).all()  # every 3-day window touching the gap
    np.testing.assert_allclose(full[0, 6], 6.0)
    partial = lookback_mean(values, 3, min_days=1)
    np.testing.assert_allclose(partial[0], [1.0, 1.5, 2.0, 2.5, 4.0, 5.5, 6.0])
    tendency = lookback_tendency(values, 3)
    np.testing.assert_allclose(tendency[0, :3], np.nan)
    np.testing.assert_allclose(tendency[0, 4], 3.0)  # 5 - 2
    assert np.isnan(tendency[0, 3]) and np.isnan(tendency[0, 6])  # NaN at either end
    count = lookback_count(np.isfinite(values), 3)
    np.testing.assert_array_equal(count[0], [1, 2, 3, 2, 2, 2, 3])
