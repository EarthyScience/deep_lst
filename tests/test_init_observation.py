"""attach_init_observation (evaluation_climpred/cubes.py) and the init/gap windows (masks.py).

Hermetic: a four-hour observation series over three pixels and two inits.
"""

import numpy as np
import xarray as xr

from deep_lst.evaluation_climpred.cubes import attach_init_observation
from deep_lst.evaluation_climpred.masks import MASKS, range_mask

LOOKBACK = 2
T0 = np.datetime64("2024-01-01T06:00")
HOUR = np.timedelta64(1, "h")


def _obs():
    """Hourly series T0-3h .. T0+1h; NaN = cloud.
    pixel 0: clear throughout; pixel 1: cloudy at T0 only; pixel 2: cloudy from T0-2h on."""
    times = T0 + np.arange(-3, 2) * HOUR
    values = np.ones((5, 3), dtype=np.float32)
    values[3, 1] = np.nan  # pixel 1 at T0
    values[1:, 2] = np.nan  # pixel 2 from T0-2h on
    return xr.Dataset({"LST": (("time", "pixel"), values)}, coords={"time": times})


def _initialised():
    inits = np.array([T0, T0 + HOUR])
    return xr.Dataset(coords={"init": inits, "pixel": np.arange(3)})


def test_init_observation_from_the_series():
    initilised = attach_init_observation(_initialised(), _obs(), LOOKBACK)

    observed = initilised["init_observed"].values
    hours = initilised["hours_since_obs"].values
    assert observed.tolist() == [[True, False, False], [True, True, False]]
    np.testing.assert_array_equal(
        hours[0], [0.0, 1.0, np.nan]
    )  # init T0: pixel 2 last clear at T0-3h, outside the lookback
    np.testing.assert_array_equal(hours[1], [0.0, 0.0, np.nan])
    assert not initilised["init_filled"].values.any()


def test_init_observation_from_the_store_lead0():
    """A store with the uniform lead-0 layout overrides the series at the init hour and
    carries the filled flag."""
    inits = _initialised()["init"].values
    lead0_obs = np.array([[1.0, 1.0, np.nan], [np.nan, 1.0, np.nan]], dtype=np.float32)  # pixel 1 clear at T0 after all
    obs_cube = np.full((2, 3, 3), np.nan, dtype=np.float32)  # (init, lead 0..2, pixel)
    obs_cube[:, 0] = lead0_obs
    filled = np.array([[False, False, True], [True, False, True]])
    forecast = xr.Dataset(
        {
            "LST_obs": (("init_time", "lead_time", "pixel"), obs_cube),
            "LST_init_filled": (("init_time", "pixel"), filled),
        },
        coords={"init_time": inits, "lead_time": np.arange(3) * HOUR},
    )

    initilised = attach_init_observation(_initialised(), _obs(), LOOKBACK, forecast)

    assert initilised["init_observed"].values.tolist() == [[True, True, False], [False, True, False]]
    np.testing.assert_array_equal(initilised["hours_since_obs"].values[1], [1.0, 0.0, np.nan])
    assert initilised["init_filled"].values.tolist() == filled.tolist()


def test_range_mask_and_init_windows():
    df = xr.Dataset(
        coords={
            "hours_since_obs": ("init", np.array([0.0, 3.0, 80.0, np.nan])),
            "init_observed": ("init", [True, False, False, False]),
        }
    )

    assert range_mask("hours_since_obs", (1, 7))(df).values.tolist() == [False, True, False, False]
    assert range_mask("hours_since_obs", (73, None))(df).values.tolist() == [False, False, True, False]
    assert MASKS["gap_0"](df).values.tolist() == [True, False, False, False]
    assert MASKS["init_gap"](df).values.tolist() == [False, True, True, True]
