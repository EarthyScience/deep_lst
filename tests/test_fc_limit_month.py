import numpy as np

from deep_lst.evaluation.fc_limit_month import _first_below_zero

DAYS = np.array([1, 2, 3, 4, 5])
HORIZON = 5


def test_normal_crossing_returns_first_below_zero_day():
    skill = np.array([0.5, 0.3, 0.1, -0.2, -0.4])
    assert _first_below_zero(DAYS, skill, HORIZON) == (4.0, False)


def test_crosses_immediately_returns_day_one_full_point():
    skill = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])
    assert _first_below_zero(DAYS, skill, HORIZON) == (1.0, False)


def test_never_crosses_returns_horizon_censored():
    skill = np.array([0.6, 0.5, 0.4, 0.3, 0.2])
    assert _first_below_zero(DAYS, skill, HORIZON) == (5.0, True)


def test_all_nan_returns_no_point():
    skill = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    assert _first_below_zero(DAYS, skill, HORIZON) == (None, False)


def test_nan_gap_is_skipped_not_a_crossing():
    skill = np.array([0.5, np.nan, -0.1, np.nan, -0.3])
    assert _first_below_zero(DAYS, skill, HORIZON) == (3.0, False)
