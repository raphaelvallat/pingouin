"""Tests for pingouin.yuen (Yuen 1974 trimmed-mean t-test).

Reference
---------
Yuen, K. K. (1974). The two-sample trimmed t for unequal population variances.
Biometrika, 61(1), 165-170.

Wilcox, R. R. (2012). Introduction to Robust Estimation and Hypothesis Testing
(3rd ed.). Academic Press.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, "src")

import pingouin as pg
from pingouin import yuen


RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_samples(n=30, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n), rng.normal(shift, 2, n)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_dataframe():
    x, y = _make_samples()
    result = yuen(x, y)
    assert isinstance(result, pd.DataFrame)


def test_index_is_Yuen():
    x, y = _make_samples()
    result = yuen(x, y)
    assert result.index.tolist() == ["Yuen"]


def test_columns_present():
    x, y = _make_samples()
    result = yuen(x, y)
    for col in ("T", "dof", "alternative", "p_val", "trim", "tail_mean_x", "tail_mean_y"):
        assert col in result.columns


def test_importable_from_pingouin():
    from pingouin import yuen as _yuen
    assert callable(_yuen)


# ---------------------------------------------------------------------------
# Correctness of trimmed means
# ---------------------------------------------------------------------------


def test_trimmed_mean_matches_scipy():
    from scipy.stats import trim_mean
    x, y = _make_samples(n=40)
    result = yuen(x, y, trim=0.2)
    assert_allclose(result["tail_mean_x"].values[0], trim_mean(x, 0.2), rtol=1e-4)
    assert_allclose(result["tail_mean_y"].values[0], trim_mean(y, 0.2), rtol=1e-4)


def test_trimmed_mean_with_different_trim():
    from scipy.stats import trim_mean
    x, y = _make_samples(n=50)
    result = yuen(x, y, trim=0.1)
    assert_allclose(result["tail_mean_x"].values[0], trim_mean(x, 0.1), rtol=1e-4)


# ---------------------------------------------------------------------------
# Correctness of test statistic (manual)
# ---------------------------------------------------------------------------


def test_statistic_matches_manual():
    x, y = _make_samples(n=30, seed=1)
    nx, ny, trim = 30, 30, 0.2
    gx = int(np.floor(trim * nx))
    gy = int(np.floor(trim * ny))
    hx, hy = nx - 2 * gx, ny - 2 * gy

    xs, ys = np.sort(x), np.sort(y)
    tx = xs[gx:nx - gx].mean()
    ty = ys[gy:ny - gy].mean()

    wx = xs.copy(); wx[:gx] = xs[gx]; wx[nx - gx:] = xs[nx - gx - 1]
    wy = ys.copy(); wy[:gy] = ys[gy]; wy[ny - gy:] = ys[ny - gy - 1]

    swx = np.sum((wx - wx.mean()) ** 2)
    swy = np.sum((wy - wy.mean()) ** 2)

    dx = swx / (hx * (hx - 1))
    dy = swy / (hy * (hy - 1))

    T_manual = (tx - ty) / np.sqrt(dx + dy)
    dof_manual = (dx + dy) ** 2 / (dx**2 / (hx - 1) + dy**2 / (hy - 1))

    result = yuen(x, y, trim=0.2)
    assert_allclose(result["T"].values[0], T_manual, rtol=1e-5)
    assert_allclose(result["dof"].values[0], dof_manual, rtol=1e-5)


# ---------------------------------------------------------------------------
# p-value properties
# ---------------------------------------------------------------------------


def test_identical_samples_p_near_one():
    x = RNG.standard_normal(30)
    result = yuen(x, x)
    assert result["p_val"].values[0] == pytest.approx(1.0, abs=1e-8)


def test_identical_samples_T_zero():
    x = RNG.standard_normal(30)
    result = yuen(x, x)
    assert result["T"].values[0] == pytest.approx(0.0, abs=1e-8)


def test_large_shift_small_p():
    """With a large location shift and big samples, p-value should be tiny."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 200)
    y = rng.normal(3, 1, 200)
    result = yuen(x, y)
    assert result["p_val"].values[0] < 1e-10


def test_p_val_in_unit_interval():
    x, y = _make_samples()
    result = yuen(x, y)
    assert 0.0 <= result["p_val"].values[0] <= 1.0


def test_two_sided_p_equals_twice_one_sided():
    x, y = _make_samples(shift=0.5)
    r_two = yuen(x, y, alternative="two-sided")
    r_less = yuen(x, y, alternative="less")
    r_greater = yuen(x, y, alternative="greater")
    # two-sided = 2 * min(one-sided)
    assert_allclose(
        r_two["p_val"].values[0],
        2 * min(r_less["p_val"].values[0], r_greater["p_val"].values[0]),
        rtol=1e-8,
    )


def test_greater_less_complement():
    x, y = _make_samples(shift=0.5)
    r_greater = yuen(x, y, alternative="greater")
    r_less = yuen(x, y, alternative="less")
    assert_allclose(
        r_greater["p_val"].values[0] + r_less["p_val"].values[0],
        1.0,
        rtol=1e-8,
    )


# ---------------------------------------------------------------------------
# alternative parameter stored
# ---------------------------------------------------------------------------


def test_alternative_stored_two_sided():
    result = yuen(*_make_samples(), alternative="two-sided")
    assert result["alternative"].values[0] == "two-sided"


def test_alternative_stored_greater():
    result = yuen(*_make_samples(), alternative="greater")
    assert result["alternative"].values[0] == "greater"


def test_alternative_stored_less():
    result = yuen(*_make_samples(), alternative="less")
    assert result["alternative"].values[0] == "less"


# ---------------------------------------------------------------------------
# trim parameter
# ---------------------------------------------------------------------------


def test_trim_stored():
    result = yuen(*_make_samples(), trim=0.15)
    assert result["trim"].values[0] == 0.15


def test_different_trim_gives_different_stat():
    x, y = _make_samples(n=60)
    r02 = yuen(x, y, trim=0.2)
    r10 = yuen(x, y, trim=0.1)
    assert r02["T"].values[0] != r10["T"].values[0]


def test_dof_greater_with_more_data():
    """More observations → higher dof (roughly)."""
    x1, y1 = _make_samples(n=20, seed=0)
    x2, y2 = _make_samples(n=200, seed=0)
    r1 = yuen(x1, y1)
    r2 = yuen(x2, y2)
    assert r2["dof"].values[0] > r1["dof"].values[0]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_raises_trim_too_small():
    x, y = _make_samples()
    with pytest.raises(ValueError, match="trim"):
        yuen(x, y, trim=0.0)


def test_raises_trim_too_large():
    x, y = _make_samples()
    with pytest.raises(ValueError, match="trim"):
        yuen(x, y, trim=0.5)


def test_raises_invalid_alternative():
    x, y = _make_samples()
    with pytest.raises(ValueError, match="alternative"):
        yuen(x, y, alternative="invalid")


def test_raises_not_enough_obs():
    with pytest.raises(ValueError, match="Not enough observations"):
        yuen(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), trim=0.4)


def test_raises_2d_input():
    x = np.ones((5, 2))
    y = np.ones(5)
    with pytest.raises(ValueError, match="1-D"):
        yuen(x, y)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_propagate_returns_nan():
    x = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = np.ones(10)
    result = yuen(x, y, nan_policy="propagate")
    assert np.isnan(result["p_val"].values[0])


def test_nan_omit_works():
    x = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = np.ones(20)
    result = yuen(x, y, nan_policy="omit")
    assert not np.isnan(result["p_val"].values[0])


def test_nan_raise_raises():
    x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    y = np.ones(5)
    with pytest.raises(ValueError, match="NaN"):
        yuen(x, y, nan_policy="raise")


# ---------------------------------------------------------------------------
# Robustness property
# ---------------------------------------------------------------------------


def test_robust_to_outlier():
    """With a single extreme outlier, Yuen should still detect the shift."""
    rng = np.random.default_rng(99)
    x = rng.normal(0, 1, 50)
    y = rng.normal(1.5, 1, 50)
    # Add an extreme outlier to x
    x_out = np.append(x, 1000.0)
    y_out = np.append(y, 0.0)
    r_clean = yuen(x, y)
    r_outlier = yuen(x_out, y_out)
    # p-value should remain small despite the outlier
    assert r_outlier["p_val"].values[0] < 0.05
