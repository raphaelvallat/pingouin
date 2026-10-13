import numpy as np
import pytest
from scipy.stats import circmean

from pingouin import read_dataset
from pingouin.circular import (
    _checkangles,
    circ_axial,
    circ_corrcc,
    circ_corrcl,
    circ_mean,
    circ_r,
    circ_rayleigh,
    circ_vtest,
    convert_angles,
)

a1 = [-1.2, 2.5, 3.1, -3.1, 0.2, -0.2]  # -np.pi / pi
a2 = np.pi + np.array(a1)  # 0 / 2 * np.pi
a3 = [150, 180, 32, 340, 54, 0, 360]  # 0 / 360 deg
a4 = [22, 23, 0.5, 1.2, 0, 24]  # Hours (0 - 24)
a5 = np.random.RandomState(123).randint(0, 1440, (2, 4))  # Minutes, 2D array
a3_nan = np.array([150, 180, 32, 340, 54, np.nan, 0, 360])
a5_nan = a5.astype(float)
a5_nan[1, 2] = np.nan


def test_helper_angles():
    """Test helper circular functions."""
    # Check angles
    _checkangles(a1)
    _checkangles(a2)
    _checkangles([-np.pi, np.pi])
    _checkangles([0, 2 * np.pi, np.nan])
    with pytest.raises(ValueError):
        _checkangles(a3)
    # Angles outside of the [-pi, pi] and [0, 2pi] ranges, even with a small spread
    for angles in [[10, 12, 13, 14, 15], [100, 101], [-4, -3.5], [-3, 3.5]]:
        with pytest.raises(ValueError, match="radians"):
            _checkangles(angles)
    with pytest.raises(ValueError, match="radians"):
        circ_mean([10, 12, 13, 14, 15])
    # Convert angles
    np.testing.assert_array_almost_equal(a1, convert_angles(a1, low=-np.pi, high=np.pi))
    np.testing.assert_array_almost_equal(
        a2, convert_angles(a2, low=0, high=2 * np.pi, positive=True)
    )
    for angles, high in [(a2, 2 * np.pi), (a3, 360), (a3_nan, 360), (a4, 24), (a5, 1440)]:
        # Default: [-pi, pi) range
        rad = convert_angles(angles, low=0, high=high)
        _checkangles(rad)
        assert np.nanmin(rad) >= -np.pi and np.nanmax(rad) < np.pi
        # positive=True: [0, 2pi] range
        rad_pos = convert_angles(angles, low=0, high=high, positive=True)
        _checkangles(rad_pos)
        np.testing.assert_allclose(rad_pos, np.asarray(angles) * 2 * np.pi / high)
        # The two ranges give the same angles, modulo 2pi
        np.testing.assert_allclose(np.exp(1j * rad), np.exp(1j * rad_pos))
    _checkangles(convert_angles(a5_nan, low=0, high=1440))
    np.testing.assert_array_equal(np.isnan(convert_angles(a5_nan, 0, 1440)), np.isnan(a5_nan))
    # Default low=0 and high=360 (degrees)
    np.testing.assert_allclose(convert_angles(a3), convert_angles(a3, low=0, high=360))
    np.testing.assert_allclose(
        convert_angles([0, 90, 180, 270]), [0, np.pi / 2, -np.pi, -np.pi / 2]
    )
    # 24-hours
    np.testing.assert_allclose(convert_angles([6, 18], low=0, high=24), [np.pi / 2, -np.pi / 2])
    assert convert_angles(a5, low=0, high=1440, positive=True).min() >= 0
    assert convert_angles(a5, low=0, high=1440).min() <= 0

    # Compare with scipy.stats.circmean
    def assert_circmean(x, low, high, axis=-1):
        m1 = convert_angles(
            circmean(x, low=low, high=high, axis=axis, nan_policy="omit"), low, high
        )
        m2 = circ_mean(convert_angles(x, low, high), axis=axis)
        assert (np.round(m1, 4) == np.round(m2, 4)).all()

    assert_circmean(a1, low=-np.pi, high=np.pi)
    assert_circmean(a2, low=0, high=2 * np.pi)
    assert_circmean(a3, low=0, high=360)
    assert_circmean(a3_nan, low=0, high=360)
    assert_circmean(a4, low=0, high=24)
    assert_circmean(a5, low=0, high=1440, axis=1)
    assert_circmean(a5, low=0, high=1440, axis=0)
    assert_circmean(a5, low=0, high=1440, axis=None)
    assert_circmean(a5_nan, low=0, high=1440, axis=-1)
    assert_circmean(a5_nan, low=0, high=1440, axis=0)
    assert_circmean(a5_nan, low=0, high=1440, axis=None)


def test_circ_axial():
    """Test function circ_axial."""
    df = read_dataset("circular")
    angles = df["Orientation"].to_numpy()
    angles = circ_axial(np.deg2rad(angles), 2)
    assert np.allclose(
        np.round(angles, 4), [0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]
    )


def test_circ_corrcc():
    """Test function circ_corrcc."""
    x = [0.785, 1.570, 3.141, 3.839, 5.934]
    y = [0.593, 1.291, 2.879, 3.892, 6.108]
    r, pval = circ_corrcc(x, y)
    # Compare with the CircStats MATLAB toolbox
    assert round(r, 3) == 0.942
    assert np.round(pval, 3) == 0.066
    # With correction for uniform marginals (regression values, also in the docstring)
    r_unif, pval_unif = circ_corrcc(x, y, correction_uniform=True)
    assert round(r_unif, 3) == 0.547
    assert round(pval_unif, 4) == 0.2859
    # Missing values are removed
    assert circ_corrcc(x + [np.nan], y + [1.0]) == (r, pval)


def test_circ_corrcl():
    """Test function circ_corrcl."""
    x = [0.785, 1.570, 3.141, 0.839, 5.934]
    y = [1.593, 1.291, -0.248, -2.892, 0.102]
    r, pval = circ_corrcl(x, y)
    # Compare with the CircStats MATLAB toolbox
    assert round(r, 3) == 0.109
    assert np.round(pval, 3) == 0.971
    # The circular variable must be in radians (the linear variable can have any range)
    # and the correlation is invariant to a linear transformation of the linear variable
    r_scaled, pval_scaled = circ_corrcl(x, np.array(y) * 100 + 5)
    assert np.isclose(r_scaled, r) and np.isclose(pval_scaled, pval)
    with pytest.raises(ValueError, match="radians"):
        circ_corrcl(np.rad2deg(x), y)


def test_circ_mean():
    """Test function circ_mean."""
    x = [0.785, 1.570, 3.141, 0.839, 5.934]
    x_nan = np.array([0.785, 1.570, 3.141, 0.839, 5.934, np.nan])
    # Compare with the CircStats MATLAB toolbox
    assert np.round(circ_mean(x), 3) == 1.013
    assert np.round(circ_mean(x_nan), 3) == 1.013
    # Binned data: w represents the number of incidences per bins, or "weights"
    nbins = 18  # Number of bins to divide the unit circle
    angles_bins = np.linspace(-np.pi, np.pi, nbins)
    w = np.array([2, 4, 2, 1, 3, 2, 4, 1, 1, 0, 2, 1, 1, 3, 4, 0, 2, 1])
    # Same as the unweighted mean of the angles repeated w times
    angles_rep = np.repeat(angles_bins, w)
    assert np.isclose(circ_mean(angles_bins, w), circ_mean(angles_rep))
    assert np.isclose(circ_mean(angles_bins, w), circmean(angles_rep, low=-np.pi, high=np.pi))


def test_circ_r():
    """Test function circ_r."""
    x = [0.785, 1.570, 3.141, 0.839, 5.934]
    x_nan = np.array([0.785, 1.570, 3.141, 0.839, 5.934, np.nan])
    # Compare with the CircStats MATLAB toolbox
    assert np.round(circ_r(x), 3) == 0.497
    assert np.round(circ_r(x_nan), 3) == 0.497


def test_circ_rayleigh():
    """Test function circ_rayleigh."""
    x = [0.785, 1.570, 3.141, 0.839, 5.934]
    z, pval = circ_rayleigh(x)
    # Compare with the CircStats MATLAB toolbox
    assert round(z, 3) == 1.236
    assert round(pval, 4) == 0.3048
    z, pval = circ_rayleigh(x, w=[0.1, 0.2, 0.3, 0.4, 0.5], d=0.2)
    assert round(z, 3) == 0.278
    assert round(pval, 4) == 0.8070
    # Missing values are not counted in the sample size
    assert circ_rayleigh(x + [np.nan]) == circ_rayleigh(x)
    w = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert circ_rayleigh(x + [np.nan], w=w + [1], d=0.2) == circ_rayleigh(x, w=w, d=0.2)


def test_circ_vtest():
    """Test function circ_vtest."""
    x = [0.785, 1.570, 3.141, 0.839, 5.934]
    v, pval = circ_vtest(x, dir=1)
    # Compare with the CircStats MATLAB toolbox
    assert round(v, 3) == 2.486
    assert round(pval, 4) == 0.0579
    v, pval = circ_vtest(x, dir=0.5, w=[0.1, 0.2, 0.3, 0.4, 0.5], d=0.2)
    assert round(v, 3) == 0.637
    assert round(pval, 4) == 0.2309
    # Missing values are not counted in the sample size
    assert circ_vtest(x + [np.nan], dir=1) == circ_vtest(x, dir=1)
