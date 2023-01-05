import pytest
import numpy as np
from unittest import TestCase
from scipy.stats import circmean
from pingouin import read_dataset
from pingouin.circular import convert_angles, _checkangles
from pingouin.circular import (
    circ_axial,
    circ_corrcc,
    circ_corrcl,
    circ_mean,
    circ_r,
    circ_rayleigh,
    circ_vtest,
)

np.random.seed(123)
a1 = [-1.2, 2.5, 3.1, -3.1, 0.2, -0.2]  # -np.pi / pi
a2 = np.pi + np.array(a1)  # 0 / 2 * np.pi
a3 = [150, 180, 32, 340, 54, 0, 360]  # 0 / 360 deg
a4 = [22, 23, 0.5, 1.2, 0, 24]  # Hours (0 - 24)
a5 = np.random.randint(0, 1440, (2, 4))  # Minutes, 2D array
a3_nan = np.array([150, 180, 32, 340, 54, np.nan, 0, 360])
a5_nan = a5.astype(float)
a5_nan[1, 2] = np.nan


class TestCircular(TestCase):
    """Test circular.py."""

    def test_helper_angles(self):
        """Test helper circular functions."""
        # Check angles
        _checkangles(a1)
        _checkangles(a2, axis=None)
        with pytest.raises(ValueError):
            _checkangles(a3)
        with pytest.raises(ValueError):
            _checkangles(a3, axis=None)
        # Convert angles
        np.testing.assert_array_almost_equal(a1, convert_angles(a1, low=-np.pi, high=np.pi))
        np.testing.assert_array_almost_equal(
            a2, convert_angles(a2, low=0, high=2 * np.pi, positive=True)
        )
        _checkangles(convert_angles(a2, low=0, high=2 * np.pi))
        _checkangles(convert_angles(a3, low=0, high=360))
        _checkangles(convert_angles(a3_nan, low=0, high=360))
        _checkangles(convert_angles(a4, low=0, high=24))
        _checkangles(convert_angles(a5, low=0, high=1440))
        _checkangles(convert_angles(a5_nan, low=0, high=1440))
        _checkangles(convert_angles(a5, low=0, high=1440, positive=True))

        convert_angles(a1, low=-np.pi, high=np.pi)
        convert_angles(a2, low=0, high=2 * np.pi)
        convert_angles(a3)
        convert_angles(a4, low=0, high=24)
        convert_angles(a5, low=0, high=1440)
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

    def test_circ_axial(self):
        """Test function circ_axial."""
        df = read_dataset("circular")
        angles = df["Orientation"].to_numpy()
        angles = circ_axial(np.deg2rad(angles), 2)
        assert np.allclose(
            np.round(angles, 4), [0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]
        )

    def test_circ_corrcc(self):
        """Test function circ_corrcc."""
        x = [0.785, 1.570, 3.141, 3.839, 5.934]
        y = [0.593, 1.291, 2.879, 3.892, 6.108]
        r, pval = circ_corrcc(x, y)
        # Compare with the CircStats MATLAB toolbox
        assert round(r, 3) == 0.942
        assert np.round(pval, 3) == 0.066
        # With correction for uniform marginals
        circ_corrcc(x, y, correction_uniform=True)

    def test_circ_corrcl(self):
        """Test function circ_corrcl."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        y = [1.593, 1.291, -0.248, -2.892, 0.102]
        r, pval = circ_corrcl(x, y)
        # Compare with the CircStats MATLAB toolbox
        assert round(r, 3) == 0.109
        assert np.round(pval, 3) == 0.971

    def test_circ_mean(self):
        """Test function circ_mean."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        x_nan = np.array([0.785, 1.570, 3.141, 0.839, 5.934, np.nan])
        # Compare with the CircStats MATLAB toolbox
        assert np.round(circ_mean(x), 3) == 1.013
        assert np.round(circ_mean(x_nan), 3) == 1.013
        # Binned data
        np.random.seed(123)
        nbins = 18  # Number of bins to divide the unit circle
        angles_bins = np.linspace(-np.pi, np.pi, nbins)
        # w represents the number of incidences per bins, or "weights".
        w = np.random.randint(low=0, high=5, size=angles_bins.size)
        assert round(circ_mean(angles_bins, w), 4) == -2.5355

    def test_circ_r(self):
        """Test function circ_r."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        x_nan = np.array([0.785, 1.570, 3.141, 0.839, 5.934, np.nan])
        # Compare with the CircStats MATLAB toolbox
        assert np.round(circ_r(x), 3) == 0.497
        assert np.round(circ_r(x_nan), 3) == 0.497

    def test_circ_rayleigh(self):
        """Test function circ_rayleigh."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        z, pval = circ_rayleigh(x)
        # Compare with the CircStats MATLAB toolbox
        assert round(z, 3) == 1.236
        assert round(pval, 4) == 0.3048
        z, pval = circ_rayleigh(x, w=[0.1, 0.2, 0.3, 0.4, 0.5], d=0.2)
        assert round(z, 3) == 0.278
        assert round(pval, 4) == 0.8070

    def test_circ_vtest(self):
        """Test function circ_vtest."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        v, pval = circ_vtest(x, dir=1)
        # Compare with the CircStats MATLAB toolbox
        assert round(v, 3) == 2.486
        assert round(pval, 4) == 0.0579
        v, pval = circ_vtest(x, dir=0.5, w=[0.1, 0.2, 0.3, 0.4, 0.5], d=0.2)
        assert round(v, 3) == 0.637
        assert round(pval, 4) == 0.2309
