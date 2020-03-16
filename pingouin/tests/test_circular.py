import pytest
import numpy as np
from unittest import TestCase
from scipy.stats import circmean
from pingouin import read_dataset
from pingouin.circular import convert_angles, _checkangles
from pingouin.circular import (circ_axial, circ_corrcc, circ_corrcl, circ_mean,
                               circ_r, circ_rayleigh, circ_vtest)

np.random.seed(123)
a1 = [-1.2, 2.5, 3.1, -3.1, 0.2, -0.2]   # -np.pi / pi
a2 = np.pi + np.array(a1)                # 0 / 2 * np.pi
a3 = [150, 180, 32, 340, 54, 0, 360]     # 0 / 360 deg
a4 = [22, 23, 0.5, 1.2, 0, 24]           # Hours (0 - 24)
a5 = np.random.randint(0, 1440, (2, 4))  # Minutes, 2D array


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
        np.testing.assert_array_almost_equal(a1, convert_angles(a1, low=-np.pi,
                                                                high=np.pi))
        _checkangles(convert_angles(a2, low=0, high=2 * np.pi))
        _checkangles(convert_angles(a3, low=0, high=360))
        _checkangles(convert_angles(a4, low=0, high=24))
        _checkangles(convert_angles(a5, low=0, high=1440))
        _checkangles(convert_angles(a3, low=0, high=1440))

        convert_angles(a1, low=-np.pi, high=np.pi)
        convert_angles(a2, low=0, high=2 * np.pi)
        convert_angles(a3)
        convert_angles(a4, low=0, high=24)
        convert_angles(a5, low=0, high=1440)

        # Compare with scipy.stats.circmean
        def assert_circmean(x, low, high):
            m1 = convert_angles(circmean(x, low=low, high=high, axis=-1),
                                low, high)
            m2 = circ_mean(convert_angles(x, low, high), axis=-1)
            assert (np.round(m1, 4) == np.round(m2, 4)).all()

        assert_circmean(a1, low=-np.pi, high=np.pi)
        assert_circmean(a2, low=0, high=2 * np.pi)
        assert_circmean(a3, low=0, high=360)
        assert_circmean(a4, low=0, high=24)
        assert_circmean(a5, low=0, high=1440)

    def test_circ_axial(self):
        """Test function circ_axial."""
        df = read_dataset('circular')
        alpha = df['Orientation'].values
        alpha = circ_axial(np.deg2rad(alpha), 2)
        assert np.allclose(np.round(alpha, 4),
                           [0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124,
                           5.4978])

    def test_circ_corrcc(self):
        """Test function circ_corrcc."""
        x = [0.785, 1.570, 3.141, 3.839, 5.934]
        y = [0.593, 1.291, 2.879, 3.892, 6.108]
        r, pval = circ_corrcc(x, y)
        # Compare with the CircStats MATLAB toolbox
        assert r == 0.942
        assert np.round(pval, 3) == 0.066
        _, pval2 = circ_corrcc(x, y, tail='one-sided')
        assert pval2 == pval / 2
        # With correction for uniform marginals
        circ_corrcc(x, y, correction_uniform=True)

    def test_circ_corrcl(self):
        """Test function circ_corrcl."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        y = [1.593, 1.291, -0.248, -2.892, 0.102]
        r, pval = circ_corrcl(x, y)
        # Compare with the CircStats MATLAB toolbox
        assert r == 0.109
        assert np.round(pval, 3) == 0.971
        _, pval2 = circ_corrcl(x, y, tail='one-sided')
        assert pval2 == pval / 2

    def test_circ_mean(self):
        """Test function circ_mean."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        mu = circ_mean(x)
        # Compare with the CircStats MATLAB toolbox
        assert np.round(mu, 3) == 1.013

    def test_circ_r(self):
        """Test function circ_r."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        r = circ_r(x)
        # Compare with the CircStats MATLAB toolbox
        assert np.round(r, 3) == 0.497

    def test_circ_rayleigh(self):
        """Test function circ_rayleigh."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        z, pval = circ_rayleigh(x)
        # Compare with the CircStats MATLAB toolbox
        assert z == 1.236
        assert np.round(pval, 4) == 0.3048
        z, pval = circ_rayleigh(x, w=[.1, .2, .3, .4, .5], d=0.2)
        assert z == 0.278
        assert np.round(pval, 4) == 0.8070

    def test_circ_vtest(self):
        """Test function circ_vtest."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        v, pval = circ_vtest(x, dir=1)
        # Compare with the CircStats MATLAB toolbox
        assert v == 2.486
        assert np.round(pval, 4) == 0.0579
        v, pval = circ_vtest(x, dir=0.5, w=[.1, .2, .3, .4, .5], d=0.2)
        assert v == 0.637
        assert np.round(pval, 4) == 0.2309
