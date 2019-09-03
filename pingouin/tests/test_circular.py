import pytest
import numpy as np
from unittest import TestCase
from pingouin import read_dataset
from pingouin.circular import (circ_axial, circ_corrcc, circ_corrcl, circ_mean,
                               circ_r, circ_rayleigh, circ_vtest)


class TestCircular(TestCase):
    """Test circular.py."""

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
        # Wrong argument
        with pytest.raises(ValueError):
            circ_corrcc(x, [0.52, 1.29, 2.87])

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
        # Wrong argument
        with pytest.raises(ValueError):
            circ_corrcl(x, [0.52, 1.29, 2.87])

    def test_circ_mean(self):
        """Test function circ_mean."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        mu = circ_mean(x)
        # Compare with the CircStats MATLAB toolbox
        assert np.round(mu, 3) == 1.013
        # Wrong argument
        with pytest.raises(ValueError):
            circ_mean(x, w=[0.1, 0.2, 0.3])

    def test_circ_r(self):
        """Test function circ_r."""
        x = [0.785, 1.570, 3.141, 0.839, 5.934]
        r = circ_r(x)
        # Compare with the CircStats MATLAB toolbox
        assert np.round(r, 3) == 0.497
        # Wrong argument
        with pytest.raises(ValueError):
            circ_r(x, w=[0.1, 0.2, 0.3])

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
        # Wrong argument
        with pytest.raises(ValueError):
            circ_rayleigh(x, w=[0.1, 0.2, 0.3])

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
        # Wrong argument
        with pytest.raises(ValueError):
            circ_vtest(x, w=[0.1, 0.2, 0.3])
