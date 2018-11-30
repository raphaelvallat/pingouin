import pytest
import numpy as np
from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.power import (power_ttest, power_anova, power_corr)


class TestPower(_TestPingouin):
    """Test power.py."""

    def test_power_ttest(self):
        """Test function power_ttest."""
        nx, ny = 20, 20
        d = 0.5
        power = power_ttest(d, nx, ny, paired=True, tail='one-sided')
        # Compare values with GPower 3.1.9
        assert np.allclose(power, 0.695)
        power = power_ttest(d, nx, ny, paired=False, tail='two-sided')
        assert np.allclose(power, 0.338)
        power_ttest(d, nx)

    def test_power_anova(self):
        """Test function power_anova."""
        ntot, ngroups = 60, 3
        eta = .20
        power = power_anova(eta, ntot, ngroups)
        assert np.allclose(power, 0.932)

    def test_power_corr(self):
        """Test function power_corr.
        Values are compared to the pwr R package.
        """
        # Two-sided
        assert np.allclose(power_corr(r=0.5, n=20), 0.6378746)
        assert np.allclose(power_corr(r=0.5, power=0.80), 28.24841)
        assert np.allclose(power_corr(n=20, power=0.80), 0.5821478)
        assert np.allclose(power_corr(r=0.5, n=20, power=0.80, alpha=None),
                           0.1377332, rtol=1e-03)

        # One-sided (e.g. = alternative = 'Greater' in R)
        assert np.allclose(power_corr(r=0.5, n=20, tail='one-sided'),
                           0.7509873)
        assert np.allclose(power_corr(r=0.5, power=0.80, tail='one-sided'),
                           22.60907)
        assert np.allclose(power_corr(n=20, power=0.80, tail='one-sided'),
                           0.5286949)

        # Error
        with pytest.raises(ValueError):
            power_corr(r=0.5)
