import numpy as np
from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.power import (ttest_power, anova_power)


class TestPower(_TestPingouin):
    """Test power.py."""

    def test_ttest_power(self):
        """Test function ttest_power."""
        nx, ny = 20, 20
        d = 0.5
        power = ttest_power(d, nx, ny, paired=True, tail='one-sided')
        # Compare values with GPower 3.1.9
        assert np.allclose(power, 0.695)
        power = ttest_power(d, nx, ny, paired=False, tail='two-sided')
        assert np.allclose(power, 0.338)
        ttest_power(d, nx)

    def test_anova_power(self):
        """Test function anova_power."""
        ntot, ngroups = 60, 3
        eta = .20
        power = anova_power(eta, ntot, ngroups)
        assert np.allclose(power, 0.932)
