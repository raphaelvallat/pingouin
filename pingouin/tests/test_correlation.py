import pytest
import numpy as np
from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.correlation import corr


class TestCorrelation(_TestPingouin):
    """Test correlation.py."""

    def test_corr(self):
        """Test function corr"""
        mean, cov = [4, 6], [(1, .5), (.5, 1)]
        x, y = np.random.multivariate_normal(mean, cov, 30).T
        # Add one outlier
        x[3] = 12
        corr(x, y, method='pearson', tail='one-sided')
        corr(x, y, method='spearman', tail='two-sided')
        corr(x, y, method='shepherd', tail='two-sided')
        corr(x, y, method='kendall')
        corr(x, y, method='percbend')
        # Not normally distributed
        z = np.random.uniform(size=30)
        corr(x, z, method='pearson')
        # With NaN values
        x[3] = np.nan
        corr(x, y)
        # Wrong argument
        with pytest.raises(ValueError):
            corr(x, y, method='error')
        with pytest.raises(ValueError):
            corr(x, y[:-10])
