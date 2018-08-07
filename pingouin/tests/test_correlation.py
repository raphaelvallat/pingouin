import pytest
import numpy as np
from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.correlation import corr, rm_corr
from pingouin.datasets import read_dataset


class TestCorrelation(_TestPingouin):
    """Test correlation.py."""

    def test_corr(self):
        """Test function corr"""
        np.random.seed(123)
        mean, cov = [4, 6], [(1, .6), (.6, 1)]
        x, y = np.random.multivariate_normal(mean, cov, 30).T
        x[3], y[5] = 12, -8
        corr(x, y, method='pearson', tail='one-sided')
        corr(x, y, method='spearman', tail='two-sided')
        corr(x, y, method='kendall')
        corr(x, y, method='shepherd', tail='two-sided')
        # Compare with robust corr toolbox
        stats = corr(x, y, method='skipped')
        assert np.round(stats['r'].values, 3) == 0.512
        stats = corr(x, y, method='percbend')
        assert np.round(stats['r'].values, 3) == 0.484
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
        # Compare with JASP
        df = read_dataset('dolan2009')
        stats = corr(df['Neuroticism'], df['Extraversion'])
        assert np.isclose(1 / stats['BF10'].values, 1.478e-13)

    def test_rmcorr(self):
        """Test function rm_corr"""
        df = read_dataset('bland1995')
        # Test again rmcorr R package.
        r, p, dof = rm_corr(data=df, x='pH', y='PacO2', subject='Subject')
        assert r == -0.507
        assert dof == 38
        assert np.round(p, 3) == 0.001
