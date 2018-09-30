import pytest
import numpy as np
import pandas as pd
from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.correlation import corr, rm_corr, intraclass_corr, partial_corr
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
        df = read_dataset('pairwise_corr')
        stats = corr(df['Neuroticism'], df['Extraversion'])
        assert np.isclose(1 / stats['BF10'].values, 1.478e-13)

    def test_partial_corr(self):
        """Test function partial_corr"""
        np.random.seed(123)
        mean, cov = [4, 6, 2], [(1, .5, .3), (.5, 1, .2), (.3, .2, 1)]
        x, y, z = np.random.multivariate_normal(mean, cov, size=30).T
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        stats = partial_corr(data=df, x='x', y='y', covar='z')
        assert stats.loc['pearson', 'r'] == 0.568
        df['w'] = np.random.normal(size=30)
        df['v'] = np.random.normal(size=30)
        # Partial correlation of x and y controlling for z, w and v
        partial_corr(data=df, x='x', y='y', covar=['z'])
        partial_corr(data=df, x='x', y='y', covar=['z', 'w', 'v'])
        partial_corr(data=df, x='x', y='y', covar=['z', 'w', 'v'],
                     method='spearman')

    def test_rmcorr(self):
        """Test function rm_corr"""
        df = read_dataset('rm_corr')
        # Test again rmcorr R package.
        r, p, dof = rm_corr(data=df, x='pH', y='PacO2', subject='Subject')
        assert r == -0.507
        assert dof == 38
        assert np.round(p, 3) == 0.001

    def test_intraclass_corr(self):
        """Test function intraclass_corr"""
        df = read_dataset('icc')
        intraclass_corr(df, 'Wine', 'Judge', 'Scores', ci=.68)
        icc, ci = intraclass_corr(df, 'Wine', 'Judge', 'Scores')
        assert np.round(icc, 3) == 0.728
        assert ci[0] == .434
        assert ci[1] == .927
        with pytest.raises(ValueError):
            intraclass_corr(df, None, 'Judge', 'Scores')
        with pytest.raises(ValueError):
            intraclass_corr(None, 'Wine', 'Judge', 'Scores')
        with pytest.raises(ValueError):
            intraclass_corr(df, 'Wine', 'Judge', 'Judge')
        with pytest.raises(ValueError):
            intraclass_corr(df.drop(index=0), 'Wine', 'Judge', 'Scores')
