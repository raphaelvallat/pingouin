import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.correlation import (corr, rm_corr, intraclass_corr, partial_corr,
                                  skipped, distance_corr)
from pingouin import read_dataset


class TestCorrelation(TestCase):
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
        assert stats['r'].values == 0.512
        assert stats['outliers'].values == 2
        stats = corr(x, y, method='shepherd')
        assert stats['outliers'].values == 2
        _, _, outliers = skipped(x, y, method='pearson')
        assert outliers.size == x.size
        assert stats['n'].values == 30
        stats = corr(x, y, method='percbend')
        assert stats['r'].values == 0.484
        # Not normally distributed
        z = np.random.uniform(size=30)
        corr(x, z, method='pearson')
        # With NaN values
        x[3] = np.nan
        corr(x, y)
        # With the same array
        assert corr(x, x).loc['pearson', 'BF10'] == np.inf
        # Wrong argument
        with pytest.raises(ValueError):
            corr(x, y, method='error')
        with pytest.raises(ValueError):
            corr(x, y[:-10])
        # Compare with JASP
        df = read_dataset('pairwise_corr')
        stats = corr(df['Neuroticism'], df['Extraversion'])
        assert np.isclose(1 / stats['BF10'].values, 1.478e-13)
        # With more than 100 values to see if BF10 is computed
        xx, yy = np.random.multivariate_normal(mean, cov, 1500).T
        c1500 = corr(xx, yy)
        assert 'BF10' not in c1500.keys()

    def test_partial_corr(self):
        """Test function partial_corr"""
        np.random.seed(123)
        mean, cov = [4, 6, 2], [(1, .5, .3), (.5, 1, .2), (.3, .2, 1)]
        x, y, z = np.random.multivariate_normal(mean, cov, size=30).T
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        stats = partial_corr(data=df, x='x', y='y', covar='z')
        # Compare with R ppcorr
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
        # Test with less than 3 subjects (same behavior as R package)
        with pytest.raises(ValueError):
            rm_corr(data=df[df['Subject'].isin([1, 2])], x='pH', y='PacO2',
                    subject='Subject')

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
            intraclass_corr('error', 'Wine', 'Judge', 'Scores')
        with pytest.raises(ValueError):
            intraclass_corr(df, 'Wine', 'Judge', 'Judge')
        with pytest.raises(ValueError):
            intraclass_corr(df.drop(index=0), 'Wine', 'Judge', 'Scores')

    def test_distance_corr(self):
        """Test function distance_corr
        We compare against the energy R package"""
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 9, 4, 4]
        dcor1 = distance_corr(a, b, n_boot=None)
        dcor, pval = distance_corr(a, b, seed=9)
        assert dcor1 == dcor
        assert np.round(dcor, 7) == 0.7626762
        assert 0.30 < pval < 0.40
        _, pval_low = distance_corr(a, b, seed=9, tail='lower')
        assert pval < pval_low
        # With 2D arrays
        np.random.seed(123)
        a = np.random.random((10, 10))
        b = np.random.random((10, 10))
        dcor, pval = distance_corr(a, b, n_boot=500, seed=9)
        assert np.round(dcor, 5) == 0.87996
        assert 0.20 < pval < 0.30

        with pytest.raises(ValueError):
            a[2, 4] = np.nan
            distance_corr(a, b)
