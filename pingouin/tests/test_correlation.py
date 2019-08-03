import pytest
import numpy as np
from unittest import TestCase
from pingouin.correlation import (corr, rm_corr, partial_corr,
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
        # Disabled because of AppVeyor failure
        # assert corr(x, x).loc['pearson', 'BF10'] == str(np.inf)
        # Wrong argument
        with pytest.raises(ValueError):
            corr(x, y, method='error')
        with pytest.raises(ValueError):
            corr(x, y[:-10])
        # Compare with JASP
        df = read_dataset('pairwise_corr')
        stats = corr(df['Neuroticism'], df['Extraversion'])
        assert np.isclose(1 / float(stats['BF10'].values), 1.478e-13)

    def test_partial_corr(self):
        """Test function partial_corr.
        Compare with the R package ppcor and JASP."""
        df = read_dataset('partial_corr')
        pc = partial_corr(data=df, x='x', y='y', covar='cv1')
        assert pc.loc['pearson', 'r'] == 0.568
        pc = df.partial_corr(x='x', y='y', covar='cv1', method='spearman')
        # Warning: Spearman slightly different than ppcor package, is this
        # caused by difference in Python / R when computing ranks?
        # assert pc.loc['spearman', 'r'] == 0.578
        # Partial correlation of x and y controlling for multiple covariates
        pc = partial_corr(data=df, x='x', y='y', covar=['cv1'])
        pc = partial_corr(data=df, x='x', y='y', covar=['cv1', 'cv2', 'cv3'])
        assert pc.loc['pearson', 'r'] == 0.493
        pc = partial_corr(data=df, x='x', y='y', covar=['cv1', 'cv2', 'cv3'],
                          method='percbend')
        # Semi-partial correlation
        df.partial_corr(x='x', y='y', y_covar='cv1')
        pc = df.partial_corr(x='x', y='y', x_covar=['cv1', 'cv2', 'cv3'])
        assert pc.loc['pearson', 'r'] == 0.463
        pc = df.partial_corr(x='x', y='y', y_covar=['cv1', 'cv2', 'cv3'])
        assert pc.loc['pearson', 'r'] == 0.421
        partial_corr(data=df, x='x', y='y', x_covar='cv1',
                     y_covar=['cv2', 'cv3'], method='spearman')
        with pytest.raises(ValueError):
            partial_corr(data=df, x='x', y='y', covar='cv2', x_covar='cv1')

    def test_rmcorr(self):
        """Test function rm_corr"""
        df = read_dataset('rm_corr')
        # Test again rmcorr R package.
        stats = rm_corr(data=df, x='pH', y='PacO2', subject='Subject')
        assert stats.loc["rm_corr", "r"] == -0.507
        assert stats.loc["rm_corr", "dof"] == 38
        assert stats.loc["rm_corr", "CI95%"] == str([-0.71, -0.23])
        assert round(stats.loc["rm_corr", "pval"], 3) == 0.001
        # Test with less than 3 subjects (same behavior as R package)
        with pytest.raises(ValueError):
            rm_corr(data=df[df['Subject'].isin([1, 2])], x='pH', y='PacO2',
                    subject='Subject')

    def test_distance_corr(self):
        """Test function distance_corr
        We compare against the energy R package"""
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 9, 4, 4]
        dcor1 = distance_corr(a, b, n_boot=None)
        dcor, pval = distance_corr(a, b, seed=9)
        assert dcor1 == dcor
        assert np.round(dcor, 7) == 0.7626762
        assert 0.25 < pval < 0.40
        _, pval_low = distance_corr(a, b, seed=9, tail='less')
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
