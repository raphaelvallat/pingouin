import pytest
import numpy as np
from unittest import TestCase
from pingouin.correlation import (corr, rm_corr, partial_corr,
                                  skipped, distance_corr, bicor)
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
        assert np.round(stats['r'].to_numpy(), 3) == 0.512
        assert stats['outliers'].to_numpy() == 2
        # Changing the method using kwargs
        sk_sp = corr(x, y, method='skipped', corr_type='spearman')
        sk_pe = corr(x, y, method='skipped', corr_type='pearson')
        assert not sk_sp.equals(sk_pe)
        stats = corr(x, y, method='shepherd')
        assert stats['outliers'].to_numpy() == 2
        _, _, outliers = skipped(x, y, corr_type='pearson')
        assert outliers.size == x.size
        assert stats['n'].to_numpy() == 30
        stats = corr(x, y, method='percbend')
        assert np.round(stats['r'].to_numpy(), 3) == 0.484
        # Compare biweight correlation to astropy
        stats = corr(x, y, method='bicor')
        assert np.isclose(stats['r'].to_numpy(), 0.4951417784979)
        # Changing the value of C using kwargs
        stats = corr(x, y, method='bicor', c=5)
        assert np.isclose(stats['r'].to_numpy(), 0.4940706950017)
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
        # Compare BF10 with JASP
        df = read_dataset('pairwise_corr')
        stats = corr(df['Neuroticism'], df['Extraversion'])
        assert np.isclose(1 / float(stats['BF10'].to_numpy()), 1.478e-13)
        # When one column is a constant, the correlation is not defined
        # and Pingouin return a DataFrame full of NaN, except for ``n``
        x, y = [1, 1, 1], [1, 2, 3]
        stats = corr(x, y)
        assert stats.at['pearson', 'n']
        assert np.isnan(stats.at['pearson', 'r'])
        # Biweight midcorrelation returns NaN when MAD is not defined
        assert np.isnan(bicor(np.array([1, 1, 1, 1, 0, 1]), np.arange(6))[0])

    def test_partial_corr(self):
        """Test function partial_corr.
        Compare with the R package ppcor and JASP.
        """
        df = read_dataset('partial_corr')
        pc = partial_corr(data=df, x='x', y='y', covar='cv1')
        assert round(pc.at['pearson', 'r'], 3) == 0.568
        pc = df.partial_corr(x='x', y='y', covar='cv1', method='spearman')
        # Warning: Spearman slightly different than ppcor package, is this
        # caused by difference in Python / R when computing ranks?
        # assert pc.at['spearman', 'r'] == 0.578
        # Partial correlation of x and y controlling for multiple covariates
        pc = partial_corr(data=df, x='x', y='y', covar=['cv1'])
        pc = partial_corr(data=df, x='x', y='y', covar=['cv1', 'cv2', 'cv3'])
        assert round(pc.at['pearson', 'r'], 3) == 0.493
        pc = partial_corr(data=df, x='x', y='y', covar=['cv1', 'cv2', 'cv3'],
                          method='percbend')
        # Semi-partial correlation
        df.partial_corr(x='x', y='y', y_covar='cv1')
        pc = df.partial_corr(x='x', y='y', x_covar=['cv1', 'cv2', 'cv3'])
        assert round(pc.at['pearson', 'r'], 3) == 0.463
        pc = df.partial_corr(x='x', y='y', y_covar=['cv1', 'cv2', 'cv3'])
        assert round(pc.at['pearson', 'r'], 3) == 0.421
        partial_corr(data=df, x='x', y='y', x_covar='cv1',
                     y_covar=['cv2', 'cv3'], method='spearman')
        with pytest.raises(ValueError):
            partial_corr(data=df, x='x', y='y', covar='cv2', x_covar='cv1')
        with pytest.raises(AssertionError) as error_info:
            partial_corr(data=df, x='cv1', y='y', covar=['cv1', 'cv2'])
        assert str(error_info.value) == "x and covar must be independent"

    def test_rmcorr(self):
        """Test function rm_corr"""
        df = read_dataset('rm_corr')
        # Test again rmcorr R package.
        stats = rm_corr(data=df, x='pH', y='PacO2', subject='Subject').round(3)
        assert stats.at["rm_corr", "r"] == -0.507
        assert stats.at["rm_corr", "dof"] == 38
        assert np.allclose(np.round(stats.at["rm_corr", "CI95%"], 2),
                           [-0.71, -0.23])
        assert stats.at["rm_corr", "pval"] == 0.001
        # Test with less than 3 subjects (same behavior as R package)
        with pytest.raises(ValueError):
            rm_corr(data=df[df['Subject'].isin([1, 2])], x='pH', y='PacO2',
                    subject='Subject')

    def test_distance_corr(self):
        """Test function distance_corr
        We compare against the energy R package
        """
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
