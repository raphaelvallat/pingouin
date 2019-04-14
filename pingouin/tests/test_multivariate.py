import numpy as np
from unittest import TestCase
from pingouin.multivariate import multivariate_normality, multivariate_ttest


class TestMultivariate(TestCase):
    """Test multivariate.py."""

    def test_multivariate_normality(self):
        """Test function multivariate_normality."""
        np.random.seed(123)
        # With 2 variables
        mean, cov, n = [4, 6], [[1, .5], [.5, 1]], 30
        X = np.random.multivariate_normal(mean, cov, n)
        normal, p = multivariate_normality(X, alpha=.05)
        # Compare with the Matlab Robust Corr toolbox
        assert normal
        assert np.round(p, 3) == 0.752
        # With 3 variables
        mean, cov = [4, 6, 5], [[1., .5, .2], [.5, 1., .1], [.2, .1, 1.]]
        X = np.random.multivariate_normal(mean, cov, 50)
        normal, p = multivariate_normality(X, alpha=.01)

    def test_multivariate_ttest(self):
        """Test function multivariate_ttest."""
        np.random.seed(123)
        # With 2 variables
        mean, cov, n = [4, 6], [[1, .5], [.5, 1]], 30
        X = np.random.multivariate_normal(mean, cov, n)
        Y = np.random.multivariate_normal(mean, cov, n)
        multivariate_ttest(X, Y, paired=True)
        multivariate_ttest(X, Y, paired=False)
        multivariate_ttest(X, Y=None, paired=False)
