import numpy as np
from unittest import TestCase
from scipy.stats import pearsonr
from pingouin.parametric import ttest
from pingouin.bayesian import bayesfactor_ttest, bayesfactor_binom
from pingouin.bayesian import bayesfactor_pearson as bfp

from pytest import approx

np.random.seed(1234)
x = np.random.normal(size=100)
y = np.random.normal(size=100)
z = np.random.normal(loc=.5, size=100)
v, w = np.random.multivariate_normal([0, 0], [[1, .8], [.8, 1]], 100).T

class TestBayesian(TestCase):
    """Test bayesian.py."""

    def test_bayesfactor_ttest(self):
        """Test function bayesfactor_ttest."""
        # check for approximate equality with 1e-3 tolerance
        # (as this is how we store the values here)
        appr = lambda x: approx(x, abs=1e-3)
        assert bayesfactor_ttest(3.5, 20, 20) == appr(26.743)
        assert bayesfactor_ttest(3.5, 20) == appr(17.185)
        assert bayesfactor_ttest(3.5, 20, 1) == appr(17.185)
        # Compare against BayesFactor::testBF
        # >>> ttestBF(df$x, df$y, paired = FALSE, rscale = "medium")
        assert ttest(x, y).at['T-test', 'BF10'] == '0.183'
        assert ttest(x, y, paired=True).at['T-test', 'BF10'] == '0.135'
        assert int(float(ttest(x, z).at['T-test', 'BF10'])) == 1290
        assert int(float(ttest(x, z, paired=True).at['T-test', 'BF10'])) == 420
        # Now check the alternative tails
        assert bayesfactor_ttest(3.5, 20, 20, tail='greater') > 1
        assert bayesfactor_ttest(3.5, 20, 20, tail='less') < 1
        assert bayesfactor_ttest(-3.5, 20, 20, tail='greater') < 1
        assert bayesfactor_ttest(-3.5, 20, 20, tail='less') > 1
        # Check with wrong T-value
        assert np.isnan(bayesfactor_ttest(np.nan, 20, paired=True))

    def test_bayesfactor_pearson(self):
        """Test function bayesfactor_pearson."""
        # Compare the analytical solution to JASP / R (method='ly')
        # Similar to JASP with kappa=1, or correlationBF with rscale='wide'
        # check for approximate equality with 1e-3 tolerance
        # (as this is how we store the values here)
        appr = lambda x: approx(x, abs=1e-3)
        assert bfp(0.1, 83) == appr(0.204)
        assert bfp(-0.1, 83) == appr(0.204)
        assert bfp(0.1, 83, tail='one-sided') == appr(0.332)
        assert bfp(0.1, 83, tail='greater') == appr(0.332)
        assert bfp(0.1, 83, tail='pos') == appr(0.332)
        assert bfp(0.1, 83, tail='less') == appr(0.076)
        assert bfp(0.1, 83, tail='neg') == appr(0.076)
        assert bfp(-0.1, 83, tail='one-sided') == appr(0.332)
        assert bfp(-0.1, 83, tail='pos') == appr(0.076)

        # Example 2. Compare with JASP.
        r, _ = pearsonr(x, y)
        n = 100
        assert bfp(r, n) == appr(0.174)
        assert bfp(r, n, tail='g') == appr(0.275)
        assert bfp(r, n, tail='g', method='wetzels') == appr(0.275)
        assert bfp(r, n, tail='l') == appr(0.073)
        r, _ = pearsonr(v, w)
        appr = lambda x: approx(x, rel=1e-3) # relative tolerance here
        assert bfp(r, n) == appr(2.321e+22)
        assert bfp(r, n, tail='g') == appr(4.643e+22)
        # assert bfp(r, n, tail='l')) == 1.677e-26

        # Compare the integral solving method (Wetzels)
        appr = lambda x: approx(x, abs=1e-3) # back to absolute tolerance
        assert bfp(0.6, 20, method='wetzels') == appr(8.221)
        assert bfp(-0.6, 20, method='wetzels') == appr(8.221)
        assert bfp(0.6, 10, method='wetzels') == appr(1.278)

        # Wrong input
        assert np.isnan(bfp(np.nan, 20))
        assert np.isnan(bfp(0.8, 1))
        assert np.isnan(bfp(np.inf, 1))
        assert np.isinf(bfp(-1, 100))

    def test_bayesfactor_binom(self):
        """Test function bayesfactor_binom.
        Compare to http://pcl.missouri.edu/bf-binomial.
        See also docstring of the function for a comparison with Wikipedia.
        """
        def bf10(x):
            return approx(1 / x, rel=1e-5)
        assert bayesfactor_binom(16, 20) == bf10(0.09703159)
        assert bayesfactor_binom(16, 20, 0.8) == bf10(4.582187)
        assert bayesfactor_binom(100, 1000, 0.1) == bf10(42.05881)
