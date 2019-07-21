import numpy as np
from unittest import TestCase
from scipy.stats import pearsonr
from pingouin.parametric import ttest
from pingouin.bayesian import bayesfactor_ttest, bayesfactor_binom
from pingouin.bayesian import bayesfactor_pearson as bfp

np.random.seed(1234)
x = np.random.normal(size=100)
y = np.random.normal(size=100)
z = np.random.normal(loc=.5, size=100)
v, w = np.random.multivariate_normal([0, 0], [[1, .8], [.8, 1]], 100).T


class TestBayesian(TestCase):
    """Test bayesian.py."""

    def test_bayesfactor_ttest(self):
        """Test function bayesfactor_ttest."""
        assert float(bayesfactor_ttest(3.5, 20, 20)) == 26.743
        assert float(bayesfactor_ttest(3.5, 20)) == 17.185
        assert float(bayesfactor_ttest(3.5, 20, 1)) == 17.185
        # Compare against BayesFactor::testBF
        # >>> ttestBF(df$x, df$y, paired = FALSE, rscale = "medium")
        assert ttest(x, y).at['T-test', 'BF10'] == '0.183'
        assert ttest(x, y, paired=True).at['T-test', 'BF10'] == '0.135'
        assert int(float(ttest(x, z).at['T-test', 'BF10'])) == 1290
        assert int(float(ttest(x, z, paired=True).at['T-test', 'BF10'])) == 420
        # Now check the alternative tails
        assert float(bayesfactor_ttest(3.5, 20, 20, tail='greater')) > 1
        assert float(bayesfactor_ttest(3.5, 20, 20, tail='less')) < 1
        assert float(bayesfactor_ttest(-3.5, 20, 20, tail='greater')) < 1
        assert float(bayesfactor_ttest(-3.5, 20, 20, tail='less')) > 1
        # Check with wrong T-value
        assert bayesfactor_ttest(np.nan, 20, paired=True) == 'nan'

    def test_bayesfactor_pearson(self):
        """Test function bayesfactor_pearson."""
        # Compare the analytical solution to JASP / R (method='ly')
        # Similar to JASP with kappa=1, or correlationBF with rscale='wide'
        assert float(bfp(0.1, 83)) == 0.204
        assert float(bfp(-0.1, 83)) == 0.204
        assert float(bfp(0.1, 83, tail='one-sided')) == 0.332
        assert float(bfp(0.1, 83, tail='greater')) == 0.332
        assert float(bfp(0.1, 83, tail='pos')) == 0.332
        assert float(bfp(0.1, 83, tail='less')) == 0.076
        assert float(bfp(0.1, 83, tail='neg')) == 0.076
        assert float(bfp(-0.1, 83, tail='one-sided')) == 0.332
        assert float(bfp(-0.1, 83, tail='pos')) == 0.076

        # Example 2. Compare with JASP.
        r, _ = pearsonr(x, y)
        n = 100
        assert float(bfp(r, n)) == 0.174
        assert float(bfp(r, n, tail='g')) == 0.275
        assert float(bfp(r, n, tail='g', method='wetzels')) == 0.275
        assert float(bfp(r, n, tail='l')) == 0.073
        r, _ = pearsonr(v, w)
        assert float(bfp(r, n)) == 2.321e+22
        assert float(bfp(r, n, tail='g')) == 4.643e+22
        # assert float(bfp(r, n, tail='l')) == 1.677e-26

        # Compare the integral solving method (Wetzels)
        assert float(bfp(0.6, 20, method='wetzels')) == 8.221
        assert float(bfp(-0.6, 20, method='wetzels')) == 8.221
        assert float(bfp(0.6, 10, method='wetzels')) == 1.278

        # Wrong input
        assert bfp(np.nan, 20) == 'nan'
        assert bfp(0.8, 1) == 'nan'
        assert bfp(np.inf, 1) == 'nan'
        assert bfp(-1, 100) == 'inf'

    def test_bayesfactor_binom(self):
        """Test function bayesfactor_binom.
        Compare to http://pcl.missouri.edu/bf-binomial.
        See also docstring of the function for a comparison with Wikipedia.
        """
        def bf10(x):
            return str(round(1 / float(x), 3))
        assert bayesfactor_binom(16, 20) == bf10(0.09703159)
        assert bayesfactor_binom(16, 20, 0.8) == bf10(4.582187)
        assert bayesfactor_binom(100, 1000, 0.1) == bf10(42.05881)
