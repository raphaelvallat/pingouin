import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.regression import (linear_regression, logistic_regression,
                                 mediation_analysis)
from pingouin.datasets import read_dataset

from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

df = read_dataset('mediation')

# For logistic regression test
np.random.seed(123)
df['Ybin'] = np.random.randint(0, 2, size=df.shape[0])


class TestRegression(_TestPingouin):
    """Test regression.py."""

    def test_linear_regression(self):
        """Test function linear_regression."""

        # Simple regression
        lm = linear_regression(df['X'], df['Y'])  # Pingouin
        sc = linregress(df['X'].values, df['Y'].values)  # SciPy
        assert_almost_equal(lm['coef'][1], sc.slope)
        assert_almost_equal(lm['coef'][0], sc.intercept)
        assert_almost_equal(lm['se'][1], sc.stderr)
        assert_almost_equal(lm['pvals'][1], sc.pvalue, decimal=5)
        assert_almost_equal(np.sqrt(lm['rsquared']), sc.rvalue)

        # Multiple regression with intercept
        X = df[['X', 'M']].values
        y = df['Y'].values
        lm = linear_regression(X, y)  # Pingouin
        sk = LinearRegression(fit_intercept=True).fit(X, y)  # SkLearn
        assert_almost_equal(lm['coef'][1:], sk.coef_)
        assert_almost_equal(lm['coef'][0], sk.intercept_)
        assert_almost_equal(sk.score(X, y), lm['rsquared'])

        # No intercept
        lm = linear_regression(X, y, add_intercept=False)
        sk = LinearRegression(fit_intercept=False).fit(X, y)
        assert_almost_equal(lm['coef'], sk.coef_)
        assert_almost_equal(sk.score(X, y), lm['rsquared'])

        # Test other arguments
        linear_regression(df[['X', 'M']], df['Y'], coef_only=True)
        linear_regression(df[['X', 'M']], df['Y'], alpha=0.01)
        linear_regression(df[['X', 'M']], df['Y'], alpha=0.10)

    def test_logistic_regression(self):
        """Test function logistic_regression."""

        # Simple regression
        lom = logistic_regression(df['X'], df['Ybin'])  # Pingouin
        # Compare to statsmodels
        assert_almost_equal(lom['coef'], [-0.0761, -0.0208], decimal=2)
        assert_almost_equal(lom['se'], [0.722, 0.116], decimal=2)
        assert_almost_equal(lom['z'], [-0.105, -0.180], decimal=2)
        assert_almost_equal(lom['pvals'], [0.916, 0.858], decimal=2)
        assert_almost_equal(lom['ll'], [-1.491, -0.248], decimal=2)
        assert_almost_equal(lom['ul'], [1.339, 0.206], decimal=2)

        # Multiple predictors
        X = df[['X', 'M']].values
        y = df['Ybin'].values
        lom = logistic_regression(X, y)  # Pingouin
        assert_almost_equal(lom['coef'], [-0.0339, -0.0051, -0.0281],
                            decimal=2)
        assert_almost_equal(lom['se'], [0.746, 0.135, 0.124], decimal=2)
        assert_almost_equal(lom['z'], [-0.045, -0.037, -0.227], decimal=2)
        assert_almost_equal(lom['pvals'], [0.964, 0.970, 0.821], decimal=2)
        assert_almost_equal(lom['ll'], [-1.496, -0.270, -0.271], decimal=2)
        assert_almost_equal(lom['ul'], [1.428, 0.260, 0.215], decimal=2)

        # Test other arguments
        logistic_regression(df[['X', 'M']], df['Ybin'], coef_only=True)

        with pytest.raises(ValueError):
            y[3] = 2
            logistic_regression(X, y)

    def test_mediation_analysis(self):
        """Test function mediation_analysis.
        TODO: compare logistic mediator to R package mediation.
        """
        ma = mediation_analysis(data=df, x='X', m='M', y='Y', n_boot=500)
        assert ma['Beta'][0] == 0.5610
        assert ma['Beta'][2] == 0.3961
        assert ma['Beta'][3] == 0.0396
        assert ma['Beta'][4] == 0.3565

        _, dist = mediation_analysis(data=df, x='X', m='M', y='Y', n_boot=1000,
                                     return_dist=True)
        assert dist.size == 1000
        mediation_analysis(data=df, x='X', m='M', y='Y', alpha=0.01)

        # Check with a binary mediator
        np.random.seed(123)
        df['M1'] = np.random.randint(0, 2, df.shape[0])
        ma = mediation_analysis(data=df, x='X', m='M1', y='Y', n_boot=500)
        assert_almost_equal(ma['Beta'][0], -0.0208, decimal=2)
        assert_almost_equal(ma['Beta'][4], 0.0027, decimal=2)
        # Check significance
        assert ma['Sig'][0] == 'No'
        assert ma['Sig'][1] == 'No'
        assert ma['Sig'][3] == 'Yes'
        assert ma['Sig'][4] == 'No'
