import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from unittest import TestCase
from pingouin.regression import (linear_regression, logistic_regression,
                                 mediation_analysis)
from pingouin.datasets import read_dataset

from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

df = read_dataset('mediation')


class TestRegression(TestCase):
    """Test regression.py."""

    def test_linear_regression(self):
        """Test function linear_regression."""

        # Simple regression
        lm = linear_regression(df['X'], df['Y'])  # Pingouin
        sc = linregress(df['X'].values, df['Y'].values)  # SciPy
        assert_equal(lm['names'].values, ['Intercept', 'X'])
        assert_almost_equal(lm['coef'][1], sc.slope)
        assert_almost_equal(lm['coef'][0], sc.intercept)
        assert_almost_equal(lm['se'][1], sc.stderr)
        assert_almost_equal(lm['pval'][1], sc.pvalue)
        assert_almost_equal(np.sqrt(lm['r2'][0]), sc.rvalue)

        # Multiple regression with intercept
        X = df[['X', 'M']].values
        y = df['Y'].values
        lm = linear_regression(X, y, as_dataframe=False)  # Pingouin
        sk = LinearRegression(fit_intercept=True).fit(X, y)  # SkLearn
        assert_equal(lm['names'], ['Intercept', 'x1', 'x2'])
        assert_almost_equal(lm['coef'][1:], sk.coef_)
        assert_almost_equal(lm['coef'][0], sk.intercept_)
        assert_almost_equal(sk.score(X, y), lm['r2'])
        # Compare values to JASP
        assert_equal([.605, .110, .101], np.round(lm['se'], 3))
        assert_equal([3.145, 0.361, 6.321], np.round(lm['T'], 3))
        assert_equal([0.002, 0.719, 0.000], np.round(lm['pval'], 3))
        assert_equal([.703, -.178, .436], np.round(lm['CI[2.5%]'], 3))
        assert_equal([3.106, .257, .835], np.round(lm['CI[97.5%]'], 3))

        # No intercept
        lm = linear_regression(X, y, add_intercept=False, as_dataframe=False)
        sk = LinearRegression(fit_intercept=False).fit(X, y)
        assert_almost_equal(lm['coef'], sk.coef_)
        assert_almost_equal(sk.score(X, y), lm['r2'])

        # Test other arguments
        linear_regression(df[['X', 'M']], df['Y'], coef_only=True)
        linear_regression(df[['X', 'M']], df['Y'], alpha=0.01)
        linear_regression(df[['X', 'M']], df['Y'], alpha=0.10)

    def test_logistic_regression(self):
        """Test function logistic_regression."""

        # Simple regression
        lom = logistic_regression(df['X'], df['Ybin'], as_dataframe=False)
        # Compare to JASP
        assert_equal(np.round(lom['coef'], 1), [1.3, -0.2])
        assert_equal(np.round(lom['se'], 2), [0.76, 0.12])
        assert_equal(np.round(lom['z'], 1), [1.7, -1.6])
        assert_equal(np.round(lom['pval'], 1), [0.1, 0.1])
        assert_equal(np.round(lom['CI[2.5%]'], 1), [-.2, -.4])
        assert_equal(np.round(lom['CI[97.5%]'], 1), [2.8, 0.0])

        # Multiple predictors
        X = df[['X', 'M']].values
        y = df['Ybin'].values
        lom = logistic_regression(X, y)  # Pingouin
        # Compare against JASP
        assert_equal(np.round(lom['coef'].values, 1), [1.3, -0.2, -0.0])
        assert_equal(np.round(lom['se'].values, 2), [0.78, 0.14, 0.13])
        assert_equal(np.round(lom['z'].values, 1), [1.7, -1.4, -0.1])
        assert_equal(np.round(lom['pval'].values, 1), [0.1, 0.2, 1.])
        assert_equal(np.round(lom['CI[2.5%]'].values, 1), [-.2, -.5, -.3])
        assert_equal(np.round(lom['CI[97.5%]'].values, 1), [2.8, 0.1, 0.2])

        # Test other arguments
        c = logistic_regression(df[['X', 'M']], df['Ybin'], coef_only=True)
        assert_equal(np.round(c, 1), [1.3, -0.2, -0.0])

        # Test **kwargs
        logistic_regression(X, y, solver='sag', C=10, max_iter=10000)
        logistic_regression(X, y, solver='sag', multi_class='auto')

        with pytest.raises(ValueError):
            y[3] = 2
            logistic_regression(X, y)

    def test_mediation_analysis(self):
        """Test function mediation_analysis.
        """
        ma = mediation_analysis(data=df, x='X', m='M', y='Y', n_boot=500)

        # Compare against R package mediation
        assert ma['Beta'][0] == 0.5610
        assert ma['Beta'][1] == 0.6542
        assert ma['Beta'][2] == 0.3961
        assert ma['Beta'][3] == 0.0396
        assert ma['Beta'][4] == 0.3565

        _, dist = mediation_analysis(data=df, x='X', m='M', y='Y', n_boot=1000,
                                     return_dist=True)
        assert dist.size == 1000
        mediation_analysis(data=df, x='X', m='M', y='Y', alpha=0.01)

        # Check with a binary mediator
        ma = mediation_analysis(data=df, x='X', m='Mbin', y='Y', n_boot=2000)
        assert_almost_equal(ma['Beta'][0], -0.0208, decimal=2)

        # Indirect effect
        assert_almost_equal(ma['Beta'][4], 0.0033, decimal=2)
        assert ma['Sig'][4] == 'No'

        # Direct effect
        assert_almost_equal(ma['Beta'][3], 0.3956, decimal=2)
        assert_almost_equal(ma['CI[2.5%]'][3], 0.1714, decimal=2)
        assert_almost_equal(ma['CI[97.5%]'][3], 0.617, decimal=1)
        assert ma['Sig'][3] == 'Yes'
