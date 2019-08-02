import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from unittest import TestCase
from pingouin.regression import (linear_regression, logistic_regression,
                                 mediation_analysis, _pval_from_bootci)
from pingouin import read_dataset

from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

df = read_dataset('mediation')
df['Zero'] = 0
df['One'] = 1
df['Two'] = 2
df_nan = df.copy()
df_nan.loc[1, 'M'] = np.nan
df_nan.loc[10, 'X'] = np.nan
df_nan.loc[12, ['Y', 'Ybin']] = np.nan


class TestRegression(TestCase):
    """Test regression.py."""

    def test_linear_regression(self):
        """Test function linear_regression."""

        # Simple regression
        lm = linear_regression(df['X'], df['Y'])  # Pingouin
        linear_regression(df['X'], df['Y'], add_intercept=False)
        sc = linregress(df['X'].values, df['Y'].values)  # SciPy
        assert_equal(lm['names'].values, ['Intercept', 'X'])
        assert_almost_equal(lm['coef'][1], sc.slope)
        assert_almost_equal(lm['coef'][0], sc.intercept)
        assert_almost_equal(lm['se'][1], sc.stderr)
        assert_almost_equal(lm['pval'][1], sc.pvalue)
        assert_almost_equal(np.sqrt(lm['r2'][0]), sc.rvalue)
        assert lm.residuals_.size == df['Y'].size

        # Multiple regression with intercept
        X = df[['X', 'M']].values
        y = df['Y'].values
        lm = linear_regression(X, y, as_dataframe=False)  # Pingouin
        sk = LinearRegression(fit_intercept=True).fit(X, y)  # SkLearn
        assert_equal(lm['names'], ['Intercept', 'x1', 'x2'])
        assert_almost_equal(lm['coef'][1:], sk.coef_)
        assert_almost_equal(lm['coef'][0], sk.intercept_)
        assert_almost_equal(sk.score(X, y), lm['r2'])
        assert lm['residuals'].size == y.size

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

        # With missing values
        linear_regression(df_nan[['X', 'M']], df_nan['Y'], remove_na=True)

        # With columns with only one unique value
        lm1 = linear_regression(df[['X', 'M', 'One']], df['Y'])
        lm2 = linear_regression(df[['X', 'M', 'One']], df['Y'],
                                add_intercept=False)
        assert lm1.shape[0] == 3
        assert lm2.shape[0] == 3
        assert np.isclose(lm1.at[0, 'r2'], lm2.at[0, 'r2'])

        # With zero-only column
        lm1 = linear_regression(df[['X', 'M', 'Zero', 'One']], df['Y'])
        lm2 = linear_regression(df[['X', 'M', 'Zero', 'One']], df['Y'].values,
                                add_intercept=False)
        lm3 = linear_regression(df[['X', 'Zero', 'M', 'Zero']].values,
                                df['Y'], add_intercept=False)
        assert np.array_equal(lm1.loc[:, 'names'], ['Intercept', 'X', 'M'])
        assert np.array_equal(lm2.loc[:, 'names'], ['X', 'M', 'One'])
        assert np.array_equal(lm3.loc[:, 'names'], ['x1', 'x3'])

        # With duplicate columns
        lm1 = linear_regression(df[['X', 'One', 'Zero', 'M', 'M', 'X']],
                                df['Y'])
        lm2 = linear_regression(df[['X', 'One', 'Zero', 'M', 'M', 'X']].values,
                                df['Y'], add_intercept=False)
        assert np.array_equal(lm1.loc[:, 'names'], ['Intercept', 'X', 'M'])
        assert np.array_equal(lm2.loc[:, 'names'], ['x1', 'x2', 'x4'])

    def test_logistic_regression(self):
        """Test function logistic_regression."""

        # Simple regression
        lom = logistic_regression(df['X'], df['Ybin'], as_dataframe=False)
        # Compare to R
        # Reproduce in jupyter notebook with rpy2 using
        # %load_ext rpy2.ipython (in a separate cell)
        # Together in one cell below
        # %%R -i df
        # summary(glm(Ybin ~ X, data=df, family=binomial))
        assert_equal(np.round(lom['coef'], 4), [1.3191, -0.1995])
        assert_equal(np.round(lom['se'], 4), [0.7582, 0.1211])
        assert_equal(np.round(lom['z'], 4), [1.7399, -1.6476])
        assert_equal(np.round(lom['pval'], 4), [0.0819, 0.0994])
        assert_equal(np.round(lom['CI[2.5%]'], 4), [-.1669, -.4367])
        assert_equal(np.round(lom['CI[97.5%]'], 4), [2.8050, 0.0378])

        # Multiple predictors
        X = df[['X', 'M']].values
        y = df['Ybin'].values
        lom = logistic_regression(X, y).round(4)  # Pingouin
        # Compare against R
        # summary(glm(Ybin ~ X+M, data=df, family=binomial))
        assert_equal(lom['coef'].values, [1.3276, -0.1960, -0.0060])
        assert_equal(lom['se'].values, [0.7784, 0.1408, 0.1253])
        assert_equal(lom['z'].values, [1.7056, -1.3926, -0.0476])
        assert_equal(lom['pval'].values, [0.0881, 0.1637, 0.9620])
        assert_equal(lom['CI[2.5%]'].values, [-.1980, -.4719, -.2516])
        assert_equal(lom['CI[97.5%]'].values, [2.8531, 0.0799, 0.2397])

        # Test other arguments
        c = logistic_regression(df[['X', 'M']], df['Ybin'], coef_only=True)
        assert_equal(np.round(c, 4), [1.3276, -0.1960, -0.0060])

        # With missing values
        logistic_regression(df_nan[['X', 'M']], df_nan['Ybin'], remove_na=True)

        # Test **kwargs
        logistic_regression(X, y, solver='sag', C=10, max_iter=10000,
                            penalty="l2")
        logistic_regression(X, y, solver='sag', multi_class='auto')

        # Test regularization coefficients are strictly closer to 0 than
        # unregularized
        c = logistic_regression(df['X'], df['Ybin'], coef_only=True)
        c_reg = logistic_regression(df['X'], df['Ybin'], coef_only=True,
                                    penalty='l2')
        assert all(np.abs(c - 0) > np.abs(c_reg - 0))

        # With one column that has only one unique value
        c = logistic_regression(df[['One', 'X']], df['Ybin'])
        assert np.array_equal(c.loc[:, 'names'], ['Intercept', 'X'])
        c = logistic_regression(df[['X', 'One', 'M', 'Zero']], df['Ybin'])
        assert np.array_equal(c.loc[:, 'names'], ['Intercept', 'X', 'M'])

        # With duplicate columns
        c = logistic_regression(df[['X', 'M', 'X']], df['Ybin'])
        assert np.array_equal(c.loc[:, 'names'], ['Intercept', 'X', 'M'])
        c = logistic_regression(df[['X', 'X', 'X']], df['Ybin'])
        assert np.array_equal(c.loc[:, 'names'], ['Intercept', 'X'])

        # Error: dependent variable is not binary
        with pytest.raises(ValueError):
            y[3] = 2
            logistic_regression(X, y)

    def test_mediation_analysis(self):
        """Test function mediation_analysis.
        """
        ma = mediation_analysis(data=df, x='X', m='M', y='Y', n_boot=500)

        # Compare against R package mediation
        assert_equal(ma['coef'].values,
                     [0.5610, 0.6542, 0.3961, 0.0396, 0.3565])

        _, dist = mediation_analysis(data=df, x='X', m='M', y='Y', n_boot=1000,
                                     return_dist=True)
        assert dist.size == 1000
        mediation_analysis(data=df, x='X', m='M', y='Y', alpha=0.01)

        # Check with a binary mediator
        ma = mediation_analysis(data=df, x='X', m='Mbin', y='Y', n_boot=2000)
        assert_almost_equal(ma['coef'][0], -0.0208, decimal=2)

        # Indirect effect
        assert_almost_equal(ma['coef'][4], 0.0033, decimal=2)
        assert ma['sig'][4] == 'No'

        # Direct effect
        assert_almost_equal(ma['coef'][3], 0.3956, decimal=2)
        assert_almost_equal(ma['CI[2.5%]'][3], 0.1714, decimal=2)
        assert_almost_equal(ma['CI[97.5%]'][3], 0.617, decimal=1)
        assert ma['sig'][3] == 'Yes'

        # With multiple mediator
        np.random.seed(42)
        df.rename(columns={"M": "M1"}, inplace=True)
        df['M2'] = np.random.randint(0, 10, df.shape[0])
        ma2 = mediation_analysis(data=df, x='X', m=['M1', 'M2'], y='Y',
                                 seed=42)
        assert ma['coef'][2] == ma2['coef'][4]

        # With covariate
        mediation_analysis(data=df, x='X', m='M1', y='Y', covar='M2')
        mediation_analysis(data=df, x='X', m='M1', y='Y', covar=['M2'])
        mediation_analysis(data=df, x='X', m=['M1', 'Ybin'], y='Y',
                           covar=['Mbin', 'M2'])

        # Test helper function _pval_from_bootci
        np.random.seed(123)
        bt2 = np.random.normal(loc=2, size=1000)
        bt3 = np.random.normal(loc=3, size=1000)
        assert _pval_from_bootci(bt2, 0) == 1
        assert _pval_from_bootci(bt2, 0.9) < 0.10
        assert _pval_from_bootci(bt3, 0.9) < _pval_from_bootci(bt2, 0.9)
