import pytest
import numpy as np
import pandas as pd
from unittest import TestCase

from scipy.stats import linregress, zscore
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal, assert_equal

from pingouin import read_dataset
from pingouin.regression import (
    linear_regression,
    logistic_regression,
    mediation_analysis,
    _pval_from_bootci,
)

# 1st dataset: mediation
df = read_dataset("mediation")
df["Zero"] = 0
df["One"] = 1
df["Two"] = 2
df_nan = df.copy()
df_nan.loc[1, "M"] = np.nan
df_nan.loc[10, "X"] = np.nan
df_nan.loc[12, ["Y", "Ybin"]] = np.nan

# 2nd dataset: penguins
data = read_dataset("penguins").dropna()
data["male"] = (data["sex"] == "male").astype(int)
data["body_mass_kg"] = data["body_mass_g"] / 1000


class TestRegression(TestCase):
    """Test regression.py."""

    def test_linear_regression(self):
        """Test function linear_regression.

        Compare against JASP and R lm() function.
        """
        # Simple regression (compare to R lm())
        lm = linear_regression(df["X"], df["Y"])  # Pingouin
        sc = linregress(df["X"], df["Y"])  # SciPy
        # When using assert_equal, we need to use .to_numpy()
        assert_equal(lm["names"].to_numpy(), ["Intercept", "X"])
        assert_almost_equal(lm["coef"][1], sc.slope)
        assert_almost_equal(lm["coef"][0], sc.intercept)
        assert_almost_equal(lm["se"][1], sc.stderr)
        assert_almost_equal(lm["pval"][1], sc.pvalue)
        assert_almost_equal(np.sqrt(lm["r2"][0]), sc.rvalue)
        assert lm.residuals_.size == df["Y"].size
        assert_equal(lm["CI[2.5%]"].round(5).to_numpy(), [1.48155, 0.17553])
        assert_equal(lm["CI[97.5%]"].round(5).to_numpy(), [4.23286, 0.61672])
        assert round(lm["r2"].iloc[0], 4) == 0.1147
        assert round(lm["adj_r2"].iloc[0], 4) == 0.1057
        assert lm.df_model_ == 1
        assert lm.df_resid_ == 98

        # Multiple regression with intercept (compare to JASP)
        X = df[["X", "M"]].to_numpy()
        y = df["Y"].to_numpy()
        lm = linear_regression(X, y, as_dataframe=False)  # Pingouin
        sk = LinearRegression(fit_intercept=True).fit(X, y)  # SkLearn
        assert_equal(lm["names"], ["Intercept", "x1", "x2"])
        assert_almost_equal(lm["coef"][1:], sk.coef_)
        assert_almost_equal(lm["coef"][0], sk.intercept_)
        assert_almost_equal(sk.score(X, y), lm["r2"])
        assert lm["residuals"].size == y.size
        # No need for .to_numpy here because we're using a dict and not pandas
        assert_equal([0.605, 0.110, 0.101], np.round(lm["se"], 3))
        assert_equal([3.145, 0.361, 6.321], np.round(lm["T"], 3))
        assert_equal([0.002, 0.719, 0.000], np.round(lm["pval"], 3))
        assert_equal([0.703, -0.178, 0.436], np.round(lm["CI[2.5%]"], 3))
        assert_equal([3.106, 0.257, 0.835], np.round(lm["CI[97.5%]"], 3))

        # No intercept
        lm = linear_regression(X, y, add_intercept=False, as_dataframe=False)
        sk = LinearRegression(fit_intercept=False).fit(X, y)
        assert_almost_equal(lm["coef"], sk.coef_)
        # Scikit-learn gives wrong R^2 score when no intercept present because
        # sklearn.metrics.r2_score always assumes that an intercept is present
        # https://stackoverflow.com/questions/54614157/scikit-learn-statsmodels-which-r-squared-is-correct
        # assert_almost_equal(sk.score(X, y), lm['r2'])
        # Instead, we compare to R lm() function:
        assert round(lm["r2"], 4) == 0.9096
        assert round(lm["adj_r2"], 4) == 0.9078
        assert lm["df_model"] == 2
        assert lm["df_resid"] == 98

        # Test other arguments
        linear_regression(df[["X", "M"]], df["Y"], coef_only=True)
        linear_regression(df[["X", "M"]], df["Y"], alpha=0.01)
        linear_regression(df[["X", "M"]], df["Y"], alpha=0.10)

        # With missing values
        linear_regression(df_nan[["X", "M"]], df_nan["Y"], remove_na=True)

        # With columns with only one unique value
        lm1 = linear_regression(df[["X", "M", "One"]], df["Y"])
        lm2 = linear_regression(df[["X", "M", "One"]], df["Y"], add_intercept=False)
        assert lm1.shape[0] == 3
        assert lm2.shape[0] == 3
        assert np.isclose(lm1.at[0, "r2"], lm2.at[0, "r2"])

        # With zero-only column
        lm1 = linear_regression(df[["X", "M", "Zero", "One"]], df["Y"])
        lm2 = linear_regression(
            df[["X", "M", "Zero", "One"]], df["Y"].to_numpy(), add_intercept=False
        )
        lm3 = linear_regression(
            df[["X", "Zero", "M", "Zero"]].to_numpy(), df["Y"], add_intercept=False
        )
        assert_equal(lm1.loc[:, "names"].to_numpy(), ["Intercept", "X", "M"])
        assert_equal(lm2.loc[:, "names"].to_numpy(), ["X", "M", "One"])
        assert_equal(lm3.loc[:, "names"].to_numpy(), ["x1", "x3"])

        # With duplicate columns
        lm1 = linear_regression(df[["X", "One", "Zero", "M", "M", "X"]], df["Y"])
        lm2 = linear_regression(
            df[["X", "One", "Zero", "M", "M", "X"]].to_numpy(), df["Y"], add_intercept=False
        )
        assert_equal(lm1.loc[:, "names"].to_numpy(), ["Intercept", "X", "M"])
        assert_equal(lm2.loc[:, "names"].to_numpy(), ["x1", "x2", "x4"])

        # with rank deficient design matrix `X`
        # see: https://github.com/raphaelvallat/pingouin/issues/130
        n = 100
        rng = np.random.RandomState(42)
        X = np.vstack([rng.permutation([0, 1, 0, 0, 0]) for i in range(n)])
        y = rng.randn(n)

        with pytest.warns(UserWarning, match="(rank 5 with 6 columns)"):
            res_pingouin = linear_regression(X, y, add_intercept=True)

        X_with_intercept = sm.add_constant(X)
        res_sm = sm.OLS(endog=y, exog=X_with_intercept).fit()

        np.testing.assert_allclose(res_pingouin.residuals_, res_sm.resid)
        np.testing.assert_allclose(res_pingouin["coef"], res_sm.params)
        np.testing.assert_allclose(res_pingouin["r2"][0], res_sm.rsquared)
        np.testing.assert_allclose(res_pingouin["adj_r2"][0], res_sm.rsquared_adj)
        np.testing.assert_allclose(res_pingouin["T"], res_sm.tvalues)
        np.testing.assert_allclose(res_pingouin["se"], res_sm.bse)
        np.testing.assert_allclose(res_pingouin["pval"], res_sm.pvalues)
        np.testing.assert_allclose(res_pingouin["CI[2.5%]"], res_sm.conf_int()[:, 0])
        np.testing.assert_allclose(res_pingouin["CI[97.5%]"], res_sm.conf_int()[:, 1])

        # Relative importance
        # Compare to R package relaimpo
        # >>> data <- read.csv('mediation.csv')
        # >>> lm1 <- lm(Y ~ X + M, data = data)
        # >>> calc.relimp(lm1, type=c("lmg"))
        lm = linear_regression(df[["X", "M"]], df["Y"], relimp=True)
        assert_almost_equal(lm.loc[[1, 2], "relimp"], [0.05778011, 0.31521913])
        assert_almost_equal(lm.loc[[1, 2], "relimp_perc"], [15.49068, 84.50932], decimal=4)
        # Now we make sure that relimp_perc sums to 100% and relimp sums to r2
        assert np.isclose(lm["relimp_perc"].sum(), 100.0)
        assert np.isclose(lm["relimp"].sum(), lm.at[0, "r2"])
        # 2 predictors, no intercept
        # Careful here, the sum of relimp is always the R^2 of the model
        # INCLUDING the intercept. Therefore, if the data are not normalized
        # and we set add_intercept to false, the sum of relimp will be
        # higher than the linear regression model.
        # A workaround is to standardize our data before:
        df_z = df[["X", "M", "Y"]].apply(zscore)
        lm = linear_regression(
            df_z[["X", "M"]], df_z["Y"], add_intercept=False, as_dataframe=False, relimp=True
        )
        assert_almost_equal(lm["relimp"], [0.05778011, 0.31521913], decimal=4)
        assert_almost_equal(lm["relimp_perc"], [15.49068, 84.50932], decimal=4)
        assert np.isclose(np.sum(lm["relimp"]), lm["r2"])
        # 3 predictors + intercept
        lm = linear_regression(df[["X", "M", "Ybin"]], df["Y"], relimp=True)
        assert_almost_equal(lm.loc[[1, 2, 3], "relimp"], [0.06010737, 0.31724368, 0.01217479])
        assert_almost_equal(
            lm.loc[[1, 2, 3], "relimp_perc"], [15.43091, 81.44355, 3.12554], decimal=4
        )
        assert np.isclose(lm["relimp"].sum(), lm.at[0, "r2"])

        ######################################################################
        # WEIGHTED REGRESSION - compare against R lm() function
        # Note that the summary function of R sometimes round to 4 decimals,
        # sometimes to 5, etc..
        lm = linear_regression(df[["X", "M"]], df["Y"], weights=df["W2"])
        assert_equal(lm["coef"].round(5).to_numpy(), [1.89530, 0.03905, 0.63912])
        assert_equal(lm["se"].round(5).to_numpy(), [0.60498, 0.10984, 0.10096])
        assert_equal(lm["T"].round(3).to_numpy(), [3.133, 0.356, 6.331])  # R round to 3
        assert_equal(lm["pval"].round(5).to_numpy(), [0.00229, 0.72296, 0.00000])
        assert_equal(lm["CI[2.5%]"].round(5).to_numpy(), [0.69459, -0.17896, 0.43874])
        assert_equal(lm["CI[97.5%]"].round(5).to_numpy(), [3.09602, 0.25706, 0.83949])
        assert round(lm["r2"].iloc[0], 4) == 0.3742
        assert round(lm["adj_r2"].iloc[0], 4) == 0.3613
        assert lm.df_model_ == 2
        assert lm.df_resid_ == 97

        # No intercept
        lm = linear_regression(df[["X", "M"]], df["Y"], add_intercept=False, weights=df["W2"])
        assert_equal(lm["coef"].round(5).to_numpy(), [0.26924, 0.71733])
        assert_equal(lm["se"].round(5).to_numpy(), [0.08525, 0.10213])
        assert_equal(lm["T"].round(3).to_numpy(), [3.158, 7.024])
        assert_equal(lm["pval"].round(5).to_numpy(), [0.00211, 0.00000])
        assert_equal(lm["CI[2.5%]"].round(5).to_numpy(), [0.10007, 0.51466])
        assert_equal(lm["CI[97.5%]"].round(4).to_numpy(), [0.4384, 0.9200])
        assert round(lm["r2"].iloc[0], 4) == 0.9090
        assert round(lm["adj_r2"].iloc[0], 4) == 0.9072
        assert lm.df_model_ == 2
        assert lm.df_resid_ == 98

        # With some weights set to zero
        # Here, R gives slightl different results than statsmodels because
        # zero weights are not taken into account when calculating the degrees
        # of freedom. Pingouin is similar to R.
        lm = linear_regression(df[["X"]], df["Y"], weights=df["W1"])
        assert_equal(lm["coef"].round(4).to_numpy(), [3.5597, 0.2820])
        assert_equal(lm["se"].round(4).to_numpy(), [0.7355, 0.1222])
        assert_equal(lm["pval"].round(4).to_numpy(), [0.0000, 0.0232])
        assert_equal(lm["CI[2.5%]"].round(5).to_numpy(), [2.09935, 0.03943])
        assert_equal(lm["CI[97.5%]"].round(5).to_numpy(), [5.02015, 0.52453])
        assert round(lm["r2"].iloc[0], 5) == 0.05364
        assert round(lm["adj_r2"].iloc[0], 5) == 0.04358
        assert lm.df_model_ == 1
        assert lm.df_resid_ == 94

        # No intercept
        lm = linear_regression(df[["X"]], df["Y"], add_intercept=False, weights=df["W1"])
        assert_equal(lm["coef"].round(5).to_numpy(), [0.85060])
        assert_equal(lm["se"].round(5).to_numpy(), [0.03719])
        assert_equal(lm["pval"].round(5).to_numpy(), [0.0000])
        assert_equal(lm["CI[2.5%]"].round(5).to_numpy(), [0.77678])
        assert_equal(lm["CI[97.5%]"].round(5).to_numpy(), [0.92443])
        assert round(lm["r2"].iloc[0], 4) == 0.8463
        assert round(lm["adj_r2"].iloc[0], 4) == 0.8447
        assert lm.df_model_ == 1
        assert lm.df_resid_ == 95

        # With all weights to one, should be equal to OLS
        assert_frame_equal(
            linear_regression(df[["X", "M"]], df["Y"]),
            linear_regression(df[["X", "M"]], df["Y"], weights=df["One"]),
        )

        # Output is a dictionary
        linear_regression(df[["X", "M"]], df["Y"], weights=df["W2"], as_dataframe=False)

        with pytest.raises(ValueError):
            linear_regression(df[["X"]], df["Y"], weights=df["W1"], relimp=True)

    def test_logistic_regression(self):
        """Test function logistic_regression."""
        # Simple regression
        df = read_dataset("mediation")
        df["Zero"], df["One"] = 0, 1
        lom = logistic_regression(df["X"], df["Ybin"], as_dataframe=False)
        # Compare to R
        # Reproduce in jupyter notebook with rpy2 using
        # %load_ext rpy2.ipython (in a separate cell)
        # Together in one cell below
        # %%R -i df
        # summary(glm(Ybin ~ X, data=df, family=binomial))
        assert_equal(np.round(lom["coef"], 3), [1.319, -0.199])
        assert_equal(np.round(lom["se"], 3), [0.758, 0.121])
        assert_equal(np.round(lom["z"], 3), [1.74, -1.647])
        assert_equal(np.round(lom["pval"], 3), [0.082, 0.099])
        assert_equal(np.round(lom["CI[2.5%]"], 3), [-0.167, -0.437])
        assert_equal(np.round(lom["CI[97.5%]"], 3), [2.805, 0.038])

        # Multiple predictors
        X = df[["X", "M"]].to_numpy()
        y = df["Ybin"].to_numpy()
        lom = logistic_regression(X, y).round(3)  # Pingouin
        # Compare against R
        # summary(glm(Ybin ~ X+M, data=df, family=binomial))
        assert_equal(lom["coef"].to_numpy(), [1.327, -0.196, -0.006])
        assert_equal(lom["se"].to_numpy(), [0.778, 0.141, 0.125])
        assert_equal(lom["z"].to_numpy(), [1.705, -1.392, -0.048])
        assert_equal(lom["pval"].to_numpy(), [0.088, 0.164, 0.962])
        assert_equal(lom["CI[2.5%]"].to_numpy(), [-0.198, -0.472, -0.252])
        assert_equal(lom["CI[97.5%]"].to_numpy(), [2.853, 0.08, 0.24])

        # Test other arguments
        c = logistic_regression(df[["X", "M"]], df["Ybin"], coef_only=True)
        assert_equal(np.round(c, 3), [1.327, -0.196, -0.006])

        # With missing values
        logistic_regression(df_nan[["X", "M"]], df_nan["Ybin"], remove_na=True)

        # Test **kwargs
        logistic_regression(X, y, solver="sag", C=10, max_iter=10000, penalty="l2")
        logistic_regression(X, y, solver="sag", multi_class="auto")

        # Test regularization coefficients are strictly closer to 0 than
        # unregularized
        c = logistic_regression(df["X"], df["Ybin"], coef_only=True)
        c_reg = logistic_regression(df["X"], df["Ybin"], coef_only=True, penalty="l2")
        assert all(np.abs(c - 0) > np.abs(c_reg - 0))

        # With one column that has only one unique value
        c = logistic_regression(df[["One", "X"]], df["Ybin"])
        assert_equal(c.loc[:, "names"].to_numpy(), ["Intercept", "X"])
        c = logistic_regression(df[["X", "One", "M", "Zero"]], df["Ybin"])
        assert_equal(c.loc[:, "names"].to_numpy(), ["Intercept", "X", "M"])

        # With duplicate columns
        c = logistic_regression(df[["X", "M", "X"]], df["Ybin"])
        assert_equal(c.loc[:, "names"].to_numpy(), ["Intercept", "X", "M"])
        c = logistic_regression(df[["X", "X", "X"]], df["Ybin"])
        assert_equal(c.loc[:, "names"].to_numpy(), ["Intercept", "X"])

        # Error: dependent variable is not binary
        with pytest.raises(ValueError):
            y[3] = 2
            logistic_regression(X, y)

        # --------------------------------------------------------------------
        # 2ND dataset (Penguin)-- compare to R
        lom = logistic_regression(data["body_mass_g"], data["male"], as_dataframe=False)
        assert np.allclose(lom["coef"], [-5.162541644, 0.001239819])
        assert_equal(np.round(lom["se"], 5), [0.72439, 0.00017])
        assert_equal(np.round(lom["z"], 3), [-7.127, 7.177])
        assert np.allclose(lom["pval"], [1.03e-12, 7.10e-13])
        assert_equal(np.round(lom["CI[2.5%]"], 3), [-6.582, 0.001])
        assert_equal(np.round(lom["CI[97.5%]"], 3), [-3.743, 0.002])

        # With a different scaling: z / p-values should be similar
        lom = logistic_regression(data["body_mass_kg"], data["male"], as_dataframe=False)
        assert np.allclose(lom["coef"], [-5.162542, 1.239819])
        assert_equal(np.round(lom["se"], 4), [0.7244, 0.1727])
        assert_equal(np.round(lom["z"], 3), [-7.127, 7.177])
        assert np.allclose(lom["pval"], [1.03e-12, 7.10e-13])
        assert_equal(np.round(lom["CI[2.5%]"], 3), [-6.582, 0.901])
        assert_equal(np.round(lom["CI[97.5%]"], 3), [-3.743, 1.578])

        # With no intercept
        lom = logistic_regression(
            data["body_mass_kg"], data["male"], as_dataframe=False, fit_intercept=False
        )
        assert np.isclose(lom["coef"], 0.04150582)
        assert np.round(lom["se"], 5) == 0.02570
        assert np.round(lom["z"], 3) == 1.615
        assert np.round(lom["pval"], 3) == 0.106
        assert np.round(lom["CI[2.5%]"], 3) == -0.009
        assert np.round(lom["CI[97.5%]"], 3) == 0.092

        # With categorical predictors
        # R: >>> glm("male ~ body_mass_kg + species", family=binomial, ...)
        #    >>> confint.default(model)  # Wald CI
        # See https://stats.stackexchange.com/a/275421/253579
        data_dum = pd.get_dummies(data, columns=["species"], drop_first=True, dtype=float)
        X = data_dum[["body_mass_kg", "species_Chinstrap", "species_Gentoo"]]
        y = data_dum["male"]
        lom = logistic_regression(X, y, as_dataframe=False)
        # See https://github.com/raphaelvallat/pingouin/pull/403
        assert_equal(np.round(lom["coef"], 2), [-27.13, 7.37, -0.26, -10.18])
        assert_equal(np.round(lom["se"], 3), [2.998, 0.814, 0.429, 1.195])
        assert_equal(np.round(lom["z"], 3), [-9.049, 9.056, -0.596, -8.520])
        assert_equal(np.round(lom["CI[2.5%]"], 1), [-33.0, 5.8, -1.1, -12.5])
        assert_equal(np.round(lom["CI[97.5%]"], 1), [-21.3, 9.0, 0.6, -7.8])

    def test_mediation_analysis(self):
        """Test function mediation_analysis."""
        df = read_dataset("mediation")
        df["Zero"], df["One"] = 0, 1
        ma = mediation_analysis(data=df, x="X", m="M", y="Y", n_boot=500)

        # Compare against R package mediation
        assert_equal(ma["coef"].round(4).to_numpy(), [0.5610, 0.6542, 0.3961, 0.0396, 0.3565])

        _, dist = mediation_analysis(data=df, x="X", m="M", y="Y", n_boot=1000, return_dist=True)
        assert dist.size == 1000
        mediation_analysis(data=df, x="X", m="M", y="Y", alpha=0.01)

        # Check with a binary mediator
        ma = mediation_analysis(data=df, x="X", m="Mbin", y="Y", n_boot=2000)
        assert_almost_equal(ma["coef"][0], -0.0208, decimal=2)

        # Indirect effect
        assert_almost_equal(ma["coef"][4], 0.0033, decimal=2)
        assert ma["sig"][4] == "No"

        # Direct effect
        assert_almost_equal(ma["coef"][3], 0.3956, decimal=2)
        assert_almost_equal(ma["CI[2.5%]"][3], 0.1714, decimal=2)
        assert_almost_equal(ma["CI[97.5%]"][3], 0.617, decimal=1)
        assert ma["sig"][3] == "Yes"

        # Check if `logreg_kwargs` is being passed on to `LogisticRegression`
        with pytest.raises(ValueError):
            mediation_analysis(
                data=df, x="X", m="Mbin", y="Y", n_boot=2000, logreg_kwargs=dict(max_iter=-1)
            )
        # Solve with 0 iterations and make sure that the results are different
        ma = mediation_analysis(
            data=df, x="X", m="Mbin", y="Y", n_boot=2000, logreg_kwargs=dict(max_iter=0)
        )
        with pytest.raises(AssertionError):
            assert_almost_equal(ma["coef"][0], -0.0208, decimal=2)
        with pytest.raises(AssertionError):
            assert_almost_equal(ma["coef"][4], 0.0033, decimal=3)

        # With multiple mediator
        np.random.seed(42)
        df.rename(columns={"M": "M1"}, inplace=True)
        df["M2"] = np.random.randint(0, 10, df.shape[0])
        ma2 = mediation_analysis(data=df, x="X", m=["M1", "M2"], y="Y", seed=42)
        assert ma["coef"][2] == ma2["coef"][4]

        # With covariate
        mediation_analysis(data=df, x="X", m="M1", y="Y", covar="M2")
        mediation_analysis(data=df, x="X", m="M1", y="Y", covar=["M2"])
        mediation_analysis(data=df, x="X", m=["M1", "Ybin"], y="Y", covar=["Mbin", "M2"])

        # Test helper function _pval_from_bootci
        np.random.seed(123)
        bt2 = np.random.normal(loc=2, size=1000)
        bt3 = np.random.normal(loc=3, size=1000)
        assert _pval_from_bootci(bt2, 0) == 1
        assert _pval_from_bootci(bt2, 0.9) < 0.10
        assert _pval_from_bootci(bt3, 0.9) < _pval_from_bootci(bt2, 0.9)
