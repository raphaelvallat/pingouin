import warnings
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from pandas.testing import assert_frame_equal
from scipy.stats import linregress, zscore
from sklearn.linear_model import LinearRegression

from pingouin import read_dataset
from pingouin.regression import (
    _duplicate_columns,
    _pval_from_bootci,
    linear_regression,
    logistic_regression,
    mediation_analysis,
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
        assert_equal(lm["CI2.5"].round(5).to_numpy(), [1.48155, 0.17553])
        assert_equal(lm["CI97.5"].round(5).to_numpy(), [4.23286, 0.61672])
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
        assert_equal([0.703, -0.178, 0.436], np.round(lm["CI2.5"], 3))
        assert_equal([3.106, 0.257, 0.835], np.round(lm["CI97.5"], 3))

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

        # Constant, all-zero and duplicate columns are kept. The design is rank deficient: the
        # fit is the same as without them, and the minimum-norm solution is returned.
        ref = linear_regression(df[["X", "M"]], df["Y"])
        # User-defined constant column without intercept: same fit as with the intercept
        lm = linear_regression(df[["X", "M", "One"]], df["Y"], add_intercept=False)
        assert_equal(lm["names"].to_numpy(), ["X", "M", "One"])
        assert_allclose(lm["coef"], ref["coef"].to_numpy()[[1, 2, 0]])
        assert_allclose(lm["se"], ref["se"].to_numpy()[[1, 2, 0]])
        assert np.isclose(lm.at[0, "r2"], ref.at[0, "r2"])
        assert (lm.df_model_, lm.df_resid_) == (ref.df_model_, ref.df_resid_)
        for X in [
            df[["X", "M", "One"]],  # Constant column in addition to the intercept
            df[["X", "M", "Two", "One"]],
            df[["X", "M", "Zero"]],
            df[["X", "Zero", "M", "Zero"]],
            df[["X", "One", "Zero", "M", "M", "X"]],  # Duplicate columns
        ]:
            with pytest.warns(UserWarning, match="rank 3 with"):
                lm = linear_regression(X, df["Y"])
            assert_equal(lm["names"].to_numpy(), ["Intercept", *X.columns])
            assert_allclose(lm["r2"], ref.at[0, "r2"])
            assert_allclose(lm["adj_r2"], ref.at[0, "adj_r2"])
            assert_allclose(lm.residuals_, ref.residuals_, atol=1e-12)
            assert (lm.df_model_, lm.df_resid_) == (ref.df_model_, ref.df_resid_)
            # T-values of the non-constant columns are unchanged
            is_x, is_m = (X.columns == "X"), (X.columns == "M")
            assert_allclose(lm["T"].iloc[1:][is_x], ref.at[1, "T"])
            assert_allclose(lm["T"].iloc[1:][is_m], ref.at[2, "T"])
            # The coefficient of a duplicated column is split equally between the copies
            assert_allclose(lm["coef"].iloc[1:][is_x].sum(), ref.at[1, "coef"])
            assert_allclose(lm["coef"].iloc[1:][is_m].sum(), ref.at[2, "coef"])
            # The coefficient of an all-zero column is zero, with NaN T-value and p-value
            is_zero = X.columns == "Zero"
            assert (lm["coef"].iloc[1:][is_zero] == 0).all()
            assert lm["T"].iloc[1:][is_zero].isna().all()
            assert lm["pval"].iloc[1:][is_zero].isna().all()

        # Duplicate and scaled copies of a column are handled consistently
        # https://github.com/raphaelvallat/pingouin/issues/522
        x1 = df["X"].to_numpy()
        ref = linear_regression(x1, df["Y"])
        for X in [np.column_stack([x1, x1]), np.column_stack([x1, 2 * x1])]:
            with pytest.warns(UserWarning, match="rank 2 with 3 columns"):
                lm = linear_regression(X, df["Y"])
            assert_allclose(lm["T"].iloc[1:], ref.at[1, "T"])
            assert_allclose(X @ lm["coef"].iloc[1:], x1 * ref.at[1, "coef"])

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
        np.testing.assert_allclose(res_pingouin["CI2.5"], res_sm.conf_int()[:, 0])
        np.testing.assert_allclose(res_pingouin["CI97.5"], res_sm.conf_int()[:, 1])

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
        # Relative importance is scale-invariant, even for a predictor in very
        # small units (see GH issue 522)
        X_scaled = df[["X", "M"]].copy()
        X_scaled["X"] *= 1e-8
        lm = linear_regression(X_scaled, df["Y"], relimp=True)
        assert_almost_equal(lm.loc[[1, 2], "relimp"], [0.05778011, 0.31521913])
        assert np.isclose(lm["relimp"].sum(), lm.at[0, "r2"])
        # User-defined constant column without intercept: the constant column
        # has zero relative importance and the others are unchanged
        X_const = df[["X", "M"]].copy()
        X_const.insert(1, "const", 0.1)
        lm = linear_regression(X_const, df["Y"], add_intercept=False, relimp=True)
        assert lm["names"].tolist() == ["X", "const", "M"]
        assert_almost_equal(lm["relimp"], [0.05778011, 0, 0.31521913])
        assert_almost_equal(lm["relimp_perc"], [15.49068, 0, 84.50932], decimal=4)

        ######################################################################
        # WEIGHTED REGRESSION - compare against R lm() function
        # Note that the summary function of R sometimes round to 4 decimals,
        # sometimes to 5, etc..
        lm = linear_regression(df[["X", "M"]], df["Y"], weights=df["W2"])
        assert_equal(lm["coef"].round(5).to_numpy(), [1.89530, 0.03905, 0.63912])
        assert_equal(lm["se"].round(5).to_numpy(), [0.60498, 0.10984, 0.10096])
        assert_equal(lm["T"].round(3).to_numpy(), [3.133, 0.356, 6.331])  # R round to 3
        assert_equal(lm["pval"].round(5).to_numpy(), [0.00229, 0.72296, 0.00000])
        assert_equal(lm["CI2.5"].round(5).to_numpy(), [0.69459, -0.17896, 0.43874])
        assert_equal(lm["CI97.5"].round(5).to_numpy(), [3.09602, 0.25706, 0.83949])
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
        assert_equal(lm["CI2.5"].round(5).to_numpy(), [0.10007, 0.51466])
        assert_equal(lm["CI97.5"].round(4).to_numpy(), [0.4384, 0.9200])
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
        assert_equal(lm["CI2.5"].round(5).to_numpy(), [2.09935, 0.03943])
        assert_equal(lm["CI97.5"].round(5).to_numpy(), [5.02015, 0.52453])
        assert round(lm["r2"].iloc[0], 5) == 0.05364
        assert round(lm["adj_r2"].iloc[0], 5) == 0.04358
        assert lm.df_model_ == 1
        assert lm.df_resid_ == 94

        # No intercept
        lm = linear_regression(df[["X"]], df["Y"], add_intercept=False, weights=df["W1"])
        assert_equal(lm["coef"].round(5).to_numpy(), [0.85060])
        assert_equal(lm["se"].round(5).to_numpy(), [0.03719])
        assert_equal(lm["pval"].round(5).to_numpy(), [0.0000])
        assert_equal(lm["CI2.5"].round(5).to_numpy(), [0.77678])
        assert_equal(lm["CI97.5"].round(5).to_numpy(), [0.92443])
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
        # Without intercept, compare to statsmodels
        lom_noint = logistic_regression(df[["X", "M"]], df["Ybin"], fit_intercept=False)
        sm_noint = sm.Logit(df["Ybin"], df[["X", "M"]]).fit(disp=False)
        assert_almost_equal(lom_noint["coef"].to_numpy(), sm_noint.params.to_numpy(), decimal=4)
        assert_almost_equal(lom_noint["se"].to_numpy(), sm_noint.bse.to_numpy(), decimal=4)
        # The default solver must converge to the maximum likelihood estimates.
        # Compare to R: summary(glm(P ~ H, family=binomial))
        H = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5]
        H += [4.75, 5, 5.5]
        P = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
        lom_h = logistic_regression(H, P, as_dataframe=False)
        np.testing.assert_allclose(lom_h["coef"], [-4.0777134, 1.5046454], atol=1e-5)
        np.testing.assert_allclose(lom_h["se"], [1.7609843, 0.6287165], atol=1e-4)
        assert_equal(np.round(lom["se"], 3), [0.758, 0.121])
        assert_almost_equal(lom["z"], [1.74, -1.647], decimal=2)
        assert_equal(np.round(lom["pval"], 3), [0.082, 0.099])
        assert_equal(np.round(lom["CI2.5"], 3), [-0.167, -0.437])
        assert_equal(np.round(lom["CI97.5"], 3), [2.805, 0.038])

        # Multiple predictors
        X = df[["X", "M"]].to_numpy()
        y = df["Ybin"].to_numpy()
        lom = logistic_regression(X, y).round(3)  # Pingouin
        # Compare against R
        # summary(glm(Ybin ~ X+M, data=df, family=binomial))
        assert_equal(lom["coef"].to_numpy(), [1.327, -0.196, -0.006])
        assert_equal(lom["se"].to_numpy(), [0.778, 0.141, 0.125])
        assert_almost_equal(lom["z"], [1.705, -1.392, -0.048], decimal=2)
        assert_equal(lom["pval"].to_numpy(), [0.088, 0.164, 0.962])
        assert_equal(lom["CI2.5"].to_numpy(), [-0.198, -0.472, -0.252])
        assert_equal(lom["CI97.5"].to_numpy(), [2.853, 0.08, 0.24])

        # Test other arguments
        c = logistic_regression(df[["X", "M"]], df["Ybin"], coef_only=True)
        assert_equal(np.round(c, 3), [1.327, -0.196, -0.006])

        # With missing values
        logistic_regression(df_nan[["X", "M"]], df_nan["Ybin"], remove_na=True)

        # Test **kwargs
        logistic_regression(X, y, solver="sag", C=10, max_iter=10000, penalty="l2")

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
        y_nonbin = y.copy()
        y_nonbin[3] = 2
        with pytest.raises(ValueError):
            logistic_regression(X, y_nonbin)

        # --------------------------------------------------------------------
        # 2ND dataset (Penguin)-- compare to R
        # sklearn >= 1.3 migrated the newton-cg solver to LinearModelLoss, producing
        # slightly different (but correct) coefficients for poorly-scaled predictors.
        # Use tolerances that accept both the old (sklearn < 1.3) and new values.
        lom = logistic_regression(data["body_mass_g"], data["male"], as_dataframe=False)
        assert np.allclose(lom["coef"], [-5.162541644, 0.001239819], atol=0.01)
        assert np.allclose(lom["se"], [0.72439, 0.00017], atol=0.001)
        assert np.allclose(lom["z"], [-7.127, 7.177], atol=0.01)
        assert np.allclose(lom["pval"], [1.03e-12, 7.10e-13], rtol=0.1)
        assert np.allclose(lom["CI2.5"], [-6.582, 0.001], atol=0.01)
        assert np.allclose(lom["CI97.5"], [-3.743, 0.002], atol=0.01)

        # With a different scaling: z / p-values should be similar
        lom = logistic_regression(data["body_mass_kg"], data["male"], as_dataframe=False)
        assert np.allclose(lom["coef"], [-5.162542, 1.239819])
        assert_equal(np.round(lom["se"], 4), [0.7244, 0.1727])
        assert_equal(np.round(lom["z"], 3), [-7.127, 7.177])
        assert np.allclose(lom["pval"], [1.03e-12, 7.10e-13])
        assert_equal(np.round(lom["CI2.5"], 3), [-6.582, 0.901])
        assert_equal(np.round(lom["CI97.5"], 3), [-3.743, 1.578])

        # With no intercept
        lom = logistic_regression(
            data["body_mass_kg"], data["male"], as_dataframe=False, fit_intercept=False
        )
        assert np.isclose(lom["coef"], 0.04150582)
        assert np.round(lom["se"], 5) == 0.02570
        assert np.round(lom["z"], 3) == 1.615
        assert np.round(lom["pval"], 3) == 0.106
        assert np.round(lom["CI2.5"], 3) == -0.009
        assert np.round(lom["CI97.5"], 3) == 0.092

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
        assert_equal(np.round(lom["CI2.5"], 1), [-33.0, 5.8, -1.1, -12.5])
        assert_equal(np.round(lom["CI97.5"], 1), [-21.3, 9.0, 0.6, -7.8])

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
        # The type of regression is chosen separately for each mediator: a binary mediator
        # is modeled with a logistic regression even when another mediator is continuous
        ma_bin = mediation_analysis(data=df, x="X", m="Mbin", y="Y", n_boot=10)
        ma_both = mediation_analysis(data=df, x="X", m=["M", "Mbin"], y="Y", n_boot=10)
        assert np.isclose(ma_both.at[1, "coef"], ma_bin.at[0, "coef"])
        assert np.isclose(ma_both.at[0, "coef"], ma.at[0, "coef"])

        # Check with a binary mediator. Each bootstrap sample fits a logistic regression, so
        # n_boot is kept small: only the significance of the indirect effect depends on it.
        ma = mediation_analysis(data=df, x="X", m="Mbin", y="Y", n_boot=500)
        assert_almost_equal(ma["coef"][0], -0.0208, decimal=2)

        # Indirect effect
        assert_almost_equal(ma["coef"][4], 0.0033, decimal=2)
        assert ma["sig"][4] == "No"

        # Direct effect
        assert_almost_equal(ma["coef"][3], 0.3956, decimal=2)
        assert_almost_equal(ma["CI2.5"][3], 0.1714, decimal=2)
        assert_almost_equal(ma["CI97.5"][3], 0.617, decimal=1)
        assert ma["sig"][3] == "Yes"

        # Check if `logreg_kwargs` is being passed on to `LogisticRegression`
        with pytest.raises(ValueError):
            mediation_analysis(
                data=df, x="X", m="Mbin", y="Y", n_boot=10, logreg_kwargs=dict(max_iter=-1)
            )
        # Solve with 0 iterations and make sure that the results are different
        ma = mediation_analysis(
            data=df, x="X", m="Mbin", y="Y", n_boot=10, logreg_kwargs=dict(max_iter=0)
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


@pytest.mark.parametrize("scale", [1.0, 1e-8, 1e8])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("add_intercept", [False, True])
def test_linear_regression_covariance_under_rescaling(scale, weighted, add_intercept):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2))
    X[:, 0] *= scale
    y = rng.normal(size=100)
    weights = rng.uniform(0.5, 2, size=100) if weighted else None
    result = linear_regression(X, y, add_intercept=add_intercept, weights=weights)
    design = sm.add_constant(X) if add_intercept else X
    reference = sm.WLS(y, design, weights=weights).fit() if weighted else sm.OLS(y, design).fit()
    np.testing.assert_allclose(result["coef"], reference.params, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(result["se"], reference.bse, rtol=1e-6)
    np.testing.assert_allclose(result["T"], reference.tvalues, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result["pval"], reference.pvalues, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result[["CI2.5", "CI97.5"]], reference.conf_int(), rtol=1e-6)


@pytest.mark.parametrize("scale", [1.0, 1e-8, 1e8])
def test_linear_regression_small_units_match_linregress(scale):
    rng = np.random.default_rng(42)
    x = rng.normal(size=100) * scale
    y = rng.normal(size=100)
    result = linear_regression(x, y)
    reference = linregress(x, y)
    np.testing.assert_allclose(result["se"].iloc[1], reference.stderr, rtol=1e-7)
    np.testing.assert_allclose(result["pval"].iloc[1], reference.pvalue, rtol=1e-7)


def test_linear_regression_exactly_collinear_dummies():
    # Intercept + full set of one-hot dummies is exactly rank deficient. With
    # lstsq's default tolerance (machine epsilon) this design is misreported
    # as full rank for this seed, giving coefficients and SE around 1e13 and
    # no warning. The truncated-SVD covariance must match statsmodels.
    rng = np.random.default_rng(3)
    n = 190
    groups = rng.integers(0, 3, n)
    X = np.column_stack([np.eye(3)[groups], rng.normal(size=n)])
    y = rng.normal(size=n)
    with pytest.warns(UserWarning, match="rank 4 with 5 columns"):
        result = linear_regression(X, y, add_intercept=True)
    reference = sm.OLS(y, sm.add_constant(X)).fit()
    np.testing.assert_allclose(result["coef"], reference.params, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(result["se"], reference.bse, rtol=1e-6)
    np.testing.assert_allclose(result["pval"], reference.pvalues, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result[["CI2.5", "CI97.5"]], reference.conf_int(), rtol=1e-6)


def test_linear_regression_saturated_design():
    # n == p (zero residual degrees of freedom) used to raise a broadcast
    # error. It should behave like the n < p case: non-finite SE and NaN p-value.
    rng = np.random.default_rng(0)
    y = rng.normal(size=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        square = linear_regression(rng.normal(size=(4, 3)), y)
        wide = linear_regression(rng.normal(size=(4, 5)), y)
    assert square.shape[0] == 4
    for res in (square, wide):
        assert not np.isfinite(res["se"]).any()
        assert res["pval"].isna().all()


def test_logistic_regression_se_small_units():
    # With a predictor in very small units, the Fisher information matrix is numerically singular
    # and inverting it with pinv gave a SE of ~0 and a p-value of 0.
    # https://github.com/raphaelvallat/pingouin/issues/522
    rng = np.random.default_rng(42)
    n, scale = 300, 1e-8
    x1, x2 = rng.normal(size=(2, n))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scikit-learn may not converge at this scale
        lr = logistic_regression(np.column_stack([x1 * scale, x2]), y)
    # Reference: Wald SE at the same linear predictor, computed on the well-conditioned unscaled
    # design and then rescaled.
    coef = lr["coef"].to_numpy()
    X = np.column_stack([np.ones(n), x1, x2])
    prob = 1 / (1 + np.exp(-(X @ (coef * [1, scale, 1]))))
    se = np.sqrt(np.diag(np.linalg.inv((X * (prob * (1 - prob))[:, None]).T @ X)))
    np.testing.assert_allclose(lr["se"], se / [1, scale, 1], rtol=1e-6)
    assert lr["pval"].iloc[1] > 0.5


def test_duplicate_columns():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 3))
    dummies = np.eye(4)[rng.integers(0, 4, 1000)]
    X = np.column_stack([x, dummies, x[:, 1], dummies[:, 2], x[:, 1] + 1e-12, -x[:, 0]])
    assert_equal(_duplicate_columns(X), [7, 8])
    assert_equal(_duplicate_columns(x[:, :1]), [])
    # Centered columns whose weighted sums cancel out
    c = np.tile([1.0, -1.0], 500)
    assert_equal(_duplicate_columns(np.column_stack([c, -c, c, 1e6 * c])), [2])


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_linear_regression_zero_column(position):
    # The coefficient and SE of an all-zero column must be exactly zero, with NaN T and p-values,
    # wherever the column is. Rounding noise of the SVD gave a coefficient and SE of ~1e-17
    # with an arbitrary, sometimes "significant", T-value.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = rng.normal(size=200)
    ref = linear_regression(x, y)
    X = np.insert(x, position, 0, axis=1)
    with pytest.warns(UserWarning, match="rank 4 with 5 columns"):
        lm = linear_regression(X, y)
    zero = position + 1  # Shifted by the intercept
    assert lm.at[zero, "coef"] == 0 and lm.at[zero, "se"] == 0
    assert np.isnan(lm.at[zero, "T"]) and np.isnan(lm.at[zero, "pval"])
    others = lm.drop(index=zero).reset_index(drop=True)
    assert_allclose(others[["coef", "se", "T", "pval"]], ref[["coef", "se", "T", "pval"]])


def test_regression_boolean_predictors():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.random(100) > 0.5, "b": rng.random(100) > 0.3})
    y = rng.normal(size=100)
    ybin = (rng.random(100) > 0.5).astype(int)
    assert_frame_equal(
        linear_regression(X, y, add_intercept=False),
        linear_regression(X.astype(float), y, add_intercept=False),
    )
    assert_frame_equal(logistic_regression(X, ybin), logistic_regression(X.astype(float), ybin))
    # Boolean target: yw @ yw returned True instead of the total sum of squares
    assert_frame_equal(
        linear_regression(X["a"], y > 0, add_intercept=False),
        linear_regression(X["a"], (y > 0).astype(float), add_intercept=False),
    )
    assert_frame_equal(
        logistic_regression(X, ybin.astype(bool)), logistic_regression(X.astype(float), ybin)
    )


def test_logistic_regression_constant_column_without_intercept():
    # With fit_intercept=False, a user-defined constant column is the intercept of the model and
    # must not be removed. All-zero and additional constant columns are still removed.
    ref = logistic_regression(df[["X", "M"]], df["Ybin"])
    lom = logistic_regression(df[["Zero", "Two", "X", "One", "M"]], df["Ybin"], fit_intercept=False)
    assert_equal(lom["names"].to_numpy(), ["Two", "X", "M"])
    assert_allclose(lom["coef"], ref["coef"] * [0.5, 1, 1], rtol=1e-5)
    assert_allclose(lom["se"], ref["se"] * [0.5, 1, 1], rtol=1e-5)
    assert_allclose(lom["pval"], ref["pval"], rtol=1e-4)
    # With the intercept of scikit-learn, all the constant columns are removed
    lom = logistic_regression(df[["Two", "X", "M"]], df["Ybin"])
    assert_equal(lom["names"].to_numpy(), ["Intercept", "X", "M"])
