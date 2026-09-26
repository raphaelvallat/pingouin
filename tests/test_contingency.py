from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

import pingouin as pg

df_ind = pg.read_dataset("chi2_independence")
df_mcnemar = pg.read_dataset("chi2_mcnemar")

data_ct = pd.DataFrame(
    {
        "A": [0, 1, 0, 0],
        "B": [False, True, False, False],
        "C": [1, 2, 3, 4],
        "D": ["No", "Yes", "No", "No"],
        "E": ["y", "y", "y", "y"],
    }
)


class TestContingency(TestCase):
    """Test contingency.py."""

    def test_chi2_independence(self):
        """Test function chi2_independence."""
        # Setup
        np.random.seed(42)
        mean, cov = [0.5, 0.5], [(1, 0.6), (0.6, 1)]
        x, y = np.random.multivariate_normal(mean, cov, 30).T
        data = pd.DataFrame({"x": x, "y": y})
        mask_class_1 = data > 0.5
        data[mask_class_1] = 1
        data[~mask_class_1] = 0

        # Comparing results with SciPy
        _, _, stats = pg.chi2_independence(data, x="x", y="y")
        contingency_table = pd.crosstab(data["x"], data["y"])
        for i in stats.index:
            lambda_ = stats.at[i, "lambda"]
            dof = stats.at[i, "dof"]
            chi2 = stats.at[i, "chi2"]
            p = round(stats.at[i, "pval"], 6)
            sp_chi2, sp_p, sp_dof, _ = chi2_contingency(contingency_table, lambda_=lambda_)
            np.testing.assert_allclose([chi2, p, dof], [sp_chi2, sp_p, sp_dof], rtol=1e-4)

        # Yates' correction cannot exceed the deviation between observed and expected
        data_small = pd.DataFrame(
            {"x": [0] * 20 + [1] * 21, "y": [0] * 10 + [1] * 10 + [0] * 10 + [1] * 11}
        )
        _, _, stats = pg.chi2_independence(data_small, x="x", y="y")
        sp_chi2, sp_p, _, _ = chi2_contingency(pd.crosstab(data_small["x"], data_small["y"]))
        np.testing.assert_allclose(stats.loc[0, ["chi2", "pval"]].astype(float), [sp_chi2, sp_p])

        # Testing resilience to NaN
        _, _, stats_full = pg.chi2_independence(data, x="x", y="y")
        data_nan = pd.concat(
            [data, pd.DataFrame({"x": [np.nan] * 20, "y": [1.0] * 20})], ignore_index=True
        )
        _, _, stats_nan = pg.chi2_independence(data_nan, x="x", y="y")
        # Rows with missing values are not counted in the sample size (Cramer's V and power)
        pd.testing.assert_frame_equal(stats_full, stats_nan)
        mask_nan = np.random.random(data.shape) > 0.8  # ~20% NaN values
        data[mask_nan] = np.nan
        pg.chi2_independence(data, x="x", y="y")

        # Testing validations
        def expect_assertion_error(*params):
            with pytest.raises(AssertionError):
                pg.chi2_independence(*params)

        expect_assertion_error(1, "x", "y")  # Not a pd.DataFrame
        expect_assertion_error(data, x, "y")  # Not a string
        expect_assertion_error(data, "x", y)  # Not a string
        expect_assertion_error(data, "x", "z")  # Not a column of data

        # Testing "no data" ValueError
        data["x"] = np.nan
        with pytest.raises(ValueError):
            pg.chi2_independence(data, x="x", y="y")

        # Testing degenerated case (observed == expected)
        data["x"] = 1
        data["y"] = 1
        expected, observed, stats = pg.chi2_independence(data, "x", "y")
        assert expected.iloc[0, 0] == observed.iloc[0, 0]
        assert stats.at[0, "dof"] == 0
        for i in stats.index:
            chi2 = stats.at[i, "chi2"]
            p = stats.at[i, "pval"]
            assert (chi2, p) == (0.0, 1.0)

        # Testing warning on low count
        data.iloc[0, 0] = 0
        with pytest.warns(UserWarning):
            pg.chi2_independence(data, "x", "y")

        # Comparing results with R
        # 2 x 2 contingency table (dof = 1)
        # >>> tbl = table(df$sex, df$target)
        # >>> chisq.test(tbl, correct = TRUE)
        # >>> cramersV(tbl)
        _, _, stats = pg.chi2_independence(df_ind, "sex", "target")
        assert round(stats.at[0, "chi2"], 3) == 22.717
        assert stats.at[0, "dof"] == 1
        assert np.isclose(stats.at[0, "pval"], 1.877e-06)
        assert round(stats.at[0, "cramer"], 2) == 0.27

        # 4 x 2 contingency table
        _, _, stats = pg.chi2_independence(df_ind, "cp", "target")
        assert round(stats.at[0, "chi2"], 3) == 81.686
        assert stats.at[0, "dof"] == 3.0
        assert stats.at[0, "pval"] < 2.2e-16
        assert round(stats.at[0, "cramer"], 3) == 0.519
        assert np.isclose(stats.at[0, "power"], 1.0)

    def test_chi2_mcnemar(self):
        """Test function chi2_mcnemar."""
        # Setup
        np.random.seed(42)
        mean, cov = [0.5, 0.5], [(1, 0.6), (0.6, 1)]
        x, y = np.random.multivariate_normal(mean, cov, 30).T
        data = pd.DataFrame({"x": x, "y": y})
        mask_class_1 = data > 0.5
        data[mask_class_1] = 1
        data[~mask_class_1] = 0

        # Testing validations
        def expect_assertion_error(*params):
            with pytest.raises(AssertionError):
                pg.chi2_mcnemar(*params)

        expect_assertion_error(1, "x", "y")  # Not a pd.DataFrame
        expect_assertion_error(data, x, "y")  # Not a string
        expect_assertion_error(data, "x", y)  # Not a string
        expect_assertion_error(data, "x", "z")  # Not a column of data

        # Testing happy-day
        pg.chi2_mcnemar(data, "x", "y")

        # Testing NaN incompatibility
        data.iloc[0, 0] = np.nan
        with pytest.raises(ValueError):
            pg.chi2_mcnemar(data, "x", "y")

        # Testing invalid dichotomous value
        data.iloc[0, 0] = 3
        with pytest.raises(ValueError):
            pg.chi2_mcnemar(data, "x", "y")

        # Testing error when b == 0 and c == 0
        data = pd.DataFrame({"x": [0, 0, 0, 1, 1, 1], "y": [0, 0, 0, 1, 1, 1]})
        with pytest.raises(ValueError):
            pg.chi2_mcnemar(data, "x", "y")

        # All the subjects switched: the crosstab has a single non-zero (discordant) cell
        observed, stats = pg.chi2_mcnemar(pd.DataFrame({"x": [0] * 10, "y": [1] * 10}), "x", "y")
        np.testing.assert_array_equal(observed.to_numpy(), [[0, 10], [0, 0]])
        assert np.isclose(stats.at["mcnemar", "p_exact"], 2 * 0.5**10)
        # NumPy scalars in an object column
        data = pd.DataFrame({"x": np.array([0, 1] * 5, dtype=object), "y": [np.int64(1)] * 10})
        pg.chi2_mcnemar(data, "x", "y")

        # Comparing results with R
        # 2 x 2 contingency table (dof = 1)
        # >>> tbl = table(df$treatment_X, df$treatment_Y)
        # >>> mcnemar.test(tbl, correct = TRUE)
        _, stats = pg.chi2_mcnemar(df_mcnemar, "treatment_X", "treatment_Y")
        assert round(stats.at["mcnemar", "chi2"], 3) == 20.021
        assert stats.at["mcnemar", "dof"] == 1
        assert np.isclose(stats.at["mcnemar", "p_approx"], 7.66e-06)
        # Results are compared to the exact2x2 R package
        # >>> exact2x2(tbl, paired = TRUE, midp = FALSE)
        assert np.isclose(stats.at["mcnemar", "p_exact"], 3.305e-06)
        # midp gives slightly different results
        # assert np.allclose(stats.at['mcnemar', 'p_mid'], 3.305e-06)

    def test_dichotomize_series(self):
        """Test function _dichotomize_series."""
        # Integer
        a = pg.contingency._dichotomize_series(data_ct, "A").to_numpy()
        b = pg.contingency._dichotomize_series(data_ct, "B").to_numpy()
        d = pg.contingency._dichotomize_series(data_ct, "D").to_numpy()
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(b, d)
        with pytest.raises(ValueError):
            pg.contingency._dichotomize_series(data_ct, "C")
        # Strings that are not recognized as dichotomous values
        with pytest.raises(ValueError, match="maybe"):
            pg.contingency._dichotomize_series(pd.DataFrame({"x": ["yes", "maybe"]}), "x")

    def test_dichotomous_crosstab(self):
        """Test function dichotomous_crosstab."""
        # Integer
        d1 = pg.dichotomous_crosstab(data_ct, "A", "B")
        d2 = pg.dichotomous_crosstab(data_ct, "A", "D")
        assert d1.equals(d2)
        pg.dichotomous_crosstab(data_ct, "A", "E")
        pg.dichotomous_crosstab(data_ct, "E", "A")
        # Missing levels are filled with zero counts
        np.testing.assert_array_equal(
            pg.dichotomous_crosstab(data_ct, "E", "E").to_numpy(), [[0, 0], [0, 4]]
        )
        np.testing.assert_array_equal(
            pg.dichotomous_crosstab(data_ct, "A", "E").to_numpy(), [[0, 3], [0, 1]]
        )
