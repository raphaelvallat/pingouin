import pytest
import numpy as np
import pandas as pd
import pingouin as pg
from unittest import TestCase
from scipy.stats import chi2_contingency

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

        # Testing resilience to NaN
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

    def test_dichotomous_crosstab(self):
        """Test function dichotomous_crosstab."""
        # Integer
        d1 = pg.dichotomous_crosstab(data_ct, "A", "B")
        d2 = pg.dichotomous_crosstab(data_ct, "A", "D")
        assert d1.equals(d2)
        pg.dichotomous_crosstab(data_ct, "A", "E")
        pg.dichotomous_crosstab(data_ct, "E", "A")
        with pytest.raises(ValueError):
            pg.dichotomous_crosstab(data_ct, "E", "E")

    def test_ransacking(self):
        """Test function ransacking."""
        # We use the 'chi2_independence' dataset that is already read into df_ind.
        # row_var = 'cp' and col_var = 'restecg' as in your example usage.
        results = pg.ransacking(
            data=df_ind, row_var="cp", col_var="restecg", alpha=0.05, adjusted=True
        )

        # 1) Check the output type
        self.assertIsInstance(results, pd.DataFrame, "Result should be a DataFrame.")

        # 2) Check the column names
        expected_columns = {
            "Row",
            "Column",
            "Odds Ratio",
            "Log Odds Ratio",
            "Standard Error",
            "Z Value",
            "Critical Z (global dof)",
            "Adjusted Critical Z (global dof)",
            "Unadjusted Result",
            "Adjusted Result",
            "2x2 Table",
            "DOF",
        }
        self.assertEqual(
            set(results.columns), expected_columns, "DataFrame does not have the expected columns."
        )

        # 3) Check the number of rows: should match r*c from a crosstab
        ctab = pd.crosstab(df_ind["cp"], df_ind["restecg"])
        expected_rows = ctab.shape[0] * ctab.shape[1]
        self.assertEqual(
            len(results), expected_rows, "Number of rows in 'ransacking' output does not match r*c."
        )

        # 4) Check DOF is present and is integer
        self.assertIn("DOF", results.columns, "'DOF' column is missing.")
        self.assertTrue(
            np.issubdtype(results["DOF"].dtype, np.integer), "'DOF' should be an integer type."
        )

        # 5) Check each '2x2 Table' is a list of lists of shape (2,2)
        for tbl in results["2x2 Table"]:
            self.assertIsInstance(tbl, list, "'2x2 Table' must be a Python list (of lists).")
            self.assertEqual(len(tbl), 2, "Each '2x2 Table' must have 2 rows.")
            for row in tbl:
                self.assertIsInstance(row, list, "Each row in '2x2 Table' must be a list.")
                self.assertEqual(len(row), 2, "Each row in '2x2 Table' must have 2 columns.")

        # 6) Check "Unadjusted Result" and "Adjusted Result" are in allowed set
        allowed_results = {"reject", "fail to reject"}
        for res in results["Unadjusted Result"]:
            self.assertIn(
                res,
                allowed_results,
                "'Unadjusted Result' should be either 'reject' or 'fail to reject'.",
            )
        for res in results["Adjusted Result"]:
            self.assertIn(
                res,
                allowed_results,
                "'Adjusted Result' should be either 'reject' or 'fail to reject'.",
            )

        # 7) Check that it raises KeyError when row/column var do not exist
        with self.assertRaises(KeyError):
            pg.ransacking(data=df_ind, row_var="DoesNotExist", col_var="restecg")
        with self.assertRaises(KeyError):
            pg.ransacking(data=df_ind, row_var="cp", col_var="DoesNotExist")

        # 8) Optionally: Check that a small toy dataset with all same values doesn't crash
        # (like a degenerate case).
        deg_data = pd.DataFrame({"A": ["X", "X"], "B": ["Y", "Y"]})
        deg_results = pg.ransacking(data=deg_data, row_var="A", col_var="B")
        self.assertEqual(len(deg_results), 1, "Degenerate table should only produce 1 result.")
