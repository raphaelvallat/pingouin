import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.reliability import cronbach_alpha, intraclass_corr
from pingouin import read_dataset


class TestReliability(TestCase):
    """Test reliability.py.
    Compare against real-statistics.com
    """

    def test_cronbach_alpha(self):
        """Test function cronbach_alpha.
        Compare results with the R package psych.
        Note that this function returns slightly different results when
        missing values are present in data.
        """
        df = read_dataset("cronbach_alpha")
        alpha, ci = cronbach_alpha(data=df, items="Items", scores="Scores", subject="Subj")
        assert round(alpha, 3) == 0.592
        assert ci[0] == 0.195
        assert ci[1] == 0.840
        # With missing values
        df.loc[2, "Scores"] = np.nan
        cronbach_alpha(data=df, items="Items", scores="Scores", subject="Subj")
        # In R = psych:alpha(data, use="complete.obs")
        cronbach_alpha(
            data=df, items="Items", scores="Scores", subject="Subj", nan_policy="listwise"
        )
        # Wide format
        data = read_dataset("cronbach_wide_missing")
        alpha, _ = cronbach_alpha(data=data)
        assert round(alpha, 2) == 0.73
        alpha, _ = cronbach_alpha(data=data, nan_policy="listwise")
        assert round(alpha, 2) == 0.80

    def test_intraclass_corr(self):
        """Test function intraclass_corr
        Compare to the ICC function of the R package psych
        """
        # Example from the R package psych (Shrout and Fleiss 1979)
        df_psych = pd.DataFrame(
            {
                "S": [1, 2, 3, 4, 5, 6],
                "J1": [9, 6, 8, 7, 10, 6],
                "J2": [2, 1, 4, 1, 5, 2],
                "J3": [5, 3, 6, 2, 6, 4],
                "J4": [8, 2, 8, 6, 9, 7],
            }
        )
        df_psych = df_psych.melt(id_vars="S", var_name="J", value_name="Y")
        icc = intraclass_corr(data=df_psych, targets="S", raters="J", ratings="Y")
        np.testing.assert_almost_equal(
            np.round(icc["ICC"].to_numpy(), 2), [0.17, 0.29, 0.71, 0.44, 0.62, 0.91]
        )
        np.testing.assert_almost_equal(np.round(icc["F"], 1), [1.8, 11.0, 11.0, 1.8, 11.0, 11.0])
        np.testing.assert_almost_equal((icc["df1"]), [5] * 6)
        np.testing.assert_almost_equal((icc["df2"]), [18, 15, 15, 18, 15, 15])
        np.testing.assert_almost_equal(
            (icc["pval"]), [0.16472, 0.00013, 0.00013, 0.16472, 0.00013, 0.00013], decimal=4
        )
        lower = icc["CI95%"].explode().to_numpy()[::2].astype(float)
        upper = icc["CI95%"].explode().to_numpy()[1::2].astype(float)
        np.testing.assert_almost_equal(lower, [-0.13, 0.02, 0.34, -0.88, 0.07, 0.68])
        np.testing.assert_almost_equal(upper, [0.72, 0.76, 0.95, 0.91, 0.93, 0.99])

        # Second example (real-statistics)
        df = read_dataset("icc")
        icc = intraclass_corr(data=df, targets="Wine", raters="Judge", ratings="Scores")
        np.testing.assert_almost_equal(
            np.round(icc["ICC"].to_numpy(), 2), [0.73, 0.73, 0.73, 0.91, 0.91, 0.92]
        )
        np.testing.assert_almost_equal(np.round(icc["F"]), [12] * 6)
        np.testing.assert_almost_equal((icc["df1"]), [7] * 6)
        np.testing.assert_almost_equal((icc["df2"]), [24, 21, 21, 24, 21, 21])
        np.testing.assert_almost_equal(
            (icc["pval"]), [2.2e-06, 5.0e-06, 5.0e-06, 2.2e-06, 5.0e-06, 5.0e-06]
        )
        lower = icc["CI95%"].explode().to_numpy()[::2].astype(float)
        upper = icc["CI95%"].explode().to_numpy()[1::2].astype(float)
        np.testing.assert_almost_equal(lower, [0.43, 0.43, 0.43, 0.75, 0.75, 0.75])
        np.testing.assert_almost_equal(upper, [0.93, 0.93, 0.93, 0.98, 0.98, 0.98])
        # Test with missing values
        df["Scores"] = df["Scores"].astype(float)
        df.at[3, "Scores"] = np.nan

        # nan_policy = 'omit'
        icc = intraclass_corr(
            data=df, targets="Wine", raters="Judge", ratings="Scores", nan_policy="omit"
        )
        np.testing.assert_almost_equal(
            np.round(icc["ICC"].to_numpy(), 2), [0.75, 0.75, 0.75, 0.92, 0.92, 0.92]
        )
        np.testing.assert_almost_equal(np.round(icc["F"]), [13] * 6)
        np.testing.assert_almost_equal((icc["df1"]), [6] * 6)
        np.testing.assert_almost_equal((icc["df2"]), [21, 18, 18, 21, 18, 18])

        # nan_policy = 'raise' (default)
        with pytest.raises(ValueError):
            intraclass_corr(
                data=df, targets="Wine", raters="Judge", ratings="Scores", nan_policy="raise"
            )
