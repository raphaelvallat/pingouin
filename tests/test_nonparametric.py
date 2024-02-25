import pytest
import scipy
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.nonparametric import (
    mad,
    madmedianrule,
    mwu,
    wilcoxon,
    kruskal,
    friedman,
    cochran,
    harrelldavis,
)

np.random.seed(1234)
x = np.random.normal(size=100)
y = np.random.normal(size=100)
z = np.random.normal(size=100)
w = np.random.normal(size=(5, 10))

x2 = [20, 22, 19, 20, 22, 18, 24, 20, 19, 24, 26, 13]
y2 = [38, 37, 33, 29, 14, 12, 20, 22, 17, 25, 26, 16]


class TestNonParametric(TestCase):
    """Test nonparametric.py."""

    def test_mad(self):
        """Test function mad."""
        from scipy.stats import median_abs_deviation as mad_scp

        a = [1.2, 3, 4.5, 2.4, 5, 6.7, 0.4]
        # Compare to Matlab
        assert mad(a, normalize=False) == 1.8
        assert np.round(mad(a), 3) == np.round(1.8 * 1.4826, 3)
        # Axes handling -- Compare to SciPy
        assert np.allclose(mad_scp(w, scale="normal"), mad(w))  # Axis = 0
        assert np.allclose(mad_scp(w, scale="normal", axis=1), mad(w, axis=1))
        assert np.allclose(mad_scp(w, scale="normal", axis=None), mad(w, axis=None))
        # Missing values
        # Note that in Scipy 1.3.0, mad(axis=0/1) does not work properly
        # if data contains NaN, even when passing (nan_policy='omit')
        wnan = w.copy()
        wnan[3, 2] = np.nan
        assert np.allclose(
            mad_scp(wnan, scale="normal", axis=None, nan_policy="omit"), mad(wnan, axis=None)
        )
        assert mad(wnan, axis=0).size == wnan.shape[1]
        assert mad(wnan, axis=1).size == wnan.shape[0]
        # Now we make sure that `w` and `wnan` returns almost the same results,
        # i.e. except for the row/column with missing values
        assert np.allclose(mad(w, axis=None), mad(wnan, axis=None), atol=1e-02)
        assert sum(mad(w, axis=0) == mad(wnan, axis=0)) == 9
        assert sum(mad(w, axis=1) == mad(wnan, axis=1)) == 4

    def test_madmedianrule(self):
        """Test function madmedianrule."""
        a = [1.2, 3, 4.5, 2.4, 5, 12.7, 0.4]
        assert np.all(madmedianrule(a) == [False, False, False, False, False, True, False])

    def test_mwu(self):
        """Test function mwu"""
        mwu_scp = scipy.stats.mannwhitneyu(x, y, use_continuity=True, alternative="two-sided")
        mwu_pg = mwu(x, y, alternative="two-sided")
        mwu_pg_less = mwu(x, y, alternative="less")
        mwu_pg_greater = mwu(x, y, alternative="greater")
        # Similar to R: wilcox.test(df$x, df$y, paired = FALSE, exact = FALSE)
        # Note that the RBC value are compared to JASP in test_pairwise.py
        assert mwu_scp[0] == mwu_pg.at["MWU", "U-val"]
        assert mwu_scp[1] == mwu_pg.at["MWU", "p-val"]
        # One-sided
        assert (
            mwu_pg_less.at["MWU", "p-val"]
            == scipy.stats.mannwhitneyu(x, y, use_continuity=True, alternative="less")[1]
        )
        # CLES is compared to:
        # https://janhove.github.io/reporting/2016/11/16/common-language-effect-sizes
        assert round(mwu_pg.at["MWU", "CLES"], 4) == 0.5363
        assert round(mwu_pg_less.at["MWU", "CLES"], 4) == 0.4637
        assert round(mwu_pg_greater.at["MWU", "CLES"], 4) == 0.5363
        with pytest.raises(ValueError):
            mwu(x, y, tail="error")

    def test_wilcoxon(self):
        """Test function wilcoxon"""
        # R: wilcox.test(df$x, df$y, paired = TRUE, exact = FALSE)
        # The V value is slightly different between SciPy and R
        # The p-value, however, is almost identical
        wc_scp = scipy.stats.wilcoxon(x2, y2, correction=True)
        wc_pg = wilcoxon(x2, y2, alternative="two-sided")
        assert wc_scp[0] == wc_pg.at["Wilcoxon", "W-val"] == 20.5  # JASP
        assert wc_scp[1] == wc_pg.at["Wilcoxon", "p-val"]
        # Same but using the pre-computed difference
        # The W and p-values should be similar
        wc_pg2 = wilcoxon(np.array(x2) - np.array(y2))
        assert wc_pg.at["Wilcoxon", "W-val"] == wc_pg2.at["Wilcoxon", "W-val"]
        assert wc_pg.at["Wilcoxon", "p-val"] == wc_pg2.at["Wilcoxon", "p-val"]
        assert wc_pg.at["Wilcoxon", "RBC"] == wc_pg2.at["Wilcoxon", "RBC"]
        assert np.isnan(wc_pg2.at["Wilcoxon", "CLES"])
        wc_pg_less = wilcoxon(x2, y2, alternative="less")
        wc_pg_greater = wilcoxon(x2, y2, alternative="greater")
        # Note that the RBC value are compared to JASP in test_pairwise.py
        # The RBC values in JASP does not change according to the tail.
        assert round(wc_pg.at["Wilcoxon", "RBC"], 3) == -0.379
        assert round(wc_pg_less.at["Wilcoxon", "RBC"], 3) == -0.379
        assert round(wc_pg_greater.at["Wilcoxon", "RBC"], 3) == -0.379
        # CLES is compared to:
        # https://janhove.github.io/reporting/2016/11/16/common-language-effect-sizes
        assert round(wc_pg.at["Wilcoxon", "CLES"], 3) == 0.396
        assert round(wc_pg_less.at["Wilcoxon", "CLES"], 3) == 0.604
        assert round(wc_pg_greater.at["Wilcoxon", "CLES"], 3) == 0.396
        with pytest.raises(ValueError):
            wilcoxon(x2, y2, tail="error")

    def test_kruskal(self):
        """Test function kruskal"""
        x_nan = x.copy()
        x_nan[10] = np.nan
        df = pd.DataFrame({"DV": np.r_[x_nan, y, z], "Group": np.repeat(["A", "B", "C"], 100)})
        kruskal(data=df, dv="DV", between="Group")
        summary = kruskal(data=df, dv="DV", between="Group")
        # Compare with SciPy built-in function
        H, p = scipy.stats.kruskal(x_nan, y, z, nan_policy="omit")
        assert np.isclose(H, summary.at["Kruskal", "H"])
        assert np.allclose(p, summary.at["Kruskal", "p-unc"])

    def test_friedman(self):
        """Test function friedman"""
        df = pd.DataFrame(
            {
                "white": {
                    0: 10,
                    1: 8,
                    2: 7,
                    3: 9,
                    4: 7,
                    5: 4,
                    6: 5,
                    7: 6,
                    8: 5,
                    9: 10,
                    10: 4,
                    11: 7,
                },
                "red": {0: 7, 1: 5, 2: 8, 3: 6, 4: 5, 5: 7, 6: 9, 7: 6, 8: 4, 9: 6, 10: 7, 11: 3},
                "rose": {0: 8, 1: 5, 2: 6, 3: 4, 4: 7, 5: 5, 6: 3, 7: 7, 8: 6, 9: 4, 10: 4, 11: 3},
            }
        )

        # Compare R and SciPy
        # >>> friedman.test(data)
        Q, p = scipy.stats.friedmanchisquare(*df.to_numpy().T)
        assert np.isclose(Q, 2)
        assert np.isclose(p, 0.3678794)

        # Wide-format
        stats = friedman(df)
        assert np.isclose(stats.at["Friedman", "Q"], Q)
        assert np.isclose(stats.at["Friedman", "p-unc"], p)
        assert np.isclose(stats.at["Friedman", "ddof1"], 2)

        # Long format
        df_long = df.melt(ignore_index=False).reset_index()
        stats = friedman(data=df_long, dv="value", within="variable", subject="index")
        assert np.isclose(stats.at["Friedman", "Q"], Q)
        assert np.isclose(stats.at["Friedman", "p-unc"], p)
        assert np.isclose(stats.at["Friedman", "ddof1"], 2)

        # Compare Kendall's W
        # WARNING: I believe that the value in JASP is wrong (as of Oct 2021), because the W is
        # calculated on the transposed dataset. Indeed, to get the correct W / Q / p, one must use:
        # >>> library(DescTools)
        # >>> KendallW(t(data), correct = T, test = T)
        # Which gives the following output:
        # Kendall chi - squared = 2, df = 2, subjects = 3, raters = 12, p - value = 0.3679
        # alternative hypothesis: Wt is greater 0
        # sample estimates: 0.08333333
        assert np.isclose(stats.at["Friedman", "W"], 0.08333333)

        # Using the F-test method, which is more conservative
        stats_f = friedman(df, method="f")
        assert stats_f.at["Friedman", "p-unc"] > stats.at["Friedman", "p-unc"]

    def test_cochran(self):
        """Test function cochran
        http://www.real-statistics.com/anova-repeated-measures/cochrans-q-test/
        """
        from pingouin import read_dataset

        df = read_dataset("cochran")
        st = cochran(dv="Energetic", within="Time", subject="Subject", data=df)
        assert round(st.at["cochran", "Q"], 3) == 6.706
        assert np.isclose(st.at["cochran", "p-unc"], 0.034981)
        # With Categorical
        df["Time"] = df["Time"].astype("category")
        df["Subject"] = df["Subject"].astype("category")
        df["Time"] = df["Time"].cat.add_categories("Unused")
        st = cochran(dv="Energetic", within="Time", subject="Subject", data=df)
        assert round(st.at["cochran", "Q"], 3) == 6.706
        assert np.isclose(st.at["cochran", "p-unc"], 0.034981)
        # With a NaN value
        df.loc[2, "Energetic"] = np.nan
        cochran(dv="Energetic", within="Time", subject="Subject", data=df)

    def test_harrelldavis(self):
        """Test Harrel-Davis estimation of :math:`q^{th}` quantile."""
        a = [
            77,
            87,
            88,
            114,
            151,
            210,
            219,
            246,
            253,
            262,
            296,
            299,
            306,
            376,
            428,
            515,
            666,
            1310,
            2611,
        ]
        assert np.isclose(harrelldavis(a, quantile=0.5), 271.72120054908913)
        harrelldavis(x=x, quantile=np.arange(0.1, 1, 0.1))
        assert np.isclose(harrelldavis(a, [0.25, 0.5, 0.75])[1], 271.72120054908913)
        # Test multiple axis
        p = np.random.normal(0, 1, (10, 100))

        def func(a, axes):
            return harrelldavis(a, [0.25, 0.75], axes)

        np.testing.assert_array_almost_equal(
            harrelldavis(p, [0.25, 0.75], 0), np.apply_over_axes(func, p, 0)
        )

        np.testing.assert_array_almost_equal(
            harrelldavis(p, [0.25, 0.75], -1), np.apply_over_axes(func, p, 1)
        )
