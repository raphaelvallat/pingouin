from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from pingouin import pairwise_corr, read_dataset
from pingouin.correlation import (
    _bvn_cdf,
    _polychoric_probs,
    bicor,
    corr,
    distance_corr,
    partial_corr,
    polychoric,
    rcorr,
    rm_corr,
    skipped,
)


class TestCorrelation(TestCase):
    """Test correlation.py.

    See the test_correlation.R file.
    """

    def test_corr(self):
        """Test function corr

        Compare to R `correlation` package. See test_correlation.R file.
        """
        np.random.seed(123)
        mean, cov = [4, 6], [(1, 0.6), (0.6, 1)]
        x, y = np.random.multivariate_normal(mean, cov, 30).T
        x2, y2 = x.copy(), y.copy()
        x[3], y[5] = 12, -8
        x2[3], y2[5] = 7, 2.6

        # Pearson correlation
        stats = corr(x, y, method="pearson")
        assert np.isclose(stats.loc["pearson", "r"], 0.1761221)
        assert np.isclose(stats.loc["pearson", "p_val"], 0.3518659)
        assert stats.loc["pearson", "CI95"][0] == round(-0.1966232, 2)
        assert stats.loc["pearson", "CI95"][1] == round(0.5043872, 2)
        # - One-sided: greater
        stats = corr(x, y, method="pearson", alternative="greater")
        assert np.isclose(stats.loc["pearson", "r"], 0.1761221)
        assert np.isclose(stats.loc["pearson", "p_val"], 0.175933)
        assert stats.loc["pearson", "CI95"][0] == round(-0.1376942, 2)
        assert stats.loc["pearson", "CI95"][1] == 1
        # - One-sided: less
        stats = corr(x, y, method="pearson", alternative="less")
        assert np.isclose(stats.loc["pearson", "r"], 0.1761221)
        assert np.isclose(stats.loc["pearson", "p_val"], 0.824067)
        assert stats.loc["pearson", "CI95"][0] == -1
        assert stats.loc["pearson", "CI95"][1] == round(0.4578044, 2)

        # Spearman correlation
        stats = corr(x, y, method="spearman")
        assert np.isclose(stats.loc["spearman", "r"], 0.4740823)
        assert np.isclose(stats.loc["spearman", "p_val"], 0.008129768)
        # CI are calculated using a different formula for Spearman in R
        # assert stats.loc['spearman', 'CI95'][0] == round(0.1262988, 2)
        # assert stats.loc['spearman', 'CI95'][1] == round(0.7180799, 2)

        # Kendall correlation
        # R uses a different estimation method than scipy for the p-value
        stats = corr(x, y, method="kendall")
        assert np.isclose(stats.loc["kendall", "r"], 0.3517241)
        # Skipped correlation -- compare with robust corr toolbox
        # https://sourceforge.net/projects/robustcorrtool/
        stats = corr(x, y, method="skipped")
        assert round(stats.loc["skipped", "r"], 4) == 0.5123
        assert stats.loc["skipped", "outliers"] == 2
        _ = corr(x2, y2, method="skipped")
        # Fails in sklearn ≥1.8, see https://github.com/scikit-learn/scikit-learn/issues/23162
        # assert round(sk_sp.loc["skipped", "r"], 4) == 0.5123
        # assert sk_sp.loc["skipped", "outliers"] == 2
        # Pearson skipped correlation
        _ = corr(x2, y2, method="skipped", corr_type="pearson")
        # assert np.round(sk_pe.loc["skipped", "r"], 4) == 0.5254
        # assert sk_pe.loc["skipped", "outliers"] == 2
        # assert not sk_sp.equals(sk_pe)
        # Shepherd
        stats = corr(x, y, method="shepherd")
        assert np.isclose(stats.loc["shepherd", "r"], 0.5123153)
        assert np.isclose(stats.loc["shepherd", "p_val"], 0.005316)
        assert stats.loc["shepherd", "outliers"] == 2
        _, _, outliers = skipped(x, y, corr_type="pearson")
        assert outliers.size == x.size
        assert stats.loc["shepherd", "n"] == 30
        # Percbend -- compare with robust corr toolbox
        stats = corr(x, y, method="percbend")
        assert round(stats.loc["percbend", "r"], 4) == 0.4843
        assert np.isclose(stats.loc["percbend", "r"], 0.4842686)
        assert np.isclose(stats.loc["percbend", "p_val"], 0.006693313)
        stats = corr(x2, y2, method="percbend")
        assert round(stats.loc["percbend", "r"], 4) == 0.4843
        stats = corr(x, y, method="percbend", beta=0.5)
        assert round(stats.loc["percbend", "r"], 4) == 0.4848
        # Compare biweight correlation to astropy
        stats = corr(x, y, method="bicor")
        assert np.isclose(stats.loc["bicor", "r"], 0.4951418)
        assert np.isclose(stats.loc["bicor", "p_val"], 0.005403701)
        assert stats.loc["bicor", "CI95"][0] == round(0.1641553, 2)
        assert stats.loc["bicor", "CI95"][1] == round(0.7259185, 2)
        stats = corr(x, y, method="bicor", c=5)
        assert np.isclose(stats.loc["bicor", "r"], 0.4940706950017)
        # Not normally distributed
        z = np.random.uniform(size=30)
        corr(x, z, method="pearson")
        # With NaN values
        x[3] = np.nan
        corr(x, y)
        # With the same array
        # Disabled because of AppVeyor failure
        # assert corr(x, x).loc['pearson', 'BF10'] == str(np.inf)
        # Wrong argument
        with pytest.raises(ValueError):
            corr(x, y, method="error")
        with pytest.raises(ValueError):
            corr(x, y, tail="error")
        # Compare BF10 with JASP
        df = read_dataset("pairwise_corr")
        stats = corr(df["Neuroticism"], df["Extraversion"])
        assert np.isclose(1 / float(stats.at["pearson", "BF10"]), 1.478e-13)
        # Perfect correlation, CI and power should be 1, BF should be Inf
        # https://github.com/raphaelvallat/pingouin/issues/195
        stats = corr(x, x)
        assert np.isclose(stats.at["pearson", "r"], 1)
        assert np.isclose(stats.at["pearson", "power"], 1)

        # Perfect correlation with percbend method
        # https://github.com/raphaelvallat/pingouin/issues/453
        stats = corr(x, x, method="percbend")  # calls _correl_pvalue
        assert np.isclose(stats.at["percbend", "r"], 1)
        assert np.isclose(stats.at["percbend", "p_val"], 0)
        # Perfect correlation in the opposite direction of a one-sided test
        assert corr(x, x, alternative="less").at["pearson", "p_val"] == 1
        assert corr(x, -x, alternative="greater").at["pearson", "p_val"] == 1
        assert corr(x, x, alternative="greater").at["pearson", "p_val"] == 0

        # One-sided Kendall p-values use the Kendall null distribution, not a t-approximation
        from scipy.stats import kendalltau

        rng = np.random.default_rng(0)
        xk = rng.normal(size=30)
        yk = 0.3 * xk + rng.normal(size=30)
        for alt in ["greater", "less"]:
            pval = corr(xk, yk, method="kendall", alternative=alt).at["kendall", "p_val"]
            assert np.isclose(pval, kendalltau(xk, yk, alternative=alt)[1])

        # When one column is a constant, the correlation is not defined
        # and Pingouin return a DataFrame full of NaN, except for ``n``
        x, y = [1, 1, 1], [1, 2, 3]
        stats = corr(x, y)
        assert stats.at["pearson", "n"]
        assert np.isnan(stats.at["pearson", "r"])
        # Biweight midcorrelation returns NaN when MAD is not defined
        assert np.isnan(bicor(np.array([1, 1, 1, 1, 0, 1]), np.arange(6))[0])

    def test_polychoric(self):
        """Test function polychoric, and method="polychoric" in corr, pairwise_corr and rcorr.

        Compare to the R packages polycor (``polychor(..., ML = FALSE, std.err = TRUE)``) and
        psych (``tetrachoric``). See the test_correlation.R file.

        polychor minimizes the same likelihood with BFGS and a numerical Hessian, and agrees with
        Pingouin up to 1e-4 for the correlation and 1e-5 for the standard error.
        """
        # 1) Contingency tables: (table, rho, se, row thresholds, column thresholds) in polycor
        t55 = [22, 14, 7, 2, 1, 16, 31, 20, 9, 3, 8, 25, 42, 24, 9, 3, 11, 26, 33, 15]
        t55 = np.reshape(t55 + [1, 4, 10, 17, 21], (5, 5))
        t55_sparse = [99, 105, 183, 11, 0, 21, 56, 123, 45, 0, 3, 33, 104, 85, 1]
        t55_sparse = np.reshape(t55_sparse + [0, 2, 33, 59, 20, 0, 0, 0, 8, 9], (5, 5))
        tables = {
            "t22": ([[40, 10], [15, 35]], 0.71271033, 0.09543623, [0], [0.125661]),
            "t33": (  # Olsson 1979
                [[13, 6, 0], [69, 113, 22], [41, 132, 104]],
                0.49136798,
                0.04730483,
                [-1.774382, -0.135774],
                [-0.687131, 0.668209],
            ),
            "t43": (
                [[131, 71, 20], [217, 207, 112], [213, 337, 257], [52, 139, 244]],
                0.42695267,
                0.02183087,
                [-1.221227, -0.308108, 0.780664],
                [-0.505796, 0.477509],
            ),
            "t55": (
                t55,
                0.59101429,
                0.03479572,
                [-1.160146, -0.428277, 0.313355, 1.072663],
                [-1.109117, -0.355887, 0.363037, 1.121601],
            ),
            "t55_sparse": (
                t55_sparse,
                0.66749876,
                0.01988958,
                [-0.258527, 0.366489, 1.121677, 2.120072],
                [-1.160120, -0.470497, 0.712751, 1.880794],
            ),
            "t33_sparse": (
                [[5, 2, 0], [3, 6, 1], [0, 2, 4]],
                0.82007140,
                0.10464230,
                [-0.511936, 0.640667],
                [-0.391196, 0.781034],
            ),
            "t22_sparse": (
                [[61661, 85], [1610, 20]],
                0.34807946,
                0.04522722,
                [1.947799],
                [2.937045],
            ),
            "t33_neg": (
                [[4, 19, 30], [12, 62, 41], [25, 33, 9]],
                -0.46825778,
                0.06447678,
                [-0.753643, 0.567738],
                [-0.936655, 0.411302],
            ),
            # Empty categories are removed
            "t33_empty": (
                [[20, 0, 5], [0, 0, 0], [6, 0, 25]],
                0.81414700,
                0.09734119,
                [-0.134690],
                [-0.089642],
            ),
        }
        for table, rho, se, tau_x, tau_y in tables.values():
            stats = polychoric(table=table)
            assert stats.index.tolist() == ["polychoric"]
            assert stats.columns.tolist() == [
                "n",
                "r",
                "se",
                "CI95",
                "p_val",
                "thresholds_x",
                "thresholds_y",
            ]
            assert stats.at["polychoric", "n"] == np.sum(table)
            assert np.isclose(stats.at["polychoric", "r"], rho, atol=1e-4)
            assert np.isclose(stats.at["polychoric", "se"], se, atol=1e-5)
            np.testing.assert_allclose(stats.at["polychoric", "thresholds_x"], tau_x, atol=1e-6)
            np.testing.assert_allclose(stats.at["polychoric", "thresholds_y"], tau_y, atol=1e-6)
            # The transposed table gives the same correlation, with swapped thresholds
            stats_t = polychoric(table=np.transpose(table))
            assert np.isclose(stats_t.at["polychoric", "r"], stats.at["polychoric", "r"])
            assert np.isclose(stats_t.at["polychoric", "se"], stats.at["polychoric", "se"])
            np.testing.assert_allclose(stats_t.at["polychoric", "thresholds_y"], tau_x, atol=1e-6)
        # A DataFrame (e.g. pd.crosstab) is a valid table
        stats = polychoric(table=pd.DataFrame(tables["t33"][0]))
        assert np.isclose(stats.at["polychoric", "r"], 0.49136798, atol=1e-4)

        # 2) Tetrachoric correlation: compare with psych::tetrachoric. An exact value is
        # available for median splits: rho = sin(2 * pi * (p11 - 1 / 4))
        assert np.isclose(polychoric(table=[[40, 10], [15, 35]]).at["polychoric", "r"], 0.71273186)
        stats = polychoric(table=[[61661, 85], [1610, 20]])
        assert np.isclose(stats.at["polychoric", "r"], 0.34807313, atol=1e-4)
        for p11 in [0.05, 0.2, 0.4, 0.49]:
            stats = polychoric(table=[[p11, 0.5 - p11], [0.5 - p11, p11]])
            assert np.isclose(stats.at["polychoric", "r"], np.sin(2 * np.pi * (p11 - 0.25)))
            assert stats.at["polychoric", "n"] == 1
        # - With the correction of psych for empty cells (correct = 0.5)
        corrected = [
            ([[30, 0], [10, 20]], 0.93945051, 0.010358, 0.415623),
            ([[44268, 14], [193, 0]], 0.23231820, 2.623568, 3.408988),
            ([[62503, 768], [105, 0]], -0.10196352, 2.935574, 2.253115),
        ]
        for table, rho, tau_x, tau_y in corrected:
            stats = polychoric(table=table, correction=0.5)
            assert stats.at["polychoric", "n"] == np.sum(table)
            assert np.isclose(stats.at["polychoric", "r"], rho, atol=1e-4)
            assert np.isclose(stats.at["polychoric", "thresholds_x"][0], tau_x, atol=1e-6)
            assert np.isclose(stats.at["polychoric", "thresholds_y"][0], tau_y, atol=1e-6)
            assert stats.equals(polychoric(table=table, correction=True))
        # - The correction has no effect without empty cells
        assert polychoric(table=tables["t22"][0], correction=0.5).equals(
            polychoric(table=tables["t22"][0])
        )

        # 3) Boundary: the likelihood is monotone and the correlation is -1 or 1
        # (polychor stops at maxcor = 0.9999)
        perfect = np.diag([10, 20, 30])
        boundary = [
            (perfect, 1),
            (perfect[:, ::-1], -1),
            ([[30, 0], [10, 20]], 1),
            ([[44268, 14], [193, 0]], -1),
        ]
        for table, rho in boundary:
            with pytest.warns(UserWarning, match="boundary"):
                stats = polychoric(table=table)
            assert stats.at["polychoric", "r"] == rho
            assert np.isnan(stats.at["polychoric", "se"])
            assert np.isnan(stats.at["polychoric", "p_val"])
            assert np.isnan(stats.at["polychoric", "CI95"]).all()
            assert stats.at["polychoric", "thresholds_x"].size == np.shape(table)[0] - 1
        # - Fewer than two categories
        for table in [[[5, 7, 9], [0, 0, 0]], [[5], [7]], np.zeros((3, 3))]:
            with pytest.warns(UserWarning, match="at least two"):
                stats = polychoric(table=table)
            assert np.isnan(stats.at["polychoric", "r"])
            assert np.isnan(stats.at["polychoric", "se"])
            assert stats.at["polychoric", "n"] == np.sum(table)
        with pytest.warns(UserWarning, match="at least two"):
            assert np.isnan(polychoric([1, 1, 1, 1], [1, 2, 3, 4]).at["polychoric", "r"])

        # 4) Raw ordinal data: (seed, rho, n, cuts_x, cuts_y), then rho, se and thresholds in R
        def ordinal(seed, rho, n, cuts_x, cuts_y):
            rng = np.random.default_rng(seed)
            z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
            return (
                np.digitize(z[:, 0], cuts_x).astype(float),
                np.digitize(z[:, 1], cuts_y).astype(float),
            )

        x1, y1 = ordinal(1, 0.5, 80, [-0.5, 0.5], [-1, 0, 1])
        x2, y2 = ordinal(2, -0.4, 60, [0], [-0.3, 0.6])
        x3, y3 = ordinal(3, 0.7, 100, [-1, 0, 1], [-1.2, -0.4, 0.4, 1.2])
        # Missing values are removed pairwise
        x3[[4, 17, 30, 58, 91]] = np.nan
        y3[[8, 17, 77]] = np.nan
        datasets = [
            (
                (x1, y1, 80, 0.54999592, 0.09381310),
                [-0.714367, 0.488776],
                [-1.281552, 0, 1.036433],
            ),
            ((x2, y2, 60, -0.27768838, 0.17300905), [0.296738], [-0.430727, 0.477040]),
            (
                (x3, y3, 93, 0.70877089, 0.05599440),
                [-0.826356, 0.231142, 1.034130],
                [-1.130978, -0.430727, 0.401337, 1.365669],
            ),
        ]
        for (x, y, n, rho, se), tau_x, tau_y in datasets:
            stats = polychoric(x, y)
            assert stats.at["polychoric", "n"] == n
            assert np.isclose(stats.at["polychoric", "r"], rho, atol=1e-4)
            assert np.isclose(stats.at["polychoric", "se"], se, atol=1e-5)
            np.testing.assert_allclose(stats.at["polychoric", "thresholds_x"], tau_x, atol=1e-6)
            np.testing.assert_allclose(stats.at["polychoric", "thresholds_y"], tau_y, atol=1e-6)
            # Same result with the contingency table
            mask = ~(np.isnan(x) | np.isnan(y))
            table = pd.crosstab(x[mask], y[mask])
            assert np.isclose(
                polychoric(table=table).at["polychoric", "r"], stats.at["polychoric", "r"]
            )
            # corr gives the same correlation, with a Wald CI and p-value, and no power
            stats_corr = corr(x, y, method="polychoric")
            assert stats_corr.columns.tolist() == ["n", "r", "CI95", "p_val", "power"]
            assert stats_corr.at["polychoric", "n"] == n
            assert stats_corr.at["polychoric", "r"] == stats.at["polychoric", "r"]
            assert stats_corr.at["polychoric", "p_val"] == stats.at["polychoric", "p_val"]
            assert np.isnan(stats_corr.at["polychoric", "power"])
            np.testing.assert_array_equal(
                stats_corr.at["polychoric", "CI95"], stats.at["polychoric", "CI95"]
            )

        # 5) Wald confidence interval and p-values, from the R estimates of the first dataset

        rho, se = 0.54999592, 0.09381310
        stats = polychoric(x1, y1)
        assert np.isclose(stats.at["polychoric", "p_val"], 2 * norm.sf(rho / se), rtol=1e-3)
        assert stats.at["polychoric", "CI95"][0] == round(rho - norm.ppf(0.975) * se, 2)
        assert stats.at["polychoric", "CI95"][1] == round(rho + norm.ppf(0.975) * se, 2)
        stats = polychoric(x1, y1, confidence=0.9)
        ci90 = [rho - norm.ppf(0.95) * se, rho + norm.ppf(0.95) * se]
        np.testing.assert_allclose(stats.at["polychoric", "CI90"], ci90, atol=1e-4)
        # - One-sided alternatives
        for func in [
            polychoric,
            lambda *args, **kwargs: corr(*args, method="polychoric", **kwargs),
        ]:
            greater = func(x1, y1, alternative="greater")
            assert np.isclose(greater.at["polychoric", "p_val"], norm.sf(rho / se), rtol=1e-3)
            assert greater.at["polychoric", "CI95"][0] == round(rho - norm.ppf(0.95) * se, 2)
            assert greater.at["polychoric", "CI95"][1] == 1
            less = func(x1, y1, alternative="less")
            assert np.isclose(less.at["polychoric", "p_val"], norm.cdf(rho / se), rtol=1e-3)
            assert less.at["polychoric", "CI95"][0] == -1
            assert less.at["polychoric", "CI95"][1] == round(rho + norm.ppf(0.95) * se, 2)
            # Negative correlation (rho = -0.2777, se = 0.1730 in R)
            less = func(x2, y2, alternative="less")
            assert np.isclose(
                less.at["polychoric", "p_val"], norm.cdf(-0.27768838 / 0.17300905), 1e-3
            )
        # - The confidence interval is clipped to [-1, 1]
        assert (
            polychoric(table=[[30, 0], [10, 20]], correction=0.5).at["polychoric", "CI95"][1] == 1
        )

        # 6) Ordered Categorical: the order of the categories is respected
        levels_x, levels_y = ["low", "medium", "high"], ["never", "rarely", "often", "always"]
        df = pd.DataFrame(
            {
                "x": pd.Categorical.from_codes(x1.astype(int), levels_x, ordered=True),
                "y": pd.Categorical.from_codes(y1.astype(int), levels_y, ordered=True),
                "x_num": x1,
                "y_num": y1,
            }
        )
        expected = polychoric(x1, y1)
        for stats in [
            polychoric(df["x"], df["y"]),
            polychoric("x", "y", data=df),
            polychoric(df["x"].array, df["y_num"]),
            polychoric("x_num", "y", data=df),
        ]:
            pd.testing.assert_frame_equal(stats, expected)
        assert (
            corr(df["x"], df["y"], method="polychoric").at["polychoric", "r"]
            == (expected.at["polychoric", "r"])
        )
        # - Reversing the order of the categories of one variable flips the sign
        reverse = df["x"].cat.reorder_categories(levels_x[::-1])
        stats = polychoric(reverse, df["y"])
        assert np.isclose(stats.at["polychoric", "r"], -expected.at["polychoric", "r"])
        # - Unused categories and missing values are ignored
        unused = df["x"].cat.add_categories(["very high"])
        pd.testing.assert_frame_equal(polychoric(unused, df["y"]), expected)
        x3_cat = pd.Categorical.from_codes(
            np.nan_to_num(x3, nan=-1).astype(int), list("abcd"), ordered=True
        )
        pd.testing.assert_frame_equal(polychoric(x3_cat, y3), polychoric(x3, y3))
        # - Boolean and integer variables
        pd.testing.assert_frame_equal(
            polychoric(x2.astype(bool), y2.astype(int)), polychoric(x2, y2)
        )
        # - Unordered Categorical and strings are not ordinal
        with pytest.raises(ValueError):
            polychoric(df["x"].cat.as_unordered(), df["y"])
        with pytest.raises(ValueError):
            polychoric(df["x"].astype(str), df["y"])
        with pytest.raises(ValueError):
            corr(df["x"].astype(str), df["y"], method="polychoric")

        # 7) Wrong arguments
        with pytest.raises(ValueError):
            polychoric(x1, y1, table=[[40, 10], [15, 35]])
        with pytest.raises(ValueError):
            polychoric(x1)
        with pytest.raises(ValueError):
            polychoric(table=[1, 2, 3])
        with pytest.raises(ValueError):
            polychoric(table=[[40, -1], [15, 35]])
        with pytest.raises(AssertionError):
            polychoric(x1, y1, alternative="error")
        with pytest.raises(AssertionError):
            polychoric(x1, y1[:-1])

        # 8) pairwise_corr and rcorr
        data = pd.DataFrame({"x1": x1, "y1": y1, "y3": y3[:80]})
        data.iloc[:4, 0] = np.nan
        pairs = pairwise_corr(data, method="polychoric", padjust="holm")
        assert pairs["method"].eq("polychoric").all()
        assert pairs.columns.tolist() == [
            "X",
            "Y",
            "method",
            "alternative",
            "n",
            "r",
            "CI95",
            "p_unc",
            "p_corr",
            "p_adjust",
        ]
        assert pairs["n"].tolist() == [76, 73, 77]
        for i, (a, b) in enumerate(zip(pairs["X"], pairs["Y"])):
            stats = polychoric(a, b, data=data)
            assert pairs.at[i, "r"] == stats.at["polychoric", "r"]
            assert pairs.at[i, "p_unc"] == stats.at["polychoric", "p_val"]
        greater = pairwise_corr(data, method="polychoric", alternative="greater")
        assert np.isclose(
            greater.at[0, "p_unc"],
            polychoric("x1", "y1", data=data, alternative="greater").at["polychoric", "p_val"],
        )
        # - rcorr: correlations on the lower triangle and Wald p-values on the upper triangle
        mat = rcorr(data, method="polychoric", stars=False, decimals=6)
        for i, (a, b) in enumerate(zip(pairs["X"], pairs["Y"])):
            assert np.isclose(float(mat.at[b, a]), pairs.at[i, "r"], atol=1e-6)
            assert np.isclose(float(mat.at[a, b]), pairs.at[i, "p_unc"], atol=1e-6)
        mat = rcorr(data, method="polychoric", stars=False, decimals=6, padjust="holm")
        for i, (a, b) in enumerate(zip(pairs["X"], pairs["Y"])):
            assert np.isclose(float(mat.at[a, b]), pairs.at[i, "p_corr"], atol=1e-6)
        assert rcorr(data, method="polychoric").at["x1", "y1"] == "***"
        assert rcorr(data, method="polychoric", upper="n").at["x1", "y3"] == 73

        # 7) No complete pair of observations: the contingency table is empty
        rng = np.random.default_rng(123)
        z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
        x, y = np.digitize(z[:, 0], [-0.5, 0.5]), np.digitize(z[:, 1], [0])
        x_half, y_half = np.r_[x[:50], np.full(50, np.nan)], np.r_[np.full(50, np.nan), y[50:]]
        with pytest.warns(UserWarning, match="at least two non-empty categories"):
            stats = polychoric(x_half, y_half)
        assert stats.at["polychoric", "n"] == 0
        assert np.isnan(stats.at["polychoric", "r"])
        assert stats.at["polychoric", "thresholds_x"].size == 0
        # Same in rcorr, without affecting the other pairs
        df_half = pd.DataFrame({"x": x_half, "y": y_half, "z": np.r_[y[:50], x[50:]]})
        mat = rcorr(df_half, method="polychoric", stars=False, decimals=6)
        assert np.isnan(float(mat.at["y", "x"])) and np.isnan(float(mat.at["x", "y"]))
        assert np.isclose(
            float(mat.at["z", "x"]), polychoric("x", "z", data=df_half).at["polychoric", "r"]
        )
        assert rcorr(df_half, method="polychoric", upper="n").at["x", "y"] == 0

        # 8) Near-perfect association with a correction for the empty cells. The smallest cell
        # probabilities are close to the precision of the bivariate normal CDF, and the Newton
        # steps stall around 1e-11, i.e. all the iterations are used. The estimate must still be
        # the maximum of the likelihood, with a standard error from its curvature.
        t42 = np.array([[40, 0], [0, 38], [0, 42], [0, 23]])
        stats = polychoric(table=t42, correction=0.5)
        r, se = stats.at["polychoric", "r"], stats.at["polychoric", "se"]
        assert 0.9 < r < 0.9999
        t42_corr = np.where(t42 == 0, 0.5, t42)
        tau_x, tau_y = (
            stats.at["polychoric", "thresholds_x"],
            stats.at["polychoric", "thresholds_y"],
        )

        def loglik(rho):
            return np.sum(t42_corr * np.log(_polychoric_probs(tau_x, tau_y, rho)))

        h = 1e-4
        assert loglik(r) > max(loglik(r - h), loglik(r + h))
        assert abs(loglik(r + h) - loglik(r - h)) / (2 * h) < 1e-3  # Score
        info = -(loglik(r + h) - 2 * loglik(r) + loglik(r - h)) / h**2
        assert np.isclose(se, 1 / np.sqrt(info), rtol=1e-3)

        # 9) The bivariate normal CDF does not depend on the precision of SciPy's
        # multivariate_normal.cdf, which is randomized with errors up to 1e-5 in SciPy 1.16.
        # Compare to Phi(h) * Phi(k) plus the integral of the bivariate density from 0 to rho,
        # including thresholds of zero and of opposite signs.
        def bvn_quad(h, k, rho):
            def pdf(t):
                return np.exp(-(h**2 - 2 * t * h * k + k**2) / (2 * (1 - t**2))) / (
                    2 * np.pi * np.sqrt(1 - t**2)
                )

            return norm.cdf(h) * norm.cdf(k) + quad(pdf, 0, rho)[0]

        for h, k in [(0, 0), (0, 0.7), (0, -0.7), (-1.2, 0), (0.4, 1.3), (-0.4, 1.3), (-2, -0.5)]:
            for rho in [-0.9, -0.3, 0, 0.5, 0.95]:
                cdf = _bvn_cdf(np.array([h]), np.array([k]), rho)[0]
                assert np.isclose(cdf, bvn_quad(h, k, rho), rtol=0, atol=1e-10)
                assert np.isclose(cdf, _bvn_cdf(np.array([k]), np.array([h]), rho)[0], atol=1e-14)

    def test_partial_corr(self):
        """Test function partial_corr.

        Compare with the R package ppcor (which is also used by JASP).
        """
        df = read_dataset("partial_corr")
        #######################################################################
        # PARTIAL CORRELATION
        #######################################################################
        # With one covariate
        pc = partial_corr(data=df, x="x", y="y", covar="cv1")
        assert round(pc.at["pearson", "r"], 7) == 0.5681692
        assert round(pc.at["pearson", "p_val"], 9) == 0.001303059
        # With two covariates
        pc = partial_corr(data=df, x="x", y="y", covar=["cv1", "cv2"])
        assert round(pc.at["pearson", "r"], 7) == 0.5344372
        assert round(pc.at["pearson", "p_val"], 9) == 0.003392904
        # With three covariates
        # in R: pcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")])
        pc = partial_corr(data=df, x="x", y="y", covar=["cv1", "cv2", "cv3"])
        assert round(pc.at["pearson", "r"], 7) == 0.4926007
        assert round(pc.at["pearson", "p_val"], 9) == 0.009044164
        # Method == "spearman"
        pc = partial_corr(data=df, x="x", y="y", covar=["cv1", "cv2", "cv3"], method="spearman")
        assert round(pc.at["spearman", "r"], 7) == 0.5209208
        assert round(pc.at["spearman", "p_val"], 9) == 0.005336187

        #######################################################################
        # SEMI-PARTIAL CORRELATION
        #######################################################################
        # With one covariate
        pc = partial_corr(data=df, x="x", y="y", y_covar="cv1")
        assert round(pc.at["pearson", "r"], 7) == 0.5670793
        assert round(pc.at["pearson", "p_val"], 9) == 0.001337718
        # With two covariates
        pc = partial_corr(data=df, x="x", y="y", y_covar=["cv1", "cv2"])
        assert round(pc.at["pearson", "r"], 7) == 0.5097489
        assert round(pc.at["pearson", "p_val"], 9) == 0.005589687
        # With three covariates
        # in R: spcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")])
        pc = partial_corr(data=df, x="x", y="y", y_covar=["cv1", "cv2", "cv3"])
        assert round(pc.at["pearson", "r"], 7) == 0.4212351
        assert round(pc.at["pearson", "p_val"], 8) == 0.02865483
        # With three covariates (x_covar)
        pc = partial_corr(data=df, x="x", y="y", x_covar=["cv1", "cv2", "cv3"])
        assert round(pc.at["pearson", "r"], 7) == 0.4631883
        assert round(pc.at["pearson", "p_val"], 8) == 0.01496857

        # Method == "spearman"
        pc = partial_corr(data=df, x="x", y="y", y_covar=["cv1", "cv2", "cv3"], method="spearman")
        assert round(pc.at["spearman", "r"], 7) == 0.4597143
        assert round(pc.at["spearman", "p_val"], 8) == 0.01584262

        #######################################################################
        # ERROR
        #######################################################################
        with pytest.raises(TypeError):
            # TypeError: partial_corr() got an unexpected keyword argument 'tail'
            partial_corr(data=df, x="x", y="y", covar="cv1", tail="error")
        with pytest.raises(ValueError):
            partial_corr(data=df, x="x", y="y", covar="cv2", x_covar="cv1")
        with pytest.raises(ValueError):
            partial_corr(data=df, x="x", y="y", x_covar="cv2", y_covar="cv1")
        with pytest.raises(AssertionError) as error_info:
            partial_corr(data=df, x="cv1", y="y", covar=["cv1", "cv2"])
        assert str(error_info.value) == "x and covar must be independent"

        # Issue #387: semi-partial correlation with x_covar/y_covar=None should not raise
        pc = partial_corr(data=df, x="x", y="y", x_covar=["cv1", "cv2", "cv3"])
        assert pc.at["pearson", "r"] is not None

        # Issue #375: covariate numerically identical to x or y raises ValueError
        df_375 = df.copy()
        df_375["z"] = df["y"].values
        with pytest.raises(ValueError, match="numerically identical to y"):
            partial_corr(data=df_375, x="x", y="y", covar="z")
        df_375["z"] = df["x"].values
        with pytest.raises(ValueError, match="numerically identical to x"):
            partial_corr(data=df_375, x="x", y="y", covar="z")

        # Issue #435: rank-deficient covariance matrix (perfect multicollinearity) warns
        df_435 = df.copy()
        df_435["cv4"] = df["cv1"] + df["cv2"]  # Perfect linear combination
        with pytest.warns(UserWarning, match="rank-deficient"):
            partial_corr(data=df_435, x="x", y="y", covar=["cv1", "cv2", "cv4"])

        # Issues #411, #509: numerical stability when variables differ by many orders of magnitude
        rng = np.random.default_rng(42)
        n = 22
        covar_1 = rng.standard_normal(n)
        covar_2_normal = rng.standard_normal(n)
        covar_2_large = covar_2_normal * 1e4
        x = rng.standard_normal(n) * 1e-4
        y = covar_1 + rng.standard_normal(n)
        df_normal = pd.DataFrame({"x": x, "y": y, "covar_1": covar_1, "covar_2": covar_2_normal})
        df_large = pd.DataFrame({"x": x, "y": y, "covar_1": covar_1, "covar_2": covar_2_large})
        pc_normal = partial_corr(data=df_normal, x="x", y="y", covar=["covar_1", "covar_2"])
        pc_large = partial_corr(data=df_large, x="x", y="y", covar=["covar_1", "covar_2"])
        assert np.isclose(pc_normal.at["pearson", "r"], pc_large.at["pearson", "r"], atol=1e-6)
        assert np.isclose(
            pc_normal.at["pearson", "p_val"], pc_large.at["pearson", "p_val"], atol=1e-6
        )

        # A constant variable (zero variance) returns NaN instead of failing
        df_const = df_normal.assign(covar_1=1.0)
        for method in ["pearson", "spearman"]:
            for kwargs in [{"x": "covar_1", "covar": "covar_2"}, {"x": "x", "covar": "covar_1"}]:
                stats = partial_corr(data=df_const, y="y", method=method, **kwargs)
                assert stats.at[method, "n"] == n
                assert np.isnan(stats.at[method, "r"])

    def test_rmcorr(self):
        """Test function rm_corr"""
        df = read_dataset("rm_corr")
        # Test again rmcorr R package.
        stats = rm_corr(data=df, x="pH", y="PacO2", subject="Subject").round(3)
        assert stats.at["rm_corr", "r"] == -0.507
        assert stats.at["rm_corr", "dof"] == 38
        assert np.allclose(np.round(stats.at["rm_corr", "CI95"], 2), [-0.71, -0.23])
        assert stats.at["rm_corr", "pval"] == 0.001
        # Test with less than 3 subjects (same behavior as R package)
        with pytest.raises(ValueError):
            rm_corr(data=df[df["Subject"].isin([1, 2])], x="pH", y="PacO2", subject="Subject")

    def test_distance_corr(self):
        """Test function distance_corr
        We compare against the energy R package
        """
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 9, 4, 4]
        dcor1 = distance_corr(a, b, n_boot=None)
        dcor, pval = distance_corr(a, b, seed=9)
        assert dcor1 == dcor
        assert np.round(dcor, 7) == 0.7626762
        assert 0.25 < pval < 0.40
        _, pval_low = distance_corr(a, b, seed=9, alternative="less")
        assert pval < pval_low
        # With 2D arrays
        np.random.seed(123)
        a = np.random.random((10, 10))
        b = np.random.random((10, 10))
        dcor, pval = distance_corr(a, b, n_boot=500, seed=9)
        assert np.round(dcor, 5) == 0.87996
        assert 0.20 < pval < 0.30

        with pytest.raises(ValueError):
            a[2, 4] = np.nan
            distance_corr(a, b)

    def test_rcorr(self):
        """Test function rcorr.

        The multiple-comparison family must contain only the n * (n - 1) / 2 unique pairs (strict
        upper triangle), not the diagonal / lower-triangle placeholders (GH #521). Adjusted
        p-values are compared against statsmodels.
        """
        from itertools import product

        from scipy.stats import pearsonr, spearmanr
        from statsmodels.stats.multitest import multipletests

        from pingouin.correlation import rcorr

        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(80, 5)))
        # Inject real correlations so that the adjusted p-values do not all clip to 1
        frame[1] += 0.4 * frame[0]
        frame[3] -= 0.5 * frame[2]
        frame_na = frame.copy()
        frame_na.iloc[:5, 0] = np.nan
        frame_na.iloc[10:15, 2] = np.nan
        i, j = np.triu_indices(frame.shape[1], k=1)
        # Pingouin -> statsmodels method names
        padjusts = {
            None: None,
            "bonf": "bonferroni",
            "sidak": "sidak",
            "holm": "holm",
            "fdr_bh": "fdr_bh",
            "fdr_by": "fdr_by",
        }
        pval_stars = {0.001: "***", 0.01: "**", 0.05: "*"}

        def to_stars(p):
            for key, value in pval_stars.items():
                if p < key:
                    return value
            return ""

        configs = product(["pearson", "spearman"], padjusts.items(), [frame, frame_na])
        for method, (padjust, sm_method), data in configs:
            corrfunc = pearsonr if method == "pearson" else spearmanr
            raw = []
            for a, b in zip(i, j):
                pair = data.iloc[:, [a, b]].dropna()
                raw.append(corrfunc(pair.iloc[:, 0], pair.iloc[:, 1])[1])
            raw = np.asarray(raw)
            expected = raw if padjust is None else multipletests(raw, method=sm_method)[1]
            # Numeric p-values on the upper triangle
            actual = rcorr(data, method=method, padjust=padjust, stars=False, decimals=12)
            actual = actual.to_numpy()[i, j].astype(float)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            # Stars: the fixture must contain both significant and non-significant pairs so that
            # this is a positive check (an all-NaN or all-1 regression would fail it)
            expected_stars = [to_stars(p) for p in expected]
            assert "" in expected_stars and "***" in expected_stars
            actual_stars = rcorr(data, method=method, padjust=padjust).to_numpy()[i, j]
            assert actual_stars.tolist() == expected_stars

        # Empty test family (constant column -> the only p-value is NaN): no correction method
        # may flag the pair as significant (Sidak used to return 0 = 1 - (1 - nan) ** 0)
        const = pd.DataFrame({"a": rng.normal(size=20), "b": np.ones(20)})
        for padjust in padjusts:
            assert rcorr(const, padjust=padjust).at["a", "b"] == ""
            assert rcorr(const, padjust=padjust, stars=False).at["a", "b"] == "nan"
