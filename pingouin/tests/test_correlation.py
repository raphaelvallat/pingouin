import pytest
import numpy as np
from unittest import TestCase
from pingouin.correlation import corr, rm_corr, partial_corr, skipped, distance_corr, bicor
from pingouin import read_dataset


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
        assert np.isclose(stats.loc["pearson", "p-val"], 0.3518659)
        assert stats.loc["pearson", "CI95%"][0] == round(-0.1966232, 2)
        assert stats.loc["pearson", "CI95%"][1] == round(0.5043872, 2)
        # - One-sided: greater
        stats = corr(x, y, method="pearson", alternative="greater")
        assert np.isclose(stats.loc["pearson", "r"], 0.1761221)
        assert np.isclose(stats.loc["pearson", "p-val"], 0.175933)
        assert stats.loc["pearson", "CI95%"][0] == round(-0.1376942, 2)
        assert stats.loc["pearson", "CI95%"][1] == 1
        # - One-sided: less
        stats = corr(x, y, method="pearson", alternative="less")
        assert np.isclose(stats.loc["pearson", "r"], 0.1761221)
        assert np.isclose(stats.loc["pearson", "p-val"], 0.824067)
        assert stats.loc["pearson", "CI95%"][0] == -1
        assert stats.loc["pearson", "CI95%"][1] == round(0.4578044, 2)

        # Spearman correlation
        stats = corr(x, y, method="spearman")
        assert np.isclose(stats.loc["spearman", "r"], 0.4740823)
        assert np.isclose(stats.loc["spearman", "p-val"], 0.008129768)
        # CI are calculated using a different formula for Spearman in R
        # assert stats.loc['spearman', 'CI95%'][0] == round(0.1262988, 2)
        # assert stats.loc['spearman', 'CI95%'][1] == round(0.7180799, 2)

        # Kendall correlation
        # R uses a different estimation method than scipy for the p-value
        stats = corr(x, y, method="kendall")
        assert np.isclose(stats.loc["kendall", "r"], 0.3517241)
        # Skipped correlation -- compare with robust corr toolbox
        # https://sourceforge.net/projects/robustcorrtool/
        stats = corr(x, y, method="skipped")
        assert round(stats.loc["skipped", "r"], 4) == 0.5123
        assert stats.loc["skipped", "outliers"] == 2
        sk_sp = corr(x2, y2, method="skipped")
        assert round(sk_sp.loc["skipped", "r"], 4) == 0.5123
        assert sk_sp.loc["skipped", "outliers"] == 2
        # Pearson skipped correlation
        sk_pe = corr(x2, y2, method="skipped", corr_type="pearson")
        assert np.round(sk_pe.loc["skipped", "r"], 4) == 0.5254
        assert sk_pe.loc["skipped", "outliers"] == 2
        assert not sk_sp.equals(sk_pe)
        # Shepherd
        stats = corr(x, y, method="shepherd")
        assert np.isclose(stats.loc["shepherd", "r"], 0.5123153)
        assert np.isclose(stats.loc["shepherd", "p-val"], 0.005316)
        assert stats.loc["shepherd", "outliers"] == 2
        _, _, outliers = skipped(x, y, corr_type="pearson")
        assert outliers.size == x.size
        assert stats.loc["shepherd", "n"] == 30
        # Percbend -- compare with robust corr toolbox
        stats = corr(x, y, method="percbend")
        assert round(stats.loc["percbend", "r"], 4) == 0.4843
        assert np.isclose(stats.loc["percbend", "r"], 0.4842686)
        assert np.isclose(stats.loc["percbend", "p-val"], 0.006693313)
        stats = corr(x2, y2, method="percbend")
        assert round(stats.loc["percbend", "r"], 4) == 0.4843
        stats = corr(x, y, method="percbend", beta=0.5)
        assert round(stats.loc["percbend", "r"], 4) == 0.4848
        # Compare biweight correlation to astropy
        stats = corr(x, y, method="bicor")
        assert np.isclose(stats.loc["bicor", "r"], 0.4951418)
        assert np.isclose(stats.loc["bicor", "p-val"], 0.005403701)
        assert stats.loc["bicor", "CI95%"][0] == round(0.1641553, 2)
        assert stats.loc["bicor", "CI95%"][1] == round(0.7259185, 2)
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
        # When one column is a constant, the correlation is not defined
        # and Pingouin return a DataFrame full of NaN, except for ``n``
        x, y = [1, 1, 1], [1, 2, 3]
        stats = corr(x, y)
        assert stats.at["pearson", "n"]
        assert np.isnan(stats.at["pearson", "r"])
        # Biweight midcorrelation returns NaN when MAD is not defined
        assert np.isnan(bicor(np.array([1, 1, 1, 1, 0, 1]), np.arange(6))[0])

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
        assert round(pc.at["pearson", "p-val"], 9) == 0.001303059
        # With two covariates
        pc = partial_corr(data=df, x="x", y="y", covar=["cv1", "cv2"])
        assert round(pc.at["pearson", "r"], 7) == 0.5344372
        assert round(pc.at["pearson", "p-val"], 9) == 0.003392904
        # With three covariates
        # in R: pcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")])
        pc = partial_corr(data=df, x="x", y="y", covar=["cv1", "cv2", "cv3"])
        assert round(pc.at["pearson", "r"], 7) == 0.4926007
        assert round(pc.at["pearson", "p-val"], 9) == 0.009044164
        # Method == "spearman"
        pc = partial_corr(data=df, x="x", y="y", covar=["cv1", "cv2", "cv3"], method="spearman")
        assert round(pc.at["spearman", "r"], 7) == 0.5209208
        assert round(pc.at["spearman", "p-val"], 9) == 0.005336187

        #######################################################################
        # SEMI-PARTIAL CORRELATION
        #######################################################################
        # With one covariate
        pc = partial_corr(data=df, x="x", y="y", y_covar="cv1")
        assert round(pc.at["pearson", "r"], 7) == 0.5670793
        assert round(pc.at["pearson", "p-val"], 9) == 0.001337718
        # With two covariates
        pc = partial_corr(data=df, x="x", y="y", y_covar=["cv1", "cv2"])
        assert round(pc.at["pearson", "r"], 7) == 0.5097489
        assert round(pc.at["pearson", "p-val"], 9) == 0.005589687
        # With three covariates
        # in R: spcor.test(x=df$x, y=df$y, z=df[, c("cv1", "cv2", "cv3")])
        pc = partial_corr(data=df, x="x", y="y", y_covar=["cv1", "cv2", "cv3"])
        assert round(pc.at["pearson", "r"], 7) == 0.4212351
        assert round(pc.at["pearson", "p-val"], 8) == 0.02865483
        # With three covariates (x_covar)
        pc = partial_corr(data=df, x="x", y="y", x_covar=["cv1", "cv2", "cv3"])
        assert round(pc.at["pearson", "r"], 7) == 0.4631883
        assert round(pc.at["pearson", "p-val"], 8) == 0.01496857

        # Method == "spearman"
        pc = partial_corr(data=df, x="x", y="y", y_covar=["cv1", "cv2", "cv3"], method="spearman")
        assert round(pc.at["spearman", "r"], 7) == 0.4597143
        assert round(pc.at["spearman", "p-val"], 8) == 0.01584262

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

    def test_rmcorr(self):
        """Test function rm_corr"""
        df = read_dataset("rm_corr")
        # Test again rmcorr R package.
        stats = rm_corr(data=df, x="pH", y="PacO2", subject="Subject").round(3)
        assert stats.at["rm_corr", "r"] == -0.507
        assert stats.at["rm_corr", "dof"] == 38
        assert np.allclose(np.round(stats.at["rm_corr", "CI95%"], 2), [-0.71, -0.23])
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
