import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from scipy.stats import pearsonr, pointbiserialr

from pingouin.effsize import compute_esci, compute_effsize, compute_effsize_from_t, compute_bootci
from pingouin.effsize import convert_effsize as cef

# Dataset
df = pd.DataFrame(
    {
        "Group": ["A", "A", "B", "B"],
        "Time": ["Mon", "Thur", "Mon", "Thur"],
        "Values": [1.52, 5.8, 8.2, 3.4],
    }
)

np.random.seed(42)
x = np.random.normal(2, 1, 20)
y = np.random.normal(2.5, 1, 20)
nx, ny = len(x), len(y)


class TestEffsize(TestCase):
    """Test effsize.py."""

    def test_compute_esci(self):
        """Test function compute_esci.

        Note that since Pingouin v0.3.5, CIs around a Cohen d are calculated
        using a T (and not Z) distribution. This is the same behavior as the
        cohen.d function of the effsize R package.

        However, note that the cohen.d function does not use the Cohen d-avg
        for paired samples, and therefore we cannot directly compare the CIs
        for paired samples. Similarly, R uses a slightly different formula to
        estimate the SE of one-sample cohen D.
        """
        # Pearson correlation
        # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/cor.test.R
        ci = compute_esci(stat=0.5543563, nx=6, eftype="r", decimals=6)
        assert np.allclose(ci, [-0.4675554, 0.9420809])
        # Alternative == "greater"
        ci = compute_esci(stat=0.8, nx=20, eftype="r", alternative="greater", decimals=6)
        assert np.allclose(ci, [0.6041625, 1])
        ci = compute_esci(stat=-0.2, nx=30, eftype="r", alternative="greater", decimals=6)
        assert np.allclose(ci, [-0.4771478, 1])
        # Alternative == "less"
        ci = compute_esci(stat=-0.8, nx=20, eftype="r", alternative="less", decimals=6)
        assert np.allclose(ci, [-1, -0.6041625])
        ci = compute_esci(stat=0.2, nx=30, eftype="r", alternative="less", decimals=6)
        assert np.allclose(ci, [-1, 0.4771478])

        # Cohen d
        # .. One sample and paired
        # Cannot compare to R because cohen.d uses different formulas for
        # Cohen d and SE.
        d = compute_effsize(np.r_[x, y], y=0)
        assert round(d, 6) == 2.086694  # Same as cohen.d
        ci = compute_esci(d, nx + ny, 1, decimals=6)
        d = compute_effsize(x, y, paired=True)
        ci = compute_esci(d, nx, ny, paired=True, decimals=6)
        # .. Independent (compare with cohen.d function)
        d = compute_effsize(x, y)
        ci = compute_esci(d, nx, ny, decimals=6)
        np.testing.assert_equal(ci, [-1.067645, 0.226762])
        # Same but with different n
        d = compute_effsize(x, y[:-5])
        ci = compute_esci(d, nx, len(y[:-5]), decimals=8)
        np.testing.assert_equal(ci, [-1.33603010, 0.08662825])

    def test_compute_boot_esci(self):
        """Test function compute_bootci

        Compare with Matlab bootci function

        See also scipy.stats.bootstrap
        """
        # This is the `lawdata` dataset in Matlab
        # >>> load lawdata
        # >>> x_m = gpa;
        # >>> y_m = lsat;
        x_m = [
            3.39,
            3.3,
            2.81,
            3.03,
            3.44,
            3.07,
            3.0,
            3.43,
            3.36,
            3.13,
            3.12,
            2.74,
            2.76,
            2.88,
            2.96,
        ]
        y_m = [576, 635, 558, 578, 666, 580, 555, 661, 651, 605, 653, 575, 545, 572, 594]
        # 1. bootci around a pearson correlation coefficient
        # Matlab: bootci(n_boot, {@corr, x_m, y_m}, 'type', 'norm');
        ci = compute_bootci(x_m, y_m, func="pearson", paired=True, method="norm", seed=123)
        assert ci[0] == 0.52 and ci[1] == 1.04
        ci = compute_bootci(x_m, y_m, func="pearson", paired=True, method="per", seed=123)
        assert ci[0] == 0.46 and ci[1] == 0.96
        ci = compute_bootci(x_m, y_m, func="pearson", paired=True, method="cper", seed=123)
        assert ci[0] == 0.41 and ci[1] == 0.95
        ci = compute_bootci(x_m, y_m, func="spearman", paired=True)  # Spearman correlation

        # 2. Univariate function: mean
        ci_n = compute_bootci(x_m, func="mean", method="norm", seed=42)
        ci_p = compute_bootci(x_m, func="mean", method="per", seed=42)
        ci_c = compute_bootci(x_m, func="mean", method="cper", seed=42)
        assert ci_n[0] == 2.98 and ci_n[1] == 3.21
        assert ci_p[0] == 2.98 and ci_p[1] == 3.21
        assert ci_c[0] == 2.98 and round(ci_c[1], 1) == 3.2

        # 2.a Univariate function: np.mean
        ci_n = compute_bootci(x_m, func=np.mean, method="norm", seed=42)
        ci_p = compute_bootci(x_m, func=np.mean, method="per", seed=42)
        ci_c = compute_bootci(x_m, func=np.mean, method="cper", seed=42)
        assert ci_n[0] == 2.98 and ci_n[1] == 3.21
        assert ci_p[0] == 2.98 and ci_p[1] == 3.21
        assert ci_c[0] == 2.98 and round(ci_c[1], 1) == 3.2

        # 3. Univariate custom function: skewness
        from scipy.stats import skew

        n_boot = 10000
        ci_n = compute_bootci(x_m, func=skew, method="norm", n_boot=n_boot, decimals=1, seed=42)
        ci_p = compute_bootci(x_m, func=skew, method="per", n_boot=n_boot, decimals=1, seed=42)
        ci_c = compute_bootci(x_m, func=skew, method="cper", n_boot=n_boot, decimals=1, seed=42)
        assert ci_n[0] == -0.7 and ci_n[1] == 0.8
        assert ci_p[0] == -0.7 and ci_p[1] == 0.8
        assert ci_c[0] == -0.7 and ci_c[1] == 0.8

        # 4. Bivariate custom function: paired T-test
        from scipy.stats import ttest_rel

        ci_n = compute_bootci(
            x_m,
            y_m,
            func=lambda x, y: ttest_rel(x, y)[0],
            method="norm",
            paired=True,
            n_boot=n_boot,
            decimals=0,
            seed=42,
        )
        ci_p = compute_bootci(
            x_m,
            y_m,
            func=lambda x, y: ttest_rel(x, y)[0],
            method="per",
            paired=True,
            n_boot=n_boot,
            decimals=0,
            seed=42,
        )
        ci_c = compute_bootci(
            x_m,
            y_m,
            func=lambda x, y: ttest_rel(x, y)[0],
            method="cper",
            paired=True,
            n_boot=n_boot,
            decimals=3,
            seed=42,
        )
        assert ci_n[0] == -70 and ci_n[1] == -34
        assert ci_p[0] == -79 and ci_p[1] == -48
        assert round(ci_c[0]) == -69 and round(ci_c[1]) == -46

        # Make sure that we use different results when using paired=False, because resampling
        # is then done separately for x and y.
        ci_c_unpaired = compute_bootci(
            x_m,
            y_m,
            func=lambda x, y: ttest_rel(x, y)[0],
            method="cper",
            paired=False,
            n_boot=n_boot,
            decimals=3,
            seed=42,
        )
        assert ci_c[0] != ci_c_unpaired[0]
        assert ci_c[1] != ci_c_unpaired[1]

        # 5. Test all combinations
        from itertools import product

        methods = ["norm", "per", "cper"]
        funcs = ["cohen", "hedges"]
        paired = [True, False]
        pr = list(product(methods, funcs, paired))
        for m, f, p in pr:
            compute_bootci(x, y, func=f, method=m, seed=123, n_boot=100)

        # Now the univariate functions
        funcs = ["mean", "std", "var"]
        for m, f in list(product(methods, funcs)):
            compute_bootci(x, func=f, method=m, seed=123, n_boot=100)

        # Using a custom function
        _, bdist = compute_bootci(
            x,
            y,
            func=lambda x, y: np.sum(np.exp(x) / np.exp(y)),
            n_boot=10000,
            decimals=4,
            confidence=0.68,
            seed=None,
            return_dist=True,
        )
        assert bdist.size == 10000

        # ERRORS
        with pytest.raises(ValueError):
            compute_bootci(x, y, func="wrong")

        with pytest.raises(AssertionError):
            compute_bootci(x, y, func="pearson", paired=False)

    def test_convert_effsize(self):
        """Test function convert_effsize.

        Compare to https://www.psychometrica.de/effect_size.html
        """
        # Cohen d
        d = 0.40
        assert cef(d, "cohen", "none") == d
        assert round(cef(d, "cohen", "pointbiserialr"), 4) == 0.1961
        cef(d, "cohen", "pointbiserialr", nx=10, ny=12)  # When nx and ny are specified
        assert np.allclose(cef(1.002549, "cohen", "pointbiserialr"), 0.4481248)  # R
        assert round(cef(d, "cohen", "eta-square"), 4) == 0.0385
        assert round(cef(d, "cohen", "odds-ratio"), 4) == 2.0658
        cef(d, "cohen", "hedges", nx=10, ny=10)
        cef(d, "cohen", "pointbiserialr")
        cef(d, "cohen", "hedges")

        # Point-biserial correlation
        rpb = 0.65
        assert cef(rpb, "pointbiserialr", "none") == rpb
        assert round(cef(rpb, "pointbiserialr", "cohen"), 4) == 1.7107
        assert np.allclose(cef(0.4481248, "pointbiserialr", "cohen"), 1.002549)
        assert round(cef(rpb, "pointbiserialr", "eta-square"), 4) == 0.4225
        assert round(cef(rpb, "pointbiserialr", "odds-ratio"), 4) == 22.2606
        # Using actual values
        np.random.seed(42)
        x1, y1 = np.random.multivariate_normal(mean=[1, 2], cov=[[1, 0.5], [0.5, 1]], size=100).T
        xy1 = np.hstack((x1, y1))
        xy1_bool = np.repeat([0, 1], 100)
        # Let's calculate the ground-truth point-biserial correlation
        r_biserial = pearsonr(xy1_bool, xy1)[0]  # 0.50247
        assert np.isclose(r_biserial, pointbiserialr(xy1_bool, xy1)[0])
        # Now the Cohen's d
        d = abs(compute_effsize(x1, y1, paired=True, eftype="cohen"))  # 1.15651
        # And now we can convert point-biserial r <--> d
        r_convert = cef(abs(d), "cohen", "pointbiserialr", nx=100, ny=100)  # 0.50247
        assert np.isclose(r_convert, r_biserial)
        d_convert = cef(r_biserial, "pointbiserialr", "cohen", nx=100, ny=100)  # 1.162
        assert abs(d - d_convert) < 0.1

        # Error
        with pytest.raises(ValueError):
            # DEPRECATED - https://github.com/raphaelvallat/pingouin/issues/302
            cef(d, "cohen", "r")
        with pytest.raises(ValueError):
            cef(d, "coucou", "hibou")
        with pytest.raises(ValueError):
            cef(d, "AUC", "eta-square")

    def test_compute_effsize(self):
        """Test function compute_effsize"""
        compute_effsize(x=x, y=y, eftype="cohen", paired=False)
        compute_effsize(x=x, y=y, eftype="AUC", paired=True)
        compute_effsize(x=x, y=y, eftype="r", paired=False)
        compute_effsize(x=x, y=y, eftype="odds-ratio", paired=False)
        compute_effsize(x=x, y=y, eftype="eta-square", paired=False)
        compute_effsize(x=x, y=y, eftype="cles", paired=False)
        compute_effsize(x=x, y=y, eftype="pointbiserialr", paired=False)
        compute_effsize(x=x, y=y, eftype="none", paired=False)
        # Unequal variances
        z = np.random.normal(2.5, 3, 30)
        compute_effsize(x=x, y=z, eftype="cohen")
        # Wrong effect size type
        with pytest.raises(ValueError):
            compute_effsize(x=x, y=y, eftype="wrong")
        # Unequal sample size with paired == True
        z = np.random.normal(2.5, 3, 25)
        compute_effsize(x=x, y=z, paired=True)
        # Compare with the effsize R package
        a = [3.2, 6.4, 1.8, 2.4, 5.8, 6.5]
        b = [2.4, 3.2, 3.2, 1.4, 2.8, 3.5]
        d = compute_effsize(x=a, y=b, eftype="cohen", paired=False)
        assert np.isclose(d, 1.002549)
        # Note that ci are different than from R because we use a normal and
        # not a T distribution to estimate the CI.
        # Also, for paired samples, effsize does not return the Cohen d-avg.
        # ci = compute_esci(ef=d, nx=na, ny=nb)
        # assert ci[0] == -.2
        # assert ci[1] == 2.2
        # With Hedges correction
        g = compute_effsize(x=a, y=b, eftype="hedges", paired=False)
        assert np.isclose(g, 0.9254296)
        # CLES
        # Compare to
        # https://janhove.github.io/reporting/2016/11/16/common-language-effect-sizes
        x2 = [20, 22, 19, 20, 22, 18, 24, 20, 19, 24, 26, 13]
        y2 = [38, 37, 33, 29, 14, 12, 20, 22, 17, 25, 26, 16]
        cl = compute_effsize(x=x2, y=y2, eftype="cles")
        assert np.isclose(cl, 0.3958333)
        assert np.isclose((1 - cl), compute_effsize(x=y2, y=x2, eftype="cles"))

    def test_compute_effsize_from_t(self):
        """Test function compute_effsize_from_t"""
        tval, nx, ny = 2.90, 35, 25
        compute_effsize_from_t(tval, nx=nx, ny=ny, eftype="hedges")
        tval, N = 2.90, 60
        compute_effsize_from_t(tval, N=N, eftype="cohen")
        # Wrong desired eftype
        with pytest.raises(ValueError):
            compute_effsize_from_t(tval, nx=x, ny=y, eftype="wrong")
        # T is not a float
        with pytest.raises(ValueError):
            compute_effsize_from_t([1, 2, 3], nx=nx, ny=ny)
        # No sample size info
        with pytest.raises(ValueError):
            compute_effsize_from_t(tval)
        # Compare with Lakens spreadsheet: https://osf.io/vbdah/
        assert np.isclose(compute_effsize_from_t(1.1, N=31), 0.395131664)
        assert np.isclose(compute_effsize_from_t(1.74, nx=6, ny=6), 1.00458946)
        assert np.isclose(compute_effsize_from_t(2.5, nx=10, ny=14), 1.0350983)
