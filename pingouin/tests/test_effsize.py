import pandas as pd
import numpy as np
import pytest

from unittest import TestCase
from pingouin.effsize import (compute_esci, compute_effsize,
                              compute_effsize_from_t, compute_bootci)
from pingouin.effsize import convert_effsize as cef

# Dataset
df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'],
                   'Time': ['Mon', 'Thur', 'Mon', 'Thur'],
                   'Values': [1.52, 5.8, 8.2, 3.4]})

x = np.random.normal(2, 1, 30)
y = np.random.normal(2.5, 1, 30)


class TestEffsize(TestCase):
    """Test effsize.py."""

    def test_compute_esci(self):
        """Test function compute_esci"""
        compute_esci(stat=.6, nx=30, eftype='r')
        compute_esci(stat=.4, nx=len(x), ny=len(y), confidence=.99, decimals=4)
        compute_esci(stat=.6, nx=30, ny=30, eftype='cohen')
        # Compare with R
        r, nx = 0.5543563, 6
        ci = compute_esci(stat=r, nx=nx, eftype='r')
        assert np.allclose(ci, [-0.47, 0.94])
        # One sample / paired T-test
        ci = compute_esci(-0.57932, nx=40, ny=1)
        ci_p = compute_esci(-0.57932, nx=40, ny=1, paired=True)
        assert np.allclose(ci, ci_p)
        assert np.allclose(ci, [-0.91, -0.24])

    def test_compute_boot_esci(self):
        """Test function compute_bootci
        Compare with Matlab bootci function
        """
        # This is the `lawdata` dataset in Matlab
        # >>> load lawdata
        # >>> x_m = gpa;
        # >>> y_m = lsat;
        x_m = [3.39, 3.3, 2.81, 3.03, 3.44, 3.07, 3.0, 3.43, 3.36, 3.13,
               3.12, 2.74, 2.76, 2.88, 2.96]
        y_m = [576, 635, 558, 578, 666, 580, 555, 661, 651, 605, 653, 575,
               545, 572, 594]
        # 1. bootci around a pearson correlation coefficient
        # Matlab: bootci(n_boot, {@corr, x_m, y_m}, 'type', 'norm');
        ci = compute_bootci(x_m, y_m, method='norm', seed=123)
        assert ci[0] == 0.52 and ci[1] == 1.05
        ci = compute_bootci(x_m, y_m, method='per', seed=123)
        assert ci[0] == 0.45 and ci[1] == 0.96
        ci = compute_bootci(x_m, y_m, method='cper', seed=123)
        assert ci[0] == 0.39 and ci[1] == 0.95
        # 2. Univariate function: mean
        ci_n = compute_bootci(x_m, func='mean', method='norm', seed=42)
        ci_p = compute_bootci(x_m, func='mean', method='per', seed=42)
        ci_c = compute_bootci(x_m, func='mean', method='cper', seed=42)
        assert ci_n[0] == 2.98 and ci_n[1] == 3.21
        assert ci_p[0] == 2.98 and ci_p[1] == 3.21
        assert ci_c[0] == 2.98 and round(ci_c[1], 1) == 3.2

        # 3. Univariate custom function: skewness
        from scipy.stats import skew
        n_boot = 10000
        ci_n = compute_bootci(x_m, func=skew, method='norm', n_boot=n_boot,
                              decimals=1, seed=42)
        ci_p = compute_bootci(x_m, func=skew, method='per', n_boot=n_boot,
                              decimals=1, seed=42)
        ci_c = compute_bootci(x_m, func=skew, method='cper', n_boot=n_boot,
                              decimals=1, seed=42)
        assert ci_n[0] == -0.7 and ci_n[1] == 0.8
        assert ci_p[0] == -0.7 and ci_p[1] == 0.8
        assert ci_c[0] == -0.7 and ci_c[1] == 0.8

        # 4. Bivariate custom function: paired T-test
        from scipy.stats import ttest_rel
        ci_n = compute_bootci(x_m, y_m, func=lambda x, y: ttest_rel(x, y)[0],
                              method='norm', n_boot=n_boot, decimals=0,
                              seed=42)
        ci_p = compute_bootci(x_m, y_m, func=lambda x, y: ttest_rel(x, y)[0],
                              method='per', n_boot=n_boot, decimals=0,
                              seed=42)
        ci_c = compute_bootci(x_m, y_m, func=lambda x, y: ttest_rel(x, y)[0],
                              method='cper', n_boot=n_boot, decimals=0,
                              seed=42)
        assert ci_n[0] == -69 and ci_n[1] == -35
        assert ci_p[0] == -79 and ci_p[1] == -48
        assert ci_c[0] == -68 and ci_c[1] == -47

        # 5. Test all combinations
        from itertools import product
        methods = ['norm', 'per', 'cper']
        funcs = ['spearman', 'pearson', 'cohen', 'hedges']
        paired = [True, False]
        pr = list(product(methods, funcs, paired))
        for m, f, p in pr:
            compute_bootci(x, y, func=f, method=m, seed=123, n_boot=100)

        # Now the univariate function
        funcs = ['mean', 'std', 'var']
        for m, f in list(product(methods, funcs)):
            compute_bootci(x, func=f, method=m, seed=123, n_boot=100)

        with pytest.raises(ValueError):
            compute_bootci(x, y, func='wrong')
        # Using a custom function
        compute_bootci(x, y,
                       func=lambda x, y: np.sum(np.exp(x) / np.exp(y)),
                       n_boot=10000, decimals=4, confidence=.68, seed=None)
        # Get the bootstrapped distribution
        _, bdist = compute_bootci(x, y, return_dist=True, n_boot=1500)
        assert bdist.size == 1500

    def test_convert_effsize(self):
        """Test function convert_effsize.
        Compare to https://www.psychometrica.de/effect_size.html"""
        # Cohen d
        d = .40
        assert cef(d, 'cohen', 'none') == d
        assert round(cef(d, 'cohen', 'r'), 4) == 0.1961
        cef(d, 'cohen', 'r', nx=10, ny=12)  # When nx and ny are specified
        assert np.allclose(cef(1.002549, 'cohen', 'r'), 0.4481248)  # R
        assert round(cef(d, 'cohen', 'eta-square'), 4) == 0.0385
        assert round(cef(d, 'cohen', 'odds-ratio'), 4) == 2.0658
        cef(d, 'cohen', 'hedges', nx=10, ny=10)
        cef(d, 'cohen', 'r')
        cef(d, 'cohen', 'hedges')
        cef(d, 'cohen', 'glass')

        # Correlation coefficient
        r = .65
        assert cef(r, 'r', 'none') == r
        assert round(cef(r, 'r', 'cohen'), 4) == 1.7107
        assert np.allclose(cef(0.4481248, 'r', 'cohen'), 1.002549)
        assert round(cef(r, 'r', 'eta-square'), 4) == 0.4225
        assert round(cef(r, 'r', 'odds-ratio'), 4) == 22.2606

        # Error
        with pytest.raises(ValueError):
            cef(d, 'coucou', 'hibou')
        with pytest.raises(ValueError):
            cef(d, 'AUC', 'eta-square')

    def test_compute_effsize(self):
        """Test function compute_effsize"""
        compute_effsize(x=x, y=y, eftype='cohen', paired=False)
        compute_effsize(x=x, y=y, eftype='AUC', paired=True)
        compute_effsize(x=x, y=y, eftype='r', paired=False)
        compute_effsize(x=x, y=y, eftype='glass', paired=False)
        compute_effsize(x=x, y=y, eftype='odds-ratio', paired=False)
        compute_effsize(x=x, y=y, eftype='eta-square', paired=False)
        compute_effsize(x=x, y=y, eftype='none', paired=False)
        # Unequal variances
        z = np.random.normal(2.5, 3, 30)
        compute_effsize(x=x, y=z, eftype='cohen')
        # Wrong effect size type
        with pytest.raises(ValueError):
            compute_effsize(x=x, y=y, eftype='wrong')
        # Unequal sample size with paired == True
        z = np.random.normal(2.5, 3, 20)
        compute_effsize(x=x, y=z, paired=True)
        # Compare with the effsize R package
        a = [3.2, 6.4, 1.8, 2.4, 5.8, 6.5]
        b = [2.4, 3.2, 3.2, 1.4, 2.8, 3.5]
        # na = len(a)
        # nb = len(b)
        d = compute_effsize(x=a, y=b, eftype='cohen', paired=False)
        assert np.isclose(d, 1.002549)
        # Note that ci are different than from R because we use a normal and
        # not a T distribution to estimate the CI
        # ci = compute_esci(ef=d, nx=na, ny=nb)
        # assert ci[0] == -.2
        # assert ci[1] == 2.2
        # With Hedges correction
        g = compute_effsize(x=a, y=b, eftype='hedges', paired=False)
        assert np.isclose(g, 0.9254296)

    def test_compute_effsize_from_t(self):
        """Test function compute_effsize_from_t"""
        tval, nx, ny = 2.90, 35, 25
        compute_effsize_from_t(tval, nx=nx, ny=ny, eftype='hedges')
        tval, N = 2.90, 60
        compute_effsize_from_t(tval, N=N, eftype='cohen')
        # Wrong desired eftype
        with pytest.raises(ValueError):
            compute_effsize_from_t(tval, nx=x, ny=y, eftype='wrong')
        # T is not a float
        with pytest.raises(ValueError):
            compute_effsize_from_t([1, 2, 3], nx=nx, ny=ny)
        # No sample size info
        with pytest.raises(ValueError):
            compute_effsize_from_t(tval)
        # Compare with R
        assert np.allclose(compute_effsize_from_t(1.736465, nx=6, ny=6),
                           1.002549)
