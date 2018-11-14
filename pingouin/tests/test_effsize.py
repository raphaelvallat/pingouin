import pandas as pd
import numpy as np
import pytest

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.effsize import (compute_esci, convert_effsize, compute_effsize,
                              compute_effsize_from_t, compute_boot_esci)

# Dataset
df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'],
                   'Time': ['Mon', 'Thur', 'Mon', 'Thur'],
                   'Values': [1.52, 5.8, 8.2, 3.4]})

x = np.random.normal(2, 1, 30)
y = np.random.normal(2.5, 1, 30)


class TestEffsize(_TestPingouin):
    """Test effsize.py."""

    def test_compute_esci(self):
        """Test function compute_esci"""
        compute_esci(stat=.6, nx=30, ny=30, eftype='r')
        compute_esci(stat=.4, nx=len(x), ny=len(y), confidence=.99, decimals=4)
        compute_esci(stat=.6, nx=30, ny=30, eftype='cohen')
        # Compare with R
        r, nx, ny = 0.5543563, 6, 6
        ci = compute_esci(stat=r, nx=nx, ny=ny, eftype='r')
        assert np.allclose(ci, [-0.47, 0.94])

    def test_compute_boot_esci(self):
        """Test function compute_boot_esci"""
        from itertools import product
        methods = ['norm', 'per', 'cper']
        funcs = ['spearman', 'pearson', 'cohen', 'hedges']
        paired = [True, False]
        pr = list(product(methods, funcs, paired))
        for m, f, p in pr:
            compute_boot_esci(x, y, func=f, method=m, seed=123)
        with pytest.raises(ValueError):
            compute_boot_esci(x, y, func='wrong')
        # Using a custom function
        compute_boot_esci(x, y,
                          func=lambda x, y: np.sum(np.exp(x) / np.exp(y)),
                          n_boot=10000, decimals=4, confidence=.68, seed=None)

    def test_convert_effsize(self):
        """Test function convert_effsize"""
        d = .40
        r = .65
        convert_effsize(d, 'cohen', 'eta-square')
        convert_effsize(d, 'cohen', 'hedges', nx=10, ny=10)
        convert_effsize(d, 'cohen', 'r', nx=10, ny=10)
        convert_effsize(r, 'r', 'cohen')
        convert_effsize(d, 'cohen', 'r')
        convert_effsize(d, 'cohen', 'hedges')
        convert_effsize(d, 'cohen', 'glass')
        convert_effsize(d, 'cohen', 'none')
        with pytest.raises(ValueError):
            convert_effsize(d, 'coucou', 'hibou')
        with pytest.raises(ValueError):
            convert_effsize(d, 'AUC', 'eta-square')
        # Compare with R
        assert np.allclose(convert_effsize(1.002549, 'cohen', 'r'), 0.4481248)
        assert np.allclose(convert_effsize(0.4481248, 'r', 'cohen'), 1.002549)

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
