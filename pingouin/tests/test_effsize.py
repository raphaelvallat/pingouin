import pandas as pd
import numpy as np
import pytest

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.effsize import (compute_esci, convert_effsize, compute_effsize,
                              compute_effsize_from_t)

# Dataset
df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'],
                   'Time': ['Mon', 'Thur', 'Mon', 'Thur'],
                   'Values': [1.52, 5.8, 8.2, 3.4]})

x = np.random.normal(2, 1, 30)
y = np.random.normal(2.5, 1, 30)
z = np.random.normal(2.5, 3, 30)


class TestEffsize(_TestPingouin):
    """Test effsize.py."""

    def test_compute_esci(self):
        """Test function compute_esci"""
        compute_esci(x=x, y=y, alpha=.95, method='bootstrap', n_boot=2000,
                     eftype='hedges', return_dist=True)
        compute_esci(x=x, y=y, alpha=.99, method='bootstrap', n_boot=1000,
                     eftype='r', return_dist=False)
        compute_esci(ef=.4, nx=len(x), ny=len(y), alpha=.95)
        # Wrong input
        compute_esci(ef=.4, nx=len(x), ny=len(y), method='bootstrap')
        with pytest.raises(ValueError):
            compute_esci(x=x, y=y, alpha=.95, method='bootstrap', n_boot=2000,
                         eftype='wrong', return_dist=True)
        with pytest.raises(ValueError):
            compute_esci()

    def test_convert_effsize(self):
        """Test function convert_effsize"""
        d = .40
        r = .65
        convert_effsize(d, 'cohen', 'eta-square')
        convert_effsize(d, 'cohen', 'hedges', nx=10, ny=10)
        convert_effsize(r, 'r', 'cohen')
        convert_effsize(d, 'cohen', 'r')
        convert_effsize(d, 'cohen', 'hedges')
        convert_effsize(d, 'cohen', 'glass')
        convert_effsize(d, 'cohen', 'none')
        with pytest.raises(ValueError):
            convert_effsize(d, 'coucou', 'hibou')
        with pytest.raises(ValueError):
            convert_effsize(d, 'AUC', 'eta-square')

    def test_compute_effsize(self):
        """Test function compute_effsize"""
        compute_effsize(x=x, y=y, eftype='cohen', paired=False)
        compute_effsize(x=x, y=y, eftype='AUC', paired=True)
        compute_effsize(x=x, y=y, eftype='r', paired=False)
        compute_effsize(x=x, y=y, eftype='glass', paired=False)
        compute_effsize(x=x, y=y, eftype='odds-ratio', paired=False)
        compute_effsize(x=x, y=y, eftype='eta-square', paired=False)
        compute_effsize(x=x, y=y, eftype='none', paired=False)
        df = pd.DataFrame({'dv': np.r_[x, y],
                           'Group': np.repeat(['Pre', 'Post'], 30)})
        compute_effsize(dv='dv', group='Group', data=df,
                        paired=True, eftype='hedges')
        # Unequal variances
        compute_effsize(x=x, y=z, eftype='cohen')
        with pytest.raises(ValueError):
            compute_effsize(x=x, y=y, eftype='wrong')

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
