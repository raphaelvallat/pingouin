import pandas as pd
import numpy as np
import pytest

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.effsize import *

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
        compute_esci(x=x, y=y, alpha=.95, method='bootstrap', n_boot=2000,
                     eftype='hedges', return_dist=True)
        compute_esci(x=x, y=y, alpha=.99, method='bootstrap', n_boot=1000,
                     eftype='r', return_dist=False)
        compute_esci(ef=.4, nx=len(x), ny=len(y), alpha=.95)
        # Wrong input
        with pytest.raises(ValueError):
            compute_esci(x=x, y=y, alpha=.95, method='bootstrap', n_boot=2000,
                         eftype='wrong', return_dist=True)
            compute_esci()


    def test_convert_effsize(self):
        """Test function convert_effsize"""
        d = .40
        r = .65
        eta = convert_effsize(d, 'cohen', 'eta-square')
        g = convert_effsize(d, 'cohen', 'hedges', nx=10, ny=10)
        d = convert_effsize(r, 'r', 'cohen')
        r = convert_effsize(d, 'cohen', 'r')
        with pytest.raises(ValueError):
            r = convert_effsize(d, 'coucou', 'hibou')
            r = convert_effsize(d, 'AUC', 'eta-square')
            g = convert_effsize(d, 'cohen', 'hedges-square')



    def test_compute_effsize(self):
        """Test function compute_effsize"""
        d = compute_effsize(x=x, y=y, eftype='cohen', paired=False)
        auc = compute_effsize(x=x, y=y, eftype='AUC', paired=True)
        r = compute_effsize(x=x, y=y, eftype='r', paired=False)
        glass = compute_effsize(x=x, y=y, eftype='glass', paired=False)
        od = compute_effsize(x=x, y=y, eftype='odds-ratio', paired=False)
        eta = compute_effsize(x=x, y=y, eftype='eta-square', paired=False)
        none = compute_effsize(x=x, y=y, eftype='none', paired=False)
        df = pd.DataFrame({'dv': np.r_[x, y],
                           'Group': np.repeat(['Pre', 'Post'], 30)})
        g = compute_effsize(dv='dv', group='Group', data=df,
                             paired=True, eftype='hedges')


    def test_compute_effsize_from_T(self):
        """Test function compute_effsize_from_T"""
        T, nx, ny = 2.90, 35, 25
        d = compute_effsize_from_T(T, nx=nx, ny=ny, eftype='hedges')
        T, N = 2.90, 60
        d = compute_effsize_from_T(T, N=N, eftype='cohen')
