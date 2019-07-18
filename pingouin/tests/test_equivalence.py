# Author: Antoine Weill--Duflos <antoine@weill-duflos.fr>
# Date July 2019
import numpy as np

from unittest import TestCase
from pingouin.equivalence import (tost)

np.random.seed(1234)
x = np.random.normal(scale=1., size=100)
y = np.random.normal(scale=0.8, size=100)


class TestEquivalence(TestCase):
    """Test equivalence.py."""
    def test_tost(self):
        """Test function tost"""
        h = np.random.normal(scale=0.9, size=95)
        stats = tost(x, y, paired=True)
        a = np.random.normal(scale=1., size=600)
        assert np.less(tost(a, a, bound=1).loc['TOST', 'p-val'], 0.05)
        assert np.greater(tost(a,  list(np.asarray(a) + 25), bound=1).loc['TOST', 'p-val'], 0.5)
        a = [4, 7, 8, 6, 3, 2]
        b = [6, 8, 7, 10, 11, 9]
        assert round(tost(a,b,bound=10).loc['TOST', 'p-val'],3) == 0.000
        assert round(tost(a, b, bound=10,parametric=False).loc['TOST', 'p-val'],3) == 0.003
        assert round(tost(a, b, bound=10,parametric=False,paired=True).loc['TOST', 'p-val'],3) == 0.018

    def test_againstRtost(self):
        """Compare tost to R implementation (package equivalence)"""
        a = [4, 7, 8, 6, 3, 2]
        b = [6, 8, 7, 10, 11, 9]
        assert round(tost(a,b,bound=1).loc['TOST', 'p-val'],3) == 0.965
        assert round(tost(a, b, bound=10).loc['TOST', 'p-val'], 6) == 0.000179
        assert round(tost(a, b, bound=1,correction=True).loc['TOST', 'p-val'], 3) == 0.964
        assert round(tost(a, b, bound=1, correction=True,paired=True).loc['TOST', 'p-val'], 3) == 0.929
        assert round(tost(a, b, bound=2, correction=False, paired=True).loc['TOST', 'p-val'], 3) == 0.829
