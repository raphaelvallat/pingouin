# Author: Antoine Weill--Duflos <antoine@weill-duflos.fr>
# Date July 2019
import numpy as np
from unittest import TestCase
from pingouin.equivalence import tost


class TestEquivalence(TestCase):
    """Test equivalence.py."""

    def test_tost(self):
        """Test function tost.
        Compare to R package equivalence (function tost)
        """
        np.random.seed(1234)
        a = np.random.normal(scale=1., size=600)
        assert np.less(tost(a, a, bound=1).at['TOST', 'p-val'], 0.05)
        assert np.greater(tost(a, a + 25, bound=1).at['TOST', 'p-val'], 0.5)
        a = [4, 7, 8, 6, 3, 2]
        b = [6, 8, 7, 10, 11, 9]
        # Check all arguments
        tost(a, b)
        tost(b, a)
        tost(a, b, paired=True)
        tost(b, a, paired=True)
        tost(a, b, parametric=False)
        tost(b, a, parametric=False)
        tost(a, b, paired=True, parametric=False)
        tost(b, a, paired=True, parametric=False)

        # Compare with R
        # R: tost(a, b, epsilon = 1, var.equal = TRUE)
        assert np.isclose(tost(a, b, bound=1).at['TOST', 'p-val'], 0.9650974)
        # R: tost(a, b)
        assert np.isclose(tost(a, b, bound=1,
                          correction=True).at['TOST', 'p-val'], 0.9643479)
        assert np.isclose(tost(a, b, bound=10).at['TOST', 'p-val'], 0.00017933)
        assert np.isclose(tost(a, b, bound=1, correction=True,
                          paired=True).at['TOST', 'p-val'], 0.9293826)
        assert np.isclose(tost(a, b, bound=2, correction=False,
                          paired=True).at['TOST', 'p-val'], 0.8286101)
