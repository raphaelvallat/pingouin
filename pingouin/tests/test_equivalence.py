# Author: Antoine Weill--Duflos <antoine@weill-duflos.fr>
# Date July 2019
import numpy as np
from unittest import TestCase
from pingouin.equivalence import tost


class TestEquivalence(TestCase):
    """Test equivalence.py."""

    def test_tost(self):
        """Test function tost.
        Compare to R package equivalence (function `tost`).
        """
        np.random.seed(1234)
        a = np.random.normal(scale=1., size=600)  # a has a mean of 0
        b = a + 25  # b has a mean of ~25

        # Simple safety check
        assert tost(a, a).at['TOST', 'pval'] < 0.05
        assert tost(a, a, paired=True).at['TOST', 'pval'] < 0.05
        assert tost(a, b).at['TOST', 'pval'] > 0.5
        assert tost(a, b, paired=True).at['TOST', 'pval'] > 0.5

        # Check all arguments with good data
        a = np.array([4, 7, 8, 6, 3, 2])
        b = np.array([6, 8, 7, 10, 11, 9])
        tost(a, b).equals(tost(b, a))
        tost(a, b).equals(tost(-1 * a, -1 * b))
        tost(a, b, paired=True).equals(tost(b, a, paired=True))

        # Compare with R
        # R: tost(a, b, epsilon = 1, var.equal = TRUE)
        assert tost(a, b).at['TOST', 'dof'] == 10
        assert np.isclose(tost(a, b).at['TOST', 'pval'], 0.9650974)

        # R: tost(a, b)
        assert tost(a, b, correction=True).at['TOST', 'dof'] == 9.49
        assert np.isclose(tost(a, b, bound=1,
                          correction=True).at['TOST', 'pval'], 0.9643479)
        assert np.isclose(tost(a, b, bound=10).at['TOST', 'pval'], 0.00017933)

        # Paired
        assert tost(a, b, paired=True).at['TOST', 'dof'] == 5
        assert np.isclose(tost(a, b, paired=True).at['TOST', 'pval'],
                          0.9293826)
        assert np.isclose(tost(a, b, bound=2, paired=True).at['TOST', 'pval'],
                          0.8286101)

        # One-sample test yield slightly different results than the equivalence
        # package. Not sure to understand why, but I think it has to do with
        # the way that they estimate the p-value of the one-sample test.
        assert tost(a, 0).at['TOST', 'pval'] > tost(a, 5).at['TOST', 'pval']
        assert tost(a, 5, bound=3).at['TOST', 'pval'] < 0.5
