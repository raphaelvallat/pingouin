import numpy as np
from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.nonparametric import (mwu, wilcoxon)

np.random.seed(1234)
x = np.random.normal(size=100)
y = np.random.normal(size=100)


class TestNonParametric(_TestPingouin):
    """Test nonparametric.py."""

    def test_mwu(self):
        """Test function mwu"""
        mwu(x, y, tail='one-sided')
        mwu(x, y, tail='two-sided')

    def test_wilcoxon(self):
        """Test function wilcoxon"""
        wilcoxon(x, y, tail='one-sided')
        wilcoxon(x, y, tail='two-sided')
