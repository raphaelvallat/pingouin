import pandas as pd
import numpy as np
import pytest

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.multicomp import fdr, bonf, holm, multicomp

# Dataset
# df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'],
#                    'Time': ['Mon', 'Thur', 'Mon', 'Thur'],
#                    'Values': [1.52, 5.8, 8.2, 3.4]})

pvals = [.52, .12, .0001, .03, .14]
pvals2 = [.52, .12, .10, .30, .14]

class TestMulticomp(_TestPingouin):
    """Test effsize.py."""

    def test_fdr(self):
        """Test function fdr"""
        fdr(pvals)
        fdr(pvals, alpha=.01, method='negcorr')
        fdr(pvals2)
        fdr(pvals2, alpha=.90, method='negcorr')
        # Wrong arguments
        with pytest.raises(ValueError):
            fdr(pvals, method='wrong')

    def test_bonf(self):
        """Test function bonf"""
        bonf(pvals)
        bonf(pvals, alpha=.01)
        bonf(pvals, alpha=.90)

    def test_holm(self):
        """Test function holm"""
        holm(pvals)
        holm(pvals, alpha=.01)
        holm(pvals, alpha=.90)

    def test_multicomp(self):
        """Test function multicomp"""
        reject, pvals_corr = multicomp(pvals, method='fdr_bh')
        reject, pvals_corr = multicomp(pvals, method='fdr_by')
        reject, pvals_corr = multicomp(pvals, method='h')
        reject, pvals_corr = multicomp(pvals, method='b')
        reject, pvals_corr = multicomp(pvals, method='none')
        reject, pvals_corr = multicomp(pvals2, method='holm')
        # Wrong arguments
        with pytest.raises(ValueError):
            reject, pvals_corr = multicomp(pvals, method='wrong')
        with pytest.raises(ValueError):
            reject, pvals_corr = multicomp(pvals=0)
