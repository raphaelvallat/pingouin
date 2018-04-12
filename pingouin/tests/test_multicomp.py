import pandas as pd
import numpy as np

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.multicomp import *

# Dataset
# df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'],
#                    'Time': ['Mon', 'Thur', 'Mon', 'Thur'],
#                    'Values': [1.52, 5.8, 8.2, 3.4]})

pvals = [.52, .12, .0001, .03, .14]

class TestMulticomp(_TestPingouin):
    """Test effsize.py."""

    def test_fdr(self):
        """Test function fdr"""
        fdr(pvals)
        fdr(pvals, alpha=.01, method='negcorr')

    def test_bonf(self):
        """Test function bonf"""
        bonf(pvals)
        bonf(pvals, alpha=.01)

    def test_holm(self):
        """Test function holm"""
        holm(pvals)
        holm(pvals, alpha=.01)

    def test_multicomp(self):
        """Test function multicomp"""
        reject, pvals_corr = multicomp(pvals, method='fdr_bh')
        reject, pvals_corr = multicomp(pvals, method='fdr_by')
        reject, pvals_corr = multicomp(pvals, method='h')
        reject, pvals_corr = multicomp(pvals, method='b')
        reject, pvals_corr = multicomp(pvals, method='none')
