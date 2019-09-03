import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from unittest import TestCase
from pingouin.multicomp import fdr, bonf, holm, sidak, multicomp

pvals = [.52, .12, .0001, .03, .14]
pvals2 = [.52, .12, .10, .30, .14]
pvals2_NA = [.52, np.nan, .10, .30, .14]
pvals_2d = np.array([pvals, pvals2_NA])


class TestMulticomp(TestCase):
    """Test multicomp.py."""

    def test_fdr(self):
        """Test function fdr.
        Compare to the p.adjust R function.
        """

        # FDR BH
        # ------
        reject, pval_corr = fdr(pvals)
        assert_array_equal(reject, [False, False, True, False, False])
        assert_array_almost_equal(pval_corr, [0.52, .175, .0005, .075, .175])

        # With NaN values
        _, pval_corr = fdr(pvals2_NA)
        assert_array_almost_equal(pval_corr, [.52, np.nan, .28, .40, .28])

        # With 2D arrays
        _, pval_corr = fdr(pvals_2d)
        pval_corr = np.round(pval_corr.ravel(), 3)
        assert_array_almost_equal(pval_corr, [.52, .21, .001, .135, .21, .52,
                                              np.nan, .21, 0.386, .21])

        # FDR BY
        # ------
        reject, pval_corr = fdr(pvals, method='fdr_by')
        assert_array_equal(reject, [False, False, True, False, False])
        assert_array_almost_equal(pval_corr, [1., 0.399583333, 0.001141667,
                                              0.171250000, 0.399583333])

        # With NaN values
        _, pval_corr = fdr(pvals2_NA, method='fdr_by')
        assert_array_almost_equal(pval_corr, [1., np.nan, 0.5833333, 0.8333333,
                                              .5833333])

        # With 2D arrays
        _, pval_corr = fdr(pvals_2d, method='fdr_by')
        pval_corr = np.round(pval_corr.ravel(), 3)
        assert_array_almost_equal(pval_corr, [1., 0.594, 0.003, 0.382, 0.594,
                                              1., np.nan, 0.594, 1., 0.594])

    def test_bonf(self):
        """Test function bonf
        Compare to the p.adjust R function.
        """
        reject, pval_corr = bonf(pvals)
        assert_array_equal(reject, [False, False, True, False, False])
        assert_array_almost_equal(pval_corr, [1, 0.6, 0.0005, 0.15, 0.7])

        # With NaN values
        _, pval_corr = bonf(pvals2_NA)
        assert_array_almost_equal(pval_corr, [1, np.nan, 0.4, 1., 0.56])

        # With 2D arrays
        _, pval_corr = bonf(pvals_2d)
        pval_corr = np.round(pval_corr.ravel(), 3)
        assert_array_almost_equal(pval_corr, [1, 1, 0.001, 0.27, 1, 1, np.nan,
                                              .9, 1., 1.])

    def test_sidak(self):
        """Test function sidak
        Compare to statsmodels function.
        """
        reject, pval_corr = sidak(pvals)
        assert_array_equal(reject, [False, False, True, False, False])
        assert_array_almost_equal(pval_corr, [9.74519603e-01, 4.72268083e-01,
                                              4.99900010e-04, 1.41265974e-01,
                                              5.29572982e-01])
        # With NaN values
        _, pval_corr = sidak(pvals2_NA)
        assert_array_almost_equal(pval_corr, [0.94691584, np.nan, 0.3439,
                                              0.7599, 0.45299184])

    def test_holm(self):
        """Test function holm.
        Compare to the p.adjust R function.
        """
        reject, pval_corr = holm(pvals)
        assert_array_equal(reject, [False, False, True, False, False])
        assert_array_equal(pval_corr, [5.2e-01, 3.6e-01, 5.0e-04,
                                       1.2e-01, 3.6e-01])
        _, pval_corr = holm(pvals2)
        assert_array_equal(pval_corr, [0.6, 0.5, 0.5, 0.6, 0.5])

        # With NaN values
        _, pval_corr = holm(pvals2_NA)
        assert_array_almost_equal(pval_corr, [0.6, np.nan, 0.4, 0.6, 0.42])

        # 2D array
        _, pval_corr = holm(pvals_2d)
        pval_corr = np.round(pval_corr.ravel(), 3)
        assert_array_almost_equal(pval_corr, [1, 0.72, 0.001, 0.24, 0.72,
                                              1., np.nan, 0.7, 0.9, 0.72])

    def test_multicomp(self):
        """Test function multicomp"""
        reject, pvals_corr = multicomp(pvals, method='fdr_bh')
        reject, pvals_corr = multicomp(pvals, method='fdr_by')
        reject, pvals_corr = multicomp(pvals, method='h')
        reject, pvals_corr = multicomp(pvals, method='b')
        reject, pvals_corr = multicomp(pvals, method='sidak')
        reject, pvals_corr = multicomp(pvals, method='none')
        assert_array_equal(pvals, pvals_corr)
        reject, pvals_corr = multicomp(pvals2, method='holm')
        # Wrong arguments
        with pytest.raises(ValueError):
            reject, pvals_corr = multicomp(pvals, method='wrong')
