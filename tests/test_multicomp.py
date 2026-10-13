import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from pingouin.multicomp import _bonf, _fdr, _holm, _sidak, multicomp

pvals = [0.52, 0.12, 0.0001, 0.03, 0.14]
pvals2 = [0.52, 0.12, 0.10, 0.30, 0.14]
pvals2_NA = [0.52, np.nan, 0.10, 0.30, 0.14]
pvals_2d = np.array([pvals, pvals2_NA])


def test_fdr():
    """Test function fdr.
    Compare to the p.adjust R function.
    """
    # FDR BH
    # ------
    reject, pval_corr = _fdr(pvals)
    assert_array_equal(reject, [False, False, True, False, False])
    assert_array_almost_equal(pval_corr, [0.52, 0.175, 0.0005, 0.075, 0.175])

    # With NaN values
    _, pval_corr = _fdr(pvals2_NA)
    assert_array_almost_equal(pval_corr, [0.52, np.nan, 0.28, 0.40, 0.28])

    # With 2D arrays
    _, pval_corr = _fdr(pvals_2d)
    pval_corr = np.round(pval_corr.ravel(), 3)
    assert_array_almost_equal(
        pval_corr, [0.52, 0.21, 0.001, 0.135, 0.21, 0.52, np.nan, 0.21, 0.386, 0.21]
    )

    # FDR BY
    # ------
    reject, pval_corr = _fdr(pvals, method="fdr_by")
    assert_array_equal(reject, [False, False, True, False, False])
    assert_array_almost_equal(pval_corr, [1.0, 0.399583333, 0.001141667, 0.171250000, 0.399583333])

    # With NaN values
    _, pval_corr = _fdr(pvals2_NA, method="fdr_by")
    assert_array_almost_equal(pval_corr, [1.0, np.nan, 0.5833333, 0.8333333, 0.5833333])

    # With 2D arrays
    _, pval_corr = _fdr(pvals_2d, method="fdr_by")
    pval_corr = np.round(pval_corr.ravel(), 3)
    assert_array_almost_equal(
        pval_corr, [1.0, 0.594, 0.003, 0.382, 0.594, 1.0, np.nan, 0.594, 1.0, 0.594]
    )


def test_bonf():
    """Test function bonf
    Compare to the p.adjust R function.
    """
    reject, pval_corr = _bonf(pvals)
    assert_array_equal(reject, [False, False, True, False, False])
    assert_array_almost_equal(pval_corr, [1, 0.6, 0.0005, 0.15, 0.7])

    # With NaN values
    _, pval_corr = _bonf(pvals2_NA)
    assert_array_almost_equal(pval_corr, [1, np.nan, 0.4, 1.0, 0.56])

    # With 2D arrays
    _, pval_corr = _bonf(pvals_2d)
    pval_corr = np.round(pval_corr.ravel(), 3)
    assert_array_almost_equal(pval_corr, [1, 1, 0.001, 0.27, 1, 1, np.nan, 0.9, 1.0, 1.0])


def test_sidak():
    """Test function sidak
    Compare to statsmodels function.
    """
    reject, pval_corr = _sidak(pvals)
    assert_array_equal(reject, [False, False, True, False, False])
    assert_array_almost_equal(
        pval_corr,
        [9.74519603e-01, 4.72268083e-01, 4.99900010e-04, 1.41265974e-01, 5.29572982e-01],
    )
    # With NaN values
    _, pval_corr = _sidak(pvals2_NA)
    assert_array_almost_equal(pval_corr, [0.94691584, np.nan, 0.3439, 0.7599, 0.45299184])
    # Empty test family (all NaN) must return NaN, not 0 (= 1 - (1 - nan) ** 0)
    reject, pval_corr = _sidak([np.nan, np.nan])
    assert_array_equal(reject, [False, False])
    assert np.isnan(pval_corr).all()
    _, pval_corr = multicomp([np.nan, np.nan], method="sidak")
    assert np.isnan(pval_corr).all()


def test_holm():
    """Test function holm.
    Compare to the p.adjust R function.
    """
    reject, pval_corr = _holm(pvals)
    assert_array_equal(reject, [False, False, True, False, False])
    assert_array_equal(pval_corr, [5.2e-01, 3.6e-01, 5.0e-04, 1.2e-01, 3.6e-01])
    _, pval_corr = _holm(pvals2)
    assert_array_equal(pval_corr, [0.6, 0.5, 0.5, 0.6, 0.5])

    # With NaN values
    _, pval_corr = _holm(pvals2_NA)
    assert_array_almost_equal(pval_corr, [0.6, np.nan, 0.4, 0.6, 0.42])

    # 2D array
    _, pval_corr = _holm(pvals_2d)
    pval_corr = np.round(pval_corr.ravel(), 3)
    assert_array_almost_equal(pval_corr, [1, 0.72, 0.001, 0.24, 0.72, 1.0, np.nan, 0.7, 0.9, 0.72])


@pytest.mark.parametrize(
    "func, methods",
    [
        (_bonf, ["b", "bonf", "bonferroni", "Bonf"]),
        (_holm, ["h", "holm"]),
        (_sidak, ["s", "sidak"]),
        (lambda p, alpha: _fdr(p, alpha=alpha, method="fdr_bh"), ["fdr", "fdr_bh", "bh"]),
        (lambda p, alpha: _fdr(p, alpha=alpha, method="fdr_by"), ["fdr_by", "by"]),
    ],
)
@pytest.mark.parametrize("p", [pvals, pvals2, pvals2_NA])
def test_multicomp(func, methods, p):
    """Test function multicomp: every alias dispatches to the right correction."""
    for alpha in [0.05, 0.2]:
        reject_exp, pvals_exp = func(np.asarray(p), alpha=alpha)
        for method in methods:
            reject, pvals_corr = multicomp(p, alpha=alpha, method=method)
            assert_array_equal(reject, reject_exp)
            assert_array_equal(pvals_corr, pvals_exp)


def test_multicomp_none():
    """Test function multicomp with method="none" and wrong arguments."""
    reject, pvals_corr = multicomp(pvals, method="none")
    assert_array_equal(pvals, pvals_corr)
    assert_array_equal(reject, np.asarray(pvals) < 0.05)
    # The output is a copy of the input
    pvals_arr = np.array(pvals)
    _, pvals_corr = multicomp(pvals_arr, method="none")
    assert pvals_corr is not pvals_arr
    # Wrong arguments
    with pytest.raises(ValueError):
        multicomp(pvals, method="wrong")
