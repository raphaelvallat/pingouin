# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
# Code borrowed from statsmodel and mne.stats
# GNU license
import numpy as np
from pingouin.utils import (_check_data, _extract_effects)

__all__ = ["fdr", "bonf", "multicomp", "pairwise_ttests"]


def _ecdf(x):
    """No frills empirical cdf used in fdrcorrection."""
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


def fdr(pvals, alpha=0.05, method='indep'):
    """P-value correction with False Discovery Rate (FDR).
    Correction for multiple comparison using FDR.
    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.
    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.
    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR
    Notes
    -----
    Reference:
    Genovese CR, Lazar NA, Nichols T.
    Thresholding of statistical maps in functional neuroimaging using the false
    discovery rate. Neuroimage. 2002 Apr;15(4):870-8.
    """
    pvals = np.asarray(pvals)
    shape_init = pvals.shape
    pvals = pvals.ravel()

    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1. / np.arange(1, len(pvals_sorted) + 1))
        ecdffactor = _ecdf(pvals_sorted) / cm
    else:
        raise ValueError("Method should be 'indep' and 'negcorr'")

    reject = pvals_sorted < (ecdffactor * alpha)
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    pvals_corrected = pvals_corrected[sortrevind].reshape(shape_init)
    reject = reject[sortrevind].reshape(shape_init)
    return reject, pvals_corrected


def bonf(pvals, alpha=0.05):
    """P-value correction with Bonferroni method.
    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing
    """
    pvals = np.asarray(pvals)
    pvals_corrected = pvals * float(pvals.size)
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    reject = pvals_corrected < alpha
    return reject, pvals_corrected


def holm(pvals, alpha=0.05):
    """P-value correction with Holm method.
    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing
    """
    pvals = np.asarray(pvals)
    ntests = pvals.size
    notreject = pvals > alpha / np.arange(ntests, 0, -1)
    nr_index = np.nonzero(notreject)[0]
    if nr_index.size == 0:
        # nonreject is empty, all rejected
        notrejectmin = ntests
    else:
        notrejectmin = np.min(nr_index)
    notreject[notrejectmin:] = True
    reject = ~notreject
    pvals_corrected_raw = pvals * np.arange(ntests, 0, -1)
    pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    reject = pvals_corrected < alpha
    return reject, pvals_corrected

def multicomp(pvals, alpha=0.05, method='holm'):
    '''Test results and p-value correction for multiple tests
    Parameters
    ----------
    pvals : array_like
        uncorrected p-values
    alpha : float
        FWER, family-wise error rate, e.g. 0.1
    method : string
        Method used for testing and adjustment of pvalues. Can be either the
        full name or initial letters. Available methods are ::
        `bonferroni` : one-step correction
        `holm` : step-down method using Bonferroni adjustments
        `fdr_bh` : Benjamini/Hochberg FDR correction
        `fdr_by` : Benjamini/Yekutieli FDR correction
    Returns
    -------
    reject : array, boolean
        true for hypothesis that can be rejected for given alpha
    pvals_corrected : array
        p-values corrected for multiple tests
    '''
    if not isinstance(pvals, (list, np.ndarray)):
        err = "pvals must be a list or a np.ndarray"
        raise ValueError(err)

    if method.lower() in ['b', 'bonf', 'bonferroni']:
        reject, pvals_corrected = bonf(pvals, alpha=alpha)
    elif method.lower() in ['h', 'holm']:
        reject, pvals_corrected = holm(pvals, alpha=alpha)
    elif method.lower() in ['fdr_bh']:
        reject, pvals_corrected = fdr(pvals, alpha=alpha, method='indep')
    elif method.lower() in ['fdr_by']:
        reject, pvals_corrected = fdr(pvals, alpha=alpha, method='negcorr')
    else:
        raise ValueError('Multiple comparison method not recognized')

    return reject, pvals_corrected


def pairwise_ttests(dv=None, between=None, within=None, effects='all', data=None,
                    alpha=.05, tailed='two-sided', padjust=None):
    '''Pairwise T-tests using Pandas
    Parameters
    ----------
    pvals : array_like
        uncorrected p-values
    alpha : float
        FWER, family-wise error rate, e.g. 0.1
    p-adjust : string
        Method used for testing and adjustment of pvalues. Can be either the
        full name or initial letters. Available methods are :
        `none` : no correction
        `bonferroni` : one-step correction
        `holm` : step-down method using Bonferroni adjustments
        `fdr_bh` : Benjamini/Hochberg FDR correction
        `fdr_by` : Benjamini/Yekutieli FDR correction
    Returns
    -------
    tvals : array
        Test statistics
    pvals : array
        (un)-corrected p-values
    '''
    from scipy.stats import ttest_ind, ttest_rel

    if tailed not in ['one-sided', 'two-sided']:
        raise ValueError('Tailed not recognized')

    # Extract data
    dt_array, nobs = _extract_effects(dv=dv, between=between, within=within,
                                     effects=effects, data=data)

    # Compute T-tests
    paired = True if effects == 'within' else False
    # Number and labels of possible comparisons
    ntests = nobs.size - 1
    tvals, pvals = np.zeros(ntests), np.zeros(ntests)
    # Start looping over columns
    for i in np.arange(ntests):
        x = dt_array[col1]
        y = dt_array[col2]
        t, p = ttest_rel(x, y) if paired else ttest_ind(x, y)
        tvals[i], pvals[i] = t, p

    # Multiple comparisons
    if padjust is not None:
        _, pvals = multicomp(pvals, alpha=alpha, method=padjust)

    if tailed=='two-sided':
        return tvals, pvals
    else:
        return tvals, 0.5 * pvals
