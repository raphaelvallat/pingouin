# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
# Code borrowed from statsmodel and mne.stats
import numpy as np

__all__ = ["fdr", "bonf", "holm", "multicomp"]


def _ecdf(x):
    """No frills empirical cdf used in fdrcorrection."""
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


def fdr(pvals, alpha=0.05, method='indep'):
    """P-values correction with False Discovery Rate (FDR).

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

    References
    ----------
    - Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery
      rate: a practical and powerful approach to multiple testing. Journal of
      the Royal Statistical Society Series B, 57, 289–300.

    - Benjamini, Y., and Yekutieli, D. (2001). The control of the false
      discovery rate in multiple testing under dependency. Annals of
      Statistics, 29, 1165–1188.

    Examples
    --------
    FDR correction of an array of p-values

    >>> from pingouin import fdr
    >>> pvals = [.50, .003, .32, .054, .0003]
    >>> reject, pvals_corr = fdr(pvals, alpha=.05)
    >>> print(reject, pvals_corr)
    [False  True False False  True] [0.5    0.0075 0.4    0.09   0.0015]
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
    """P-values correction with Bonferroni method.

    Parameters
    ----------
    pvals : array_like
        Array of p-values of the individual tests.
    alpha : float
        Error rate (= alpha level).

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        P-values adjusted for multiple hypothesis testing using the Bonferroni
        procedure (= multiplied by the number of tests).

    Notes
    -----
    From Wikipedia:

    Statistical hypothesis testing is based on rejecting the null hypothesis
    if the likelihood of the observed data under the null hypotheses is low.
    If multiple hypotheses are tested, the chance of a rare event increases,
    and therefore, the likelihood of incorrectly rejecting a null hypothesis
    (i.e., making a Type I error) increases.
    The Bonferroni correction compensates for that increase by testing each
    individual hypothesis :math:`p_i` at a significance level of
    :math:`p_i = \\alpha / n` where :math:`\\alpha` is the desired overall
    alpha level and :math:`n` is the number of hypotheses. For example, if a
    trial is testing :math:`n=20` hypotheses with a desired
    :math:`\\alpha=0.05`, then the Bonferroni correction would test each
    individual hypothesis at :math:`\\alpha=0.05/20=0.0025``.

    The Bonferroni correction tends to be a bit too conservative.

    References
    ----------
    - Bonferroni, C. E. (1935). Il calcolo delle assicurazioni su gruppi
      di teste. Studi in onore del professore salvatore ortu carboni, 13-60.

    - https://en.wikipedia.org/wiki/Bonferroni_correction

    Examples
    --------
    >>> from pingouin import bonf
    >>> pvals = [.50, .003, .32, .054, .0003]
    >>> reject, pvals_corr = bonf(pvals, alpha=.05)
    >>> print(reject, pvals_corr)
    [False  True False False  True] [1.     0.015  1.     0.27   0.0015]
    """
    pvals = np.asarray(pvals)
    pvals_corrected = pvals * float(pvals.size)
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    reject = pvals_corrected < alpha
    return reject, pvals_corrected


def holm(pvals, alpha=.05):
    """P-values correction with Holm method.

    Parameters
    ----------
    pvals : array_like
        Array of p-values of the individual tests.
    alpha : float
        Error rate (= alpha level).

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corr : array
        P-values adjusted for multiple hypothesis testing using the Holm
        procedure.

    Notes
    -----
    From Wikipedia:

    In statistics, the Holm–Bonferroni method (also called the Holm method) is
    used to counteract the problem of multiple comparisons. It is intended to
    control the family-wise error rate and offers a simple test uniformly more
    powerful than the Bonferroni correction.

    Note that NaN values are not yaken into account in the p-values correction.

    References
    ----------
    - Holm, S. (1979). A simple sequentially rejective multiple test procedure.
      Scandinavian journal of statistics, 65-70.

    - https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method

    Examples
    --------
    >>> from pingouin import holm
    >>> pvals = [.50, .003, .32, .054, .0003]
    >>> reject, pvals_corr = holm(pvals, alpha=.05)
    >>> print(reject, pvals_corr)
    [False  True False False  True] [0.64   0.012  0.64   0.162  0.0015]
    """
    # Convert to array and save original shape
    pvals = np.asarray(pvals)
    shape_init = pvals.shape
    pvals = pvals.ravel()
    num_nan = np.isnan(pvals).sum()

    # Sort the (flattened) p-values
    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()
    ntests = pvals.size - num_nan

    # Now we adjust the p-values
    pvals_corr = np.diag(pvals_sorted * np.arange(ntests, 0, -1)[..., None])
    pvals_corr = np.maximum.accumulate(pvals_corr)
    pvals_corr[pvals_corr > 1.0] = 1.0
    pvals_corr = np.append(pvals_corr, np.full(num_nan, np.nan))

    # And revert to the original shape and order
    pvals_corr = pvals_corr[sortrevind].reshape(shape_init)
    with np.errstate(invalid='ignore'):
        reject = np.less(pvals_corr, alpha)
    return reject, pvals_corr


def multicomp(pvals, alpha=0.05, method='holm'):
    """P-values correction for multiple comparisons.

    Parameters
    ----------
    pvals : array_like
        uncorrected p-values.
    alpha : float
        Significance level.
    method : string
        Method used for testing and adjustment of pvalues. Can be either the
        full name or initial letters. Available methods are ::

        'bonf' : one-step Bonferroni correction
        'holm' : step-down method using Bonferroni adjustments
        'fdr_bh' : Benjamini/Hochberg FDR correction
        'fdr_by' : Benjamini/Yekutieli FDR correction

    Returns
    -------
    reject : array, boolean
        True for hypothesis that can be rejected for given alpha.
    pvals_corrected : array
        P-values corrected for multiple testing.

    See Also
    --------
    pairwise_ttests : Pairwise post-hocs T-tests

    Notes
    -----
    This function is similar to the `p.adjust` R function.

    The correction methods include the Bonferroni correction ("bonf")
    in which the p-values are multiplied by the number of comparisons.
    Less conservative methods are also included such as Holm (1979) ("holm"),
    Benjamini & Hochberg (1995) ("fdr_bh"), and Benjamini
    & Yekutieli (2001) ("fdr_by"), respectively.

    The first two methods are designed to give strong control of the
    family-wise error rate. Note that the Holm's method is usually preferred
    over the Bonferroni correction.
    The "fdr_bh" and "fdr_by" methods control the false discovery rate, i.e.
    the expected proportion of false discoveries amongst the rejected
    hypotheses. The false discovery rate is a less stringent condition than
    the family-wise error rate, so these methods are more powerful than the
    others.

    References
    ----------
    - Bonferroni, C. E. (1935). Il calcolo delle assicurazioni su gruppi
      di teste. Studi in onore del professore salvatore ortu carboni, 13-60.

    - Holm, S. (1979). A simple sequentially rejective multiple test procedure.
      Scandinavian Journal of Statistics, 6, 65–70.

    - Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery
      rate: a practical and powerful approach to multiple testing. Journal of
      the Royal Statistical Society Series B, 57, 289–300.

    - Benjamini, Y., and Yekutieli, D. (2001). The control of the false
      discovery rate in multiple testing under dependency. Annals of
      Statistics, 29, 1165–1188.

    Examples
    --------
    FDR correction of an array of p-values

    >>> from pingouin import multicomp
    >>> pvals = [.50, .003, .32, .054, .0003]
    >>> reject, pvals_corr = multicomp(pvals, method='fdr_bh')
    >>> print(reject, pvals_corr)
    [False  True False False  True] [0.5    0.0075 0.4    0.09   0.0015]
    """
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
    elif method.lower() == 'none':
        return None, None
    else:
        raise ValueError('Multiple comparison method not recognized')
    return reject, pvals_corrected
