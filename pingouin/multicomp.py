# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np

__all__ = ["multicomp"]


##############################################################################
# INTERNAL FUNCTIONS
##############################################################################

def fdr(pvals, alpha=0.05, method='fdr_bh'):
    """P-values FDR correction with Benjamini/Hochberg and
    Benjamini/Yekutieli procedure.

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like
        Array of p-values of the individual tests.
    alpha : float
        Error rate (= alpha level).
    method : str
        FDR correction methods ::

        'fdr_bh' : Benjamini/Hochberg for independent / posit correlated tests
        'fdr_by' : Benjamini/Yekutieli for negatively correlated tests

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        P-values adjusted for multiple hypothesis testing using the BH or BY
        correction.

    See also
    --------
    bonf : Bonferroni correction
    holm : Holm-Bonferroni correction

    Notes
    -----
    From Wikipedia:

    The **Benjamini–Hochberg** procedure (BH step-up procedure) controls the
    false discovery rate (FDR) at level :math:`\\alpha`. It works as follows:

    1. For a given :math:`\\alpha`, find the largest :math:`k` such that
    :math:`P_{(k)}\\leq \\frac {k}{m}\\alpha.`

    2. Reject the null hypothesis (i.e., declare discoveries) for all
    :math:`H_{(i)}` for :math:`i = 1, \\ldots, k`.

    The BH procedure is valid when the m tests are independent, and also in
    various scenarios of dependence, but is not universally valid.

    The **Benjamini–Yekutieli** procedure (BY) controls the FDR under arbitrary
    dependence assumptions. This refinement modifies the threshold and finds
    the largest :math:`k` such that:

    .. math::
        P_{(k)} \\leq \\frac{k}{m \\cdot c(m)} \\alpha

    References
    ----------
    - Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery
      rate: a practical and powerful approach to multiple testing. Journal of
      the Royal Statistical Society Series B, 57, 289–300.

    - Benjamini, Y., and Yekutieli, D. (2001). The control of the false
      discovery rate in multiple testing under dependency. Annals of
      Statistics, 29, 1165–1188.

    - https://en.wikipedia.org/wiki/False_discovery_rate

    Examples
    --------
    FDR correction of an array of p-values

    >>> import pingouin as pg
    >>> pvals = [.50, .003, .32, .054, .0003]
    >>> reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh', alpha=.05)
    >>> print(reject, pvals_corr)
    [False  True False False  True] [0.5    0.0075 0.4    0.09   0.0015]
    """
    assert method.lower() in ['fdr_bh', 'fdr_by']
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

    # Empirical CDF factor
    ecdffactor = np.arange(1, ntests + 1) / float(ntests)

    if method.lower() == 'fdr_by':
        cm = np.sum(1. / np.arange(1, ntests + 1))
        ecdffactor /= cm

    # Now we adjust the p-values
    pvals_corr = np.diag(pvals_sorted / ecdffactor[..., None])
    pvals_corr = np.minimum.accumulate(pvals_corr[::-1])[::-1]
    pvals_corr = np.clip(pvals_corr, None, 1)

    # And revert to the original shape and order
    pvals_corr = np.append(pvals_corr, np.full(num_nan, np.nan))
    pvals_corrected = pvals_corr[sortrevind].reshape(shape_init)
    with np.errstate(invalid='ignore'):
        reject = np.less(pvals_corrected, alpha)
    # reject = reject[sortrevind].reshape(shape_init)
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

    See also
    --------
    holm : Holm-Bonferroni correction
    fdr : Benjamini/Hochberg and Benjamini/Yekutieli FDR correction

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

    The Bonferroni adjusted p-values are defined as:

    .. math::
        \\widetilde {p}_{{(i)}}= n \\cdot p_{{(i)}}

    The Bonferroni correction tends to be a bit too conservative.

    Note that NaN values are not taken into account in the p-values correction.

    References
    ----------
    - Bonferroni, C. E. (1935). Il calcolo delle assicurazioni su gruppi
      di teste. Studi in onore del professore salvatore ortu carboni, 13-60.

    - https://en.wikipedia.org/wiki/Bonferroni_correction

    Examples
    --------
    >>> import pingouin as pg
    >>> pvals = [.50, .003, .32, .054, .0003]
    >>> reject, pvals_corr = pg.multicomp(pvals, method='bonf', alpha=.05)
    >>> print(reject, pvals_corr)
    [False  True False False  True] [1.     0.015  1.     0.27   0.0015]
    """
    pvals = np.asarray(pvals)
    num_nan = np.isnan(pvals).sum()
    pvals_corrected = pvals * (float(pvals.size) - num_nan)
    pvals_corrected = np.clip(pvals_corrected, None, 1)
    with np.errstate(invalid='ignore'):
        reject = np.less(pvals_corrected, alpha)
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
    pvals_corrected : array
        P-values adjusted for multiple hypothesis testing using the Holm
        procedure.

    See also
    --------
    bonf : Bonferroni correction
    fdr : Benjamini/Hochberg and Benjamini/Yekutieli FDR correction

    Notes
    -----
    From Wikipedia:

    In statistics, the Holm–Bonferroni method (also called the Holm method) is
    used to counteract the problem of multiple comparisons. It is intended to
    control the family-wise error rate and offers a simple test uniformly more
    powerful than the Bonferroni correction.

    The Holm adjusted p-values are the running maximum of the sorted p-values
    divided by the corresponding increasing alpha level:

    .. math::

        \\frac{\\alpha}{n}, \\frac{\\alpha}{n-1}, ..., \\frac{\\alpha}{1}

    where :math:`n` is the number of test.

    The full mathematical formula is:

    .. math::
        \\widetilde {p}_{{(i)}}=\\max _{{j\\leq i}}\\left\\{(n-j+1)p_{{(j)}}
        \\right\\}_{{1}}

    Note that NaN values are not taken into account in the p-values correction.

    References
    ----------
    - Holm, S. (1979). A simple sequentially rejective multiple test procedure.
      Scandinavian journal of statistics, 65-70.

    - https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method

    Examples
    --------
    >>> import pingouin as pg
    >>> pvals = [.50, .003, .32, .054, .0003]
    >>> reject, pvals_corr = pg.multicomp(pvals, method='holm', alpha=.05)
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
    pvals_corr = np.clip(pvals_corr, None, 1)

    # And revert to the original shape and order
    pvals_corr = np.append(pvals_corr, np.full(num_nan, np.nan))
    pvals_corrected = pvals_corr[sortrevind].reshape(shape_init)
    with np.errstate(invalid='ignore'):
        reject = np.less(pvals_corrected, alpha)
    return reject, pvals_corrected


def sidak(pvals, alpha=0.05):
    """P-values correction with Sidak method.

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
        P-values adjusted for multiple hypothesis testing using the Sidak
        procedure.

    See also
    --------
    bonf, holm, fdr, multicomp

    Notes
    -----
    The Sidak adjusted p-values are defined as:

    .. math::
        \\widetilde {p}_{{(i)}}= 1 - (1 - p_{{(i)}})^{n}

    The Sidak correction is slightly more liberal than the Bonferroni
    correction.

    Note that NaN values are not taken into account in the p-values correction.

    References
    ----------
    - Šidák, Z. K. (1967). "Rectangular Confidence Regions for the Means of
      Multivariate Normal Distributions". Journal of the American Statistical
      Association. 62 (318): 626–633.

    - https://en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction

    Examples
    --------
    >>> import numpy as np
    >>> import pingouin as pg
    >>> pvals = [.50, .003, .32, .054, .0003]
    >>> reject, pvals_corr = pg.multicomp(pvals, method='sidak', alpha=.05)
    >>> print(reject, np.round(pvals_corr, 4))
    [False  True False False  True] [0.9688 0.0149 0.8546 0.2424 0.0015]
    """
    pvals = np.asarray(pvals)
    num_nan = np.isnan(pvals).sum()
    ntests = (float(pvals.size) - num_nan)
    pvals_corrected = 1 - np.power((1. - pvals), ntests)
    pvals_corrected = np.clip(pvals_corrected, None, 1)
    with np.errstate(invalid='ignore'):
        reject = np.less(pvals_corrected, alpha)
    return reject, pvals_corrected


##############################################################################
# EXTERNAL FUNCTION
##############################################################################

def multicomp(pvals, alpha=0.05, method='holm'):
    """P-values correction for multiple comparisons.

    Parameters
    ----------
    pvals : array_like
        Uncorrected p-values.
    alpha : float
        Significance level.
    method : string
        Method used for testing and adjustment of p-values. Can be either the
        full name or initial letters. Available methods are ::

        'bonf' : one-step Bonferroni correction
        'sidak' : one-step Sidak correction
        'holm' : step-down method using Bonferroni adjustments
        'fdr_bh' : Benjamini/Hochberg FDR correction
        'fdr_by' : Benjamini/Yekutieli FDR correction
        'none' : pass-through option (no correction applied)

    Returns
    -------
    reject : array, boolean
        True for hypothesis that can be rejected for given alpha.
    pvals_corrected : array
        P-values corrected for multiple testing.

    Notes
    -----
    This function is similar to the `p.adjust` R function.

    The correction methods include the Bonferroni correction (``bonf``)
    in which the p-values are multiplied by the number of comparisons.
    Less conservative methods are also included such as Sidak (1967)
    (``sidak``), Holm (1979) (``holm``), Benjamini & Hochberg (1995)
    (``fdr_bh``), and Benjamini & Yekutieli (2001) (``fdr_by``), respectively.

    The first three methods are designed to give strong control of the
    family-wise error rate. Note that the Holm's method is usually preferred.
    The ``fdr_bh`` and ``fdr_by`` methods control the false discovery rate,
    i.e. the expected proportion of false discoveries amongst the rejected
    hypotheses. The false discovery rate is a less stringent condition than
    the family-wise error rate, so these methods are more powerful than the
    others.

    The **Bonferroni** adjusted p-values are defined as:

    .. math::
        \\widetilde {p}_{{(i)}}= n \\cdot p_{{(i)}}

    where :math:`n` is the number of *finite* p-values (i.e. excluding NaN).

    The **Sidak** adjusted p-values are defined as:

    .. math::
        \\widetilde {p}_{{(i)}}= 1 - (1 - p_{{(i)}})^{n}

    The **Holm** adjusted p-values are the running maximum of the sorted
    p-values divided by the corresponding increasing alpha level:

    .. math::
        \\widetilde {p}_{{(i)}}=\\max _{{j\\leq i}}\\left\\{(n-j+1)p_{{(j)}}
        \\right\\}_{{1}}

    The **Benjamini–Hochberg** procedure (BH step-up procedure) controls the
    false discovery rate (FDR) at level :math:`\\alpha`. It works as follows:

    1. For a given :math:`\\alpha`, find the largest :math:`k` such that
    :math:`P_{(k)}\\leq \\frac {k}{n}\\alpha.`

    2. Reject the null hypothesis for all
    :math:`H_{(i)}` for :math:`i = 1, \\ldots, k`.

    The BH procedure is valid when the :math:`n` tests are independent, and
    also in various scenarios of dependence, but is not universally valid.

    The **Benjamini–Yekutieli** procedure (BY) controls the FDR under arbitrary
    dependence assumptions. This refinement modifies the threshold and finds
    the largest :math:`k` such that:

    .. math::
        P_{(k)} \\leq \\frac{k}{n \\cdot c(n)} \\alpha

    References
    ----------
    - Bonferroni, C. E. (1935). Il calcolo delle assicurazioni su gruppi
      di teste. Studi in onore del professore salvatore ortu carboni, 13-60.

    - Šidák, Z. K. (1967). "Rectangular Confidence Regions for the Means of
      Multivariate Normal Distributions". Journal of the American Statistical
      Association. 62 (318): 626–633.

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

    >>> import pingouin as pg
    >>> pvals = [.50, .003, .32, .054, .0003]
    >>> reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
    >>> print(reject, pvals_corr)
    [False  True False False  True] [0.5    0.0075 0.4    0.09   0.0015]

    Holm correction with missing values

    >>> import numpy as np
    >>> pvals[2] = np.nan
    >>> reject, pvals_corr = pg.multicomp(pvals, method='holm')
    >>> print(reject, pvals_corr)
    [False  True False False  True] [0.5    0.009     nan 0.108  0.0012]
    """
    # Safety check
    assert isinstance(pvals, (list, np.ndarray)), "pvals must be list or array"
    pvals = np.squeeze(np.asarray(pvals))
    assert isinstance(alpha, float), 'alpha must be a float.'
    assert isinstance(method, str), 'method must be a string.'
    assert 0 < alpha < 1, 'alpha must be between 0 and 1.'

    if method.lower() in ['b', 'bonf', 'bonferroni']:
        reject, pvals_corrected = bonf(pvals, alpha=alpha)
    elif method.lower() in ['h', 'holm']:
        reject, pvals_corrected = holm(pvals, alpha=alpha)
    elif method.lower() in ['s', 'sidak']:
        reject, pvals_corrected = sidak(pvals, alpha=alpha)
    elif method.lower() in ['fdr', 'fdr_bh', 'bh']:
        reject, pvals_corrected = fdr(pvals, alpha=alpha, method='fdr_bh')
    elif method.lower() in ['fdr_by', 'by']:
        reject, pvals_corrected = fdr(pvals, alpha=alpha, method='fdr_by')
    elif method.lower() == 'none':
        pvals_corrected = pvals
        with np.errstate(invalid='ignore'):
            reject = np.less(pvals_corrected, alpha)
    else:
        raise ValueError('Multiple comparison method not recognized')
    return reject, pvals_corrected
