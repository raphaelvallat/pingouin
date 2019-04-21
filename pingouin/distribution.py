import numpy as np

__all__ = ["gzscore", "normality", "homoscedasticity", "anderson",
           "epsilon", "sphericity"]


def gzscore(x):
    """Geometric standard (Z) score.

    Parameters
    ----------
    x : array_like
        Array of raw values

    Returns
    -------
    gzscore : array_like
        Array of geometric z-scores (same shape as x)

    Notes
    -----
    Geometric Z-scores are better measures of dispersion than arithmetic
    z-scores when the sample data come from a log-normally distributed
    population.

    Given the raw scores :math:`x`, the geometric mean :math:`\\mu_g` and
    the geometric standard deviation :math:`\\sigma_g`,
    the standard score is given by the formula:

    .. math:: z = \\frac{log(x) - log(\\mu_g)}{log(\\sigma_g)}

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Geometric_standard_deviation

    Examples
    --------
    Standardize a log-normal array

    >>> import numpy as np
    >>> from pingouin import gzscore
    >>> np.random.seed(123)
    >>> raw = np.random.lognormal(size=100)
    >>> z = gzscore(raw)
    >>> print(round(z.mean(), 3), round(z.std(), 3))
    -0.0 0.995
    """
    from scipy.stats import gmean
    # Geometric mean
    geo_mean = gmean(x)
    # Geometric standard deviation
    gstd = np.exp(np.sqrt(np.sum((np.log(x / geo_mean))**2) / (len(x) - 1)))
    # Geometric z-score
    return np.log(x / geo_mean) / np.log(gstd)


def normality(*args, alpha=.05):
    """Shapiro-Wilk univariate normality test.

    Parameters
    ----------
    sample1, sample2,... : array_like
        Array of sample data. May be of different lengths.

    Returns
    -------
    normal : boolean
        True if x comes from a normal distribution.
    p : float
        P-value.

    See Also
    --------
    homoscedasticity : Test equality of variance.
    sphericity : Mauchly's test for sphericity.

    Notes
    -----
    The Shapiro-Wilk test calculates a :math:`W` statistic that tests whether a
    random sample :math:`x_1, x_2, ..., x_n` comes from a normal distribution.

    The :math:`W` statistic is calculated as follows:

    .. math::

        W = \\frac{(\\sum_{i=1}^n a_i x_{i})^2}
        {\\sum_{i=1}^n (x_i - \\overline{x})^2}

    where the :math:`x_i` are the ordered sample values (in ascending
    order) and the :math:`a_i` are constants generated from the means,
    variances and covariances of the order statistics of a sample of size
    :math:`n` from a standard normal distribution. Specifically:

    .. math:: (a_1, ..., a_n) = \\frac{m^TV^{-1}}{(m^TV^{-1}V^{-1}m)^{1/2}}

    with :math:`m = (m_1, ..., m_n)^T` and :math:`(m_1, ..., m_n)` are the
    expected values of the order statistics of independent and identically
    distributed random variables sampled from the standard normal distribution,
    and :math:`V` is the covariance matrix of those order statistics.

    The null-hypothesis of this test is that the population is normally
    distributed. Thus, if the p-value is less than the
    chosen alpha level (typically set at 0.05), then the null hypothesis is
    rejected and there is evidence that the data tested are not normally
    distributed.

    The result of the Shapiro-Wilk test should be interpreted with caution in
    the case of large sample sizes. Indeed, quoting from Wikipedia:

    *"Like most statistical significance tests, if the sample size is
    sufficiently large this test may detect even trivial departures from the
    null hypothesis (i.e., although there may be some statistically significant
    effect, it may be too small to be of any practical significance); thus,
    additional investigation of the effect size is typically advisable,
    e.g., a Q–Q plot in this case."*

    References
    ----------
    .. [1] Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
           for normality (complete samples). Biometrika, 52(3/4), 591-611.

    .. [2] https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm

    .. [3] https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test

    Examples
    --------
    1. Test the normality of one array.

    >>> import numpy as np
    >>> from pingouin import normality
    >>> np.random.seed(123)
    >>> x = np.random.normal(size=100)
    >>> normal, p = normality(x, alpha=.05)
    >>> print(normal, p)
    True 0.275

    2. Test the normality of two arrays.

    >>> import numpy as np
    >>> from pingouin import normality
    >>> np.random.seed(123)
    >>> x = np.random.normal(size=100)
    >>> y = np.random.rand(100)
    >>> normal, p = normality(x, y, alpha=.05)
    >>> print(normal, p)
    [ True False] [0.275 0.001]
    """
    from scipy.stats import shapiro
    k = len(args)
    p = np.zeros(k)
    normal = np.zeros(k, 'bool')
    for j in range(k):
        _, p[j] = shapiro(args[j])
        normal[j] = True if p[j] > alpha else False

    if k == 1:
        normal = bool(normal)
        p = float(p)

    return normal, np.round(p, 3)


def homoscedasticity(*args, alpha=.05):
    """Test equality of variance.

    Parameters
    ----------
    sample1, sample2,... : array_like
        Array of sample data. May be different lengths.

    Returns
    -------
    equal_var : boolean
        True if data have equal variance.
    p : float
        P-value.

    See Also
    --------
    normality : Test the univariate normality of one or more array(s).
    sphericity : Mauchly's test for sphericity.

    Notes
    -----
    This function first tests if the data are normally distributed using the
    Shapiro-Wilk test. If yes, then the homogeneity of variances is measured
    using the Bartlett test. If the data are not normally distributed, the
    Levene (1960) test, which is less sensitive to departure from
    normality, is used.

    The **Bartlett** :math:`T` statistic is defined as:

    .. math::

        T = \\frac{(N-k) \\ln{s^{2}_{p}} - \\sum_{i=1}^{k}(N_{i} - 1)
        \\ln{s^{2}_{i}}}{1 + (1/(3(k-1)))((\\sum_{i=1}^{k}{1/(N_{i} - 1))}
        - 1/(N-k))}

    where :math:`s_i^2` is the variance of the :math:`i^{th}` group,
    :math:`N` is the total sample size, :math:`N_i` is the sample size of the
    :math:`i^{th}` group, :math:`k` is the number of groups,
    and :math:`s_p^2` is the pooled variance.

    The pooled variance is a weighted average of the group variances and is
    defined as:

    .. math:: s^{2}_{p} = \\sum_{i=1}^{k}(N_{i} - 1)s^{2}_{i}/(N-k)

    The p-value is then computed using a chi-square distribution:

    .. math:: T \\sim \\chi^2(k-1)

    The **Levene** :math:`W` statistic is defined as:

    .. math::

        W = \\frac{(N-k)} {(k-1)}
        \\frac{\\sum_{i=1}^{k}N_{i}(\\overline{Z}_{i.}-\\overline{Z})^{2} }
        {\\sum_{i=1}^{k}\\sum_{j=1}^{N_i}(Z_{ij}-\\overline{Z}_{i.})^{2} }

    where :math:`Z_{ij} = |Y_{ij} - median({Y}_{i.})|`,
    :math:`\\overline{Z}_{i.}` are the group means of :math:`Z_{ij}` and
    :math:`\\overline{Z}` is the grand mean of :math:`Z_{ij}`.

    The p-value is then computed using a F-distribution:

    .. math:: W \\sim F(k-1, N-k)

    References
    ----------
    .. [1] Bartlett, M. S. (1937). Properties of sufficiency and statistical
           tests. Proc. R. Soc. Lond. A, 160(901), 268-282.

    .. [2] Brown, M. B., & Forsythe, A. B. (1974). Robust tests for the
           equality of variances. Journal of the American Statistical
           Association, 69(346), 364-367.

    .. [3] NIST/SEMATECH e-Handbook of Statistical Methods,
           http://www.itl.nist.gov/div898/handbook/

    Examples
    --------
    Test the homoscedasticity of two arrays.

    >>> import numpy as np
    >>> from pingouin import homoscedasticity
    >>> np.random.seed(123)
    >>> # Scale = standard deviation of the distribution.
    >>> x = np.random.normal(loc=0, scale=1., size=100)
    >>> y = np.random.normal(loc=0, scale=0.8,size=100)
    >>> equal_var, p = homoscedasticity(x, y, alpha=.05)
    >>> print(round(np.var(x), 3), round(np.var(y), 3), equal_var, p)
    1.273 0.602 False 0.0
    """
    from scipy.stats import levene, bartlett
    k = len(args)
    if k < 2:
        raise ValueError("Must enter at least two input sample vectors.")

    # Test normality of data
    normal, _ = normality(*args)
    if np.count_nonzero(normal) != normal.size:
        # print('Data are not normally distributed. Using Levene test.')
        _, p = levene(*args)
    else:
        _, p = bartlett(*args)

    equal_var = True if p > alpha else False
    return equal_var, np.round(p, 3)


def anderson(*args, dist='norm'):
    """Anderson-Darling test of distribution.

    Parameters
    ----------
    sample1, sample2,... : array_like
        Array of sample data. May be different lengths.
    dist : string
        Distribution ('norm', 'expon', 'logistic', 'gumbel')

    Returns
    -------
    from_dist : boolean
        True if data comes from this distribution.
    sig_level : float
        The significance levels for the corresponding critical values in %.
        (See :py:func:`scipy.stats.anderson` for more details)

    Examples
    --------
    1. Test that an array comes from a normal distribution

    >>> from pingouin import anderson
    >>> x = [2.3, 5.1, 4.3, 2.6, 7.8, 9.2, 1.4]
    >>> anderson(x, dist='norm')
    (False, 15.0)

    2. Test that two arrays comes from an exponential distribution

    >>> y = [2.8, 12.4, 28.3, 3.2, 16.3, 14.2]
    >>> anderson(x, y, dist='expon')
    (array([False, False]), array([15., 15.]))
    """
    from scipy.stats import anderson as ads
    k = len(args)
    from_dist = np.zeros(k, 'bool')
    sig_level = np.zeros(k)
    for j in range(k):
        st, cr, sig = ads(args[j], dist=dist)
        from_dist[j] = True if (st > cr).any() else False
        sig_level[j] = sig[np.argmin(np.abs(st - cr))]

    if k == 1:
        from_dist = bool(from_dist)
        sig_level = float(sig_level)
    return from_dist, sig_level


def epsilon(data, correction='gg'):
    """Epsilon adjustement factor for repeated measures.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the repeated measurements.
        ``data`` must be in wide-format. To convert from wide to long format,
        use the :py:func:`pandas.pivot_table` function.
    correction : string
        Specify the epsilon version ::

            'gg' : Greenhouse-Geisser
            'hf' : Huynh-Feldt
            'lb' : Lower bound

    Returns
    -------
    eps : float
        Epsilon adjustement factor.

    Notes
    -----
    The **lower bound** for epsilon is:

    .. math:: lb = \\frac{1}{k - 1}

    where :math:`k` is the number of groups (= data.shape[1]).

    The **Greenhouse-Geisser epsilon** is given by:

    .. math::

        \\epsilon_{GG} = \\frac{k^2(\\overline{diag(S)} - \\overline{S})^2}
        {(k-1)(\\sum_{i=1}^{k}\\sum_{j=1}^{k}s_{ij}^2 - 2k\\sum_{j=1}^{k}
        \\overline{s_i}^2 + k^2\\overline{S}^2)}

    where :math:`S` is the covariance matrix, :math:`\\overline{S}` the
    grandmean of S and :math:`\\overline{diag(S)}` the mean of all the elements
    on the diagonal of S (i.e. mean of the variances).

    The **Huynh-Feldt epsilon** is given by:

    .. math::

        \\epsilon_{HF} = \\frac{n(k-1)\\epsilon_{GG}-2}{(k-1)
        (n-1-(k-1)\\epsilon_{GG})}

    where :math:`n` is the number of subjects.

    References
    ----------
    .. [1] http://www.real-statistics.com/anova-repeated-measures/sphericity/

    Examples
    --------

    >>> import pandas as pd
    >>> from pingouin import epsilon
    >>> data = pd.DataFrame({'A': [2.2, 3.1, 4.3, 4.1, 7.2],
    ...                      'B': [1.1, 2.5, 4.1, 5.2, 6.4],
    ...                      'C': [8.2, 4.5, 3.4, 6.2, 7.2]})
    >>> epsilon(data, correction='gg')
    0.5587754577585018

    >>> epsilon(data, correction='hf')
    0.6223448311539781

    >>> epsilon(data, correction='lb')
    0.5
    """
    # Covariance matrix
    S = data.cov()
    n = data.shape[0]
    k = data.shape[1]

    if correction == 'lb':
        return 1 / (k - 1)

    mean_var = np.diag(S).mean()
    S_mean = S.mean().mean()
    ss_mat = (S**2).sum().sum()
    ss_rows = (S.mean(1)**2).sum().sum()

    # Compute GGEpsilon
    num = (k * (mean_var - S_mean))**2
    den = (k - 1) * (ss_mat - 2 * k * ss_rows + k**2 * S_mean**2)
    eps = num / den

    if correction == 'hf':
        num = n * (k - 1) * eps - 2
        den = (k - 1) * (n - 1 - (k - 1) * eps)
        eps = np.min([num / den, 1])
    return eps


def sphericity(data, method='mauchly', alpha=.05):
    """Mauchly and JNS test for sphericity.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the repeated measurements.
        ``data`` must be in wide-format. To convert from wide to long format,
        use the :py:func:`pandas.pivot_table` function.
    method : str
        Method to compute sphericity ::

        'jns' : John, Nagao and Sugiura test.
        'mauchly' : Mauchly test.

    alpha : float
        Significance level

    Returns
    -------
    spher : boolean
        True if data have the sphericity property.
    W : float
        Test statistic
    chi_sq : float
        Chi-square statistic
    ddof : int
        Degrees of freedom
    p : float
        P-value.

    See Also
    --------
    homoscedasticity : Test equality of variance.
    normality : Test the univariate normality of one or more array(s).

    Notes
    -----
    The **Mauchly** :math:`W` statistic is defined by:

    .. math::

        W = \\frac{\\prod_{j=1}^{r-1} \\lambda_j}{(\\frac{1}{r-1}
        \\cdot \\sum_{j=1}^{^{r-1}} \\lambda_j)^{r-1}}

    where :math:`\\lambda_j` are the eigenvalues of the population
    covariance matrix (= double-centered sample covariance matrix) and
    :math:`r` is the number of conditions.

    From then, the :math:`W` statistic is transformed into a chi-square
    score using the number of observations per condition :math:`n`

    .. math:: f = \\frac{2(r-1)^2+r+1}{6(r-1)(n-1)}
    .. math:: \\chi_w^2 = (f-1)(n-1) log(W)

    The p-value is then approximated using a chi-square distribution:

    .. math:: \\chi_w^2 \\sim \\chi^2(\\frac{r(r-1)}{2}-1)

    The **JNS** :math:`V` statistic is defined by:

    .. math::

        V = \\frac{(\\sum_j^{r-1} \\lambda_j)^2}{\\sum_j^{r-1} \\lambda_j^2}

    .. math:: \\chi_v^2 = \\frac{n}{2}  (r-1)^2 (V - \\frac{1}{r-1})

    and the p-value approximated using a chi-square distribution

    .. math:: \\chi_v^2 \\sim \\chi^2(\\frac{r(r-1)}{2}-1)


    References
    ----------
    .. [1] Mauchly, J. W. (1940). Significance test for sphericity of a normal
           n-variate distribution. The Annals of Mathematical Statistics,
           11(2), 204-209.

    .. [2] Nagao, H. (1973). On some test criteria for covariance matrix.
           The Annals of Statistics, 700-709.

    .. [3] Sugiura, N. (1972). Locally best invariant test for sphericity and
           the limiting distributions. The Annals of Mathematical Statistics,
           1312-1316.

    .. [4] John, S. (1972). The distribution of a statistic used for testing
           sphericity of normal distributions. Biometrika, 59(1), 169-173.

    .. [5] http://www.real-statistics.com/anova-repeated-measures/sphericity/

    Examples
    --------
    1. Mauchly test for sphericity

    >>> import pandas as pd
    >>> from pingouin import sphericity
    >>> data = pd.DataFrame({'A': [2.2, 3.1, 4.3, 4.1, 7.2],
    ...                      'B': [1.1, 2.5, 4.1, 5.2, 6.4],
    ...                      'C': [8.2, 4.5, 3.4, 6.2, 7.2]})
    >>> sphericity(data)
    (True, 0.21, 4.677, 2, 0.09649016283209666)

    2. JNS test for sphericity

    >>> sphericity(data, method='jns')
    (False, 1.118, 6.176, 2, 0.04560424030751982)
    """
    from scipy.stats import chi2
    S = data.cov().values
    n = data.shape[0]
    p = data.shape[1]
    d = p - 1

    # Estimate of the population covariance (= double-centered)
    S_pop = S - S.mean(0)[:, np.newaxis] - S.mean(1)[np.newaxis, :] + S.mean()

    # p - 1 eigenvalues (sorted by ascending importance)
    eig = np.linalg.eigvalsh(S_pop)[1:]

    if method == 'jns':
        # eps = epsilon(data, correction='gg')
        # W = eps * d
        W = eig.sum()**2 / np.square(eig).sum()
        chi_sq = 0.5 * n * d ** 2 * (W - 1 / d)

    if method == 'mauchly':
        # Mauchly's statistic
        W = np.product(eig) / (eig.sum() / d)**d
        # Chi-square
        f = (2 * d**2 + p + 1) / (6 * d * (n - 1))
        chi_sq = (f - 1) * (n - 1) * np.log(W)

    # Compute dof and pval
    ddof = 0.5 * d * p - 1
    # Ensure that dof is not zero
    ddof = 1 if ddof == 0 else ddof
    pval = chi2.sf(chi_sq, ddof)

    # Second order approximation
    # pval2 = chi2.sf(chi_sq, ddof + 4)
    # w2 = (d + 2) * (d - 1) * (d - 2) * (2 * d**3 + 6 * d * d + 3 * d + 2) / \
    #      (288 * d * d * nr * nr * dd * dd)
    # pval += w2 * (pval2 - pval)

    sphericity = True if pval > alpha else False
    return sphericity, np.round(W, 3), np.round(chi_sq, 3), int(ddof), pval
