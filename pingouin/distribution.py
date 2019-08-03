import warnings
import scipy.stats
import numpy as np
import pandas as pd
from .utils import remove_na
from .utils import _flatten_list as _fl

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
    # Geometric mean
    geo_mean = scipy.stats.gmean(x)
    # Geometric standard deviation
    gstd = np.exp(np.sqrt(np.sum((np.log(x / geo_mean))**2) / (len(x) - 1)))
    # Geometric z-score
    return np.log(x / geo_mean) / np.log(gstd)


def normality(data, dv=None, group=None, method="shapiro", alpha=.05):
    """Univariate normality test.

    Parameters
    ----------
    data : dataframe, series, list or 1D np.array
        Iterable. Can be either a single list, 1D numpy array,
        or a wide- or long-format pandas dataframe.
    dv : str
        Dependent variable (only when ``data`` is a long-format dataframe).
    group : str
        Grouping variable (only when ``data`` is a long-format dataframe).
    method : str
        Normality test. 'shapiro' (default) performs the Shapiro-Wilk test
        using :py:func:`scipy.stats.shapiro`, and 'normaltest' performs the
        omnibus test of normality using :py:func:`scipy.stats.normaltest`.
        The latter is more appropriate for large samples.
    alpha : float
        Significance level.

    Returns
    -------
    stats : dataframe
        Pandas DataFrame with columns:

        * ``'W'``: test statistic
        * ``'pval'``: p-value
        * ``'normal'``: True if ``data`` is normally distributed.

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

    Note that missing values are automatically removed (casewise deletion).

    References
    ----------
    .. [1] Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
           for normality (complete samples). Biometrika, 52(3/4), 591-611.

    .. [2] https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm

    .. [3] https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test

    Examples
    --------
    1. Shapiro-Wilk test on a 1D array.

    >>> import numpy as np
    >>> import pingouin as pg
    >>> np.random.seed(123)
    >>> x = np.random.normal(size=100)
    >>> pg.normality(x)
             W      pval  normal
    0  0.98414  0.274886    True

    2. Omnibus test on a wide-format dataframe with missing values

    >>> data = pg.read_dataset('mediation')
    >>> data.loc[1, 'X'] = np.nan
    >>> pg.normality(data, method='normaltest')
                   W           pval  normal
    X       1.791839   4.082320e-01    True
    M       0.492349   7.817859e-01    True
    Y       0.348676   8.400129e-01    True
    Mbin  839.716156  4.549393e-183   False
    Ybin  814.468158  1.381932e-177   False

    3. Pandas Series

    >>> pg.normality(data['X'], method='normaltest')
              W      pval  normal
    X  1.791839  0.408232    True

    4. Long-format dataframe

    >>> data = pg.read_dataset('rm_anova2')
    >>> pg.normality(data, dv='Performance', group='Time')
                 W      pval  normal
    Pre   0.967718  0.478773    True
    Post  0.940728  0.095157    True
    """
    assert isinstance(data, (pd.DataFrame, pd.Series, list, np.ndarray))
    assert method in ['shapiro', 'normaltest']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    col_names = ['W', 'pval']
    func = getattr(scipy.stats, method)
    if isinstance(data, (list, np.ndarray)):
        data = np.asarray(data)
        assert data.ndim == 1, 'Data must be 1D.'
        assert data.size > 3, 'Data must have more than 3 samples.'
        data = remove_na(data)
        stats = pd.DataFrame(func(data)).T
        stats.columns = col_names
        stats['normal'] = np.where(stats['pval'] > alpha, True, False)
    else:
        # Data is a Pandas DataFrame
        if dv is None and group is None:
            # Wide-format
            # Get numeric data only
            numdata = data._get_numeric_data()
            stats = numdata.apply(lambda x: func(x.dropna()),
                                  result_type='expand', axis=0).T
            stats.columns = col_names
            stats['normal'] = np.where(stats['pval'] > alpha, True, False)
        else:
            # Long-format
            stats = pd.DataFrame([])
            assert group in data.columns
            assert dv in data.columns
            grp = data.groupby(group, sort=False)
            cols = grp.groups.keys()
            for _, tmp in grp:
                stats = stats.append(normality(tmp[dv].values, method=method,
                                               alpha=alpha))
            stats.index = cols
    return stats


def homoscedasticity(data, dv=None, group=None, method="levene", alpha=.05):
    """Test equality of variance.

    Parameters
    ----------
    data : dataframe, list or dict
        Iterable. Can be either a list / dictionnary of iterables
        or a wide- or long-format pandas dataframe.
    dv : str
        Dependent variable (only when ``data`` is a long-format dataframe).
    group : str
        Grouping variable (only when ``data`` is a long-format dataframe).
    method : str
        Statistical test. 'levene' (default) performs the Levene test
        using :py:func:`scipy.stats.levene`, and 'bartlett' performs the
        Bartlett test using :py:func:`scipy.stats.bartlett`.
        The former is more robust to departure from normality.
    alpha : float
        Significance level.

    Returns
    -------
    stats : dataframe
        Pandas DataFrame with columns:

        * ``'W/T'``: test statistic ('W' for Levene, 'T' for Bartlett)
        * ``'pval'``: p-value
        * ``'equal_var'``: True if ``data`` has equal variance

    See Also
    --------
    normality : Univariate normality test.
    sphericity : Mauchly's test for sphericity.

    Notes
    -----
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

    .. warning:: Missing values are not supported for this function.
        Make sure to remove them before using the
        :py:meth:`pandas.DataFrame.dropna` or :py:func:`pingouin.remove_na`
        functions.

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
    1. Levene test on a wide-format dataframe

    >>> import numpy as np
    >>> import pingouin as pg
    >>> data = pg.read_dataset('mediation')
    >>> pg.homoscedasticity(data[['X', 'Y', 'M']])
                W      pval  equal_var
    levene  0.435  0.999997       True

    2. Bartlett test using a list of iterables

    >>> data = [[4, 8, 9, 20, 14], np.array([5, 8, 15, 45, 12])]
    >>> pg.homoscedasticity(data, method="bartlett", alpha=.05)
                  T      pval  equal_var
    bartlett  2.874  0.090045       True

    3. Long-format dataframe

    >>> data = pg.read_dataset('rm_anova2')
    >>> pg.homoscedasticity(data, dv='Performance', group='Time')
                W      pval  equal_var
    levene  3.192  0.079217       True
    """
    assert isinstance(data, (pd.DataFrame, list, dict))
    assert method.lower() in ['levene', 'bartlett']
    func = getattr(scipy.stats, method)
    if isinstance(data, pd.DataFrame):
        # Data is a Pandas DataFrame
        if dv is None and group is None:
            # Wide-format
            # Get numeric data only
            numdata = data._get_numeric_data()
            assert numdata.shape[1] > 1, 'Data must have at least two columns.'
            statistic, p = func(*numdata.values)
        else:
            # Long-format
            assert group in data.columns
            assert dv in data.columns
            grp = data.groupby(group)[dv]
            assert grp.ngroups > 1, 'Data must have at least two columns.'
            statistic, p = func(*grp.apply(list))
    elif isinstance(data, list):
        # Check that list contains other list or np.ndarray
        assert all(isinstance(el, (list, np.ndarray)) for el in data)
        assert len(data) > 1, 'Data must have at least two iterables.'
        statistic, p = func(*data)
    else:
        # Data is a dict
        assert all(isinstance(el, (list, np.ndarray)) for el in data.values())
        assert len(data) > 1, 'Data must have at least two iterables.'
        statistic, p = func(*data.values())

    equal_var = True if p > alpha else False
    stat_name = 'W' if method.lower() == 'levene' else 'T'

    stats = {
        stat_name: round(statistic, 3),
        'pval': p,
        'equal_var': equal_var
    }

    return pd.DataFrame(stats, columns=stats.keys(), index=[method])


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
    k = len(args)
    from_dist = np.zeros(k, 'bool')
    sig_level = np.zeros(k)
    for j in range(k):
        st, cr, sig = scipy.stats.anderson(args[j], dist=dist)
        from_dist[j] = True if (st > cr).any() else False
        sig_level[j] = sig[np.argmin(np.abs(st - cr))]

    if k == 1:
        from_dist = bool(from_dist)
        sig_level = float(sig_level)
    return from_dist, sig_level

###############################################################################
# REPEATED MEASURES
###############################################################################


def _check_multilevel_rm(data, func='epsilon'):
    """Check if data has multilevel columns for wide-format repeated measures.
    ``func`` can be either epsilon or mauchly
    """
    # Support for two-way factor of shape (2, N)
    if data.columns.nlevels == 1:
        # For code clarity only
        return data
    elif data.columns.nlevels == 2:
        # We sort the multiindex so that the higher factor has fewer levels
        # Make sure to use remove_unused_levels to get the "true" shape
        levshape = data.columns.remove_unused_levels().levshape
        data = data.reorder_levels(np.argsort(levshape), axis=1)
        levshape = np.sort(levshape)
        # The first factor can have only one level (see if .. below), however,
        # the second factor must have at least two levels.
        assert levshape[1] >= 2, 'Factor must have at least two levels.'
        if levshape[0] == 1:
            # Two factors but first factor has only one level (= one-way)
            data = data.droplevel(level=0, axis=1)
        elif levshape[0] == 2:
            # One factor has only two-level, e.g. (2, N) or (N, 2)
            # Let's make sure that the first factor is sorted
            data = data.sort_index(level=0, axis=1)
            # Now let's compute the difference matrix of the first level
            # We end up with a one-way design. It is similar to applying
            # a paired T-test to gain scores instead of using repeated measures
            # on two time points. Here we have computed the gain scores.
            data = data.groupby(level=1, axis=1).diff(axis=1).dropna(axis=1)
            data = data.droplevel(level=0, axis=1)
        else:
            # Both factors have more than 2 levels -- differ from R / JASP
            if func == 'epsilon':
                warnings.warn("Epsilon values might be innaccurate in "
                              "two-way repeated measures design where each  "
                              "factor has more than 2 levels. Please  "
                              "double-check your results.")
            else:
                raise ValueError("If using two-way repeated measures design, "
                                 "at least one factor must have exactly two "
                                 "levels. More complex designs are not yet "
                                 "supported.")
        return data
    else:
        raise ValueError("Only one-way or two-way designs are supported.")


def _long_to_wide_rm(data, dv=None, within=None, subject=None):
    """Convert long-format dataframe to wide-format.
    This internal function is used in pingouin.epsilon and pingouin.sphericity.
    """
    # Check arguments
    assert isinstance(dv, str), 'dv must be a string.'
    assert isinstance(subject, str), 'subject must be a string.'
    assert isinstance(within, (str, list)), 'within must be a string or list.'
    # Check that all columns are present
    assert dv in data.columns, '%s not in data' % dv
    assert data[dv].dtype.kind in 'bfi', '%s must be numeric' % dv
    assert subject in data.columns, '%s not in data' % subject
    assert not data[subject].isnull().any(), 'Cannot have NaN in %s' % subject
    if isinstance(within, str):
        within = [within]  # within = ['fac1'] or ['fac1', 'fac2']
    for w in within:
        assert w in data.columns, '%s not in data' % w
    # Keep all relevant columns and reset index
    data = data[_fl([subject, within, dv])]
    # Convert to wide-format + collapse to the mean
    data = pd.pivot_table(data, index=subject, values=dv, columns=within,
                          aggfunc='mean', dropna=True)
    return data


def epsilon(data, dv=None, within=None, subject=None, correction='gg'):
    """Epsilon adjustement factor for repeated measures.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the repeated measurements.
        Both wide and long-format dataframe are supported for this function.
        To test for an interaction term between two repeated measures factors
        with a wide-format dataframe, ``data`` must have a two-levels
        :py:class:`pandas.MultiIndex` columns.
    dv : string
        Name of column containing the dependant variable (only required if
        ``data`` is in long format).
    within : string
        Name of column containing the within factor (only required if ``data``
        is in long format).
        If ``within`` is a list with two strings, this function computes
        the epsilon factor for the interaction between the two within-subject
        factor.
    subject : string
        Name of column containing the subject identifier (only required if
        ``data`` is in long format).
    correction : string
        Specify the epsilon version ::

            'gg' : Greenhouse-Geisser
            'hf' : Huynh-Feldt
            'lb' : Lower bound

    Returns
    -------
    eps : float
        Epsilon adjustement factor.

    See Also
    --------
    sphericity : Mauchly and JNS test for sphericity.
    homoscedasticity : Test equality of variance.

    Notes
    -----
    The **lower bound epsilon** is:

    .. math:: lb = \\frac{1}{\\text{dof}},

    where the degrees of freedom :math:`\\text{dof}` is the number of groups
    :math:`k` minus 1 for one-way design and :math:`(k_1 - 1)(k_2 - 1)`
    for two-way design

    The **Greenhouse-Geisser epsilon** is given by:

    .. math::

        \\epsilon_{GG} = \\frac{k^2(\\overline{\\text{diag}(S)} -
        \\overline{S})^2}{(k-1)(\\sum_{i=1}^{k}\\sum_{j=1}^{k}s_{ij}^2 -
        2k\\sum_{j=1}^{k}\\overline{s_i}^2 + k^2\\overline{S}^2)}

    where :math:`S` is the covariance matrix, :math:`\\overline{S}` the
    grandmean of S and :math:`\\overline{\\text{diag}(S)}` the mean of all the
    elements on the diagonal of S (i.e. mean of the variances).

    The **Huynh-Feldt epsilon** is given by:

    .. math::

        \\epsilon_{HF} = \\frac{n(k-1)\\epsilon_{GG}-2}{(k-1)
        (n-1-(k-1)\\epsilon_{GG})}

    where :math:`n` is the number of observations.

    Missing values are automatically removed from ``data`` (listwise deletion).

    References
    ----------
    .. [1] http://www.real-statistics.com/anova-repeated-measures/sphericity/

    Examples
    --------
    Using a wide-format dataframe

    >>> import pandas as pd
    >>> import pingouin as pg
    >>> data = pd.DataFrame({'A': [2.2, 3.1, 4.3, 4.1, 7.2],
    ...                      'B': [1.1, 2.5, 4.1, 5.2, 6.4],
    ...                      'C': [8.2, 4.5, 3.4, 6.2, 7.2]})
    >>> gg = pg.epsilon(data, correction='gg')
    >>> hf = pg.epsilon(data, correction='hf')
    >>> lb = pg.epsilon(data, correction='lb')
    >>> print(lb, gg, hf)
    0.5 0.5587754577585018 0.6223448311539781

    Now using a long-format dataframe

    >>> data = pg.read_dataset('rm_anova2')
    >>> data.head()
       Subject Time   Metric  Performance
    0        1  Pre  Product           13
    1        2  Pre  Product           12
    2        3  Pre  Product           17
    3        4  Pre  Product           12
    4        5  Pre  Product           19

    Let's first calculate the epsilon of the *Time* within-subject factor

    >>> pg.epsilon(data, dv='Performance', subject='Subject',
    ...            within='Time')
    1.0

    Since *Time* has only two levels (Pre and Post), the sphericity assumption
    is necessarily met, and therefore the epsilon adjustement factor is 1.

    The *Metric* factor, however, has three levels:

    >>> pg.epsilon(data, dv='Performance', subject='Subject',
    ...            within=['Metric'])
    0.9691029584899856

    The epsilon value is very close to 1, meaning that there is no major
    violation of sphericity.

    Now, let's calculate the epsilon for the interaction between the two
    repeated measures factor:

    >>> pg.epsilon(data, dv='Performance', subject='Subject',
    ...            within=['Time', 'Metric'])
    0.727166420214127

    Alternatively, we could use a wide-format dataframe with two column
    levels:

    >>> # Pivot from long-format to wide-format
    >>> piv = data.pivot_table(index='Subject', columns=['Time', 'Metric'],
    ...                        values='Performance')
    >>> piv.head()
    Time      Post                   Pre
    Metric  Action Client Product Action Client Product
    Subject
    1           34     30      18     17     12      13
    2           30     18       6     18     19      12
    3           32     31      21     24     19      17
    4           40     39      18     25     25      12
    5           27     28      18     19     27      19

    >>> pg.epsilon(piv)
    0.727166420214127

    which gives the same epsilon value as the long-format dataframe.
    """
    assert isinstance(data, pd.DataFrame), 'Data must be a pandas Dataframe.'

    # If data is in long-format, convert to wide-format
    if all([v is not None for v in [dv, within, subject]]):
        data = _long_to_wide_rm(data, dv=dv, within=within, subject=subject)

    # Drop rows with missing values
    data = data.dropna()

    # Support for two-way factor of shape (2, N)
    data = _check_multilevel_rm(data, func='epsilon')

    # Covariance matrix
    S = data.cov()
    n, k = data.shape

    # Epsilon is always 1 with only two repeated measures.
    if k <= 2:
        return 1.

    # Degrees of freedom
    if S.columns.nlevels == 1:
        # One-way design
        dof = k - 1
    else:
        # Two-way design (>2, >2)
        ka, kb = S.columns.levshape
        dof = (ka - 1) * (kb - 1)

    # Lower bound
    if correction == 'lb':
        return 1 / dof

    # Greenhouse-Geisser
    # Method 1. Sums of squares. (see real-statistics.com)
    mean_var = np.diag(S).mean()
    S_mean = S.mean().mean()
    ss_mat = (S**2).sum().sum()
    ss_rows = (S.mean(1)**2).sum().sum()
    num = (k * (mean_var - S_mean))**2
    den = (k - 1) * (ss_mat - 2 * k * ss_rows + k**2 * S_mean**2)
    eps = np.min([num / den, 1])

    # Method 2. Eigenvalues.
    # Sv = S.values
    # S_pop = Sv - Sv.mean(0)[:, None] - Sv.mean(1)[None, :] + Sv.mean()
    # eig = np.linalg.eigvalsh(S_pop)
    # eig = eig[eig > 0.1]
    # V = eig.sum()**2 / np.sum(eig**2)
    # eps = np.min([V / dof, 1])

    # Huynh-Feldt
    if correction == 'hf':
        num = n * dof * eps - 2
        den = dof * (n - 1 - dof * eps)
        eps = np.min([num / den, 1])
    return eps


def sphericity(data, dv=None, within=None, subject=None, method='mauchly',
               alpha=.05):
    """Mauchly and JNS test for sphericity.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the repeated measurements.
        Both wide and long-format dataframe are supported for this function.
        To test for an interaction term between two repeated measures factors
        with a wide-format dataframe, ``data`` must have a two-levels
        :py:class:`pandas.MultiIndex` columns.
    dv : string
        Name of column containing the dependant variable (only required if
        ``data`` is in long format).
    within : string
        Name of column containing the within factor (only required if ``data``
        is in long format).
        If ``within`` is a list with two strings, this function computes
        the epsilon factor for the interaction between the two within-subject
        factor.
    subject : string
        Name of column containing the subject identifier (only required if
        ``data`` is in long format).
    method : str
        Method to compute sphericity ::

        'jns' : John, Nagao and Sugiura test.
        'mauchly' : Mauchly test (default).

    alpha : float
        Significance level

    Returns
    -------
    spher : boolean
        True if data have the sphericity property.
    W : float
        Test statistic.
    chi_sq : float
        Chi-square statistic.
    ddof : int
        Degrees of freedom.
    p : float
        P-value.

    Raises
    ------
    ValueError
        When testing for an interaction, if both within-subject factors have
        more than 2 levels (not yet supported in Pingouin).

    See Also
    --------
    epsilon : Epsilon adjustement factor for repeated measures.
    homoscedasticity : Test equality of variance.
    normality : Univariate normality test.

    Notes
    -----
    The **Mauchly** :math:`W` statistic is defined by:

    .. math::

        W = \\frac{\\prod \\lambda_j}{(\\frac{1}{k-1} \\sum \\lambda_j)^{k-1}}

    where :math:`\\lambda_j` are the eigenvalues of the population
    covariance matrix (= double-centered sample covariance matrix) and
    :math:`k` is the number of conditions.

    From then, the :math:`W` statistic is transformed into a chi-square
    score using the number of observations per condition :math:`n`

    .. math:: f = \\frac{2(k-1)^2+k+1}{6(k-1)(n-1)}
    .. math:: \\chi_w^2 = (f-1)(n-1) \\text{log}(W)

    The p-value is then approximated using a chi-square distribution:

    .. math:: \\chi_w^2 \\sim \\chi^2(\\frac{k(k-1)}{2}-1)

    The **JNS** :math:`V` statistic is defined by:

    .. math::

        V = \\frac{(\\sum_j^{k-1} \\lambda_j)^2}{\\sum_j^{k-1} \\lambda_j^2}

    .. math:: \\chi_v^2 = \\frac{n}{2}  (k-1)^2 (V - \\frac{1}{k-1})

    and the p-value approximated using a chi-square distribution

    .. math:: \\chi_v^2 \\sim \\chi^2(\\frac{k(k-1)}{2}-1)

    Missing values are automatically removed from ``data`` (listwise deletion).

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
    Mauchly test for sphericity using a wide-format dataframe

    >>> import pandas as pd
    >>> import pingouin as pg
    >>> data = pd.DataFrame({'A': [2.2, 3.1, 4.3, 4.1, 7.2],
    ...                      'B': [1.1, 2.5, 4.1, 5.2, 6.4],
    ...                      'C': [8.2, 4.5, 3.4, 6.2, 7.2]})
    >>> pg.sphericity(data)
    (True, 0.21, 4.677, 2, 0.09649016283209666)

    John, Nagao and Sugiura (JNS) test

    >>> pg.sphericity(data, method='jns')
    (False, 1.118, 6.176, 2, 0.0456042403075203)

    Now using a long-format dataframe

    >>> data = pg.read_dataset('rm_anova2')
    >>> data.head()
       Subject Time   Metric  Performance
    0        1  Pre  Product           13
    1        2  Pre  Product           12
    2        3  Pre  Product           17
    3        4  Pre  Product           12
    4        5  Pre  Product           19

    Let's first test sphericity for the *Time* within-subject factor

    >>> pg.sphericity(data, dv='Performance', subject='Subject',
    ...            within='Time')
    (True, nan, nan, 1, 1.0)

    Since *Time* has only two levels (Pre and Post), the sphericity assumption
    is necessarily met.

    The *Metric* factor, however, has three levels:

    >>> pg.sphericity(data, dv='Performance', subject='Subject',
    ...            within=['Metric'])
    (True, 0.968, 0.259, 2, 0.8784417991645136)

    The p-value value is very large, and the test therefore indicates that
    there is no violation of sphericity.

    Now, let's calculate the epsilon for the interaction between the two
    repeated measures factor. The current implementation in Pingouin only works
    if at least one of the two within-subject factors has no more than two
    levels.

    >>> pg.sphericity(data, dv='Performance', subject='Subject',
    ...            within=['Time', 'Metric'])
    (True, 0.625, 3.763, 2, 0.15239168046050933)

    Here again, there is no violation of sphericity acccording to Mauchly's
    test.

    Alternatively, we could use a wide-format dataframe with two column
    levels:

    >>> # Pivot from long-format to wide-format
    >>> piv = data.pivot_table(index='Subject', columns=['Time', 'Metric'],
    ...                        values='Performance')
    >>> piv.head()
    Time      Post                   Pre
    Metric  Action Client Product Action Client Product
    Subject
    1           34     30      18     17     12      13
    2           30     18       6     18     19      12
    3           32     31      21     24     19      17
    4           40     39      18     25     25      12
    5           27     28      18     19     27      19

    >>> pg.sphericity(piv)
    (True, 0.625, 3.763, 2, 0.15239168046050933)

    which gives the same output as the long-format dataframe.
    """
    assert isinstance(data, pd.DataFrame), 'Data must be a pandas Dataframe.'

    # If data is in long-format, convert to wide-format
    if all([v is not None for v in [dv, within, subject]]):
        data = _long_to_wide_rm(data, dv=dv, within=within, subject=subject)

    # Remove rows with missing values in wide-format dataframe
    data = data.dropna()

    # Support for two-way factor of shape (2, N)
    data = _check_multilevel_rm(data, func='mauchly')

    # From here, we work only with one-way design
    n, k = data.shape
    d = k - 1

    # Sphericity is always met with only two repeated measures.
    if k <= 2:
        return True, np.nan, np.nan, 1, 1.

    # Compute dof of the test
    ddof = (d * (d + 1)) / 2 - 1
    ddof = 1 if ddof == 0 else ddof

    if method.lower() == 'mauchly':
        # Method 1. Contrast matrix. Similar to R & Matlab implementation.
        # Only works for one-way design or two-way design with shape (2, N).
        # 1 - Compute the successive difference matrix Z.
        #     (Note that the order of columns does not matter.)
        # 2 - Find the contrast matrix that M so that data * M = Z
        # 3 - Performs the QR decomposition of this matrix (= contrast matrix)
        # 4 - Compute sample covariance matrix S
        # 5 - Compute Mauchly's statistic
        # Z = data.diff(axis=1).dropna(axis=1)
        # M = np.linalg.lstsq(data, Z, rcond=None)[0]
        # C, _ = np.linalg.qr(M)
        # S = data.cov()
        # A = C.T.dot(S).dot(C)
        # logW = np.log(np.linalg.det(A)) - d * np.log(np.trace(A / d))
        # W = np.exp(logW)

        # Method 2. Eigenvalue-based method. Faster.
        # 1 - Estimate the population covariance (= double-centered)
        # 2 - Calculate n-1 eigenvalues
        # 3 - Compute Mauchly's statistic
        S = data.cov().values  # values here, otherwise S.mean() != grandmean
        S_pop = S - S.mean(0)[:, None] - S.mean(1)[None, :] + S.mean()
        eig = np.linalg.eigvalsh(S_pop)[1:]
        eig = eig[eig > 0.001]  # Additional check to remove very low eig
        W = np.product(eig) / (eig.sum() / d)**d
        logW = np.log(W)

        # Compute chi-square and p-value (adapted from the ezANOVA R package)
        f = 1 - (2 * d**2 + d + 2) / (6 * d * (n - 1))
        w2 = ((d + 2) * (d - 1) * (d - 2) * (2 * d**3 + 6 * d**2 + 3 * k + 2)
              / (288 * ((n - 1) * d * f)**2))
        chi_sq = -(n - 1) * f * logW
        p1 = scipy.stats.chi2.sf(chi_sq, ddof)
        p2 = scipy.stats.chi2.sf(chi_sq, ddof + 4)
        pval = p1 + w2 * (p2 - p1)
    else:
        # Method = JNS
        eps = epsilon(data, correction='gg')
        W = eps * d
        chi_sq = 0.5 * n * d**2 * (W - 1 / d)
        pval = scipy.stats.chi2.sf(chi_sq, ddof)

    sphericity = True if pval > alpha else False
    return sphericity, np.round(W, 3), np.round(chi_sq, 3), int(ddof), pval
