import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats

from .utils import _flatten_list as _fl
from .utils import _postprocess_dataframe, remove_na

__all__ = ["normality", "homoscedasticity", "anderson", "epsilon", "sphericity"]


def normality(data, dv=None, group=None, method="shapiro", alpha=0.05):
    """Univariate normality test.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`, series, list or 1D np.array
        Iterable. Can be either a single list, 1D numpy array,
        or a wide- or long-format pandas dataframe.
    dv : str
        Dependent variable (only when ``data`` is a long-format dataframe).
    group : str
        Grouping variable (only when ``data`` is a long-format dataframe).
    method : str
        Normality test. `'shapiro'` (default) performs the Shapiro-Wilk test
        using :py:func:`scipy.stats.shapiro`, `'normaltest'` performs the
        omnibus test of normality using :py:func:`scipy.stats.normaltest`, `'jarque_bera'` performs
        the Jarque-Bera test using :py:func:`scipy.stats.jarque_bera`.
        The Omnibus and Jarque-Bera tests are more suitable than the Shapiro test for
        large samples.
    alpha : float
        Significance level.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'W'``: Test statistic.
        * ``'pval'``: p-value.
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
    the case of large sample sizes. Indeed, quoting from
    `Wikipedia <https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test>`_:

        *"Like most statistical significance tests, if the sample size is
        sufficiently large this test may detect even trivial departures from
        the null hypothesis (i.e., although there may be some statistically
        significant effect, it may be too small to be of any practical
        significance); thus, additional investigation of the effect size is
        typically advisable, e.g., a Q–Q plot in this case."*

    Note that missing values are automatically removed (casewise deletion).

    References
    ----------
    * Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
      for normality (complete samples). Biometrika, 52(3/4), 591-611.

    * https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm

    Examples
    --------
    1. Shapiro-Wilk test on a 1D array.

    >>> import numpy as np
    >>> import pingouin as pg
    >>> np.random.seed(123)
    >>> x = np.random.normal(size=100)
    >>> pg.normality(x).round(3)
           W   pval  normal
    0  0.984  0.275    True

    2. Omnibus test on a wide-format dataframe with missing values

    >>> data = pg.read_dataset("mediation")
    >>> data.loc[1, "X"] = np.nan
    >>> pg.normality(data, method="normaltest").round(3)
                W   pval  normal
    X       1.792  0.408    True
    M       0.492  0.782    True
    Y       0.349  0.840    True
    Mbin  839.716  0.000   False
    Ybin  814.468  0.000   False
    W1     24.816  0.000   False
    W2     43.400  0.000   False

    3. Pandas Series

    >>> pg.normality(data["X"], method="normaltest")
              W      pval  normal
    X  1.791839  0.408232    True

    4. Long-format dataframe

    >>> data = pg.read_dataset("rm_anova2")
    >>> pg.normality(data, dv="Performance", group="Time").round(3)
              W   pval  normal
    Time
    Pre   0.968  0.479    True
    Post  0.941  0.095    True

    5. Same but using the Jarque-Bera test

    >>> pg.normality(data, dv="Performance", group="Time", method="jarque_bera")
                 W      pval  normal
    Time
    Pre   0.304021  0.858979    True
    Post  1.265656  0.531088    True
    """
    assert isinstance(data, (pd.DataFrame, pd.Series, list, np.ndarray))
    assert method in ["shapiro", "normaltest", "jarque_bera"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    col_names = ["W", "pval"]
    func = getattr(scipy.stats, method)
    if isinstance(data, (list, np.ndarray)):
        data = np.asarray(data)
        assert data.ndim == 1, "Data must be 1D."
        assert data.size > 3, "Data must have more than 3 samples."
        data = remove_na(data)
        stats = pd.DataFrame(func(data)).T
        stats.columns = col_names
        stats["normal"] = np.where(stats["pval"] > alpha, True, False)
    else:
        # Data is a Pandas DataFrame
        if dv is None and group is None:
            # Wide-format
            # Get numeric data only
            numdata = data._get_numeric_data()
            stats = numdata.apply(lambda x: func(x.dropna()), result_type="expand", axis=0).T
            stats.columns = col_names
            stats["normal"] = np.where(stats["pval"] > alpha, True, False)
        else:
            # Long-format
            assert group in data.columns
            assert dv in data.columns
            rows = {}
            for idx, x in data.groupby(group, observed=True, sort=False)[dv]:
                x = x.dropna().to_numpy()
                if x.size <= 3:
                    warnings.warn(f"Group {idx} has less than 4 valid samples. Returning NaN.")
                    rows[idx] = (np.nan, np.nan)
                else:
                    rows[idx] = tuple(func(x))[:2]
            stats = pd.DataFrame.from_dict(rows, orient="index", columns=col_names)
            stats["normal"] = stats["pval"] > alpha
            stats.index.name = group
    return _postprocess_dataframe(stats)


def homoscedasticity(data, dv=None, group=None, method="levene", alpha=0.05, **kwargs):
    """Test equality of variance.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`, list or dict
        Iterable. Can be either a list / dictionnary of iterables
        or a wide- or long-format pandas dataframe.
    dv : str
        Dependent variable (only when ``data`` is a long-format dataframe).
    group : str
        Grouping variable (only when ``data`` is a long-format dataframe).
    method : str
        Statistical test. `'levene'` (default) performs the Levene test
        using :py:func:`scipy.stats.levene`, and `'bartlett'` performs the
        Bartlett test using :py:func:`scipy.stats.bartlett`.
        The former is more robust to departure from normality.
    alpha : float
        Significance level.
    **kwargs : optional
        Optional argument(s) passed to the lower-level :py:func:`scipy.stats.levene` function.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'W/T'``: Test statistic ('W' for Levene, 'T' for Bartlett)
        * ``'pval'``: p-value
        * ``'equal_var'``: True if ``data`` has equal variance

    See Also
    --------
    normality : Univariate normality test.
    sphericity : Mauchly's test for sphericity.

    Notes
    -----
    The **Bartlett** :math:`T` statistic [1]_ is defined as:

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

    The **Levene** :math:`W` statistic [2]_ is defined as:

    .. math::

        W = \\frac{(N-k)} {(k-1)}
        \\frac{\\sum_{i=1}^{k}N_{i}(\\overline{Z}_{i.}-\\overline{Z})^{2} }
        {\\sum_{i=1}^{k}\\sum_{j=1}^{N_i}(Z_{ij}-\\overline{Z}_{i.})^{2} }

    where :math:`Z_{ij} = |Y_{ij} - \\text{median}({Y}_{i.})|`,
    :math:`\\overline{Z}_{i.}` are the group means of :math:`Z_{ij}` and
    :math:`\\overline{Z}` is the grand mean of :math:`Z_{ij}`.

    The p-value is then computed using a F-distribution:

    .. math:: W \\sim F(k-1, N-k)

    Missing values are automatically removed from each sample.

    References
    ----------
    .. [1] Bartlett, M. S. (1937). Properties of sufficiency and statistical
           tests. Proc. R. Soc. Lond. A, 160(901), 268-282.

    .. [2] Brown, M. B., & Forsythe, A. B. (1974). Robust tests for the
           equality of variances. Journal of the American Statistical
           Association, 69(346), 364-367.

    Examples
    --------
    1. Levene test on a wide-format dataframe

    >>> import numpy as np
    >>> import pingouin as pg
    >>> data = pg.read_dataset("mediation")
    >>> pg.homoscedasticity(data[["X", "Y", "M"]])
                   W      pval  equal_var
    levene  1.173518  0.310707       True

    2. Same data but using a long-format dataframe

    >>> data_long = data[["X", "Y", "M"]].melt()
    >>> pg.homoscedasticity(data_long, dv="value", group="variable")
                   W      pval  equal_var
    levene  1.173518  0.310707       True

    3. Same but using a mean center

    >>> pg.homoscedasticity(data_long, dv="value", group="variable", center="mean")
                   W      pval  equal_var
    levene  1.572239  0.209303       True

    4. Bartlett test using a list of iterables

    >>> data = [[4, 8, 9, 20, 14], np.array([5, 8, 15, 45, 12])]
    >>> pg.homoscedasticity(data, method="bartlett", alpha=0.05)
                     T      pval  equal_var
    bartlett  2.873569  0.090045       True
    """
    assert isinstance(data, (pd.DataFrame, list, dict))
    method = method.lower()
    assert method in ["levene", "bartlett"]
    func = getattr(scipy.stats, method)
    if isinstance(data, pd.DataFrame):
        # Data is a Pandas DataFrame
        if dv is None and group is None:
            # Wide-format
            # Get numeric data only
            numdata = data._get_numeric_data()
            assert numdata.shape[1] > 1, "Data must have at least two columns."
            samples = numdata.to_numpy().T
        else:
            # Long-format
            assert group in data.columns
            assert dv in data.columns
            grp = data.groupby(group, observed=True)[dv]
            assert grp.ngroups > 1, "Data must have at least two columns."
            samples = grp.apply(list)
    elif isinstance(data, list):
        # Check that list contains other list or np.ndarray
        assert all(isinstance(el, (list, np.ndarray)) for el in data)
        assert len(data) > 1, "Data must have at least two iterables."
        samples = data
    else:
        # Data is a dict
        assert all(isinstance(el, (list, np.ndarray)) for el in data.values())
        assert len(data) > 1, "Data must have at least two iterables."
        samples = data.values()

    # Cast to float: scipy.stats.bartlett fails with integer inputs in SciPy >= 1.17
    # Missing values are removed separately in each sample.
    samples = [remove_na(np.asarray(x, dtype=float)) for x in samples]
    statistic, p = func(*samples, **kwargs)
    equal_var = True if p > alpha else False
    stat_name = "W" if method == "levene" else "T"
    stats = pd.DataFrame({stat_name: statistic, "pval": p, "equal_var": equal_var}, index=[method])

    return _postprocess_dataframe(stats)


def anderson(*args, dist="norm"):
    """Anderson-Darling test of distribution.

    The Anderson-Darling test tests the null hypothesis that a sample is drawn from a population
    that follows a particular distribution. For the Anderson-Darling test, the critical values
    depend on which distribution is being tested against.

    This function is a wrapper around :py:func:`scipy.stats.anderson`.

    Parameters
    ----------
    sample1, sample2,... : array_like
        Array of sample data. They may be of different lengths.
    dist : string
        The type of distribution to test against. The default is 'norm'.
        Must be one of 'norm', 'expon', 'logistic', 'gumbel'.

    Returns
    -------
    from_dist : boolean
        A boolean indicating if the data comes from the tested distribution (True) or not (False).
    sig_level : float
        The significance levels for the corresponding critical values, in %.
        See :py:func:`scipy.stats.anderson` for more details.

    Examples
    --------
    1. Test that an array comes from a normal distribution

    >>> from pingouin import anderson
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> x = np.random.normal(size=100)
    >>> y = np.random.normal(size=10000)
    >>> z = np.random.random(1000)
    >>> anderson(x)
    (True, 15.0)

    2. Test that multiple arrays comes from the normal distribution

    >>> anderson(x, y, z)
    (array([ True,  True, False]), array([15., 15.,  1.]))

    3. Test that an array comes from the exponential distribution

    >>> x = np.random.exponential(size=1000)
    >>> anderson(x, dist="expon")
    (True, 15.0)
    """
    k = len(args)
    from_dist = np.zeros(k, dtype="bool")
    sig_level = np.zeros(k)
    for j in range(k):
        st, cr, sig = scipy.stats.anderson(args[j], dist=dist)
        from_dist[j] = True if (st < cr).any() else False
        sig_level[j] = sig[np.argmin(np.abs(st - cr))]

    if k == 1:
        from_dist = from_dist[0]
        sig_level = sig_level[0]
    return from_dist, sig_level


###############################################################################
# REPEATED MEASURES
###############################################################################


def _orthonormal_contrasts(k):
    """(k, k - 1) matrix of orthonormal contrasts, i.e. orthogonal to the grand mean."""
    Q, _ = np.linalg.qr(np.column_stack([np.ones(k), np.eye(k)[:, :-1]]))
    return Q[:, 1:]


def _rm_contrasts(columns):
    """Orthonormal contrasts of the highest-order effect of a wide-format repeated measures design.

    ``columns`` are the columns of the wide-format dataframe: a single level for a one-way design
    (contrasts of the main effect), or a two-level :py:class:`pandas.MultiIndex` for a two-way
    design (contrasts of the interaction). The interaction contrasts are the Kronecker product of
    the contrasts of each factor, as in R (``mauchly.test``, car, afex), SPSS and JASP.
    Returns an array of shape (n_columns, dof).
    """
    if columns.nlevels > 2:
        raise ValueError("Only one-way or two-way designs are supported.")
    codes = [pd.factorize(columns.get_level_values(i))[0] for i in range(columns.nlevels)]
    n_levels = [c.max() + 1 for c in codes]
    if np.prod(n_levels) != len(columns):
        raise ValueError("Each combination of the within-subject factors must appear exactly once.")
    C = np.ones((len(columns), 1))
    for c, k in zip(codes, n_levels):
        # Row-wise Kronecker product. A factor with only one level has no contrast and is ignored,
        # e.g. the interaction of a (1, k) design is the main effect of the second factor.
        if k > 1:
            C_fac = _orthonormal_contrasts(k)[c]
            C = (C[:, :, None] * C_fac[:, None, :]).reshape(len(columns), -1)
    return C


def _contrast_cov(data):
    """Covariance matrix of the orthonormal contrasts, of shape (dof, dof).

    ``data`` is a wide-format dataframe without missing values. Epsilon and sphericity tests are
    computed on this matrix.
    """
    S = data.cov(numeric_only=True)
    C = _rm_contrasts(S.columns)
    return C.T @ S.to_numpy() @ C


def _mauchly(M, df_resid, k):
    """Mauchly's test of sphericity from the (d, d) covariance matrix of orthonormal contrasts.

    ``df_resid`` is the residual degrees of freedom of the covariance matrix,
    i.e. n - 1 in a repeated measures design and n - n_groups when ``M`` is computed from
    the pooled within-group covariance of a mixed design. ``k`` is the number of repeated
    measures conditions (e.g. ka * kb for the interaction of a two-way design).
    Returns W, chi-square, dof and p-value. Same as R ``mauchly.test``.
    """
    d = M.shape[0]
    # Compute dof of the test
    ddof = (d * (d + 1)) / 2 - 1
    # W = det(M) / (tr(M) / d)^d
    sign, logdet = np.linalg.slogdet(M)
    logW = logdet - d * np.log(np.trace(M) / d) if sign > 0 else -np.inf
    W = np.exp(logW)

    # Compute chi-square and p-value. Note that R uses the number of conditions k, and not d,
    # in the second-order term w2 (Anderson 2003 uses d). We follow R to get the same p-values.
    f = 1 - (2 * d**2 + d + 2) / (6 * d * df_resid)
    w2 = (
        (d + 2)
        * (d - 1)
        * (d - 2)
        * (2 * d**3 + 6 * d**2 + 3 * k + 2)
        / (288 * (df_resid * d * f) ** 2)
    )
    chi_sq = -df_resid * f * logW
    p1 = scipy.stats.chi2.sf(chi_sq, ddof)
    p2 = scipy.stats.chi2.sf(chi_sq, ddof + 4)
    pval = p1 + w2 * (p2 - p1)
    return W, chi_sq, ddof, pval


def _gg_epsilon(M):
    """Greenhouse-Geisser epsilon from the (d, d) covariance matrix of orthonormal contrasts.

    Epsilon is always 1 with only one degree of freedom (e.g. two repeated measures).
    """
    d = M.shape[0]
    if d <= 1:
        return 1.0
    return np.min([np.trace(M) ** 2 / (d * np.trace(M @ M)), 1])


def _mauchly_sphericity(M, df_resid, k, alpha=0.05):
    """Mauchly's test of sphericity, as reported in the repeated measures ANOVA tables.

    Same parameters as :py:func:`_mauchly`. Returns whether sphericity is met, W and the p-value.
    Sphericity is always met with only one degree of freedom (e.g. two repeated measures).
    """
    if M.shape[0] <= 1:
        return True, np.nan, 1.0
    W, _, _, pval = _mauchly(M, df_resid, k)
    return bool(pval > alpha), W, pval


def _long_to_wide_rm(data, dv=None, within=None, subject=None):
    """Convert long-format dataframe to wide-format.
    This internal function is used in pingouin.epsilon and pingouin.sphericity.
    """
    # Check that all columns are present
    assert dv in data.columns, "%s not in data" % dv
    assert data[dv].dtype.kind in "bfiu", "%s must be numeric" % dv
    assert subject in data.columns, "%s not in data" % subject
    assert not data[subject].isnull().any(), "Cannot have missing values in %s" % subject
    if isinstance(within, (str, int)):
        within = [within]  # within = ['fac1'] or ['fac1', 'fac2']
    for w in within:
        assert w in data.columns, "%s not in data" % w
    # Keep all relevant columns and reset index
    data = data[_fl([subject, within, dv])]
    # Convert to wide-format + collapse to the mean
    data = pd.pivot_table(
        data, index=subject, values=dv, columns=within, aggfunc="mean", dropna=True, observed=True
    )
    return data


def _wide_rm(data, dv=None, within=None, subject=None):
    """Wide-format dataframe of a repeated measures design, without missing values.

    ``data`` is converted from long to wide format if ``dv``, ``within`` and ``subject`` are
    specified. Rows with missing values are removed (listwise deletion).
    This internal function is used in pingouin.epsilon and pingouin.sphericity.
    """
    assert isinstance(data, pd.DataFrame), "Data must be a pandas Dataframe."
    if all([v is not None for v in [dv, within, subject]]):
        data = _long_to_wide_rm(data, dv=dv, within=within, subject=subject)
    return data.dropna()


def epsilon(data, dv=None, within=None, subject=None, correction="gg"):
    """Epsilon adjustement factor for repeated measures.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame containing the repeated measurements.
        Both wide and long-format dataframe are supported for this function.
        To test for an interaction term between two repeated measures factors
        with a wide-format dataframe, ``data`` must have a two-levels
        :py:class:`pandas.MultiIndex` columns.
    dv : string
        Name of column containing the dependent variable (only required if
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
        Specify the epsilon version:

        * ``'gg'``: Greenhouse-Geisser
        * ``'hf'``: Huynh-Feldt
        * ``'lb'``: Lower bound

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
    The lower bound epsilon is:

    .. math:: lb = \\frac{1}{\\text{dof}},

    where the degrees of freedom :math:`\\text{dof}` is the number of groups
    :math:`k` minus 1 for one-way design and :math:`(k_1 - 1)(k_2 - 1)`
    for two-way design

    The Greenhouse-Geisser epsilon is given by:

    .. math::

        \\epsilon_{GG} = \\frac{\\text{tr}(M)^2}{\\text{dof} \\cdot \\text{tr}(M^2)}

    where :math:`M = C^T S C`, :math:`S` is the covariance matrix and :math:`C` is a
    :math:`(k, \\text{dof})` matrix of orthonormal contrasts. For the interaction of a two-way
    design, :math:`C` is the Kronecker product of the orthonormal contrasts of each factor, as in
    R, SPSS and JASP.

    The Huynh-Feldt epsilon is given by:

    .. math::

        \\epsilon_{HF} = \\frac{n \\cdot \\text{dof} \\cdot \\epsilon_{GG}-2}{\\text{dof}
        (n-1-\\text{dof} \\cdot \\epsilon_{GG})}

    where :math:`n` is the number of observations.

    Missing values are automatically removed from data (listwise deletion).

    Examples
    --------
    Using a wide-format dataframe

    >>> import pandas as pd
    >>> import pingouin as pg
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": [2.2, 3.1, 4.3, 4.1, 7.2],
    ...         "B": [1.1, 2.5, 4.1, 5.2, 6.4],
    ...         "C": [8.2, 4.5, 3.4, 6.2, 7.2],
    ...     }
    ... )
    >>> gg = pg.epsilon(data, correction="gg")
    >>> hf = pg.epsilon(data, correction="hf")
    >>> lb = pg.epsilon(data, correction="lb")
    >>> print("%.2f %.2f %.2f" % (lb, gg, hf))
    0.50 0.56 0.62

    Now using a long-format dataframe

    >>> data = pg.read_dataset("rm_anova2")
    >>> data.head()
       Subject Time   Metric  Performance
    0        1  Pre  Product           13
    1        2  Pre  Product           12
    2        3  Pre  Product           17
    3        4  Pre  Product           12
    4        5  Pre  Product           19

    Let's first calculate the epsilon of the *Time* within-subject factor

    >>> pg.epsilon(data, dv="Performance", subject="Subject", within="Time")
    1.0

    Since *Time* has only two levels (Pre and Post), the sphericity assumption
    is necessarily met, and therefore the epsilon adjustement factor is 1.

    The *Metric* factor, however, has three levels:

    >>> round(pg.epsilon(data, dv="Performance", subject="Subject", within=["Metric"]), 3)
    0.969

    The epsilon value is very close to 1, meaning that there is no major
    violation of sphericity.

    Now, let's calculate the epsilon for the interaction between the two
    repeated measures factor:

    >>> round(pg.epsilon(data, dv="Performance", subject="Subject", within=["Time", "Metric"]), 3)
    0.727

    Alternatively, we could use a wide-format dataframe with two column
    levels:

    >>> # Pivot from long-format to wide-format
    >>> piv = data.pivot(index="Subject", columns=["Time", "Metric"], values="Performance")
    >>> piv.head()
    Time        Pre                  Post
    Metric  Product Client Action Product Client Action
    Subject
    1            13     12     17      18     30     34
    2            12     19     18       6     18     30
    3            17     19     24      21     31     32
    4            12     25     25      18     39     40
    5            19     27     19      18     28     27

    >>> round(pg.epsilon(piv), 3)
    0.727

    which gives the same epsilon value as the long-format dataframe.
    """
    # Wide-format data without missing values, and covariance matrix of the orthonormal contrasts
    data = _wide_rm(data, dv=dv, within=within, subject=subject)
    M = _contrast_cov(data)
    n, dof = data.shape[0], M.shape[0]

    # Epsilon is always 1 with only one degree of freedom (e.g. two repeated measures).
    if dof <= 1:
        return 1.0

    # Lower bound
    if correction == "lb":
        return 1 / dof

    # Greenhouse-Geisser
    eps = _gg_epsilon(M)

    # Huynh-Feldt
    if correction == "hf":
        num = n * dof * eps - 2
        den = dof * (n - 1 - dof * eps)
        eps = np.min([num / den, 1])
    return eps


def sphericity(data, dv=None, within=None, subject=None, method="mauchly", alpha=0.05):
    """Mauchly and JNS test for sphericity.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame containing the repeated measurements.
        Both wide and long-format dataframe are supported for this function.
        To test for an interaction term between two repeated measures factors
        with a wide-format dataframe, ``data`` must have a two-levels
        :py:class:`pandas.MultiIndex` columns.
    dv : string
        Name of column containing the dependent variable (only required if
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
        Method to compute sphericity:

        * `'jns'`: John, Nagao and Sugiura test.
        * `'mauchly'`: Mauchly test (default).

    alpha : float
        Significance level

    Returns
    -------
    spher : boolean
        True if data have the sphericity property.
    W : float
        Test statistic.
    chi2 : float
        Chi-square statistic.
    dof : int
        Degrees of freedom.
    pval : float
        P-value.

    See Also
    --------
    epsilon : Epsilon adjustement factor for repeated measures.
    homoscedasticity : Test equality of variance.
    normality : Univariate normality test.

    Notes
    -----
    The **Mauchly** :math:`W` statistic [1]_ is defined by:

    .. math::

        W = \\frac{\\prod \\lambda_j}{(\\frac{1}{k-1} \\sum \\lambda_j)^{k-1}}

    where :math:`\\lambda_j` are the eigenvalues of the covariance matrix of the orthonormal
    contrasts :math:`M = C^T S C` (see :py:func:`pingouin.epsilon`) and :math:`k` is the number of
    conditions. For the interaction of a two-way design, :math:`k - 1` is replaced by
    :math:`(k_1 - 1)(k_2 - 1)`.

    From then, the :math:`W` statistic is transformed into a chi-square
    score using the number of observations per condition :math:`n`

    .. math:: f = \\frac{2(k-1)^2+k+1}{6(k-1)(n-1)}
    .. math:: \\chi_w^2 = (f-1)(n-1) \\text{log}(W)

    The p-value is then approximated using a chi-square distribution:

    .. math:: \\chi_w^2 \\sim \\chi^2(\\frac{k(k-1)}{2}-1)

    The **JNS** :math:`V` statistic ([2]_, [3]_, [4]_) is defined by:

    .. math::

        V = \\frac{\\sum_j^{k-1} \\lambda_j^2}{(\\sum_j^{k-1} \\lambda_j)^2}

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

    See also http://www.real-statistics.com/anova-repeated-measures/sphericity/

    Examples
    --------
    Mauchly test for sphericity using a wide-format dataframe

    >>> import pandas as pd
    >>> import pingouin as pg
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": [2.2, 3.1, 4.3, 4.1, 7.2],
    ...         "B": [1.1, 2.5, 4.1, 5.2, 6.4],
    ...         "C": [8.2, 4.5, 3.4, 6.2, 7.2],
    ...     }
    ... )
    >>> spher, W, chisq, dof, pval = pg.sphericity(data)
    >>> print(spher, round(W, 3), round(chisq, 3), dof, round(pval, 3))
    True 0.21 4.677 2 0.096

    John, Nagao and Sugiura (JNS) test

    >>> round(pg.sphericity(data, method="jns")[-1], 3)  # P-value only
    0.139

    Now using a long-format dataframe

    >>> data = pg.read_dataset("rm_anova2")
    >>> data.head()
       Subject Time   Metric  Performance
    0        1  Pre  Product           13
    1        2  Pre  Product           12
    2        3  Pre  Product           17
    3        4  Pre  Product           12
    4        5  Pre  Product           19

    Let's first test sphericity for the *Time* within-subject factor

    >>> pg.sphericity(data, dv="Performance", subject="Subject", within="Time")
    (True, nan, nan, 1, 1.0)

    Since *Time* has only two levels (Pre and Post), the sphericity assumption
    is necessarily met.

    The *Metric* factor, however, has three levels:

    >>> round(pg.sphericity(data, dv="Performance", subject="Subject", within=["Metric"])[-1], 3)
    0.878

    The p-value value is very large, and the test therefore indicates that
    there is no violation of sphericity.

    Now, let's test sphericity for the interaction between the two
    repeated measures factor.

    >>> spher, _, chisq, dof, pval = pg.sphericity(
    ...     data, dv="Performance", subject="Subject", within=["Time", "Metric"]
    ... )
    >>> print(spher, round(chisq, 3), dof, round(pval, 3))
    True 3.763 2 0.152

    Here again, there is no violation of sphericity acccording to Mauchly's
    test.

    Alternatively, we could use a wide-format dataframe with two column
    levels:

    >>> # Pivot from long-format to wide-format
    >>> piv = data.pivot(index="Subject", columns=["Time", "Metric"], values="Performance")
    >>> piv.head()
    Time        Pre                  Post
    Metric  Product Client Action Product Client Action
    Subject
    1            13     12     17      18     30     34
    2            12     19     18       6     18     30
    3            17     19     24      21     31     32
    4            12     25     25      18     39     40
    5            19     27     19      18     28     27

    >>> spher, _, chisq, dof, pval = pg.sphericity(piv)
    >>> print(spher, round(chisq, 3), dof, round(pval, 3))
    True 3.763 2 0.152

    which gives the same output as the long-format dataframe.
    """
    # Wide-format data without missing values, and covariance matrix of the orthonormal contrasts
    data = _wide_rm(data, dv=dv, within=within, subject=subject)
    M = _contrast_cov(data)
    n, d = data.shape[0], M.shape[0]

    # Sphericity is always met with only one degree of freedom (e.g. two repeated measures).
    if d <= 1:
        return True, np.nan, np.nan, 1, 1.0

    if method.lower() == "mauchly":
        W, chi_sq, ddof, pval = _mauchly(M, n - 1, data.shape[1])
    else:
        # Method = JNS. John's statistic is equal to 1 / d under sphericity.
        ddof = (d * (d + 1)) / 2 - 1
        W = np.trace(M @ M) / np.trace(M) ** 2
        chi_sq = 0.5 * n * d**2 * (W - 1 / d)
        pval = scipy.stats.chi2.sf(chi_sq, ddof)

    spher = True if pval > alpha else False

    SpherResults = namedtuple("SpherResults", ["spher", "W", "chi2", "dof", "pval"])
    return SpherResults(spher, W, chi_sq, int(ddof), pval)
