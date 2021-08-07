# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: May 2018
import scipy
import numpy as np
import pandas as pd
from pingouin import (remove_na, remove_rm_na, _check_dataframe,
                      _postprocess_dataframe)

__all__ = ["mad", "madmedianrule", "mwu", "wilcoxon", "kruskal", "friedman",
           "cochran", "harrelldavis"]


def mad(a, normalize=True, axis=0):
    """
    Median Absolute Deviation (MAD) along given axis of an array.

    Parameters
    ----------
    a : array-like
        Input array.
    normalize : boolean.
        If True, scale by a normalization constant :math:`c \\approx  0.67`
        to ensure consistency with the standard deviation for normally
        distributed data.
    axis : int or None, optional
        Axis along which the MAD is computed. Default is 0.
        Can also be None to compute the MAD over the entire array.

    Returns
    -------
    mad : float
        mad = median(abs(a - median(a))) / c

    See also
    --------
    madmedianrule, numpy.std

    Notes
    -----
    The `median absolute deviation
    <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_ (MAD) computes
    the median over the absolute deviations from the median. It is a measure of
    dispersion similar to the standard deviation, but is more robust to
    outliers.

    SciPy 1.3 and higher includes a similar function:
    :py:func:`scipy.stats.median_abs_deviation`.

    Please note that missing values are automatically removed.

    Examples
    --------
    >>> from pingouin import mad
    >>> a = [1.2, 5.4, 3.2, 7.8, 2.5]
    >>> mad(a)
    2.965204437011204

    >>> mad(a, normalize=False)
    2.0

    2D arrays with missing values (axis handling example)

    >>> import numpy as np
    >>> np.random.seed(123)
    >>> w = np.random.normal(size=(5, 10))
    >>> w[3, 2] = np.nan
    >>> mad(w)  # Axis = 0 (default) = iterate over the columns
    array([0.60304023, 2.35057834, 0.90350696, 1.28599837, 1.16024152,
           0.38653752, 1.92564066, 1.2480913 , 0.42580373, 1.69814622])

    >>> mad(w, axis=1)  # Axis = 1 = iterate over the rows
    array([1.32639022, 1.19295036, 1.41198672, 0.78020689, 1.01531254])

    >>> mad(w, axis=None)  # Axis = None = over the entire array
    1.1607762457644006

    Compare with Scipy >= 1.3

    >>> from scipy.stats import median_abs_deviation
    >>> median_abs_deviation(w, scale='normal', axis=None, nan_policy='omit')
    1.1607762457644006
    """
    a = np.asarray(a)
    if axis is None:
        # Calculate the MAD over the entire array
        a = np.ravel(a)
        axis = 0
    c = scipy.stats.norm.ppf(3 / 4.) if normalize else 1
    center = np.apply_over_axes(np.nanmedian, a, axis)
    return np.nanmedian((np.fabs(a - center)) / c, axis=axis)


def madmedianrule(a):
    """Robust outlier detection based on the MAD-median rule.

    Parameters
    ----------
    a : array-like
        Input array. Must be one-dimensional.

    Returns
    -------
    outliers: boolean (same shape as a)
        Boolean array indicating whether each sample is an outlier (True) or
        not (False).

    See also
    --------
    mad

    Notes
    -----
    The MAD-median-rule ([1]_, [2]_) will refer to declaring :math:`X_i`
    an outlier if

    .. math::

        \\frac{\\left | X_i - M \\right |}{\\text{MAD}_{\\text{norm}}} > K,

    where :math:`M` is the median of :math:`X`,
    :math:`\\text{MAD}_{\\text{norm}}` the normalized median absolute deviation
    of :math:`X`, and :math:`K` is the square
    root of the .975 quantile of a :math:`X^2` distribution with one degree
    of freedom, which is roughly equal to 2.24.

    References
    ----------
    .. [1] Hall, P., Welsh, A.H., 1985. Limit theorems for the median
       deviation. Ann. Inst. Stat. Math. 37, 27–36.
       https://doi.org/10.1007/BF02481078

    .. [2] Wilcox, R. R. Introduction to Robust Estimation and Hypothesis
       Testing. (Academic Press, 2011).

    Examples
    --------
    >>> import pingouin as pg
    >>> a = [-1.09, 1., 0.28, -1.51, -0.58, 6.61, -2.43, -0.43]
    >>> pg.madmedianrule(a)
    array([False, False, False, False, False,  True, False, False])
    """
    a = np.asarray(a)
    assert a.ndim == 1, 'Only 1D array / list are supported for this function.'
    k = np.sqrt(scipy.stats.chi2.ppf(0.975, 1))
    return (np.fabs(a - np.median(a)) / mad(a)) > k


def mwu(x, y, alternative='two-sided', **kwargs):
    """Mann-Whitney U Test (= Wilcoxon rank-sum test). It is the non-parametric
    version of the independent T-test.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. ``x`` and ``y`` must be
        independent.
    alternative : string
        Defines the alternative hypothesis, or tail of the test. Must be one of
        "two-sided" (default), "greater" or "less". See :py:func:`scipy.stats.mannwhitneyu` for
        more details.
    **kwargs : dict
        Additional keywords arguments that are passed to :py:func:`scipy.stats.mannwhitneyu`.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'U-val'``: U-value
        * ``'alternative'``: tail of the test
        * ``'p-val'``: p-value
        * ``'RBC'``   : rank-biserial correlation
        * ``'CLES'``  : common language effect size

    See also
    --------
    scipy.stats.mannwhitneyu, wilcoxon, ttest

    Notes
    -----
    The Mann–Whitney U test [1]_ (also called Wilcoxon rank-sum test) is a
    non-parametric test of the null hypothesis that it is equally likely that
    a randomly selected value from one sample will be less than or greater
    than a randomly selected value from a second sample. The test assumes
    that the two samples are independent. This test corrects for ties and by
    default uses a continuity correction (see :py:func:`scipy.stats.mannwhitneyu` for details).

    The rank biserial correlation [2]_ is the difference between
    the proportion of favorable evidence minus the proportion of unfavorable
    evidence.

    The common language effect size is the proportion of pairs where ``x`` is
    higher than ``y``. It was first introduced by McGraw and Wong (1992) [3]_.
    Pingouin uses a brute-force version of the formula given by Vargha and
    Delaney 2000 [4]_:

    .. math:: \\text{CL} = P(X > Y) + .5 \\times P(X = Y)

    The advantage is of this method are twofold. First, the brute-force
    approach pairs each observation of ``x`` to its ``y`` counterpart, and
    therefore does not require normally distributed data. Second, the formula
    takes ties into account and therefore works with ordinal data.

    When tail is ``'less'``, the CLES is then set to :math:`1 - \\text{CL}`,
    which gives the proportion of pairs where ``x`` is *lower* than ``y``.

    References
    ----------
    .. [1] Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of
           two random variables is stochastically larger than the other.
           The annals of mathematical statistics, 50-60.

    .. [2] Kerby, D. S. (2014). The simple difference formula: An approach to
           teaching nonparametric correlation. Comprehensive Psychology,
           3, 11-IT.

    .. [3] McGraw, K. O., & Wong, S. P. (1992). A common language effect size
           statistic. Psychological bulletin, 111(2), 361.

    .. [4] Vargha, A., & Delaney, H. D. (2000). A Critique and Improvement of
        the “CL” Common Language Effect Size Statistics of McGraw and Wong.
        Journal of Educational and Behavioral Statistics: A Quarterly
        Publication Sponsored by the American Educational Research
        Association and the American Statistical Association, 25(2),
        101–132. https://doi.org/10.2307/1165329

    Examples
    --------
    >>> import numpy as np
    >>> import pingouin as pg
    >>> np.random.seed(123)
    >>> x = np.random.uniform(low=0, high=1, size=20)
    >>> y = np.random.uniform(low=0.2, high=1.2, size=20)
    >>> pg.mwu(x, y, alternative='two-sided')
         U-val alternative    p-val    RBC    CLES
    MWU   97.0   two-sided  0.00556  0.515  0.2425

    Compare with SciPy

    >>> import scipy
    >>> scipy.stats.mannwhitneyu(x, y, use_continuity=True, alternative='two-sided')
    MannwhitneyuResult(statistic=97.0, pvalue=0.0055604599321374135)

    One-sided test

    >>> pg.mwu(x, y, alternative='greater')
         U-val alternative     p-val    RBC    CLES
    MWU   97.0     greater  0.997442  0.515  0.2425

    >>> pg.mwu(x, y, alternative='less')
         U-val alternative    p-val    RBC    CLES
    MWU   97.0        less  0.00278  0.515  0.7575

    Passing keyword arguments to :py:func:`scipy.stats.mannwhitneyu`:

    >>> pg.mwu(x, y, alternative='two-sided', method='exact')
         U-val alternative     p-val    RBC    CLES
    MWU   97.0   two-sided  0.004681  0.515  0.2425
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NA
    x, y = remove_na(x, y, paired=False)

    # Check tails
    assert alternative in ['two-sided', 'greater', 'less'], (
        "Alternative must be one of 'two-sided' (default), 'greater' or 'less'.")
    if "tail" in kwargs:
        raise ValueError(
            "Since Pingouin 0.4.0, the 'tail' argument has been renamed to 'alternative'.")
    uval, pval = scipy.stats.mannwhitneyu(x, y, alternative=alternative, **kwargs)

    # Effect size 1: Common Language Effect Size
    # CLES is tail-specific and calculated according to the formula given in
    # Vargha and Delaney 2000 which works with ordinal data.
    diff = x[:, None] - y
    # cles = max((diff < 0).sum(), (diff > 0).sum()) / diff.size
    # Tail = 'greater', with ties set to 0.5
    # Note that tail = 'two-sided' gives same output as tail = 'greater'
    cles = np.where(diff == 0, 0.5, diff > 0).mean()
    cles = 1 - cles if alternative == 'less' else cles

    # Effect size 2: rank biserial correlation (Wendt 1972)
    rbc = 1 - (2 * uval) / diff.size  # diff.size = x.size * y.size

    # Fill output DataFrame
    stats = pd.DataFrame({
        'U-val': uval,
        'alternative': alternative,
        'p-val': pval,
        'RBC': rbc,
        'CLES': cles}, index=['MWU'])
    return _postprocess_dataframe(stats)


def wilcoxon(x, y=None, alternative='two-sided', **kwargs):
    """
    Wilcoxon signed-rank test. It is the non-parametric version of the paired T-test.

    Parameters
    ----------
    x : array_like
        Either the first set of measurements
        (in which case y is the second set of measurements),
        or the differences between two sets of measurements
        (in which case y is not to be specified.) Must be one-dimensional.
    y : array_like
        Either the second set of measurements (if x is the first set of
        measurements), or not specified (if x is the differences between
        two sets of measurements.) Must be one-dimensional.
    alternative : string
        Defines the alternative hypothesis, or tail of the test. Must be one of
        "two-sided" (default), "greater" or "less". See :py:func:`scipy.stats.wilcoxon` for
        more details.
    **kwargs : dict
        Additional keywords arguments that are passed to :py:func:`scipy.stats.wilcoxon`.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'W-val'``: W-value
        * ``'alternative'``: tail of the test
        * ``'p-val'``: p-value
        * ``'RBC'``   : matched pairs rank-biserial correlation (effect size)
        * ``'CLES'``  : common language effect size

    See also
    --------
    scipy.stats.wilcoxon, mwu

    Notes
    -----
    The Wilcoxon signed-rank test [1]_ tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero.

    .. important:: Pingouin automatically applies a continuity correction.
        Therefore, the p-values will be slightly different than
        :py:func:`scipy.stats.wilcoxon` unless ``correction=True`` is
        explicitly passed to the latter.

    In addition to the test statistic and p-values, Pingouin also computes two
    measures of effect size. The matched pairs rank biserial correlation [2]_
    is the simple difference between the proportion of favorable and
    unfavorable evidence; in the case of the Wilcoxon signed-rank test,
    the evidence consists of rank sums (Kerby 2014):

    .. math:: r = f - u

    The common language effect size is the proportion of pairs where ``x`` is
    higher than ``y``. It was first introduced by McGraw and Wong (1992) [3]_.
    Pingouin uses a brute-force version of the formula given by Vargha and
    Delaney 2000 [4]_:

    .. math:: \\text{CL} = P(X > Y) + .5 \\times P(X = Y)

    The advantage is of this method are twofold. First, the brute-force
    approach pairs each observation of ``x`` to its ``y`` counterpart, and
    therefore does not require normally distributed data. Second, the formula
    takes ties into account and therefore works with ordinal data.

    When tail is ``'less'``, the CLES is then set to :math:`1 - \\text{CL}`,
    which gives the proportion of pairs where ``x`` is *lower* than ``y``.

    References
    ----------
    .. [1] Wilcoxon, F. (1945). Individual comparisons by ranking methods.
           Biometrics bulletin, 1(6), 80-83.

    .. [2] Kerby, D. S. (2014). The simple difference formula: An approach to
           teaching nonparametric correlation. Comprehensive Psychology,
           3, 11-IT.

    .. [3] McGraw, K. O., & Wong, S. P. (1992). A common language effect size
           statistic. Psychological bulletin, 111(2), 361.

    .. [4] Vargha, A., & Delaney, H. D. (2000). A Critique and Improvement of
           the “CL” Common Language Effect Size Statistics of McGraw and Wong.
           Journal of Educational and Behavioral Statistics: A Quarterly
           Publication Sponsored by the American Educational Research
           Association and the American Statistical Association, 25(2),
           101–132. https://doi.org/10.2307/1165329

    Examples
    --------
    Wilcoxon test on two related samples.

    >>> import numpy as np
    >>> import pingouin as pg
    >>> x = np.array([20, 22, 19, 20, 22, 18, 24, 20, 19, 24, 26, 13])
    >>> y = np.array([38, 37, 33, 29, 14, 12, 20, 22, 17, 25, 26, 16])
    >>> pg.wilcoxon(x, y, alternative='two-sided')
              W-val alternative     p-val       RBC      CLES
    Wilcoxon   20.5   two-sided  0.285765 -0.378788  0.395833

    Same but using pre-computed differences. However, the CLES effect size
    cannot be computed as it requires the raw data.

    >>> pg.wilcoxon(x - y)
              W-val alternative     p-val       RBC  CLES
    Wilcoxon   20.5   two-sided  0.285765 -0.378788   NaN

    Compare with SciPy

    >>> import scipy
    >>> scipy.stats.wilcoxon(x, y)
    WilcoxonResult(statistic=20.5, pvalue=0.2661660677806492)

    The p-value is not exactly similar to Pingouin. This is because Pingouin automatically applies
    a continuity correction. Disabling it gives the same p-value as scipy:

    >>> pg.wilcoxon(x, y, alternative='two-sided', correction=False)
              W-val alternative     p-val       RBC      CLES
    Wilcoxon   20.5   two-sided  0.266166 -0.378788  0.395833

    One-sided test

    >>> pg.wilcoxon(x, y, alternative='greater')
              W-val alternative     p-val       RBC      CLES
    Wilcoxon   20.5     greater  0.876244 -0.378788  0.395833

    >>> pg.wilcoxon(x, y, alternative='less')
              W-val alternative     p-val       RBC      CLES
    Wilcoxon   20.5        less  0.142883 -0.378788  0.604167
    """
    x = np.asarray(x)
    if y is not None:
        y = np.asarray(y)
        x, y = remove_na(x, y, paired=True)  # Remove NA
    else:
        x = x[~np.isnan(x)]

    # Check tails
    assert alternative in ['two-sided', 'greater', 'less'], (
        "Alternative must be one of 'two-sided' (default), 'greater' or 'less'.")
    if "tail" in kwargs:
        raise ValueError(
            "Since Pingouin 0.4.0, the 'tail' argument has been renamed to 'alternative'.")

    # Compute test
    if "correction" not in kwargs:
        kwargs["correction"] = True
    wval, pval = scipy.stats.wilcoxon(x=x, y=y, alternative=alternative, **kwargs)

    # Effect size 1: Common Language Effect Size
    # Since Pingouin v0.3.5, CLES is tail-specific and calculated
    # according to the formula given in Vargha and Delaney 2000 which
    # works with ordinal data.
    if y is not None:
        diff = x[:, None] - y
        # cles = max((diff < 0).sum(), (diff > 0).sum()) / diff.size
        # alternative = 'greater', with ties set to 0.5
        # Note that alternative = 'two-sided' gives same output as alternative = 'greater'
        cles = np.where(diff == 0, 0.5, diff > 0).mean()
        cles = 1 - cles if alternative == 'less' else cles
    else:
        # CLES cannot be computed if y is None
        cles = np.nan

    # Effect size 2: matched-pairs rank biserial correlation (Kerby 2014)
    if y is not None:
        d = x - y
        d = d[d != 0]
    else:
        d = x[x != 0]
    r = scipy.stats.rankdata(abs(d))
    rsum = r.sum()
    r_plus = np.sum((d > 0) * r)
    r_minus = np.sum((d < 0) * r)
    rbc = r_plus / rsum - r_minus / rsum

    # Fill output DataFrame
    stats = pd.DataFrame({
        'W-val': wval,
        'alternative': alternative,
        'p-val': pval,
        'RBC': rbc,
        'CLES': cles}, index=['Wilcoxon'])
    return _postprocess_dataframe(stats)


def kruskal(data=None, dv=None, between=None, detailed=False):
    """Kruskal-Wallis H-test for independent samples.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame
    dv : string
        Name of column containing the dependent variable.
    between : string
        Name of column containing the between factor.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'H'``: The Kruskal-Wallis H statistic, corrected for ties
        * ``'p-unc'``: Uncorrected p-value
        * ``'dof'``: degrees of freedom

    Notes
    -----
    The Kruskal-Wallis H-test tests the null hypothesis that the population
    median of all of the groups are equal. It is a non-parametric version of
    ANOVA. The test works on 2 or more independent samples, which may have
    different sizes.

    Due to the assumption that H has a chi square distribution, the number of
    samples in each group must not be too small. A typical rule is that each
    sample must have at least 5 measurements.

    NaN values are automatically removed.

    Examples
    --------
    Compute the Kruskal-Wallis H-test for independent samples.

    >>> from pingouin import kruskal, read_dataset
    >>> df = read_dataset('anova')
    >>> kruskal(data=df, dv='Pain threshold', between='Hair color')
                 Source  ddof1         H     p-unc
    Kruskal  Hair color      3  10.58863  0.014172
    """
    # Check data
    _check_dataframe(dv=dv, between=between, data=data,
                     effects='between')

    # Remove NaN values
    data = data[[dv, between]].dropna()

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)

    # Extract number of groups and total sample size
    n_groups = data[between].nunique()
    n = data[dv].size

    # Rank data, dealing with ties appropriately
    data['rank'] = scipy.stats.rankdata(data[dv])

    # Find the total of rank per groups
    grp = data.groupby(between, observed=True)['rank']
    sum_rk_grp = grp.sum().to_numpy()
    n_per_grp = grp.count().to_numpy()

    # Calculate chi-square statistic (H)
    H = (12 / (n * (n + 1)) * np.sum(sum_rk_grp**2 / n_per_grp)) - 3 * (n + 1)

    # Correct for ties
    H /= scipy.stats.tiecorrect(data['rank'].to_numpy())

    # Calculate DOF and p-value
    ddof1 = n_groups - 1
    p_unc = scipy.stats.chi2.sf(H, ddof1)

    # Create output dataframe
    stats = pd.DataFrame({'Source': between,
                          'ddof1': ddof1,
                          'H': H,
                          'p-unc': p_unc,
                          }, index=['Kruskal'])
    return _postprocess_dataframe(stats)


def friedman(data=None, dv=None, within=None, subject=None, method='chisq'):
    """Friedman test for repeated measurements.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame
    dv : string
        Name of column containing the dependent variable.
    within : string
        Name of column containing the within-subject factor.
    subject : string
        Name of column containing the subject identifier.
    method : string
        Statistical test to perform. Must be ``'chisq'`` (chi-square test) or ``'f'`` (F test).
        See notes below for explanation.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'W'``: Kendall's coefficient of concordance, corrected for ties

        If ``method='chisq'``

            * ``'Q'``: The Friedman chi-square statistic, corrected for ties
            * ``'dof'``: degrees of freedom
            * ``'p-unc'``: Uncorrected p-value of the chi squared test


        If ``method='f'``

            * ``'F'``: The Friedman F statistic, corrected for ties
            * ``'dof1'``: degrees of freedom of the numerator
            * ``'dof2'``: degrees of freedom of the denominator
            * ``'p-unc'``: Uncorrected p-value of the F test

    Notes
    -----
    The Friedman test is used for one-way repeated measures ANOVA by ranks.

    Data are expected to be in long-format.

    Note that if the dataset contains one or more other within subject
    factors, an automatic collapsing to the mean is applied on the dependent
    variable (same behavior as the ezANOVA R package). As such, results can
    differ from those of JASP. If you can, always double-check the results.

    NaN values are automatically removed.

    The Friedman test is equivalent to the test of significance of Kendalls's
    coefficient of concordance (Kendall's W). Most commonly a Q statistic,
    which has asymptotical chi-squared distribution, is computed and used for
    testing. However, in [1]_ they showed the chi-squared test to be overly
    conservative for small numbers of samples and repeated measures. Instead
    they recommend the F test, which has the correct size and behaves like a
    permutation test, but is computationaly much easier.

    References
    ----------
    .. [1] Marozzi, M. (2014). Testing for concordance between several
           criteria. Journal of Statistical Computation and Simulation,
           84(9), 1843–1850. https://doi.org/10.1080/00949655.2013.766189

    Examples
    --------
    Compute the Friedman test for repeated measurements.

    >>> from pingouin import friedman, read_dataset
    >>> df = read_dataset('rm_anova')
    >>> friedman(data=df, dv='DesireToKill', within='Disgustingness',
    ...          subject='Subject')
                      Source         W  ddof1         Q     p-unc
    Friedman  Disgustingness  0.099224      1  9.227848  0.002384


    This time we will use the F test method.

    >>> from pingouin import friedman, read_dataset
    >>> df = read_dataset('rm_anova')
    >>> friedman(data=df, dv='DesireToKill', within='Disgustingness',
    ...          subject='Subject', method='f')
                      Source         W     ddof1      ddof2         F     p-unc
    Friedman  Disgustingness  0.099224  0.978495  90.021505  10.13418  0.002138

    We can see, compared to the previous example, that the p-value is slightly
    lower. This is expected, since the F test is more powerful (see Notes).
    """
    # Check data
    _check_dataframe(dv=dv, within=within, data=data, subject=subject,
                     effects='within')

    # Convert Categorical columns to string
    # This is important otherwise all the groupby will return different results
    # unless we specify .groupby(..., observed = True).
    for c in [subject, within]:
        if data[c].dtype.name == 'category':
            data[c] = data[c].astype(str)

    # Collapse to the mean
    data = data.groupby([subject, within]).mean().reset_index()

    # Remove NaN
    if data[dv].isnull().any():
        data = remove_rm_na(dv=dv, within=within, subject=subject,
                            data=data[[subject, within, dv]])

    # Extract number of groups and total sample size
    grp = data.groupby(within)[dv]
    rm = list(data[within].unique())
    k = len(rm)
    X = np.array([grp.get_group(r).to_numpy() for r in rm]).T
    n = X.shape[0]

    # Rank per subject
    ranked = np.zeros(X.shape)
    for i in range(n):
        ranked[i] = scipy.stats.rankdata(X[i, :])

    ssbn = (ranked.sum(axis=0)**2).sum()

    # Correction for ties
    ties = 0
    for i in range(n):
        replist, repnum = scipy.stats.find_repeats(X[i])
        for t in repnum:
            ties += t * (t * t - 1)

    # Compute Kendall's W corrected for ties
    W = (12 * ssbn - 3 * n * n * k * (k + 1) * (k + 1)) / (n * n * k * (k - 1) * (k + 1) - n * ties)

    if method == 'chisq':
        # Compute the Q statistic
        Q = n * (k - 1) * W

        # Approximate the p-value
        ddof1 = k - 1
        p_unc = scipy.stats.chi2.sf(Q, ddof1)

        # Create output dataframe
        stats = pd.DataFrame({'Source': within,
                              'W': W,
                              'ddof1': ddof1,
                              'Q': Q,
                              'p-unc': p_unc,
                              }, index=['Friedman'])
    elif method == 'f':
        # Compute the F statistic
        F = W * (n - 1) / (1 - W)

        # Approximate the p-value
        ddof1 = k - 1 - 2 / n
        ddof2 = (n - 1) * ddof1
        p_unc = scipy.stats.f.sf(F, ddof1, ddof2)

        # Create output dataframe
        stats = pd.DataFrame({'Source': within,
                              'W': W,
                              'ddof1': ddof1,
                              'ddof2': ddof2,
                              'F': F,
                              'p-unc': p_unc,
                              }, index=['Friedman'])

    return _postprocess_dataframe(stats)


def cochran(data=None, dv=None, within=None, subject=None):
    """Cochran Q test. A special case of the Friedman test when the dependent
    variable is binary.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame
    dv : string
        Name of column containing the binary dependent variable.
    within : string
        Name of column containing the within-subject factor.
    subject : string
        Name of column containing the subject identifier.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'Q'``: The Cochran Q statistic
        * ``'p-unc'``: Uncorrected p-value
        * ``'dof'``: degrees of freedom

    Notes
    -----
    The Cochran Q test [1]_ is a non-parametric test for ANOVA with repeated
    measures where the dependent variable is binary.

    Data are expected to be in long-format. NaN are automatically removed
    from the data.

    The Q statistics is defined as:

    .. math:: Q = \\frac{(r-1)(r\\sum_j^rx_j^2-N^2)}{rN-\\sum_i^nx_i^2}

    where :math:`N` is the total sum of all observations, :math:`j=1,...,r`
    where :math:`r` is the number of repeated measures, :math:`i=1,...,n` where
    :math:`n` is the number of observations per condition.

    The p-value is then approximated using a chi-square distribution with
    :math:`r-1` degrees of freedom:

    .. math:: Q \\sim \\chi^2(r-1)

    References
    ----------
    .. [1] Cochran, W.G., 1950. The comparison of percentages in matched
       samples. Biometrika 37, 256–266.
       https://doi.org/10.1093/biomet/37.3-4.256

    Examples
    --------
    Compute the Cochran Q test for repeated measurements.

    >>> from pingouin import cochran, read_dataset
    >>> df = read_dataset('cochran')
    >>> cochran(data=df, dv='Energetic', within='Time', subject='Subject')
            Source  dof         Q     p-unc
    cochran   Time    2  6.705882  0.034981
    """
    # Check data
    _check_dataframe(dv=dv, within=within, data=data, subject=subject,
                     effects='within')

    # Convert Categorical columns to string
    # This is important otherwise all the groupby will return different results
    # unless we specify .groupby(..., observed = True).
    for c in [subject, within]:
        if data[c].dtype.name == 'category':
            data[c] = data[c].astype(str)

    # Remove NaN
    if data[dv].isnull().any():
        data = remove_rm_na(dv=dv, within=within, subject=subject,
                            data=data[[subject, within, dv]])

    # Groupby and extract size
    grp = data.groupby(within)[dv]
    grp_s = data.groupby(subject)[dv]
    k = data[within].nunique()
    dof = k - 1
    # n = grp.count().unique()[0]

    # Q statistic and p-value
    q = (dof * (k * np.sum(grp.sum()**2) - grp.sum().sum()**2)) / \
        (k * grp.sum().sum() - np.sum(grp_s.sum()**2))
    p_unc = scipy.stats.chi2.sf(q, dof)

    # Create output dataframe
    stats = pd.DataFrame({'Source': within,
                          'dof': dof,
                          'Q': q,
                          'p-unc': p_unc,
                          }, index=['cochran'])

    return _postprocess_dataframe(stats)


def harrelldavis(x, quantile=0.5, axis=-1):
    """Harrell-Davis robust estimate of the :math:`q^{th}` quantile(s) of the
    data.

    .. versionadded:: 0.2.9

    Parameters
    ----------
    x : array_like
        Data, must be a one or two-dimensional vector.
    quantile : float or array_like
        Quantile or sequence of quantiles to compute, must be between 0 and 1.
        Default is ``0.5``.
    axis : int
        Axis along which the MAD is computed. Default is the last axis (-1).
        Can be either 0, 1 or -1.

    Returns
    -------
    y : float or array_like
        The estimated quantile(s). If ``quantile`` is a single quantile, will
        return a float, otherwise will compute each quantile separately and
        returns an array of floats.

    Notes
    -----
    The Harrell-Davis method [1]_ estimates the :math:`q^{th}` quantile by a
    linear combination of  the  order statistics. Results have been tested
    against a Matlab implementation [2]_. Note that this method is also
    used to measure the confidence intervals of the difference between
    quantiles of two groups, as implemented in the shift function [3]_.

    See Also
    --------
    plot_shift

    References
    ----------
    .. [1] Frank E. Harrell, C. E. Davis, A new distribution-free quantile
       estimator, Biometrika, Volume 69, Issue 3, December 1982, Pages
       635–640, https://doi.org/10.1093/biomet/69.3.635

    .. [2] https://github.com/GRousselet/matlab_stats/blob/master/hd.m

    .. [3] Rousselet, G. A., Pernet, C. R. and Wilcox, R. R. (2017). Beyond
       differences in means: robust graphical methods to compare two groups
       in neuroscience. Eur J Neurosci, 46: 1738-1748.
       https://doi.org/doi:10.1111/ejn.13610

    Examples
    --------
    Estimate the 0.5 quantile (i.e median) of 100 observation picked from a
    normal distribution with zero mean and unit variance.

    >>> import numpy as np
    >>> import pingouin as pg
    >>> np.random.seed(123)
    >>> x = np.random.normal(0, 1, 100)
    >>> round(pg.harrelldavis(x, quantile=0.5), 4)
    -0.0499

    Several quantiles at once

    >>> pg.harrelldavis(x, quantile=[0.25, 0.5, 0.75])
    array([-0.84133224, -0.04991657,  0.95897233])

    On the last axis of a 2D vector (default)

    >>> np.random.seed(123)
    >>> x = np.random.normal(0, 1, (10, 100))
    >>> pg.harrelldavis(x, quantile=[0.25, 0.5, 0.75])
    array([[-0.84133224, -0.52346777, -0.81801193, -0.74611216, -0.64928321,
            -0.48565262, -0.64332799, -0.8178394 , -0.70058282, -0.73088088],
           [-0.04991657,  0.02932655, -0.08905073, -0.1860034 ,  0.06970415,
             0.15129817,  0.00430958, -0.13784786, -0.08648077, -0.14407123],
           [ 0.95897233,  0.49543002,  0.57712236,  0.48620599,  0.85899005,
             0.7903462 ,  0.76558585,  0.62528436,  0.60421847,  0.52620286]])

    On the first axis

    >>> pg.harrelldavis(x, quantile=[0.5], axis=0).shape
    (100,)
    """
    x = np.asarray(x)
    assert x.ndim <= 2, 'Only 1D or 2D array are supported for this function.'
    assert axis in [0, 1, -1], 'Axis must be 0, 1 or -1.'

    # Sort the input array
    x = np.sort(x, axis=axis)

    n = x.shape[axis]
    vec = np.arange(n)
    if isinstance(quantile, float):
        quantile = [quantile]

    y = []
    for q in quantile:
        # Harrell-Davis estimate of the qth quantile
        m1 = (n + 1) * q
        m2 = (n + 1) * (1 - q)
        w = (scipy.stats.beta.cdf((vec + 1) / n, m1, m2) -
             scipy.stats.beta.cdf((vec) / n, m1, m2))
        if axis != 0:
            y.append((w * x).sum(axis))
        else:
            y.append((w[..., None] * x).sum(axis))  # Store results

    if len(y) == 1:
        y = y[0]  # Return a float instead of a list if n quantile is 1
    else:
        y = np.array(y)
    return y
