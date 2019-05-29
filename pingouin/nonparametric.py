# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: May 2018
import scipy
import numpy as np
import pandas as pd
from pingouin import remove_na, remove_rm_na, _check_dataframe, _export_table

__all__ = ["mad", "madmedianrule", "mwu", "wilcoxon", "kruskal", "friedman",
           "cochran"]


def mad(a, normalize=True, axis=0):
    """
    Median Absolute Deviation along given axis of an array.

    Parameters
    ----------
    a : array-like
        Input array.
    normalize : boolean.
        If True, scale by a normalization constant (~0.67)
    axis : int, optional
        The defaul is 0. Can also be None.

    Returns
    -------
    mad : float
        mad = median(abs(a - median(a))) / c

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Median_absolute_deviation

    Examples
    --------
    >>> from pingouin import mad
    >>> a = [1.2, 5.4, 3.2, 7.8, 2.5]
    >>> mad(a)
    2.965204437011204

    >>> mad(a, normalize=False)
    2.0
    """
    a = np.asarray(a)
    c = scipy.stats.norm.ppf(3 / 4.) if normalize else 1
    center = np.apply_over_axes(np.median, a, axis)
    return np.median((np.fabs(a - center)) / c, axis=axis)


def madmedianrule(a):
    """Outlier detection based on the MAD-median rule.

    Parameters
    ----------
    a : array-like
        Input array.

    Returns
    -------
    outliers: boolean (same shape as a)
        Boolean array indicating whether each sample is an outlier (True) or
        not (False).

    References
    ----------
    .. [1] Hall, P., Welsh, A.H., 1985. Limit theorems for the median
       deviation. Ann. Inst. Stat. Math. 37, 27–36.
       https://doi.org/10.1007/BF02481078

    Examples
    --------
    >>> from pingouin import madmedianrule
    >>> a = [-1.09, 1., 0.28, -1.51, -0.58, 6.61, -2.43, -0.43]
    >>> madmedianrule(a)
    array([False, False, False, False, False,  True, False, False])
    """
    a = np.asarray(a)
    k = np.sqrt(scipy.stats.chi2.ppf(0.975, 1))
    return (np.fabs(a - np.median(a)) / mad(a)) > k


def mwu(x, y, tail='two-sided'):
    """Mann-Whitney U Test (= Wilcoxon rank-sum test). It is the non-parametric
    version of the independent T-test.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.

    Returns
    -------
    stats : pandas DataFrame
        Test summary ::

        'U-val' : U-value
        'p-val' : p-value
        'RBC'   : rank-biserial correlation (effect size)
        'CLES'  : common language effect size

    Notes
    -----
    mwu tests the hypothesis that data in x and y are samples from continuous
    distributions with equal medians. The test assumes that x and y
    are independent. This test corrects for ties and by default
    uses a continuity correction (see :py:func:`scipy.stats.mannwhitneyu`
    for details).

    The rank biserial correlation is the difference between the proportion of
    favorable evidence minus the proportion of unfavorable evidence
    (see Kerby 2014).

    The common language effect size is the probability (from 0 to 1) that a
    randomly selected observation from the first sample will be greater than a
    randomly selected observation from the second sample.

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

    Examples
    --------
    >>> import numpy as np
    >>> import pingouin as pg
    >>> np.random.seed(123)
    >>> x = np.random.uniform(low=0, high=1, size=20)
    >>> y = np.random.uniform(low=0.2, high=1.2, size=20)
    >>> pg.mwu(x, y, tail='two-sided')
         U-val    p-val    RBC   CLES
    MWU   97.0  0.00556  0.515  0.758

    Compare with SciPy

    >>> import scipy
    >>> scipy.stats.mannwhitneyu(x, y, use_continuity=True,
    ...                          alternative='two-sided')
    MannwhitneyuResult(statistic=97.0, pvalue=0.0055604599321374135)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NA
    x, y = remove_na(x, y, paired=False)

    # Compute test
    if tail == 'one-sided':
        tail = 'less' if np.median(x) < np.median(y) else 'greater'
    uval, pval = scipy.stats.mannwhitneyu(x, y, use_continuity=True,
                                          alternative=tail)

    # Effect size 1: common language effect size (McGraw and Wong 1992)
    diff = x[:, None] - y
    cles = max((diff < 0).sum(), (diff > 0).sum()) / diff.size

    # Effect size 2: rank biserial correlation (Wendt 1972)
    rbc = 1 - (2 * uval) / diff.size  # diff.size = x.size * y.size

    # Fill output DataFrame
    stats = pd.DataFrame({}, index=['MWU'])
    stats['U-val'] = round(uval, 3)
    stats['p-val'] = pval
    stats['RBC'] = round(rbc, 3)
    stats['CLES'] = round(cles, 3)

    col_order = ['U-val', 'p-val', 'RBC', 'CLES']
    stats = stats.reindex(columns=col_order)
    return stats


def wilcoxon(x, y, tail='two-sided'):
    """Wilcoxon signed-rank test. It is the non-parametric version of the
    paired T-test.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be related (e.g
        repeated measures).
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.

    Returns
    -------
    stats : pandas DataFrame
        Test summary ::

        'W-val' : W-value
        'p-val' : p-value
        'RBC'   : matched pairs rank-biserial correlation (effect size)
        'CLES'  : common language effect size

    Notes
    -----
    The Wilcoxon signed-rank test tests the null hypothesis that two related
    paired samples come from the same distribution.
    A continuity correction is applied by default
    (see :py:func:`scipy.stats.wilcoxon` for details).

    The rank biserial correlation is the difference between the proportion of
    favorable evidence minus the proportion of unfavorable evidence
    (see Kerby 2014).

    The common language effect size is the probability (from 0 to 1) that a
    randomly selected observation from the first sample will be greater than a
    randomly selected observation from the second sample.

    .. warning :: Versions of Pingouin below 0.2.6 gave wrong p-values for
        the Wilcoxon test. This issue has been resolved in
        Pingouin>=0.2.6. Make sure to always use the latest release.

    References
    ----------
    .. [1] Wilcoxon, F. (1945). Individual comparisons by ranking methods.
           Biometrics bulletin, 1(6), 80-83.

    .. [2] Kerby, D. S. (2014). The simple difference formula: An approach to
           teaching nonparametric correlation. Comprehensive Psychology,
           3, 11-IT.

    .. [3] McGraw, K. O., & Wong, S. P. (1992). A common language effect size
           statistic. Psychological bulletin, 111(2), 361.

    Examples
    --------
    1. Wilcoxon test on two related samples.

    >>> import numpy as np
    >>> import pingouin as pg
    >>> x = [20, 22, 19, 20, 22, 18, 24, 20, 19, 24, 26, 13]
    >>> y = [38, 37, 33, 29, 14, 12, 20, 22, 17, 25, 26, 16]
    >>> pg.wilcoxon(x, y, tail='two-sided')
              W-val     p-val    RBC   CLES
    Wilcoxon   20.5  0.285765  0.333  0.583

    Compare with SciPy

    >>> import scipy
    >>> scipy.stats.wilcoxon(x, y, correction=True)
    WilcoxonResult(statistic=20.5, pvalue=0.2857652190231508)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NA
    x, y = remove_na(x, y, paired=True)

    # Compute test
    # TODO: scipy 1.3.0 includes an alternative arg as for mannwhitney
    # For now keeping this way to keep the compatibility with previous scipy
    wval, pval = scipy.stats.wilcoxon(x, y, zero_method='wilcox',
                                      correction=True)
    pval = pval * .5 if tail == 'one-sided' else pval

    # Effect size 1: common language effect size (McGraw and Wong 1992)
    diff = x[:, None] - y
    cles = max((diff < 0).sum(), (diff > 0).sum()) / diff.size

    # Effect size 2: matched-pairs rank biserial correlation (Kerby 2014)
    rank = np.arange(x.size, 0, -1)
    rsum = rank.sum()
    fav = rank[np.sign(y - x) > 0].sum()
    unfav = rank[np.sign(y - x) < 0].sum()
    rbc = fav / rsum - unfav / rsum

    # Fill output DataFrame
    stats = pd.DataFrame({}, index=['Wilcoxon'])
    stats['W-val'] = round(wval, 3)
    stats['p-val'] = pval
    stats['RBC'] = round(rbc, 3)
    stats['CLES'] = round(cles, 3)

    col_order = ['W-val', 'p-val', 'RBC', 'CLES']
    stats = stats.reindex(columns=col_order)
    return stats


def kruskal(dv=None, between=None, data=None, detailed=False,
            export_filename=None):
    """Kruskal-Wallis H-test for independent samples.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    between : string
        Name of column containing the between factor.
    data : pandas DataFrame
        DataFrame
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    stats : DataFrame
        Test summary ::

        'H' : The Kruskal-Wallis H statistic, corrected for ties
        'p-unc' : Uncorrected p-value
        'dof' : degrees of freedom

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
    >>> kruskal(dv='Pain threshold', between='Hair color', data=df)
                 Source  ddof1       H     p-unc
    Kruskal  Hair color      3  10.589  0.014172
    """
    # Check data
    _check_dataframe(dv=dv, between=between, data=data,
                     effects='between')

    # Remove NaN values
    data = data.dropna()

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)

    # Extract number of groups and total sample size
    groups = list(data[between].unique())
    n_groups = len(groups)
    n = data[dv].size

    # Rank data, dealing with ties appropriately
    data['rank'] = scipy.stats.rankdata(data[dv])

    # Find the total of rank per groups
    grp = data.groupby(between)['rank']
    sum_rk_grp = grp.sum().values
    n_per_grp = grp.count().values

    # Calculate chi-square statistic (H)
    H = (12 / (n * (n + 1)) * np.sum(sum_rk_grp**2 / n_per_grp)) - 3 * (n + 1)

    # Correct for ties
    H /= scipy.stats.tiecorrect(data['rank'].values)

    # Calculate DOF and p-value
    ddof1 = n_groups - 1
    p_unc = scipy.stats.chi2.sf(H, ddof1)

    # Create output dataframe
    stats = pd.DataFrame({'Source': between,
                          'ddof1': ddof1,
                          'H': np.round(H, 3),
                          'p-unc': p_unc,
                          }, index=['Kruskal'])

    col_order = ['Source', 'ddof1', 'H', 'p-unc']

    stats = stats.reindex(columns=col_order)
    stats.dropna(how='all', axis=1, inplace=True)

    # Export to .csv
    if export_filename is not None:
        _export_table(stats, export_filename)
    return stats


def friedman(dv=None, within=None, subject=None, data=None,
             export_filename=None):
    """Friedman test for repeated measurements.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    within : string
        Name of column containing the within-subject factor.
    subject : string
        Name of column containing the subject identifier.
    data : pandas DataFrame
        DataFrame
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    stats : DataFrame
        Test summary ::

        'Q' : The Friedman Q statistic, corrected for ties
        'p-unc' : Uncorrected p-value
        'dof' : degrees of freedom

    Notes
    -----
    The Friedman test is used for one-way repeated measures ANOVA by ranks.

    Data are expected to be in long-format.

    Note that if the dataset contains one or more other within subject
    factors, an automatic collapsing to the mean is applied on the dependant
    variable (same behavior as the ezANOVA R package). As such, results can
    differ from those of JASP. If you can, always double-check the results.

    Due to the assumption that the test statistic has a chi squared
    distribution, the p-value is only reliable for n > 10 and more than 6
    repeated measurements.

    NaN values are automatically removed.

    Examples
    --------
    Compute the Friedman test for repeated measurements.

    >>> from pingouin import friedman, read_dataset
    >>> df = read_dataset('rm_anova')
    >>> friedman(dv='DesireToKill', within='Disgustingness',
    ...          subject='Subject', data=df)
                      Source  ddof1      Q     p-unc
    Friedman  Disgustingness      1  9.228  0.002384
    """
    # Check data
    _check_dataframe(dv=dv, within=within, data=data, subject=subject,
                     effects='within')

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
    X = np.array([grp.get_group(r).values for r in rm]).T
    n = X.shape[0]

    # Rank per subject
    ranked = np.zeros(X.shape)
    for i in range(n):
        ranked[i] = scipy.stats.rankdata(X[i, :])

    ssbn = (ranked.sum(axis=0)**2).sum()

    # Compute the test statistic
    Q = (12 / (n * k * (k + 1))) * ssbn - 3 * n * (k + 1)

    # Correct for ties
    ties = 0
    for i in range(n):
        replist, repnum = scipy.stats.find_repeats(X[i])
        for t in repnum:
            ties += t * (t * t - 1)

    c = 1 - ties / float(k * (k * k - 1) * n)
    Q /= c

    # Approximate the p-value
    ddof1 = k - 1
    p_unc = scipy.stats.chi2.sf(Q, ddof1)

    # Create output dataframe
    stats = pd.DataFrame({'Source': within,
                          'ddof1': ddof1,
                          'Q': np.round(Q, 3),
                          'p-unc': p_unc,
                          }, index=['Friedman'])

    col_order = ['Source', 'ddof1', 'Q', 'p-unc']

    stats = stats.reindex(columns=col_order)
    stats.dropna(how='all', axis=1, inplace=True)

    # Export to .csv
    if export_filename is not None:
        _export_table(stats, export_filename)
    return stats


def cochran(dv=None, within=None, subject=None, data=None,
            export_filename=None):
    """Cochran Q test. Special case of the Friedman test when the dependant
    variable is binary.

    Parameters
    ----------
    dv : string
        Name of column containing the binary dependant variable.
    within : string
        Name of column containing the within-subject factor.
    subject : string
        Name of column containing the subject identifier.
    data : pandas DataFrame
        DataFrame
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    stats : DataFrame
        Test summary ::

        'Q' : The Cochran Q statistic
        'p-unc' : Uncorrected p-value
        'dof' : degrees of freedom

    Notes
    -----
    The Cochran Q Test is a non-parametric test for ANOVA with repeated
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
    >>> cochran(dv='Energetic', within='Time', subject='Subject', data=df)
            Source  dof      Q     p-unc
    cochran   Time    2  6.706  0.034981
    """
    # Check data
    _check_dataframe(dv=dv, within=within, data=data, subject=subject,
                     effects='within')

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
                          'Q': np.round(q, 3),
                          'p-unc': p_unc,
                          }, index=['cochran'])

    # Export to .csv
    if export_filename is not None:
        _export_table(stats, export_filename)
    return stats
