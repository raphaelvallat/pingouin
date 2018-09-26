# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: May 2018
import numpy as np
import pandas as pd
from pingouin import _remove_na, _remove_rm_na, _check_dataframe, _export_table

__all__ = ["mad", "madmedianrule", "mwu", "wilcoxon", "kruskal", "friedman",
           "cochran"]


def mad(a, normalize=True, axis=0):
    """
    Median Absolute Deviation along given axis of an array

    See https://en.wikipedia.org/wiki/Median_absolute_deviation

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

    Examples
    --------

        >>>  a = [1.2, 5.4, 3.2, 7.8, 2.5]
        >>> mad(a)
            2.965
        >>> mad(a, normalize=False)
            2.0
    """
    from scipy.stats import norm
    c = norm.ppf(3 / 4.) if normalize else 1
    return np.median(np.abs(a - np.median(a)) / c, axis=axis)


def madmedianrule(a):
    """Outlier detection based on the MAD-median rule described in
    Wilcox 2012.

    Parameters
    ----------
    a : array-like
        Input array.

    Returns
    -------
    outliers: boolean (same shape as a)
        Boolean array indicating whether each sample is an outlier (True) or
        not (False).

    Examples
    --------

        >>> a = [-1.09, 1., 0.28, -1.51, -0.58, 6.61, -2.43, -0.43]
        >>> madmedianrule(a)
            array([False, False, False, False, False, True, False, False])
    """
    from scipy.stats import chi2
    k = np.sqrt(chi2.ppf(0.975, 1))
    return (np.abs(a - np.median(a)) / mad(a)) > k


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
    uses a continuity correction.

    Examples
    --------
    1. Compare the medians of two independent samples.

        >>> import numpy as np
        >>> from pingouin import mwu
        >>> np.random.seed(123)
        >>> x = np.random.uniform(low=0, high=1, size=20)
        >>> y = np.random.uniform(low=0.2, high=1.2, size=20)
        >>> print("Medians = %.2f - %.2f" % (np.median(x), np.median(y)))
        >>> mwu(x, y, tail='two-sided')
            U-val   p-val     RBC    CLES
            97.0    0.006    0.51    0.75
    """
    from scipy.stats import mannwhitneyu
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NA
    x, y = _remove_na(x, y, paired=False)
    nx = x.size
    ny = y.size

    # Compute test
    if tail == 'one-sided':
        tail = 'less' if np.median(x) < np.median(y) else 'greater'
    uval, pval = mannwhitneyu(x, y, use_continuity=True, alternative=tail)

    # Effect size 1: common language effect size (McGraw and Wong 1992)
    c = np.array([(a, b) for a in x for b in y])
    num = max((c[:, 0] < c[:, 1]).sum(), (c[:, 0] > c[:, 1]).sum())
    cles = num / (nx * ny)

    # Effect size 2: rank biserial correlation (Wendt 1972)
    rbc = 1 - (2 * uval) / (nx * ny)

    # Fill output DataFrame
    stats = pd.DataFrame({}, index=['MWU'])
    stats['U-val'] = uval.round(3)
    stats['p-val'] = pval
    stats['RBC'] = rbc
    stats['CLES'] = cles

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
    A continuity correction is applied by default.


    Examples
    --------
    1. Wilcoxon test on two related samples.

        >>> import numpy as np
        >>> from pingouin import wilcoxon
        >>> x = [20, 22, 19, 20, 22, 18, 24, 20]
        >>> y = [38, 37, 33, 29, 14, 12, 20, 22]
        >>> print("Medians = %.2f - %.2f" % (np.median(x), np.median(y)))
        >>> wilcoxon(x, y, tail='two-sided')
    """
    from scipy.stats import wilcoxon
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NA
    x, y = _remove_na(x, y, paired=True)
    nx = x.size
    ny = y.size

    # Compute test
    wval, pval = wilcoxon(x, y, zero_method='wilcox', correction=False)
    pval *= .5 if tail == 'one-sided' else pval

    # Effect size 1: common language effect size (McGraw and Wong 1992)
    c = np.array([(a, b) for a in x for b in y])
    num = max((c[:, 0] < c[:, 1]).sum(), (c[:, 0] > c[:, 1]).sum())
    cles = num / (nx * ny)

    # Effect size 2: matched-pairs rank biserial correlation (Kerby 2014)
    rank = np.arange(x.size, 0, -1)
    rsum = rank.sum()
    fav = rank[np.sign(y - x) > 0].sum()
    unfav = rank[np.sign(y - x) < 0].sum()
    rbc = fav / rsum - unfav / rsum

    # Fill output DataFrame
    stats = pd.DataFrame({}, index=['Wilcoxon'])
    stats['W-val'] = wval.round(3)
    stats['p-val'] = pval
    stats['RBC'] = rbc
    stats['CLES'] = cles

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

        >>> import pandas as pd
        >>> from pingouin import kruskal, print_table
        >>> df = pd.read_csv('dataset.csv')
        >>> stats = kruskal(dv='DV', between='Group', data=df)
        >>> print_table(stats)
    """
    from scipy.stats import chi2, rankdata, tiecorrect

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
    data['rank'] = rankdata(data['DV'])

    # Find the total of rank per groups
    grp = data.groupby(between)['rank']
    sum_rk_grp = grp.sum().values
    n_per_grp = grp.count().values

    # Calculate chi-square statistic (H)
    H = (12 / (n * (n + 1)) * np.sum(sum_rk_grp**2 / n_per_grp)) - 3 * (n + 1)

    # Correct for ties
    H /= tiecorrect(data['rank'].values)

    # Calculate DOF and p-value
    ddof1 = n_groups - 1
    p_unc = chi2.sf(H, ddof1)

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

        >>> import pandas as pd
        >>> from pingouin import friedman, print_table
        >>> df = pd.read_csv('dataset.csv')
        >>> stats = friedman(dv='DV', within='Time', subject='Ss', data=df)
        >>> print_table(stats)
    """
    from scipy.stats import rankdata, chi2, find_repeats

    # Check data
    _check_dataframe(dv=dv, within=within, data=data, subject=subject,
                     effects='within')

    # Collapse to the mean
    data = data.groupby([subject, within]).mean().reset_index()

    # Remove NaN
    if data[dv].isnull().any():
        data = _remove_rm_na(dv=dv, within=within, subject=subject, data=data)

    # Extract number of groups and total sample size
    grp = data.groupby(within)[dv]
    rm = list(data[within].unique())
    k = len(rm)
    X = np.array([grp.get_group(r).values for r in rm]).T
    n = X.shape[0]

    # Rank per subject
    ranked = np.zeros(X.shape)
    for i in range(n):
        ranked[i] = rankdata(X[i, :])

    ssbn = (ranked.sum(axis=0)**2).sum()

    # Compute the test statistic
    Q = (12 / (n * k * (k + 1))) * ssbn - 3 * n * (k + 1)

    # Correct for ties
    ties = 0
    for i in range(n):
        replist, repnum = find_repeats(X[i])
        for t in repnum:
            ties += t * (t * t - 1)

    c = 1 - ties / float(k * (k * k - 1) * n)
    Q /= c

    # Approximate the p-value
    ddof1 = k - 1
    p_unc = chi2.sf(Q, ddof1)

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

    Data are expected to be in long-format.

    NaN are automatically removed from the data.

    Examples
    --------
    Compute the Cochran Q test for repeated measurements.

        >>> from pingouin.datasets import read_dataset
        >>> from pingouin import cochran
        >>> df = read_dataset('cochran')
        >>> cochran(dv='Energetic', within='Time', subject='Subject', data=df)
    """
    from scipy.stats import chi2

    # Check data
    _check_dataframe(dv=dv, within=within, data=data, subject=subject,
                     effects='within')

    # Remove NaN
    if data[dv].isnull().any():
        data = _remove_rm_na(dv=dv, within=within, subject=subject, data=data)

    # Groupby and extract size
    grp = data.groupby(within)[dv]
    grp_s = data.groupby(subject)[dv]
    k = data[within].unique().size
    dof = k - 1
    # n = grp.count().unique()[0]

    # Q statistic and p-value
    q = (dof * (k * np.sum(grp.sum()**2) - grp.sum().sum()**2)) / \
        (k * grp.sum().sum() - np.sum(grp_s.sum()**2))
    p_unc = chi2.sf(q, dof)

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
