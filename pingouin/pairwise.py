# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
import pandas as pd
from itertools import combinations, product
from pingouin.parametric import anova
from pingouin.multicomp import multicomp
from pingouin.effsize import compute_effsize, convert_effsize
from pingouin.utils import _remove_rm_na, _export_table, _check_dataframe

__all__ = ["pairwise_ttests", "pairwise_tukey", "pairwise_gameshowell",
           "pairwise_corr"]


def _append_stats_dataframe(stats, x, y, xlabel, ylabel, alpha, paired,
                            df_ttest, ef, eftype, time=np.nan):
    stats = stats.append({
        'A': xlabel,
        'B': ylabel,
        'mean(A)': np.round(x.mean(), 3),
        'mean(B)': np.round(y.mean(), 3),
        # Use ddof=1 for unibiased estimator (pandas default)
        'std(A)': np.round(x.std(ddof=1), 3),
        'std(B)': np.round(y.std(ddof=1), 3),
        'Paired': paired,
        'tail': df_ttest.loc['T-test', 'tail'],
        # 'Alpha': alpha,
        'T': df_ttest.loc['T-test', 'T'],
        'p-unc': df_ttest.loc['T-test', 'p-val'],
        'BF10': df_ttest.loc['T-test', 'BF10'],
        'efsize': ef,
        'eftype': eftype,
        'Time': time}, ignore_index=True, sort=False)
    return stats


def pairwise_ttests(dv=None, between=None, within=None, subject=None,
                    effects='all', data=None, alpha=.05, tail='two-sided',
                    padjust='none', effsize='hedges', return_desc=False,
                    export_filename=None):
    '''Pairwise T-tests.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    between: string
        Name of column containing the between factor.
    within : string
        Name of column containing the within factor.
    subject : string
        Name of column containing the subject identifier. Only useful when
        effects == 'within' or effects == 'interaction'.
    data : pandas DataFrame
        DataFrame
    alpha : float
        Significance level
    tail : string
        Indicates whether to return the 'two-sided' or 'one-sided' p-values
    padjust : string
        Method used for testing and adjustment of pvalues.
        Available methods are ::

        'none' : no correction
        'bonferroni' : one-step Bonferroni correction
        'holm' : step-down method using Bonferroni adjustments
        'fdr_bh' : Benjamini/Hochberg FDR correction
        'fdr_by' : Benjamini/Yekutieli FDR correction
    effsize : string or None
        Effect size type. Available methods are ::

        'none' : no effect size
        'cohen' : Unbiased Cohen d
        'hedges' : Hedges g
        'glass': Glass delta
        'eta-square' : Eta-square
        'odds-ratio' : Odds ratio
        'AUC' : Area Under the Curve
    return_desc : boolean
        If True, return group means and std
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    stats : DataFrame
        Stats summary ::

        'A' : Name of first measurement
        'B' : Name of second measurement
        'Paired' : indicates whether the two measurements are paired or not
        'Tail' : indicate whether the p-values are one-sided or two-sided
        'T' : T-values
        'p-unc' : Uncorrected p-values
        'p-corr' : Corrected p-values
        'p-adjust' : p-values correction method
        'BF10' : Bayes Factor
        'efsize' : effect sizes
        'eftype' : type of effect size

    Examples
    --------
    Compute Bonferroni-corrected pairwise post-hocs T-tests from a mixed model
    design.

        >>> import pandas as pd
        >>> from pingouin import pairwise_ttests, print_table
        >>> df = pd.read_csv('dataset.csv')
        >>> post_hocs = pairwise_ttests(dv='DV', within='Time', subject='Ss',
        >>>                             between='Group', data=df,
        >>>                             effects='all',
        >>>                             padjust='bonf', effsize='hedges')
        >>> # Print the table with 3 decimals
        >>> print_table(post_hocs, floatfmt=".3f")
    '''
    from pingouin.parametric import ttest

    # Safety checks
    effects_orig = '%s' % effects  # For later use (ensure deep copy)
    effects = 'within' if between is None else effects
    effects = 'between' if within is None else effects
    _check_dataframe(dv=dv, between=between, within=within, subject=subject,
                     effects=effects, data=data)

    if tail not in ['one-sided', 'two-sided']:
        raise ValueError('Tail not recognized')

    if not isinstance(alpha, float):
        raise ValueError('Alpha must be float')

    # Check if we have multiple between or within factors
    multiple_between = False
    multiple_within = False
    if isinstance(between, list):
        if len(between) > 1:
            multiple_between = True
        else:
            between = between[0]

    if isinstance(within, list):
        if len(within) > 1:
            multiple_within = True
        else:
            within = within[0]

    if multiple_within is True and multiple_between is True:
        raise ValueError("Multiple between and within factors are",
                         "currently not supported. Please select only one.")

    multiple = any([multiple_within, multiple_between])
    effects = 'all' if multiple is True else effects

    # Initialize empty variables
    stats = pd.DataFrame([])
    ddic = {}

    # OPTION A: SIMPLE MAIN EFFECTS, WITHIN OR BETWEEN
    if effects.lower() in ['within', 'between']:
        # Compute T-tests
        paired = True if effects == 'within' else False
        col = within if effects == 'within' else between

        # Remove NAN in repeated measurements
        if within is not None and data[dv].isnull().values.any():
            data = _remove_rm_na(dv=dv, within=within, subject=subject,
                                 data=data)

        # Extract effects
        labels = data[col].unique().tolist()
        for l in labels:
            ddic[l] = data.loc[data[col] == l, dv].values

        # Number and labels of possible comparisons
        if len(labels) >= 2:
            combs = list(combinations(labels, 2))
        else:
            raise ValueError('Data must have at least two columns')

        # Initialize vectors
        for comb in combs:
            col1, col2 = comb
            x = ddic.get(col1)
            y = ddic.get(col2)
            df_ttest = ttest(x, y, paired=paired, tail=tail)
            ef = compute_effsize(x=x, y=y, eftype=effsize, paired=paired)
            stats = _append_stats_dataframe(stats, x, y, col1, col2, alpha,
                                            paired, df_ttest, ef, effsize)
            stats['Contrast'] = col

        # Multiple comparisons
        padjust = None if stats['p-unc'].size <= 1 else padjust
        if padjust is not None:
            if padjust.lower() != 'none':
                reject, stats['p-corr'] = multicomp(stats['p-unc'].values,
                                                    alpha=alpha,
                                                    method=padjust)
                stats['p-adjust'] = padjust
        else:
            stats['p-corr'] = None
            stats['p-adjust'] = None

    # OPTION B: TWO BETWEEN OR WITHIN FACTORS AND/OR WITHIN + BETWEEN + INTER
    # B1: BETWEEN1 + BETWEEN2 + BETWEEN1 * BETWEEN2
    # B2: WITHIN1 + WITHIN2 + WITHIN1 * WITHIN2
    # B3: WITHIN + BETWEEN + WITHIN * BETWEEN
    if effects.lower() in ['all', 'interaction']:
        # Define cases
        if multiple_between is True:
            # Interaction between1 * between2
            factors = between
            fbt = factors
            fwt = [None, None]
            eft = ['between', 'between']
            paired = False
        elif multiple_within is True:
            # Interaction within1 * within2
            factors = within
            fbt = [None, None]
            fwt = factors
            eft = ['within', 'within']
            paired = True
        else:
            # Interaction within * between
            factors = [within, between]
            fbt = [None, between]
            fwt = [within, None]
            eft = ['within', 'between']
            paired = False

        if effects == 'all' and effects_orig != 'interaction':
            for i, f in enumerate(factors):
                stats = stats.append(pairwise_ttests(dv=dv,
                                                     between=fbt[i],
                                                     within=fwt[i],
                                                     subject=subject,
                                                     effects=eft[i],
                                                     data=data,
                                                     alpha=alpha,
                                                     tail=tail,
                                                     padjust=padjust,
                                                     effsize=effsize,
                                                     return_desc=return_desc),
                                     ignore_index=True, sort=False)

        # Then compute the interaction between the factors
        labels_fac1 = data[factors[0]].unique().tolist()
        labels_fac2 = data[factors[1]].unique().tolist()
        comb_fac1 = list(combinations(labels_fac1, 2))
        comb_fac2 = list(combinations(labels_fac2, 2))

        for lw in labels_fac1:
            for l in labels_fac2:
                tmp = data.loc[data[factors[0]] == lw]
                ddic[lw, l] = tmp.loc[tmp[factors[1]] == l, dv].values

        # Pairwise comparisons
        combs = list(product(labels_fac1, comb_fac2))
        for comb in combs:
            fac1, (col1, col2) = comb
            x = ddic.get((fac1, col1))
            y = ddic.get((fac1, col2))
            df_ttest = ttest(x, y, paired=paired, tail=tail)
            ef = compute_effsize(x=x, y=y, eftype=effsize, paired=paired)
            stats = _append_stats_dataframe(stats, x, y, col1, col2,
                                            alpha, paired, df_ttest, ef,
                                            effsize, fac1)

        # Finally update the Contrast and the p-adjust columns
        txt_inter = factors[0] + ' * ' + factors[1]
        if effects == 'all' and effects_orig != 'interaction':
            lc_fac1 = len(comb_fac1)
            lc_fac2 = len(comb_fac2)
            idxitr = np.arange(lc_fac1 + lc_fac2, stats.shape[0]).tolist()
            stats.loc[idxitr, 'Contrast'] = txt_inter
        else:
            stats['Contrast'] = txt_inter
            idxitr = np.arange(stats.shape[0]).tolist()

        if padjust is not None and padjust.lower() != 'none':
            _, pcor = multicomp(stats.loc[idxitr, 'p-unc'].values,
                                alpha=alpha, method=padjust)
            stats.loc[idxitr, 'p-corr'] = pcor
            stats.loc[idxitr, 'p-adjust'] = padjust

    # ---------------------------------------------------------------------
    stats['Paired'] = stats['Paired'].astype(bool)

    # Reorganize column order
    col_order = ['Contrast', 'Time', 'A', 'B', 'mean(A)', 'std(A)', 'mean(B)',
                 'std(B)', 'Paired', 'T', 'tail', 'p-unc',
                 'p-corr', 'p-adjust', 'BF10', 'efsize', 'eftype']

    if return_desc is False:
        stats.drop(columns=['mean(A)', 'mean(B)', 'std(A)', 'std(B)'],
                   inplace=True)

    stats = stats.reindex(columns=col_order)
    stats.dropna(how='all', axis=1, inplace=True)

    # Rename if multiple == True
    if multiple is True:
        stats['Time'].fillna('-', inplace=True)
        stats.rename(columns={'Time': factors[0]}, inplace=True)

    if 'Time' in stats.keys().tolist():
        stats['Time'].fillna('-', inplace=True)

    if export_filename is not None:
        _export_table(stats, export_filename)
    return stats


def pairwise_tukey(dv=None, between=None, data=None, alpha=.05,
                   tail='two-sided', effsize='hedges'):
    '''Pairwise Tukey-HSD post-hoc test.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    between: string
        Name of column containing the between factor.
    data : pandas DataFrame
        DataFrame
    alpha : float
        Significance level
    tail : string
        Indicates whether to return the 'two-sided' or 'one-sided' p-values
    effsize : string or None
        Effect size type. Available methods are ::

        'none' : no effect size
        'cohen' : Unbiased Cohen d
        'hedges' : Hedges g
        'glass': Glass delta
        'eta-square' : Eta-square
        'odds-ratio' : Odds ratio
        'AUC' : Area Under the Curve

    Returns
    -------
    stats : DataFrame
        Stats summary ::

        'A' : Name of first measurement
        'B' : Name of second measurement
        'mean(A)' : Mean of first measurement
        'mean(B)' : Mean of second measurement
        'diff' : Mean difference
        'SE' : Standard error
        'tail' : indicate whether the p-values are one-sided or two-sided
        'T' : T-values
        'p-tukey' : Tukey-HSD corrected p-values
        'efsize' : effect sizes
        'eftype' : type of effect size

    Notes
    -----
    Tukey HSD post-hoc is best for balanced one-way ANOVA.
    It has been proven to be conservative for one-way ANOVA with unequal
    sample sizes. However, it is not robust if the groups have unequal
    variances, in which case the Games-Howell test is more adequate.
    Tukey HSD is not valid for repeated measures ANOVA.

    Note that when the sample sizes are unequal, this function actually
    performs the Tukey-Kramer test (which allows for unequal sample sizes).

    The T-values are defined as:

    .. math::

        t = \dfrac{\overline{x}_i - \overline{x}_j}{\sqrt{2 \cdot MS_w / n}}

    where :math:`\overline{x}_i` and :math:`\overline{x}_j` are the means of
    the first and second group, respectively, :math:`MS_w` the mean squares of
    the error (computed using ANOVA) and :math:`n` the sample size.

    If the sample sizes are unequal, the Tukey-Kramer procedure is
    automatically used:

    .. math::

        t = \dfrac{\overline{x}_i - \overline{x}_j}{\sqrt{\dfrac{MS_w}{n_i}
        + \dfrac{MS_w}{n_j}}}

    where :math:`n_i` and :math:`n_j` are the sample sizes of the first and
    second group, respectively.

    The p-values are then approximated using the Studentized range distribution
    :math:`Q(\sqrt2*|t_i|, r, N - r)` where :math:`r` is the total number of
    groups and :math:`N` is the total sample size.

    Note that the p-values might be slightly different than those obtained
    using R or Matlab since the studentized range approximation is done using
    the Gleason (1999) algorithm, which is more efficient and accurate than
    the algorithms used in Matlab or R.

    References
    ----------
    .. [1] Tukey, John W. "Comparing individual means in the analysis of
           variance." Biometrics (1949): 99-114.

    .. [2] Gleason, John R. "An accurate, non-iterative approximation for
           studentized range quantiles." Computational statistics & data
           analysis 31.2 (1999): 147-158.

    Examples
    --------
    Pairwise Tukey post-hocs on the pain threshold dataset.

        >>> from pingouin import pairwise_tukey
        >>> from pingouin.datasets import read_dataset
        >>> df = read_dataset('anova')
        >>> pairwise_tukey(dv='Pain threshold', between='Hair color', data=df)
    '''
    from pingouin.external.qsturng import psturng

    # First compute the ANOVA
    aov = anova(dv=dv, data=data, between=between, detailed=True)
    df = aov.loc[1, 'DF']
    ng = aov.loc[0, 'DF'] + 1
    grp = data.groupby(between)[dv]
    n = grp.count().values
    gmeans = grp.mean().values
    gvar = aov.loc[1, 'MS'] / n

    # Pairwise combinations
    g1, g2 = np.array(list(combinations(np.arange(ng), 2))).T
    mn = gmeans[g1] - gmeans[g2]
    se = np.sqrt(gvar[g1] + gvar[g2])
    tval = mn / se

    # Critical values and p-values
    # from pingouin.external.qsturng import qsturng
    # crit = qsturng(1 - alpha, ng, df) / np.sqrt(2)
    pval = psturng(np.sqrt(2) * np.abs(tval), ng, df)
    pval *= 0.5 if tail == 'one-sided' else 1

    # Uncorrected p-values
    # from scipy.stats import t
    # punc = t.sf(np.abs(tval), n[g1].size + n[g2].size - 2) * 2

    # Effect size
    d = tval * np.sqrt(1 / n[g1] + 1 / n[g2])
    ef = convert_effsize(d, 'cohen', effsize, n[g1], n[g2])

    # Create dataframe
    # Careful: pd.unique does NOT sort whereas numpy does
    stats = pd.DataFrame({
                         'A': np.unique(data[between])[g1],
                         'B': np.unique(data[between])[g2],
                         'mean(A)': gmeans[g1],
                         'mean(B)': gmeans[g2],
                         'diff': mn,
                         'SE': np.round(se, 3),
                         'tail': tail,
                         'T': np.round(tval, 3),
                         # 'alpha': alpha,
                         # 'crit': np.round(crit, 3),
                         'p-tukey': pval,
                         'efsize': np.round(ef, 3),
                         'eftype': effsize,
                         })
    return stats


def pairwise_gameshowell(dv=None, between=None, data=None, alpha=.05,
                         tail='two-sided', effsize='hedges'):
    '''Pairwise Games-Howell post-hoc test.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    between: string
        Name of column containing the between factor.
    data : pandas DataFrame
        DataFrame
    alpha : float
        Significance level
    tail : string
        Indicates whether to return the 'two-sided' or 'one-sided' p-values
    effsize : string or None
        Effect size type. Available methods are ::

        'none' : no effect size
        'cohen' : Unbiased Cohen d
        'hedges' : Hedges g
        'glass': Glass delta
        'eta-square' : Eta-square
        'odds-ratio' : Odds ratio
        'AUC' : Area Under the Curve

    Returns
    -------
    stats : DataFrame
        Stats summary ::

        'A' : Name of first measurement
        'B' : Name of second measurement
        'mean(A)' : Mean of first measurement
        'mean(B)' : Mean of second measurement
        'diff' : Mean difference
        'SE' : Standard error
        'tail' : indicate whether the p-values are one-sided or two-sided
        'T' : T-values
        'df' : adjusted degrees of freedom
        'pval' : Games-Howell corrected p-values
        'efsize' : effect sizes
        'eftype' : type of effect size

    Notes
    -----
    Games-Howell is very similar to the Tukey HSD post-hoc test but is much
    more robust to heterogeneity of variances. While the
    Tukey-HSD post-hoc is optimal after a classic one-way ANOVA, the
    Games-Howell is optimal after a Welch ANOVA.
    Games-Howell is not valid for repeated measures ANOVA.

    Compared to the Tukey-HSD test, the Games-Howell test uses different pooled
    variances for each pair of variables instead of the same pooled variance.

    The T-values are defined as:

    .. math::

        t = \dfrac{\overline{x}_i - \overline{x}_j}{\sqrt{(\dfrac{s_i^2}{n_i}
        + \dfrac{s_j^2}{n_j})}}

    and the corrected degrees of freedom are:

    .. math::

        v = \dfrac{(\dfrac{s_i^2}{n_i} + \dfrac{s_j^2}{n_j})^2}
        {\dfrac{(\dfrac{s_i^2}{n_i})^2}{n_i-1} +
        \dfrac{(\dfrac{s_j^2}{n_j})^2}{n_j-1}}

    where :math:`\overline{x}_i`, :math:`s_i^2`, and :math:`n_i`
    are the mean, variance and sample size of the first group and
    :math:`\overline{x}_j`, :math:`s_j^2`, and :math:`n_j` the mean, variance
    and sample size of the second group.

    The p-values are then approximated using the Studentized range distribution
    :math:`Q(\sqrt2*|t_i|, r, v_i)`.

    Note that the p-values might be slightly different than those obtained
    using R or Matlab since the studentized range approximation is done using
    the Gleason (1999) algorithm, which is more efficient and accurate than
    the algorithms used in Matlab or R.

    References
    ----------
    .. [1] Games, Paul A., and John F. Howell. "Pairwise multiple comparison
           procedures with unequal n’s and/or variances: a Monte Carlo study."
           Journal of Educational Statistics 1.2 (1976): 113-125.

    .. [2] Gleason, John R. "An accurate, non-iterative approximation for
           studentized range quantiles." Computational statistics & data
           analysis 31.2 (1999): 147-158.

    Examples
    --------
    Pairwise Games-Howell post-hocs on the pain threshold dataset.

        >>> from pingouin import pairwise_gameshowell
        >>> from pingouin.datasets import read_dataset
        >>> df = read_dataset('anova')
        >>> pairwise_gameshowell(dv='Pain threshold', between='Hair color',
        >>>                      data=df)
    '''
    from pingouin.external.qsturng import psturng

    # Check the dataframe
    _check_dataframe(dv=dv, between=between, effects='between', data=data)

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)

    # Extract infos
    ng = data[between].unique().size
    grp = data.groupby(between)[dv]
    n = grp.count().values
    gmeans = grp.mean().values
    gvars = grp.var().values

    # Pairwise combinations
    g1, g2 = np.array(list(combinations(np.arange(ng), 2))).T
    mn = gmeans[g1] - gmeans[g2]
    se = np.sqrt(0.5 * (gvars[g1] / n[g1] + gvars[g2] / n[g2]))
    tval = mn / np.sqrt(gvars[g1] / n[g1] + gvars[g2] / n[g2])
    df = (gvars[g1] / n[g1] + gvars[g2] / n[g2])**2 / \
         ((((gvars[g1] / n[g1])**2) / (n[g1] - 1)) +
          (((gvars[g2] / n[g2])**2) / (n[g2] - 1)))

    # Compute corrected p-values
    pval = psturng(np.sqrt(2) * np.abs(tval), ng, df)
    pval *= 0.5 if tail == 'one-sided' else 1

    # Uncorrected p-values
    # from scipy.stats import t
    # punc = t.sf(np.abs(tval), n[g1].size + n[g2].size - 2) * 2

    # Effect size
    d = tval * np.sqrt(1 / n[g1] + 1 / n[g2])
    ef = convert_effsize(d, 'cohen', effsize, n[g1], n[g2])

    # Create dataframe
    # Careful: pd.unique does NOT sort whereas numpy does
    stats = pd.DataFrame({
                         'A': np.unique(data[between])[g1],
                         'B': np.unique(data[between])[g2],
                         'mean(A)': gmeans[g1],
                         'mean(B)': gmeans[g2],
                         'diff': mn,
                         'SE': se,
                         'tail': tail,
                         'T': tval,
                         'df': df,
                         'pval': pval,
                         'efsize': ef,
                         'eftype': effsize,
                         })
    col_round = ['mean(A)', 'mean(B)', 'diff', 'SE', 'T', 'df', 'efsize']
    stats[col_round] = stats[col_round].round(3)
    return stats


def pairwise_corr(data, columns=None, tail='two-sided', method='pearson',
                  padjust='none', export_filename=None):
    '''Pairwise correlations between columns of a pandas dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame
    columns : list or str
        Column names in data ::

        '["a", "b", "c"]' : combination between columns a, b, and c
        '["a"]' : product between a and all the other numeric columns
        '[["a"], ["b", "c"]]' : product between ["a"] and ["b", "c"]
        '[["a", "d"], ["b", "c"]]' : product between ["a", "d"] and ["b", "c"]
        '[["a", "d"], None]' : product between ["a", "d"] and all other columns

        Note that if column is not specified, then the function will return the
        pairwise correlation between the combination of all the numeric columns
        in data. See the examples section for more details on this.
    tail : string
        Indicates whether to return the 'two-sided' or 'one-sided' p-values
    method : string
        Specify which method to use for the computation of the correlation
        coefficient. Available methods are ::

        'pearson' : Pearson product-moment correlation
        'spearman' : Spearman rank-order correlation
        'kendall' : Kendall’s tau (ordinal data)
        'percbend' : percentage bend correlation (robust)
        'shepherd' : Shepherd's pi correlation (robust Spearman)
    padjust : string
        Method used for testing and adjustment of pvalues.
        Available methods are ::

        'none' : no correction
        'bonferroni' : one-step Bonferroni correction
        'holm' : step-down method using Bonferroni adjustments
        'fdr_bh' : Benjamini/Hochberg FDR correction
        'fdr_by' : Benjamini/Yekutieli FDR correction
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    stats : DataFrame
        Stats summary ::

        'X' : Name(s) of first columns
        'Y' : Name(s) of second columns
        'method' : method used to compute the correlation
        'tail' : indicates whether the p-values are one-sided or two-sided
        'n' : Sample size (after NaN removal)
        'r' : Correlation coefficients
        'CI95' : 95% parametric confidence intervals
        'r2' : R-squared values
        'adj_r2' : Adjusted R-squared values
        'z' : Standardized correlation coefficients
        'p-unc' : uncorrected one or two tailed p-values
        'p-corr' : corrected one or two tailed p-values
        'p-adjust' : Correction method

    Notes
    -----
    Please refer to the `pingouin.corr()` function for a description of the
    different methods. NaN are automatically removed from the data.

    This function is more flexible and gives a much more detailed
    output than the `pandas.DataFrame.corr()` method (i.e. p-values,
    confidence interval, Bayes Factor..). This comes however at
    an increased computational cost. While this should not be discernible for
    dataframe with less than 10,000 rows and/or less than 20 columns, this
    function can be slow for very large dataset.

    For speed purpose, the Bayes Factor is only computed when the sample size
    is less than 1000 (and method='pearson').

    Examples
    --------
    1. One-tailed spearman correlation corrected for multiple comparisons

        >>> from pingouin.datasets import read_dataset
        >>> from pingouin import pairwise_corr
        >>> data = read_dataset('pairwise_corr').iloc[:, 1:]
        >>> stats = pairwise_corr(data, method='spearman', tail='two-sided',
        >>>                       padjust='bonf')
        >>> print_table(stats)

    2. Robust two-sided correlation with uncorrected p-values

        >>> pairwise_corr(data, columns=['Openness', 'Extraversion',
        >>>                              'Neuroticism'], method='percbend')

    3. Export the results to a .csv file

        >>> pairwise_corr(data, export_filename='pairwise_corr.csv')

    4. One-versus-others pairwise correlations

        >>> pairwise_corr(data, columns=['Neuroticism'])

    5. Pairwise correlations between two lists of columns (cartesian product)

        >>> pairwise_corr(data, columns=[['Neuroticism', 'Extraversion'],
        >>>                              ['Openness', 'Agreeableness'])
    '''
    from pingouin.correlation import corr

    if tail not in ['one-sided', 'two-sided']:
        raise ValueError('Tail not recognized')

    # Keep only numeric columns
    data = data._get_numeric_data()
    keys = data.keys().tolist()

    # Initialize empty DataFrame
    stats = pd.DataFrame()

    # First ensure that columns is a list
    if isinstance(columns, str):
        columns = [columns]

    # Then define combinations / products between columns
    if columns is None:
        # Case A: column is not defined --> corr between all numeric columns
        combs = list(combinations(keys, 2))
    else:
        # Case B: column is specified
        if isinstance(columns[0], list):
            group1 = [e for e in columns[0] if e in keys]
            # Assert that column is two-dimensional
            if len(columns) == 1:
                columns.append(None)
            if isinstance(columns[1], list) and len(columns[1]):
                # B1: [['a', 'b'], ['c', 'd']]
                group2 = [e for e in columns[1] if e in keys]
            else:
                # B2: [['a', 'b']], [['a', 'b'], None] or [['a', 'b'], 'all']
                group2 = [e for e in keys if e not in group1]
            combs = list(product(group1, group2))
        else:
            # Column is a simple list
            if len(columns) == 1:
                # Case B3: one-versus-all, e.g. ['a'] or 'a'
                others = [e for e in keys if e != columns[0]]
                combs = list(product(columns, others))
            else:
                # Combinations between all specified columns ['a', 'b', 'c']
                # Make sure that we keep numeric columns
                columns = np.intersect1d(keys, columns)
                if len(columns) == 1:
                    # If only one-column is left, equivalent to ['a']
                    others = [e for e in keys if e != columns[0]]
                    combs = list(product(columns, others))
                else:
                    # combinations between ['a', 'b', 'c']
                    combs = list(combinations(columns, 2))

    # Assert that all columns do exist in DataFrame
    # If you see this error, check for column name errors in `columns=[]`
    for comb in combs:
        assert comb[0] in keys
        assert comb[1] in keys

    # Initialize vectors
    for comb in combs:
        col1, col2 = comb
        cor_st = corr(data[col1].values,
                      data[col2].values,
                      tail=tail, method=method).reset_index(drop=True)
        stats = stats.append({
            'X': col1,
            'Y': col2,
            'method': method,
            'tail': tail,
            'n': cor_st['n'][0],
            'r': cor_st['r'][0],
            'CI95%': cor_st['CI95%'][0],
            'r2': cor_st['r2'][0],
            'adj_r2': cor_st['adj_r2'][0],
            'p-unc': cor_st['p-val'][0],
            'BF10': cor_st['BF10'][0] if 'BF10' in cor_st.keys() else np.nan},
            ignore_index=True)

    # Multiple comparisons
    padjust = None if stats['p-unc'].size <= 1 else padjust
    if padjust is not None:
        if padjust.lower() != 'none':
            reject, stats['p-corr'] = multicomp(stats['p-unc'].values,
                                                method=padjust)
            stats['p-adjust'] = padjust
    else:
        stats['p-corr'] = None
        stats['p-adjust'] = None

    # Standardize correlation coefficients (Fisher z-transformation)
    stats['z'] = np.arctanh(stats['r'].values)

    # Round values
    for c in ['r', 'r2', 'adj_r2', 'z']:
        stats[c] = stats[c].round(3)

    col_order = ['X', 'Y', 'method', 'tail', 'n', 'r', 'CI95%', 'r2', 'adj_r2',
                 'z', 'p-unc', 'p-corr', 'p-adjust', 'BF10']

    # Convert n to int
    stats['n'] = stats['n'].astype(int)
    stats = stats.reindex(columns=col_order)
    stats.dropna(how='all', axis=1, inplace=True)
    if export_filename is not None:
        _export_table(stats, export_filename)
    return stats
