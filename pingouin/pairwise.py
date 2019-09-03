# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
import pandas as pd
import pandas_flavor as pf
from itertools import combinations, product
from pingouin.parametric import anova
from pingouin.multicomp import multicomp
from pingouin.effsize import compute_effsize, convert_effsize
from pingouin.utils import (remove_rm_na, _export_table, _check_dataframe,
                            _flatten_list)

__all__ = ["pairwise_ttests", "pairwise_tukey", "pairwise_gameshowell",
           "pairwise_corr"]


@pf.register_dataframe_method
def pairwise_ttests(data=None, dv=None, between=None, within=None,
                    subject=None, parametric=True, alpha=.05, tail='two-sided',
                    padjust='none', effsize='hedges', nan_policy='listwise',
                    return_desc=False, interaction=True,
                    export_filename=None):
    '''Pairwise T-tests.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    dv : string
        Name of column containing the dependant variable.
    between : string or list with 2 elements
        Name of column(s) containing the between factor(s).
    within : string or list with 2 elements
        Name of column(s) containing the within factor(s).
    subject : string
        Name of column containing the subject identifier. Compulsory for
        contrast including a within-subject factor.
    parametric : boolean
        If True (default), use the parametric :py:func:`ttest` function.
        If False, use :py:func:`pingouin.wilcoxon` or :py:func:`pingouin.mwu`
        for paired or unpaired samples, respectively.
    alpha : float
        Significance level
    tail : string
        Specify whether the alternative hypothesis is `'two-sided'` or
        `'one-sided'`. Can also be `'greater'` or `'less'` to specify the
        direction of the test. `'greater'` tests the alternative that ``x``
        has a larger mean than ``y``. If tail is `'one-sided'`, Pingouin will
        automatically infer the one-sided alternative hypothesis of the test
        based on the test statistic.
    padjust : string
        Method used for testing and adjustment of pvalues.
        Available methods are ::

        'none' : no correction
        'bonf' : one-step Bonferroni correction
        'sidak' : one-step Sidak correction
        'holm' : step-down method using Bonferroni adjustments
        'fdr_bh' : Benjamini/Hochberg FDR correction
        'fdr_by' : Benjamini/Yekutieli FDR correction
    effsize : string or None
        Effect size type. Available methods are ::

        'none' : no effect size
        'cohen' : Unbiased Cohen d
        'hedges' : Hedges g
        'glass': Glass delta
        'r' : Pearson correlation coefficient
        'eta-square' : Eta-square
        'odds-ratio' : Odds ratio
        'AUC' : Area Under the Curve
        'CLES' : Common Language Effect Size
    nan_policy : string
        Can be `'listwise'` for listwise deletion of missing values in repeated
        measures design (= complete-case analysis) or `'pairwise'` for the
        more liberal pairwise deletion (= available-case analysis).

        .. versionadded:: 0.2.9
    return_desc : boolean
        If True, append group means and std to the output dataframe
    interaction : boolean
        If there are multiple factors and ``interaction`` is True (default),
        Pingouin will also calculate T-tests for the interaction term (see
        Notes).

        .. versionadded:: 0.2.9
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
        'Parametric' : indicates if (non)-parametric tests were used
        'Tail' : indicate whether the p-values are one-sided or two-sided
        'T' : T statistic (only if parametric=True)
        'U-val' : Mann-Whitney U stat (if parametric=False and unpaired data)
        'W-val' : Wilcoxon W stat (if parametric=False and paired data)
        'dof' : degrees of freedom (only if parametric=True)
        'p-unc' : Uncorrected p-values
        'p-corr' : Corrected p-values
        'p-adjust' : p-values correction method
        'BF10' : Bayes Factor
        'hedges' : effect size (or any effect size defined in ``effsize``)

    See also
    --------
    ttest, mwu, wilcoxon, compute_effsize, multicomp

    Notes
    -----
    Data are expected to be in long-format. If your data is in wide-format,
    you can use the :py:func:`pandas.melt` function to convert from wide to
    long format.

    If ``between`` or ``within`` is a list (e.g. ['col1', 'col2']),
    the function returns 1) the pairwise T-tests between each values of the
    first column, 2) the pairwise T-tests between each values of the second
    column and 3) the interaction between col1 and col2. The interaction is
    dependent of the order of the list, so ['col1', 'col2'] will not yield the
    same results as ['col2', 'col1'], and will only be calculated if
    ``interaction=True``.

    In other words, if ``between`` is a list with two elements, the output
    model is between1 + between2 + between1 * between2.

    Similarly, if `within`` is a list with two elements, the output model is
    within1 + within2 + within1 * within2.

    If both ``between`` and ``within`` are specified, the function return
    within + between + within * between.

    Missing values in repeated measurements are automatically removed using a
    listwise (default) or pairwise deletion strategy. However, you should be
    very careful since it can result in undesired values removal (especially
    for the interaction effect). We strongly recommend that you preprocess
    your data and remove the missing values before using this function.

    This function has been tested against the `pairwise.t.test` R function.

    Examples
    --------
    1. One between-factor

    >>> from pingouin import pairwise_ttests, read_dataset
    >>> df = read_dataset('mixed_anova.csv')
    >>> post_hocs = pairwise_ttests(dv='Scores', between='Group', data=df)

    2. One within-factor

    >>> post_hocs = pairwise_ttests(dv='Scores', within='Time',
    ...                             subject='Subject', data=df)
    >>> print(post_hocs)  # doctest: +SKIP

    3. Non-parametric pairwise paired test (wilcoxon)

    >>> pairwise_ttests(dv='Scores', within='Time', subject='Subject',
    ...                 data=df, parametric=False)  # doctest: +SKIP

    4. Within + Between + Within * Between with corrected p-values

    >>> posthocs = pairwise_ttests(dv='Scores', within='Time',
    ...                            subject='Subject', between='Group',
    ...                            padjust='bonf', data=df)

    5. Between1 + Between2 + Between1 * Between2

    >>> posthocs = pairwise_ttests(dv='Scores', between=['Group', 'Time'],
    ...                            data=df)

    6. Between1 + Between2, no interaction

    >>> posthocs = df.pairwise_ttests(dv='Scores', between=['Group', 'Time'],
    ...                               interaction=False)
    '''
    from .parametric import ttest
    from .nonparametric import wilcoxon, mwu

    # Safety checks
    _check_dataframe(dv=dv, between=between, within=within, subject=subject,
                     effects='all', data=data)

    assert tail in ['one-sided', 'two-sided', 'greater', 'less']
    assert isinstance(alpha, float), 'alpha must be float.'
    assert nan_policy in ['listwise', 'pairwise']

    # Check if we have multiple between or within factors
    multiple_between = False
    multiple_within = False
    contrast = None

    if isinstance(between, list):
        if len(between) > 1:
            multiple_between = True
            contrast = 'multiple_between'
            assert all([b in data.keys() for b in between])
        else:
            between = between[0]

    if isinstance(within, list):
        if len(within) > 1:
            multiple_within = True
            contrast = 'multiple_within'
            assert all([w in data.keys() for w in within])
        else:
            within = within[0]

    if all([multiple_within, multiple_between]):
        raise ValueError("Multiple between and within factors are",
                         "currently not supported. Please select only one.")

    # Check the other cases
    if isinstance(between, str) and within is None:
        contrast = 'simple_between'
        assert between in data.keys()
    if isinstance(within, str) and between is None:
        contrast = 'simple_within'
        assert within in data.keys()
    if isinstance(between, str) and isinstance(within, str):
        contrast = 'within_between'
        assert all([between in data.keys(), within in data.keys()])

    # Reorganize column order
    col_order = ['Contrast', 'Time', 'A', 'B', 'mean(A)', 'std(A)', 'mean(B)',
                 'std(B)', 'Paired', 'Parametric', 'T', 'U-val', 'W-val',
                 'dof', 'Tail', 'p-unc', 'p-corr', 'p-adjust', 'BF10',
                 effsize]

    if contrast in ['simple_within', 'simple_between']:
        # OPTION A: SIMPLE MAIN EFFECTS, WITHIN OR BETWEEN
        paired = True if contrast == 'simple_within' else False
        col = within if contrast == 'simple_within' else between
        # Remove NAN in repeated measurements
        if contrast == 'simple_within' and data[dv].isnull().values.any():
            # Only if nan_policy == 'listwise'. For pairwise deletion,
            # missing values will be removed directly in the lower-level
            # functions (e.g. pg.ttest)
            if nan_policy == 'listwise':
                data = remove_rm_na(dv=dv, within=within, subject=subject,
                                    data=data)
            else:
                # The `remove_rm_na` also aggregate other repeated measures
                # factor using the mean. Here, we ensure this behavior too.
                data = data.groupby([subject, within])[dv].mean().reset_index()
            # Now we check that subjects are present in all conditions
            # For example, if we have four subjects and 3 conditions,
            # and if subject 2 have missing data at the third condition,
            # we still need a row with missing values for this subject.
            if data.groupby(within)[subject].count().nunique() != 1:
                raise ValueError("Repeated measures dataframe is not balanced."
                                 " `Subjects` must have the same number of "
                                 "elements in all conditions, "
                                 "even when missing values are present.")

        # Extract effects
        grp_col = data.groupby(col, sort=False)[dv]
        labels = grp_col.groups.keys()
        # Number and labels of possible comparisons
        if len(labels) >= 2:
            combs = list(combinations(labels, 2))
            combs = np.array(combs)
            A = combs[:, 0]
            B = combs[:, 1]
        else:
            raise ValueError('Columns must have at least two unique values.')

        # Initialize dataframe
        stats = pd.DataFrame(dtype=np.float64, index=range(len(combs)),
                             columns=col_order)

        # Force dtype conversion
        cols_str = ['Contrast', 'Time', 'A', 'B', 'Tail', 'p-adjust', 'BF10']
        cols_bool = ['Parametric', 'Paired']
        stats[cols_str] = stats[cols_str].astype(object)
        stats[cols_bool] = stats[cols_bool].astype(bool)

        # Fill str columns
        stats.loc[:, 'A'] = A
        stats.loc[:, 'B'] = B
        stats.loc[:, 'Contrast'] = col
        stats.loc[:, 'Tail'] = tail
        stats.loc[:, 'Paired'] = paired

        for i in range(stats.shape[0]):
            col1, col2 = stats.at[i, 'A'], stats.at[i, 'B']
            x = grp_col.get_group(col1).to_numpy(dtype=np.float64)
            y = grp_col.get_group(col2).to_numpy(dtype=np.float64)
            if parametric:
                stat_name = 'T'
                df_ttest = ttest(x, y, paired=paired, tail=tail)
                stats.at[i, 'BF10'] = df_ttest.at['T-test', 'BF10']
                stats.at[i, 'dof'] = df_ttest.at['T-test', 'dof']
            else:
                if paired:
                    stat_name = 'W-val'
                    df_ttest = wilcoxon(x, y, tail=tail)
                else:
                    stat_name = 'U-val'
                    df_ttest = mwu(x, y, tail=tail)

            # Compute Hedges / Cohen
            ef = np.round(compute_effsize(x=x, y=y, eftype=effsize,
                                          paired=paired), 3)

            if return_desc:
                stats.at[i, 'mean(A)'] = np.round(np.nanmean(x), 3)
                stats.at[i, 'mean(B)'] = np.round(np.nanmean(y), 3)
                stats.at[i, 'std(A)'] = np.round(np.nanstd(x), 3)
                stats.at[i, 'std(B)'] = np.round(np.nanstd(y), 3)
            stats.at[i, stat_name] = df_ttest[stat_name].iat[0]
            stats.at[i, 'p-unc'] = df_ttest['p-val'].iat[0]
            stats.at[i, effsize] = ef

        # Multiple comparisons
        padjust = None if stats['p-unc'].size <= 1 else padjust
        if padjust is not None:
            if padjust.lower() != 'none':
                _, stats['p-corr'] = multicomp(stats['p-unc'].values,
                                               alpha=alpha, method=padjust)
                stats['p-adjust'] = padjust
        else:
            stats['p-corr'] = None
            stats['p-adjust'] = None
    else:
        # B1: BETWEEN1 + BETWEEN2 + BETWEEN1 * BETWEEN2
        # B2: WITHIN1 + WITHIN2 + WITHIN1 * WITHIN2
        # B3: WITHIN + BETWEEN + WITHIN * BETWEEN
        if contrast == 'multiple_between':
            # B1
            factors = between
            fbt = factors
            fwt = [None, None]
            # eft = ['between', 'between']
            paired = False
        elif contrast == 'multiple_within':
            # B2
            factors = within
            fbt = [None, None]
            fwt = factors
            # eft = ['within', 'within']
            paired = True
        else:
            # B3
            factors = [within, between]
            fbt = [None, between]
            fwt = [within, None]
            # eft = ['within', 'between']
            paired = False

        stats = pd.DataFrame()
        for i, f in enumerate(factors):
            stats = stats.append(pairwise_ttests(dv=dv,
                                                 between=fbt[i],
                                                 within=fwt[i],
                                                 subject=subject,
                                                 data=data,
                                                 parametric=parametric,
                                                 alpha=alpha,
                                                 tail=tail,
                                                 padjust=padjust,
                                                 effsize=effsize,
                                                 return_desc=return_desc),
                                 ignore_index=True, sort=False)

        # Then compute the interaction between the factors
        if interaction:
            nrows = stats.shape[0]
            grp_fac1 = data.groupby(factors[0], sort=False)[dv]
            grp_fac2 = data.groupby(factors[1], sort=False)[dv]
            grp_both = data.groupby(factors, sort=False)[dv]
            labels_fac1 = grp_fac1.groups.keys()
            labels_fac2 = grp_fac2.groups.keys()
            # comb_fac1 = list(combinations(labels_fac1, 2))
            comb_fac2 = list(combinations(labels_fac2, 2))

            # Pairwise comparisons
            combs_list = list(product(labels_fac1, comb_fac2))
            ncombs = len(combs_list)
            # np.array(combs_list) does not work because of tuples
            # we therefore need to flatten the tupple
            combs = np.zeros(shape=(ncombs, 3), dtype=object)
            for i in range(ncombs):
                combs[i] = _flatten_list(combs_list[i], include_tuple=True)

            # Append empty rows
            idxiter = np.arange(nrows, nrows + ncombs)
            stats = stats.append(pd.DataFrame(columns=stats.columns,
                                 index=idxiter), ignore_index=True)
            # Update other columns
            stats.loc[idxiter, 'Contrast'] = factors[0] + ' * ' + factors[1]
            stats.loc[idxiter, 'Time'] = combs[:, 0]
            stats.loc[idxiter, 'Paired'] = paired
            stats.loc[idxiter, 'Tail'] = tail
            stats.loc[idxiter, 'A'] = combs[:, 1]
            stats.loc[idxiter, 'B'] = combs[:, 2]

            for i, comb in enumerate(combs):
                ic = nrows + i  # Take into account previous rows
                fac1, col1, col2 = comb
                x = grp_both.get_group((fac1, col1)).to_numpy(dtype=np.float64)
                y = grp_both.get_group((fac1, col2)).to_numpy(dtype=np.float64)
                ef = np.round(compute_effsize(x=x, y=y, eftype=effsize,
                                              paired=paired), 3)
                if parametric:
                    stat_name = 'T'
                    df_ttest = ttest(x, y, paired=paired, tail=tail)
                    stats.at[ic, 'BF10'] = df_ttest.at['T-test', 'BF10']
                    stats.at[ic, 'dof'] = df_ttest.at['T-test', 'dof']
                else:
                    if paired:
                        stat_name = 'W-val'
                        df_ttest = wilcoxon(x, y, tail=tail)
                    else:
                        stat_name = 'U-val'
                        df_ttest = mwu(x, y, tail=tail)

                # Append to stats
                if return_desc:
                    stats.at[ic, 'mean(A)'] = np.round(np.nanmean(x), 3)
                    stats.at[ic, 'mean(B)'] = np.round(np.nanmean(y), 3)
                    stats.at[ic, 'std(A)'] = np.round(np.nanstd(x), 3)
                    stats.at[ic, 'std(B)'] = np.round(np.nanstd(y), 3)
                stats.at[ic, stat_name] = df_ttest[stat_name].iat[0]
                stats.at[ic, 'p-unc'] = df_ttest['p-val'].iat[0]
                stats.at[ic, effsize] = ef

            # Multi-comparison columns
            if padjust is not None and padjust.lower() != 'none':
                _, pcor = multicomp(stats.loc[idxiter, 'p-unc'].values,
                                    alpha=alpha, method=padjust)
                stats.loc[idxiter, 'p-corr'] = pcor
                stats.loc[idxiter, 'p-adjust'] = padjust

    # ---------------------------------------------------------------------
    # Append parametric columns
    stats.loc[:, 'Parametric'] = parametric

    # Reorder and drop empty columns
    stats = stats[np.array(col_order)[np.isin(col_order, stats.columns)]]
    stats = stats.dropna(how='all', axis=1)

    # Rename Time columns
    if (contrast in ['multiple_within', 'multiple_between', 'within_between']
       and interaction):
        stats['Time'].fillna('-', inplace=True)
        stats.rename(columns={'Time': factors[0]}, inplace=True)

    if export_filename is not None:
        _export_table(stats, export_filename)
    return stats


@pf.register_dataframe_method
def pairwise_tukey(data=None, dv=None, between=None, alpha=.05,
                   tail='two-sided', effsize='hedges'):
    '''Pairwise Tukey-HSD post-hoc test.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    dv : string
        Name of column containing the dependant variable.
    between: string
        Name of column containing the between factor.
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
        'r' : Pearson correlation coefficient
        'eta-square' : Eta-square
        'odds-ratio' : Odds ratio
        'AUC' : Area Under the Curve
        'CLES' : Common Language Effect Size

    Returns
    -------
    stats : DataFrame
        Stats summary ::

        'A' : Name of first measurement
        'B' : Name of second measurement
        'mean(A)' : Mean of first measurement
        'mean(B)' : Mean of second measurement
        'diff' : Mean difference (= mean(A) - mean(B))
        'se' : Standard error
        'tail' : indicate whether the p-values are one-sided or two-sided
        'T' : T-values
        'p-tukey' : Tukey-HSD corrected p-values
        'hedges' : effect size (or any effect size defined in ``effsize``)

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

        t = \\frac{\\overline{x}_i - \\overline{x}_j}
        {\\sqrt{2 \\cdot MS_w / n}}

    where :math:`\\overline{x}_i` and :math:`\\overline{x}_j` are the means of
    the first and second group, respectively, :math:`MS_w` the mean squares of
    the error (computed using ANOVA) and :math:`n` the sample size.

    If the sample sizes are unequal, the Tukey-Kramer procedure is
    automatically used:

    .. math::

        t = \\frac{\\overline{x}_i - \\overline{x}_j}{\\sqrt{\\frac{MS_w}{n_i}
        + \\frac{MS_w}{n_j}}}

    where :math:`n_i` and :math:`n_j` are the sample sizes of the first and
    second group, respectively.

    The p-values are then approximated using the Studentized range distribution
    :math:`Q(\\sqrt2*|t_i|, r, N - r)` where :math:`r` is the total number of
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

    >>> from pingouin import pairwise_tukey, read_dataset
    >>> df = read_dataset('anova')
    >>> pt = pairwise_tukey(data=df, dv='Pain threshold', between='Hair color')
    '''
    from pingouin.external.qsturng import psturng

    # First compute the ANOVA
    aov = anova(dv=dv, data=data, between=between, detailed=True)
    df = aov.at[1, 'DF']
    ng = aov.at[0, 'DF'] + 1
    grp = data.groupby(between)[dv]
    n = grp.count().values
    gmeans = grp.mean().values
    gvar = aov.at[1, 'MS'] / n

    # Pairwise combinations
    g1, g2 = np.array(list(combinations(np.arange(ng), 2))).T
    mn = gmeans[g1] - gmeans[g2]
    se = np.sqrt(gvar[g1] + gvar[g2])
    tval = mn / se

    # Critical values and p-values
    # from pingouin.external.qsturng import qsturng
    # crit = qsturng(1 - alpha, ng, df) / np.sqrt(2)
    pval = psturng(np.sqrt(2) * np.abs(tval), ng, df)
    pval = pval * 0.5 if tail == 'one-sided' else pval

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
                         'mean(A)': np.round(gmeans[g1], 3),
                         'mean(B)': np.round(gmeans[g2], 3),
                         'diff': np.round(mn, 3),
                         'se': np.round(se, 3),
                         'tail': tail,
                         'T': np.round(tval, 3),
                         # 'alpha': alpha,
                         # 'crit': np.round(crit, 3),
                         'p-tukey': pval,
                         effsize: np.round(ef, 3),
                         })
    return stats


def pairwise_gameshowell(data=None, dv=None, between=None, alpha=.05,
                         tail='two-sided', effsize='hedges'):
    '''Pairwise Games-Howell post-hoc test.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame
    dv : string
        Name of column containing the dependant variable.
    between: string
        Name of column containing the between factor.
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
        'diff' : Mean difference (= mean(A) - mean(B))
        'se' : Standard error
        'tail' : indicate whether the p-values are one-sided or two-sided
        'T' : T-values
        'df' : adjusted degrees of freedom
        'pval' : Games-Howell corrected p-values
        'hedges' : effect size (or any effect size defined in ``effsize``)

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

        t = \\frac{\\overline{x}_i - \\overline{x}_j}
        {\\sqrt{(\\frac{s_i^2}{n_i} + \\frac{s_j^2}{n_j})}}

    and the corrected degrees of freedom are:

    .. math::

        v = \\frac{(\\frac{s_i^2}{n_i} + \\frac{s_j^2}{n_j})^2}
        {\\frac{(\\frac{s_i^2}{n_i})^2}{n_i-1} +
        \\frac{(\\frac{s_j^2}{n_j})^2}{n_j-1}}

    where :math:`\\overline{x}_i`, :math:`s_i^2`, and :math:`n_i`
    are the mean, variance and sample size of the first group and
    :math:`\\overline{x}_j`, :math:`s_j^2`, and :math:`n_j` the mean, variance
    and sample size of the second group.

    The p-values are then approximated using the Studentized range distribution
    :math:`Q(\\sqrt2*|t_i|, r, v_i)`.

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

    >>> from pingouin import pairwise_gameshowell, read_dataset
    >>> df = read_dataset('anova')
    >>> pairwise_gameshowell(data=df, dv='Pain threshold',
    ...                      between='Hair color')  # doctest: +SKIP
    '''
    from pingouin.external.qsturng import psturng

    # Check the dataframe
    _check_dataframe(dv=dv, between=between, effects='between', data=data)

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)

    # Extract infos
    ng = data[between].nunique()
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
    pval = pval * 0.5 if tail == 'one-sided' else pval

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
                         'se': se,
                         'tail': tail,
                         'T': tval,
                         'df': df,
                         'pval': pval,
                         effsize: ef,
                         })
    col_round = ['mean(A)', 'mean(B)', 'diff', 'se', 'T', 'df', effsize]
    stats[col_round] = stats[col_round].round(3)
    return stats


@pf.register_dataframe_method
def pairwise_corr(data, columns=None, covar=None, tail='two-sided',
                  method='pearson', padjust='none', nan_policy='pairwise',
                  export_filename=None):
    """Pairwise (partial) correlations between columns of a pandas dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
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
    covar : None, string or list
        Covariate(s) for partial correlation. Must be one or more columns
        in data. Use a list if there are more than one covariate. If
        ``covar`` is not None, a partial correlation will be computed using
        :py:func:`pingouin.partial_corr` function.
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
        'bonf' : one-step Bonferroni correction
        'sidak' : one-step Sidak correction
        'holm' : step-down method using Bonferroni adjustments
        'fdr_bh' : Benjamini/Hochberg FDR correction
        'fdr_by' : Benjamini/Yekutieli FDR correction
    nan_policy : string
        Can be `'listwise'` for listwise deletion of missing values
        (= complete-case analysis) or `'pairwise'` (default) for the more
        liberal pairwise deletion (= available-case analysis).

        .. versionadded:: 0.2.9
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
        'covar' : List of specified covariate(s) (only for partial correlation)
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
    Please refer to the :py:func:`pingouin.corr()` function for a description
    of the different methods. NaN are automatically removed from the data using
    a pairwise deletion.

    This function is more flexible and gives a much more detailed
    output than the :py:func:`pandas.DataFrame.corr()` method (i.e. p-values,
    confidence interval, Bayes Factor..). This comes however at
    an increased computational cost. While this should not be discernible for
    dataframe with less than 10,000 rows and/or less than 20 columns, this
    function can be slow for very large dataset. For speed purpose, the Bayes
    Factor is only computed when the sample size is less than 1000
    (and method='pearson').

    A faster alternative to get the r-values and p-values in a matrix format is
    to use the :py:func:`pingouin.rcorr` function, which works directly as a
    :py:class:`pandas.DataFrame` method (see example below).

    This function also works with two-dimensional multi-index columns. In this
    case, columns must be list(s) of tuple(s). See the Jupyter notebook
    for more details:
    https://github.com/raphaelvallat/pingouin/blob/master/notebooks/04_Correlations.ipynb

    If ``covar`` is specified, this function will compute the pairwise partial
    correlation between the variables. If you are only interested in computing
    the partial correlation matrix (i.e. the raw pairwise partial correlation
    coefficient matrix, without the p-values, sample sizes, etc), a better
    alternative is to use the :py:func:`pingouin.pcorr` function (see
    example 7).

    Examples
    --------
    1. One-sided spearman correlation corrected for multiple comparisons

    >>> from pingouin import pairwise_corr, read_dataset
    >>> data = read_dataset('pairwise_corr').iloc[:, 1:]
    >>> pairwise_corr(data, method='spearman', tail='one-sided',
    ...               padjust='bonf')  # doctest: +SKIP

    2. Robust two-sided correlation with uncorrected p-values

    >>> pcor = pairwise_corr(data, columns=['Openness', 'Extraversion',
    ...                                     'Neuroticism'], method='percbend')

    3. One-versus-all pairwise correlations

    >>> pairwise_corr(data, columns=['Neuroticism'])  # doctest: +SKIP

    4. Pairwise correlations between two lists of columns (cartesian product)

    >>> columns = [['Neuroticism', 'Extraversion'], ['Openness']]
    >>> pairwise_corr(data, columns)   # doctest: +SKIP

    5. As a Pandas method

    >>> pcor = data.pairwise_corr(covar='Neuroticism', method='spearman')

    6. Pairwise partial correlation

    >>> pcor = pairwise_corr(data, covar='Neuroticism')  # One covariate
    >>> pcor = pairwise_corr(data, covar=['Neuroticism', 'Openness'])  # Two

    7. Pairwise partial correlation matrix using :py:func:`pingouin.pcorr`

    >>> data[['Neuroticism', 'Openness', 'Extraversion']].pcorr()
                  Neuroticism  Openness  Extraversion
    Neuroticism      1.000000  0.092097     -0.360421
    Openness         0.092097  1.000000      0.281312
    Extraversion    -0.360421  0.281312      1.000000

    8. Correlation matrix with p-values using :py:func:`pingouin.rcorr`

    >>> data[['Neuroticism', 'Openness', 'Extraversion']].rcorr()
                 Neuroticism Openness Extraversion
    Neuroticism            -                   ***
    Openness           -0.01        -          ***
    Extraversion       -0.35    0.267            -
    """
    from pingouin.correlation import corr, partial_corr

    # Check arguments
    assert tail in ['one-sided', 'two-sided']
    assert nan_policy in ['listwise', 'pairwise']

    # Keep only numeric columns
    data = data._get_numeric_data()
    # Remove columns with constant value and/or NaN
    data = data.loc[:, data.nunique(dropna=True) >= 2]
    # Extract columns names
    keys = data.columns.tolist()

    # First ensure that columns is a list
    if isinstance(columns, (str, tuple)):
        columns = [columns]

    def traverse(o, tree_types=(list, tuple)):
        """Helper function to flatten nested lists.
        From https://stackoverflow.com/a/6340578
        """
        if isinstance(o, tree_types):
            for value in o:
                for subvalue in traverse(value, tree_types):
                    yield subvalue
        else:
            yield o

    # Check if columns index has multiple levels
    if isinstance(data.columns, pd.core.index.MultiIndex):
        multi_index = True
        if columns is not None:
            # Simple List with one element: [('L0', 'L1')]
            # Simple list with >= 2 elements: [('L0', 'L1'), ('L0', 'L2')]
            # Nested lists: [[('L0', 'L1')], ...] or [..., [('L0', 'L1')]]
            col_flatten = list(traverse(columns, tree_types=list))
            assert all(isinstance(c, (tuple, type(None))) for c in col_flatten)
    else:
        multi_index = False

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
                # Check that this column exist
                if columns[0] not in keys:
                    msg = ('"%s" is not in data or is not numeric.'
                           % columns[0])
                    raise ValueError(msg)
                others = [e for e in keys if e != columns[0]]
                combs = list(product(columns, others))
            else:
                # Combinations between all specified columns ['a', 'b', 'c']
                # Make sure that we keep numeric columns
                columns = [c for c in columns if c in keys]
                if len(columns) == 1:
                    # If only one-column is left, equivalent to ['a']
                    others = [e for e in keys if e != columns[0]]
                    combs = list(product(columns, others))
                else:
                    # combinations between ['a', 'b', 'c']
                    combs = list(combinations(columns, 2))

    combs = np.array(combs)
    if len(combs) == 0:
        raise ValueError("No column combination found. Please make sure that "
                         "the specified columns exist in the dataframe, are "
                         "numeric, and contains at least two unique values.")

    # Initialize empty dataframe
    if multi_index:
        X = list(zip(combs[:, 0, 0], combs[:, 0, 1]))
        Y = list(zip(combs[:, 1, 0], combs[:, 1, 1]))
    else:
        X = combs[:, 0]
        Y = combs[:, 1]
    stats = pd.DataFrame({'X': X, 'Y': Y, 'method': method, 'tail': tail},
                         index=range(len(combs)),
                         columns=['X', 'Y', 'method', 'tail', 'n', 'outliers',
                                  'r', 'CI95%', 'r2', 'adj_r2', 'p-val',
                                  'BF10', 'power'])

    # Now we check if covariates are present
    if covar is not None:
        assert isinstance(covar, (str, list)), 'covar must be list or string.'
        if isinstance(covar, str):
            covar = [covar]
        # Check that columns exist and are numeric
        assert all([c in keys for c in covar]), 'covar not in data or not num.'
        # And we make sure that X or Y does not contain covar
        stats = stats[~stats[['X', 'Y']].isin(covar).any(1)]
        stats = stats.reset_index(drop=True)
        if stats.shape[0] == 0:
            raise ValueError("No column combination found. Please make sure "
                             "that the specified columns and covar exist in "
                             "the dataframe, are numeric, and contains at "
                             "least two unique values.")

    # Listwise deletion of missing values
    if nan_policy == 'listwise':
        all_cols = np.unique(stats[['X', 'Y']].values).tolist()
        if covar is not None:
            all_cols.extend(covar)
        data = data[all_cols].dropna()

    # Compute pairwise correlations and fill dataframe
    dvs = ['n', 'r', 'CI95%', 'r2', 'adj_r2', 'p-val', 'power']
    dvs_out = dvs + ['outliers']
    dvs_bf10 = dvs + ['BF10']
    for i in range(stats.shape[0]):
        col1, col2 = stats.at[i, 'X'], stats.at[i, 'Y']
        if covar is None:
            cor_st = corr(data[col1].values, data[col2].values, tail=tail,
                          method=method)
        else:
            cor_st = partial_corr(data=data, x=col1, y=col2, covar=covar,
                                  tail=tail, method=method)
        cor_st_keys = cor_st.columns.tolist()
        if 'BF10' in cor_st_keys:
            stats.loc[i, dvs_bf10] = cor_st[dvs_bf10].values
        elif 'outliers' in cor_st_keys:
            stats.loc[i, dvs_out] = cor_st[dvs_out].values
        else:
            stats.loc[i, dvs] = cor_st[dvs].values

    # Force conversion to numeric
    stats = stats.astype({'r': float, 'r2': float, 'adj_r2': float,
                          'n': int, 'p-val': float, 'outliers': float,
                          'power': float})

    # Multiple comparisons
    stats = stats.rename(columns={'p-val': 'p-unc'})
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
    stats['z'] = np.round(np.arctanh(stats['r'].values), 3)

    col_order = ['X', 'Y', 'method', 'tail', 'n', 'outliers', 'r', 'CI95%',
                 'r2', 'adj_r2', 'z', 'p-unc', 'p-corr', 'p-adjust',
                 'BF10', 'power']

    # Reorder columns and remove empty ones
    stats = stats.reindex(columns=col_order).dropna(how='all', axis=1)

    # Add covariates names if present
    if covar is not None:
        stats.insert(loc=3, column='covar', value=str(covar))

    if export_filename is not None:
        _export_table(stats, export_filename)
    return stats
