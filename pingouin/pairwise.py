# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
import pandas as pd
import pandas_flavor as pf
from itertools import combinations, product
from pingouin.config import options
from pingouin.parametric import anova
from pingouin.multicomp import multicomp
from pingouin.effsize import compute_effsize, convert_effsize
from pingouin.utils import (remove_rm_na, _check_dataframe, _flatten_list,
                            _postprocess_dataframe)

__all__ = ["pairwise_ttests", "pairwise_tukey", "pairwise_gameshowell",
           "pairwise_corr"]


@pf.register_dataframe_method
def pairwise_ttests(data=None, dv=None, between=None, within=None, subject=None,
                    parametric=True, marginal=True, alpha=.05, alternative='two-sided',
                    padjust='none', effsize='hedges', correction='auto', nan_policy='listwise',
                    return_desc=False, interaction=True, within_first=True):
    """Pairwise T-tests.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    dv : string
        Name of column containing the dependent variable.
    between : string or list with 2 elements
        Name of column(s) containing the between-subject factor(s).

        .. warning:: Note that Pingouin gives slightly different T and
            p-values compared to JASP posthoc tests for 2-way factorial design,
            because Pingouin does not pool the standard error for each factor,
            but rather calculate each pairwise T-test completely independent
            of others.
    within : string or list with 2 elements
        Name of column(s) containing the within-subject factor(s), i.e. the
        repeated measurements.
    subject : string
        Name of column containing the subject identifier. This is mandatory
        when ``within`` is specified.
    parametric : boolean
        If True (default), use the parametric :py:func:`ttest` function.
        If False, use :py:func:`pingouin.wilcoxon` or :py:func:`pingouin.mwu`
        for paired or unpaired samples, respectively.
    marginal : boolean
        If True, the between-subject pairwise T-test(s) will be calculated
        after averaging across all levels of the within-subject factor in mixed
        design. This is recommended to avoid violating the assumption of
        independence and conflating the degrees of freedom by the
        number of repeated measurements.

        .. versionadded:: 0.3.2
    alpha : float
        Significance level
    alternative : string
        Defines the alternative hypothesis, or tail of the test. Must be one of
        "two-sided" (default), "greater" or "less". Both "greater" and "less" return one-sided
        p-values. "greater" tests against the alternative hypothesis that the mean of ``x``
        is greater than the mean of ``y``.
    padjust : string
        Method used for testing and adjustment of pvalues.

        * ``'none'``: no correction
        * ``'bonf'``: one-step Bonferroni correction
        * ``'sidak'``: one-step Sidak correction
        * ``'holm'``: step-down method using Bonferroni adjustments
        * ``'fdr_bh'``: Benjamini/Hochberg FDR correction
        * ``'fdr_by'``: Benjamini/Yekutieli FDR correction
    effsize : string or None
        Effect size type. Available methods are:

        * ``'none'``: no effect size
        * ``'cohen'``: Unbiased Cohen d
        * ``'hedges'``: Hedges g
        * ``'r'``: Pearson correlation coefficient
        * ``'eta-square'``: Eta-square
        * ``'odds-ratio'``: Odds ratio
        * ``'AUC'``: Area Under the Curve
        * ``'CLES'``: Common Language Effect Size
    correction : string or boolean
        For unpaired two sample T-tests, specify whether or not to correct for
        unequal variances using Welch separate variances T-test. If `'auto'`,
        it will automatically uses Welch T-test when the sample sizes are
        unequal, as recommended by Zimmerman 2004.

        .. versionadded:: 0.3.2
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
    within_first : boolean
        Determines the order of the interaction in mixed design. Pingouin will
        return within * between when this parameter is set to True (default),
        and between * within otherwise.

        .. versionadded:: 0.3.6

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'Contrast'``: Contrast (= independent variable or interaction)
        * ``'A'``: Name of first measurement
        * ``'B'``: Name of second measurement
        * ``'Paired'``: indicates whether the two measurements are paired or
          independent
        * ``'Parametric'``: indicates if (non)-parametric tests were used
        * ``'T'``: T statistic (only if parametric=True)
        * ``'U-val'``: Mann-Whitney U stat (if parametric=False and unpaired
          data)
        * ``'W-val'``: Wilcoxon W stat (if parametric=False and paired data)
        * ``'dof'``: degrees of freedom (only if parametric=True)
        * ``'alternative'``: tail of the test
        * ``'p-unc'``: Uncorrected p-values
        * ``'p-corr'``: Corrected p-values
        * ``'p-adjust'``: p-values correction method
        * ``'BF10'``: Bayes Factor
        * ``'hedges'``: effect size (or any effect size defined in
          ``effsize``)

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
    same results as ['col2', 'col1']. Furthermore, the interaction will only be
    calculated if ``interaction=True``.

    If ``between`` is a list with two elements, the output
    model is between1 + between2 + between1 * between2.

    Similarly, if ``within`` is a list with two elements, the output model is
    within1 + within2 + within1 * within2.

    If both ``between`` and ``within`` are specified, the output model is
    within + between + within * between (= mixed design), unless
    ``within_first=False`` in which case the model becomes between + within +
    between * within.

    Missing values in repeated measurements are automatically removed using a
    listwise (default) or pairwise deletion strategy. However, you should be
    very careful since it can result in undesired values removal (especially
    for the interaction effect). We strongly recommend that you preprocess
    your data and remove the missing values before using this function.

    .. warning:: Versions of Pingouin below 0.3.9 gave INCORRECT RESULTS
        for one-way and two-way repeated measures when the dataframe was not
        sorted by subject (see issue 151). Please make sure you're using the
        LATEST VERSION of Pingouin, and always DOUBLE CHECK your results with
        another statistical software.

    Examples
    --------
    For more examples, please refer to the `Jupyter notebooks
    <https://github.com/raphaelvallat/pingouin/blob/master/notebooks/01_ANOVA.ipynb>`_

    1. One between-subject factor

    >>> import pandas as pd
    >>> import pingouin as pg
    >>> pd.set_option('expand_frame_repr', False)
    >>> pd.set_option('max_columns', 20)
    >>> df = pg.read_dataset('mixed_anova.csv')
    >>> pg.pairwise_ttests(dv='Scores', between='Group', data=df).round(3)
      Contrast        A           B  Paired  Parametric     T    dof alternative  p-unc   BF10  hedges
    0    Group  Control  Meditation   False        True -2.29  178.0   two-sided  0.023  1.813   -0.34

    2. One within-subject factor

    >>> post_hocs = pg.pairwise_ttests(dv='Scores', within='Time', subject='Subject', data=df)
    >>> post_hocs.round(3)
      Contrast        A        B  Paired  Parametric      T   dof alternative  p-unc   BF10  hedges
    0     Time   August  January    True        True -1.740  59.0   two-sided  0.087  0.582  -0.328
    1     Time   August     June    True        True -2.743  59.0   two-sided  0.008  4.232  -0.483
    2     Time  January     June    True        True -1.024  59.0   two-sided  0.310  0.232  -0.170

    3. Non-parametric pairwise paired test (wilcoxon)

    >>> pg.pairwise_ttests(dv='Scores', within='Time', subject='Subject',
    ...                    data=df, parametric=False).round(3)
      Contrast        A        B  Paired  Parametric  W-val alternative  p-unc  hedges
    0     Time   August  January    True       False  716.0   two-sided  0.144  -0.328
    1     Time   August     June    True       False  564.0   two-sided  0.010  -0.483
    2     Time  January     June    True       False  887.0   two-sided  0.840  -0.170

    4. Mixed design (within and between) with bonferroni-corrected p-values

    >>> posthocs = pg.pairwise_ttests(dv='Scores', within='Time', subject='Subject',
    ...                               between='Group', padjust='bonf', data=df)
    >>> posthocs.round(3)
           Contrast     Time        A           B Paired  Parametric      T   dof alternative  p-unc  p-corr p-adjust   BF10  hedges
    0          Time        -   August     January   True        True -1.740  59.0   two-sided  0.087   0.261     bonf  0.582  -0.328
    1          Time        -   August        June   True        True -2.743  59.0   two-sided  0.008   0.024     bonf  4.232  -0.483
    2          Time        -  January        June   True        True -1.024  59.0   two-sided  0.310   0.931     bonf  0.232  -0.170
    3         Group        -  Control  Meditation  False        True -2.248  58.0   two-sided  0.028     NaN      NaN  2.096  -0.573
    4  Time * Group   August  Control  Meditation  False        True  0.316  58.0   two-sided  0.753   1.000     bonf  0.274   0.081
    5  Time * Group  January  Control  Meditation  False        True -1.434  58.0   two-sided  0.157   0.471     bonf  0.619  -0.365
    6  Time * Group     June  Control  Meditation  False        True -2.744  58.0   two-sided  0.008   0.024     bonf  5.593  -0.699

    5. Two between-subject factors. The order of the ``between`` factors matters!

    >>> pg.pairwise_ttests(dv='Scores', between=['Group', 'Time'], data=df).round(3)
           Contrast       Group        A           B Paired  Parametric      T    dof alternative  p-unc     BF10  hedges
    0         Group           -  Control  Meditation  False        True -2.290  178.0   two-sided  0.023    1.813  -0.340
    1          Time           -   August     January  False        True -1.806  118.0   two-sided  0.074    0.839  -0.328
    2          Time           -   August        June  False        True -2.660  118.0   two-sided  0.009    4.499  -0.483
    3          Time           -  January        June  False        True -0.934  118.0   two-sided  0.352    0.288  -0.170
    4  Group * Time     Control   August     January  False        True -0.383   58.0   two-sided  0.703    0.279  -0.098
    5  Group * Time     Control   August        June  False        True -0.292   58.0   two-sided  0.771    0.272  -0.074
    6  Group * Time     Control  January        June  False        True  0.045   58.0   two-sided  0.964    0.263   0.011
    7  Group * Time  Meditation   August     January  False        True -2.188   58.0   two-sided  0.033    1.884  -0.558
    8  Group * Time  Meditation   August        June  False        True -4.040   58.0   two-sided  0.000  148.302  -1.030
    9  Group * Time  Meditation  January        June  False        True -1.442   58.0   two-sided  0.155    0.625  -0.367

    6. Same but without the interaction, and using a directional test

    >>> df.pairwise_ttests(dv='Scores', between=['Group', 'Time'], alternative="less",
    ...                    interaction=False).round(3)
      Contrast        A           B  Paired  Parametric      T    dof alternative  p-unc   BF10  hedges
    0    Group  Control  Meditation   False        True -2.290  178.0        less  0.012  3.626  -0.340
    1     Time   August     January   False        True -1.806  118.0        less  0.037  1.679  -0.328
    2     Time   August        June   False        True -2.660  118.0        less  0.004  8.998  -0.483
    3     Time  January        June   False        True -0.934  118.0        less  0.176  0.577  -0.170
    """
    from .parametric import ttest
    from .nonparametric import wilcoxon, mwu

    # Safety checks
    _check_dataframe(dv=dv, between=between, within=within, subject=subject,
                     effects='all', data=data)
    assert alternative in ['two-sided', 'greater', 'less'], (
        "Alternative must be one of 'two-sided' (default), 'greater' or 'less'.")
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
                 'dof', 'alternative', 'p-unc', 'p-corr', 'p-adjust', 'BF10',
                 effsize]

    if contrast in ['simple_within', 'simple_between']:
        # OPTION A: SIMPLE MAIN EFFECTS, WITHIN OR BETWEEN
        paired = True if contrast == 'simple_within' else False
        col = within if contrast == 'simple_within' else between

        if contrast == 'simple_within':
            # Check missing values
            isna = data[dv].isnull().to_numpy().any()
            # Only if nan_policy == 'listwise'. For pairwise deletion,
            # missing values will be removed directly in the lower-level
            # functions (e.g. pg.ttest)
            if all([isna, nan_policy == 'listwise']):
                data = remove_rm_na(dv=dv, within=within, subject=subject,
                                    data=data)
            else:
                # BUGFIX 0.3.9: If repeated measures, we sort by subject and
                # within and aggregate other repeated measures factor using
                # the mean. Note that these two steps are already done in the
                # remove_rm_na function.
                data = data.groupby([subject, within], observed=True, sort=True
                                    )[dv].mean().reset_index()
            # Now we check that subjects are present in all conditions
            # For example, if we have four subjects and 3 conditions,
            # and if subject 2 have missing data at the third condition,
            # we still need a row with missing values for this subject.
            if data.groupby(within,
                            observed=True)[subject].count().nunique() != 1:
                raise ValueError("Repeated measures dataframe is not balanced."
                                 " `Subjects` must have the same number of "
                                 "elements in all conditions, "
                                 "even when missing values are present.")

        # Extract levels of the grouping variable, sorted in alphabetical order
        grp_col = data.groupby(col, sort=True, observed=True)[dv]
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
        cols_str = ['Contrast', 'Time', 'A', 'B', 'alternative', 'p-adjust', 'BF10']
        cols_bool = ['Parametric', 'Paired']
        stats[cols_str] = stats[cols_str].astype(object)
        stats[cols_bool] = stats[cols_bool].astype(bool)

        # Fill str columns
        stats.loc[:, 'A'] = A
        stats.loc[:, 'B'] = B
        stats.loc[:, 'Contrast'] = col
        stats.loc[:, 'alternative'] = alternative
        stats.loc[:, 'Paired'] = paired

        # For max precision, make sure rounding is disabled
        old_options = options.copy()
        options['round'] = None

        for i in range(stats.shape[0]):
            col1, col2 = stats.at[i, 'A'], stats.at[i, 'B']
            x = grp_col.get_group(col1).to_numpy(dtype=np.float64)
            y = grp_col.get_group(col2).to_numpy(dtype=np.float64)
            if parametric:
                stat_name = 'T'
                df_ttest = ttest(x, y, paired=paired, alternative=alternative,
                                 correction=correction)
                stats.at[i, 'BF10'] = df_ttest.at['T-test', 'BF10']
                stats.at[i, 'dof'] = df_ttest.at['T-test', 'dof']
            else:
                if paired:
                    stat_name = 'W-val'
                    df_ttest = wilcoxon(x, y, alternative=alternative)
                else:
                    stat_name = 'U-val'
                    df_ttest = mwu(x, y, alternative=alternative)

            options.update(old_options)  # restore options

            # Compute Hedges / Cohen
            ef = compute_effsize(x=x, y=y, eftype=effsize, paired=paired)

            if return_desc:
                stats.at[i, 'mean(A)'] = np.nanmean(x)
                stats.at[i, 'mean(B)'] = np.nanmean(y)
                stats.at[i, 'std(A)'] = np.nanstd(x, ddof=1)
                stats.at[i, 'std(B)'] = np.nanstd(y, ddof=1)
            stats.at[i, stat_name] = df_ttest[stat_name].iat[0]
            stats.at[i, 'p-unc'] = df_ttest['p-val'].iat[0]
            stats.at[i, effsize] = ef

        # Multiple comparisons
        padjust = None if stats['p-unc'].size <= 1 else padjust
        if padjust is not None:
            if padjust.lower() != 'none':
                _, stats['p-corr'] = multicomp(stats['p-unc'].to_numpy(),
                                               alpha=alpha, method=padjust)
                stats['p-adjust'] = padjust
        else:
            stats['p-corr'] = None
            stats['p-adjust'] = None
    else:
        # Multiple factors
        if contrast == 'multiple_between':
            # B1: BETWEEN1 + BETWEEN2 + BETWEEN1 * BETWEEN2
            factors = between
            fbt = factors
            fwt = [None, None]
            paired = False  # the interaction is not paired
            agg = [False, False]
            # TODO: add a pool SD option, as in JASP and JAMOVI?
        elif contrast == 'multiple_within':
            # B2: WITHIN1 + WITHIN2 + WITHIN1 * WITHIN2
            factors = within
            fbt = [None, None]
            fwt = factors
            paired = True
            agg = [True, True]  # Calculate marginal means for both factors
        else:
            # B3: WITHIN + BETWEEN + INTERACTION
            # Decide which order should be reported
            if within_first:
                # within + between + within * between
                factors = [within, between]
                fbt = [None, between]
                fwt = [within, None]
                paired = False  # only for interaction
                agg = [False, True]
            else:
                # between + within + between * within
                factors = [between, within]
                fbt = [between, None]
                fwt = [None, within]
                paired = True
                agg = [True, False]

        stats = pd.DataFrame()
        for i, f in enumerate(factors):
            # Introduced in Pingouin v0.3.2
            # Note that is only has an impact in the between test of mixed
            # designs. Indeed, a similar groupby is applied by default on
            # each within-subject factor of a two-way repeated measures design.
            if all([agg[i], marginal]):
                tmp = data.groupby([subject, f], as_index=False,
                                   observed=True, sort=True).mean()
            else:
                tmp = data
            # Recursive call to pairwise_ttests
            stats = stats.append(pairwise_ttests(dv=dv,
                                                 between=fbt[i],
                                                 within=fwt[i],
                                                 subject=subject,
                                                 data=tmp,
                                                 parametric=parametric,
                                                 marginal=marginal,
                                                 alpha=alpha,
                                                 alternative=alternative,
                                                 padjust=padjust,
                                                 effsize=effsize,
                                                 correction=correction,
                                                 nan_policy=nan_policy,
                                                 return_desc=return_desc),
                                 ignore_index=True, sort=False)

        # Then compute the interaction between the factors
        if interaction:
            nrows = stats.shape[0]
            # BUGFIX 0.3.9: If subject is present, make sure that we respect
            # the order of subjects.
            if subject is not None:
                data = data.set_index(subject).sort_index()
            # Extract interaction levels, sorted in alphabetical order
            grp_fac1 = data.groupby(factors[0], observed=True, sort=True)[dv]
            grp_fac2 = data.groupby(factors[1], observed=True, sort=True)[dv]
            grp_both = data.groupby(factors, observed=True, sort=True)[dv]
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
            stats.loc[idxiter, 'alternative'] = alternative
            stats.loc[idxiter, 'A'] = combs[:, 1]
            stats.loc[idxiter, 'B'] = combs[:, 2]

            # For max precision, make sure rounding is disabled
            old_options = options.copy()
            options['round'] = None

            for i, comb in enumerate(combs):
                ic = nrows + i  # Take into account previous rows
                fac1, col1, col2 = comb
                x = grp_both.get_group((fac1, col1)).to_numpy(dtype=np.float64)
                y = grp_both.get_group((fac1, col2)).to_numpy(dtype=np.float64)
                ef = compute_effsize(x=x, y=y, eftype=effsize, paired=paired)
                if parametric:
                    stat_name = 'T'
                    df_ttest = ttest(x, y, paired=paired, alternative=alternative,
                                     correction=correction)
                    stats.at[ic, 'BF10'] = df_ttest.at['T-test', 'BF10']
                    stats.at[ic, 'dof'] = df_ttest.at['T-test', 'dof']
                else:
                    if paired:
                        stat_name = 'W-val'
                        df_ttest = wilcoxon(x, y, alternative=alternative)
                    else:
                        stat_name = 'U-val'
                        df_ttest = mwu(x, y, alternative=alternative)

                options.update(old_options)  # restore options

                # Append to stats
                if return_desc:
                    stats.at[ic, 'mean(A)'] = np.nanmean(x)
                    stats.at[ic, 'mean(B)'] = np.nanmean(y)
                    stats.at[ic, 'std(A)'] = np.nanstd(x, ddof=1)
                    stats.at[ic, 'std(B)'] = np.nanstd(y, ddof=1)
                stats.at[ic, stat_name] = df_ttest[stat_name].iat[0]
                stats.at[ic, 'p-unc'] = df_ttest['p-val'].iat[0]
                stats.at[ic, effsize] = ef

            # Multi-comparison columns
            if padjust is not None and padjust.lower() != 'none':
                _, pcor = multicomp(stats.loc[idxiter, 'p-unc'].to_numpy(),
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

    return _postprocess_dataframe(stats)


@pf.register_dataframe_method
def pairwise_tukey(data=None, dv=None, between=None, effsize='hedges'):
    """Pairwise Tukey-HSD post-hoc test.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    dv : string
        Name of column containing the dependent variable.
    between: string
        Name of column containing the between factor.
    effsize : string or None
        Effect size type. Available methods are:

        * ``'none'``: no effect size
        * ``'cohen'``: Unbiased Cohen d
        * ``'hedges'``: Hedges g
        * ``'r'``: Pearson correlation coefficient
        * ``'eta-square'``: Eta-square
        * ``'odds-ratio'``: Odds ratio
        * ``'AUC'``: Area Under the Curve
        * ``'CLES'``: Common Language Effect Size

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'A'``: Name of first measurement
        * ``'B'``: Name of second measurement
        * ``'mean(A)'``: Mean of first measurement
        * ``'mean(B)'``: Mean of second measurement
        * ``'diff'``: Mean difference (= mean(A) - mean(B))
        * ``'se'``: Standard error
        * ``'T'``: T-values
        * ``'p-tukey'``: Tukey-HSD corrected p-values
        * ``'hedges'``: Hedges effect size (or any effect size defined in
          ``effsize``)

    See also
    --------
    pairwise_ttests, pairwise_gameshowell

    Notes
    -----
    Tukey HSD post-hoc [1]_ is best for balanced one-way ANOVA.

    It has been proven to be conservative for one-way ANOVA with unequal
    sample sizes. However, it is not robust if the groups have unequal
    variances, in which case the Games-Howell test is more adequate.
    Tukey HSD is not valid for repeated measures ANOVA.
    Only one-way ANOVA design are supported.

    The T-values are defined as:

    .. math::

        t = \\frac{\\overline{x}_i - \\overline{x}_j}
        {\\sqrt{2 \\cdot \\text{MS}_w / n}}

    where :math:`\\overline{x}_i` and :math:`\\overline{x}_j` are the means of
    the first and second group, respectively, :math:`\\text{MS}_w` the mean
    squares of the error (computed using ANOVA) and :math:`n` the sample size.

    If the sample sizes are unequal, the Tukey-Kramer procedure is
    automatically used:

    .. math::

        t = \\frac{\\overline{x}_i - \\overline{x}_j}{\\sqrt{\\frac{MS_w}{n_i}
        + \\frac{\\text{MS}_w}{n_j}}}

    where :math:`n_i` and :math:`n_j` are the sample sizes of the first and
    second group, respectively.

    The p-values are then approximated using the Studentized range distribution
    :math:`Q(\\sqrt2|t_i|, r, N - r)` where :math:`r` is the total number of
    groups and :math:`N` is the total sample size.

    .. warning:: Versions of Pingouin below 0.3.10 used a wrong algorithm for
        the studentized range approximation [2]_, which resulted in (slightly)
        incorrect p-values. Please make sure you're using the
        LATEST VERSION of Pingouin, and always DOUBLE CHECK your results with
        another statistical software.

    References
    ----------
    .. [1] Tukey, John W. "Comparing individual means in the analysis of
           variance." Biometrics (1949): 99-114.

    .. [2] Gleason, John R. "An accurate, non-iterative approximation for
           studentized range quantiles." Computational statistics & data
           analysis 31.2 (1999): 147-158.

    Examples
    --------
    Pairwise Tukey post-hocs on the Penguins dataset.

    >>> import pingouin as pg
    >>> df = pg.read_dataset('penguins')
    >>> df.pairwise_tukey(dv='body_mass_g', between='species').round(3)
               A          B   mean(A)   mean(B)      diff      se       T  p-tukey  hedges
    0     Adelie  Chinstrap  3700.662  3733.088   -32.426  67.512  -0.480    0.869  -0.070
    1     Adelie     Gentoo  3700.662  5076.016 -1375.354  56.148 -24.495    0.001  -2.967
    2  Chinstrap     Gentoo  3733.088  5076.016 -1342.928  69.857 -19.224    0.001  -2.894
    """
    from statsmodels.stats.libqsturng import psturng

    # First compute the ANOVA
    # For max precision, make sure rounding is disabled
    old_options = options.copy()
    options['round'] = None
    aov = anova(dv=dv, data=data, between=between, detailed=True)
    options.update(old_options)  # Restore original options
    df = aov.at[1, 'DF']
    ng = aov.at[0, 'DF'] + 1
    grp = data.groupby(between, observed=True)[dv]  # default is sort=True
    # Careful: pd.unique does NOT sort whereas numpy does
    # The line below should be equal to labels = np.unique(data[between])
    # However, this does not work if between is a Categorical column, because
    # Pandas applies a custom, not alphabetical, sorting.
    # See https://github.com/raphaelvallat/pingouin/issues/111
    labels = np.array(list(grp.groups.keys()))
    n = grp.count().to_numpy()
    gmeans = grp.mean().to_numpy()
    gvar = aov.at[1, 'MS'] / n

    # Pairwise combinations
    g1, g2 = np.array(list(combinations(np.arange(ng), 2))).T
    mn = gmeans[g1] - gmeans[g2]
    se = np.sqrt(gvar[g1] + gvar[g2])
    tval = mn / se

    # Critical values and p-values
    # from statsmodels.stats.libqsturng import qsturng
    # crit = qsturng(1 - alpha, ng, df) / np.sqrt(2)
    pval = psturng(np.sqrt(2) * np.abs(tval), ng, df)

    # Uncorrected p-values
    # from scipy.stats import t
    # punc = t.sf(np.abs(tval), n[g1].size + n[g2].size - 2) * 2

    # Effect size
    d = tval * np.sqrt(1 / n[g1] + 1 / n[g2])
    ef = convert_effsize(d, 'cohen', effsize, n[g1], n[g2])

    # Create dataframe
    stats = pd.DataFrame({
                         'A': labels[g1],
                         'B': labels[g2],
                         'mean(A)': gmeans[g1],
                         'mean(B)': gmeans[g2],
                         'diff': mn,
                         'se': se,
                         'T': tval,
                         'p-tukey': pval,
                         effsize: ef,
                         })
    return _postprocess_dataframe(stats)


def pairwise_gameshowell(data=None, dv=None, between=None, effsize='hedges'):
    """Pairwise Games-Howell post-hoc test.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame
    dv : string
        Name of column containing the dependent variable.
    between: string
        Name of column containing the between factor.
    effsize : string or None
        Effect size type. Available methods are:

        * ``'none'``: no effect size
        * ``'cohen'``: Unbiased Cohen d
        * ``'hedges'``: Hedges g
        * ``'r'``: Pearson correlation coefficient
        * ``'eta-square'``: Eta-square
        * ``'odds-ratio'``: Odds ratio
        * ``'AUC'``: Area Under the Curve
        * ``'CLES'``: Common Language Effect Size

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`
        Stats summary:

        * ``'A'``: Name of first measurement
        * ``'B'``: Name of second measurement
        * ``'mean(A)'``: Mean of first measurement
        * ``'mean(B)'``: Mean of second measurement
        * ``'diff'``: Mean difference (= mean(A) - mean(B))
        * ``'se'``: Standard error
        * ``'T'``: T-values
        * ``'df'``: adjusted degrees of freedom
        * ``'pval'``: Games-Howell corrected p-values
        * ``'hedges'``: Hedges effect size (or any effect size defined in
          ``effsize``)

    See also
    --------
    pairwise_ttests, pairwise_tukey

    Notes
    -----
    Games-Howell [1]_ is very similar to the Tukey HSD post-hoc test but is
    much more robust to heterogeneity of variances. While the
    Tukey-HSD post-hoc is optimal after a classic one-way ANOVA, the
    Games-Howell is optimal after a Welch ANOVA. Please note that Games-Howell
    is not valid for repeated measures ANOVA.
    Only one-way ANOVA design are supported.

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
    :math:`Q(\\sqrt2|t_i|, r, v_i)`.

    .. warning:: Versions of Pingouin below 0.3.10 used a wrong algorithm for
        the studentized range approximation [2]_, which resulted in (slightly)
        incorrect p-values. Please make sure you're using the
        LATEST VERSION of Pingouin, and always DOUBLE CHECK your results with
        another statistical software.

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
    Pairwise Games-Howell post-hocs on the Penguins dataset.

    >>> import pingouin as pg
    >>> df = pg.read_dataset('penguins')
    >>> pg.pairwise_gameshowell(data=df, dv='body_mass_g',
    ...                         between='species').round(3)
               A          B   mean(A)   mean(B)      diff      se       T       df   pval  hedges
    0     Adelie  Chinstrap  3700.662  3733.088   -32.426  59.706  -0.543  152.455  0.834  -0.079
    1     Adelie     Gentoo  3700.662  5076.016 -1375.354  58.811 -23.386  249.643  0.001  -2.833
    2  Chinstrap     Gentoo  3733.088  5076.016 -1342.928  65.103 -20.628  170.404  0.001  -3.105
    """
    from statsmodels.stats.libqsturng import psturng

    # Check the dataframe
    _check_dataframe(dv=dv, between=between, effects='between', data=data)

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)

    # Extract infos
    ng = data[between].nunique()
    grp = data.groupby(between, observed=True)[dv]  # default is sort=True
    # Careful: pd.unique does NOT sort whereas numpy does
    # The line below should be equal to labels = np.unique(data[between])
    # However, this does not work if between is a Categorical column, because
    # Pandas applies a custom, not alphabetical, sorting.
    # See https://github.com/raphaelvallat/pingouin/issues/111
    labels = np.array(list(grp.groups.keys()))
    n = grp.count().to_numpy()
    gmeans = grp.mean().to_numpy()
    gvars = grp.var().to_numpy()

    # Pairwise combinations
    g1, g2 = np.array(list(combinations(np.arange(ng), 2))).T
    mn = gmeans[g1] - gmeans[g2]
    se = np.sqrt(gvars[g1] / n[g1] + gvars[g2] / n[g2])
    tval = mn / np.sqrt(gvars[g1] / n[g1] + gvars[g2] / n[g2])
    df = (gvars[g1] / n[g1] + gvars[g2] / n[g2])**2 / \
         ((((gvars[g1] / n[g1])**2) / (n[g1] - 1)) +
          (((gvars[g2] / n[g2])**2) / (n[g2] - 1)))

    # Compute corrected p-values
    pval = psturng(np.sqrt(2) * np.abs(tval), ng, df)

    # Uncorrected p-values
    # from scipy.stats import t
    # punc = t.sf(np.abs(tval), n[g1].size + n[g2].size - 2) * 2

    # Effect size
    d = tval * np.sqrt(1 / n[g1] + 1 / n[g2])
    ef = convert_effsize(d, 'cohen', effsize, n[g1], n[g2])

    # Create dataframe
    stats = pd.DataFrame({
                         'A': labels[g1],
                         'B': labels[g2],
                         'mean(A)': gmeans[g1],
                         'mean(B)': gmeans[g2],
                         'diff': mn,
                         'se': se,
                         'T': tval,
                         'df': df,
                         'pval': pval,
                         effsize: ef,
                         })
    return _postprocess_dataframe(stats)


@pf.register_dataframe_method
def pairwise_corr(data, columns=None, covar=None, alternative='two-sided',
                  method='pearson', padjust='none', nan_policy='pairwise'):
    """Pairwise (partial) correlations between columns of a pandas dataframe.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    columns : list or str
        Column names in data:

        * ``["a", "b", "c"]``: combination between columns a, b, and c.
        * ``["a"]``: product between a and all the other numeric columns.
        * ``[["a"], ["b", "c"]]``: product between ["a"] and ["b", "c"].
        * ``[["a", "d"], ["b", "c"]]``: product between ["a", "d"] and
          ["b", "c"].
        * ``[["a", "d"], None]``: product between ["a", "d"] and all other
          numeric columns in dataframe.

        If column is None, the function will return the pairwise correlation
        between the combination of all the numeric columns in data.
        See the examples section for more details on this.
    covar : None, string or list
        Covariate(s) for partial correlation. Must be one or more columns
        in data. Use a list if there are more than one covariate. If
        ``covar`` is not None, a partial correlation will be computed using
        :py:func:`pingouin.partial_corr` function.

        .. important:: Only ``method='pearson'`` and ``method='spearman'``
            are currently supported in partial correlation.
    alternative : string
        Defines the alternative hypothesis, or tail of the correlation. Must be one of
        "two-sided" (default), "greater" or "less". Both "greater" and "less" return a one-sided
        p-value. "greater" tests against the alternative hypothesis that the correlation is
        positive (greater than zero), "less" tests against the hypothesis that the correlation is
        negative.
    method : string
        Correlation type:

        * ``'pearson'``: Pearson :math:`r` product-moment correlation
        * ``'spearman'``: Spearman :math:`\\rho` rank-order correlation
        * ``'kendall'``: Kendall's :math:`\\tau_B` correlation
          (for ordinal data)
        * ``'bicor'``: Biweight midcorrelation (robust)
        * ``'percbend'``: Percentage bend correlation (robust)
        * ``'shepherd'``: Shepherd's pi correlation (robust)
        * ``'skipped'``: Skipped correlation (robust)
    padjust : string
        Method used for testing and adjustment of pvalues.

        * ``'none'``: no correction
        * ``'bonf'``: one-step Bonferroni correction
        * ``'sidak'``: one-step Sidak correction
        * ``'holm'``: step-down method using Bonferroni adjustments
        * ``'fdr_bh'``: Benjamini/Hochberg FDR correction
        * ``'fdr_by'``: Benjamini/Yekutieli FDR correction
    nan_policy : string
        Can be ``'listwise'`` for listwise deletion of missing values
        (= complete-case analysis) or ``'pairwise'`` (default) for the more
        liberal pairwise deletion (= available-case analysis).

        .. versionadded:: 0.2.9

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'X'``: Name(s) of first columns.
        * ``'Y'``: Name(s) of second columns.
        * ``'method'``: Correlation type.
        * ``'covar'``: List of specified covariate(s), only when covariates are passed.
        * ``'alternative'``: Tail of the test.
        * ``'n'``: Sample size (after removal of missing values).
        * ``'r'``: Correlation coefficients.
        * ``'CI95'``: 95% parametric confidence intervals.
        * ``'p-unc'``: Uncorrected p-values.
        * ``'p-corr'``: Corrected p-values.
        * ``'p-adjust'``: P-values correction method.
        * ``'BF10'``: Bayes Factor of the alternative hypothesis (only for Pearson correlation)
        * ``'power'``: achieved power of the test (= 1 - type II error).

    Notes
    -----
    Please refer to the :py:func:`pingouin.corr()` function for a description
    of the different methods. Missing values are automatically removed from the
    data using a pairwise deletion.

    This function is more flexible and gives a much more detailed
    output than the :py:func:`pandas.DataFrame.corr()` method (i.e. p-values,
    confidence interval, Bayes Factor...). This comes however at
    an increased computational cost. While this should not be discernible for
    a dataframe with less than 10,000 rows and/or less than 20 columns, this
    function can be slow for very large datasets.

    A faster alternative to get the r-values and p-values in a matrix format is
    to use the :py:func:`pingouin.rcorr` function, which works directly as a
    :py:class:`pandas.DataFrame` method (see example below).

    This function also works with two-dimensional multi-index columns. In this
    case, columns must be list(s) of tuple(s). Please refer to this `example
    Jupyter notebook
    <https://github.com/raphaelvallat/pingouin/blob/master/notebooks/04_Correlations.ipynb>`_
    for more details.

    If and only if ``covar`` is specified, this function will compute the
    pairwise partial correlation between the variables. If you are only
    interested in computing the partial correlation matrix (i.e. the raw
    pairwise partial correlation coefficient matrix, without the p-values,
    sample sizes, etc), a better alternative is to use the
    :py:func:`pingouin.pcorr` function (see example 7).

    Examples
    --------
    1. One-sided spearman correlation corrected for multiple comparisons

    >>> import pandas as pd
    >>> import pingouin as pg
    >>> pd.set_option('expand_frame_repr', False)
    >>> pd.set_option('max_columns', 20)
    >>> data = pg.read_dataset('pairwise_corr').iloc[:, 1:]
    >>> pg.pairwise_corr(data, method='spearman', alternative='greater', padjust='bonf').round(3)
                   X                  Y    method alternative    n      r         CI95%  p-unc  p-corr p-adjust  power
    0    Neuroticism       Extraversion  spearman     greater  500 -0.325  [-0.39, 1.0]  1.000   1.000     bonf  0.000
    1    Neuroticism           Openness  spearman     greater  500 -0.028   [-0.1, 1.0]  0.735   1.000     bonf  0.012
    2    Neuroticism      Agreeableness  spearman     greater  500 -0.151  [-0.22, 1.0]  1.000   1.000     bonf  0.000
    3    Neuroticism  Conscientiousness  spearman     greater  500 -0.356  [-0.42, 1.0]  1.000   1.000     bonf  0.000
    4   Extraversion           Openness  spearman     greater  500  0.243   [0.17, 1.0]  0.000   0.000     bonf  1.000
    5   Extraversion      Agreeableness  spearman     greater  500  0.062  [-0.01, 1.0]  0.083   0.832     bonf  0.398
    6   Extraversion  Conscientiousness  spearman     greater  500  0.056  [-0.02, 1.0]  0.106   1.000     bonf  0.345
    7       Openness      Agreeableness  spearman     greater  500  0.170    [0.1, 1.0]  0.000   0.001     bonf  0.985
    8       Openness  Conscientiousness  spearman     greater  500 -0.007  [-0.08, 1.0]  0.560   1.000     bonf  0.036
    9  Agreeableness  Conscientiousness  spearman     greater  500  0.161   [0.09, 1.0]  0.000   0.002     bonf  0.976

    2. Robust two-sided biweight midcorrelation with uncorrected p-values

    >>> pcor = pg.pairwise_corr(data, columns=['Openness', 'Extraversion',
    ...                                        'Neuroticism'], method='bicor')
    >>> pcor.round(3)
                  X             Y method alternative    n      r           CI95%  p-unc  power
    0      Openness  Extraversion  bicor   two-sided  500  0.247    [0.16, 0.33]  0.000  1.000
    1      Openness   Neuroticism  bicor   two-sided  500 -0.028   [-0.12, 0.06]  0.535  0.095
    2  Extraversion   Neuroticism  bicor   two-sided  500 -0.343  [-0.42, -0.26]  0.000  1.000

    3. One-versus-all pairwise correlations

    >>> pg.pairwise_corr(data, columns=['Neuroticism']).round(3)
                 X                  Y   method alternative    n      r           CI95%  p-unc       BF10  power
    0  Neuroticism       Extraversion  pearson   two-sided  500 -0.350  [-0.42, -0.27]  0.000  6.765e+12  1.000
    1  Neuroticism           Openness  pearson   two-sided  500 -0.010    [-0.1, 0.08]  0.817      0.058  0.056
    2  Neuroticism      Agreeableness  pearson   two-sided  500 -0.134  [-0.22, -0.05]  0.003      5.122  0.854
    3  Neuroticism  Conscientiousness  pearson   two-sided  500 -0.368  [-0.44, -0.29]  0.000  2.644e+14  1.000

    4. Pairwise correlations between two lists of columns (cartesian product)

    >>> columns = [['Neuroticism', 'Extraversion'], ['Openness']]
    >>> pg.pairwise_corr(data, columns).round(3)
                  X         Y   method alternative    n      r         CI95%  p-unc       BF10  power
    0   Neuroticism  Openness  pearson   two-sided  500 -0.010  [-0.1, 0.08]  0.817      0.058  0.056
    1  Extraversion  Openness  pearson   two-sided  500  0.267  [0.18, 0.35]  0.000  5.277e+06  1.000

    5. As a Pandas method

    >>> pcor = data.pairwise_corr(covar='Neuroticism', method='spearman')

    6. Pairwise partial correlation

    >>> pg.pairwise_corr(data, covar=['Neuroticism', 'Openness'])
                   X                  Y   method                        covar alternative    n         r          CI95%     p-unc
    0   Extraversion      Agreeableness  pearson  ['Neuroticism', 'Openness']   two-sided  500 -0.038737  [-0.13, 0.05]  0.388361
    1   Extraversion  Conscientiousness  pearson  ['Neuroticism', 'Openness']   two-sided  500 -0.071427  [-0.16, 0.02]  0.111389
    2  Agreeableness  Conscientiousness  pearson  ['Neuroticism', 'Openness']   two-sided  500  0.123108   [0.04, 0.21]  0.005944

    7. Pairwise partial correlation matrix using :py:func:`pingouin.pcorr`

    >>> data[['Neuroticism', 'Openness', 'Extraversion']].pcorr().round(3)
                  Neuroticism  Openness  Extraversion
    Neuroticism         1.000     0.092        -0.360
    Openness            0.092     1.000         0.281
    Extraversion       -0.360     0.281         1.000

    8. Correlation matrix with p-values using :py:func:`pingouin.rcorr`

    >>> data[['Neuroticism', 'Openness', 'Extraversion']].rcorr()
                 Neuroticism Openness Extraversion
    Neuroticism            -                   ***
    Openness           -0.01        -          ***
    Extraversion       -0.35    0.267            -
    """
    from pingouin.correlation import corr, partial_corr

    # Check arguments
    assert alternative in ['two-sided', 'greater', 'less'], (
        "Alternative must be one of 'two-sided' (default), 'greater' or 'less'.")
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
    if isinstance(data.columns, pd.MultiIndex):
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
    stats = pd.DataFrame({'X': X, 'Y': Y, 'method': method, 'alternative': alternative},
                         index=range(len(combs)),
                         columns=['X', 'Y', 'method', 'alternative', 'n', 'outliers',
                                  'r', 'CI95%', 'p-val', 'BF10', 'power'])

    # Now we check if covariates are present
    if covar is not None:
        assert isinstance(covar, (str, list, pd.Index)), (
            'covar must be list or string.'
        )
        if isinstance(covar, str):
            covar = [covar]
        elif isinstance(covar, pd.Index):
            covar = covar.tolist()
        # Check that columns exist and are numeric
        assert all([c in keys for c in covar]), (
            'Covariate(s) are either not in data or not numeric.'
        )
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
        all_cols = np.unique(stats[['X', 'Y']].to_numpy()).tolist()
        if covar is not None:
            all_cols.extend(covar)
        data = data[all_cols].dropna()

    # For max precision, make sure rounding is disabled
    old_options = options.copy()
    options['round'] = None

    # Compute pairwise correlations and fill dataframe
    for i in range(stats.shape[0]):
        col1, col2 = stats.at[i, 'X'], stats.at[i, 'Y']
        if covar is None:
            cor_st = corr(data[col1].to_numpy(), data[col2].to_numpy(),
                          alternative=alternative, method=method)
        else:
            cor_st = partial_corr(data=data, x=col1, y=col2, covar=covar,
                                  alternative=alternative, method=method)
        cor_st_keys = cor_st.columns.tolist()

        for c in cor_st_keys:
            stats.loc[i, c] = cor_st.loc[method, c]

    options.update(old_options)  # restore options

    # Force conversion to numeric
    stats = stats.astype({
        'r': float, 'n': int, 'p-val': float, 'outliers': float,
        'power': float})

    # Multiple comparisons
    stats = stats.rename(columns={'p-val': 'p-unc'})
    padjust = None if stats['p-unc'].size <= 1 else padjust
    if padjust is not None:
        if padjust.lower() != 'none':
            reject, stats['p-corr'] = multicomp(
                stats['p-unc'].to_numpy(), method=padjust)
            stats['p-adjust'] = padjust
    else:
        stats['p-corr'] = None
        stats['p-adjust'] = None

    # Standardize correlation coefficients (Fisher z-transformation)
    # stats['z'] = np.arctanh(stats['r'].to_numpy())

    col_order = ['X', 'Y', 'method', 'alternative', 'n', 'outliers', 'r', 'CI95%',
                 'p-unc', 'p-corr', 'p-adjust', 'BF10', 'power']

    # Reorder columns and remove empty ones
    stats = stats.reindex(columns=col_order).dropna(how='all', axis=1)

    # Add covariates names if present
    if covar is not None:
        stats.insert(loc=3, column='covar', value=str(covar))

    return _postprocess_dataframe(stats)
