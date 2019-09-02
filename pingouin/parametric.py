# Author: Raphael Vallat <raphaelvallat9@gmail.com>
import warnings
import numpy as np
import pandas as pd
from scipy.stats import f
import pandas_flavor as pf
from pingouin import (_check_dataframe, remove_rm_na, remove_na, _flatten_list,
                      _export_table, bayesfactor_ttest, epsilon, sphericity)

__all__ = ["ttest", "rm_anova", "anova", "welch_anova", "mixed_anova",
           "ancova"]


def ttest(x, y, paired=False, tail='two-sided', correction='auto', r=.707):
    """T-test.

    Parameters
    ----------
    x : array_like
        First set of observations.
    y : array_like or float
        Second set of observations. If ``y`` is a single value, a one-sample
        T-test is computed.
    paired : boolean
        Specify whether the two observations are related (i.e. repeated
        measures) or independent.
    tail : string
        Specify whether the alternative hypothesis is `'two-sided'` or
        `'one-sided'`. Can also be `'greater'` or `'less'` to specify the
        direction of the test. `'greater'` tests the alternative that ``x``
        has a larger mean than ``y``. If tail is `'one-sided'`, Pingouin will
        automatically infer the one-sided alternative hypothesis based on the
        test statistic.
    correction : string or boolean
        For unpaired two sample T-tests, specify whether or not to correct for
        unequal variances using Welch separate variances T-test. If 'auto', it
        will automatically uses Welch T-test when the sample sizes are unequal,
        as recommended by Zimmerman 2004.
    r : float
        Cauchy scale factor for computing the Bayes Factor.
        Smaller values of r (e.g. 0.5), may be appropriate when small effect
        sizes are expected a priori; larger values of r are appropriate when
        large effect sizes are expected (Rouder et al 2009).
        The default is 0.707 (= :math:`\\sqrt{2} / 2`).

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`
        T-test summary::

        'T' : T-value
        'p-val' : p-value
        'dof' : degrees of freedom
        'cohen-d' : Cohen d effect size
        'CI95%' : 95% confidence intervals of the difference in means
        'power' : achieved power of the test ( = 1 - type II error)
        'BF10' : Bayes Factor of the alternative hypothesis

    See also
    --------
    mwu, wilcoxon, anova, rm_anova, pairwise_ttests, compute_effsize

    Notes
    -----
    Missing values are automatically removed from the data. If ``x`` and
    ``y`` are paired, the entire row is removed (= listwise deletion).

    The **two-sample T-test for unpaired data** is defined as:

    .. math::

        t = \\frac{\\overline{x} - \\overline{y}}
        {\\sqrt{\\frac{s^{2}_{x}}{n_{x}} + \\frac{s^{2}_{y}}{n_{y}}}}

    where :math:`\\overline{x}` and :math:`\\overline{y}` are the sample means,
    :math:`n_{x}` and :math:`n_{y}` are the sample sizes, and
    :math:`s^{2}_{x}` and :math:`s^{2}_{y}` are the sample variances.
    The degrees of freedom :math:`v` are :math:`n_x + n_y - 2` when the sample
    sizes are equal. When the sample sizes are unequal or when
    :code:`correction=True`, the Welch–Satterthwaite equation is used to
    approximate the adjusted degrees of freedom:

    .. math::

        v = \\frac{(\\frac{s^{2}_{x}}{n_{x}} + \\frac{s^{2}_{y}}{n_{y}})^{2}}
        {\\frac{(\\frac{s^{2}_{x}}{n_{x}})^{2}}{(n_{x}-1)} +
        \\frac{(\\frac{s^{2}_{y}}{n_{y}})^{2}}{(n_{y}-1)}}

    The p-value is then calculated using a T distribution with :math:`v`
    degrees of freedom.

    The T-value for **paired samples** is defined by:

    .. math:: t = \\frac{\\overline{x}_d}{s_{\\overline{x}}}

    where

    .. math:: s_{\\overline{x}} = \\frac{s_d}{\\sqrt n}

    where :math:`\\overline{x}_d` is the sample mean of the differences
    between the two paired samples, :math:`n` is the number of observations
    (sample size), :math:`s_d` is the sample standard deviation of the
    differences and :math:`s_{\\overline{x}}` is the estimated standard error
    of the mean of the differences.

    The p-value is then calculated using a T-distribution with :math:`n-1`
    degrees of freedom.

    The scaled Jeffrey-Zellner-Siow (JZS) Bayes Factor is approximated using
    the :py:func:`pingouin.bayesfactor_ttest` function.

    Results have been tested against JASP and the `t.test` R function.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm

    .. [2] Delacre, M., Lakens, D., & Leys, C. (2017). Why psychologists should
           by default use Welch’s t-test instead of Student’s t-test.
           International Review of Social Psychology, 30(1).

    .. [3] Zimmerman, D. W. (2004). A note on preliminary tests of equality of
           variances. British Journal of Mathematical and Statistical
           Psychology, 57(1), 173-181.

    .. [4] Rouder, J.N., Speckman, P.L., Sun, D., Morey, R.D., Iverson, G.,
           2009. Bayesian t tests for accepting and rejecting the null
           hypothesis. Psychon. Bull. Rev. 16, 225–237.
           https://doi.org/10.3758/PBR.16.2.225

    Examples
    --------
    1. One-sample T-test.

    >>> from pingouin import ttest
    >>> x = [5.5, 2.4, 6.8, 9.6, 4.2]
    >>> ttest(x, 4).round(2)
              T  dof       tail  p-val          CI95%  cohen-d   BF10  power
    T-test  1.4    4  two-sided   0.23  [-1.68, 5.08]     0.62  0.766   0.19

    2. Paired T-test. Since tail is `'one-sided'`, Pingouin will
    automatically infer the alternative hypothesis based on the T-value. In
    the example below, the T-value is negative so the tail is set to `'less'`..

    >>> pre = [5.5, 2.4, 6.8, 9.6, 4.2]
    >>> post = [6.4, 3.4, 6.4, 11., 4.8]
    >>> ttest(pre, post, paired=True, tail='one-sided').round(2)
               T  dof  tail  p-val          CI95%  cohen-d   BF10  power
    T-test -2.31    4  less   0.04  [-inf, -0.05]     0.25  3.122   0.12

    ..which is indeed equivalent to directly testing that ``x`` has a
    smaller mean than ``y`` (``tail = 'less'``)

    >>> ttest(pre, post, paired=True, tail='less').round(2)
               T  dof  tail  p-val          CI95%  cohen-d   BF10  power
    T-test -2.31    4  less   0.04  [-inf, -0.05]     0.25  3.122   0.12

    3. Now testing the opposite alternative hypothesis (``tail = 'greater'``)

    >>> ttest(pre, post, paired=True, tail='greater').round(2)
               T  dof     tail  p-val         CI95%  cohen-d  BF10  power
    T-test -2.31    4  greater   0.96  [-1.35, inf]     0.25  0.32   0.02

    4. Paired T-test with missing values.

    >>> import numpy as np
    >>> pre = [5.5, 2.4, np.nan, 9.6, 4.2]
    >>> post = [6.4, 3.4, 6.4, 11., 4.8]
    >>> stats = ttest(pre, post, paired=True)

    5. Independent two-sample T-test (equal sample size).

    >>> np.random.seed(123)
    >>> x = np.random.normal(loc=7, size=20)
    >>> y = np.random.normal(loc=4, size=20)
    >>> stats = ttest(x, y, correction='auto')

    6. Independent two-sample T-test (unequal sample size).

    >>> np.random.seed(123)
    >>> x = np.random.normal(loc=7, size=20)
    >>> y = np.random.normal(loc=6.5, size=15)
    >>> stats = ttest(x, y, correction='auto')
    """
    from scipy.stats import t, ttest_rel, ttest_ind, ttest_1samp
    from scipy.stats.stats import (_unequal_var_ttest_denom,
                                   _equal_var_ttest_denom)
    from pingouin import (power_ttest, power_ttest2n, compute_effsize)

    # Check tails
    possible_tails = ['two-sided', 'one-sided', 'greater', 'less']
    assert tail in possible_tails, 'Invalid tail argument.'

    x = np.asarray(x)
    y = np.asarray(y)

    if x.size != y.size and paired:
        warnings.warn("x and y have unequal sizes. Switching to "
                      "paired == False. Check your data.")
        paired = False

    # Remove rows with missing values
    x, y = remove_na(x, y, paired=paired)
    nx, ny = x.size, y.size

    if ny == 1:
        # Case one sample T-test
        tval, pval = ttest_1samp(x, y)
        dof = nx - 1
        se = np.sqrt(x.var(ddof=1) / nx)
    if ny > 1 and paired is True:
        # Case paired two samples T-test
        # Do not compute if two arrays are identical (avoid SciPy warning)
        if np.array_equal(x, y):
            warnings.warn("x and y are equals. Cannot compute T or p-value.")
            tval, pval = np.nan, np.nan
        else:
            tval, pval = ttest_rel(x, y)
        dof = nx - 1
        se = np.sqrt(np.var(x - y, ddof=1) / nx)
        bf = bayesfactor_ttest(tval, nx, ny, paired=True, r=r)
    elif ny > 1 and paired is False:
        dof = nx + ny - 2
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        # Case unpaired two samples T-test
        if correction is True or (correction == 'auto' and nx != ny):
            # Use the Welch separate variance T-test
            tval, pval = ttest_ind(x, y, equal_var=False)
            # Compute sample standard deviation
            # dof are approximated using Welch–Satterthwaite equation
            dof, se = _unequal_var_ttest_denom(vx, nx, vy, ny)
        else:
            tval, pval = ttest_ind(x, y, equal_var=True)
            _, se = _equal_var_ttest_denom(vx, nx, vy, ny)

    # Tail of the test
    # - Two-sided: default returned by SciPy
    # - One-sided: 0.5 * two-sided
    # - Greater / less: (1 - one-sided) or one-sided depending on the means
    if tail == 'one-sided':
        # Automatically decide which tail to use based on the T-value
        tail = 'greater' if tval > 0 else 'less'

    if tail == 'greater':
        pval = pval / 2 if tval > 0 else 1 - pval / 2
    elif tail == 'less':
        pval = pval / 2 if tval < 0 else 1 - pval / 2

    # Effect size
    d = compute_effsize(x, y, paired=paired, eftype='cohen')

    # 95% confidence interval for the effect size
    # ci = compute_esci(d, nx, ny, paired=paired, eftype='cohen',
    #                   confidence=.95)

    # 95% confidence interval for the difference in means
    # Compare to the t.test r function
    conf = 0.975 if tail == 'two-sided' else 0.95
    tcrit = t.ppf(conf, dof)
    ci = np.array([tval - tcrit, tval + tcrit]) * se
    if tail == 'greater':
        ci[1] = np.inf
    elif tail == 'less':
        ci[0] = -np.inf

    # Achieved power
    if ny == 1:
        # One-sample
        power = power_ttest(d=d, n=nx, power=None, alpha=0.05,
                            contrast='one-sample', tail=tail)
    if ny > 1 and paired is True:
        # Paired two-sample
        power = power_ttest(d=d, n=nx, power=None, alpha=0.05,
                            contrast='paired', tail=tail)
    elif ny > 1 and paired is False:
        # Independent two-samples
        if nx == ny:
            # Equal sample sizes
            power = power_ttest(d=d, n=nx, power=None, alpha=0.05,
                                contrast='two-samples', tail=tail)
        else:
            # Unequal sample sizes
            power = power_ttest2n(nx, ny, d=d, power=None, alpha=0.05,
                                  tail=tail)

    # Bayes factor
    bf = bayesfactor_ttest(tval, nx, ny, paired=paired, tail=tail, r=r)

    # Create output dictionnary
    stats = {'dof': np.round(dof, 2),
             'T': round(tval, 3),
             'p-val': pval,
             'tail': tail,
             'cohen-d': round(abs(d), 3),
             'CI95%': [np.round(ci, 2)],
             'power': round(power, 3),
             'BF10': bf}

    # Convert to dataframe
    col_order = ['T', 'dof', 'tail', 'p-val', 'CI95%', 'cohen-d', 'BF10',
                 'power']
    stats = pd.DataFrame.from_records(stats, columns=col_order,
                                      index=['T-test'])
    return stats


@pf.register_dataframe_method
def rm_anova(data=None, dv=None, within=None, subject=None, correction='auto',
             detailed=False, export_filename=None):
    """One-way and two-way repeated measures ANOVA.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        :py:class:`pandas.DataFrame` method, in which case this argument is no
        longer needed.
        Both wide and long-format dataframe are supported for one-way repeated
        measures ANOVA. However, ``data`` must be in long format for two-way
        repeated measures.
    dv : string
        Name of column containing the dependant variable (only required if
        ``data`` is in long format).
    within : string
        Name of column containing the within factor (only required if ``data``
        is in long format).
        If ``within`` is a single string, then compute a one-way repeated
        measures ANOVA, if ``within`` is a list with two strings,
        compute a two-way repeated measures ANOVA.
    subject : string
        Name of column containing the subject identifier (only required if
        ``data`` is in long format).
    correction : string or boolean
        If True, also return the Greenhouse-Geisser corrected p-value.

        The default for one-way design is to compute Mauchly's test of
        sphericity to determine whether the p-values needs to be corrected
        (see :py:func:`pingouin.sphericity`).

        The default for two-way design is to return both the uncorrected and
        Greenhouse-Geisser corrected p-values. Note that sphericity test for
        two-way design are not currently implemented in Pingouin.
    detailed : boolean
        If True, return a full ANOVA table.
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    aov : DataFrame
        ANOVA summary ::

        'Source' : Name of the within-group factor
        'ddof1' : Degrees of freedom (numerator)
        'ddof2' : Degrees of freedom (denominator)
        'F' : F-value
        'p-unc' : Uncorrected p-value
        'np2' : Partial eta-square effect size
        'eps' : Greenhouse-Geisser epsilon factor (= index of sphericity)
        'p-GG-corr' : Greenhouse-Geisser corrected p-value
        'W-spher' : Sphericity test statistic
        'p-spher' : p-value of the sphericity test
        'sphericity' : sphericity of the data (boolean)

    See Also
    --------
    anova : One-way and N-way ANOVA
    mixed_anova : Two way mixed ANOVA
    friedman : Non-parametric one-way repeated measures ANOVA

    Notes
    -----
    Data can be in wide or long format for one-way repeated measures ANOVA but
    *must* be in long format for two-way repeated measures ANOVA.

    In one-way repeated-measures ANOVA, the total variance (sums of squares)
    is divided into three components

    .. math:: SS_{total} = SS_{treatment} + (SS_{subjects} + SS_{error})

    with

    .. math:: SS_{total} = \\sum_i^r \\sum_j^n (Y_{ij} - \\overline{Y})^2
    .. math:: SS_{treatment} = \\sum_i^r n_i(\\overline{Y_i} - \\overline{Y})^2
    .. math:: SS_{subjects} = r\\sum (\\overline{Y}_s - \\overline{Y})^2
    .. math:: SS_{error} = SS_{total} - SS_{treatment} - SS_{subjects}

    where :math:`i=1,...,r; j=1,...,n_i`, :math:`r` is the number of
    conditions, :math:`n_i` the number of observations for each condition,
    :math:`\\overline{Y}` the grand mean of the data, :math:`\\overline{Y_i}`
    the mean of the :math:`i^{th}` condition and :math:`\\overline{Y}_{subj}`
    the mean of the :math:`s^{th}` subject.

    The F-statistics is then defined as:

    .. math::

        F^* = \\frac{MS_{treatment}}{MS_{error}} =
        \\frac{\\frac{SS_{treatment}}
        {r-1}}{\\frac{SS_{error}}{(n - 1)(r - 1)}}

    and the p-value can be calculated using a F-distribution with
    :math:`v_{treatment} = r - 1` and
    :math:`v_{error} = (n - 1)(r - 1)` degrees of freedom.

    The effect size reported in Pingouin is the partial eta-square, which is
    equivalent to eta-square for one-way repeated measures ANOVA.

    .. math:: \\eta_p^2 = \\frac{SS_{treatment}}{SS_{treatment} + SS_{error}}

    Results have been tested against R and JASP. Note however that if the
    dataset contains one or more other within subject factors, an automatic
    collapsing to the mean is applied on the dependant variable (same behavior
    as the ezANOVA R package). As such, results can differ from those of JASP.

    Missing values are automatically removed (listwise deletion) using the
    :py:func:`pingouin.remove_rm_na` function. This could drastically decrease
    the power of the ANOVA if many missing values are present. In that case,
    it might be better to use linear mixed effects models.

    .. warning:: The epsilon adjustement factor of the interaction in
        two-way repeated measures ANOVA where both factors have more than
        two levels slightly differs than from R and JASP.
        Please always make sure to double-check your results with another
        software.

    .. warning:: Sphericity tests for the interaction term of a two-way
        repeated measures ANOVA are not currently supported in Pingouin.
        Instead, please refer to the Greenhouse-Geisser epsilon value
        (a value close to 1 indicates that sphericity is met.) For more
        details, see :py:func:`pingouin.sphericity`.

    References
    ----------
    .. [1] Bakeman, R. (2005). Recommended effect size statistics for
           repeated measures designs. Behavior research methods, 37(3),
           379-384.

    .. [2] Richardson, J. T. (2011). Eta squared and partial eta squared as
           measures of effect size in educational research. Educational
           Research Review, 6(2), 135-147.

    .. [3] https://en.wikipedia.org/wiki/Repeated_measures_design

    Examples
    --------
    One-way repeated measures ANOVA using a wide-format dataset

    >>> import pingouin as pg
    >>> data = pg.read_dataset('rm_anova_wide')
    >>> pg.rm_anova(data)
       Source  ddof1  ddof2      F     p-unc    np2    eps
    0  Within      3     24  5.201  0.006557  0.394  0.694

    One-way repeated-measures ANOVA using a long-format dataset

    >>> df = pg.read_dataset('rm_anova')
    >>> aov = pg.rm_anova(dv='DesireToKill', within='Disgustingness',
    ...                   subject='Subject', data=df, detailed=True)
    >>> print(aov)
               Source       SS  DF      MS       F        p-unc    np2 eps
    0  Disgustingness   27.485   1  27.485  12.044  0.000793016  0.116   1
    1           Error  209.952  92   2.282       -            -      -   -

    Two-way repeated-measures ANOVA

    >>> aov = pg.rm_anova(dv='DesireToKill',
    ...                   within=['Disgustingness', 'Frighteningness'],
    ...                   subject='Subject', data=df)

    As a :py:class:`pandas.DataFrame` method

    >>> df.rm_anova(dv='DesireToKill', within='Disgustingness',
    ...             subject='Subject',  detailed=True)
               Source       SS  DF      MS       F        p-unc    np2 eps
    0  Disgustingness   27.485   1  27.485  12.044  0.000793016  0.116   1
    1           Error  209.952  92   2.282       -            -      -   -
    """
    if isinstance(within, list):
        assert len(within) > 0, 'Within is empty.'
        if len(within) == 1:
            within = within[0]
        elif len(within) == 2:
            return rm_anova2(dv=dv, within=within, data=data, subject=subject,
                             export_filename=export_filename)
        else:
            raise ValueError('Repeated measures ANOVA with more than three '
                             'factors are not yet supported.')

    # Check data format
    if all([v is None for v in [dv, within, subject]]):
        # Convert from wide to long format
        assert isinstance(data, pd.DataFrame)
        data = data._get_numeric_data().dropna()
        assert data.shape[0] > 2, 'Data must have at least 3 rows.'
        assert data.shape[1] > 1, 'Data must contain at least two columns.'
        data['Subj'] = np.arange(data.shape[0])
        data = data.melt(id_vars='Subj', var_name='Within', value_name='DV')
        subject, within, dv = 'Subj', 'Within', 'DV'

    # Check dataframe
    _check_dataframe(dv=dv, within=within, data=data, subject=subject,
                     effects='within')

    # Collapse to the mean
    data = data.groupby([subject, within]).mean().reset_index()

    # Remove NaN
    if data[dv].isnull().any():
        data = remove_rm_na(dv=dv, within=within, subject=subject,
                            data=data[[subject, within, dv]])
    assert not data[within].isnull().any(), 'Cannot have NaN in `within`.'
    assert not data[subject].isnull().any(), 'Cannot have NaN in `subject`.'

    # Groupby
    grp_with = data.groupby(within)[dv]
    rm = list(data[within].unique())
    n_rm = len(rm)
    n_obs = int(data.groupby(within)[dv].count().max())
    grandmean = data[dv].mean()

    # Calculate sums of squares
    sstime = ((grp_with.mean() - grandmean)**2 * grp_with.count()).sum()
    sswithin = grp_with.apply(lambda x: (x - x.mean())**2).sum()
    grp_subj = data.groupby(subject)[dv]
    sssubj = n_rm * np.sum((grp_subj.mean() - grandmean)**2)
    sserror = sswithin - sssubj

    # Calculate degrees of freedom
    ddof1 = n_rm - 1
    ddof2 = ddof1 * (n_obs - 1)

    # Calculate F and p-values
    mserror = sserror / (ddof2 / ddof1)
    fval = sstime / mserror
    p_unc = f(ddof1, ddof2).sf(fval)

    # Calculating partial eta-square
    # Similar to (fval * ddof1) / (fval * ddof1 + ddof2)
    np2 = sstime / (sstime + sserror)

    # Reshape and remove NAN for sphericity estimation and correction
    data_pivot = data.pivot(index=subject, columns=within, values=dv).dropna()

    # Compute sphericity using Mauchly test
    # Sphericity assumption only applies if there are more than 2 levels
    if correction == 'auto' or (correction is True and n_rm >= 3):
        spher, W_spher, chi_sq_spher, ddof_spher, \
            p_spher = sphericity(data_pivot, alpha=.05)
        if correction == 'auto':
            correction = True if not spher else False
    else:
        correction = False

    # Compute epsilon adjustement factor
    eps = epsilon(data_pivot, correction='gg')

    # If required, apply Greenhouse-Geisser correction for sphericity
    if correction:
        corr_ddof1, corr_ddof2 = [np.maximum(d * eps, 1.) for d in
                                  (ddof1, ddof2)]
        p_corr = f(corr_ddof1, corr_ddof2).sf(fval)

    # Create output dataframe
    if not detailed:
        aov = pd.DataFrame({'Source': within,
                            'ddof1': ddof1,
                            'ddof2': ddof2,
                            'F': fval,
                            'p-unc': p_unc,
                            'np2': np2,
                            'eps': eps,
                            }, index=[0])
        if correction:
            aov['p-GG-corr'] = p_corr
            aov['W-spher'] = W_spher
            aov['p-spher'] = p_spher
            aov['sphericity'] = spher

        col_order = ['Source', 'ddof1', 'ddof2', 'F', 'p-unc',
                     'p-GG-corr', 'np2', 'eps', 'sphericity', 'W-spher',
                     'p-spher']
    else:
        aov = pd.DataFrame({'Source': [within, 'Error'],
                            'SS': np.round([sstime, sserror], 3),
                            'DF': [ddof1, ddof2],
                            'MS': np.round([sstime / ddof1, sserror / ddof2],
                                           3),
                            'F': [fval, np.nan],
                            'p-unc': [p_unc, np.nan],
                            'np2': [np2, np.nan],
                            'eps': [eps, np.nan]
                            })
        if correction:
            aov['p-GG-corr'] = [p_corr, np.nan]
            aov['W-spher'] = [W_spher, np.nan]
            aov['p-spher'] = [p_spher, np.nan]
            aov['sphericity'] = [spher, np.nan]

        col_order = ['Source', 'SS', 'DF', 'MS', 'F', 'p-unc', 'p-GG-corr',
                     'np2', 'eps', 'sphericity', 'W-spher', 'p-spher']

    # Round
    aov[['F', 'eps', 'np2']] = aov[['F', 'eps', 'np2']].round(3)

    # Replace NaN
    aov = aov.fillna('-')

    aov = aov.reindex(columns=col_order)
    aov.dropna(how='all', axis=1, inplace=True)
    # Export to .csv
    if export_filename is not None:
        _export_table(aov, export_filename)
    return aov


def rm_anova2(data=None, dv=None, within=None, subject=None,
              export_filename=None):
    """Two-way repeated measures ANOVA.

    This is an internal function. The main call to this function should be done
    by the :py:func:`pingouin.rm_anova` function.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame
    dv : string
        Name of column containing the dependant variable.
    within : list
        Names of column containing the two within factor
        (e.g. ['Time', 'Treatment'])
    subject : string
        Name of column containing the subject identifier.
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    aov : DataFrame
        ANOVA summary ::

        'Source' : Name of the within-group factors
        'ddof1' : Degrees of freedom (numerator)
        'ddof2' : Degrees of freedom (denominator)
        'F' : F-value
        'p-unc' : Uncorrected p-value
        'np2' : Partial eta-square effect size
        'eps' : Greenhouse-Geisser epsilon factor (= index of sphericity)
        'p-GG-corr' : Greenhouse-Geisser corrected p-value
    """
    a, b = within

    # Validate the dataframe
    _check_dataframe(dv=dv, within=within, data=data, subject=subject,
                     effects='within')

    # Remove NaN
    if data[[subject, a, b, dv]].isnull().any().any():
        data = remove_rm_na(dv=dv, subject=subject, within=[a, b],
                            data=data[[subject, a, b, dv]])

    # Collapse to the mean (that this is also done in remove_rm_na)
    data = data.groupby([subject, a, b]).mean().reset_index()

    assert not data[a].isnull().any(), 'Cannot have NaN in %s' % a
    assert not data[b].isnull().any(), 'Cannot have NaN in %s' % b
    assert not data[subject].isnull().any(), 'Cannot have NaN in %s' % subject

    # Group sizes and grandmean
    n_a = data[a].nunique()
    n_b = data[b].nunique()
    n_s = data[subject].nunique()
    mu = data[dv].mean()

    # Groupby means
    grp_s = data.groupby(subject)[dv].mean()
    grp_a = data.groupby([a])[dv].mean()
    grp_b = data.groupby([b])[dv].mean()
    grp_ab = data.groupby([a, b])[dv].mean()
    grp_as = data.groupby([a, subject])[dv].mean()
    grp_bs = data.groupby([b, subject])[dv].mean()

    # Sums of squares
    ss_tot = np.sum((data[dv] - mu)**2)
    ss_s = (n_a * n_b) * np.sum((grp_s - mu)**2)
    ss_a = (n_b * n_s) * np.sum((grp_a - mu)**2)
    ss_b = (n_a * n_s) * np.sum((grp_b - mu)**2)
    ss_ab_er = n_s * np.sum((grp_ab - mu)**2)
    ss_ab = ss_ab_er - ss_a - ss_b
    ss_as_er = n_b * np.sum((grp_as - mu)**2)
    ss_as = ss_as_er - ss_s - ss_a
    ss_bs_er = n_a * np.sum((grp_bs - mu)**2)
    ss_bs = ss_bs_er - ss_s - ss_b
    ss_abs = ss_tot - ss_a - ss_b - ss_s - ss_ab - ss_as - ss_bs

    # DOF
    df_a = n_a - 1
    df_b = n_b - 1
    df_s = n_s - 1
    df_ab_er = n_a * n_b - 1
    df_ab = df_ab_er - df_a - df_b
    df_as_er = n_a * n_s - 1
    df_as = df_as_er - df_s - df_a
    df_bs_er = n_b * n_s - 1
    df_bs = df_bs_er - df_s - df_b
    df_tot = n_a * n_b * n_s - 1
    df_abs = df_tot - df_a - df_b - df_s - df_ab - df_as - df_bs

    # Mean squares
    ms_a = ss_a / df_a
    ms_b = ss_b / df_b
    ms_ab = ss_ab / df_ab
    ms_as = ss_as / df_as
    ms_bs = ss_bs / df_bs
    ms_abs = ss_abs / df_abs

    # F-values
    f_a = ms_a / ms_as
    f_b = ms_b / ms_bs
    f_ab = ms_ab / ms_abs

    # P-values
    p_a = f(df_a, df_as).sf(f_a)
    p_b = f(df_b, df_bs).sf(f_b)
    p_ab = f(df_ab, df_abs).sf(f_ab)

    # Partial eta-square
    eta_a = (f_a * df_a) / (f_a * df_a + df_as)
    eta_b = (f_b * df_b) / (f_b * df_b + df_bs)
    eta_ab = (f_ab * df_ab) / (f_ab * df_ab + df_abs)

    # Epsilon
    piv_a = data.pivot_table(index=subject, columns=a, values=dv)
    piv_b = data.pivot_table(index=subject, columns=b, values=dv)
    piv_ab = data.pivot_table(index=subject, columns=[a, b], values=dv)
    eps_a = epsilon(piv_a, correction='gg')
    eps_b = epsilon(piv_b, correction='gg')
    # Note that the GG epsilon of the interaction slightly differs between
    # R and Pingouin. An alternative is to use the lower bound, which is
    # very conservative (same behavior as described on real-statistics.com).
    eps_ab = epsilon(piv_ab, correction='gg')

    # Greenhouse-Geisser correction
    df_a_c, df_as_c = [np.maximum(d * eps_a, 1.) for d in (df_a, df_as)]
    df_b_c, df_bs_c = [np.maximum(d * eps_b, 1.) for d in (df_b, df_bs)]
    df_ab_c, df_abs_c = [np.maximum(d * eps_ab, 1.) for d in (df_ab, df_abs)]
    p_a_corr = f(df_a_c, df_as_c).sf(f_a)
    p_b_corr = f(df_b_c, df_bs_c).sf(f_b)
    p_ab_corr = f(df_ab_c, df_abs_c).sf(f_ab)

    # Create dataframe
    aov = pd.DataFrame({'Source': [a, b, a + ' * ' + b],
                        'SS': [ss_a, ss_b, ss_ab],
                        'ddof1': [df_a, df_b, df_ab],
                        'ddof2': [df_as, df_bs, df_abs],
                        'MS': [ms_a, ms_b, ms_ab],
                        'F': [f_a, f_b, f_ab],
                        'p-unc': [p_a, p_b, p_ab],
                        'p-GG-corr': [p_a_corr, p_b_corr, p_ab_corr],
                        'np2': [eta_a, eta_b, eta_ab],
                        'eps': [eps_a, eps_b, eps_ab],
                        })

    col_order = ['Source', 'SS', 'ddof1', 'ddof2', 'MS', 'F', 'p-unc',
                 'p-GG-corr', 'np2', 'eps']

    # Round
    aov[['SS', 'MS', 'F', 'eps', 'np2']] = aov[['SS', 'MS', 'F', 'eps',
                                                'np2']].round(3)

    aov = aov.reindex(columns=col_order)
    # Export to .csv
    if export_filename is not None:
        _export_table(aov, export_filename)
    return aov


@pf.register_dataframe_method
def anova(data=None, dv=None, between=None, ss_type=2, detailed=False,
          export_filename=None):
    """One-way and *N*-way ANOVA.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    dv : string
        Name of column in ``data`` containing the dependent variable.
    between : string or list with *N* elements
        Name of column(s) in ``data`` containing the between-subject factor(s).
        If ``between`` is a single string, a one-way ANOVA is computed.
        If ``between`` is a list with two or more elements, a *N*-way ANOVA is
        performed.
        Note that Pingouin will internally call statsmodels to calculate
        ANOVA with 3 or more factors, or unbalanced two-way ANOVA.
    ss_type : int
        Specify how the sums of squares is calculated for *unbalanced* design
        with 2 or more factors. Can be 1, 2 (default), or 3. This has no impact
        on one-way design or N-way ANOVA with balanced data.
    detailed : boolean
        If True, return a detailed ANOVA table
        (default True for N-way ANOVA).
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    aov : DataFrame
        ANOVA summary ::

        'Source' : Factor names
        'SS' : Sums of squares
        'DF' : Degrees of freedom
        'MS' : Mean squares
        'F' : F-values
        'p-unc' : uncorrected p-values
        'np2' : Partial eta-square effect sizes

    See Also
    --------
    rm_anova : One-way and two-way repeated measures ANOVA
    mixed_anova : Two way mixed ANOVA
    welch_anova : One-way Welch ANOVA
    kruskal : Non-parametric one-way ANOVA

    Notes
    -----
    The classic ANOVA is very powerful when the groups are normally distributed
    and have equal variances. However, when the groups have unequal variances,
    it is best to use the Welch ANOVA (:py:func:`pingouin.welch_anova`) that
    better controls for type I error (Liu 2015). The homogeneity of variances
    can be measured with the :py:func:`pingouin.homoscedasticity` function.

    The main idea of ANOVA is to partition the variance (sums of squares)
    into several components. For example, in one-way ANOVA:

    .. math:: SS_{total} = SS_{treatment} + SS_{error}
    .. math:: SS_{total} = \\sum_i \\sum_j (Y_{ij} - \\overline{Y})^2
    .. math:: SS_{treatment} = \\sum_i n_i (\\overline{Y_i} - \\overline{Y})^2
    .. math:: SS_{error} = \\sum_i \\sum_j (Y_{ij} - \\overline{Y}_i)^2

    where :math:`i=1,...,r; j=1,...,n_i`, :math:`r` is the number of groups,
    and :math:`n_i` the number of observations for the :math:`i` th group.

    The F-statistics is then defined as:

    .. math::

        F^* = \\frac{MS_{treatment}}{MS_{error}} = \\frac{SS_{treatment}
        / (r - 1)}{SS_{error} / (n_t - r)}

    and the p-value can be calculated using a F-distribution with
    :math:`r-1, n_t-1` degrees of freedom.

    When the groups are balanced and have equal variances, the optimal post-hoc
    test is the Tukey-HSD test (:py:func:`pingouin.pairwise_tukey`).
    If the groups have unequal variances, the Games-Howell test is more
    adequate (:py:func:`pingouin.pairwise_gameshowell`).

    The effect size reported in Pingouin is the partial eta-square.
    However, one should keep in mind that for one-way ANOVA
    partial eta-square is the same as eta-square and generalized eta-square.
    For more details, see Bakeman 2005; Richardson 2011.

    .. math:: \\eta_p^2 = \\frac{SS_{treatment}}{SS_{treatment} + SS_{error}}

    Note that missing values are automatically removed. Results have been
    tested against R, Matlab and JASP.

    .. warning :: Versions of Pingouin below 0.2.5 gave wrong results for
        **unbalanced N-way ANOVA**. This issue has been resolved in
        Pingouin>=0.2.5. In such cases, the ANOVA is calculated via an
        internal call to the statsmodels package.

    References
    ----------
    .. [1] Liu, Hangcheng. "Comparing Welch's ANOVA, a Kruskal-Wallis test and
           traditional ANOVA in case of Heterogeneity of Variance." (2015).

    .. [2] Bakeman, Roger. "Recommended effect size statistics for repeated
           measures designs." Behavior research methods 37.3 (2005): 379-384.

    .. [3] Richardson, John TE. "Eta squared and partial eta squared as
           measures of effect size in educational research." Educational
           Research Review 6.2 (2011): 135-147.

    Examples
    --------
    One-way ANOVA

    >>> import pingouin as pg
    >>> df = pg.read_dataset('anova')
    >>> aov = pg.anova(dv='Pain threshold', between='Hair color', data=df,
    ...             detailed=True)
    >>> aov
           Source        SS  DF       MS      F       p-unc    np2
    0  Hair color  1360.726   3  453.575  6.791  0.00411423  0.576
    1      Within  1001.800  15   66.787      -           -      -

    Note that this function can also directly be used as a Pandas method

    >>> df.anova(dv='Pain threshold', between='Hair color', detailed=True)
           Source        SS  DF       MS      F       p-unc    np2
    0  Hair color  1360.726   3  453.575  6.791  0.00411423  0.576
    1      Within  1001.800  15   66.787      -           -      -

    Two-way ANOVA with balanced design

    >>> data = pg.read_dataset('anova2')
    >>> data.anova(dv="Yield", between=["Blend", "Crop"]).round(3)
             Source        SS  DF        MS      F  p-unc    np2
    0         Blend     2.042   1     2.042  0.004  0.952  0.000
    1          Crop  2736.583   2  1368.292  2.525  0.108  0.219
    2  Blend * Crop  2360.083   2  1180.042  2.178  0.142  0.195
    3      Residual  9753.250  18   541.847    NaN    NaN    NaN

    Two-way ANOVA with unbalanced design (requires statsmodels)

    >>> data = pg.read_dataset('anova2_unbalanced')
    >>> data.anova(dv="Scores", between=["Diet", "Exercise"]).round(3)
                Source       SS   DF       MS      F  p-unc    np2
    0             Diet  390.625  1.0  390.625  7.423  0.034  0.553
    1         Exercise  180.625  1.0  180.625  3.432  0.113  0.364
    2  Diet * Exercise   15.625  1.0   15.625  0.297  0.605  0.047
    3         Residual  315.750  6.0   52.625    NaN    NaN    NaN

    Three-way ANOVA, type 3 sums of squares (requires statsmodels)

    >>> data = pg.read_dataset('anova3')
    >>> data.anova(dv='Cholesterol', between=['Sex', 'Risk', 'Drug'],
    ...            ss_type=3)
                  Source      SS    DF      MS       F     p-unc    np2
    0                Sex   2.075   1.0   2.075   2.462  0.123191  0.049
    1               Risk  11.332   1.0  11.332  13.449  0.000613  0.219
    2               Drug   0.816   2.0   0.408   0.484  0.619249  0.020
    3         Sex * Risk   0.117   1.0   0.117   0.139  0.710541  0.003
    4         Sex * Drug   2.564   2.0   1.282   1.522  0.228711  0.060
    5        Risk * Drug   2.438   2.0   1.219   1.446  0.245485  0.057
    6  Sex * Risk * Drug   1.844   2.0   0.922   1.094  0.343041  0.044
    7           Residual  40.445  48.0   0.843     NaN       NaN    NaN
    """
    if isinstance(between, list):
        if len(between) == 0:
            raise ValueError('between is empty.')
        elif len(between) == 1:
            between = between[0]
        elif len(between) == 2:
            # Two factors with balanced design = Pingouin implementation
            # Two factors with unbalanced design = statsmodels
            return anova2(dv=dv, between=between, data=data,
                          ss_type=ss_type, export_filename=export_filename)
        else:
            # 3 or more factors with (un)-balanced design = statsmodels
            return anovan(dv=dv, between=between, data=data,
                          ss_type=ss_type, export_filename=export_filename)

    # Check data
    _check_dataframe(dv=dv, between=between, data=data, effects='between')

    # Drop missing values
    data = data[[dv, between]].dropna()

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)

    groups = list(data[between].unique())
    n_groups = len(groups)
    N = data[dv].size

    # Calculate sums of squares
    grp = data.groupby(between)[dv]
    # Between effect
    ssbetween = ((grp.mean() - data[dv].mean())**2 * grp.count()).sum()
    # Within effect (= error between)
    #  = (grp.var(ddof=0) * grp.count()).sum()
    sserror = grp.apply(lambda x: (x - x.mean())**2).sum()

    # Calculate DOF, MS, F and p-values
    ddof1 = n_groups - 1
    ddof2 = N - n_groups
    msbetween = ssbetween / ddof1
    mserror = sserror / ddof2
    fval = msbetween / mserror
    p_unc = f(ddof1, ddof2).sf(fval)

    # Calculating partial eta-square
    # Similar to (fval * ddof1) / (fval * ddof1 + ddof2)
    np2 = ssbetween / (ssbetween + sserror)

    # Create output dataframe
    if not detailed:
        aov = pd.DataFrame({'Source': between,
                            'ddof1': ddof1,
                            'ddof2': ddof2,
                            'F': fval,
                            'p-unc': p_unc,
                            'np2': np2
                            }, index=[0])

        col_order = ['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'np2']
    else:
        aov = pd.DataFrame({'Source': [between, 'Within'],
                            'SS': np.round([ssbetween, sserror], 3),
                            'DF': [ddof1, ddof2],
                            'MS': np.round([msbetween, mserror], 3),
                            'F': [fval, np.nan],
                            'p-unc': [p_unc, np.nan],
                            'np2': [np2, np.nan]
                            })
        col_order = ['Source', 'SS', 'DF', 'MS', 'F', 'p-unc', 'np2']

    # Round
    aov[['F', 'np2']] = aov[['F', 'np2']].round(3)

    # Replace NaN
    aov = aov.fillna('-')

    aov = aov.reindex(columns=col_order)
    aov.dropna(how='all', axis=1, inplace=True)
    # Export to .csv
    if export_filename is not None:
        _export_table(aov, export_filename)
    return aov


def anova2(data=None, dv=None, between=None, ss_type=2, export_filename=None):
    """Two-way balanced ANOVA in pure Python + Pandas.

    This is an internal function. The main call to this function should be done
    by the :py:func:`pingouin.anova` function.
    """
    # Validate the dataframe
    _check_dataframe(dv=dv, between=between, data=data, effects='between')

    assert len(between) == 2, 'Must have exactly two between-factors variables'
    fac1, fac2 = between

    # Drop missing values
    data = data[[dv, fac1, fac2]].dropna()
    assert data.shape[0] >= 5, 'Data must have at least 5 non-missing values.'

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)
    grp_both = data.groupby(between)[dv]

    if grp_both.count().nunique() == 1:
        # BALANCED DESIGN
        aov_fac1 = anova(data=data, dv=dv, between=fac1, detailed=True)
        aov_fac2 = anova(data=data, dv=dv, between=fac2, detailed=True)
        ng1, ng2 = data[fac1].nunique(), data[fac2].nunique()
        # Sums of squares
        ss_fac1 = aov_fac1.at[0, 'SS']
        ss_fac2 = aov_fac2.at[0, 'SS']
        ss_tot = ((data[dv] - data[dv].mean())**2).sum()
        ss_resid = np.sum(grp_both.apply(lambda x: (x - x.mean())**2))
        ss_inter = ss_tot - (ss_resid + ss_fac1 + ss_fac2)
        # Degrees of freedom
        df_fac1 = aov_fac1.at[0, 'DF']
        df_fac2 = aov_fac2.at[0, 'DF']
        df_inter = (ng1 - 1) * (ng2 - 1)
        df_resid = data[dv].size - (ng1 * ng2)
    else:
        # UNBALANCED DESIGN
        return anovan(dv=dv, between=between, data=data, ss_type=ss_type,
                      export_filename=export_filename)

    # Mean squares
    ms_fac1 = ss_fac1 / df_fac1
    ms_fac2 = ss_fac2 / df_fac2
    ms_inter = ss_inter / df_inter
    ms_resid = ss_resid / df_resid

    # F-values
    fval_fac1 = ms_fac1 / ms_resid
    fval_fac2 = ms_fac2 / ms_resid
    fval_inter = ms_inter / ms_resid

    # P-values
    pval_fac1 = f(df_fac1, df_resid).sf(fval_fac1)
    pval_fac2 = f(df_fac2, df_resid).sf(fval_fac2)
    pval_inter = f(df_inter, df_resid).sf(fval_inter)

    # Partial eta-square
    np2_fac1 = ss_fac1 / (ss_fac1 + ss_resid)
    np2_fac2 = ss_fac2 / (ss_fac2 + ss_resid)
    np2_inter = ss_inter / (ss_inter + ss_resid)

    # Create output dataframe
    aov = pd.DataFrame({'Source': [fac1, fac2, fac1 + ' * ' + fac2,
                                   'Residual'],
                        'SS': np.round([ss_fac1, ss_fac2, ss_inter,
                                        ss_resid], 3),
                        'DF': [df_fac1, df_fac2, df_inter, df_resid],
                        'MS': np.round([ms_fac1, ms_fac2, ms_inter,
                                        ms_resid], 3),
                        'F': [fval_fac1, fval_fac2, fval_inter, np.nan],
                        'p-unc': [pval_fac1, pval_fac2, pval_inter, np.nan],
                        'np2': [np2_fac1, np2_fac2, np2_inter, np.nan]
                        })
    col_order = ['Source', 'SS', 'DF', 'MS', 'F', 'p-unc', 'np2']

    aov = aov.reindex(columns=col_order)
    aov.dropna(how='all', axis=1, inplace=True)
    # Export to .csv
    if export_filename is not None:
        _export_table(aov, export_filename)
    return aov


def anovan(data=None, dv=None, between=None, ss_type=2, export_filename=None):
    """N-way ANOVA using statsmodels.

    This is an internal function. The main call to this function should be done
    by the :py:func:`pingouin.anova` function.
    """
    # Check that stasmodels is installed
    from pingouin.utils import _is_statsmodels_installed
    _is_statsmodels_installed(raise_error=True)
    from statsmodels.api import stats
    from statsmodels.formula.api import ols

    # Validate the dataframe
    _check_dataframe(dv=dv, between=between, data=data, effects='between')
    all_cols = _flatten_list([dv, between])
    bad_chars = [',', '(', ')', ':']
    if not all([c not in v for c in bad_chars for v in all_cols]):
        err_msg = "comma, bracket, and colon are not allowed in column names."
        raise ValueError(err_msg)

    # Drop missing values
    data = data[all_cols].dropna()
    assert data.shape[0] >= 5, 'Data must have at least 5 non-missing values.'

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)

    # Create R-like formula
    formula = dv + ' ~ '
    for fac in between:
        formula += 'C(' + fac + ', Sum) * '
    formula = formula[:-3]  # Remove last * and space

    # Fit using statsmodels
    lm = ols(formula, data=data).fit()
    aov = stats.anova_lm(lm, typ=ss_type)

    # Convert to Pingouin-like dataframe
    if ss_type == 1:
        # statsmodels output is not exactly the same when ss_type = 1
        aov = aov[['sum_sq', 'df', 'F', 'PR(>F)']]
    if ss_type == 3:
        # Remove intercept row
        aov = aov.iloc[1:, :]

    aov = aov.reset_index()
    aov = aov.rename(columns={'index': 'Source', 'sum_sq': 'SS',
                              'df': 'DF', 'PR(>F)': 'p-unc'})
    aov['MS'] = aov['SS'] / aov['DF']
    aov['np2'] = (aov['F'] * aov['DF']) / (aov['F'] * aov['DF'] +
                                           aov.iloc[-1, 2])

    def format_source(x):
        for fac in between:
            x = x.replace('C(%s, Sum)' % fac, fac)
        return x.replace(':', ' * ')

    aov['Source'] = aov['Source'].apply(format_source)

    # Re-index and round
    col_order = ['Source', 'SS', 'DF', 'MS', 'F', 'p-unc', 'np2']
    aov = aov.reindex(columns=col_order)
    aov[['SS', 'MS', 'F', 'np2']] = aov[['SS', 'MS', 'F', 'np2']].round(3)
    aov.dropna(how='all', axis=1, inplace=True)

    # Add formula to dataframe
    aov.formula_ = formula

    # Export to .csv
    if export_filename is not None:
        _export_table(aov, export_filename)
    return aov


@pf.register_dataframe_method
def welch_anova(data=None, dv=None, between=None, export_filename=None):
    """One-way Welch ANOVA.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    between : string
        Name of column containing the between factor.
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    aov : DataFrame
        ANOVA summary ::

        'Source' : Factor names
        'SS' : Sums of squares
        'DF' : Degrees of freedom
        'MS' : Mean squares
        'F' : F-values
        'p-unc' : uncorrected p-values

    See Also
    --------
    anova : One-way and N-way ANOVA
    rm_anova : One-way and two-way repeated measures ANOVA
    mixed_anova : Two way mixed ANOVA
    kruskal : Non-parametric one-way ANOVA

    Notes
    -----
    From Wikipedia:

    *It is named for its creator, Bernard Lewis Welch, and is an adaptation of
    Student's t-test, and is more reliable when the two samples have
    unequal variances and/or unequal sample sizes.*

    The classic ANOVA is very powerful when the groups are normally distributed
    and have equal variances. However, when the groups have unequal variances,
    it is best to use the Welch ANOVA that better controls for
    type I error (Liu 2015). The homogeneity of variances can be measured with
    the `homoscedasticity` function. The two other assumptions of
    normality and independance remain.

    The main idea of Welch ANOVA is to use a weight :math:`w_i` to reduce
    the effect of unequal variances. This weight is calculated using the sample
    size :math:`n_i` and variance :math:`s_i^2` of each group
    :math:`i=1,...,r`:

    .. math:: w_i = \\frac{n_i}{s_i^2}

    Using these weights, the adjusted grand mean of the data is:

    .. math::

        \\overline{Y}_{welch} = \\frac{\\sum_{i=1}^r w_i\\overline{Y}_i}
        {\\sum w}

    where :math:`\\overline{Y}_i` is the mean of the :math:`i` group.

    The treatment sums of squares is defined as:

    .. math::

        SS_{treatment} = \\sum_{i=1}^r w_i
        (\\overline{Y}_i - \\overline{Y}_{welch})^2

    We then need to calculate a term lambda:

    .. math::

        \\Lambda = \\frac{3\\sum_{i=1}^r(\\frac{1}{n_i-1})
        (1 - \\frac{w_i}{\\sum w})^2}{r^2 - 1}

    from which the F-value can be calculated:

    .. math::

        F_{welch} = \\frac{SS_{treatment} / (r-1)}
        {1 + \\frac{2\\Lambda(r-2)}{3}}

    and the p-value approximated using a F-distribution with
    :math:`(r-1, 1 / \\Lambda)` degrees of freedom.

    When the groups are balanced and have equal variances, the optimal post-hoc
    test is the Tukey-HSD test (:py:func:`pingouin.pairwise_tukey`).
    If the groups have unequal variances, the Games-Howell test is more
    adequate (:py:func:`pingouin.pairwise_gameshowell`).

    Results have been tested against R.

    References
    ----------
    .. [1] Liu, Hangcheng. "Comparing Welch's ANOVA, a Kruskal-Wallis test and
           traditional ANOVA in case of Heterogeneity of Variance." (2015).

    .. [2] Welch, Bernard Lewis. "On the comparison of several mean values:
           an alternative approach." Biometrika 38.3/4 (1951): 330-336.

    Examples
    --------
    1. One-way Welch ANOVA on the pain threshold dataset.

    >>> from pingouin import welch_anova, read_dataset
    >>> df = read_dataset('anova')
    >>> aov = welch_anova(dv='Pain threshold', between='Hair color',
    ...                   data=df, export_filename='pain_anova.csv')
    >>> aov
           Source  ddof1  ddof2     F     p-unc
    0  Hair color      3   8.33  5.89  0.018813
    """
    # Check data
    _check_dataframe(dv=dv, between=between, data=data, effects='between')

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)

    # Number of groups
    r = data[between].nunique()
    ddof1 = r - 1

    # Compute weights and ajusted means
    grp = data.groupby(between)[dv]
    weights = grp.count() / grp.var()
    adj_grandmean = (weights * grp.mean()).sum() / weights.sum()

    # Treatment sum of squares
    ss_tr = np.sum(weights * np.square(grp.mean() - adj_grandmean))
    ms_tr = ss_tr / ddof1

    # Calculate lambda, F-value and p-value
    lamb = (3 * np.sum((1 / (grp.count() - 1)) *
                       (1 - (weights / weights.sum()))**2)) / (r**2 - 1)
    fval = ms_tr / (1 + (2 * lamb * (r - 2)) / 3)
    pval = f.sf(fval, ddof1, 1 / lamb)

    # Create output dataframe
    aov = pd.DataFrame({'Source': between,
                        'ddof1': ddof1,
                        'ddof2': 1 / lamb,
                        'F': fval,
                        'p-unc': pval,
                        }, index=[0])

    col_order = ['Source', 'ddof1', 'ddof2', 'F', 'p-unc']
    aov = aov.reindex(columns=col_order)
    aov[['F', 'ddof2']] = aov[['F', 'ddof2']].round(3)

    # Export to .csv
    if export_filename is not None:
        _export_table(aov, export_filename)
    return aov


@pf.register_dataframe_method
def mixed_anova(data=None, dv=None, within=None, subject=None, between=None,
                correction='auto', export_filename=None):
    """Mixed-design (split-plot) ANOVA.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    dv : string
        Name of column containing the dependant variable.
    within : string
        Name of column containing the within factor.
    subject : string
        Name of column containing the subject identifier.
    between : string
        Name of column containing the between factor.
    correction : string or boolean
        If True, return Greenhouse-Geisser corrected p-value.
        If 'auto' (default), compute Mauchly's test of sphericity to determine
        whether the p-values needs to be corrected.
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    aov : DataFrame
        ANOVA summary ::

        'Source' : Names of the factor considered
        'ddof1' : Degrees of freedom (numerator)
        'ddof2' : Degrees of freedom (denominator)
        'F' : F-values
        'p-unc' : Uncorrected p-values
        'np2' : Partial eta-square effect sizes
        'eps' : Greenhouse-Geisser epsilon factor ( = index of sphericity)
        'p-GG-corr' : Greenhouse-Geisser corrected p-values
        'W-spher' : Sphericity test statistic
        'p-spher' : p-value of the sphericity test
        'sphericity' : sphericity of the data (boolean)

    See Also
    --------
    anova : One-way and N-way ANOVA
    rm_anova : One-way and two-way repeated measures ANOVA

    Notes
    -----
    Results have been tested against R and JASP.

    Missing values are automatically removed (listwise deletion) using the
    :py:func:`pingouin.remove_rm_na` function. This could drastically decrease
    the power of the ANOVA if many missing values are present. In that case,
    it might be better to use linear mixed effects models.

    If the between-subject groups are unbalanced (=  unequal sample sizes), a
    type II ANOVA will be computed.

    Examples
    --------
    Compute a two-way mixed model ANOVA.

    >>> from pingouin import mixed_anova, read_dataset
    >>> df = read_dataset('mixed_anova')
    >>> aov = mixed_anova(dv='Scores', between='Group',
    ...                   within='Time', subject='Subject', data=df)
    >>> aov
            Source     SS  DF1  DF2     MS      F     p-unc    np2    eps
    0        Group  5.460    1   58  5.460  5.052  0.028420  0.080      -
    1         Time  7.628    2  116  3.814  4.027  0.020373  0.065  0.999
    2  Interaction  5.168    2  116  2.584  2.728  0.069530  0.045      -
    """
    # Check data
    _check_dataframe(dv=dv, within=within, between=between, data=data,
                     subject=subject, effects='interaction')

    # Collapse to the mean
    data = data.groupby([subject, within, between]).mean().reset_index()

    # Remove NaN
    if data[dv].isnull().any():
        data = remove_rm_na(dv=dv, within=within, subject=subject,
                            data=data[[subject, within, between, dv]])

    # SUMS OF SQUARES
    grandmean = data[dv].mean()
    # Extract main effects of time and between
    mtime = rm_anova(dv=dv, within=within, subject=subject, data=data,
                     correction=correction, detailed=True)
    mbetw = anova(dv=dv, between=between, data=data, detailed=True)
    # Extract SS total, residuals and interactions
    grp = data.groupby([between, within])[dv]
    sstotal = grp.apply(lambda x: (x - grandmean)**2).sum()
    # sst = residuals within + residuals between
    sst = grp.apply(lambda x: (x - x.mean())**2).sum()
    # Interaction
    ssinter = sstotal - (sst + mtime.at[0, 'SS'] + mbetw.at[0, 'SS'])
    sswg = mtime.at[1, 'SS'] - ssinter
    sseb = sstotal - (mtime.at[0, 'SS'] + mbetw.at[0, 'SS'] + sswg + ssinter)

    # DEGREES OF FREEDOM
    n_obs = data.groupby(within)[dv].count().max()
    dftime = mtime.at[0, 'DF']
    dfbetween = mbetw.at[0, 'DF']
    dfeb = n_obs - data.groupby(between)[dv].count().count()
    dfwg = dftime * dfeb
    dfinter = mtime.at[0, 'DF'] * mbetw.at[0, 'DF']

    # MEAN SQUARES
    mseb = sseb / dfeb
    mswg = sswg / dfwg
    msinter = ssinter / dfinter

    # F VALUES
    fbetween = mbetw.at[0, 'MS'] / mseb
    ftime = mtime.at[0, 'MS'] / mswg
    finter = msinter / mswg

    # P-values
    pbetween = f(dfbetween, dfeb).sf(fbetween)
    ptime = f(dftime, dfwg).sf(ftime)
    pinter = f(dfinter, dfwg).sf(finter)

    # Effects sizes
    npsq_between = fbetween * dfbetween / (fbetween * dfbetween + dfeb)
    npsq_time = ftime * dftime / (ftime * dftime + dfwg)
    npsq_inter = ssinter / (ssinter + sswg)

    # Stats table
    aov = pd.concat([mbetw.drop(1), mtime.drop(1)], sort=False,
                    ignore_index=True)
    # Update values
    aov.rename(columns={'DF': 'DF1'}, inplace=True)
    aov.at[0, 'F'], aov.at[1, 'F'] = fbetween, ftime
    aov.at[0, 'p-unc'], aov.at[1, 'p-unc'] = pbetween, ptime
    aov.at[0, 'np2'], aov.at[1, 'np2'] = npsq_between, npsq_time
    aov = aov.append({'Source': 'Interaction',
                      'SS': ssinter,
                      'DF1': dfinter,
                      'MS': msinter,
                      'F': finter,
                      'p-unc': pinter,
                      'np2': npsq_inter,
                      }, ignore_index=True)

    aov['SS'] = aov['SS'].round(3)
    aov['MS'] = aov['MS'].round(3)
    aov['DF2'] = [dfeb, dfwg, dfwg]
    aov['eps'] = [np.nan, mtime.at[0, 'eps'], np.nan]
    col_order = ['Source', 'SS', 'DF1', 'DF2', 'MS', 'F', 'p-unc',
                 'p-GG-corr', 'np2', 'eps', 'sphericity', 'W-spher',
                 'p-spher']

    # Replace NaN
    aov = aov.fillna('-')

    aov = aov.reindex(columns=col_order)
    aov.dropna(how='all', axis=1, inplace=True)

    # Round
    aov[['F', 'eps', 'np2']] = aov[['F', 'eps', 'np2']].round(3)

    # Export to .csv
    if export_filename is not None:
        _export_table(aov, export_filename)
    return aov


@pf.register_dataframe_method
def ancova(data=None, dv=None, between=None, covar=None, export_filename=None):
    """ANCOVA with one or more covariate(s).

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    dv : string
        Name of column containing the dependant variable.
    between : string
        Name of column containing the between factor.
    covar : string or list
        Name(s) of column(s) containing the covariate.
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    aov : DataFrame
        ANCOVA summary ::

        'Source' : Names of the factor considered
        'SS' : Sums of squares
        'DF' : Degrees of freedom
        'F' : F-values
        'p-unc' : Uncorrected p-values

    Notes
    -----
    Analysis of covariance (ANCOVA) is a general linear model which blends
    ANOVA and regression. ANCOVA evaluates whether the means of a dependent
    variable (dv) are equal across levels of a categorical independent
    variable (between) often called a treatment, while statistically
    controlling for the effects of other continuous variables that are not
    of primary interest, known as covariates or nuisance variables (covar).

    Note that in the case of one covariate, Pingouin will use a built-in
    function. However, if there are more than one covariate, Pingouin will
    use the statsmodels package to compute the ANCOVA.

    Rows with missing values are automatically removed (listwise deletion).

    See Also
    --------
    anova : One-way and N-way ANOVA

    Examples
    --------
    1. Evaluate the reading scores of students with different teaching method
    and family income as a covariate.

    >>> from pingouin import ancova, read_dataset
    >>> df = read_dataset('ancova')
    >>> ancova(data=df, dv='Scores', covar='Income', between='Method')
         Source           SS  DF          F     p-unc
    0    Method   571.030045   3   3.336482  0.031940
    1    Income  1678.352687   1  29.419438  0.000006
    2  Residual  1768.522365  31        NaN       NaN

    2. Evaluate the reading scores of students with different teaching method
    and family income + BMI as a covariate.

    >>> ancova(data=df, dv='Scores', covar=['Income', 'BMI'], between='Method')
         Source        SS  DF       F     p-unc
    0    Method   552.284   3   3.233  0.036113
    1    Income  1573.952   1  27.637  0.000011
    2       BMI    60.014   1   1.054  0.312842
    3  Residual  1708.509  30     NaN       NaN
    """
    # Safety checks
    assert isinstance(data, pd.DataFrame)
    assert dv in data.columns, '%s is not in data.' % dv
    assert between in data.columns, '%s is not in data.' % between
    assert isinstance(covar, (str, list)), 'covar must be a str or a list.'

    # Drop missing values
    data = data[_flatten_list([dv, between, covar])].dropna()

    # Check the number of covariates
    if isinstance(covar, list):
        if len(covar) > 1:
            return ancovan(dv=dv, covar=covar, between=between, data=data,
                           export_filename=export_filename)
        else:
            covar = covar[0]

    # Assert that covariate is numeric
    assert data[covar].dtype.kind in 'fi', 'Covariate must be numeric.'

    def linreg(x, y):
        return np.corrcoef(x, y)[0, 1] * np.std(y, ddof=1) / np.std(x, ddof=1)

    # Compute slopes
    groups = data[between].unique()
    slopes = np.zeros(shape=groups.shape)
    ss = np.zeros(shape=groups.shape)
    for i, b in enumerate(groups):
        dt_covar = data[data[between] == b][covar].values
        ss[i] = ((dt_covar - dt_covar.mean())**2).sum()
        slopes[i] = linreg(dt_covar, data[data[between] == b][dv].values)
    ss_slopes = ss * slopes
    bw = np.nansum(ss_slopes) / np.sum(ss)
    bt = linreg(data[covar], data[dv])

    # Run the ANOVA
    aov_dv = anova(data=data, dv=dv, between=between, detailed=True)
    aov_covar = anova(data=data, dv=covar, between=between, detailed=True)

    # Create full ANCOVA
    ss_t_dv = aov_dv.at[0, 'SS'] + aov_dv.at[1, 'SS']
    ss_t_covar = aov_covar.at[0, 'SS'] + aov_covar.at[1, 'SS']
    # Sums of squares
    ss_t = ss_t_dv - bt**2 * ss_t_covar
    ss_w = aov_dv.at[1, 'SS'] - bw**2 * aov_covar.at[1, 'SS']
    ss_b = ss_t - ss_w
    ss_c = np.nansum(ss_slopes) * bw
    # DOF
    df_c = 1
    df_b = aov_dv.at[0, 'DF']
    df_w = aov_dv.at[1, 'DF'] - 1
    # Mean squares
    ms_c = ss_c / df_c
    ms_b = ss_b / df_b
    ms_w = ss_w / df_w
    # F-values
    f_c = ms_c / ms_w
    f_b = ms_b / ms_w
    # P-values
    p_c = f(df_c, df_w).sf(f_c)
    p_b = f(df_b, df_w).sf(f_b)

    # Create dataframe
    aov = pd.DataFrame({'Source': [between, covar, 'Residual'],
                        'SS': [ss_b, ss_c, ss_w],
                        'DF': [df_b, df_c, df_w],
                        'F': [f_b, f_c, np.nan],
                        'p-unc': [p_b, p_c, np.nan],
                        })

    # Export to .csv
    if export_filename is not None:
        _export_table(aov, export_filename)

    # Add bw as an attribute (for rm_corr function)
    aov.bw_ = bw
    return aov


def ancovan(data=None, dv=None, covar=None, between=None,
            export_filename=None):
    """ANCOVA with n covariates.

    This is an internal function. The main call to this function should be done
    by the :py:func:`pingouin.ancova` function.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame
    dv : string
        Name of column containing the dependant variable.
    covar : string
        Name(s) of columns containing the covariates.
    between : string
        Name of column containing the between factor.
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.

    Returns
    -------
    aov : DataFrame
        ANCOVA summary ::

        'Source' : Names of the factor considered
        'SS' : Sums of squares
        'DF' : Degrees of freedom
        'F' : F-values
        'p-unc' : Uncorrected p-values
    """
    # Check that stasmodels is installed
    from pingouin.utils import _is_statsmodels_installed
    _is_statsmodels_installed(raise_error=True)
    from statsmodels.api import stats
    from statsmodels.formula.api import ols

    # Check that covariates are numeric ('float', 'int')
    assert all([data[covar[i]].dtype.kind in 'fi' for i in range(len(covar))])

    # Fit ANCOVA model
    formula = dv + ' ~ C(' + between + ')'
    for c in covar:
        formula += ' + ' + c
    model = ols(formula, data=data).fit()
    aov = stats.anova_lm(model, typ=2).reset_index()

    aov.rename(columns={'index': 'Source', 'sum_sq': 'SS',
                        'df': 'DF', 'PR(>F)': 'p-unc'}, inplace=True)

    aov.at[0, 'Source'] = between

    aov['DF'] = aov['DF'].astype(int)
    aov[['SS', 'F']] = aov[['SS', 'F']].round(3)

    # Export to .csv
    if export_filename is not None:
        _export_table(aov, export_filename)

    return aov
