# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
import pandas as pd
from pingouin import (_check_dataframe, _remove_rm_na, _remove_na,
                      _export_table, bayesfactor_ttest, epsilon, sphericity)

__all__ = ["ttest", "rm_anova", "rm_anova2", "anova", "anova2",
           "mixed_anova", "ancova", "ancovan"]


def ttest(x, y, paired=False, tail='two-sided', correction='auto', r=.707):
    """T-test.

    Parameters
    ----------
    x : array_like
        First set of observations.
    y : array_like or float
        Second set of observations. If y is a single value, a one-sample T-test
        is computed.
    paired : boolean
        Specify whether the two observations are related (i.e. repeated
        measures) or independant.
    tail : string
        Specify whether to return two-sided or one-sided p-value.
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
        The default is 0.707 (= np.sqrt(2) / 2).

    Returns
    -------
    stats : pandas DataFrame
        T-test summary ::

        'T-val' : T-value
        'p-val' : p-value
        'dof' : degrees of freedom
        'cohen-d' : Cohen d effect size
        'power' : achieved power of the test ( = 1 - type II error)
        'BF10' : Bayes Factor of the alternative hypothesis

    Notes
    -----
    Missing values are automatically removed from the data. If x and y are
    paired, the entire row is removed.

    The Bayes Factor is approximated using the formula described in ref [1]_:

    .. math::

        BF_{10} = \dfrac{\int_{0}^{\infty}(1 + Ngr^2)^{-1/2}
        (1 + \dfrac{t^2}{v(1 + Ngr^2)})^{-(v+1) / 2}(2\pi)^{-1/2}g^
        {-3/2}e^{-1/2g}}{(1 + \dfrac{t^2}{v})^{-(v+1) / 2}}

    where **t** is the T-value, **v** the degrees of freedom, **N** the
    sample size and **r** the Cauchy scale factor (i.e. prior on effect size).

    References
    ----------

    .. [1] Rouder, J.N., Speckman, P.L., Sun, D., Morey, R.D., Iverson, G.,
       2009. Bayesian t tests for accepting and rejecting the null hypothesis.
       Psychon. Bull. Rev. 16, 225–237. https://doi.org/10.3758/PBR.16.2.225

    Examples
    --------
    1. One-sample T-test.

        >>> from pingouin import ttest
        >>> x = [5.5, 2.4, 6.8, 9.6, 4.2]
        >>> ttest(x, 4)
            T-val   p-val  dof  cohen-d  power   BF10
            1.397  0.2348    4    0.625  0.919  0.766

    2. Paired two-sample T-test (one-tailed).

        >>> from pingouin import ttest
        >>> pre = [5.5, 2.4, 6.8, 9.6, 4.2]
        >>> post = [6.4, 3.4, 6.4, 11., 4.8]
        >>> ttest(pre, post, paired=True, tail='one-sided')
            T-val   p-val  dof  cohen-d  power   BF10
            -2.308   0.04    4     0.25  0.121  3.122

    3. Paired two-sample T-test with missing values.

        >>> from pingouin import ttest
        >>> from numpy import nan
        >>> pre = [5.5, 2.4, nan, 9.6, 4.2]
        >>> post = [6.4, 3.4, 6.4, 11., 4.8]
        >>> ttest(pre, post, paired=True)
            T-val    p-val  dof  cohen-d  power    BF10
            -5.902  0.0097    3   0.306   0.065   7.169

    4. Independant two-sample T-test (equal sample size).

        >>> from pingouin import ttest
        >>> import numpy as np
        >>> np.random.seed(123)
        >>> x = np.random.normal(loc=7, size=20)
        >>> y = np.random.normal(loc=4, size=20)
        >>> ttest(x, y, correction='auto')
            T-val     p-val  dof  cohen-d  power   BF10
            9.106  4.30e-11   38     2.88    1.0  1.4e8

    5. Independant two-sample T-test (unequal sample size).

        >>> from pingouin import ttest
        >>> import numpy as np
        >>> np.random.seed(123)
        >>> x = np.random.normal(loc=7, size=20)
        >>> y = np.random.normal(loc=6.5, size=15)
        >>> ttest(x, y, correction='auto')
            T-val     p-val  dof   dof-corr  cohen-d  power   BF10
            2.327     0.027   33      30.75    0.792  0.614  2.454
    """
    from scipy.stats import ttest_rel, ttest_ind, ttest_1samp
    from pingouin import ttest_power, compute_effsize
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size != y.size and paired:
        print('x and y have unequal sizes. Switching to paired == False.')
        paired = False

    # Remove NA
    x, y = _remove_na(x, y, paired=paired)
    nx = x.size
    ny = y.size
    stats = pd.DataFrame({}, index=['T-test'])

    if ny == 1:
        # Case one sample T-test
        tval, pval = ttest_1samp(x, y)
        dof = nx - 1
        pval = pval / 2 if tail == 'one-sided' else pval

    if ny > 1 and paired is True:
        # Case paired two samples T-test
        tval, pval = ttest_rel(x, y)
        dof = nx - 1
        bf = bayesfactor_ttest(tval, nx, ny, paired=True, r=r)

    elif ny > 1 and paired is False:
        dof = nx + ny - 2
        # Case unpaired two samples T-test
        if correction is True or (correction == 'auto' and nx != ny):
            # Use the Welch separate variance T-test
            tval, pval = ttest_ind(x, y, equal_var=False)
            # dof are approximated using Welch–Satterthwaite equation
            vx = x.var(ddof=1)
            vy = y.var(ddof=1)
            dof_corr = (vx / nx + vy / ny)**2 / ((vx / nx)**2 / (nx - 1)
                                                 + (vy / ny)**2 / (ny - 1))
            stats['dof-corr'] = dof_corr
        else:
            tval, pval = ttest_ind(x, y, equal_var=True)

    pval = pval / 2 if tail == 'one-sided' else pval

    # Effect size and achieved power
    d = compute_effsize(x, y, paired=paired, eftype='cohen')
    power = ttest_power(d, nx, ny, paired=paired, tail=tail)

    # Bayes factor
    bf = bayesfactor_ttest(tval, nx, ny, paired=paired, tail=tail, r=r)

    # Fill output DataFrame
    stats['dof'] = dof
    stats['T-val'] = tval.round(3)
    stats['p-val'] = pval
    stats['tail'] = tail
    stats['cohen-d'] = np.abs(d).round(3)
    stats['power'] = power
    stats['BF10'] = bf

    col_order = ['T-val', 'p-val', 'dof', 'dof-corr', 'tail', 'cohen-d',
                 'power', 'BF10']
    stats = stats.reindex(columns=col_order)
    stats.dropna(how='all', axis=1, inplace=True)
    return stats


def rm_anova(dv=None, within=None, subject=None, data=None, correction='auto',
             remove_na=True, detailed=False, export_filename=None):
    """One-way repeated measures ANOVA.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    within : string
        Name of column containing the within factor.
        If within is a single string, then compute a one-way repeated
        measures ANOVA, if within is a list with two strings, compute a two-way
        repeated measures ANOVA using the rm_anova2 function.
    subject : string
        Name of column containing the subject identifier.
    data : pandas DataFrame
        DataFrame
    correction : string or boolean
        If True, return Greenhouse-Geisser corrected p-value.
        If 'auto' (default), compute Mauchly's test of sphericity to determine
        whether the p-values needs to be corrected.
    remove_na : boolean
        If True, automatically remove from the analysis subjects with one or
        more missing values::

            Ss    x1       x2       x3
            1     5.0      4.2      nan
            2     4.6      3.6      3.9

        In this example, if remove_na == True, Ss 1 will be removed from the
        ANOVA because of the x3 missing value. If False, the two non-missing
        values will be included in the analysis.
    detailed : boolean
        If True, return a full ANOVA table
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
    anova : One-way and two-way ANOVA
    rm_anova2 : Two-way repeated measures ANOVA
    mixed_anova : Two way mixed ANOVA
    friedman : Non-parametric one-way repeated measures ANOVA

    Notes
    -----
    Data are expected to be in long-format.

    Results have been tested against R and JASP.

    Note that if the dataset contains one or more other within subject
    factors, an automatic collapsing to the mean is applied on the dependant
    variable (same behavior as the ezANOVA R package). As such, results can
    differ from those of JASP. If you can, always double-check the results.

    The effect size reported in Pingouin is the partial eta-square.
    However, one should keep in mind that for one-way repeated-measures ANOVA,
    partial eta-square is the same as eta-square
    (Bakeman 2005; Richardson 2011).

    Examples
    --------
    One-way repeated-measures ANOVA (Ryan et al 2013 dataset).

        >>> from pingouin.datasets import read_dataset
        >>> from pingouin import rm_anova, print_table
        >>> df = read_dataset('rm_anova')
        >>> aov = rm_anova(dv='DesireToKill', within='Disgustingness',
                           subject='Subject', data=df, detailed=True)
        >>> print(aov)
    """
    from scipy.stats import f
    if isinstance(within, list):
        if len(within) == 2:
            return rm_anova2(dv=dv, within=within, data=data, subject=subject,
                             export_filename=export_filename)
        elif len(within) == 1:
            within = within[0]

    # Check data
    _check_dataframe(dv=dv, within=within, data=data, subject=subject,
                     effects='within')

    # Collapse to the mean
    data = data.groupby([subject, within]).mean().reset_index()

    # Remove NaN
    if remove_na and data[dv].isnull().any():
        data = _remove_rm_na(dv=dv, within=within, subject=subject,
                             data=data[[subject, within, dv]])

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


def rm_anova2(dv=None, within=None, subject=None, data=None,
              export_filename=None):
    """Two-way repeated measures ANOVA.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    within : list
        Names of column containing the two within factor
        (e.g. ['Time', 'Treatment'])
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

    Notes
    -----
    Data are expected to be in long-format and perfectly balanced.

    Results have been tested against ezANOVA (R) and MNE.

    See Also
    --------
    anova : One-way and two-way ANOVA
    rm_anova : One-way repeated measures ANOVA
    mixed_anova : Two way mixed ANOVA
    friedman : Non-parametric one-way repeated measures ANOVA

    Examples
    --------
    Two-way repeated-measures ANOVA.

        >>> from pingouin.datasets import read_dataset
        >>> from pingouin import rm_anova2, print_table
        >>> df = read_dataset('rm_anova')
        >>> aov = rm_anova2(dv='DesireToKill',
                           within=['Disgustingness', 'Frighteningness'],
                           subject='Subject', data=df)
        >>> print(aov)
    """
    from scipy.stats import f
    a, b = within

    # Validate the dataframe
    _check_dataframe(dv=dv, within=within, data=data, subject=subject,
                     effects='within')

    # Remove NaN
    if data[[a, b, subject, dv]].isnull().any().any():
        data = _remove_rm_na(dv=dv, within=within, subject=subject,
                             data=data[[a, b, subject, dv]])

    # Group sizes and grandmean
    n_a = data[a].unique().size
    n_b = data[b].unique().size
    n_s = data[subject].unique().size
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


def anova(dv=None, between=None, data=None, detailed=False,
          export_filename=None):
    """One-way and two-way ANOVA.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    between : string or list
        Name(s) of column containing the between factor.
        If between is a single string, then compute a one-way ANOVA, if between
        is a list with two strings, compute a two-way ANOVA.
    data : pandas DataFrame
        DataFrame
    detailed : boolean
        If True, return a detailed ANOVA table
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
        'SS' :Sums of squares
        'DF' : Degrees of freedom
        'MS' : Mean squares
        'F' : F-values
        'p-unc' : uncorrected p-values
        'np2' : Partial eta-square effect sizes

    See Also
    --------
    rm_anova : One-way repeated measures ANOVA
    rm_anova2 : Two-way repeated measures ANOVA
    mixed_anova : Two way mixed ANOVA
    kruskal : Non-parametric one-way ANOVA


    Notes
    -----
    The effect size reported in Pingouin is the partial eta-square.
    However, one should keep in mind that for one-way ANOVA
    partial eta-square is the same as eta-square and generalized eta-square.
    For more details, see Bakeman 2005; Richardson 2011.

    Results have been tested against R, Matlab and JASP.

    Examples
    --------
    1. One-way ANOVA on the pain threshold dataset.

        >>> from pingouin import anova, print_table
        >>> from pingouin.datasets import read_dataset
        >>> df = read_dataset('anova')
        >>> aov = anova(dv='Pain threshold', between='Hair color', data=df,
                        detailed=True, export_filename='pain_anova.csv')
        >>> print_table(aov)

    2. Two-way ANOVA.

        >>> import pandas as pd
        >>> from pingouin import anova, print_table
        >>> df = pd.read_csv('dataset.csv')
        >>> aov = anova(dv='DV', between=['factor1, 'factor2'], data=df,
                        export_filename='anova.csv')
        >>> print_table(aov)

    """
    if isinstance(between, list):
        if len(between) == 2:
            return anova2(dv=dv, between=between, data=data,
                          export_filename=export_filename)
        elif len(between) == 1:
            between = between[0]

    # Check data
    _check_dataframe(dv=dv, between=between, data=data, effects='between')

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
    from scipy.stats import f
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


def anova2(dv=None, between=None, data=None, export_filename=None):
    """Two-way ANOVA.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    between : list of string
        Name of column containing the two between factors. Must contain exactly
        two values (e.g. ['factor1', 'factor2'])
    data : pandas DataFrame
        DataFrame
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
        'SS' :Sums of squares
        'DF' : Degrees of freedom
        'MS' : Mean squares
        'F' : F-values
        'p-unc' : uncorrected p-values
        'np2' : Partial eta-square effect sizes

    See Also
    --------
    anova : One-way and two way ANOVA
    rm_anova : One-way repeated measures ANOVA
    rm_anova2 : Two-way repeated measures ANOVA
    mixed_anova : Two way mixed ANOVA
    kruskal : Non-parametric one-way ANOVA

    Notes
    -----
    Results have been tested against JASP.

    Examples
    --------
    Compute a two-way ANOVA.

        >>> import pandas as pd
        >>> from pingouin import anova2, print_table
        >>> df = pd.read_csv('dataset.csv')
        >>> aov = anova(dv='DV', between=['factor1', 'factor2'], data=df,
                        export_filename='anova.csv')
        >>> print_table(aov)
    """
    from scipy.stats import f

    # Validate the dataframe
    _check_dataframe(dv=dv, between=between, data=data, effects='between')

    # Assert that there are two factors
    if not isinstance(between, list):
        return anova(dv=dv, between=between, data=data,
                     export_filename=export_filename, detailed=True)

    if len(between) == 1:
        return anova(dv=dv, between=between[0], data=data,
                     export_filename=export_filename, detailed=True)

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)

    fac1, fac2 = between
    aov_fac1 = anova(data=data, dv=dv, between=fac1, detailed=True)
    aov_fac2 = anova(data=data, dv=dv, between=fac2, detailed=True)

    # Sums of squares
    ss_fac1 = aov_fac1.loc[0, 'SS']
    ss_fac2 = aov_fac2.loc[0, 'SS']
    ss_tot = ((data[dv] - data[dv].mean())**2).sum()
    ss_resid = np.sum(data.groupby([fac1, fac2]).apply(lambda x:
                                                       (x - x.mean())**2))[0]
    ss_inter = ss_tot - (ss_resid + ss_fac1 + ss_fac2)

    # Degrees of freedom
    df_fac1 = aov_fac1.loc[0, 'DF']
    df_fac2 = aov_fac2.loc[0, 'DF']
    df_inter = (data[fac1].unique().size - 1) * (data[fac2].unique().size - 1)
    df_resid = data[dv].size - (data[fac1].unique().size
                                * data[fac2].unique().size)

    # Mean squares
    ms_fac1 = aov_fac1.loc[0, 'MS']
    ms_fac2 = aov_fac2.loc[0, 'MS']
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
    np2_fac1 = (fval_fac1 * df_fac1) / (fval_fac1 * df_fac1 + df_resid)
    np2_fac2 = (fval_fac2 * df_fac2) / (fval_fac2 * df_fac2 + df_resid)
    np2_inter = (fval_inter * df_inter) / (fval_inter * df_inter + df_resid)

    # Create output dataframe
    aov = pd.DataFrame({'Source': [fac1, fac2, fac1 + ' * ' + fac2,
                                   'residual'],
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


def mixed_anova(dv=None, within=None, subject=None, between=None, data=None,
                correction='auto', remove_na=True, export_filename=None):
    """Mixed-design (split-plot) ANOVA.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    within : string
        Name of column containing the within factor.
    subject : string
        Name of column containing the subject identifier.
    between : string
        Name of column containing the between factor.
    data : pandas DataFrame
        DataFrame
    correction : string or boolean
        If True, return Greenhouse-Geisser corrected p-value.
        If 'auto' (default), compute Mauchly's test of sphericity to determine
        whether the p-values needs to be corrected.
    remove_na : boolean
        If True, automatically remove from the analysis subjects with one or
        more missing values::

            Ss    x1       x2       x3
            1     5.0      4.2      nan
            2     4.6      3.6      3.9

        In this example, if remove_na == True, Ss 1 will be removed from the
        ANOVA because of the x3 missing value. If False, the two non-missing
        values will be included in the analysis.
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
    anova : One-way and two-way ANOVA
    rm_anova : One-way repeated measures ANOVA
    rm_anova2 : Two-way repeated measures ANOVA

    Notes
    -----
    Results have been tested against R and JASP.

    Examples
    --------
    Compute a two-way mixed model ANOVA.

        >>> from pingouin.datasets import read_dataset
        >>> from pingouin import mixed_anova, print_table
        >>> df = read_dataset('rm_anova')
        >>> aov = mixed_anova(dv='DesireToKill', between='Gender',
                              within='Disgustingness',
                              subject='Subject', data=df)
        >>> print(aov)
    """
    from scipy.stats import f

    # Check data
    _check_dataframe(dv=dv, within=within, between=between, data=data,
                     subject=subject, effects='interaction')

    # Collapse to the mean
    data = data.groupby([subject, within, between]).mean().reset_index()

    # Remove NaN
    if remove_na and data[dv].isnull().any():
        data = _remove_rm_na(dv=dv, within=within, subject=subject,
                             data=data[[subject, between, within, dv]])

    # SUMS OF SQUARES
    grandmean = data[dv].mean()
    # Extract main effects of time and between
    mtime = rm_anova(dv=dv, within=within, subject=subject, data=data,
                     correction=correction, remove_na=False, detailed=True)
    mbetw = anova(dv=dv, between=between, data=data, detailed=True)
    # Extract SS total, residuals and interactions
    grp = data.groupby([between, within])[dv]
    sstotal = grp.apply(lambda x: (x - grandmean)**2).sum()
    # sst = residuals within + residuals between
    sst = grp.apply(lambda x: (x - x.mean())**2).sum()
    # Interaction
    ssinter = sstotal - (sst + mtime.loc[0, 'SS'] + mbetw.loc[0, 'SS'])
    sswg = mtime.loc[1, 'SS'] - ssinter
    sseb = sstotal - (mtime.loc[0, 'SS'] + mbetw.loc[0, 'SS'] + sswg + ssinter)

    # DEGREES OF FREEDOM
    n_obs = data.groupby(within)[dv].count().max()
    dftime = mtime.loc[0, 'DF']
    dfbetween = mbetw.loc[0, 'DF']
    dfeb = n_obs - data.groupby(between)[dv].count().count()
    dfwg = dftime * dfeb
    dfinter = mtime.loc[0, 'DF'] * mbetw.loc[0, 'DF']

    # MEAN SQUARES
    mseb = sseb / dfeb
    mswg = sswg / dfwg
    msinter = ssinter / dfinter

    # F VALUES
    fbetween = mbetw.loc[0, 'MS'] / mseb
    ftime = mtime.loc[0, 'MS'] / mswg
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
    aov.loc[0, 'F'], aov.loc[1, 'F'] = fbetween, ftime
    aov.loc[0, 'p-unc'], aov.loc[1, 'p-unc'] = pbetween, ptime
    aov.loc[0, 'np2'], aov.loc[1, 'np2'] = npsq_between, npsq_time
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
    aov['eps'] = [np.nan, mtime.loc[0, 'eps'], np.nan]
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


def ancova(dv=None, covar=None, between=None, data=None,
           export_filename=None, return_bw=False):
    """ANCOVA with one or more covariate(s).

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    covar : string or list
        Name(s) of column(s) containing the covariate.
    between : string
        Name of column containing the between factor.
    data : pandas DataFrame
        DataFrame
    export_filename : string
        Filename (without extension) for the output file.
        If None, do not export the table.
        By default, the file will be created in the current python console
        directory. To change that, specify the filename with full path.
    return_bw : bool
        If True, return beta within parameter (used for the rm_corr function)

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

    See Also
    --------
    anova : One-way and two-way ANOVA

    Examples
    --------
    1. Evaluate the reading scores of students with different teaching method
    and family income as a covariate.

        >>> from pingouin.datasets import read_dataset
        >>> from pingouin import ancova
        >>> df = read_dataset('ancova')
        >>> ancova(data=df, dv='Scores', covar='Income', between='Method')

    2. Evaluate the reading scores of students with different teaching method
    and family income + BMI as a covariate.

        >>> from pingouin.datasets import read_dataset
        >>> from pingouin import ancova
        >>> df = read_dataset('ancova')
        >>> ancova(data=df, dv='Scores', covar=['Income', 'BMI'],
            between='Method')
    """
    if isinstance(covar, list):
        if len(covar) > 1:
            return ancovan(dv=dv, covar=covar, between=between, data=data,
                           export_filename=export_filename)
        else:
            covar = covar[0]

    # Assert that covariate is numeric
    assert data[covar].dtype.kind in 'fi'

    from scipy.stats import f

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
    bw = ss_slopes.sum() / ss.sum()
    bt = linreg(data[covar], data[dv])

    # Run the ANOVA
    aov_dv = anova(data=data, dv=dv, between=between, detailed=True)
    aov_covar = anova(data=data, dv=covar, between=between, detailed=True)

    # Create full ANCOVA
    ss_t_dv = aov_dv.loc[0, 'SS'] + aov_dv.loc[1, 'SS']
    ss_t_covar = aov_covar.loc[0, 'SS'] + aov_covar.loc[1, 'SS']
    # Sums of squares
    ss_t = ss_t_dv - bt**2 * ss_t_covar
    ss_w = aov_dv.loc[1, 'SS'] - bw**2 * aov_covar.loc[1, 'SS']
    ss_b = ss_t - ss_w
    ss_c = ss_slopes.sum() * bw
    # DOF
    df_c = 1
    df_b = aov_dv.loc[0, 'DF']
    df_w = aov_dv.loc[1, 'DF'] - 1
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

    if return_bw:
        return aov, bw
    else:
        return aov


def ancovan(dv=None, covar=None, between=None, data=None,
            export_filename=None):
    """ANCOVA with n covariates.

    Requires statsmodels.

    Parameters
    ----------
    dv : string
        Name of column containing the dependant variable.
    covar : string
        Name(s) of columns containing the covariates.
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

    See Also
    --------
    ancova : ANCOVA with one covariate
    anova : One-way and two-way ANOVA

    Examples
    --------
    1. Evaluate the reading scores of students with different teaching method
    and family income and BMI as covariates.

        >>> from pingouin.datasets import read_dataset
        >>> from pingouin import ancovan
        >>> df = read_dataset('ancova')
        >>> ancovan(data=df, dv='Scores', covar=['Income', 'BMI'],
                    between='Method')
    """
    # Assert that there are at least two covariates
    if not isinstance(covar, list):
        return ancova(dv=dv, covar=covar, between=between, data=data,
                      export_filename=export_filename)

    if len(covar) == 1:
        return ancova(dv=dv, covar=covar[0], between=between, data=data,
                      export_filename=export_filename)

    # Check that stasmodels is installed
    from pingouin.utils import is_statsmodels_installed
    is_statsmodels_installed(raise_error=True)
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

    aov.loc[0, 'Source'] = between

    aov['DF'] = aov['DF'].astype(int)
    aov[['SS', 'F']] = aov[['SS', 'F']].round(3)

    # Export to .csv
    if export_filename is not None:
        _export_table(aov, export_filename)

    return aov
