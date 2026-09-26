# Author: Raphael Vallat <raphaelvallat9@gmail.com>
import warnings

import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.stats import f

from .bayesian import bayesfactor_ttest
from .distribution import _contrast_cov, _gg_epsilon, _mauchly_sphericity
from .utils import (
    _check_alternative,
    _check_dataframe,
    _flatten_list,
    _postprocess_dataframe,
    remove_na,
)

__all__ = ["ttest", "rm_anova", "anova", "welch_anova", "mixed_anova", "ancova"]


def ttest(x, y, paired=False, alternative="two-sided", correction="auto", r=0.707, confidence=0.95):
    """T-test.

    Parameters
    ----------
    x : array_like
        First set of observations.
    y : array_like or float
        Second set of observations. If ``y`` is a single value, a one-sample
        T-test is computed against that value (= "mu" in the t.test R
        function).
    paired : boolean
        Specify whether the two observations are related (i.e. repeated
        measures) or independent.
    alternative : string
        Defines the alternative hypothesis, or tail of the test. Must be one of
        "two-sided" (default), "greater" or "less". Both "greater" and "less" return one-sided
        p-values. "greater" tests against the alternative hypothesis that the mean of ``x``
        is greater than the mean of ``y``.
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
    confidence : float
        Confidence level for the confidence intervals (0.95 = 95%)

        .. versionadded:: 0.3.9

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'T'``: T-value
        * ``'dof'``: degrees of freedom
        * ``'alternative'``: alternative of the test
        * ``'p_val'``: p-value
        * ``'CI95'``: confidence intervals of the difference in means
        * ``'cohen_d'``: Cohen d effect size
        * ``'power'``: achieved power of the test ( = 1 - type II error)
        * ``'BF10'``: Bayes Factor of the alternative hypothesis

    See also
    --------
    mwu, wilcoxon, anova, rm_anova, pairwise_tests, compute_effsize

    Notes
    -----
    Missing values are automatically removed from the data. If ``x`` and
    ``y`` are paired, the entire row is removed (= listwise deletion).

    The **T-value for unpaired samples** is defined as:

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
    of the mean of the differences. The p-value is then calculated using a
    T-distribution with :math:`n-1` degrees of freedom.

    The scaled Jeffrey-Zellner-Siow (JZS) Bayes Factor is approximated
    using the :py:func:`pingouin.bayesfactor_ttest` function. It is only provided for two-sided
    tests; the BF value will be set to NaN for one-sided tests.

    Results have been tested against JASP and the `t.test` R function.

    References
    ----------
    * https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm

    * Delacre, M., Lakens, D., & Leys, C. (2017). Why psychologists should
      by default use Welch’s t-test instead of Student’s t-test.
      International Review of Social Psychology, 30(1).

    * Zimmerman, D. W. (2004). A note on preliminary tests of equality of
      variances. British Journal of Mathematical and Statistical
      Psychology, 57(1), 173-181.

    * Rouder, J.N., Speckman, P.L., Sun, D., Morey, R.D., Iverson, G.,
      2009. Bayesian t tests for accepting and rejecting the null
      hypothesis. Psychon. Bull. Rev. 16, 225–237.
      https://doi.org/10.3758/PBR.16.2.225

    Examples
    --------
    1. One-sample T-test.

    >>> from pingouin import ttest
    >>> x = [5.5, 2.4, 6.8, 9.6, 4.2]
    >>> ttest(x, 4).round(2)
              T  dof alternative  p_val          CI95  cohen_d  power   BF10
    T_test  1.4    4   two-sided   0.23  [2.32, 9.08]     0.62   0.19  0.766

    2. One sided paired T-test.

    >>> pre = [5.5, 2.4, 6.8, 9.6, 4.2]
    >>> post = [6.4, 3.4, 6.4, 11.0, 4.8]
    >>> ttest(pre, post, paired=True, alternative="less").round(2)
               T  dof alternative  p_val           CI95  cohen_d  power
    T_test -2.31    4        less   0.04  [-inf, -0.05]     0.25   0.12

    Now testing the opposite alternative hypothesis

    >>> ttest(pre, post, paired=True, alternative="greater").round(2)
               T  dof alternative  p_val          CI95  cohen_d  power
    T_test -2.31    4     greater   0.96  [-1.35, inf]     0.25   0.02

    3. Paired T-test with missing values.

    >>> import numpy as np
    >>> pre = [5.5, 2.4, np.nan, 9.6, 4.2]
    >>> post = [6.4, 3.4, 6.4, 11.0, 4.8]
    >>> ttest(pre, post, paired=True).round(3)
                T  dof alternative  p_val           CI95  cohen_d  power   BF10
    T_test -5.902    3   two-sided   0.01  [-1.5, -0.45]    0.306  0.073  7.169

    Compare with SciPy

    >>> from scipy.stats import ttest_rel
    >>> np.round(ttest_rel(pre, post, nan_policy="omit"), 3)
    array([-5.902,  0.01 ])

    4. Independent two-sample T-test with equal sample size.

    >>> np.random.seed(123)
    >>> x = np.random.normal(loc=7, size=20)
    >>> y = np.random.normal(loc=4, size=20)
    >>> ttest(x, y)
                   T  dof alternative         p_val          CI95   cohen_d  power       BF10
    T_test  9.106452   38   two-sided  4.306971e-11  [2.64, 4.15]  2.879713    1.0  1.366e+08

    5. Independent two-sample T-test with unequal sample size. A Welch's T-test is used.

    >>> np.random.seed(123)
    >>> y = np.random.normal(loc=6.5, size=15)
    >>> ttest(x, y)
                   T        dof alternative     p_val           CI95   cohen_d     power   BF10
    T_test  1.996537  31.567592   two-sided  0.054561  [-0.02, 1.65]  0.673518  0.481867  1.469

    6. However, the Welch's correction can be disabled:

    >>> ttest(x, y, correction=False)
                   T  dof alternative     p_val           CI95   cohen_d     power   BF10
    T_test  1.971859   33   two-sided  0.057056  [-0.03, 1.66]  0.673518  0.481867  1.418

    Compare with SciPy

    >>> from scipy.stats import ttest_ind
    >>> np.round(ttest_ind(x, y, equal_var=True), 6)  # T value and p-value
    array([1.971859, 0.057056])
    """
    from scipy.stats import t, ttest_1samp, ttest_ind, ttest_rel

    from pingouin import compute_effsize, power_ttest, power_ttest2n

    # Check arguments
    _check_alternative(alternative)
    assert 0 < confidence < 1, "confidence must be between 0 and 1."

    x = np.asarray(x)
    y = np.asarray(y)
    # Accept any truthy / falsy value (e.g. numpy bool or 0 / 1)
    paired = bool(paired)

    if x.size != y.size and paired:
        warnings.warn("x and y have unequal sizes. Switching to paired == False. Check your data.")
        paired = False

    # Remove rows with missing values
    x, y = remove_na(x, y, paired=paired)
    nx, ny = x.size, y.size

    if ny == 1:
        # Case one sample T-test
        # Pass the population mean as a scalar so that SciPy returns a scalar t-value
        tval, pval = ttest_1samp(x, y.item(), alternative=alternative)
        dof = nx - 1
        se = np.sqrt(x.var(ddof=1) / nx)
    elif paired:
        # Case paired two samples T-test
        # Do not compute if two arrays are identical (avoid SciPy warning)
        if np.array_equal(x, y):
            warnings.warn("x and y are equals. Cannot compute T or p-value.")
            tval, pval = np.nan, np.nan
        else:
            tval, pval = ttest_rel(x, y, alternative=alternative)
        dof = nx - 1
        se = np.sqrt(np.var(x - y, ddof=1) / nx)
    else:
        dof = nx + ny - 2
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        # Case unpaired two samples T-test
        if correction is True or (correction == "auto" and nx != ny):
            # Use the Welch separate variance T-test
            tval, pval = ttest_ind(x, y, equal_var=False, alternative=alternative)
            # Compute sample standard deviation
            # dof are approximated using Welch–Satterthwaite equation
            vnx, vny = vx / nx, vy / ny
            dof = (vnx + vny) ** 2 / (vnx**2 / (nx - 1) + vny**2 / (ny - 1))
            se = np.sqrt(vnx + vny)
        else:
            tval, pval = ttest_ind(x, y, equal_var=True, alternative=alternative)
            # Pooled variance
            svar = ((nx - 1) * vx + (ny - 1) * vy) / dof
            se = np.sqrt(svar * (1 / nx + 1 / ny))

    # Effect size
    d = compute_effsize(x, y, paired=paired, eftype="cohen")

    # Confidence interval for the (difference in) means
    # Compare to the t.test r function
    if alternative == "two-sided":
        alpha = 1 - confidence
        conf = 1 - alpha / 2  # 0.975
    else:
        conf = confidence
    tcrit = t.ppf(conf, dof)
    ci = np.array([tval - tcrit, tval + tcrit]) * se
    if ny == 1:
        ci += y

    if alternative == "greater":
        ci[1] = np.inf
    elif alternative == "less":
        ci[0] = -np.inf

    # Rename CI
    ci_name = "CI%.0f" % (100 * confidence)

    # Achieved power
    if ny == 1:
        # One-sample
        power = power_ttest(
            d=d, n=nx, power=None, alpha=0.05, contrast="one-sample", alternative=alternative
        )
    elif paired:
        # Paired two-sample
        power = power_ttest(
            d=d, n=nx, power=None, alpha=0.05, contrast="paired", alternative=alternative
        )
    else:
        # Independent two-samples
        if nx == ny:
            # Equal sample sizes
            power = power_ttest(
                d=d, n=nx, power=None, alpha=0.05, contrast="two-samples", alternative=alternative
            )
        else:
            # Unequal sample sizes
            power = power_ttest2n(nx, ny, d=d, power=None, alpha=0.05, alternative=alternative)

    # Create output dictionnary
    stats = {
        "T": tval,
        "dof": dof,
        "alternative": alternative,
        "p_val": pval,
        ci_name: [ci],
        "cohen_d": abs(d),
        "power": power,
    }

    if alternative == "two-sided":
        stats["BF10"] = bayesfactor_ttest(tval, nx, ny, paired=paired, alternative=alternative, r=r)

    # Convert to dataframe
    stats = pd.DataFrame(stats, index=["T_test"])
    return _postprocess_dataframe(stats)


def _pivot_rm(data, dv, within, subject, between=None):
    """Wide-format table of a repeated measures design, and the matching long-format data.

    Pivoting and melting the table has several effects:
    1) Force missing values to be explicit (a NaN cell is created)
    2) Automatic collapsing to the mean if multiple within factors are present
    3) Remove subjects with missing values (listwise deletion).
    The latter is the same behavior as JASP (= strict complete-case analysis).
    For a mixed design, ``between`` is added to the index of the wide-format table.
    """
    index = subject if between is None else [subject, between]
    data_piv = data.pivot_table(index=index, columns=within, values=dv, observed=True).dropna()
    data = data_piv.melt(ignore_index=False, value_name=dv).reset_index()
    return data_piv, data


def _gg_corrected_pval(fval, ddof1, ddof2, eps):
    """P-value(s) of the F-test(s) with Greenhouse-Geisser corrected degrees of freedom."""
    return f(np.maximum(ddof1 * eps, 1.0), np.maximum(ddof2 * eps, 1.0)).sf(fval)


@pf.register_dataframe_method
def rm_anova(
    data=None, dv=None, within=None, subject=None, correction="auto", detailed=False, effsize="ng2"
):
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
        Name of column containing the dependent variable (only required if
        ``data`` is in long format).
    within : string or list of string
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

        For two-way design, this argument is ignored: both the uncorrected and
        Greenhouse-Geisser corrected p-values are always returned, together with
        Mauchly's test of sphericity for each effect.
    detailed : boolean
        If True, return a full ANOVA table.
    effsize : string
        Effect size. Must be one of 'np2' (partial eta-squared), 'n2'
        (eta-squared) or 'ng2'(generalized eta-squared, default). Note that for
        one-way repeated measure ANOVA, eta-squared is the same as the generalized eta-squared.

    Returns
    -------
    aov : :py:class:`pandas.DataFrame`
        ANOVA summary:

        * ``'Source'``: Name of the within-group factor
        * ``'ddof1'``: Degrees of freedom (numerator)
        * ``'ddof2'``: Degrees of freedom (denominator)
        * ``'F'``: F-value
        * ``'p_unc'``: Uncorrected p-value
        * ``'ng2'``: Generalized eta-square effect size
        * ``'eps'``: Greenhouse-Geisser epsilon factor (= index of sphericity)
        * ``'p_GG_corr'``: Greenhouse-Geisser corrected p-value
        * ``'W_spher'``: Sphericity test statistic
        * ``'p_spher'``: p-value of the sphericity test
        * ``'sphericity'``: sphericity of the data (boolean)

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

    .. math::
        SS_{\\text{total}} = SS_{\\text{effect}} +
        (SS_{\\text{subjects}} + SS_{\\text{error}})

    with

    .. math::
        SS_{\\text{total}} = \\sum_i^r \\sum_j^n (Y_{ij} - \\overline{Y})^2

        SS_{\\text{effect}} = \\sum_i^r n_i(\\overline{Y_i} - \\overline{Y})^2

        SS_{\\text{subjects}} = r\\sum (\\overline{Y}_s - \\overline{Y})^2

        SS_{\\text{error}} = SS_{\\text{total}} - SS_{\\text{effect}} -
        SS_{\\text{subjects}}


    where :math:`i=1,...,r; j=1,...,n_i`, :math:`r` is the number of
    conditions, :math:`n_i` the number of observations for each condition,
    :math:`\\overline{Y}` the grand mean of the data, :math:`\\overline{Y_i}`
    the mean of the :math:`i^{th}` condition and :math:`\\overline{Y}_{subj}`
    the mean of the :math:`s^{th}` subject.

    The F-statistics is then defined as:

    .. math::

        F^* = \\frac{MS_{\\text{effect}}}{MS_{\\text{error}}} =
        \\frac{\\frac{SS_{\\text{effect}}}
        {r-1}}{\\frac{SS_{\\text{error}}}{(n - 1)(r - 1)}}

    and the p-value can be calculated using a F-distribution with
    :math:`v_{\\text{effect}} = r - 1` and
    :math:`v_{\\text{error}} = (n - 1)(r - 1)` degrees of freedom.

    The default effect size reported in Pingouin is the generalized eta-squared,
    which is equivalent to eta-squared for one-way repeated measures ANOVA.

    .. math::
        \\eta_g^2 = \\frac{SS_{\\text{effect}}}{SS_{\\text{total}}}

    The partial eta-squared is defined as:

    .. math::
        \\eta_p^2 = \\frac{SS_{\\text{effect}}}{SS_{\\text{effect}} + SS_{\\text{error}}}

    Missing values are automatically removed using a strict listwise approach (= complete-case
    analysis). In other words, any subject with one or more missing value(s) is completely removed
    from the dataframe prior to running the test. This could drastically decrease the power of the
    ANOVA if many missing values are present. In that case, we strongly recommend using linear
    mixed effect modelling, which can handle missing values in repeated measures.

    For two-way repeated measures ANOVA, the epsilon and Mauchly's test of sphericity of the
    interaction are computed using the Kronecker product of the orthonormal contrasts of each
    factor, as in R, SPSS and JASP. For the main effects, the p-value of Mauchly's test uses the
    same chi-square approximation as R (afex / car), and can therefore very slightly differ from
    :py:func:`pingouin.sphericity` applied to the main effect alone.

    Examples
    --------
    1. One-way repeated measures ANOVA using a wide-format dataset

    >>> import pingouin as pg
    >>> data = pg.read_dataset("rm_anova_wide")
    >>> pg.rm_anova(data)
       Source  ddof1  ddof2         F     p_unc       ng2       eps
    0  Within      3     24  5.200652  0.006557  0.346392  0.694329

    2. One-way repeated-measures ANOVA using a long-format dataset.

    We're also specifying two additional options here: ``detailed=True`` means
    that we'll get a more detailed ANOVA table, and ``effsize='np2'``
    means that we want to get the partial eta-squared effect size instead
    of the default (generalized) eta-squared.

    >>> df = pg.read_dataset("rm_anova")
    >>> aov = pg.rm_anova(
    ...     dv="DesireToKill",
    ...     within="Disgustingness",
    ...     subject="Subject",
    ...     data=df,
    ...     detailed=True,
    ...     effsize="np2",
    ... )
    >>> aov.round(3)
               Source       SS  DF      MS       F  p_unc    np2  eps
    0  Disgustingness   27.485   1  27.485  12.044  0.001  0.116  1.0
    1           Error  209.952  92   2.282     NaN    NaN    NaN  NaN

    3. Two-way repeated-measures ANOVA

    >>> aov = pg.rm_anova(
    ...     dv="DesireToKill",
    ...     within=["Disgustingness", "Frighteningness"],
    ...     subject="Subject",
    ...     data=df,
    ... )

    4. As a :py:class:`pandas.DataFrame` method

    >>> df.rm_anova(dv="DesireToKill", within="Disgustingness", subject="Subject", detailed=False)
               Source  ddof1  ddof2          F     p_unc       ng2  eps
    0  Disgustingness      1     92  12.043878  0.000793  0.025784  1.0
    """
    assert effsize in ["n2", "np2", "ng2"], "effsize must be n2, np2 or ng2."
    if isinstance(within, list):
        assert len(within) > 0, "Within cannot be empty."
        if len(within) == 1:
            within = within[0]
        elif len(within) == 2:
            return rm_anova2(dv=dv, within=within, data=data, subject=subject, effsize=effsize)
        else:
            raise ValueError("Repeated measures ANOVA with three or more factors is not supported.")

    # Convert from wide to long-format, if needed
    if all([v is None for v in [dv, within, subject]]):
        assert isinstance(data, pd.DataFrame)
        data = data._get_numeric_data().dropna()  # Listwise deletion of missing values
        assert data.shape[0] > 2, "Data must have at least 3 non-missing rows."
        assert data.shape[1] > 1, "Data must contain at least two columns."
        data["Subj"] = np.arange(data.shape[0])
        data = data.melt(id_vars="Subj", var_name="Within", value_name="DV")
        subject, within, dv = "Subj", "Within", "DV"

    # Check dataframe
    data = _check_dataframe(dv=dv, within=within, data=data, subject=subject, effects="within")

    assert not data[within].isnull().any(), "Cannot have missing values in `within`."
    assert not data[subject].isnull().any(), "Cannot have missing values in `subject`."

    # Wide-format table, and complete-case long-format data
    data_piv, data = _pivot_rm(data, dv, within, subject)

    # Groupby
    # I think that observed=True is actually not needed here since we have already used
    # `observed=True` in pivot_table.
    grp_with = data.groupby(within, observed=True, group_keys=False)[dv]
    rm = list(data[within].unique())
    n_rm = len(rm)
    n_obs = int(grp_with.count().max())
    grandmean = data[dv].mean(numeric_only=True)

    # Calculate sums of squares
    ss_with = ((grp_with.mean(numeric_only=True) - grandmean) ** 2 * grp_with.count()).sum()
    ss_resall = grp_with.apply(lambda x: (x - x.mean()) ** 2).sum()
    # sstotal = sstime + ss_resall =  sstime + (sssubj + sserror)
    # ss_total = ((data[dv] - grandmean)**2).sum()
    # We can further divide the residuals into a within and between component:
    grp_subj = data.groupby(subject, observed=True)[dv]
    ss_resbetw = n_rm * np.sum((grp_subj.mean(numeric_only=True) - grandmean) ** 2)
    ss_reswith = ss_resall - ss_resbetw

    # Calculate degrees of freedom
    ddof1 = n_rm - 1
    ddof2 = ddof1 * (n_obs - 1)

    # Calculate MS, F and p-values
    ms_with = ss_with / ddof1
    ms_reswith = ss_reswith / ddof2
    fval = ms_with / ms_reswith
    p_unc = f(ddof1, ddof2).sf(fval)

    # Calculating effect sizes (see Bakeman 2005; Lakens 2013)
    # https://github.com/raphaelvallat/pingouin/issues/251
    if effsize == "np2":
        # Partial eta-squared
        ef = ss_with / (ss_with + ss_reswith)
    else:
        # (Generalized) eta-squared, ng2 == n2
        ef = ss_with / (ss_with + ss_resall)

    # Epsilon and Mauchly's test of sphericity, from the covariance matrix of the contrasts
    M = _contrast_cov(data_piv)
    eps = _gg_epsilon(M)
    spher, W_spher, p_spher = _mauchly_sphericity(M, n_obs - 1, n_rm)
    # Sphericity assumption only applies if there are more than 2 levels
    correction = not spher if correction == "auto" else (bool(correction) and n_rm >= 3)

    # If required, apply Greenhouse-Geisser correction for sphericity
    if correction:
        p_corr = _gg_corrected_pval(fval, ddof1, ddof2, eps)

    # Create output dataframe
    if not detailed:
        aov = pd.DataFrame(
            {
                "Source": within,
                "ddof1": ddof1,
                "ddof2": ddof2,
                "F": fval,
                "p_unc": p_unc,
                effsize: ef,
                "eps": eps,
            },
            index=[0],
        )
        if correction:
            aov["p_GG_corr"] = p_corr
            aov["W_spher"] = W_spher
            aov["p_spher"] = p_spher
            aov["sphericity"] = spher

        col_order = [
            "Source",
            "ddof1",
            "ddof2",
            "F",
            "p_unc",
            "p_GG_corr",
            effsize,
            "eps",
            "sphericity",
            "W_spher",
            "p_spher",
        ]
    else:
        aov = pd.DataFrame(
            {
                "Source": [within, "Error"],
                "SS": [ss_with, ss_reswith],
                "DF": [ddof1, ddof2],
                "MS": [ms_with, ms_reswith],
                "F": [fval, np.nan],
                "p_unc": [p_unc, np.nan],
                effsize: [ef, np.nan],
                "eps": [eps, np.nan],
            }
        )
        if correction:
            aov["p_GG_corr"] = [p_corr, np.nan]
            aov["W_spher"] = [W_spher, np.nan]
            aov["p_spher"] = [p_spher, np.nan]
            aov["sphericity"] = [spher, np.nan]

        col_order = [
            "Source",
            "SS",
            "DF",
            "MS",
            "F",
            "p_unc",
            "p_GG_corr",
            effsize,
            "eps",
            "sphericity",
            "W_spher",
            "p_spher",
        ]

    aov = aov.reindex(columns=col_order)
    aov.dropna(how="all", axis=1, inplace=True)
    return _postprocess_dataframe(aov)


def rm_anova2(data=None, dv=None, within=None, subject=None, effsize="ng2"):
    """Two-way repeated measures ANOVA.

    This is an internal function. The main call to this function should be done
    by the :py:func:`pingouin.rm_anova` function.
    """
    a, b = within

    # Validate the dataframe
    data = _check_dataframe(dv=dv, within=within, data=data, subject=subject, effects="within")

    assert not data[a].isnull().any(), "Cannot have missing values in %s" % a
    assert not data[b].isnull().any(), "Cannot have missing values in %s" % b
    assert not data[subject].isnull().any(), "Cannot have missing values in %s" % subject

    # Wide-format table, and complete-case long-format data
    data_piv, data = _pivot_rm(data, dv, within, subject)

    # Group sizes and grandmean
    n_a = data[a].nunique()
    n_b = data[b].nunique()
    n_s = data[subject].nunique()
    mu = data[dv].mean(numeric_only=True)

    # Groupby means
    # I think that observed=True is actually not needed here since we have already used
    # `observed=True` in pivot_table.
    grp_s = data.groupby(subject, observed=True)[dv].mean(numeric_only=True)
    grp_a = data.groupby([a], observed=True)[dv].mean(numeric_only=True)
    grp_b = data.groupby([b], observed=True)[dv].mean(numeric_only=True)
    grp_ab = data.groupby([a, b], observed=True)[dv].mean(numeric_only=True)
    grp_as = data.groupby([a, subject], observed=True)[dv].mean(numeric_only=True)
    grp_bs = data.groupby([b, subject], observed=True)[dv].mean(numeric_only=True)

    # Sums of squares
    ss_tot = np.sum((data[dv] - mu) ** 2)
    ss_s = (n_a * n_b) * np.sum((grp_s - mu) ** 2)
    ss_a = (n_b * n_s) * np.sum((grp_a - mu) ** 2)
    ss_b = (n_a * n_s) * np.sum((grp_b - mu) ** 2)
    ss_ab_er = n_s * np.sum((grp_ab - mu) ** 2)
    ss_ab = ss_ab_er - ss_a - ss_b
    ss_as_er = n_b * np.sum((grp_as - mu) ** 2)
    ss_as = ss_as_er - ss_s - ss_a
    ss_bs_er = n_a * np.sum((grp_bs - mu) ** 2)
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

    # Effects of A, B and A * B, and their respective error terms
    ss = np.array([ss_a, ss_b, ss_ab])
    ddof1 = np.array([df_a, df_b, df_ab])
    ss_err = np.array([ss_as, ss_bs, ss_abs])
    ddof2 = np.array([df_as, df_bs, df_abs])

    # Mean squares, F-values and p-values
    ms = ss / ddof1
    fval = ms / (ss_err / ddof2)
    p_unc = f(ddof1, ddof2).sf(fval)

    # Effect sizes
    if effsize == "n2":
        # ..Eta-squared
        ef = ss / ss_tot
    elif effsize == "ng2":
        # .. Generalized eta-squared (from Bakeman 2005 Table 1) -- default
        ef = ss / (ss + ss_s + ss_as + ss_bs + ss_abs)
    else:
        # .. Partial eta squared
        ef = ss / (ss + ss_err)

    # Epsilon and Mauchly's test of sphericity of each effect, from the covariance matrix of the
    # orthonormal contrasts. The main effects use the subject means across the other factor.
    # Same as R / afex, Mauchly's test uses the total number of conditions (n_a * n_b) in the
    # chi-square approximation, including for the main effects.
    eps, spher = [], []
    for fac in [a, b, None]:
        piv = data_piv if fac is None else data_piv.T.groupby(level=fac, observed=True).mean().T
        M = _contrast_cov(piv)
        eps.append(_gg_epsilon(M))
        spher.append(_mauchly_sphericity(M, n_s - 1, n_a * n_b))
    eps = np.array(eps)

    # Create dataframe
    aov = pd.DataFrame(
        {
            "Source": [a, b, a + " * " + b],
            "SS": ss,
            "ddof1": ddof1,
            "ddof2": ddof2,
            "MS": ms,
            "F": fval,
            "p_unc": p_unc,
            "p_GG_corr": _gg_corrected_pval(fval, ddof1, ddof2, eps),
            effsize: ef,
            "eps": eps,
            "sphericity": [sph[0] for sph in spher],
            "W_spher": [sph[1] for sph in spher],
            "p_spher": [sph[2] for sph in spher],
        }
    )
    return _postprocess_dataframe(aov)


@pf.register_dataframe_method
def anova(data=None, dv=None, between=None, ss_type=2, detailed=False, effsize="np2"):
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
    effsize : str
        Effect size. Must be 'np2' (partial eta-squared) or 'n2'
        (eta-squared). Note that for one-way ANOVA partial eta-squared is the
        same as eta-squared.

    Returns
    -------
    aov : :py:class:`pandas.DataFrame`
        ANOVA summary:

        * ``'Source'``: Factor names
        * ``'SS'``: Sums of squares
        * ``'DF'``: Degrees of freedom
        * ``'MS'``: Mean squares
        * ``'F'``: F-values
        * ``'p_unc'``: uncorrected p-values
        * ``'np2'``: Partial eta-square effect sizes

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

    .. math::
        SS_{\\text{total}} = SS_{\\text{effect}} + SS_{\\text{error}}

        SS_{\\text{total}} = \\sum_i \\sum_j (Y_{ij} - \\overline{Y})^2

        SS_{\\text{effect}} = \\sum_i n_i (\\overline{Y_i} - \\overline{Y})^2

        SS_{\\text{error}} = \\sum_i \\sum_j (Y_{ij} - \\overline{Y}_i)^2

    where :math:`i=1,...,r; j=1,...,n_i`, :math:`r` is the number of groups,
    and :math:`n_i` the number of observations for the :math:`i` th group.

    The F-statistics is then defined as:

    .. math::

        F^* = \\frac{MS_{\\text{effect}}}{MS_{\\text{error}}} =
        \\frac{SS_{\\text{effect}} / (r - 1)}{SS_{\\text{error}} / (n_t - r)}

    and the p-value can be calculated using a F-distribution with
    :math:`r-1, n_t-1` degrees of freedom.

    When the groups are balanced and have equal variances, the optimal post-hoc
    test is the Tukey-HSD test (:py:func:`pingouin.pairwise_tukey`).
    If the groups have unequal variances, the Games-Howell test is more
    adequate (:py:func:`pingouin.pairwise_gameshowell`).

    The default effect size reported in Pingouin is the partial eta-square,
    which, for one-way ANOVA is the same as eta-square and generalized
    eta-square.

    .. math::
        \\eta_p^2 = \\frac{SS_{\\text{effect}}}{SS_{\\text{effect}} +
        SS_{\\text{error}}}

    Missing values are automatically removed. Results have been tested against
    R, Matlab and JASP.

    Examples
    --------
    One-way ANOVA

    >>> import pingouin as pg
    >>> df = pg.read_dataset("anova")
    >>> aov = pg.anova(dv="Pain threshold", between="Hair color", data=df, detailed=True)
    >>> aov.round(3)
           Source        SS  DF       MS      F  p_unc    np2
    0  Hair color  1360.726   3  453.575  6.791  0.004  0.576
    1      Within  1001.800  15   66.787    NaN    NaN    NaN

    Same but using a standard eta-squared instead of a partial eta-squared
    effect size. Also note how here we're using the anova function directly as
    a method (= built-in function) of our pandas dataframe. In that case,
    we don't have to specify ``data`` anymore.

    >>> df.anova(dv="Pain threshold", between="Hair color", detailed=False, effsize="n2")
           Source  ddof1  ddof2         F     p_unc        n2
    0  Hair color      3     15  6.791407  0.004114  0.575962

    Two-way ANOVA with balanced design

    >>> data = pg.read_dataset("anova2")
    >>> data.anova(dv="Yield", between=["Blend", "Crop"]).round(3)
             Source        SS  DF        MS      F  p_unc    np2
    0         Blend     2.042   1     2.042  0.004  0.952  0.000
    1          Crop  2736.583   2  1368.292  2.525  0.108  0.219
    2  Blend * Crop  2360.083   2  1180.042  2.178  0.142  0.195
    3      Residual  9753.250  18   541.847    NaN    NaN    NaN

    Two-way ANOVA with unbalanced design (requires statsmodels)

    >>> data = pg.read_dataset("anova2_unbalanced")
    >>> data.anova(dv="Scores", between=["Diet", "Exercise"], effsize="n2").round(3)
                Source       SS   DF       MS      F  p_unc     n2
    0             Diet  390.625  1.0  390.625  7.423  0.034  0.433
    1         Exercise  180.625  1.0  180.625  3.432  0.113  0.200
    2  Diet * Exercise   15.625  1.0   15.625  0.297  0.605  0.017
    3         Residual  315.750  6.0   52.625    NaN    NaN    NaN

    Three-way ANOVA, type 3 sums of squares (requires statsmodels)

    >>> data = pg.read_dataset("anova3")
    >>> data.anova(dv="Cholesterol", between=["Sex", "Risk", "Drug"], ss_type=3).round(3)
                  Source      SS    DF      MS       F  p_unc    np2
    0                Sex   2.075   1.0   2.075   2.462  0.123  0.049
    1               Risk  11.332   1.0  11.332  13.449  0.001  0.219
    2               Drug   0.816   2.0   0.408   0.484  0.619  0.020
    3         Sex * Risk   0.117   1.0   0.117   0.139  0.711  0.003
    4         Sex * Drug   2.564   2.0   1.282   1.522  0.229  0.060
    5        Risk * Drug   2.438   2.0   1.219   1.446  0.245  0.057
    6  Sex * Risk * Drug   1.844   2.0   0.922   1.094  0.343  0.044
    7           Residual  40.445  48.0   0.843     NaN    NaN    NaN
    """
    assert effsize in ["np2", "n2"], "effsize must be 'np2' or 'n2'."
    if isinstance(between, list):
        if len(between) == 0:
            raise ValueError("between is empty.")
        elif len(between) == 1:
            between = between[0]
        elif len(between) == 2:
            # Two factors with balanced design = Pingouin implementation
            # Two factors with unbalanced design = statsmodels
            return anova2(dv=dv, between=between, data=data, ss_type=ss_type, effsize=effsize)
        else:
            # 3 or more factors with (un)-balanced design = statsmodels
            return anovan(dv=dv, between=between, data=data, ss_type=ss_type, effsize=effsize)

    # Check data
    data = _check_dataframe(dv=dv, between=between, data=data, effects="between")

    # Drop missing values
    data = data[[dv, between]].dropna()
    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)
    n_groups = data[between].nunique()
    N = data[dv].size

    # Calculate sums of squares
    grp = data.groupby(between, observed=True, group_keys=False)[dv]
    # Between effect
    ssbetween = (
        (grp.mean(numeric_only=True) - data[dv].mean(numeric_only=True)) ** 2 * grp.count()
    ).sum()
    # Within effect (= error between)
    #  = (grp.var(ddof=0) * grp.count()).sum()
    sserror = grp.transform(lambda x: (x - x.mean()) ** 2).sum()
    # In 1-way ANOVA, sstotal = ssbetween + sserror
    # sstotal = ssbetween + sserror

    # Calculate DOF, MS, F and p-values
    ddof1 = n_groups - 1
    ddof2 = N - n_groups
    msbetween = ssbetween / ddof1
    mserror = sserror / ddof2
    fval = msbetween / mserror
    p_unc = f(ddof1, ddof2).sf(fval)

    # Calculating effect sizes (see Bakeman 2005; Lakens 2013)
    # In one-way ANOVA, partial eta2 = eta2 = generalized eta2
    # Similar to (fval * ddof1) / (fval * ddof1 + ddof2)
    np2 = ssbetween / (ssbetween + sserror)  # = ssbetween / sstotal
    # Omega-squared
    # o2 = (ddof1 * (msbetween - mserror)) / (sstotal + mserror)

    # Create output dataframe
    if not detailed:
        aov = pd.DataFrame(
            {
                "Source": between,
                "ddof1": ddof1,
                "ddof2": ddof2,
                "F": fval,
                "p_unc": p_unc,
                effsize: np2,
            },
            index=[0],
        )

    else:
        aov = pd.DataFrame(
            {
                "Source": [between, "Within"],
                "SS": [ssbetween, sserror],
                "DF": [ddof1, ddof2],
                "MS": [msbetween, mserror],
                "F": [fval, np.nan],
                "p_unc": [p_unc, np.nan],
                effsize: [np2, np.nan],
            }
        )

    aov.dropna(how="all", axis=1, inplace=True)
    return _postprocess_dataframe(aov)


def anova2(data=None, dv=None, between=None, ss_type=2, effsize="np2"):
    """Two-way balanced ANOVA in pure Python + Pandas.

    This is an internal function. The main call to this function should be done
    by the :py:func:`pingouin.anova` function.
    """
    # Validate the dataframe
    data = _check_dataframe(dv=dv, between=between, data=data, effects="between")

    assert len(between) == 2, "Must have exactly two between-factors variables"
    fac1, fac2 = between

    # Drop missing values
    data = data[[dv, fac1, fac2]].dropna()
    assert data.shape[0] >= 5, "Data must have at least 5 non-missing values."

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)
    _check_no_empty_cells(data, between)
    grp_both = data.groupby(between, observed=True, group_keys=False)[dv]
    if grp_both.count().nunique() > 1:
        # UNBALANCED DESIGN
        return anovan(dv=dv, between=between, data=data, ss_type=ss_type, effsize=effsize)

    # BALANCED DESIGN
    mu = data[dv].mean()
    ss_tot = ((data[dv] - mu) ** 2).sum()
    ss_resid = ((data[dv] - grp_both.transform("mean")) ** 2).sum()
    df_resid = data[dv].size - grp_both.ngroups
    # Main effects of each factor, and interaction
    ss, ddof1 = [], []
    for fac in between:
        grp = data.groupby(fac, observed=True)[dv]
        ss.append(((grp.mean() - mu) ** 2 * grp.count()).sum())
        ddof1.append(grp.ngroups - 1)
    ss.append(ss_tot - (ss_resid + sum(ss)))
    ddof1.append(ddof1[0] * ddof1[1])
    ss, ddof1 = np.array(ss), np.array(ddof1)

    # Mean squares, F-values and p-values
    ms = ss / ddof1
    ms_resid = ss_resid / df_resid
    fval = ms / ms_resid
    pval = f(ddof1, df_resid).sf(fval)

    # Effect size
    if effsize == "n2":
        # Standard eta-square
        ef = ss / ss_tot
    else:
        # ..Partial eta-square
        ef = ss / (ss + ss_resid)

    # Create output dataframe
    aov = pd.DataFrame(
        {
            "Source": [fac1, fac2, fac1 + " * " + fac2, "Residual"],
            "SS": [*ss, ss_resid],
            "DF": [*ddof1, df_resid],
            "MS": [*ms, ms_resid],
            "F": [*fval, np.nan],
            "p_unc": [*pval, np.nan],
            effsize: [*ef, np.nan],
        }
    )

    aov.dropna(how="all", axis=1, inplace=True)
    return _postprocess_dataframe(aov)


def _check_no_empty_cells(data, between):
    """Raise an error if a combination of the between-subject factors has no observation.

    The interaction(s) cannot be estimated with empty cells: statsmodels would silently report
    too many degrees of freedom, and the balanced two-way ANOVA negative sums of squares.
    """
    n_cells = np.prod([data[fac].nunique() for fac in between])
    if data.groupby(between, observed=True).ngroups < n_cells:
        raise ValueError(
            "Each combination of the between-subject factors must have at least one "
            "observation. The interaction cannot be estimated with empty cells."
        )


def _eta_squared(ss, effsize):
    """Eta-squared (``"n2"``) or partial eta-squared (``"np2"``) of each row of an ANOVA table.

    ``ss`` are the sums of squares of the table, the last row being the residuals.
    """
    ss = np.asarray(ss, dtype=float)
    ef = ss / ss.sum() if effsize == "n2" else ss / (ss + ss[-1])
    ef[-1] = np.nan
    return ef


def _remove_unused_categories(data):
    """Remove the unused levels of the categorical columns of ``data``.

    Statsmodels / patsy build one dummy column per category, including unused ones (e.g. after
    dropping missing values), which makes the design matrix rank deficient and silently
    corrupts the sums of squares.
    """
    return data.apply(
        lambda s: (
            s.cat.remove_unused_categories() if isinstance(s.dtype, pd.CategoricalDtype) else s
        )
    )


def anovan(data=None, dv=None, between=None, ss_type=2, effsize="np2"):
    """N-way ANOVA using statsmodels.

    This is an internal function. The main call to this function should be done
    by the :py:func:`pingouin.anova` function.
    """
    from statsmodels.api import stats
    from statsmodels.formula.api import ols

    # Validate the dataframe
    data = _check_dataframe(dv=dv, between=between, data=data, effects="between")
    all_cols = _flatten_list([dv, between])

    # Drop missing values
    data = _remove_unused_categories(data[all_cols].dropna())
    assert data.shape[0] >= 5, "Data must have at least 5 non-missing values."
    _check_no_empty_cells(data, between)

    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)

    # Create R-like formula: dv ~ C(factor0, Sum) * C(factor1, Sum) * ...
    # https://patsy.readthedocs.io/en/latest/builtins-reference.html
    # The formula is evaluated as Python code by patsy, so it must never contain the column names
    # (which can have any character, e.g. quotes). The columns are renamed to placeholders instead.
    terms = {f"C(factor{i}, Sum)": fac for i, fac in enumerate(between)}
    model_data = pd.DataFrame(
        {"dv": data[dv]} | {f"factor{i}": data[fac] for i, fac in enumerate(between)}
    )
    formula = "dv ~ " + " * ".join(terms)

    # Fit using statsmodels
    lm = ols(formula, data=model_data).fit()
    aov = stats.anova_lm(lm, typ=ss_type)

    # Convert to Pingouin-like dataframe
    if ss_type == 1:
        # statsmodels output is not exactly the same when ss_type = 1
        aov = aov[["sum_sq", "df", "F", "PR(>F)"]]
    if ss_type == 3:
        # Remove intercept row
        aov = aov.iloc[1:, :]

    aov = aov.reset_index()
    aov = aov.rename(columns={"index": "Source", "sum_sq": "SS", "df": "DF", "PR(>F)": "p_unc"})
    aov["MS"] = aov["SS"] / aov["DF"]

    # Effect size
    aov[effsize] = _eta_squared(aov["SS"], effsize)

    # Map the model terms back to the column names, e.g. "C(factor0, Sum):C(factor1, Sum)" -> "A * B"
    aov["Source"] = [
        " * ".join(str(terms.get(term, term)) for term in source.split(":"))
        for source in aov["Source"]
    ]

    # Re-index and round
    col_order = ["Source", "SS", "DF", "MS", "F", "p_unc", effsize]
    aov = aov.reindex(columns=col_order)
    aov.dropna(how="all", axis=1, inplace=True)

    # Add formula to dataframe, with the original column names (for display only, it is never
    # evaluated)
    aov = _postprocess_dataframe(aov)
    aov.formula_ = f"Q('{dv}') ~ " + " * ".join(f"C(Q('{fac}'), Sum)" for fac in between)
    return aov


@pf.register_dataframe_method
def welch_anova(data=None, dv=None, between=None):
    """One-way Welch ANOVA.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    dv : string
        Name of column containing the dependent variable.
    between : string
        Name of column containing the between factor.

    Returns
    -------
    aov : :py:class:`pandas.DataFrame`
        ANOVA summary:

        * ``'Source'``: Factor names
        * ``'ddof1'``: Numerator degrees of freedom
        * ``'ddof2'``: Denominator degrees of freedom
        * ``'F'``: F-values
        * ``'p_unc'``: uncorrected p-values
        * ``'np2'``: Partial eta-squared

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
    normality and independence remain.

    The main idea of Welch ANOVA is to use a weight :math:`w_i` to reduce
    the effect of unequal variances. This weight is calculated using the sample
    size :math:`n_i` and variance :math:`s_i^2` of each group
    :math:`i=1,...,r`:

    .. math:: w_i = \\frac{n_i}{s_i^2}

    Using these weights, the adjusted grand mean of the data is:

    .. math::

        \\overline{Y}_{\\text{welch}} = \\frac{\\sum_{i=1}^r
        w_i\\overline{Y}_i}{\\sum w}

    where :math:`\\overline{Y}_i` is the mean of the :math:`i` group.

    The effect sums of squares is defined as:

    .. math::

        SS_{\\text{effect}} = \\sum_{i=1}^r w_i
        (\\overline{Y}_i - \\overline{Y}_{\\text{welch}})^2

    We then need to calculate a term lambda:

    .. math::

        \\Lambda = \\frac{3\\sum_{i=1}^r(\\frac{1}{n_i-1})
        (1 - \\frac{w_i}{\\sum w})^2}{r^2 - 1}

    from which the F-value can be calculated:

    .. math::

        F_{\\text{welch}} = \\frac{SS_{\\text{effect}} / (r-1)}
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
    >>> df = read_dataset("anova")
    >>> aov = welch_anova(dv="Pain threshold", between="Hair color", data=df)
    >>> aov
           Source  ddof1     ddof2         F     p_unc       np2
    0  Hair color      3  8.329841  5.890115  0.018813  0.575962
    """
    # Check data
    data = _check_dataframe(dv=dv, between=between, data=data, effects="between")

    # Drop missing values
    data = data[[dv, between]].dropna()
    # Reset index (avoid duplicate axis error)
    data = data.reset_index(drop=True)

    # Number of groups
    r = data[between].nunique()
    ddof1 = r - 1

    # Compute weights and ajusted means
    grp = data.groupby(between, observed=True, group_keys=False)[dv]
    if (grp.count() < 2).any() or (grp.var() == 0).any():
        raise ValueError(
            "Each group must have at least two observations and a non-zero variance. "
            "The Welch ANOVA weights are undefined otherwise."
        )
    weights = grp.count() / grp.var(numeric_only=True)
    adj_grandmean = (weights * grp.mean(numeric_only=True)).sum() / weights.sum()

    # Sums of squares (regular and adjusted)
    ss_res = grp.apply(lambda x: (x - x.mean()) ** 2).sum()
    ss_bet = (
        (grp.mean(numeric_only=True) - data[dv].mean(numeric_only=True)) ** 2 * grp.count()
    ).sum()
    ss_betadj = np.sum(weights * np.square(grp.mean(numeric_only=True) - adj_grandmean))
    ms_betadj = ss_betadj / ddof1

    # Calculate lambda, F-value, p-value and np2
    lamb = (3 * np.sum((1 / (grp.count() - 1)) * (1 - (weights / weights.sum())) ** 2)) / (r**2 - 1)
    ddof2 = 1 / lamb
    fval = ms_betadj / (1 + (2 * lamb * (r - 2)) / 3)
    pval = f.sf(fval, ddof1, ddof2)
    np2 = ss_bet / (ss_bet + ss_res)

    # Create output dataframe
    aov = pd.DataFrame(
        {
            "Source": between,
            "ddof1": ddof1,
            "ddof2": ddof2,
            "F": fval,
            "p_unc": pval,
            "np2": np2,
        },
        index=[0],
    )
    return _postprocess_dataframe(aov)


@pf.register_dataframe_method
def mixed_anova(
    data=None, dv=None, within=None, subject=None, between=None, correction="auto", effsize="np2"
):
    """Mixed-design (split-plot) ANOVA.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    dv : string
        Name of column containing the dependent variable.
    within : string
        Name of column containing the within-subject factor
        (repeated measurements).
    subject : string
        Name of column containing the between-subject identifier.
    between : string
        Name of column containing the between factor.
    correction : string or boolean
        If True, return Greenhouse-Geisser corrected p-values for the within factor and
        the interaction. If `'auto'` (default), compute Mauchly's test of sphericity to
        determine whether the p-values needs to be corrected.
    effsize : str
        Effect size. Must be one of 'np2' (partial eta-squared), 'n2'
        (eta-squared) or 'ng2'(generalized eta-squared).

    Returns
    -------
    aov : :py:class:`pandas.DataFrame`
        ANOVA summary:

        * ``'Source'``: Names of the factor considered
        * ``'ddof1'``: Degrees of freedom (numerator)
        * ``'ddof2'``: Degrees of freedom (denominator)
        * ``'F'``: F-values
        * ``'p_unc'``: Uncorrected p-values
        * ``'np2'``: Partial eta-squared effect sizes
        * ``'eps'``: Greenhouse-Geisser epsilon factor (= index of sphericity)
        * ``'p_GG_corr'``: Greenhouse-Geisser corrected p-values
        * ``'W_spher'``: Sphericity test statistic
        * ``'p_spher'``: p-value of the sphericity test
        * ``'sphericity'``: sphericity of the data (boolean)

    See Also
    --------
    anova, rm_anova, pairwise_tests

    Notes
    -----
    Data are expected to be in long-format (even the repeated measures).
    If your data is in wide-format, you can use the :py:func:`pandas.melt()`
    function to convert from wide to long format.

    The Greenhouse-Geisser epsilon and Mauchly's test of sphericity are computed from the pooled
    within-group covariance matrix of the repeated measurements, i.e. after removing the
    between-group differences in means. This is the same as SPSS, JASP and the afex / car R
    packages. The same epsilon is used to correct the p-values of both the within factor and the
    interaction.

    Missing values are automatically removed using a strict listwise approach (= complete-case
    analysis). In other words, any subject with one or more missing value(s) is completely removed
    from the dataframe prior to running the test. This could drastically decrease the power of the
    ANOVA if many missing values are present. In that case, we strongly recommend using linear
    mixed effect modelling, which can handle missing values in repeated measures.

    .. warning :: If the between-subject groups are unbalanced (= unequal sample sizes),
        a type II ANOVA will be computed. Note however that SPSS, JAMOVI and JASP by default
        return a type III ANOVA, which may lead to slightly different results.

    Examples
    --------
    For more examples, please refer to the `Jupyter notebooks
    <https://github.com/raphaelvallat/pingouin/blob/master/notebooks/01_ANOVA.ipynb>`_

    Compute a two-way mixed model ANOVA.

    >>> from pingouin import mixed_anova, read_dataset
    >>> df = read_dataset("mixed_anova")
    >>> aov = mixed_anova(dv="Scores", between="Group", within="Time", subject="Subject", data=df)
    >>> aov.round(3)
            Source     SS  DF1  DF2     MS      F  p_unc    np2    eps
    0        Group  5.460    1   58  5.460  5.052  0.028  0.080    NaN
    1         Time  7.628    2  116  3.814  4.027  0.020  0.065  0.998
    2  Interaction  5.167    2  116  2.584  2.728  0.070  0.045  0.998

    Same but reporting a generalized eta-squared effect size. Notice how we
    can also apply this function directly as a method of the dataframe, in
    which case we do not need to specify ``data=df`` anymore.

    >>> df.mixed_anova(
    ...     dv="Scores", between="Group", within="Time", subject="Subject", effsize="ng2"
    ... ).round(3)
            Source     SS  DF1  DF2     MS      F  p_unc    ng2    eps
    0        Group  5.460    1   58  5.460  5.052  0.028  0.031    NaN
    1         Time  7.628    2  116  3.814  4.027  0.020  0.042  0.998
    2  Interaction  5.167    2  116  2.584  2.728  0.070  0.029  0.998
    """
    assert effsize in ["n2", "np2", "ng2"], "effsize must be n2, np2 or ng2."

    # Check that only a single within and between factor are provided
    one_is_list = isinstance(within, list) or isinstance(between, list)
    both_are_str = isinstance(within, (str, int)) and isinstance(between, (str, int))
    if one_is_list or not both_are_str:
        raise ValueError(
            "within and between factors must both be strings referring to a column in the data. "
            "Specifying multiple within and between factors is currently not supported. "
            "For more information, see: https://github.com/raphaelvallat/pingouin/issues/136"
        )

    # Check data
    data = _check_dataframe(
        dv=dv, within=within, between=between, data=data, subject=subject, effects="interaction"
    )

    # Wide-format table (subjects x within), and complete-case long-format data
    data_piv, data = _pivot_rm(data, dv, within, subject, between=between)

    # Check that subject IDs do not overlap between groups: the subject ID
    # should have a unique range / set of values for each between-subject
    # group e.g. group1= 1 --> 20 and group2 = 21 --> 40.
    if not (data.groupby([subject, within], observed=True)[between].nunique() == 1).all():
        raise ValueError(
            "Subject IDs cannot overlap between groups: each "
            "group in `%s` must have a unique set of "
            "subject IDs, e.g. group1 = [1, 2, 3, ..., 10] "
            "and group2 = [11, 12, 13, ..., 20]" % between
        )

    # SUMS OF SQUARES
    grandmean = data[dv].mean(numeric_only=True)
    ss_total = ((data[dv] - grandmean) ** 2).sum()
    grp_betw = data.groupby(between, observed=True)[dv]
    grp_with = data.groupby(within, observed=True)[dv]
    ss_betw = ((grp_betw.mean() - grandmean) ** 2 * grp_betw.count()).sum()
    ss_with = ((grp_with.mean() - grandmean) ** 2 * grp_with.count()).sum()
    # ssresall = residuals within + residuals between
    grp = data.groupby([between, within], observed=True)[dv]
    ss_resall = ((data[dv] - grp.transform("mean")) ** 2).sum()
    # Interaction
    ss_inter = ss_total - (ss_resall + ss_with + ss_betw)
    # Variability between subjects, which is split into the between-subject factor and its error
    n_rm = data_piv.shape[1]
    ss_subj = n_rm * ((data_piv.mean(axis=1) - grandmean) ** 2).sum()
    ss_resbetw = ss_subj - ss_betw
    ss_reswith = ss_total - (ss_with + ss_subj + ss_inter)

    # DEGREES OF FREEDOM
    n_subj, n_groups = data_piv.shape[0], grp_betw.ngroups
    df_betw = n_groups - 1
    df_with = n_rm - 1
    df_inter = df_with * df_betw
    df_resbetw = n_subj - n_groups
    df_reswith = df_with * df_resbetw

    # Between, within and interaction effects, and their respective error terms
    ss = np.array([ss_betw, ss_with, ss_inter])
    ddof1 = np.array([df_betw, df_with, df_inter])
    ss_err = np.array([ss_resbetw, ss_reswith, ss_reswith])
    ddof2 = np.array([df_resbetw, df_reswith, df_reswith])

    # MEAN SQUARES, F-VALUES AND P-VALUES
    ms = ss / ddof1
    fval = ms / (ss_err / ddof2)
    p_unc = f(ddof1, ddof2).sf(fval)

    # SPHERICITY
    # Epsilon and Mauchly's test are computed from the pooled within-group covariance matrix,
    # i.e. after removing the between-group mean differences at each level of the within factor
    # (same as SPSS, R car::Anova / afex / ez). Using the total covariance instead would treat
    # group differences as departure from sphericity.
    resid = data_piv - data_piv.groupby(level=between, observed=True).transform("mean")
    M = _contrast_cov(resid)
    # GG epsilon is invariant to the scaling of the covariance matrix (N - 1 vs N - n_groups)
    eps = _gg_epsilon(M)
    spher, W_spher, p_spher = _mauchly_sphericity(M, df_resbetw, n_rm)
    # Sphericity is always met with only two repeated measures
    correction = not spher if correction == "auto" else (bool(correction) and n_rm >= 3)

    # Effects sizes (see Bakeman 2005)
    if effsize == "n2":
        # Standard eta-squared
        ef = ss / ss_total
    elif effsize == "ng2":
        # Generalized eta-square
        ef = ss / (ss + ss_resall)
    else:
        # Partial eta-squared (default)
        ef = ss / (ss + ss_err)

    # Stats table
    aov = pd.DataFrame(
        {
            "Source": [between, within, "Interaction"],
            "SS": ss,
            "DF1": ddof1,
            "DF2": ddof2,
            "MS": ms,
            "F": fval,
            "p_unc": p_unc,
            effsize: ef,
            "eps": [np.nan, eps, eps],
        }
    )
    if correction:
        # Same epsilon for the within factor and the interaction
        aov["p_GG_corr"] = [np.nan, *_gg_corrected_pval(fval[1:], ddof1[1:], ddof2[1:], eps)]
        aov["W_spher"] = [np.nan, W_spher, W_spher]
        aov["p_spher"] = [np.nan, p_spher, p_spher]
        aov["sphericity"] = [np.nan, spher, spher]
    col_order = [
        "Source",
        "SS",
        "DF1",
        "DF2",
        "MS",
        "F",
        "p_unc",
        "p_GG_corr",
        effsize,
        "eps",
        "sphericity",
        "W_spher",
        "p_spher",
    ]
    aov = aov.reindex(columns=col_order)
    aov.dropna(how="all", axis=1, inplace=True)
    return _postprocess_dataframe(aov)


@pf.register_dataframe_method
def ancova(data=None, dv=None, between=None, covar=None, effsize="np2"):
    """ANCOVA with one or more covariate(s).

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    dv : string
        Name of column in data with the dependent variable.
    between : string
        Name of column in data with the between factor.
    covar : string or list
        Name(s) of column(s) in data with the covariate.
    effsize : str
        Effect size. Must be 'np2' (partial eta-squared) or 'n2'
        (eta-squared).

    Returns
    -------
    aov : :py:class:`pandas.DataFrame`
        ANCOVA summary:

        * ``'Source'``: Names of the factor considered
        * ``'SS'``: Sums of squares
        * ``'DF'``: Degrees of freedom
        * ``'F'``: F-values
        * ``'p_unc'``: Uncorrected p-values
        * ``'np2'``: Partial eta-squared

    Notes
    -----
    Analysis of covariance (ANCOVA) is a general linear model which blends
    ANOVA and regression. ANCOVA evaluates whether the means of a dependent
    variable (dv) are equal across levels of a categorical independent
    variable (between) often called a treatment, while statistically
    controlling for the effects of other continuous variables that are not
    of primary interest, known as covariates or nuisance variables (covar).

    Pingouin uses :py:class:`statsmodels.regression.linear_model.OLS` to
    compute the ANCOVA.

    .. important:: Rows with missing values are automatically removed
        (listwise deletion).

    See Also
    --------
    anova : One-way and N-way ANOVA

    Examples
    --------
    1. Evaluate the reading scores of students with different teaching method
    and family income as a covariate.

    >>> from pingouin import ancova, read_dataset
    >>> df = read_dataset("ancova")
    >>> ancova(data=df, dv="Scores", covar="Income", between="Method")
         Source           SS  DF          F     p_unc       np2
    0    Method   571.029883   3   3.336482  0.031940  0.244077
    1    Income  1678.352687   1  29.419438  0.000006  0.486920
    2  Residual  1768.522313  31        NaN       NaN       NaN

    2. Evaluate the reading scores of students with different teaching method
    and family income + BMI as a covariate.

    >>> ancova(data=df, dv="Scores", covar=["Income", "BMI"], between="Method", effsize="n2")
         Source           SS  DF          F     p_unc        n2
    0    Method   552.284043   3   3.232550  0.036113  0.141802
    1    Income  1573.952434   1  27.637304  0.000011  0.404121
    2       BMI    60.013656   1   1.053790  0.312842  0.015409
    3  Residual  1708.508657  30        NaN       NaN       NaN
    """
    from statsmodels.api import stats
    from statsmodels.formula.api import ols

    # Safety checks
    assert effsize in ["np2", "n2"], "effsize must be 'np2' or 'n2'."
    assert isinstance(data, pd.DataFrame), "data must be a pandas dataframe."
    assert isinstance(between, str), (
        "between must be a string. Pingouin does not support multiple "
        "between factors. For more details, please see "
        "https://github.com/raphaelvallat/pingouin/issues/173."
    )
    assert dv in data.columns, "%s is not in data." % dv
    assert between in data.columns, "%s is not in data." % between
    assert isinstance(covar, (str, list)), "covar must be a str or a list."
    if isinstance(covar, str):
        covar = [covar]
    for c in covar:
        assert c in data.columns, "covariate %s is not in data" % c
        assert data[c].dtype.kind in "bfi", "covariate %s is not numeric" % c

    # Drop missing values
    data = _remove_unused_categories(data[_flatten_list([dv, between, covar])].dropna())

    # Fit ANCOVA model: dv ~ 1 + between + covar1 + covar2 + ...
    # The formula is evaluated as Python code by patsy, so it must never contain the column names
    # (which can have any character, e.g. quotes). The columns are renamed to placeholders instead.
    covar_names = [f"covar{i}" for i in range(len(covar))]
    model_data = pd.DataFrame(
        {"dv": data[dv], "between": data[between]}
        | {name: data[c] for name, c in zip(covar_names, covar)}
    )
    formula = " + ".join(["dv ~ C(between)", *covar_names])
    model = ols(formula, data=model_data).fit()

    # Create output dataframe
    aov = stats.anova_lm(model, typ=2).reset_index()
    aov.rename(
        columns={"index": "Source", "sum_sq": "SS", "df": "DF", "PR(>F)": "p_unc"}, inplace=True
    )
    aov.at[0, "Source"] = between
    for i in range(len(covar)):
        aov.at[i + 1, "Source"] = covar[i]
    aov["DF"] = aov["DF"].astype(int)

    # Add effect sizes
    aov[effsize] = _eta_squared(aov["SS"], effsize)

    # Add bw as an attribute (for rm_corr function)
    aov = _postprocess_dataframe(aov)
    aov.bw_ = model.params.iloc[-1]
    return aov
