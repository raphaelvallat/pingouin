# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import warnings
import numpy as np
from scipy.stats import pearsonr
from pingouin.utils import _check_eftype, remove_na

# from pingouin.distribution import homoscedasticity


__all__ = [
    "compute_esci",
    "compute_bootci",
    "convert_effsize",
    "compute_effsize",
    "compute_effsize_from_t",
]


def compute_esci(
    stat=None,
    nx=None,
    ny=None,
    paired=False,
    eftype="cohen",
    confidence=0.95,
    decimals=2,
    alternative="two-sided",
):
    """Parametric confidence intervals around a Cohen d or a correlation coefficient.

    Parameters
    ----------
    stat : float
        Original effect size. Must be either a correlation coefficient or a Cohen-type effect size
        (Cohen d or Hedges g).
    nx, ny : int
        Length of vector x and y.
    paired : bool
        Indicates if the effect size was estimated from a paired sample. This is only relevant for
        cohen or hedges effect size.
    eftype : string
        Effect size type. Must be "r" (correlation) or "cohen" (Cohen d or Hedges g).
    confidence : float
        Confidence level (0.95 = 95%)
    decimals : int
        Number of rounded decimals.
    alternative : string
        Defines the alternative hypothesis, or tail for the correlation coefficient. Must be one of
        "two-sided" (default), "greater" or "less". This parameter only has an effect if ``eftype``
        is "r".

    Returns
    -------
    ci : array
        Desired converted effect size

    Notes
    -----
    To compute the parametric confidence interval around a **Pearson r correlation** coefficient,
    one must first apply a Fisher's r-to-z transformation:

    .. math:: z = 0.5 \\cdot \\ln \\frac{1 + r}{1 - r} = \\text{arctanh}(r)

    and compute the standard error:

    .. math:: \\text{SE} = \\frac{1}{\\sqrt{n - 3}}

    where :math:`n` is the sample size.

    The lower and upper confidence intervals - *in z-space* - are then given by:

    .. math:: \\text{ci}_z = z \\pm \\text{crit} \\cdot \\text{SE}

    where :math:`\\text{crit}` is the critical value of the normal distribution corresponding to
    the desired confidence level (e.g. 1.96 in case of a 95% confidence interval).

    These confidence intervals can then be easily converted back to *r-space*:

    .. math::

        \\text{ci}_r = \\frac{\\exp(2 \\cdot \\text{ci}_z) - 1}
        {\\exp(2 \\cdot \\text{ci}_z) + 1} = \\text{tanh}(\\text{ci}_z)

    A formula for calculating the confidence interval for a **Cohen d effect size** is given by
    Hedges and Olkin (1985, p86). If the effect size estimate from the sample is :math:`d`, then
    it follows a T distribution with standard error:

    .. math::

        \\text{SE} = \\sqrt{\\frac{n_x + n_y}{n_x \\cdot n_y} +
        \\frac{d^2}{2 (n_x + n_y)}}

    where :math:`n_x` and :math:`n_y` are the sample sizes of the two groups.

    In one-sample test or paired test, this becomes:

    .. math::

        \\text{SE} = \\sqrt{\\frac{1}{n_x} + \\frac{d^2}{2 n_x}}

    The lower and upper confidence intervals are then given by:

    .. math:: \\text{ci}_d = d \\pm \\text{crit} \\cdot \\text{SE}

    where :math:`\\text{crit}` is the critical value of the T distribution corresponding to the
    desired confidence level.

    References
    ----------
    * https://en.wikipedia.org/wiki/Fisher_transformation

    * Hedges, L., and Ingram Olkin. "Statistical models for meta-analysis." (1985).

    * http://www.leeds.ac.uk/educol/documents/00002182.htm

    * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5133225/

    Examples
    --------
    1. Confidence interval of a Pearson correlation coefficient

    >>> import pingouin as pg
    >>> x = [3, 4, 6, 7, 5, 6, 7, 3, 5, 4, 2]
    >>> y = [4, 6, 6, 7, 6, 5, 5, 2, 3, 4, 1]
    >>> nx, ny = len(x), len(y)
    >>> stat = pg.compute_effsize(x, y, eftype='r')
    >>> ci = pg.compute_esci(stat=stat, nx=nx, ny=ny, eftype='r')
    >>> print(round(stat, 4), ci)
    0.7468 [0.27 0.93]

    2. Confidence interval of a Cohen d

    >>> stat = pg.compute_effsize(x, y, eftype='cohen')
    >>> ci = pg.compute_esci(stat, nx=nx, ny=ny, eftype='cohen', decimals=3)
    >>> print(round(stat, 4), ci)
    0.1538 [-0.737  1.045]
    """
    from scipy.stats import norm, t

    assert eftype.lower() in ["r", "pearson", "spearman", "cohen", "d", "g", "hedges"]
    assert alternative in [
        "two-sided",
        "greater",
        "less",
    ], "Alternative must be one of 'two-sided' (default), 'greater' or 'less'."
    assert stat is not None and nx is not None
    assert isinstance(confidence, float)
    assert 0 < confidence < 1, "confidence must be between 0 and 1."

    if eftype.lower() in ["r", "pearson", "spearman"]:
        z = np.arctanh(stat)  # R-to-z transform
        se = 1 / np.sqrt(nx - 3)
        # See https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/cor.test.R
        if alternative == "two-sided":
            crit = np.abs(norm.ppf((1 - confidence) / 2))
            ci_z = np.array([z - crit * se, z + crit * se])
        elif alternative == "greater":
            crit = norm.ppf(confidence)
            ci_z = np.array([z - crit * se, np.inf])
        else:  # alternative = "less"
            crit = norm.ppf(confidence)
            ci_z = np.array([-np.inf, z + crit * se])
        ci = np.tanh(ci_z)  # Transform back to r
    else:
        # Cohen d. Results are different than JASP which uses a non-central T
        # distribution. See github.com/jasp-stats/jasp-issues/issues/525
        if ny == 1 or paired:
            # One-sample or paired. Results vary slightly from the cohen.d R
            # function which uses instead:
            # >>> sqrt((n / (n / 2)^2) + .5*(dd^2 / n)) -- one-sample
            # >>> sqrt( (1/n1 + dd^2/(2*n1))*(2-2*r)); -- paired
            # where r is the correlation between samples
            # https://github.com/mtorchiano/effsize/blob/master/R/CohenD.R
            # However, Pingouin uses the formulas on www.real-statistics.com
            se = np.sqrt(1 / nx + stat**2 / (2 * nx))
            dof = nx - 1
        else:
            # Independent two-samples: give same results as R:
            # >>> cohen.d(..., paired = FALSE, noncentral=FALSE)
            se = np.sqrt(((nx + ny) / (nx * ny)) + (stat**2) / (2 * (nx + ny)))
            dof = nx + ny - 2
        crit = np.abs(t.ppf((1 - confidence) / 2, dof))
        ci = np.array([stat - crit * se, stat + crit * se])
    return np.round(ci, decimals)


def compute_bootci(
    x,
    y=None,
    func=None,
    method="cper",
    paired=False,
    confidence=0.95,
    n_boot=2000,
    decimals=2,
    seed=None,
    return_dist=False,
):
    """Bootstrapped confidence intervals of univariate and bivariate functions.

    Parameters
    ----------
    x : 1D-array or list
        First sample. Required for both bivariate and univariate functions.
    y : 1D-array, list, or None
        Second sample. Required only for bivariate functions.
    func : str or custom function
        Function to compute the bootstrapped statistic. Accepted string values are:

        * ``'pearson'``: Pearson correlation (bivariate, paired x and y)
        * ``'spearman'``: Spearman correlation (bivariate, paired x and y)
        * ``'cohen'``: Cohen d effect size (bivariate, paired or unpaired x and y)
        * ``'hedges'``: Hedges g effect size (bivariate, paired or unpaired x and y)
        * ``'mean'``: Mean (univariate = only x)
        * ``'std'``: Standard deviation (univariate)
        * ``'var'``: Variance (univariate)
    method : str
        Method to compute the confidence intervals (see Notes):

        * ``'cper'``: Bias-corrected percentile method (default)
        * ``'norm'``: Normal approximation with bootstrapped bias and standard error
        * ``'per'``: Simple percentile
    paired : boolean
        Indicates whether x and y are paired or not. For example, for correlation functions or
        paired T-test, x and y are assumed to be paired. Pingouin will resample the pairs
        (x_i, y_i) when paired=True, and resample x and y separately when paired=False.
        If paired=True, x and y must have the same number of elements.
    confidence : float
        Confidence level (0.95 = 95%)
    n_boot : int
        Number of bootstrap iterations. The higher, the better, the slower.
    decimals : int
        Number of rounded decimals.
    seed : int or None
        Random seed for generating bootstrap samples.
    return_dist : boolean
        If True, return the confidence intervals and the bootstrapped distribution (e.g. for
        plotting purposes).

    Returns
    -------
    ci : array
        Bootstrapped confidence intervals.

    Notes
    -----
    Results have been tested against the
    `bootci <https://www.mathworks.com/help/stats/bootci.html>`_ Matlab function.

    Since version 1.7, SciPy also includes a built-in bootstrap function
    :py:func:`scipy.stats.bootstrap`. The SciPy implementation has two advantages over Pingouin: it
    is faster when using ``vectorized=True``, and it supports the bias-corrected and accelerated
    (BCa) confidence intervals for univariate functions. However, unlike Pingouin, it does not
    return the bootstrap distribution.

    The percentile bootstrap method (``per``) is defined as the
    :math:`100 \\times \\frac{\\alpha}{2}` and :math:`100 \\times \\frac{1 - \\alpha}{2}`
    percentiles of the distribution of :math:`\\theta` estimates obtained from resampling, where
    :math:`\\alpha` is the level of significance (1 - confidence, default = 0.05 for 95% CIs).

    The bias-corrected percentile method (``cper``) corrects for bias of the bootstrap
    distribution. This method is different from the BCa method — the default in Matlab and SciPy —
    which corrects for both bias and skewness of the bootstrap distribution using jackknife
    resampling.

    The normal approximation method (``norm``) calculates the confidence intervals with the
    standard normal distribution using bootstrapped bias and standard error.

    References
    ----------
    * DiCiccio, T. J., & Efron, B. (1996). Bootstrap confidence intervals. Statistical science,
      189-212.

    * Davison, A. C., & Hinkley, D. V. (1997). Bootstrap methods and their application (Vol. 1).
      Cambridge university press.

    * Jung, Lee, Gupta, & Cho (2019). Comparison of bootstrap confidence interval methods for
      GSCA using a Monte Carlo simulation. Frontiers in psychology, 10, 2215.

    Examples
    --------
    1. Bootstrapped 95% confidence interval of a Pearson correlation

    >>> import pingouin as pg
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> x = rng.normal(loc=4, scale=2, size=100)
    >>> y = rng.normal(loc=3, scale=1, size=100)
    >>> stat = np.corrcoef(x, y)[0][1]
    >>> ci = pg.compute_bootci(x, y, func='pearson', paired=True, seed=42, decimals=4)
    >>> print(round(stat, 4), ci)
    0.0945 [-0.098   0.2738]

    Let's compare to SciPy's built-in bootstrap function

    >>> from scipy.stats import bootstrap
    >>> bt_scipy = bootstrap(
    ...       data=(x, y), statistic=lambda x, y: np.corrcoef(x, y)[0][1],
    ...       method="basic", vectorized=False, n_resamples=2000, paired=True, random_state=42)
    >>> np.round(bt_scipy.confidence_interval, 4)
    array([-0.0952,  0.2883])

    2. Bootstrapped 95% confidence interval of a Cohen d

    >>> stat = pg.compute_effsize(x, y, eftype='cohen')
    >>> ci = pg.compute_bootci(x, y, func='cohen', seed=42, decimals=3)
    >>> print(round(stat, 4), ci)
    0.7009 [0.403 1.009]

    3. Bootstrapped confidence interval of a standard deviation (univariate)

    >>> import numpy as np
    >>> stat = np.std(x, ddof=1)
    >>> ci = pg.compute_bootci(x, func='std', seed=123)
    >>> print(round(stat, 4), ci)
    1.5534 [1.38 1.8 ]

    Compare to SciPy's built-in bootstrap function, which returns the bias-corrected and
    accelerated CIs (see Notes).

    >>> def std(x, axis):
    ...     return np.std(x, ddof=1, axis=axis)
    >>> bt_scipy = bootstrap(data=(x, ), statistic=std, n_resamples=2000, random_state=123)
    >>> np.round(bt_scipy.confidence_interval, 2)
    array([1.39, 1.81])

    Changing the confidence intervals type in Pingouin

    >>> pg.compute_bootci(x, func='std', seed=123, method="norm")
    array([1.37, 1.76])

    >>> pg.compute_bootci(x, func='std', seed=123, method="percentile")
    array([1.35, 1.75])

    4. Bootstrapped confidence interval using a custom univariate function

    >>> from scipy.stats import skew
    >>> round(skew(x), 4), pg.compute_bootci(x, func=skew, n_boot=10000, seed=123)
    (-0.137, array([-0.55,  0.32]))

    5. Bootstrapped confidence interval using a custom bivariate function. Here, x and y are not
    paired and can therefore have different sizes.

    >>> def mean_diff(x, y):
    ...     return np.mean(x) - np.mean(y)
    >>> y2 = rng.normal(loc=3, scale=1, size=200)  # y2 has 200 samples, x has 100
    >>> ci = pg.compute_bootci(x, y2, func=mean_diff, n_boot=10000, seed=123)
    >>> print(round(mean_diff(x, y2), 2), ci)
    0.88 [0.54 1.21]

    We can also get the bootstrapped distribution

    >>> ci, bt = pg.compute_bootci(x, y2, func=mean_diff, n_boot=10000, return_dist=True, seed=9)
    >>> print(f"The bootstrap distribution has {bt.size} samples. The mean and standard "
    ...       f"{bt.mean():.4f} ± {bt.std():.4f}")
    The bootstrap distribution has 10000 samples. The mean and standard 0.8807 ± 0.1704
    """
    from inspect import isfunction, isroutine
    from scipy.stats import norm

    # Check other arguments
    assert isinstance(confidence, float)
    assert 0 < confidence < 1, "confidence must be between 0 and 1."
    assert method in ["norm", "normal", "percentile", "per", "cpercentile", "cper"]
    assert isfunction(func) or isinstance(func, str) or isroutine(func), (
        "func must be a function (e.g. np.mean, custom function) or a string (e.g. 'pearson'). "
        "See documentation for more details."
    )
    vectorizable = False

    # Check x
    x = np.asarray(x)
    nx = x.size
    assert x.ndim == 1, "x must be one-dimensional."
    assert nx > 1, "x must have more than one element."

    # Check y
    if y is not None:
        y = np.asarray(y)
        ny = y.size
        assert y.ndim == 1, "y must be one-dimensional."
        assert ny > 1, "y must have more than one element."
        if paired:
            assert nx == ny, "x and y must have the same number of elements when paired=True."

    # Check string functions
    if isinstance(func, str):
        func_str = "%s" % func
        if func == "pearson":
            assert paired, "Paired should be True if using correlation functions."

            def func(x, y):
                return pearsonr(x, y)[0]  # Faster than np.corrcoef

        elif func == "spearman":
            from scipy.stats import spearmanr

            assert paired, "Paired should be True if using correlation functions."

            def func(x, y):
                return spearmanr(x, y)[0]

        elif func in ["cohen", "hedges"]:
            from pingouin.effsize import compute_effsize

            def func(x, y):
                return compute_effsize(x, y, paired=paired, eftype=func_str)

        elif func == "mean":
            vectorizable = True

            def func(x):
                return np.mean(x, axis=0)

        elif func == "std":
            vectorizable = True

            def func(x):
                return np.std(x, ddof=1, axis=0)

        elif func == "var":
            vectorizable = True

            def func(x):
                return np.var(x, ddof=1, axis=0)

        else:
            raise ValueError("Function string not recognized.")

    # Bootstrap
    bootstat = np.empty(n_boot)
    rng = np.random.default_rng(seed)  # Random seed
    boot_x = rng.choice(np.arange(nx), size=(nx, n_boot), replace=True)

    if y is not None:
        reference = func(x, y)
        if paired:
            for i in range(n_boot):
                # Note that here we use a bootstrapping procedure with replacement
                # of all the pairs (Xi, Yi). This is NOT suited for
                # hypothesis testing such as p-value estimation). Instead, for the
                # latter, one must only shuffle the Y values while keeping the X
                # values constant, i.e.:
                # >>> boot_x = rng.random_sample((n_boot, n)).argsort(axis=1)
                # >>> for i in range(n_boot):
                # >>>   bootstat[i] = func(x, y[boot_x[i, :]])
                bootstat[i] = func(x[boot_x[:, i]], y[boot_x[:, i]])
        else:
            boot_y = rng.choice(np.arange(ny), size=(ny, n_boot), replace=True)
            for i in range(n_boot):
                bootstat[i] = func(x[boot_x[:, i]], y[boot_y[:, i]])
    else:
        reference = func(x)
        if vectorizable:
            bootstat = func(x[boot_x])
        else:
            for i in range(n_boot):
                bootstat[i] = func(x[boot_x[:, i]])

    # CONFIDENCE INTERVALS
    # See Matlab bootci function
    alpha = (1 - confidence) / 2
    if method in ["norm", "normal"]:
        # Normal approximation
        za = norm.ppf(alpha)  # = 1.96
        se = np.std(bootstat, ddof=1)
        bias = np.mean(bootstat - reference)
        ci = np.array([reference - bias + se * za, reference - bias - se * za])
    elif method in ["percentile", "per"]:
        # Simple percentile
        interval = 100 * np.array([alpha, 1 - alpha])
        ci = np.percentile(bootstat, interval)
        pass
    else:
        # Bias-corrected percentile bootstrap
        from pingouin.regression import _bias_corrected_ci

        ci = _bias_corrected_ci(bootstat, reference, alpha=(1 - confidence))

    ci = np.round(ci, decimals)

    if return_dist:
        return ci, bootstat
    else:
        return ci


def convert_effsize(ef, input_type, output_type, nx=None, ny=None):
    """Conversion between effect sizes.

    Parameters
    ----------
    ef : float
        Original effect size.
    input_type : string
        Effect size type of ef. Must be ``'cohen'`` or ``'pointbiserialr'``.
    output_type : string
        Desired effect size type. Available methods are:

        * ``'cohen'``: Unbiased Cohen d
        * ``'hedges'``: Hedges g
        * ``'pointbiserialr'``: Point-biserial correlation
        * ``'eta-square'``: Eta-square
        * ``'odds-ratio'``: Odds ratio
        * ``'AUC'``: Area Under the Curve
        * ``'none'``: pass-through (return ``ef``)

    nx, ny : int, optional
        Length of vector x and y. Required to convert to Hedges g.

    Returns
    -------
    ef : float
        Desired converted effect size

    See Also
    --------
    compute_effsize : Calculate effect size between two set of observations.
    compute_effsize_from_t : Convert a T-statistic to an effect size.

    Notes
    -----
    The formula to convert from a`point-biserial correlation
    <https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient>`_ **r** to **d** is
    given in [1]_:

    .. math:: d = \\frac{2r_{pb}}{\\sqrt{1 - r_{pb}^2}}

    The formula to convert **d** to a point-biserial correlation **r** is given in [2]_:

    .. math::

        r_{pb} = \\frac{d}{\\sqrt{d^2 + \\frac{(n_x + n_y)^2 - 2(n_x + n_y)}
        {n_xn_y}}}

    The formula to convert **d** to :math:`\\eta^2` is given in [3]_:

    .. math:: \\eta^2 = \\frac{(0.5 d)^2}{1 + (0.5 d)^2}

    The formula to convert **d** to an odds-ratio is given in [4]_:

    .. math:: \\text{OR} = \\exp (\\frac{d \\pi}{\\sqrt{3}})

    The formula to convert **d** to area under the curve is given in [5]_:

    .. math:: \\text{AUC} = \\mathcal{N}_{cdf}(\\frac{d}{\\sqrt{2}})

    References
    ----------
    .. [1] Rosenthal, Robert. "Parametric measures of effect size."
       The handbook of research synthesis 621 (1994): 231-244.

    .. [2] McGrath, Robert E., and Gregory J. Meyer. "When effect sizes
       disagree: the case of r and d." Psychological methods 11.4 (2006): 386.

    .. [3] Cohen, Jacob. "Statistical power analysis for the behavioral
       sciences. 2nd." (1988).

    .. [4] Borenstein, Michael, et al. "Effect sizes for continuous data."
       The handbook of research synthesis and meta-analysis 2 (2009): 221-235.

    .. [5] Ruscio, John. "A probability-based measure of effect size:
       Robustness to base rates and other factors." Psychological methods 1
       3.1 (2008): 19.

    Examples
    --------
    1. Convert from Cohen d to eta-square

    >>> import pingouin as pg
    >>> d = .45
    >>> eta = pg.convert_effsize(d, 'cohen', 'eta-square')
    >>> print(eta)
    0.048185603807257595

    2. Convert from Cohen d to Hegdes g (requires the sample sizes of each
       group)

    >>> pg.convert_effsize(.45, 'cohen', 'hedges', nx=10, ny=10)
    0.4309859154929578

    3. Convert a point-biserial correlation to Cohen d

    >>> rpb = 0.40
    >>> d = pg.convert_effsize(rpb, 'pointbiserialr', 'cohen')
    >>> print(d)
    0.8728715609439696

    4. Reverse operation: convert Cohen d to a point-biserial correlation

    >>> pg.convert_effsize(d, 'cohen', 'pointbiserialr')
    0.4000000000000001
    """
    it = input_type.lower()
    ot = output_type.lower()

    # Check input and output type
    for inp in [it, ot]:
        if not _check_eftype(inp):
            err = f"Could not interpret input '{inp}'"
            raise ValueError(err)
    if it not in ["pointbiserialr", "cohen"]:
        raise ValueError("Input type must be 'cohen' or 'pointbiserialr'")

    # Pass-through option
    if it == ot or ot == "none":
        return ef

    # Convert point-biserial r to Cohen d (Rosenthal 1994)
    d = (2 * ef) / np.sqrt(1 - ef**2) if it == "pointbiserialr" else ef

    # Then convert to the desired output type
    if ot == "cohen":
        return d
    elif ot == "hedges":
        if all(v is not None for v in [nx, ny]):
            return d * (1 - (3 / (4 * (nx + ny) - 9)))
        else:
            # If shapes of x and y are not known, return cohen's d
            warnings.warn(
                "You need to pass nx and ny arguments to compute "
                "Hedges g. Returning Cohen's d instead"
            )
            return d
    elif ot == "pointbiserialr":
        # McGrath and Meyer 2006
        if all(v is not None for v in [nx, ny]):
            a = ((nx + ny) ** 2 - 2 * (nx + ny)) / (nx * ny)
        else:
            a = 4
        return d / np.sqrt(d**2 + a)
    elif ot == "eta-square":
        # Cohen 1988
        return (d / 2) ** 2 / (1 + (d / 2) ** 2)
    elif ot == "odds-ratio":
        # Borenstein et al. 2009
        return np.exp(d * np.pi / np.sqrt(3))
    elif ot == "r":
        # https://github.com/raphaelvallat/pingouin/issues/302
        raise ValueError(
            "Using effect size 'r' in `pingouin.convert_effsize` has been deprecated. "
            "Please use 'pointbiserialr' instead."
        )
    else:  # ['auc']
        # Ruscio 2008
        from scipy.stats import norm

        return norm.cdf(d / np.sqrt(2))


def compute_effsize(x, y, paired=False, eftype="cohen"):
    """Calculate effect size between two set of observations.

    Parameters
    ----------
    x : np.array or list
        First set of observations.
    y : np.array or list
        Second set of observations.
    paired : boolean
        If True, uses Cohen d-avg formula to correct for repeated measurements
        (see Notes).
    eftype : string
        Desired output effect size.
        Available methods are:

        * ``'none'``: no effect size
        * ``'cohen'``: Unbiased Cohen d
        * ``'hedges'``: Hedges g
        * ``'r'``: Pearson correlation coefficient
        * ``'pointbiserialr'``: Point-biserial correlation
        * ``'eta-square'``: Eta-square
        * ``'odds-ratio'``: Odds ratio
        * ``'AUC'``: Area Under the Curve
        * ``'CLES'``: Common Language Effect Size

    Returns
    -------
    ef : float
        Effect size

    See Also
    --------
    convert_effsize : Conversion between effect sizes.
    compute_effsize_from_t : Convert a T-statistic to an effect size.

    Notes
    -----
    Missing values are automatically removed from the data. If ``x`` and ``y`` are paired, the
    entire row is removed.

    If ``x`` and ``y`` are independent, the Cohen :math:`d` is:

    .. math::

        d = \\frac{\\overline{X} - \\overline{Y}}
        {\\sqrt{\\frac{(n_{1} - 1)\\sigma_{1}^{2} + (n_{2} - 1)
        \\sigma_{2}^{2}}{n1 + n2 - 2}}}

    If ``x`` and ``y`` are paired, the Cohen :math:`d_{avg}` is computed:

    .. math::

        d_{avg} = \\frac{\\overline{X} - \\overline{Y}}
        {\\sqrt{\\frac{(\\sigma_1^2 + \\sigma_2^2)}{2}}}

    The Cohen's d is a biased estimate of the population effect size, especially for small samples
    (n < 20). It is often preferable to use the corrected Hedges :math:`g` instead:

    .. math:: g = d \\times (1 - \\frac{3}{4(n_1 + n_2) - 9})

    The common language effect size is the proportion of pairs where ``x`` is higher than ``y``
    (calculated with a brute-force approach where each observation of ``x`` is paired to each
    observation of ``y``, see :py:func:`pingouin.wilcoxon` for more details):

    .. math:: \\text{CL} = P(X > Y) + .5 \\times P(X = Y)

    For other effect sizes, Pingouin will first calculate a Cohen :math:`d` and then use the
    :py:func:`pingouin.convert_effsize` to convert to the desired effect size.

    References
    ----------
    * Lakens, D., 2013. Calculating and reporting effect sizes to
      facilitate cumulative science: a practical primer for t-tests and
      ANOVAs. Front. Psychol. 4, 863. https://doi.org/10.3389/fpsyg.2013.00863

    * Cumming, Geoff. Understanding the new statistics: Effect sizes,
      confidence intervals, and meta-analysis. Routledge, 2013.

    * https://osf.io/vbdah/

    Examples
    --------
    1. Cohen d from two independent samples.

    >>> import numpy as np
    >>> import pingouin as pg
    >>> x = [1, 2, 3, 4]
    >>> y = [3, 4, 5, 6, 7]
    >>> pg.compute_effsize(x, y, paired=False, eftype='cohen')
    -1.707825127659933

    The sign of the Cohen d will be opposite if we reverse the order of
    ``x`` and ``y``:

    >>> pg.compute_effsize(y, x, paired=False, eftype='cohen')
    1.707825127659933

    2. Hedges g from two paired samples.

    >>> x = [1, 2, 3, 4, 5, 6, 7]
    >>> y = [1, 3, 5, 7, 9, 11, 13]
    >>> pg.compute_effsize(x, y, paired=True, eftype='hedges')
    -0.8222477210374874

    3. Common Language Effect Size.

    >>> pg.compute_effsize(x, y, eftype='cles')
    0.2857142857142857

    In other words, there are ~29% of pairs where ``x`` is higher than ``y``,
    which means that there are ~71% of pairs where ``x`` is *lower* than ``y``.
    This can be easily verified by changing the order of ``x`` and ``y``:

    >>> pg.compute_effsize(y, x, eftype='cles')
    0.7142857142857143
    """
    # Check arguments
    if not _check_eftype(eftype):
        err = f"Could not interpret input '{eftype}'"
        raise ValueError(err)

    x = np.asarray(x)
    y = np.asarray(y)

    if x.size != y.size and paired:
        warnings.warn("x and y have unequal sizes. Switching to " "paired == False.")
        paired = False

    # Remove rows with missing values
    x, y = remove_na(x, y, paired=paired)
    nx, ny = x.size, y.size

    if ny == 1:
        # Case 1: One-sample Test
        d = (x.mean() - y) / x.std(ddof=1)
        return d
    if eftype.lower() == "r":
        # Return correlation coefficient (useful for CI bootstrapping)
        r, _ = pearsonr(x, y)
        return r
    elif eftype.lower() == "cles":
        # Compute exact CLES (see pingouin.wilcoxon)
        diff = x[:, None] - y
        return np.where(diff == 0, 0.5, diff > 0).mean()
    else:
        # Compute unbiased Cohen's d effect size
        if not paired:
            # https://en.wikipedia.org/wiki/Effect_size
            dof = nx + ny - 2
            poolsd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / dof)
            d = (x.mean() - y.mean()) / poolsd
        else:
            # Report Cohen d-avg (Cumming 2012; Lakens 2013)
            # Careful, the formula in Lakens 2013 is wrong. Updated in Pingouin
            # v0.3.4 to use the formula provided by Cummings 2012.
            # Before that the denominator was just (SD1 + SD2) / 2
            d = (x.mean() - y.mean()) / np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        return convert_effsize(d, "cohen", eftype, nx=nx, ny=ny)


def compute_effsize_from_t(tval, nx=None, ny=None, N=None, eftype="cohen"):
    """Compute effect size from a T-value.

    Parameters
    ----------
    tval : float
        T-value
    nx, ny : int, optional
        Group sample sizes.
    N : int, optional
        Total sample size (will not be used if nx and ny are specified)
    eftype : string, optional
        Desired output effect size.

    Returns
    -------
    ef : float
        Effect size

    See Also
    --------
    compute_effsize : Calculate effect size between two set of observations.
    convert_effsize : Conversion between effect sizes.

    Notes
    -----
    If both nx and ny are specified, the formula to convert from *t* to *d* is:

    .. math:: d = t * \\sqrt{\\frac{1}{n_x} + \\frac{1}{n_y}}

    If only N (total sample size) is specified, the formula is:

    .. math:: d = \\frac{2t}{\\sqrt{N}}

    Examples
    --------
    1. Compute effect size from a T-value when both sample sizes are known.

    >>> from pingouin import compute_effsize_from_t
    >>> tval, nx, ny = 2.90, 35, 25
    >>> d = compute_effsize_from_t(tval, nx=nx, ny=ny, eftype='cohen')
    >>> print(d)
    0.7593982580212534

    2. Compute effect size when only total sample size is known (nx+ny)

    >>> tval, N = 2.90, 60
    >>> d = compute_effsize_from_t(tval, N=N, eftype='cohen')
    >>> print(d)
    0.7487767802667672
    """
    if not _check_eftype(eftype):
        err = f"Could not interpret input '{eftype}'"
        raise ValueError(err)

    if not isinstance(tval, float):
        err = "T-value must be float"
        raise ValueError(err)

    # Compute Cohen d (Lakens, 2013)
    if nx is not None and ny is not None:
        d = tval * np.sqrt(1 / nx + 1 / ny)
    elif N is not None:
        d = 2 * tval / np.sqrt(N)
    else:
        raise ValueError("You must specify either nx + ny, or just N")
    return convert_effsize(d, "cohen", eftype, nx=nx, ny=ny)
