# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import warnings
import numpy as np
from pingouin.utils import _check_eftype, remove_na
# from pingouin.distribution import homoscedasticity


__all__ = ["compute_esci", "compute_bootci", "convert_effsize",
           "compute_effsize", "compute_effsize_from_t"]


def compute_esci(stat=None, nx=None, ny=None, paired=False, eftype='cohen',
                 confidence=.95, decimals=2):
    """Parametric confidence intervals around a Cohen d or a
    correlation coefficient.

    Parameters
    ----------
    stat : float
        Original effect size. Must be either a correlation coefficient or a
        Cohen-type effect size (Cohen d or Hedges g).
    nx, ny : int
        Length of vector x and y.
    paired : bool
        Indicates if the effect size was estimated from a paired sample.
        This is only relevant for cohen or hedges effect size.
    eftype : string
        Effect size type. Must be 'r' (correlation) or 'cohen'
        (Cohen d or Hedges g).
    confidence : float
        Confidence level (0.95 = 95%)
    decimals : int
        Number of rounded decimals.

    Returns
    -------
    ci : array
        Desired converted effect size

    Notes
    -----
    To compute the parametric confidence interval around a
    **Pearson r correlation** coefficient, one must first apply a
    Fisher's r-to-z transformation:

    .. math:: z = 0.5 \\cdot \\ln \\frac{1 + r}{1 - r} = \\text{arctanh}(r)

    and compute the standard deviation:

    .. math:: se = \\frac{1}{\\sqrt{n - 3}}

    where :math:`n` is the sample size.

    The lower and upper confidence intervals - *in z-space* - are then
    given by:

    .. math:: ci_z = z \\pm crit \\cdot se

    where :math:`crit` is the critical value of the nomal distribution
    corresponding to the desired confidence level (e.g. 1.96 in case of a 95%
    confidence interval).

    These confidence intervals can then be easily converted back to *r-space*:

    .. math::

        ci_r = \\frac{\\exp(2 \\cdot ci_z) - 1}{\\exp(2 \\cdot ci_z) + 1} =
        \\text{tanh}(ci_z)

    A formula for calculating the confidence interval for a
    **Cohen d effect size** is given by Hedges and Olkin (1985, p86).
    If the effect size estimate from the sample is :math:`d`, then it is
    normally distributed, with standard deviation:

    .. math::

        se = \\sqrt{\\frac{n_x + n_y}{n_x \\cdot n_y} +
        \\frac{d^2}{2 (n_x + n_y)}}

    where :math:`n_x` and :math:`n_y` are the sample sizes of the two groups.

    In one-sample test or paired test, this becomes:

    .. math::

        se = \\sqrt{\\frac{1}{n_x} + \\frac{d^2}{2 \\cdot n_x}}

    The lower and upper confidence intervals are then given by:

    .. math:: ci_d = d \\pm crit \\cdot se

    where :math:`crit` is the critical value of the nomal distribution
    corresponding to the desired confidence level (e.g. 1.96 in case of a 95%
    confidence interval).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Fisher_transformation

    .. [2] Hedges, L., and Ingram Olkin. "Statistical models for
           meta-analysis." (1985).

    .. [3] http://www.leeds.ac.uk/educol/documents/00002182.htm

    Examples
    --------
    1. Confidence interval of a Pearson correlation coefficient

    >>> import pingouin as pg
    >>> x = [3, 4, 6, 7, 5, 6, 7, 3, 5, 4, 2]
    >>> y = [4, 6, 6, 7, 6, 5, 5, 2, 3, 4, 1]
    >>> nx, ny = len(x), len(y)
    >>> stat = np.corrcoef(x, y)[0][1]
    >>> ci = pg.compute_esci(stat=stat, nx=nx, ny=ny, eftype='r')
    >>> print(stat, ci)
    0.7468280049029223 [0.27 0.93]

    2. Confidence interval of a Cohen d

    >>> import pingouin as pg
    >>> x = [3, 4, 6, 7, 5, 6, 7, 3, 5, 4, 2]
    >>> y = [4, 6, 6, 7, 6, 5, 5, 2, 3, 4, 1]
    >>> nx, ny = len(x), len(y)
    >>> stat = pg.compute_effsize(x, y, eftype='cohen')
    >>> ci = pg.compute_esci(stat=stat, nx=nx, ny=ny, eftype='cohen')
    >>> print(stat, ci)
    0.1537753990658328 [-0.68  0.99]
    """
    from scipy.stats import norm
    # Safety check
    assert eftype.lower() in['r', 'pearson', 'spearman', 'cohen',
                             'd', 'g', 'hedges']
    assert stat is not None and nx is not None
    assert isinstance(confidence, float)
    assert 0 < confidence < 1

    # Note that we are using a normal dist and not a T dist:
    # from scipy.stats import t
    # crit = np.abs(t.ppf((1 - confidence) / 2), dof)
    crit = np.abs(norm.ppf((1 - confidence) / 2))

    if eftype.lower() in ['r', 'pearson', 'spearman']:
        # Standardize correlation coefficient
        z = np.arctanh(stat)
        se = 1 / np.sqrt(nx - 3)
        ci_z = np.array([z - crit * se, z + crit * se])
        # Transform back to r
        ci = np.tanh(ci_z)
    else:
        if ny == 1 or paired:
            # One sample or paired
            se = np.sqrt(1 / nx + stat**2 / (2 * nx))
        else:
            # Two-sample test
            se = np.sqrt(((nx + ny) / (nx * ny)) + (stat**2) / (2 * (nx + ny)))
        ci = np.array([stat - crit * se, stat + crit * se])
    return np.round(ci, decimals)


def compute_bootci(x, y=None, func='pearson', method='cper', paired=False,
                   confidence=.95, n_boot=2000, decimals=2, seed=None,
                   return_dist=False):
    """Bootstrapped confidence intervals of univariate and bivariate functions.

    Parameters
    ----------
    x : 1D-array or list
        First sample. Required for both bivariate and univariate functions.
    y : 1D-array, list, or None
        Second sample. Required only for bivariate functions.
    func : str or custom function
        Function to compute the bootstrapped statistic.
        Accepted string values are::

        'pearson': Pearson correlation (bivariate, requires x and y)
        'spearman': Spearman correlation (bivariate)
        'cohen': Cohen d effect size (bivariate)
        'hedges': Hedges g effect size (bivariate)
        'mean': Mean (univariate, requires only x)
        'std': Standard deviation (univariate)
        'var': Variance (univariate)
    method : str
        Method to compute the confidence intervals::

        'norm': Normal approximation with bootstrapped bias and standard error
        'per': basic percentile method
        'cper': Bias corrected percentile method (default)
    paired : boolean
        Indicates whether x and y are paired or not. Only useful when computing
        bivariate Cohen d or Hedges g bootstrapped confidence intervals.
    confidence : float
        Confidence level (0.95 = 95%)
    n_boot : int
        Number of bootstrap iterations. The higher, the better, the slower.
    decimals : int
        Number of rounded decimals.
    seed : int or None
        Random seed for generating bootstrap samples.
    return_dist : boolean
        If True, return the confidence intervals and the bootstrapped
        distribution  (e.g. for plotting purposes).

    Returns
    -------
    ci : array
        Desired converted effect size

    Notes
    -----
    Results have been tested against the *bootci* Matlab function.

    References
    ----------
    .. [1] https://www.mathworks.com/help/stats/bootci.html

    .. [2] DiCiccio, T. J., & Efron, B. (1996). Bootstrap confidence intervals.
           Statistical science, 189-212.

    .. [3] Davison, A. C., & Hinkley, D. V. (1997). Bootstrap methods and their
           application (Vol. 1). Cambridge university press.

    Examples
    --------
    1. Bootstrapped 95% confidence interval of a Pearson correlation

    >>> import pingouin as pg
    >>> x = [3, 4, 6, 7, 5, 6, 7, 3, 5, 4, 2]
    >>> y = [4, 6, 6, 7, 6, 5, 5, 2, 3, 4, 1]
    >>> stat = np.corrcoef(x, y)[0][1]
    >>> ci = pg.compute_bootci(x, y, func='pearson', seed=42)
    >>> print(stat, ci)
    0.7468280049029223 [0.27 0.93]

    2. Bootstrapped 95% confidence interval of a Cohen d

    >>> stat = pg.compute_effsize(x, y, eftype='cohen')
    >>> ci = pg.compute_bootci(x, y, func='cohen', seed=42, decimals=3)
    >>> print(stat, ci)
    0.1537753990658328 [-0.327  0.562]

    3. Bootstrapped confidence interval of a standard deviation (univariate)

    >>> import numpy as np
    >>> stat = np.std(x, ddof=1)
    >>> ci = pg.compute_bootci(x, func='std', seed=123)
    >>> print(stat, ci)
    1.6787441193290351 [1.21 2.16]

    4. Bootstrapped confidence interval using a custom univariate function

    >>> from scipy.stats import skew
    >>> skew(x), pg.compute_bootci(x, func=skew, n_boot=10000, seed=123)
    (-0.08244607271328411, array([-1.03,  0.77]))

    5. Bootstrapped confidence interval using a custom bivariate function

    >>> stat = np.sum(np.exp(x) / np.exp(y))
    >>> ci = pg.compute_bootci(x, y, func=lambda x, y: np.sum(np.exp(x)
    ...                           / np.exp(y)), n_boot=10000, seed=123)
    >>> print(stat, ci)
    26.80405184881793 [12.76 45.15]


    6. Get the bootstrapped distribution around a Pearson correlation

    >>> ci, bstat = pg.compute_bootci(x, y, return_dist=True)
    >>> print(bstat.size)
    2000
    """
    from inspect import isfunction
    from scipy.stats import norm

    x = np.asarray(x)
    n = x.size
    assert x.ndim == 1
    assert n > 1

    if y is not None:
        y = np.asarray(y)
        ny = y.size
        assert y.ndim == 1
        assert ny > 1
        n = min(n, ny)

    assert isinstance(confidence, float)
    assert 0 < confidence < 1
    assert method in ['norm', 'normal', 'percentile', 'per', 'cpercentile',
                      'cper']
    assert isfunction(func) or isinstance(func, str)

    if isinstance(func, str):
        func_str = '%s' % func
        if func == 'pearson':

            def func(x, y):
                return np.corrcoef(x, y)[0][1]

        elif func == 'spearman':
            from scipy.stats import spearmanr

            def func(x, y):
                spr, _ = spearmanr(x, y)
                return spr

        elif func in ['cohen', 'hedges']:
            from pingouin.effsize import compute_effsize

            def func(x, y):
                return compute_effsize(x, y, paired=paired, eftype=func_str)

        elif func == 'mean':

            def func(x):
                return np.mean(x)

        elif func == 'std':

            def func(x):
                return np.std(x, ddof=1)

        elif func == 'var':

            def func(x):
                return np.var(x, ddof=1)
        else:
            raise ValueError('Function string not recognized.')

    # Bootstrap
    rng = np.random.RandomState(seed)  # Random seed
    bootsam = rng.choice(np.arange(n), size=(n, n_boot), replace=True)
    bootstat = np.empty(n_boot)

    if y is not None:
        reference = func(x, y)
        for i in range(n_boot):
            # Note that here we use a bootstrapping procedure with replacement
            # of all the pairs (Xi, Yi). This is NOT suited for
            # hypothesis testing such as p-value estimation). Instead, for the
            # latter, one must only shuffle the Y values while keeping the X
            # values constant, i.e.:
            # >>> bootsam = rng.random_sample((n_boot, n)).argsort(axis=1)
            # >>> for i in range(n_boot):
            # >>>   bootstat[i] = func(x, y[bootsam[i, :]])
            bootstat[i] = func(x[bootsam[:, i]], y[bootsam[:, i]])
    else:
        reference = func(x)
        for i in range(n_boot):
            bootstat[i] = func(x[bootsam[:, i]])

    # CONFIDENCE INTERVALS
    alpha = 1 - confidence
    dist_sorted = np.sort(bootstat)

    if method in ['norm', 'normal']:
        # Normal approximation
        za = norm.ppf(alpha / 2)
        se = np.std(bootstat, ddof=1)

        bias = np.mean(bootstat - reference)
        ll = reference - bias + se * za
        ul = reference - bias - se * za
        ci = [ll, ul]
    elif method in ['percentile', 'per']:
        # Uncorrected percentile
        pct_ll = int(n_boot * (alpha / 2))
        pct_ul = int(n_boot * (1 - alpha / 2))
        ci = [dist_sorted[pct_ll], dist_sorted[pct_ul]]
    else:
        # Corrected percentile bootstrap
        # Compute bias-correction constant z0
        z_0 = norm.ppf(np.mean(bootstat < reference) +
                       np.mean(bootstat == reference) / 2)
        z_alpha = norm.ppf(alpha / 2)
        pct_ul = 100 * norm.cdf(2 * z_0 - z_alpha)
        pct_ll = 100 * norm.cdf(2 * z_0 + z_alpha)
        ll = np.percentile(bootstat, pct_ll)
        ul = np.percentile(bootstat, pct_ul)
        ci = [ll, ul]

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
        Original effect size
    input_type : string
        Effect size type of ef. Must be 'r' or 'd'.
    output_type : string
        Desired effect size type.
        Available methods are ::

        'cohen' : Unbiased Cohen d
        'hedges' : Hedges g
        'glass': Glass delta
        'eta-square' : Eta-square
        'odds-ratio' : Odds ratio
        'AUC' : Area Under the Curve
        'none' : pass-through (return ``ef``)

    nx, ny : int, optional
        Length of vector x and y.
        nx and ny are required to convert to Hedges g

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
    The formula to convert **r** to **d** is given in ref [1]:

    .. math:: d = \\frac{2r}{\\sqrt{1 - r^2}}

    The formula to convert **d** to **r** is given in ref [2]:

    .. math::

        r = \\frac{d}{\\sqrt{d^2 + \\frac{(n_x + n_y)^2 - 2(n_x + n_y)}
        {n_xn_y}}}

    The formula to convert **d** to :math:`\\eta^2` is given in ref [3]:

    .. math:: \\eta^2 = \\frac{(0.5 * d)^2}{1 + (0.5 * d)^2}

    The formula to convert **d** to an odds-ratio is given in ref [4]:

    .. math:: OR = e(\\frac{d * \\pi}{\\sqrt{3}})

    The formula to convert **d** to area under the curve is given in ref [5]:

    .. math:: AUC = \\mathcal{N}_{cdf}(\\frac{d}{\\sqrt{2}})

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

    >>> from pingouin import convert_effsize
    >>> d = .45
    >>> eta = convert_effsize(d, 'cohen', 'eta-square')
    >>> print(eta)
    0.048185603807257595

    2. Convert from Cohen d to Hegdes g (requires the sample sizes of each
       group)

    >>> d = .45
    >>> g = convert_effsize(d, 'cohen', 'hedges', nx=10, ny=10)
    >>> print(g)
    0.4309859154929578

    3. Convert Pearson r to Cohen d

    >>> r = 0.40
    >>> d = convert_effsize(r, 'r', 'cohen')
    >>> print(d)
    0.8728715609439696

    4. Reverse operation: convert Cohen d to Pearson r

    >>> d = 0.873
    >>> r = convert_effsize(d, 'cohen', 'r')
    >>> print(r)
    0.40004943911648533
    """
    it = input_type.lower()
    ot = output_type.lower()

    # Check input and output type
    for input in [it, ot]:
        if not _check_eftype(input):
            err = "Could not interpret input '{}'".format(input)
            raise ValueError(err)
    if it not in ['r', 'cohen']:
        raise ValueError("Input type must be 'r' or 'cohen'")

    # Pass-through option
    if it == ot or ot == 'none':
        return ef

    # Convert r to Cohen d (Rosenthal 1994)
    d = (2 * ef) / np.sqrt(1 - ef**2) if it == 'r' else ef

    # Then convert to the desired output type
    if ot == 'cohen':
        return d
    elif ot == 'hedges':
        if all(v is not None for v in [nx, ny]):
            return d * (1 - (3 / (4 * (nx + ny) - 9)))
        else:
            # If shapes of x and y are not known, return cohen's d
            warnings.warn("You need to pass nx and ny arguments to compute "
                          "Hedges g. Returning Cohen's d instead")
            return d
    elif ot == 'glass':
        warnings.warn("Returning original effect size instead of Glass "
                      "because variance is not known.")
        return ef
    elif ot == 'r':
        # McGrath and Meyer 2006
        if all(v is not None for v in [nx, ny]):
            a = ((nx + ny)**2 - 2 * (nx + ny)) / (nx * ny)
        else:
            a = 4
        return d / np.sqrt(d**2 + a)
    elif ot == 'eta-square':
        # Cohen 1988
        return (d / 2)**2 / (1 + (d / 2)**2)
    elif ot == 'odds-ratio':
        # Borenstein et al. 2009
        return np.exp(d * np.pi / np.sqrt(3))
    elif ot in ['auc', 'cles']:
        # Ruscio 2008
        from scipy.stats import norm
        return norm.cdf(d / np.sqrt(2))


def compute_effsize(x, y, paired=False, eftype='cohen'):
    """Calculate effect size between two set of observations.

    Parameters
    ----------
    x : np.array or list
        First set of observations.
    y : np.array or list
        Second set of observations.
    paired : boolean
        If True, uses Cohen d-avg formula to correct for repeated measurements
        (Cumming 2012)
    eftype : string
        Desired output effect size.
        Available methods are ::

        'none' : no effect size
        'cohen' : Unbiased Cohen d
        'hedges' : Hedges g
        'glass': Glass delta
        'r' : correlation coefficient
        'eta-square' : Eta-square
        'odds-ratio' : Odds ratio
        'AUC' : Area Under the Curve
        'CLES' : Common language effect size

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
    Missing values are automatically removed from the data. If ``x`` and ``y``
    are paired, the entire row is removed.

    If ``x`` and ``y`` are independent, the Cohen's d is:

    .. math::

        d = \\frac{\\overline{X} - \\overline{Y}}
        {\\sqrt{\\frac{(n_{1} - 1)\\sigma_{1}^{2} + (n_{2} - 1)
        \\sigma_{2}^{2}}{n1 + n2 - 2}}}

    If ``x`` and ``y`` are paired, the Cohen :math:`d_{avg}` is computed:

    .. math::

        d_{avg} = \\frac{\\overline{X} - \\overline{Y}}
        {0.5 * (\\sigma_1 + \\sigma_2)}

    The Cohen’s d is a biased estimate of the population effect size,
    especially for small samples (n < 20). It is often preferable
    to use the corrected effect size, or Hedges’g, instead:

    .. math:: g = d * (1 - \\frac{3}{4(n_1 + n_2) - 9})

    If eftype = 'glass', the Glass :math:`\\delta` is reported, using the
    group with the lowest variance as the control group:

    .. math::

        \\delta = \\frac{\\overline{X} - \\overline{Y}}{\\sigma_{control}}

    References
    ----------
    .. [1] Lakens, D., 2013. Calculating and reporting effect sizes to
       facilitate cumulative science: a practical primer for t-tests and
       ANOVAs. Front. Psychol. 4, 863. https://doi.org/10.3389/fpsyg.2013.00863

    .. [2] Cumming, Geoff. Understanding the new statistics: Effect sizes,
           confidence intervals, and meta-analysis. Routledge, 2013.

    Examples
    --------
    1. Compute Cohen d from two independent set of observations.

    >>> import numpy as np
    >>> from pingouin import compute_effsize
    >>> np.random.seed(123)
    >>> x = np.random.normal(2, size=100)
    >>> y = np.random.normal(2.3, size=95)
    >>> d = compute_effsize(x=x, y=y, eftype='cohen', paired=False)
    >>> print(d)
    -0.2835170152506578

    2. Compute Hedges g from two paired set of observations.

    >>> import numpy as np
    >>> from pingouin import compute_effsize
    >>> x = [1.62, 2.21, 3.79, 1.66, 1.86, 1.87, 4.51, 4.49, 3.3 , 2.69]
    >>> y = [0.91, 3., 2.28, 0.49, 1.42, 3.65, -0.43, 1.57, 3.27, 1.13]
    >>> g = compute_effsize(x=x, y=y, eftype='hedges', paired=True)
    >>> print(g)
    0.8370985097811404

    3. Compute Glass delta from two independent set of observations. The group
       with the lowest variance will automatically be selected as the control.

    >>> import numpy as np
    >>> from pingouin import compute_effsize
    >>> np.random.seed(123)
    >>> x = np.random.normal(2, scale=1, size=50)
    >>> y = np.random.normal(2, scale=2, size=45)
    >>> d = compute_effsize(x=x, y=y, eftype='glass')
    >>> print(d)
    -0.1170721973604153
    """
    # Check arguments
    if not _check_eftype(eftype):
        err = "Could not interpret input '{}'".format(eftype)
        raise ValueError(err)

    x = np.asarray(x)
    y = np.asarray(y)

    if x.size != y.size and paired:
        warnings.warn("x and y have unequal sizes. Switching to "
                      "paired == False.")
        paired = False

    # Remove rows with missing values
    x, y = remove_na(x, y, paired=paired)
    nx, ny = x.size, y.size

    if ny == 1:
        # Case 1: One-sample Test
        d = (x.mean() - y) / x.std(ddof=1)
        return d
    if eftype.lower() == 'glass':
        # Find group with lowest variance
        sd_control = np.min([x.std(ddof=1), y.std(ddof=1)])
        d = (x.mean() - y.mean()) / sd_control
        return d
    elif eftype.lower() == 'r':
        # Return correlation coefficient (useful for CI bootstrapping)
        from scipy.stats import pearsonr
        r, _ = pearsonr(x, y)
        return r
    elif eftype.lower() == 'cles':
        # Compute exact CLES
        diff = x[:, None] - y
        return max((diff < 0).sum(), (diff > 0).sum()) / diff.size
    else:
        # Test equality of variance of data with a stringent threshold
        # equal_var, p = homoscedasticity(x, y, alpha=.001)
        # if not equal_var:
        #     print('Unequal variances (p<.001). You should report',
        #           'Glass delta instead.')

        # Compute unbiased Cohen's d effect size
        if not paired:
            # https://en.wikipedia.org/wiki/Effect_size
            dof = nx + ny - 2
            poolsd = np.sqrt(((nx - 1) * x.var(ddof=1)
                              + (ny - 1) * y.var(ddof=1)) / dof)
            d = (x.mean() - y.mean()) / poolsd
        else:
            # Report Cohen d-avg (Cumming 2012; Lakens 2013)
            d = (x.mean() - y.mean()) / (.5 * (x.std(ddof=1)
                                               + y.std(ddof=1)))
        return convert_effsize(d, 'cohen', eftype, nx=nx, ny=ny)


def compute_effsize_from_t(tval, nx=None, ny=None, N=None, eftype='cohen'):
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
        desired output effect size

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
        err = "Could not interpret input '{}'".format(eftype)
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
        raise ValueError('You must specify either nx + ny, or just N')
    return convert_effsize(d, 'cohen', eftype, nx=nx, ny=ny)
