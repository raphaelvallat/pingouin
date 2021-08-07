# Author: Raphael Vallat <raphaelvallat9@gmail.com>
import warnings
import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau

from pingouin.config import options
from pingouin.power import power_corr
from pingouin.multicomp import multicomp
from pingouin.effsize import compute_esci
from pingouin.utils import remove_na, _perm_pval, _postprocess_dataframe
from pingouin.bayesian import bayesfactor_pearson


__all__ = ["corr", "partial_corr", "pcorr", "rcorr", "rm_corr", "distance_corr"]


def _correl_pvalue(r, n, k=0, alternative="two-sided"):
    """Compute the p-value of a correlation coefficient.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Using_the_exact_distribution

    See also scipy.stats._ttest_finish

    Parameters
    ----------
    r : float
        Correlation coefficient.
    n : int
        Sample size
    k : int
        Number of covariates for (semi)-partial correlation.
    alternative : string
        Tail of the test.

    Returns
    -------
    pval : float
        p-value.

    Notes
    -----
    This uses the same approach as :py:func:`scipy.stats.pearsonr` to calculate
    the p-value (i.e. using a beta distribution)
    """
    from scipy.stats import t
    assert alternative in ['two-sided', 'greater', 'less'], (
        "Alternative must be one of 'two-sided' (default), 'greater' or 'less'.")

    # Method 1: using a student T distribution
    dof = n - k - 2
    tval = r * np.sqrt(dof / (1 - r**2))
    if alternative == 'less':
        pval = t.cdf(tval, dof)
    elif alternative == 'greater':
        pval = t.sf(tval, dof)
    elif alternative == 'two-sided':
        pval = 2 * t.sf(np.abs(tval), dof)

    # Method 2: beta distribution (similar to scipy.stats.pearsonr, faster)
    # from scipy.special import btdtr
    # ab = (n - k) / 2 - 1
    # pval = 2 * btdtr(ab, ab, 0.5 * (1 - abs(np.float64(r))))
    return pval


def skipped(x, y, corr_type='spearman'):
    """Skipped correlation (Rousselet and Pernet 2012).

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    corr_type : str
        Method used to compute the correlation after outlier removal. Can be
        either 'spearman' (default) or 'pearson'.

    Returns
    -------
    r : float
        Skipped correlation coefficient.
    pval : float
        Two-tailed p-value.
    outliers : array of bool
        Indicate if value is an outlier or not

    Notes
    -----
    The skipped correlation involves multivariate outlier detection using a
    projection technique (Wilcox, 2004, 2005). First, a robust estimator of
    multivariate location and scatter, for instance the minimum covariance
    determinant estimator (MCD; Rousseeuw, 1984; Rousseeuw and van Driessen,
    1999; Hubert et al., 2008) is computed. Second, data points are
    orthogonally projected on lines joining each of the data point to the
    location estimator. Third, outliers are detected using a robust technique.
    Finally, Spearman correlations are computed on the remaining data points
    and calculations are adjusted by taking into account the dependency among
    the remaining data points.

    Code inspired by Matlab code from Cyril Pernet and Guillaume
    Rousselet [1]_.

    Requires scikit-learn.

    References
    ----------
    .. [1] Pernet CR, Wilcox R, Rousselet GA. Robust Correlation Analyses:
       False Positive and Power Validation Using a New Open Source Matlab
       Toolbox. Frontiers in Psychology. 2012;3:606.
       doi:10.3389/fpsyg.2012.00606.
    """
    # Check that sklearn is installed
    from pingouin.utils import _is_sklearn_installed
    _is_sklearn_installed(raise_error=True)
    from scipy.stats import chi2
    from sklearn.covariance import MinCovDet
    X = np.column_stack((x, y))
    nrows, ncols = X.shape
    gval = np.sqrt(chi2.ppf(0.975, 2))
    # Compute center and distance to center
    center = MinCovDet(random_state=42).fit(X).location_

    # https://github.com/raphaelvallat/pingouin/issues/164
    warnings.warn(
        "The skipped correlation relies on the Minimum Covariance Determinant "
        "algorithm, which gives slightly different results in Python "
        "(scikit-learn) than in the original Matlab library (LIBRA). As such, "
        "the skipped correlation may be different from the Matlab robust "
        "correlation toolbox (see issue 164 on Pingouin's GitHub). "
        "Make sure to double check your results or use another robust "
        "correlation method."
    )

    B = X - center
    bot = (B**2).sum(axis=1)
    # Loop over rows
    dis = np.zeros(shape=(nrows, nrows))
    for i in np.arange(nrows):
        if bot[i] != 0:  # Avoid division by zero error
            dis[i, :] = np.linalg.norm(
                B.dot(B[i, :, None]) * B[i, :] / bot[i], axis=1)

    # Detect outliers
    def idealf(x):
        """Compute the ideal fourths IQR (Wilcox 2012).
        """
        n = len(x)
        j = int(np.floor(n / 4 + 5 / 12))
        y = np.sort(x)
        g = (n / 4) - j + (5 / 12)
        low = (1 - g) * y[j - 1] + g * y[j]
        k = n - j + 1
        up = (1 - g) * y[k - 1] + g * y[k - 2]
        return up - low

    # One can either use the MAD or the IQR (see Wilcox 2012)
    # MAD = mad(dis, axis=1)
    iqr = np.apply_along_axis(idealf, 1, dis)
    thresh = (np.median(dis, axis=1) + gval * iqr)
    outliers = np.apply_along_axis(np.greater, 0, dis, thresh).any(axis=0)
    # Compute correlation on remaining data
    if corr_type == 'spearman':
        r, pval = spearmanr(X[~outliers, 0], X[~outliers, 1])
    else:
        r, pval = pearsonr(X[~outliers, 0], X[~outliers, 1])
    return r, pval, outliers


def bsmahal(a, b, n_boot=200):
    """
    Bootstraps Mahalanobis distances for Shepherd's pi correlation.

    Parameters
    ----------
    a : ndarray (shape=(n, 2))
        Data
    b : ndarray (shape=(n, 2))
        Data
    n_boot : int
        Number of bootstrap samples to calculate.

    Returns
    -------
    m : ndarray (shape=(n,))
        Mahalanobis distance for each row in a, averaged across all the
        bootstrap resamples.
    """
    n, m = b.shape
    MD = np.zeros((n, n_boot))
    nr = np.arange(n)
    xB = np.random.choice(nr, size=(n_boot, n), replace=True)
    # Bootstrap the MD
    for i in np.arange(n_boot):
        s1 = b[xB[i, :], 0]
        s2 = b[xB[i, :], 1]
        X = np.column_stack((s1, s2))
        mu = X.mean(0)
        _, R = np.linalg.qr(X - mu)
        sol = np.linalg.solve(R.T, (a - mu).T)
        MD[:, i] = np.sum(sol**2, 0) * (n - 1)
    # Average across all bootstraps
    return MD.mean(1)


def shepherd(x, y, n_boot=200):
    """
    Shepherd's Pi correlation, equivalent to Spearman's rho after outliers
    removal.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    n_boot : int
        Number of bootstrap samples to calculate.

    Returns
    -------
    r : float
        Pi correlation coefficient
    pval : float
        Two-tailed adjusted p-value.
    outliers : array of bool
        Indicate if value is an outlier or not

    Notes
    -----
    It first bootstraps the Mahalanobis distances, removes all observations
    with m >= 6 and finally calculates the correlation of the remaining data.

    Pi is Spearman's Rho after outlier removal.
    """
    X = np.column_stack((x, y))
    # Bootstrapping on Mahalanobis distance
    m = bsmahal(X, X, n_boot)
    # Determine outliers
    outliers = (m >= 6)
    # Compute correlation
    r, pval = spearmanr(x[~outliers], y[~outliers])
    # (optional) double the p-value to achieve a nominal false alarm rate
    # pval *= 2
    # pval = 1 if pval > 1 else pval
    return r, pval, outliers


def percbend(x, y, beta=.2):
    """
    Percentage bend correlation (Wilcox 1994).

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    beta : float
        Bending constant for omega (0 <= beta <= 0.5).

    Returns
    -------
    r : float
        Percentage bend correlation coefficient.
    pval : float
        Two-tailed p-value.

    Notes
    -----
    Code inspired by Matlab code from Cyril Pernet and Guillaume Rousselet.

    References
    ----------
    .. [1] Wilcox, R.R., 1994. The percentage bend correlation coefficient.
       Psychometrika 59, 601–616. https://doi.org/10.1007/BF02294395

    .. [2] Pernet CR, Wilcox R, Rousselet GA. Robust Correlation Analyses:
       False Positive and Power Validation Using a New Open Source Matlab
       Toolbox. Frontiers in Psychology. 2012;3:606.
       doi:10.3389/fpsyg.2012.00606.
    """
    X = np.column_stack((x, y))
    nx = X.shape[0]
    M = np.tile(np.median(X, axis=0), nx).reshape(X.shape)
    W = np.sort(np.abs(X - M), axis=0)
    m = int((1 - beta) * nx)
    omega = W[m - 1, :]
    P = (X - M) / omega
    P[np.isinf(P)] = 0
    P[np.isnan(P)] = 0

    # Loop over columns
    a = np.zeros((2, nx))
    for c in [0, 1]:
        psi = P[:, c]
        i1 = np.where(psi < -1)[0].size
        i2 = np.where(psi > 1)[0].size
        s = X[:, c].copy()
        s[np.where(psi < -1)[0]] = 0
        s[np.where(psi > 1)[0]] = 0
        pbos = (np.sum(s) + omega[c] * (i2 - i1)) / (s.size - i1 - i2)
        a[c] = (X[:, c] - pbos) / omega[c]

    # Bend
    a[a <= -1] = -1
    a[a >= 1] = 1

    # Correlation coefficient
    a, b = a
    r = (a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum())
    pval = _correl_pvalue(r, nx, k=0)
    return r, pval


def bicor(x, y, c=9):
    """
    Biweight midcorrelation.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    c : float
        Tuning constant for the biweight estimator (default = 9.0).

    Returns
    -------
    r : float
        Correlation coefficient.
    pval : float
        Two-tailed p-value.

    Notes
    -----
    This function will return (np.nan, np.nan) if mad(x) == 0 or mad(y) == 0.

    References
    ----------
    https://en.wikipedia.org/wiki/Biweight_midcorrelation

    https://docs.astropy.org/en/stable/api/astropy.stats.biweight.biweight_midcovariance.html

    Langfelder, P., & Horvath, S. (2012). Fast R Functions for Robust
    Correlations and Hierarchical Clustering. Journal of Statistical Software,
    46(11). https://www.ncbi.nlm.nih.gov/pubmed/23050260
    """
    # Calculate median
    nx = x.size
    x_median = np.median(x)
    y_median = np.median(y)

    # Raw median absolute deviation
    x_mad = np.median(np.abs(x - x_median))
    y_mad = np.median(np.abs(y - y_median))
    if x_mad == 0 or y_mad == 0:
        # From Langfelder and Horvath 2012:
        # "Strictly speaking, a call to bicor in R should return a missing
        # value if mad(x) = 0 or mad(y) = 0." This avoids division by zero.
        return np.nan, np.nan

    # Calculate weights
    u = (x - x_median) / (c * x_mad)
    v = (y - y_median) / (c * y_mad)
    w_x = (1 - u**2)**2 * ((1 - np.abs(u)) > 0)
    w_y = (1 - v**2)**2 * ((1 - np.abs(v)) > 0)

    # Normalize x and y by weights
    x_norm = (x - x_median) * w_x
    y_norm = (y - y_median) * w_y
    denom = (np.sqrt((x_norm**2).sum()) * np.sqrt((y_norm**2).sum()))

    # Correlation coefficient
    r = (x_norm * y_norm).sum() / denom
    pval = _correl_pvalue(r, nx, k=0)
    return r, pval


def corr(x, y, alternative='two-sided', method='pearson', **kwargs):
    """(Robust) correlation between two variables.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. ``x`` and ``y`` must be
        independent.
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
        * ``'kendall'``: Kendall's :math:`\\tau_B` correlation (for ordinal data)
        * ``'bicor'``: Biweight midcorrelation (robust)
        * ``'percbend'``: Percentage bend correlation (robust)
        * ``'shepherd'``: Shepherd's pi correlation (robust)
        * ``'skipped'``: Skipped correlation (robust)
    **kwargs : optional
        Optional argument(s) passed to the lower-level correlation functions.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'n'``: Sample size (after removal of missing values)
        * ``'outliers'``: number of outliers, only if a robust method was used
        * ``'r'``: Correlation coefficient
        * ``'CI95'``: 95% parametric confidence intervals around :math:`r`
        * ``'p-val'``: p-value
        * ``'BF10'``: Bayes Factor of the alternative hypothesis (only for Pearson correlation)
        * ``'power'``: achieved power of the test with an alpha of 0.05.

    See also
    --------
    pairwise_corr : Pairwise correlation between columns of a pandas DataFrame
    partial_corr : Partial correlation
    rm_corr : Repeated measures correlation

    Notes
    -----
    The `Pearson correlation coefficient
    <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
    measures the linear relationship between two datasets. Strictly speaking,
    Pearson's correlation requires that each dataset be normally distributed.
    Correlations of -1 or +1 imply a perfect negative and positive linear
    relationship, respectively, with 0 indicating the absence of association.

    .. math::
        r_{xy} = \\frac{\\sum_i(x_i - \\bar{x})(y_i - \\bar{y})}
        {\\sqrt{\\sum_i(x_i - \\bar{x})^2} \\sqrt{\\sum_i(y_i - \\bar{y})^2}}
        = \\frac{\\text{cov}(x, y)}{\\sigma_x \\sigma_y}

    where :math:`\\text{cov}` is the sample covariance and :math:`\\sigma`
    is the sample standard deviation.

    If ``method='pearson'``, The Bayes Factor is calculated using the
    :py:func:`pingouin.bayesfactor_pearson` function.

    The `Spearman correlation coefficient
    <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_
    is a non-parametric measure of the monotonicity of the relationship between
    two datasets. Unlike the Pearson correlation, the Spearman correlation does
    not assume that both datasets are normally distributed. Correlations of -1
    or +1 imply an exact negative and positive monotonic relationship,
    respectively. Mathematically, the Spearman correlation coefficient is
    defined as the Pearson correlation coefficient between the
    `rank variables <https://en.wikipedia.org/wiki/Ranking>`_.

    The `Kendall correlation coefficient
    <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_
    is a measure of the correspondence between two rankings. Values also range
    from -1 (perfect disagreement) to 1 (perfect agreement), with 0 indicating
    the absence of association. Consistent with
    :py:func:`scipy.stats.kendalltau`, Pingouin returns the Tau-b coefficient,
    which adjusts for ties:

    .. math:: \\tau_B = \\frac{(P - Q)}{\\sqrt{(P + Q + T) (P + Q + U)}}

    where :math:`P` is the number of concordant pairs, :math:`Q` the number of
    discordand pairs, :math:`T` the number of ties in x, and :math:`U`
    the number of ties in y.

    The `biweight midcorrelation
    <https://en.wikipedia.org/wiki/Biweight_midcorrelation>`_ and
    percentage bend correlation [1]_ are both robust methods that
    protects against *univariate* outliers by down-weighting observations that
    deviate too much from the median.

    The Shepherd pi [2]_ correlation and skipped [3]_, [4]_ correlation are
    both robust methods that returns the Spearman correlation coefficient after
    removing *bivariate* outliers. Briefly, the Shepherd pi uses a
    bootstrapping of the Mahalanobis distance to identify outliers, while the
    skipped correlation is based on the minimum covariance determinant
    (which requires scikit-learn). Note that these two methods are
    significantly slower than the previous ones.

    The confidence intervals for the correlation coefficient are estimated
    using the Fisher transformation.

    .. important:: Rows with missing values (NaN) are automatically removed.

    References
    ----------
    .. [1] Wilcox, R.R., 1994. The percentage bend correlation coefficient.
       Psychometrika 59, 601–616. https://doi.org/10.1007/BF02294395

    .. [2] Schwarzkopf, D.S., De Haas, B., Rees, G., 2012. Better ways to
       improve standards in brain-behavior correlation analysis. Front.
       Hum. Neurosci. 6, 200. https://doi.org/10.3389/fnhum.2012.00200

    .. [3] Rousselet, G.A., Pernet, C.R., 2012. Improving standards in
       brain-behavior correlation analyses. Front. Hum. Neurosci. 6, 119.
       https://doi.org/10.3389/fnhum.2012.00119

    .. [4] Pernet, C.R., Wilcox, R., Rousselet, G.A., 2012. Robust correlation
       analyses: false positive and power validation using a new open
       source matlab toolbox. Front. Psychol. 3, 606.
       https://doi.org/10.3389/fpsyg.2012.00606

    Examples
    --------
    1. Pearson correlation

    >>> import numpy as np
    >>> import pingouin as pg
    >>> # Generate random correlated samples
    >>> np.random.seed(123)
    >>> mean, cov = [4, 6], [(1, .5), (.5, 1)]
    >>> x, y = np.random.multivariate_normal(mean, cov, 30).T
    >>> # Compute Pearson correlation
    >>> pg.corr(x, y).round(3)
              n      r         CI95%  p-val  BF10  power
    pearson  30  0.491  [0.16, 0.72]  0.006  8.55  0.809

    2. Pearson correlation with two outliers

    >>> x[3], y[5] = 12, -8
    >>> pg.corr(x, y).round(3)
              n      r          CI95%  p-val   BF10  power
    pearson  30  0.147  [-0.23, 0.48]  0.439  0.302  0.121

    3. Spearman correlation (robust to outliers)

    >>> pg.corr(x, y, method="spearman").round(3)
               n      r         CI95%  p-val  power
    spearman  30  0.401  [0.05, 0.67]  0.028   0.61

    4. Biweight midcorrelation (robust)

    >>> pg.corr(x, y, method="bicor").round(3)
            n      r         CI95%  p-val  power
    bicor  30  0.393  [0.04, 0.66]  0.031  0.592

    5. Percentage bend correlation (robust)

    >>> pg.corr(x, y, method='percbend').round(3)
               n      r         CI95%  p-val  power
    percbend  30  0.389  [0.03, 0.66]  0.034  0.581

    6. Shepherd's pi correlation (robust)

    >>> pg.corr(x, y, method='shepherd').round(3)
               n  outliers      r        CI95%  p-val  power
    shepherd  30         2  0.437  [0.08, 0.7]   0.02  0.662

    7. Skipped spearman correlation (robust)

    >>> pg.corr(x, y, method='skipped').round(3)
              n  outliers      r        CI95%  p-val  power
    skipped  30         2  0.437  [0.08, 0.7]   0.02  0.662

    8. One-tailed Pearson correlation

    >>> pg.corr(x, y, alternative="greater", method='pearson').round(3)
            n      r           CI95%  p-val   BF10  power
    pearson  30  0.147  [-0.17, 1.0]   0.22  0.467  0.194

    >>> pg.corr(x, y, alternative="less", method='pearson').round(3)
            n        r         CI95%  p-val   BF10  power
    pearson  30  0.147  [-1.0, 0.43]   0.78  0.137  0.008

    9. Perfect correlation

    >>> pg.corr(x, -x).round(3)
              n    r         CI95%  p-val BF10  power
    pearson  30 -1.0  [-1.0, -1.0]    0.0  inf      1

    10. Using columns of a pandas dataframe

    >>> import pandas as pd
    >>> data = pd.DataFrame({'x': x, 'y': y})
    >>> pg.corr(data['x'], data['y']).round(3)
              n      r          CI95%  p-val   BF10  power
    pearson  30  0.147  [-0.23, 0.48]  0.439  0.302  0.121
    """
    # Safety check
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == y.ndim == 1, 'x and y must be 1D array.'
    assert x.size == y.size, 'x and y must have the same length.'
    assert alternative in ['two-sided', 'greater', 'less'], (
        "Alternative must be one of 'two-sided' (default), 'greater' or 'less'.")
    if "tail" in kwargs:
        raise ValueError(
            "Since Pingouin 0.4.0, the 'tail' argument has been renamed to 'alternative'.")

    # Remove rows with missing values
    x, y = remove_na(x, y, paired=True)
    n = x.size

    # Compute correlation coefficient and two-sided p-value
    if method == 'pearson':
        r, pval = pearsonr(x, y)
    elif method == 'spearman':
        r, pval = spearmanr(x, y, **kwargs)
    elif method == 'kendall':
        r, pval = kendalltau(x, y, **kwargs)
    elif method == 'bicor':
        r, pval = bicor(x, y, **kwargs)
    elif method == 'percbend':
        r, pval = percbend(x, y, **kwargs)
    elif method == 'shepherd':
        r, pval, outliers = shepherd(x, y, **kwargs)
    elif method == 'skipped':
        r, pval, outliers = skipped(x, y, **kwargs)
    else:
        raise ValueError(f'Method "{method}" not recognized.')

    if np.isnan(r):
        # Correlation failed -- new in version v0.3.4, instead of raising an
        # error we just return a dataframe full of NaN (except sample size).
        # This avoid sudden stop in pingouin.pairwise_corr.
        return pd.DataFrame({
            'n': n, 'r': np.nan, 'CI95%': np.nan, 'p-val': np.nan, 'BF10': np.nan,
            'power': np.nan}, index=[method])

    # Sample size after outlier removal
    n_outliers = sum(outliers) if "outliers" in locals() else 0
    n_clean = n - n_outliers

    # Compute the parametric 95% confidence interval and power
    if abs(r) == 1:
        ci = [r, r]
        pr = 1
    else:
        ci = compute_esci(
            stat=r, nx=n_clean, ny=n_clean, eftype='r', decimals=6, alternative=alternative)
        pr = power_corr(r=r, n=n_clean, power=None, alpha=0.05, alternative=alternative)

    # Recompute p-value if tail is one-sided
    if alternative != "two-sided":
        pval = _correl_pvalue(r, n_clean, k=0, alternative=alternative)

    # Create dictionnary
    stats = {'n': n, 'r': r, 'CI95%': [ci], 'p-val': pval, 'power': pr}

    if method in ['shepherd', 'skipped']:
        stats['outliers'] = n_outliers

    # Compute the BF10 for Pearson correlation only
    if method == 'pearson':
        stats['BF10'] = bayesfactor_pearson(r, n_clean, alternative=alternative)

    # Convert to DataFrame
    stats = pd.DataFrame.from_records(stats, index=[method])

    # Define order
    col_keep = ['n', 'outliers', 'r', 'CI95%', 'p-val', 'BF10', 'power']
    col_order = [k for k in col_keep if k in stats.keys().tolist()]
    return _postprocess_dataframe(stats)[col_order]


@pf.register_dataframe_method
def partial_corr(data=None, x=None, y=None, covar=None, x_covar=None,
                 y_covar=None, alternative='two-sided', method='pearson', **kwargs):
    """Partial and semi-partial correlation.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Pandas Dataframe. Note that this function can also directly be used
        as a :py:class:`pandas.DataFrame` method, in which case this argument
        is no longer needed.
    x, y : string
        x and y. Must be names of columns in ``data``.
    covar : string or list
        Covariate(s). Must be a names of columns in ``data``. Use a list if
        there are two or more covariates.
    x_covar : string or list
        Covariate(s) for the ``x`` variable. This is used to compute
        semi-partial correlation (i.e. the effect of ``x_covar`` is removed
        from ``x`` but not from ``y``). Only one of ``covar``,  ``x_covar`` and
        ``y_covar`` can be specified.
    y_covar : string or list
        Covariate(s) for the ``y`` variable. This is used to compute
        semi-partial correlation (i.e. the effect of ``y_covar`` is removed
        from ``y`` but not from ``x``). Only one of ``covar``,  ``x_covar`` and
        ``y_covar`` can be specified.
    alternative : string
        Defines the alternative hypothesis, or tail of the partial correlation. Must be one of
        "two-sided" (default), "greater" or "less". Both "greater" and "less" return a one-sided
        p-value. "greater" tests against the alternative hypothesis that the partial correlation is
        positive (greater than zero), "less" tests against the hypothesis that the partial
        correlation is negative.
    method : string
        Correlation type:

        * ``'pearson'``: Pearson :math:`r` product-moment correlation
        * ``'spearman'``: Spearman :math:`\\rho` rank-order correlation

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'n'``: Sample size (after removal of missing values)
        * ``'r'``: Partial correlation coefficient
        * ``'CI95'``: 95% parametric confidence intervals around :math:`r`
        * ``'p-val'``: p-value

    See also
    --------
    corr, pcorr, pairwise_corr, rm_corr

    Notes
    -----
    Partial correlation [1]_ measures the degree of association between ``x``
    and ``y``, after removing the effect of one or more controlling variables
    (``covar``, or :math:`Z`). Practically, this is achieved by calculating the
    correlation coefficient between the residuals of two linear regressions:

    .. math:: x \\sim Z, y \\sim Z

    Like the correlation coefficient, the partial correlation
    coefficient takes on a value in the range from –1 to 1, where 1 indicates a
    perfect positive association.

    The semipartial correlation is similar to the partial correlation,
    with the exception that the set of controlling variables is only
    removed for either ``x`` or ``y``, but not both.

    Pingouin uses the method described in [2]_ to calculate the (semi)partial
    correlation coefficients and associated p-values. This method is based on
    the inverse covariance matrix and is significantly faster than the
    traditional regression-based method. Results have been tested against the
    `ppcor <https://cran.r-project.org/web/packages/ppcor/index.html>`_
    R package.

    .. important:: Rows with missing values are automatically removed from
        data.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Partial_correlation

    .. [2] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4681537/

    Examples
    --------
    1. Partial correlation with one covariate

    >>> import pingouin as pg
    >>> df = pg.read_dataset('partial_corr')
    >>> pg.partial_corr(data=df, x='x', y='y', covar='cv1').round(3)
              n      r         CI95%  p-val
    pearson  30  0.568  [0.25, 0.77]  0.001

    2. Spearman partial correlation with several covariates

    >>> # Partial correlation of x and y controlling for cv1, cv2 and cv3
    >>> pg.partial_corr(data=df, x='x', y='y', covar=['cv1', 'cv2', 'cv3'],
    ...                 method='spearman').round(3)
               n      r         CI95%  p-val
    spearman  30  0.521  [0.18, 0.75]  0.005

    3. Same but one-sided test

    >>> pg.partial_corr(data=df, x='x', y='y', covar=['cv1', 'cv2', 'cv3'],
    ...                 alternative="greater", method='spearman').round(3)
               n      r        CI95%  p-val
    spearman  30  0.521  [0.24, 1.0]  0.003

    >>> pg.partial_corr(data=df, x='x', y='y', covar=['cv1', 'cv2', 'cv3'],
    ...                 alternative="less", method='spearman').round(3)
               n      r         CI95%  p-val
    spearman  30  0.521  [-1.0, 0.72]  0.997

    4. As a pandas method

    >>> df.partial_corr(x='x', y='y', covar=['cv1'], method='spearman').round(3)
               n      r         CI95%  p-val
    spearman  30  0.578  [0.27, 0.78]  0.001

    5. Partial correlation matrix (returns only the correlation coefficients)

    >>> df.pcorr().round(3)
             x      y    cv1    cv2    cv3
    x    1.000  0.493 -0.095  0.130 -0.385
    y    0.493  1.000 -0.007  0.104 -0.002
    cv1 -0.095 -0.007  1.000 -0.241 -0.470
    cv2  0.130  0.104 -0.241  1.000 -0.118
    cv3 -0.385 -0.002 -0.470 -0.118  1.000

    6. Semi-partial correlation on x

    >>> pg.partial_corr(data=df, x='x', y='y', x_covar=['cv1', 'cv2', 'cv3']).round(3)
              n      r        CI95%  p-val
    pearson  30  0.463  [0.1, 0.72]  0.015
    """
    from pingouin.utils import _flatten_list
    # Safety check
    assert alternative in ['two-sided', 'greater', 'less'], (
        "Alternative must be one of 'two-sided' (default), 'greater' or 'less'.")
    if "tail" in kwargs:
        raise ValueError(
            "Since Pingouin 0.4.0, the 'tail' argument has been renamed to 'alternative'.")
    assert method in ['pearson', 'spearman'], (
        'only "pearson" and "spearman" are supported for partial correlation.')
    assert isinstance(data, pd.DataFrame), 'data must be a pandas DataFrame.'
    assert data.shape[0] > 2, 'Data must have at least 3 samples.'
    assert isinstance(x, (str, tuple)), 'x must be a string.'
    assert isinstance(y, (str, tuple)), 'y must be a string.'
    assert isinstance(covar, (str, list, type(None)))
    assert isinstance(x_covar, (str, list, type(None)))
    assert isinstance(y_covar, (str, list, type(None)))
    if covar is not None and (x_covar is not None or y_covar is not None):
        raise ValueError('Cannot specify both covar and {x,y}_covar.')
    if x_covar is not None and y_covar is not None:
        raise ValueError('Cannot specify both x_covar and y_covar.')
    assert x != covar, 'x and covar must be independent'
    assert y != covar, 'y and covar must be independent'
    assert x != y, 'x and y must be independent'
    if isinstance(covar, list):
        assert x not in covar, 'x and covar must be independent'
        assert y not in covar, 'y and covar must be independent'
    # Check that columns exist
    col = _flatten_list([x, y, covar, x_covar, y_covar])
    assert all([c in data for c in col]), 'columns are not in dataframe.'
    # Check that columns are numeric
    assert all([data[c].dtype.kind in 'bfiu' for c in col])

    # Drop rows with NaN
    data = data[col].dropna()
    n = data.shape[0]  # Number of samples
    k = data.shape[1] - 2  # Number of covariates
    assert n > 2, 'Data must have at least 3 non-NAN samples.'

    # Calculate the partial corrrelation matrix - similar to pingouin.pcorr()
    if method == "spearman":
        # Convert the data to rank, similar to R cov()
        V = data.rank(na_option='keep').cov()
    else:
        V = data.cov()
    Vi = np.linalg.pinv(V)  # Inverse covariance matrix
    Vi_diag = Vi.diagonal()
    D = np.diag(np.sqrt(1 / Vi_diag))
    pcor = -1 * (D @ Vi @ D)  # Partial correlation matrix

    if covar is not None:
        r = pcor[0, 1]
    else:
        # Semi-partial correlation matrix
        with np.errstate(divide='ignore'):
            spcor = pcor / \
                np.sqrt(np.diag(V))[..., None] / \
                np.sqrt(np.abs(Vi_diag - Vi ** 2 / Vi_diag[..., None])).T
        if y_covar is not None:
            r = spcor[0, 1]  # y_covar is removed from y
        else:
            r = spcor[1, 0]  # x_covar is removed from x

    if np.isnan(r):
        # Correlation failed. Return NaN. When would this happen?
        return pd.DataFrame(
            {'n': n, 'r': np.nan, 'CI95%': np.nan, 'p-val': np.nan}, index=[method])

    # Compute the two-sided p-value and confidence intervals
    # https://online.stat.psu.edu/stat505/lesson/6/6.3
    pval = _correl_pvalue(r, n, k, alternative)
    ci = compute_esci(
        stat=r, nx=(n - k), ny=(n - k), eftype='r', decimals=6, alternative=alternative)

    # Create dictionnary
    stats = {
        'n': n,
        'r': r,
        'CI95%': [ci],
        'p-val': pval,
    }

    # Convert to DataFrame
    stats = pd.DataFrame.from_records(stats, index=[method])

    # Define order
    col_keep = ['n', 'r', 'CI95%', 'p-val']
    col_order = [k for k in col_keep if k in stats.keys().tolist()]
    return _postprocess_dataframe(stats)[col_order]


@pf.register_dataframe_method
def pcorr(self):
    """Partial correlation matrix (:py:class:`pandas.DataFrame` method).

    Returns
    ----------
    pcormat : :py:class:`pandas.DataFrame`
        Partial correlation matrix.

    Notes
    -----
    This function calculates the pairwise partial correlations for each pair of
    variables in a :py:class:`pandas.DataFrame` given all the others. It has
    the same behavior as the pcor function in the
    `ppcor <https://cran.r-project.org/web/packages/ppcor/index.html>`_
    R package.

    Note that this function only returns the raw Pearson correlation
    coefficient. If you want to calculate the test statistic and p-values, or
    use more robust estimates of the correlation coefficient, please refer to
    the :py:func:`pingouin.pairwise_corr` or :py:func:`pingouin.partial_corr`
    functions.

    Examples
    --------
    >>> import pingouin as pg
    >>> data = pg.read_dataset('mediation')
    >>> data.pcorr().round(3)
              X      M      Y   Mbin   Ybin     W1     W2
    X     1.000  0.359  0.074 -0.019 -0.147 -0.148 -0.067
    M     0.359  1.000  0.555 -0.024 -0.112 -0.138 -0.176
    Y     0.074  0.555  1.000 -0.001  0.169  0.101  0.108
    Mbin -0.019 -0.024 -0.001  1.000 -0.080 -0.032 -0.040
    Ybin -0.147 -0.112  0.169 -0.080  1.000 -0.000 -0.140
    W1   -0.148 -0.138  0.101 -0.032 -0.000  1.000 -0.394
    W2   -0.067 -0.176  0.108 -0.040 -0.140 -0.394  1.000

    On a subset of columns

    >>> data[['X', 'Y', 'M']].pcorr()
              X         Y         M
    X  1.000000  0.036649  0.412804
    Y  0.036649  1.000000  0.540140
    M  0.412804  0.540140  1.000000
    """
    V = self.cov()  # Covariance matrix
    Vi = np.linalg.pinv(V)  # Inverse covariance matrix
    D = np.diag(np.sqrt(1 / np.diag(Vi)))
    pcor = -1 * (D @ Vi @ D)  # Partial correlation matrix
    pcor[np.diag_indices_from(pcor)] = 1
    return pd.DataFrame(pcor, index=V.index, columns=V.columns)


@pf.register_dataframe_method
def rcorr(self, method='pearson', upper='pval', decimals=3, padjust=None,
          stars=True, pval_stars={0.001: '***', 0.01: '**', 0.05: '*'}):
    """
    Correlation matrix of a dataframe with p-values and/or sample size on the
    upper triangle (:py:class:`pandas.DataFrame` method).

    This method is a faster, but less exhaustive, matrix-version of the
    :py:func:`pingouin.pairwise_corr` function. It is based on the
    :py:func:`pandas.DataFrame.corr` method. Missing values are automatically
    removed from each pairwise correlation.

    Parameters
    ----------
    self : :py:class:`pandas.DataFrame`
        Input dataframe.
    method : str
        Correlation method. Can be either 'pearson' or 'spearman'.
    upper : str
        If 'pval', the upper triangle of the output correlation matrix shows
        the p-values. If 'n', the upper triangle is the sample size used in
        each pairwise correlation.
    decimals : int
        Number of decimals to display in the output correlation matrix.
    padjust : string or None
        Method used for testing and adjustment of pvalues.

        * ``'none'``: no correction
        * ``'bonf'``: one-step Bonferroni correction
        * ``'sidak'``: one-step Sidak correction
        * ``'holm'``: step-down method using Bonferroni adjustments
        * ``'fdr_bh'``: Benjamini/Hochberg FDR correction
        * ``'fdr_by'``: Benjamini/Yekutieli FDR correction
    stars : boolean
        If True, only significant p-values are displayed as stars using the
        pre-defined thresholds of ``pval_stars``. If False, all the raw
        p-values are displayed.
    pval_stars : dict
        Significance thresholds. Default is 3 stars for p-values < 0.001,
        2 stars for p-values < 0.01 and 1 star for p-values < 0.05.

    Returns
    -------
    rcorr : :py:class:`pandas.DataFrame`
        Correlation matrix, of type str.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import pingouin as pg
    >>> # Load an example dataset of personality dimensions
    >>> df = pg.read_dataset('pairwise_corr').iloc[:, 1:]
    >>> # Add some missing values
    >>> df.iloc[[2, 5, 20], 2] = np.nan
    >>> df.iloc[[1, 4, 10], 3] = np.nan
    >>> df.head().round(2)
       Neuroticism  Extraversion  Openness  Agreeableness  Conscientiousness
    0         2.48          4.21      3.94           3.96               3.46
    1         2.60          3.19      3.96            NaN               3.23
    2         2.81          2.90       NaN           2.75               3.50
    3         2.90          3.56      3.52           3.17               2.79
    4         3.02          3.33      4.02            NaN               2.85

    >>> # Correlation matrix on the four first columns
    >>> df.iloc[:, 0:4].rcorr()
                  Neuroticism Extraversion Openness Agreeableness
    Neuroticism             -          ***                     **
    Extraversion        -0.35            -      ***
    Openness            -0.01        0.265        -           ***
    Agreeableness      -0.134        0.054    0.161             -

    >>> # Spearman correlation and Holm adjustement for multiple comparisons
    >>> df.iloc[:, 0:4].rcorr(method='spearman', padjust='holm')
                  Neuroticism Extraversion Openness Agreeableness
    Neuroticism             -          ***                     **
    Extraversion       -0.325            -      ***
    Openness           -0.027         0.24        -           ***
    Agreeableness       -0.15         0.06    0.173             -

    >>> # Compare with the pg.pairwise_corr function
    >>> pairwise = df.iloc[:, 0:4].pairwise_corr(method='spearman',
    ...                                          padjust='holm')
    >>> pairwise[['X', 'Y', 'r', 'p-corr']].round(3)  # Do not show all columns
                  X              Y      r  p-corr
    0   Neuroticism   Extraversion -0.325   0.000
    1   Neuroticism       Openness -0.027   0.543
    2   Neuroticism  Agreeableness -0.150   0.002
    3  Extraversion       Openness  0.240   0.000
    4  Extraversion  Agreeableness  0.060   0.358
    5      Openness  Agreeableness  0.173   0.000

    >>> # Display the raw p-values with four decimals
    >>> df.iloc[:, [0, 1, 3]].rcorr(stars=False, decimals=4)
                  Neuroticism Extraversion Agreeableness
    Neuroticism             -       0.0000        0.0028
    Extraversion      -0.3501            -        0.2305
    Agreeableness      -0.134       0.0539             -

    >>> # With the sample size on the upper triangle instead of the p-values
    >>> df.iloc[:, [0, 1, 2]].rcorr(upper='n')
                 Neuroticism Extraversion Openness
    Neuroticism            -          500      497
    Extraversion       -0.35            -      497
    Openness           -0.01        0.265        -
    """
    from numpy import triu_indices_from as tif
    from numpy import format_float_positional as ffp
    from scipy.stats import pearsonr, spearmanr

    # Safety check
    assert isinstance(pval_stars, dict), 'pval_stars must be a dictionnary.'
    assert isinstance(decimals, int), 'decimals must be an int.'
    assert method in ['pearson', 'spearman'], 'Method is not recognized.'
    assert upper in ['pval', 'n'], 'upper must be either `pval` or `n`.'
    mat = self.corr(method=method).round(decimals)
    if upper == 'n':
        mat_upper = self.corr(method=lambda x, y: len(x)).astype(int)
    else:
        if method == 'pearson':
            mat_upper = self.corr(method=lambda x, y: pearsonr(x, y)[1])
        else:
            # Method = 'spearman'
            mat_upper = self.corr(method=lambda x, y: spearmanr(x, y)[1])

        if padjust is not None:
            pvals = mat_upper.to_numpy()[tif(mat, k=1)]
            mat_upper.to_numpy()[tif(mat, k=1)] = multicomp(pvals, alpha=0.05, method=padjust)[1]

    # Convert r to text
    mat = mat.astype(str)
    # Inplace modification of the diagonal
    np.fill_diagonal(mat.to_numpy(), '-')

    if upper == 'pval':

        def replace_pval(x):
            for key, value in pval_stars.items():
                if x < key:
                    return value
            return ''

        if stars:
            # Replace p-values by stars
            mat_upper = mat_upper.applymap(replace_pval)
        else:
            mat_upper = mat_upper.applymap(lambda x: ffp(x, precision=decimals))

    # Replace upper triangle by p-values or n
    mat.to_numpy()[tif(mat, k=1)] = mat_upper.to_numpy()[tif(mat, k=1)]
    return mat


def rm_corr(data=None, x=None, y=None, subject=None):
    """Repeated measures correlation.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Dataframe.
    x, y : string
        Name of columns in ``data`` containing the two dependent variables.
    subject : string
        Name of column in ``data`` containing the subject indicator.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'r'``: Repeated measures correlation coefficient
        * ``'dof'``: Degrees of freedom
        * ``'pval'``: p-value
        * ``'CI95'``: 95% parametric confidence intervals
        * ``'power'``: achieved power of the test (= 1 - type II error).

    See also
    --------
    plot_rm_corr

    Notes
    -----
    Repeated measures correlation (rmcorr) is a statistical technique for determining the common
    within-individual association for paired measures assessed on two or more occasions for
    multiple individuals.

    From `Bakdash and Marusich (2017)
    <https://doi.org/10.3389/fpsyg.2017.00456>`_:

        *Rmcorr accounts for non-independence among observations using analysis
        of covariance (ANCOVA) to statistically adjust for inter-individual
        variability. By removing measured variance between-participants,
        rmcorr provides the best linear fit for each participant using parallel
        regression lines (the same slope) with varying intercepts.
        Like a Pearson correlation coefficient, the rmcorr coefficient
        is bounded by − 1 to 1 and represents the strength of the linear
        association between two variables.*

    Results have been tested against the `rmcorr <https://github.com/cran/rmcorr>`_ R package.

    Missing values are automatically removed from the dataframe (listwise deletion).

    Examples
    --------
    >>> import pingouin as pg
    >>> df = pg.read_dataset('rm_corr')
    >>> pg.rm_corr(data=df, x='pH', y='PacO2', subject='Subject')
                   r  dof      pval           CI95%     power
    rm_corr -0.50677   38  0.000847  [-0.71, -0.23]  0.929579

    Now plot using the :py:func:`pingouin.plot_rm_corr` function:

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset('rm_corr')
        >>> g = pg.plot_rm_corr(data=df, x='pH', y='PacO2', subject='Subject')
    """
    from pingouin import ancova, power_corr
    # Safety checks
    assert isinstance(data, pd.DataFrame), 'Data must be a DataFrame'
    assert x in data.columns, 'The %s column is not in data.' % x
    assert y in data.columns, 'The %s column is not in data.' % y
    assert data[x].dtype.kind in 'bfiu', '%s must be numeric.' % x
    assert data[y].dtype.kind in 'bfiu', '%s must be numeric.' % y
    assert subject in data.columns, 'The %s column is not in data.' % subject
    if data[subject].nunique() < 3:
        raise ValueError('rm_corr requires at least 3 unique subjects.')

    # Remove missing values
    data = data[[x, y, subject]].dropna(axis=0)

    # Using PINGOUIN
    # For max precision, make sure rounding is disabled
    old_options = options.copy()
    options['round'] = None
    aov = ancova(dv=y, covar=x, between=subject, data=data)
    options.update(old_options)  # restore options
    bw = aov.bw_  # Beta within parameter
    sign = np.sign(bw)
    dof = int(aov.at[2, 'DF'])
    n = dof + 2
    ssfactor = aov.at[1, 'SS']
    sserror = aov.at[2, 'SS']
    rm = sign * np.sqrt(ssfactor / (ssfactor + sserror))
    pval = aov.at[1, 'p-unc']
    ci = compute_esci(stat=rm, nx=n, eftype='pearson').tolist()
    pwr = power_corr(r=rm, n=n, alternative="two-sided")
    # Convert to Dataframe
    stats = pd.DataFrame({"r": rm,
                          "dof": int(dof),
                          "pval": pval,
                          "CI95%": [ci],
                          "power": pwr}, index=["rm_corr"])
    return _postprocess_dataframe(stats)


def _dcorr(y, n2, A, dcov2_xx):
    """Helper function for distance correlation bootstrapping.
    """
    # Pairwise Euclidean distances
    b = squareform(pdist(y, metric='euclidean'))
    # Double centering
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    # Compute squared distance covariances
    dcov2_yy = np.vdot(B, B) / n2
    dcov2_xy = np.vdot(A, B) / n2
    return np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))


def distance_corr(x, y, alternative='greater', n_boot=1000, seed=None):
    """Distance correlation between two arrays.

    Statistical significance (p-value) is evaluated with a permutation test.

    Parameters
    ----------
    x, y : array_like
        1D or 2D input arrays, shape (n_samples, n_features).
        ``x`` and ``y`` must have the same number of samples and must not
        contain missing values.
    alternative : str
        Alternative of the test. Can be either "two-sided", "greater" (default) or "less".
        To be consistent with the original R implementation, the default is to calculate the
        one-sided "greater" p-value.
    n_boot : int or None
        Number of bootstrap to perform. If None, no bootstrapping is performed and the function
        only returns the distance correlation (no p-value). Default is 1000 (thus giving a
        precision of 0.001).
    seed : int or None
        Random state seed.

    Returns
    -------
    dcor : float
        Sample distance correlation (range from 0 to 1).
    pval : float
        P-value.

    Notes
    -----
    From Wikipedia:

        *Distance correlation is a measure of dependence between two paired
        random vectors of arbitrary, not necessarily equal, dimension. The
        distance correlation coefficient is zero if and only if the random
        vectors are independent. Thus, distance correlation measures both
        linear and nonlinear association between two random variables or
        random vectors. This is in contrast to Pearson's correlation, which can
        only detect linear association between two random variables.*

    The distance correlation of two random variables is obtained by
    dividing their distance covariance by the product of their distance
    standard deviations:

    .. math::

        \\text{dCor}(X, Y) = \\frac{\\text{dCov}(X, Y)}
        {\\sqrt{\\text{dVar}(X) \\cdot \\text{dVar}(Y)}}

    where :math:`\\text{dCov}(X, Y)` is the square root of the arithmetic
    average of the product of the double-centered pairwise Euclidean distance
    matrices.

    Note that by contrast to Pearson's correlation, the distance correlation
    cannot be negative, i.e :math:`0 \\leq \\text{dCor} \\leq 1`.

    Results have been tested against the
    `energy <https://cran.r-project.org/web/packages/energy/energy.pdf>`_
    R package.

    References
    ----------
    * https://en.wikipedia.org/wiki/Distance_correlation

    * Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
      Measuring and testing dependence by correlation of distances.
      The annals of statistics, 35(6), 2769-2794.

    * https://gist.github.com/satra/aa3d19a12b74e9ab7941

    * https://gist.github.com/wladston/c931b1495184fbb99bec

    Examples
    --------
    1. With two 1D vectors

    >>> from pingouin import distance_corr
    >>> a = [1, 2, 3, 4, 5]
    >>> b = [1, 2, 9, 4, 4]
    >>> dcor, pval = distance_corr(a, b, seed=9)
    >>> print(round(dcor, 3), pval)
    0.763 0.312

    2. With two 2D arrays and no p-value

    >>> import numpy as np
    >>> np.random.seed(123)
    >>> from pingouin import distance_corr
    >>> a = np.random.random((10, 10))
    >>> b = np.random.random((10, 10))
    >>> round(distance_corr(a, b, n_boot=None), 3)
    0.88
    """
    assert alternative in ['two-sided', 'greater', 'less'], (
        "Alternative must be one of 'two-sided' (default), 'greater' or 'less'.")
    x = np.asarray(x)
    y = np.asarray(y)
    # Check for NaN values
    if any([np.isnan(np.min(x)), np.isnan(np.min(y))]):
        raise ValueError('Input arrays must not contain NaN values.')
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    assert x.shape[0] == y.shape[0], 'x and y must have same number of samples'

    # Extract number of samples
    n = x.shape[0]
    n2 = n**2

    # Process first array to avoid redundancy when performing bootstrap
    a = squareform(pdist(x, metric='euclidean'))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    dcov2_xx = np.vdot(A, A) / n2

    # Process second array and compute final distance correlation
    dcor = _dcorr(y, n2, A, dcov2_xx)

    # Compute one-sided p-value using a bootstrap procedure
    if n_boot is not None and n_boot > 1:
        # Define random seed and permutation
        rng = np.random.RandomState(seed)
        bootsam = rng.random_sample((n_boot, n)).argsort(axis=1)
        bootstat = np.empty(n_boot)
        for i in range(n_boot):
            bootstat[i] = _dcorr(y[bootsam[i, :]], n2, A, dcov2_xx)

        pval = _perm_pval(bootstat, dcor, alternative=alternative)
        return dcor, pval
    else:
        return dcor
