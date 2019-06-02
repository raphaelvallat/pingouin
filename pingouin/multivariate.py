import numpy as np
import pandas as pd
from pingouin.utils import remove_na

__all__ = ["multivariate_normality", "multivariate_ttest"]


def multivariate_normality(X, alpha=.05):
    """Henze-Zirkler multivariate normality test.

    Parameters
    ----------
    X : np.array
        Data matrix of shape (n_samples, n_features).
    alpha : float
        Significance level.

    Returns
    -------
    normal : boolean
        True if X comes from a multivariate normal distribution.
    p : float
        P-value.

    See Also
    --------
    normality : Test the univariate normality of one or more variables.
    homoscedasticity : Test equality of variance.
    sphericity : Mauchly's test for sphericity.

    Notes
    -----
    The Henze-Zirkler test has a good overall power against alternatives
    to normality and is feasable for any dimension and any sample size.

    Adapted to Python from a Matlab code by Antonio Trujillo-Ortiz and
    tested against the R package MVN.

    Rows with missing values are automatically removed using the
    :py:func:`remove_na` function.

    References
    ----------
    .. [1] Henze, N., & Zirkler, B. (1990). A class of invariant consistent
       tests for multivariate normality. Communications in Statistics-Theory
       and Methods, 19(10), 3595-3617.

    .. [2] Trujillo-Ortiz, A., R. Hernandez-Walls, K. Barba-Rojo and L.
       Cupul-Magana. (2007). HZmvntest: Henze-Zirkler's Multivariate
       Normality Test. A MATLAB file.

    Examples
    --------
    >>> import pingouin as pg
    >>> data = pg.read_dataset('multivariate')
    >>> X = data[['Fever', 'Pressure', 'Aches']]
    >>> normal, p = pg.multivariate_normality(X, alpha=.05)
    >>> print(normal, round(p, 3))
    True 0.717
    """
    from scipy.stats import lognorm

    # Check input and remove missing values
    X = np.asarray(X)
    assert X.ndim == 2, 'X must be of shape (n_samples, n_features).'
    X = X[~np.isnan(X).any(axis=1)]
    n, p = X.shape
    assert n >= 3, 'X must have at least 3 rows.'
    assert p >= 2, 'X must have at least two columns.'

    # Covariance matrix
    S = np.cov(X, rowvar=False, bias=True)
    S_inv = np.linalg.pinv(S)
    difT = X - X.mean(0)

    # Squared-Mahalanobis distances
    Dj = np.diag(np.linalg.multi_dot([difT, S_inv, difT.T]))
    Y = np.linalg.multi_dot([X, S_inv, X.T])
    Djk = -2 * Y.T + np.repeat(np.diag(Y.T), n).reshape(n, -1) + \
        np.tile(np.diag(Y.T), (n, 1))

    # Smoothing parameter
    b = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4)**(1 / (p + 4)) * \
        (n**(1 / (p + 4)))

    hz = n * 4
    if np.linalg.matrix_rank(S) == p:
        hz = n * (1 / (n**2) * np.sum(np.sum(np.exp(-(b**2) / 2 * Djk))) - 2
                  * ((1 + (b**2))**(-p / 2)) * (1 / n)
                  * (np.sum(np.exp(-((b**2) / (2 * (1 + (b**2)))) * Dj)))
                  + ((1 + (2 * (b**2)))**(-p / 2)))

    wb = (1 + b**2) * (1 + 3 * b**2)
    a = 1 + 2 * b**2
    # Mean and variance
    mu = 1 - a**(-p / 2) * (1 + p * b**2 / a + (p * (p + 2)
                                                * (b**4)) / (2 * a**2))
    si2 = 2 * (1 + 4 * b**2)**(-p / 2) + 2 * a**(-p) * \
        (1 + (2 * p * b**4) / a**2 + (3 * p * (p + 2) * b**8) / (4 * a**4)) \
        - 4 * wb**(-p / 2) * (1 + (3 * p * b**4) / (2 * wb)
                              + (p * (p + 2) * b**8) / (2 * wb**2))

    # Lognormal mean and variance
    pmu = np.log(np.sqrt(mu**4 / (si2 + mu**2)))
    psi = np.sqrt(np.log((si2 + mu**2) / mu**2))

    # P-value
    pval = lognorm.sf(hz, psi, scale=np.exp(pmu))
    normal = True if pval > alpha else False
    return normal, pval


def multivariate_ttest(X, Y=None, paired=False):
    """Hotelling T-squared test (= multivariate T-test)

    Parameters
    ----------
    X : np.array
        First data matrix of shape (n_samples, n_features).
    Y : np.array or None
        Second data matrix of shape (n_samples, n_features). If ``Y`` is a 1D
        array of shape (n_features), a one-sample test is performed where the
        null hypothesis is defined in ``Y``. If ``Y`` is None, a one-sample
        is performed against np.zeros(n_features).
    paired : boolean
        Specify whether the two observations are related (i.e. repeated
        measures) or independent. If ``paired`` is True, ``X`` and ``Y`` must
        have exactly the same shape.

    Returns
    -------
    stats : pandas DataFrame
        Hotelling T-squared test summary ::

        'T2' : T-squared value
        'F' : F-value
        'df1' : first degree of freedom
        'df2' : second degree of freedom
        'p-val' : p-value

    See Also
    --------
    multivariate_normality : Multivariate normality test
    ttest : Univariate T-test.

    Notes
    -----
    Hotelling 's T-squared test is the multivariate counterpart of the T-test.

    Rows with missing values are automatically removed using the
    :py:func:`remove_na` function.

    Tested against the R package Hotelling.

    References
    ----------
    .. [1] Hotelling, H. The Generalization of Student's Ratio. Ann. Math.
           Statist. 2 (1931), no. 3, 360--378.

    .. [2] http://www.real-statistics.com/multivariate-statistics/

    .. [3] https://cran.r-project.org/web/packages/Hotelling/Hotelling.pdf

    Examples
    --------
    Two-sample independent Hotelling T-squared test

    >>> import pingouin as pg
    >>> data = pg.read_dataset('multivariate')
    >>> dvs = ['Fever', 'Pressure', 'Aches']
    >>> X = data[data['Condition'] == 'Drug'][dvs]
    >>> Y = data[data['Condition'] == 'Placebo'][dvs]
    >>> pg.multivariate_ttest(X, Y)
                  T2      F  df1  df2      pval
    hotelling  4.229  1.327    3   32  0.282898

    Two-sample paired Hotelling T-squared test

    >>> pg.multivariate_ttest(X, Y, paired=True)
                  T2      F  df1  df2      pval
    hotelling  4.468  1.314    3   15  0.306542

    One-sample Hotelling T-squared test with a specified null hypothesis

    >>> null_hypothesis_means = [37.5, 70, 5]
    >>> pg.multivariate_ttest(X, Y=null_hypothesis_means)
                    T2      F  df1  df2          pval
    hotelling  253.231  74.48    3   15  3.081281e-09
    """
    from scipy.stats import f
    x = np.asarray(X)
    assert x.ndim == 2, 'x must be of shape (n_samples, n_features)'

    if Y is None:
        y = np.zeros(x.shape[1])
        # Remove rows with missing values in x
        x = x[~np.isnan(x).any(axis=1)]
    else:
        nx, kx = x.shape
        y = np.asarray(Y)
        if y.ndim == 1:
            # One sample with specified null
            assert y.size == kx
        elif y.ndim == 2:
            # Two-sample
            err = 'X and Y must have the same number of features (= columns).'
            assert y.shape[1] == kx, err
            if paired:
                err = 'X and Y must have the same number of rows if paired.'
                assert y.shape[0] == nx, err
        # Remove rows with missing values in both x and y
        x, y = remove_na(x, y, paired=paired, axis='rows')

    # Shape of arrays
    nx, k = x.shape
    ny = y.shape[0]
    assert nx >= 5, 'At least five samples are required.'

    if y.ndim == 1 or paired is True:
        n = nx
        if y.ndim == 1:
            # One sample test
            cov = np.cov(x, rowvar=False)
            diff = x.mean(0) - y
        else:
            # Paired two sample
            cov = np.cov(x - y, rowvar=False)
            diff = x.mean(0) - y.mean(0)
        inv_cov = np.linalg.pinv(cov)
        t2 = (diff @ inv_cov) @ diff * n
    else:
        n = nx + ny - 1
        x_cov = np.cov(x, rowvar=False)
        y_cov = np.cov(y, rowvar=False)
        pooled_cov = ((nx - 1) * x_cov + (ny - 1) * y_cov) / (n - 1)
        inv_cov = np.linalg.pinv((1 / nx + 1 / ny) * pooled_cov)
        diff = x.mean(0) - y.mean(0)
        t2 = (diff @ inv_cov) @ diff

    # F-value, degrees of freedom and p-value
    fval = t2 * (n - k) / (k * (n - 1))
    df1 = k
    df2 = n - k
    pval = f.sf(fval, df1, df2)

    # Create output dictionnary
    stats = {'T2': t2, 'F': fval, 'df1': df1, 'df2': df2, 'pval': pval}
    stats = pd.DataFrame(stats, index=['hotelling'])
    stats[['T2', 'F']] = stats[['T2', 'F']].round(3)
    return stats
