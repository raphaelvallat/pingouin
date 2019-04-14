import numpy as np

__all__ = ["multivariate_normality"]


def multivariate_normality(X, alpha=.05):
    """Henze-Zirkler multivariate normality test.

    Parameters
    ----------
    X : np.array
        Data matrix of shape (n, p) where n are the observations and p the
        variables.
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

    Aapted to Python from a Matlab code by Antonio Trujillo-Ortiz.

    Tested against the R package MVN.

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
    1. Test for multivariate normality of 2 variables

    >>> import numpy as np
    >>> from pingouin import multivariate_normality
    >>> np.random.seed(123)
    >>> mean, cov, n = [4, 6], [[1, .5], [.5, 1]], 30
    >>> X = np.random.multivariate_normal(mean, cov, n)
    >>> normal, p = multivariate_normality(X, alpha=.05)
    >>> print(normal, p)
    True 0.7523511059223078

    2. Test for multivariate normality of 3 variables

    >>> import numpy as np
    >>> from pingouin import multivariate_normality
    >>> np.random.seed(123)
    >>> mean, cov = [4, 6, 5], [[1, .5, .2], [.5, 1, .1], [.2, .1, 1]]
    >>> X = np.random.multivariate_normal(mean, cov, 50)
    >>> normal, p = multivariate_normality(X, alpha=.05)
    >>> print(normal, p)
    True 0.4607466031757833
    """
    from scipy.stats import lognorm

    # Check input
    X = np.asarray(X)
    assert X.ndim == 2
    n, p = X.shape
    assert p >= 2

    # Covariance matrix
    S = np.cov(X, rowvar=False, bias=True)
    S_inv = np.linalg.inv(S)
    difT = X - X.mean(0)
    # Squared-Mahalanobis distances
    Dj = np.diag(np.linalg.multi_dot([difT, S_inv, difT.T]))
    Y = np.linalg.multi_dot([X, S_inv, X.T])
    Djk = -2 * Y.T + np.repeat(np.diag(Y.T), n).reshape(n, -1) + \
        np.tile(np.diag(Y.T), (n, 1))

    # Smoothing parameter
    b = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4)**(1 / (p + 4)) * \
        (n**(1 / (p + 4)))

    if np.linalg.matrix_rank(S) == p:
        hz = n * (1 / (n**2) * np.sum(np.sum(np.exp(-(b**2) / 2 * Djk))) - 2
                  * ((1 + (b**2))**(-p / 2)) * (1 / n)
                  * (np.sum(np.exp(-((b**2) / (2 * (1 + (b**2)))) * Dj)))
                  + ((1 + (2 * (b**2)))**(-p / 2)))
    else:
        hz = n * 4

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
