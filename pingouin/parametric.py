# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np

__all__ = ["gzscore", "test_normality", "test_homoscedasticity", "test_dist"]

def gzscore(x):
    """Compute the geometric standard score of a 1D array.

    Geometric Z-score are better than arithmetic z-scores when the data
    comes from a log-normal or chi-squares distribution.

    Parameters
    ----------
    x: array_like
        Array of raw values

    Returns
    -------
    gzscore: array_like
        Array of geometric z-scores (gzscore.shape == x.shape)
    """
    from scipy.stats import gmean
    # Geometric mean
    geo_mean = gmean(x)
    # Geometric standard deviation
    gstd = np.exp(np.sqrt(np.sum((np.log(x/geo_mean))**2) / (len(x) - 1)))
    # Geometric z-score
    return np.log(x/geo_mean) / np.log(gstd)

# MAIN FUNCTIONS
def test_normality(*args, alpha=.05):
    """Test normality of an array.

    Parameters
    ----------
    sample1, sample2,... : array_like
        Array of sample data. May be different lengths.

    Returns
    -------
    normal: boolean
        True if x comes from a normal distribution.
    p: float
        P-value.
    """
    from scipy.stats import shapiro
    # Handle empty input
    for a in args:
        if np.asanyarray(a).size == 0:
            return np.nan, np.nan

    k = len(args)
    p = np.zeros(k)
    normal = np.zeros(k, 'bool')
    for j in range(k):
        _, p[j] = shapiro(args[j])
        normal[j] = True if p[j] > alpha else False

    if k == 1:
        normal = bool(normal)
        p = float(p)

    return normal, p


def test_homoscedasticity(*args, alpha=.05):
    """Test equality of variance.

    If data are normally distributed, uses Bartlett (1937).
    If data are not-normally distributed, uses Levene (1960).

    Parameters
    ----------
    sample1, sample2,... : array_like
        Array of sample data. May be different lengths.

    Returns
    -------
    equal_var: boolean
        True if data have equal variance.
    p: float
        P-value.
    """
    from scipy.stats import levene, bartlett
    # Handle empty input
    for a in args:
        if np.asanyarray(a).size == 0:
            return np.nan, np.nan

    k = len(args)
    if k < 2:
        raise ValueError("Must enter at least two input sample vectors.")

    # Test normality of data
    normal, _ = test_normality(*args)

    if np.count_nonzero(normal) != normal.size:
        # print('Data are not normally distributed. Using Levene test.')
        _, p = levene(*args)
    else:
        _, p = bartlett(*args)

    equal_var = True if p > alpha else False
    return equal_var, p


def test_dist(*args, dist='norm'):
    """Anderson-Darling test for data coming from a particular distribution.

    Parameters
    ----------
    sample1, sample2,... : array_like
        Array of sample data. May be different lengths.

    Returns
    -------
    from_dist: boolean
        True if data comes from this distribution.
    """
    from scipy.stats import anderson
    # Handle empty input
    for a in args:
        if np.asanyarray(a).size == 0:
            return np.nan, np.nan

    k = len(args)
    from_dist = np.zeros(k, 'bool')
    sig_level = np.zeros(k)
    for j in range(k):
        st, cr, sig = anderson(args[j], dist=dist)
        from_dist[j] = True if (st > cr).any() else False
        sig_level[j] = sig[np.argmin(np.abs(st - cr))]

    if k == 1:
        from_dist = bool(from_dist)
        sig_level = float(sig_level)
    return from_dist, sig_level
