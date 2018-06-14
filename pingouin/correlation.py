# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: May 2018
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from pingouin import (_remove_na, test_normality)

__all__ = ["corr"]


def mahal(Y, X):
    """Mahalanobis distance.

    Equivalent to the Matlab mahal function.

    Parameters
    ----------
    Y : ndarray (shape=(n, m))
        Data
    X : ndarray (shape=(p, m))
        Reference samples

    Returns
    -------
    MD : 1D-array (shape=(n,))
        Squared Mahalanobis distance of each observation in Y to the
        reference samples in X.
    """
    rx, cx = X.shape
    ry, cy = Y.shape

    m = np.mean(X, 0)
    M = np.tile(m, ry).reshape(ry, 2)
    C = X - np.tile(m, rx).reshape(rx, 2)
    Q, R = np.linalg.qr(C)
    ri = np.linalg.solve(R.T, (Y - M).T)
    return np.sum(ri**2, 0) * (rx - 1)


def bsmahal(a, b, j=5000):
    """
    Bootstraps Mahalanobis distances for Shepherd's pi correlation.

    Parameters
    ----------
    a : ndarray (shape=(n, 2))
        Data
    b : ndarray (shape=(n, 2))
        Data
    j : int
        Number of bootstrap samples to calculate.

    Returns
    -------
    m : ndarray (shape=(n,))
        Mahalanobis distance for each row in a, averaged across all the
        bootstrap resamples.
    """
    n = b.shape[0]
    MD = np.zeros((n, j))
    nr = np.arange(n)

    # Bootstrap the MD
    for i in np.arange(j):
        x = np.random.choice(nr, size=n, replace=True)
        s1 = b[x, 0]
        s2 = b[x, 1]
        Y = np.vstack([s1, s2]).T
        m = mahal(a, Y)
        MD[:, i] = m

    # Average across all bootstraps
    return np.mean(MD, 1)


def shepherd(x, y):
    """
    Shepherd's Pi correlation, equivalent to Spearman's rho after outliers
    removal.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.

    Returns
    -------
    Pi : float
        Pi correlation coefficient
    pval : float
        Two-tailed adjusted p-value.

    Notes
    -----
    It first bootstraps the Mahalanobis distances, removes all observations
    with m >= 6 and finally calculates the correlation of the remaining data.

    Pi is Spearman's Rho after outlier removal.

    The p-value is multiplied by 2 to achieve a nominal false alarm rate.
    """
    from scipy.stats import spearmanr

    X = np.vstack([x, y]).T

    # Bootstrapping on Mahalanobis distance
    m = bsmahal(X, X)

    # Determine outliers
    outliers = (m >= 6)

    # Compute correlation
    pi, pval = spearmanr(x[~outliers], y[~outliers])

    # Adjust p-values
    pval *= 2
    pval = 1 if pval > 1 else pval

    return pi, pval


def percbend(x, y, beta=.2):
    """
    Compute the percentage bend correlation (Wilcox 1994).

    Code inspired by Matlab code from Cyril Pernet and Guillaume Rousselet:

    Pernet CR, Wilcox R, Rousselet GA. Robust Correlation Analyses:
    False Positive and Power Validation Using a New Open Source Matlab Toolbox.
    Frontiers in Psychology. 2012;3:606. doi:10.3389/fpsyg.2012.00606.

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
    """
    from scipy.stats import t
    X = np.c_[x, y]
    nx = X.shape[0]
    M = np.tile(np.median(X, axis=0), nx).reshape(X.shape)
    W = np.sort(np.abs(X - M), axis=0)
    m = int((1 - beta) * nx)
    omega = W[m - 1, :]

    # Compute correlation
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

    # Get r, tval and pval
    a, b = a
    r = (a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum())
    tval = r * np.sqrt((nx - 2) / (1 - r**2))
    pval = 2 * t.sf(abs(tval), nx - 2)
    return r, pval


def corr(x, y, tail='two-sided', method='pearson'):
    """(Robust) correlation between two variables.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.
    method : string
        Specify which method to use for the computation of the correlation
        coefficient. Available methods are ::

        'pearson' : Pearson product-moment correlation
        'spearman' : Spearman rank-order correlation
        'kendall' : Kendall’s tau (ordinal data)
        'percbend' : percentage bend correlation (robust)
        'shepherd' : Shepherd's pi correlation (robust Spearman)

    Returns
    -------
    stats : pandas DataFrame
        Test summary ::

        'r' : Correlation coefficient
        'CI95' : 95% parametric confidence intervals
        'r2' : R-squared
        'adj_r2' : Adjusted R-squared
        'p-val' : one or two tailed p-value
        'BF10' : Bayes Factor of the alternative hypothesis (Pearson only)

    Notes
    -----
    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Correlations of -1 or +1 imply
    an exact linear relationship.

    The Spearman correlation is a nonparametric measure of the monotonicity of
    the relationship between two datasets. Unlike the Pearson correlation,
    the Spearman correlation does not assume that both datasets are normally
    distributed. Correlations of -1 or +1 imply an exact monotonic
    relationship.

    Kendall’s tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, values close to -1 indicate
    strong disagreement.

    The percentage bend correlation (Wilcox 1994) is a robust method that
    protects against univariate outliers.

    The Shepherd's pi correlation (Schwarzkopf et al. 2012) is a robust method
    that returns the equivalent of the Spearman's rho after outliers removal.

    Please note that NaN are automatically removed from datasets.

    References
    ----------
    Wilcox, R. R. (1994). The percentage bend correlation
    coefficient. Psychometrika, 59(4), 601-616.

    Schwarzkopf, D. S., De Haas, B., & Rees, G. (2012).
    Better ways to improve standards in brain-behavior correlation analysis.
    Frontiers in human neuroscience, 6, 200.

    Examples
    --------
    1. Pearson correlation

        >>> # Generate random correlated samples
        >>> np.random.seed(123)
        >>> mean, cov = [4, 6], [(1, .5), (.5, 1)]
        >>> x, y = np.random.multivariate_normal(mean, cov, 30).T
        >>> # Compute Pearson correlation
        >>> from pingouin import corr
        >>> corr(x, y)
            method   r      CI95%         r2     adj_r2  p-val   BF10
            pearson  0.491  [0.16, 0.72]  0.242  0.185   0.0058  6.135

    2. Pearson correlation with two outliers

        >>> x[3], y[5] = 12, -8
        >>> corr(x, y)
            method   r      CI95%          r2     adj_r2  p-val  BF10
            pearson  0.147  [-0.23, 0.48]  0.022  -0.051  0.439  0.19

    3. Spearman correlation

        >>> corr(x, y, method="spearman")
            method    r      CI95%         r2     adj_r2  p-val
            spearman  0.401  [0.05, 0.67]  0.161  0.099   0.028

    4. Percentage bend correlation (robust)

        >>> corr(x, y, method='percbend')
            method    r      CI95%         r2     adj_r2  p-val
            percbend  0.389  [0.03, 0.66]  0.151  0.089   0.034

    5. Shepherd's pi correlation (robust)

        >>> corr(x, y, method='shepherd')
            method    r      CI95%         r2     adj_r2  p-val
            percbend  0.437  [0.09, 0.69]  0.191  0.131   0.040

    6. One-tailed Spearman correlation

        >>> corr(x, y, tail="one-sided", method='shepherd')
            method    r      CI95%         r2     adj_r2  p-val
            spearman  0.401  [0.05, 0.67]  0.161  0.099   0.014
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Check size
    if x.size != y.size:
        raise ValueError('x and y must have the same length.')

    # Remove NA
    x, y = _remove_na(x, y, paired=True)
    nx = x.size

    # Compute correlation coefficient
    if method == 'pearson':
        # Test normality of the data
        normal, pnorm = test_normality(x, y)
        if not normal.all():
            print('Warning: data are not normaly distributed (x = %.3f, y =' %
                  pnorm[0], '%.3f). Consider using alternative methods.' %
                  pnorm[1])
        r, pval = pearsonr(x, y)
    elif method == 'spearman':
        r, pval = spearmanr(x, y)
    elif method == 'kendall':
        r, pval = kendalltau(x, y)
    elif method == 'percbend':
        r, pval = percbend(x, y)
    elif method == 'shepherd':
        r, pval = shepherd(x, y)
    else:
        raise ValueError('Method not recognized.')

    # Compute adj_r2
    adj_r2 = 1 - (((1 - r**2) * (nx - 1)) / (nx - 3))

    # Compute the parametric 95% confidence interval
    from pingouin.effsize import compute_esci
    ci = compute_esci(ef=r, nx=nx, ny=nx, eftype='r')

    stats = pd.DataFrame({}, index=[method])
    stats['r'] = np.round(r, 3)
    stats['CI95%'] = [ci]
    stats['r2'] = np.round(r**2, 3)
    stats['adj_r2'] = np.round(adj_r2, 3)
    stats['p-val'] = pval if tail == 'two-sided' else .5 * pval

    # Compute the BF10 for Pearson correlation only
    from pingouin.bayesian import bayesfactor_pearson
    if method == 'pearson':
        stats['BF10'] = bayesfactor_pearson(r, nx)

    col_order = ['r', 'CI95%', 'r2', 'adj_r2', 'p-val', 'BF10']
    stats = stats.reindex(columns=col_order)
    stats.dropna(how='all', axis=1, inplace=True)
    return stats
