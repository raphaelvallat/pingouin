# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: May 2018
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from pingouin import (_check_dataframe, _remove_na, test_normality)

__all__ = ["corr"]


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
    M =  np.tile(np.median(X, axis=0), nx).reshape(X.shape)
    W = np.sort(np.abs(X - M), axis=0)
    m = int((1 - beta) * nx)
    omega = W[m-1, :]

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
        pbos = (np.sum(s) + omega[c] * (i2 - i1)) /  (s.size - i1 - i2)
        a[c] = (X[:, c] - pbos) / omega[c]

    # Bend
    a[a <= -1] = -1
    a[a >= 1] = 1

    # Get r, tval and pval
    a, b = a
    r = (a*b).sum() / np.sqrt((a**2).sum() * (b**2).sum())
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

    Returns
    -------
    stats : pandas DataFrame
        Test summary ::

        'r' : Correlation coefficient
        'r2' : R-squared
        'adj_r2' : Adjusted R-squared
        'p-val' : one or two tailed p-value

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

    Please note that NaN are automatically removed from datasets.

    Examples
    --------
    1. Pearson correlation

        >>> import numpy as np
        >>> from pingouin import corr
        >>> x = [20, 22, 19, 20, 22, 18, 24, 20]
        >>> y = [38, 37, 33, 29, 14, 12, 20, 22]
        >>> corr(x, y)
                    r       r2     adj_r2   p-val
            pearson -0.023  0.001  -0.399   0.957

    2. One-tailed spearman correlation

        >>> import numpy as np
        >>> from pingouin import corr
        >>> x = [20, 22, 19, 20, 22, 18, 24, 20]
        >>> y = [38, 37, 33, 29, 14, 12, 20, 22]
        >>> corr(x, y, method='spearman', tail='one-sided')
                      r       r2     adj_r2  p-val
            spearman  0.0368  0.001  -0.398  0.465

    3. Robust correlation (percentage bend)

        >>> import numpy as np
        >>> from pingouin import corr
        >>> x = [20, 22, 19, 20, 22, 18, 24, 20]
        >>> y = [38, 37, 33, 29, 14, 12, 20, 22]
        >>> corr(x, y, method='percbend')
                     r        r2     adj_r2  p-val
            percbend -0.0411  0.002  -0.397  0.923


    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NA
    x, y = _remove_na(x, y, paired=True)
    nx = x.size
    ny = y.size

    # Compute correlation coefficient
    if method == 'pearson':
        # Test normality of the data
        normal, pnorm = test_normality(x, y)
        if not normal.all():
            print('Warning: data are not normaly distributed (x = %.3f, y =' %
            pnorm[0], '%.3f). Consider using alternative methods.' % pnorm[1])
        r, pval = pearsonr(x, y)
    elif method == 'spearman':
        r, pval = spearmanr(x, y)
    elif method == 'kendall':
        r, pval = kendalltau(x, y)
    elif method == 'percbend':
        r, pval = percbend(x, y)
    else:
        raise ValueError('Method not recognized.')

    # Compute adj_r2
    adj_r2 = 1 - ( ((1 - r**2) * (nx - 1)) / (nx - 3))

    stats = pd.DataFrame({}, index=[method])
    stats['r'] = r.round(4)
    stats['r2'] = r**2
    stats['adj_r2'] = adj_r2
    stats['p-val'] = pval if tail == 'two-sided' else .5 * pval

    col_order = ['r', 'r2', 'adj_r2', 'p-val']
    stats = stats.reindex(columns=col_order)
    return stats
