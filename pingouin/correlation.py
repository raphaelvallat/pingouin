# Author: Raphael Vallat <raphaelvallat9@gmail.com>
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from pingouin.power import power_corr
from pingouin.utils import _remove_na
from pingouin.effsize import compute_esci
from pingouin.bayesian import bayesfactor_pearson

from pingouin.nonparametric import mad, madmedianrule

__all__ = ["corr", "partial_corr", "rm_corr", "intraclass_corr"]


def skipped(x, y, method='spearman'):
    """
    Skipped correlation (Rousselet and Pernet 2012).

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    method : str
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
    from pingouin.utils import is_sklearn_installed
    is_sklearn_installed(raise_error=True)
    from scipy.stats import chi2
    from sklearn.covariance import MinCovDet
    X = np.column_stack((x, y))
    center = MinCovDet().fit(X).location_

    # Detect outliers based on robust covariance
    nrows, ncols = X.shape
    gval = np.sqrt(chi2.ppf(0.975, 2))

    # Loop over rows
    record = np.zeros(shape=(nrows, nrows))
    for i in np.arange(nrows):
        dis = np.zeros(nrows)
        B = (X[i, :] - center).T
        bot = np.sum(B**2)
        if bot != 0:
            for j in np.arange(nrows):
                A = X[j, :] - center
                dis[j] = np.linalg.norm(A * B / bot * B)

            # Apply the MAD median rule
            MAD = mad(dis)
            outliers = madmedianrule(dis)
            record[i, :] = dis > (np.median(dis) + gval * MAD)

    outliers = np.sum(record, axis=0) >= 1

    # Compute correlation on remaining data
    if method == 'spearman':
        r, pval = spearmanr(X[~outliers, 0], X[~outliers, 1])
    else:
        r, pval = pearsonr(X[~outliers, 0], X[~outliers, 1])
    return r, pval, outliers


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

    m = X.mean(0)
    M = np.tile(m, ry).reshape(ry, 2)
    C = X - np.tile(m, rx).reshape(rx, 2)
    _, R = np.linalg.qr(C)
    ri = np.linalg.solve(R.T, (Y - M).T)
    return np.sum(ri**2, 0) * (rx - 1)


def bsmahal(a, b, n_boot=2000):
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
    n = b.shape[0]
    MD = np.zeros((n, n_boot))
    nr = np.arange(n)
    xB = np.random.choice(nr, size=(n_boot, n), replace=True)

    # Bootstrap the MD
    for i in np.arange(n_boot):
        s1 = b[xB[i, :], 0]
        s2 = b[xB[i, :], 1]
        Y = np.column_stack((s1, s2))
        MD[:, i] = mahal(a, Y)

    # Average across all bootstraps
    return MD.mean(1)


def shepherd(x, y, n_boot=2000):
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

    Notes
    -----
    It first bootstraps the Mahalanobis distances, removes all observations
    with m >= 6 and finally calculates the correlation of the remaining data.

    Pi is Spearman's Rho after outlier removal.
    """
    from scipy.stats import spearmanr

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

    return r, pval


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
    from scipy.stats import t
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
        'skipped' : skipped correlation (robust Spearman, requires sklearn)

    Returns
    -------
    stats : pandas DataFrame
        Test summary ::

        'n' : Sample size (after NaN removal)
        'r' : Correlation coefficient
        'CI95' : 95% parametric confidence intervals
        'r2' : R-squared
        'adj_r2' : Adjusted R-squared
        'p-val' : one or two tailed p-value
        'BF10' : Bayes Factor of the alternative hypothesis (Pearson only)
        'power' : achieved power of the test (= 1 - type II error).

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

    The percentage bend correlation [1]_ is a robust method that
    protects against univariate outliers.

    The Shepherd's pi [2]_ and skipped [3]_, [4]_ correlations are both robust
    methods that returns the Spearman's rho after bivariate outliers removal.
    Note that the skipped correlation requires that the scikit-learn
    package is installed (for computing the minimum covariance determinant).

    Please note that rows with NaN are automatically removed.

    If method='pearson', The JZS Bayes Factor is approximated using the
    :py:func:`pingouin.bayesfactor_pearson` function.

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

        >>> # Generate random correlated samples
        >>> np.random.seed(123)
        >>> mean, cov = [4, 6], [(1, .5), (.5, 1)]
        >>> x, y = np.random.multivariate_normal(mean, cov, 30).T
        >>> # Compute Pearson correlation
        >>> from pingouin import corr
        >>> corr(x, y)
            method   n   r      CI95%         r2     adj_r2  p-val   BF10
            pearson  30  0.491  [0.16, 0.72]  0.242  0.185   0.0058  6.135

    2. Pearson correlation with two outliers

        >>> x[3], y[5] = 12, -8
        >>> corr(x, y)
            method   n    r      CI95%          r2     adj_r2  p-val  BF10
            pearson  30   0.147  [-0.23, 0.48]  0.022  -0.051  0.439  0.19

    3. Spearman correlation

        >>> corr(x, y, method="spearman")
            method    n   r      CI95%         r2     adj_r2  p-val
            spearman  30  0.401  [0.05, 0.67]  0.161  0.099   0.028

    4. Percentage bend correlation (robust)

        >>> corr(x, y, method='percbend')
            method    n   r      CI95%         r2     adj_r2  p-val
            percbend  30  0.389  [0.03, 0.66]  0.151  0.089   0.034

    5. Shepherd's pi correlation (robust)

        >>> corr(x, y, method='shepherd')
            method    n   r      CI95%         r2     adj_r2  p-val
            percbend  30  0.437  [0.09, 0.69]  0.191  0.131   0.020

    6. Skipped spearman correlation (robust)

        >>> corr(x, y, method='skipped')
            method    n   r      CI95%         r2     adj_r2  p-val
            percbend  30  0.437  [0.09, 0.69]  0.191  0.131   0.020

    7. One-tailed Spearman correlation

        >>> corr(x, y, tail="one-sided", method='shepherd')
            method    n   r      CI95%         r2     adj_r2  p-val
            spearman  30  0.401  [0.05, 0.67]  0.161  0.099   0.014

    8. Using columns of a pandas dataframe

        >>> import pandas as pd
        >>> data = pd.DataFrame({'x': x, 'y': y})
        >>> corr(data['x'], data['y'])
            method   n    r      CI95%          r2     adj_r2  p-val  BF10
            pearson  30   0.147  [-0.23, 0.48]  0.022  -0.051  0.439  0.19
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
        r, pval = pearsonr(x, y)
    elif method == 'spearman':
        r, pval = spearmanr(x, y)
    elif method == 'kendall':
        r, pval = kendalltau(x, y)
    elif method == 'percbend':
        r, pval = percbend(x, y)
    elif method == 'shepherd':
        r, pval = shepherd(x, y)
    elif method == 'skipped':
        r, pval, _ = skipped(x, y, method='spearman')
    else:
        raise ValueError('Method not recognized.')

    # Compute adj_r2
    adj_r2 = 1 - (((1 - r**2) * (nx - 1)) / (nx - 3))

    # Compute the parametric 95% confidence interval
    ci = compute_esci(stat=r, nx=nx, ny=nx, eftype='r')

    stats = pd.DataFrame({}, index=[method])
    stats['n'] = nx
    stats['r'] = np.round(r, 3)
    stats['CI95%'] = [ci]
    stats['r2'] = np.round(r**2, 3)
    stats['adj_r2'] = np.round(adj_r2, 3)
    stats['p-val'] = pval if tail == 'two-sided' else .5 * pval

    # Compute achieved power
    if not np.isnan(r):
        stats['power'] = np.round(power_corr(r=r, n=nx, power=None,
                                             alpha=0.05, tail=tail), 3)

    # Compute the BF10 for Pearson correlation only
    if method == 'pearson' and nx < 1000 and not np.isnan(r):
        stats['BF10'] = bayesfactor_pearson(r, nx)

    col_order = ['n', 'r', 'CI95%', 'r2', 'adj_r2', 'p-val', 'BF10', 'power']
    stats = stats.reindex(columns=col_order)
    stats.dropna(how='all', axis=1, inplace=True)
    return stats


def partial_corr(data=None, x=None, y=None, covar=None, tail='two-sided',
                 method='pearson'):
    """Partial correlation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe
    x, y : string
        x and y. Must be names of columns in data.
    covar : string or list
        Covariate(s). Must be a names of columns in data. Use a list if there
        are more than one covariate.
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
        'skipped' : skipped correlation (robust Spearman, requires sklearn)

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
    Partial correlation is a measure of the strength and direction of a
    linear relationship between two continuous variables whilst controlling
    for the effect of one or more other continuous variables
    (also known as covariates or control variables).

    Inspired from a code found at:
    https://gist.github.com/fabianp/9396204419c7b638d38f

    Results have been tested against the ppcor R package.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Partial_correlation

    .. [2] https://cran.r-project.org/web/packages/ppcor/index.html

    Examples
    --------
    1. Partial correlation with one covariate

        >>> import numpy as np
        >>> import pandas as pd
        >>> from pingouin import partial_corr
        >>> # Generate random correlated samples
        >>> np.random.seed(123)
        >>> mean, cov = [4, 6, 2], [(1, .5, .3), (.5, 1, .2), (.3, .2, 1)]
        >>> x, y, z = np.random.multivariate_normal(mean, cov, size=30).T
        >>> # Append in a dataframe
        >>> df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        >>> # Partial correlation of x and y controlling for z
        >>> partial_corr(data=df, x='x', y='y', covar='z')
            method   r      CI95%         r2     adj_r2  p-val   BF10
            pearson  0.568  [0.26, 0.77]  0.323  0.273   0.0010  28.695

    2. Partial correlation with several covariates

        >>> # Add new random columns to the dataframe of the first example
        >>> np.random.seed(123)
        >>> df['w'] = np.random.normal(size=30)
        >>> df['v'] = np.random.normal(size=30)
        >>> # Partial correlation of x and y controlling for z, w and v
        >>> partial_corr(data=df, x='x', y='y', covar=['z', 'w', 'v'])
            method   r      CI95%         r2     adj_r2  p-val   BF10
            pearson  0.493  [0.16, 0.72]  0.243  0.187   0.0056  6.258
    """
    # Check arguments
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert isinstance(covar, (str, list))
    assert isinstance(data, pd.DataFrame)
    # Check that columns exist
    if isinstance(covar, str):
        col = [x] + [y] + [covar]
        n_cvr = 1
    if isinstance(covar, list):
        col = [x] + [y] + covar
        n_cvr = len(covar)
        covar = covar[0] if n_cvr == 1 else covar
    assert all([c in data for c in col])
    # Check that columns are numeric
    assert all([data[c].dtype.kind in 'bfi' for c in col])

    # Standardize
    C = (data[col] - data[col].mean(axis=0)) / data[col].std(axis=0)

    # Covariates
    cvar = C[covar].values[..., np.newaxis] if n_cvr == 1 else C[covar].values

    # Compute beta
    beta_x = np.linalg.lstsq(cvar, C[y].values, rcond=None)[0]
    beta_y = np.linalg.lstsq(cvar, C[x].values, rcond=None)[0]

    # Compute residuals
    res_x = C[x].values - np.dot(cvar, beta_y)
    res_y = C[y].values - np.dot(cvar, beta_x)

    # Partial correlation = corr between these residuals
    return corr(res_x, res_y, method=method, tail=tail)


def rm_corr(data=None, x=None, y=None, subject=None, tail='two-sided'):
    """Repeated measures correlation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the variables
    x, y : string
        Name of columns in data containing the two dependent variables
    subject : string
        Name of column in data containing the subject indicator
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.

    Returns
    -------
    r : float
        Repeated measures correlation coefficient
    p : float
        P-value
    dof : int
        Degrees of freedom

    Notes
    -----
    Repeated measures correlation [1]_ is a statistical technique
    for determining the common within-individual association for paired
    measures assessed on two or more occasions for multiple individuals.

    Results have been tested against the `rmcorr` R package.

    Please note that NaN are automatically removed from the dataframe.

    References
    ----------

    .. [1] Bakdash, J.Z., Marusich, L.R., 2017. Repeated Measures Correlation.
       Front. Psychol. 8, 456. https://doi.org/10.3389/fpsyg.2017.00456

    Examples
    --------
    1. Repeated measures correlation

        >>> from pingouin import rm_corr
        >>> from pingouin.datasets import read_dataset
        >>> df = read_dataset('rm_corr')
        >>> # Compute the repeated measure correlation
        >>> rm_corr(data=df, x='pH', y='PacO2', subject='Subject')
            (-0.507, 0.0008, 38)
    """
    # Remove Nans
    data = data[[x, y, subject]].dropna(axis=0)

    # Using STATSMODELS
    # from pingouin.utils import is_statsmodels_installed
    # is_statsmodels_installed(raise_error=True)
    # from statsmodels.api import stats
    # from statsmodels.formula.api import ols
    # # ANCOVA model
    # formula = y + ' ~ ' + 'C(' + subject + ') + ' + x
    # model = ols(formula, data=data).fit()
    # table = stats.anova_lm(model, typ=3)
    # # Extract the sign of the correlation and dof
    # sign = np.sign(model.params[x])
    # dof = int(table.loc['Residual', 'df'])
    # # Extract correlation coefficient from sum of squares
    # ssfactor = table.loc[x, 'sum_sq']
    # sserror = table.loc['Residual', 'sum_sq']
    # rm = sign * np.sqrt(ssfactor / (ssfactor + sserror))
    # # Extract p-value
    # pval = table.loc[x, 'PR(>F)']
    # pval *= 0.5 if tail == 'one-sided' else 1

    # Using PINGOUIN
    from pingouin import ancova
    aov, bw = ancova(dv=y, covar=x, between=subject, data=data,
                     return_bw=True)
    sign = np.sign(bw)
    dof = int(aov.loc[2, 'DF'])
    ssfactor = aov.loc[1, 'SS']
    sserror = aov.loc[2, 'SS']
    rm = sign * np.sqrt(ssfactor / (ssfactor + sserror))
    pval = aov.loc[1, 'p-unc']
    pval *= 0.5 if tail == 'one-sided' else 1

    return np.round(rm, 3), pval, dof


def intraclass_corr(data=None, groups=None, raters=None, scores=None, ci=.95):
    """Intra-class correlation coefficient.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the variables
    groups : string
        Name of column in data containing the groups.
    raters : string
        Name of column in data containing the raters (scorers).
    scores : string
        Name of column in data containing the scores (ratings).
    ci : float
        Confidence interval

    Returns
    -------
    icc : float
        Intraclass correlation coefficient
    ci : list
        Lower and upper confidence intervals

    Notes
    -----
    The intraclass correlation (ICC) assesses the reliability of ratings by
    comparing the variability of different ratings of the same subject to the
    total variation across all ratings and all subjects. The ratings are
    quantitative (e.g. Likert scale).

    Inspired from:
    http://www.real-statistics.com/reliability/intraclass-correlation/

    Examples
    --------
    1. ICC of wine quality assessed by 4 judges.

        >>> from pingouin.datasets import read_dataset
        >>> from pingouin import intraclass_corr
        >>> data = read_dataset('icc')
        >>> intraclass_corr(data, 'Wine', 'Judge', 'Scores')
            (0.727525596259691, array([0.434, 0.927]))

    """
    from pingouin import anova
    from scipy.stats import f

    # Check dataframe
    if any(v is None for v in [data, groups, raters, scores]):
        raise ValueError('Data, groups, raters and scores must be specified')
    if not isinstance(data, pd.DataFrame):
        raise ValueError('Data must be a pandas dataframe.')
    # Check that scores is a numeric variable
    if data[scores].dtype.kind not in 'fi':
        raise ValueError('Scores must be numeric.')
    # Check that data are fully balanced
    if data.groupby(raters)[scores].count().unique().size > 1:
        raise ValueError('Data must be balanced.')

    # Extract sizes
    k = data[raters].unique().size
    # n = data[groups].unique().size

    # ANOVA and ICC
    aov = anova(dv=scores, data=data, between=groups, detailed=True)
    icc = (aov.loc[0, 'MS'] - aov.loc[1, 'MS']) / \
          (aov.loc[0, 'MS'] + (k - 1) * aov.loc[1, 'MS'])

    # Confidence interval
    alpha = 1 - ci
    df_num, df_den = aov.loc[0, 'DF'], aov.loc[1, 'DF']
    f_lower = aov.loc[0, 'F'] / f.isf(alpha / 2, df_num, df_den)
    f_upper = aov.loc[0, 'F'] * f.isf(alpha / 2, df_den, df_num)
    lower = (f_lower - 1) / (f_lower + k - 1)
    upper = (f_upper - 1) / (f_upper + k - 1)

    return icc, np.round([lower, upper], 3)
