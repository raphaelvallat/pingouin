import numpy as np
import pandas as pd
from collections import namedtuple
from pingouin.utils import remove_na, _postprocess_dataframe

__all__ = ["multivariate_normality", "multivariate_ttest", "box_m"]


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
    hz : float
        The Henze-Zirkler test statistic.
    pval : float
        P-value.
    normal : boolean
        True if X comes from a multivariate normal distribution.

    See Also
    --------
    normality : Test the univariate normality of one or more variables.
    homoscedasticity : Test equality of variance.
    sphericity : Mauchly's test for sphericity.

    Notes
    -----
    The Henze-Zirkler test [1]_ has a good overall power against alternatives
    to normality and works for any dimension and sample size.

    Adapted to Python from a Matlab code [2]_ by Antonio Trujillo-Ortiz and
    tested against the
    `MVN <https://cran.r-project.org/web/packages/MVN/MVN.pdf>`_ R package.

    Rows with missing values are automatically removed.

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
    >>> pg.multivariate_normality(X, alpha=.05)
    HZResults(hz=0.5400861018514641, pval=0.7173686509624891, normal=True)
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
    S_inv = np.linalg.pinv(S).astype(X.dtype)  # Preserving original dtype
    difT = X - X.mean(0)

    # Squared-Mahalanobis distances
    Dj = np.diag(np.linalg.multi_dot([difT, S_inv, difT.T]))
    Y = np.linalg.multi_dot([X, S_inv, X.T])
    Djk = -2 * Y.T + np.repeat(np.diag(Y.T), n).reshape(n, -1) + \
        np.tile(np.diag(Y.T), (n, 1))

    # Smoothing parameter
    b = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4)**(1 / (p + 4)) * \
        (n**(1 / (p + 4)))

    # Is matrix full-rank (columns are linearly independent)?
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

    HZResults = namedtuple('HZResults', ['hz', 'pval', 'normal'])
    return HZResults(hz=hz, pval=pval, normal=normal)


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
    stats : :py:class:`pandas.DataFrame`

        * ``'T2'``: T-squared value
        * ``'F'``: F-value
        * ``'df1'``: first degree of freedom
        * ``'df2'``: second degree of freedom
        * ``'p-val'``: p-value

    See Also
    --------
    multivariate_normality : Multivariate normality test.
    ttest : Univariate T-test.

    Notes
    -----
    The Hotelling 's T-squared test [1]_ is the multivariate counterpart of
    the T-test.

    Rows with missing values are automatically removed using the
    :py:func:`remove_na` function.

    Tested against the `Hotelling
    <https://cran.r-project.org/web/packages/Hotelling/Hotelling.pdf>`_ R
    package.

    References
    ----------
    .. [1] Hotelling, H. The Generalization of Student's Ratio. Ann. Math.
           Statist. 2 (1931), no. 3, 360--378.

    See also http://www.real-statistics.com/multivariate-statistics/

    Examples
    --------
    Two-sample independent Hotelling T-squared test

    >>> import pingouin as pg
    >>> data = pg.read_dataset('multivariate')
    >>> dvs = ['Fever', 'Pressure', 'Aches']
    >>> X = data[data['Condition'] == 'Drug'][dvs]
    >>> Y = data[data['Condition'] == 'Placebo'][dvs]
    >>> pg.multivariate_ttest(X, Y)
                     T2         F  df1  df2      pval
    hotelling  4.228679  1.326644    3   32  0.282898

    Two-sample paired Hotelling T-squared test

    >>> pg.multivariate_ttest(X, Y, paired=True)
                     T2         F  df1  df2      pval
    hotelling  4.468456  1.314252    3   15  0.306542

    One-sample Hotelling T-squared test with a specified null hypothesis

    >>> null_hypothesis_means = [37.5, 70, 5]
    >>> pg.multivariate_ttest(X, Y=null_hypothesis_means)
                       T2          F  df1  df2          pval
    hotelling  253.230991  74.479703    3   15  3.081281e-09
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
        assert y.ndim in [1, 2], 'Y must be 1D or 2D.'
        if y.ndim == 1:
            # One sample with specified null
            assert y.size == kx
        else:
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
    return _postprocess_dataframe(stats)


def box_m(data, dvs, group, alpha=.001):
    """Test equality of covariance matrices using the Box's M test.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Long-format dataframe.
    dvs : list
        Dependent variables.
    group : str
        Grouping variable.
    alpha : float
        Significance level. Default is 0.001 as recommended in [2]_. A
        non-significant p-value (higher than alpha) indicates that the
        covariance matrices are homogenous (= equal).

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'Chi2'``: Test statistic
        * ``'pval'``: p-value
        * ``'df'``: The Chi-Square statistic's degree of freedom
        * ``'equal_cov'``: True if ``data`` has equal covariance

    Notes
    -----
    .. warning:: Box's M test is susceptible to errors if the data does not
        meet the assumption of multivariate normality or if the sample size is
        too large or small [3]_.

    Pingouin uses :py:meth:`pandas.DataFrameGroupBy.cov` to calculate the
    variance-covariance matrix of each group. Missing values are automatically
    excluded from the calculation by Pandas.

    Mathematical expressions can be found in [1]_.

    This function has been tested against the boxM package of the `biotools`
    R package [4]_.

    References
    ----------
    .. [1] Rencher, A. C. (2003). Methods of multivariate analysis (Vol. 492).
           John Wiley & Sons.

    .. [2] Hahs-Vaughn, D. (2016). Applied Multivariate Statistical Concepts.
           Taylor & Francis.

    .. [3] https://en.wikipedia.org/wiki/Box%27s_M_test

    .. [4] https://cran.r-project.org/web/packages/biotools/index.html

    Examples
    --------
    1. Box M test with 3 dependent variables of 4 groups (equal sample size)

    >>> import pandas as pd
    >>> import pingouin as pg
    >>> from scipy.stats import multivariate_normal as mvn
    >>> data = pd.DataFrame(mvn.rvs(size=(100, 3), random_state=42),
    ...                     columns=['A', 'B', 'C'])
    >>> data['group'] = [1] * 25 + [2] * 25 + [3] * 25 + [4] * 25
    >>> data.head()
              A         B         C  group
    0  0.496714 -0.138264  0.647689      1
    1  1.523030 -0.234153 -0.234137      1
    2  1.579213  0.767435 -0.469474      1
    3  0.542560 -0.463418 -0.465730      1
    4  0.241962 -1.913280 -1.724918      1

    >>> pg.box_m(data, dvs=['A', 'B', 'C'], group='group')
              Chi2    df      pval  equal_cov
    box  11.634185  18.0  0.865537       True

    2. Box M test with 3 dependent variables of 2 groups (unequal sample size)

    >>> data = pd.DataFrame(mvn.rvs(size=(30, 2), random_state=42),
    ...                     columns=['A', 'B'])
    >>> data['group'] = [1] * 20 + [2] * 10
    >>> pg.box_m(data, dvs=['A', 'B'], group='group')
             Chi2   df      pval  equal_cov
    box  0.706709  3.0  0.871625       True
    """
    # Safety checks
    from scipy.stats import chi2
    assert isinstance(data, pd.DataFrame), "data must be a pandas dataframe."
    assert group in data.columns, "The grouping variable is not in data."
    assert set(dvs).issubset(data.columns), "The DVs are not in data."
    grp = data.groupby(group, observed=True)[dvs]
    assert grp.ngroups > 1, 'Data must have at least two columns.'

    # Calculate covariance matrix and descriptive statistics
    # - n_covs is the number of covariance matrices
    # - n_dvs is the number of variables
    # - n_samp is the number of samples in each covariance matrix
    # - nobs is the total number of observations
    covs = grp.cov()
    n_covs, n_dvs = covs.index.levshape
    n_samp = grp.count().iloc[:, 0].to_numpy()  # NaN are excluded by .count
    nobs = n_samp.sum()
    v = n_samp - 1

    # Calculate pooled covariance matrix (S) and M statistics
    covs = covs.to_numpy().reshape(n_covs, n_dvs, -1)
    S = (covs * v[..., None, None]).sum(axis=0) / (nobs - n_covs)
    # The following lines might raise an error if the covariance matrices are
    # not invertible (e.g. missing values in input).
    S_det = np.linalg.det(S)
    M = ((np.linalg.det(covs) / S_det)**(v / 2)).prod()

    # Calculate C in reference [1] (page 257-259)
    if len(np.unique(n_samp)) == 1:
        # All groups have same number of samples
        c = ((n_covs + 1) * (2 * n_dvs ** 2 + 3 * n_dvs - 1)) \
            / (6 * n_covs * (n_dvs + 1) * (nobs / n_covs - 1))
    else:
        # Unequal sample size
        c = (2 * n_dvs ** 2 + 3 * n_dvs - 1) / (6 * (n_dvs + 1) * (n_covs - 1))
        c *= ((1 / v).sum() - 1 / v.sum())

    # Calculate U statistics and degree of fredom
    u = -2 * (1 - c) * np.log(M)
    df = 0.5 * n_dvs * (n_dvs + 1) * (n_covs - 1)
    p = chi2.sf(u, df)
    equal_cov = True if p > alpha else False
    stats = pd.DataFrame(index=["box"], data={
        'Chi2': [u], 'df': [df], 'pval': [p], 'equal_cov': [equal_cov]})
    return _postprocess_dataframe(stats)
