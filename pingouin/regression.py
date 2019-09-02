import itertools
import numpy as np
import pandas as pd
from scipy.stats import t, norm
import pandas_flavor as pf
from pingouin.utils import remove_na as rm_na
from pingouin.utils import _flatten_list as _fl

__all__ = ['linear_regression', 'logistic_regression', 'mediation_analysis']


def linear_regression(X, y, add_intercept=True, coef_only=False, alpha=0.05,
                      as_dataframe=True, remove_na=False):
    """(Multiple) Linear regression.

    Parameters
    ----------
    X : pd.DataFrame, np.array or list
        Predictor(s). Shape = (n_samples, n_features) or (n_samples,).
    y : pd.Series, np.array or list
        Dependent variable. Shape = (n_samples).
    add_intercept : bool
        If False, assume that the data are already centered. If True, add a
        constant term to the model. In this case, the first value in the
        output dict is the intercept of the model.

        .. note:: It is generally recommanded to include a constant term
            (intercept) to the model to limit the bias and force the residual
            mean to equal zero. Note that intercept coefficient and p-values
            are however rarely meaningful.
    coef_only : bool
        If True, return only the regression coefficients.
    alpha : float
        Alpha value used for the confidence intervals.
        :math:`\\text{CI} = [\\alpha / 2 ; 1 - \\alpha / 2]`
    as_dataframe : bool
        If True, returns a pandas DataFrame. If False, returns a dictionnary.
    remove_na : bool
        If True, apply a listwise deletion of missing values (i.e. the entire
        row is removed). Default is False, which will raise an error if missing
        values are present in either the predictor(s) or dependent
        variable.

    Returns
    -------
    stats : dataframe or dict
        Linear regression summary::

        'names' : name of variable(s) in the model (e.g. x1, x2...)
        'coef' : regression coefficients
        'se' : standard error of the estimate
        'T' : T-values
        'pval' : p-values
        'r2' : coefficient of determination (R2)
        'adj_r2' : adjusted R2
        'CI[2.5%]' : lower confidence interval
        'CI[97.5%]' : upper confidence interval
        'residuals' : residuals (only if as_dataframe is False)

    See also
    --------
    logistic_regression, mediation_analysis, corr

    Notes
    -----
    The :math:`\\beta` coefficients are estimated using an ordinary least
    squares (OLS) regression, as implemented in the
    :py:func:`numpy.linalg.lstsq` function. The OLS method minimizes
    the sum of squared residuals, and leads to a closed-form expression for
    the estimated :math:`\\beta`:

    .. math:: \\hat{\\beta} = (X^TX)^{-1} X^Ty

    It is generally recommanded to include a constant term (intercept) to the
    model to limit the bias and force the residual mean to equal zero.
    Note that intercept coefficient and p-values are however rarely meaningful.

    The standard error of the estimates is a measure of the accuracy of the
    prediction defined as:

    .. math:: \\sigma = \\sqrt{\\text{MSE} \\cdot (X^TX)^{-1}}

    where :math:`\\text{MSE}` is the mean squared error,

    .. math::

        \\text{MSE} = \\frac{SS_{\\text{resid}}}{n - p - 1}
         = \\frac{\\sum{(\\text{true} - \\text{pred})^2}}{n - p - 1}

    :math:`p` is the total number of predictor variables in the model
    (excluding the intercept) and :math:`n` is the sample size.

    Using the :math:`\\beta` coefficients and the standard errors,
    the T-values can be obtained:

    .. math:: T = \\frac{\\beta}{\\sigma}

    and the p-values approximated using a T-distribution with
    :math:`n - p - 1` degrees of freedom.

    The coefficient of determination (:math:`R^2`) is defined as:

    .. math:: R^2 = 1 - (\\frac{SS_{\\text{resid}}}{SS_{\\text{total}}})

    The adjusted :math:`R^2` is defined as:

    .. math:: \\overline{R}^2 = 1 - (1 - R^2) \\frac{n - 1}{n - p - 1}

    The residuals can be accessed via :code:`stats.residuals_` if ``stats``
    is a pandas DataFrame or :code:`stats['residuals']` if ``stats`` is a
    dict.

    Note that Pingouin will automatically remove any duplicate columns
    from :math:`X`, as well as any column with only one unique value
    (constant), excluding the intercept.

    Results have been compared against sklearn, statsmodels and JASP.

    Examples
    --------
    1. Simple linear regression

    >>> import numpy as np
    >>> from pingouin import linear_regression
    >>> np.random.seed(123)
    >>> mean, cov, n = [4, 6], [[1, 0.5], [0.5, 1]], 30
    >>> x, y = np.random.multivariate_normal(mean, cov, n).T
    >>> lm = linear_regression(x, y)
    >>> lm.round(2)
           names  coef    se     T  pval    r2  adj_r2  CI[2.5%]  CI[97.5%]
    0  Intercept  4.40  0.54  8.16  0.00  0.24    0.21      3.29       5.50
    1         x1  0.39  0.13  2.99  0.01  0.24    0.21      0.12       0.67

    2. Multiple linear regression

    >>> np.random.seed(42)
    >>> z = np.random.normal(size=n)
    >>> X = np.column_stack((x, z))
    >>> lm = linear_regression(X, y)
    >>> print(lm['coef'].values)
    [4.54123324 0.36628301 0.17709451]

    3. Get the residuals

    >>> np.round(lm.residuals_, 2)
    array([ 1.18, -1.17,  1.32,  0.76, -1.25,  0.34, -1.54, -0.2 ,  0.36,
           -0.39,  0.69,  1.39,  0.2 , -1.14, -0.21, -1.68,  0.67, -0.69,
            0.62,  0.92, -1.  ,  0.64, -0.21, -0.78,  1.08, -0.03, -1.3 ,
            0.64,  0.81, -0.04])

    4. Using a Pandas DataFrame

    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    >>> lm = linear_regression(df[['x', 'z']], df['y'])
    >>> print(lm['coef'].values)
    [4.54123324 0.36628301 0.17709451]

    5. No intercept and return coef only

    >>> linear_regression(X, y, add_intercept=False, coef_only=True)
    array([ 1.40935593, -0.2916508 ])

    6. Return a dictionnary instead of a DataFrame

    >>> lm_dict = linear_regression(X, y, as_dataframe=False)

    7. Remove missing values

    >>> X[4, 1] = np.nan
    >>> y[7] = np.nan
    >>> linear_regression(X, y, remove_na=True, coef_only=True)
    array([4.64069731, 0.35455398, 0.1888135 ])
    """
    # Extract names if X is a Dataframe or Series
    if isinstance(X, pd.DataFrame):
        names = X.keys().tolist()
    elif isinstance(X, pd.Series):
        names = [X.name]
    else:
        names = []

    # Convert input to numpy array
    X = np.asarray(X)
    y = np.asarray(y)
    assert y.ndim == 1, 'y must be one-dimensional.'
    assert 0 < alpha < 1

    if X.ndim == 1:
        # Convert to (n_samples, n_features) shape
        X = X[..., np.newaxis]

    # Check for NaN / Inf
    if remove_na:
        X, y = rm_na(X, y[..., np.newaxis], paired=True, axis='rows')
        y = np.squeeze(y)
    y_gd = np.isfinite(y).all()
    X_gd = np.isfinite(X).all()
    assert y_gd, ("Target (y) contains NaN or Inf. Please remove them "
                  "manually or use remove_na=True.")
    assert X_gd, ("Predictors (X) contain NaN or Inf. Please remove them "
                  "manually or use remove_na=True.")

    # Check that X and y have same length
    assert y.shape[0] == X.shape[0], 'X and y must have same number of samples'

    if not names:
        names = ['x' + str(i + 1) for i in range(X.shape[1])]

    if add_intercept:
        # Add intercept
        X = np.column_stack((np.ones(X.shape[0]), X))
        names.insert(0, "Intercept")

    # FINAL CHECKS BEFORE RUNNING LEAST SQUARES REGRESSION
    # 1. Let's remove the column with only zero, otherwise the regression fails
    n_nonzero = np.count_nonzero(X, axis=0)
    idx_zero = np.flatnonzero(n_nonzero == 0)  # Find columns that are only 0
    if len(idx_zero):
        X = np.delete(X, idx_zero, 1)
        names = np.delete(names, idx_zero)

    # 2. We also want to make sure that there is no more than one column
    # (= Intercept) with only one unique value, otherwise the regression fails
    # This is equivalent, but much faster, to pd.DataFrame(X).nunique()
    idx_unique = np.where(np.all(X == X[0, :], axis=0))[0]
    if len(idx_unique) > 1:
        # Houston, we have a problem!
        # We remove all but the first "Intercept" column.
        X = np.delete(X, idx_unique[1:], 1)
        names = np.delete(names, idx_unique[1:])

    # 3. Finally, we want to remove duplicate columns
    if X.shape[1] > 1:
        idx_duplicate = []
        for pair in itertools.combinations(range(X.shape[1]), 2):
            if np.array_equal(X[:, pair[0]], X[:, pair[1]]):
                idx_duplicate.append(pair[1])
        if len(idx_duplicate):
            X = np.delete(X, idx_duplicate, 1)
            names = np.delete(names, idx_duplicate)

    # LEAST-SQUARE REGRESSION + STATISTICS
    # Compute beta coefficient and predictions
    n, p = X.shape[0], X.shape[1]
    assert n >= 3, 'At least three valid samples are required in X.'
    assert p >= 1, 'X must have at least one valid column.'
    coef, ss_res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    if coef_only:
        return coef
    ss_res = np.squeeze(ss_res)
    pred = np.dot(X, coef)
    resid = y - pred
    # Degrees of freedom should not include the intercept
    dof = n - p if add_intercept else n - p - 1
    # Compute mean squared error, variance and SE
    MSE = ss_res / dof
    beta_var = MSE * (np.linalg.pinv(np.dot(X.T, X)).diagonal())
    beta_se = np.sqrt(beta_var)

    # Compute R2, adjusted R2 and RMSE
    ss_tot = np.square(y - y.mean()).sum()
    # ss_exp = np.square(pred - y.mean()).sum()
    r2 = 1 - (ss_res / ss_tot)
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof

    # Compute T and p-values
    T = coef / beta_se
    pval = 2 * t.sf(np.fabs(T), dof)

    # Compute confidence intervals
    crit = t.ppf(1 - alpha / 2, dof)
    marg_error = crit * beta_se
    ll = coef - marg_error
    ul = coef + marg_error

    # Rename CI
    ll_name = 'CI[%.1f%%]' % (100 * alpha / 2)
    ul_name = 'CI[%.1f%%]' % (100 * (1 - alpha / 2))

    # Create dict
    stats = {'names': names, 'coef': coef, 'se': beta_se, 'T': T,
             'pval': pval, 'r2': r2, 'adj_r2': adj_r2, ll_name: ll,
             ul_name: ul}

    if as_dataframe:
        stats = pd.DataFrame(stats)
        stats.residuals_ = 0  # Trick to avoid Pandas warning
        stats.residuals_ = resid  # Residuals is a hidden attribute
    else:
        stats['residuals'] = resid
    return stats


def logistic_regression(X, y, coef_only=False, alpha=0.05,
                        as_dataframe=True, remove_na=False, **kwargs):
    """(Multiple) Binary logistic regression.

    Parameters
    ----------
    X : np.array or list
        Predictor(s). Shape = (n_samples, n_features) or (n_samples,).
    y : np.array or list
        Dependent variable. Shape = (n_samples).
        Must be binary.
    coef_only : bool
        If True, return only the regression coefficients.
    alpha : float
        Alpha value used for the confidence intervals.
        :math:`\\text{CI} = [\\alpha / 2 ; 1 - \\alpha / 2]`
    as_dataframe : bool
        If True, returns a pandas DataFrame. If False, returns a dictionnary.
    remove_na : bool
        If True, apply a listwise deletion of missing values (i.e. the entire
        row is removed). Default is False, which will raise an error if missing
        values are present in either the predictor(s) or dependent
        variable.
    **kwargs : optional
        Optional arguments passed to
        :py:class:`sklearn.linear_model.LogisticRegression`.

    Returns
    -------
    stats : dataframe or dict
        Logistic regression summary::

        'names' : name of variable(s) in the model (e.g. x1, x2...)
        'coef' : regression coefficients
        'se' : standard error
        'z' : z-scores
        'pval' : two-tailed p-values
        'CI[2.5%]' : lower confidence interval
        'CI[97.5%]' : upper confidence interval

    See also
    --------
    linear_regression

    Notes
    -----
    This is a wrapper around the
    :py:class:`sklearn.linear_model.LogisticRegression` class. Note that
    Pingouin automatically disables the l2 regularization applied by
    scikit-learn. This can be modified by changing the `penalty` argument.

    The calculation of the p-values and confidence interval is adapted from a
    code found at
    https://gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d

    Note that the first coefficient is always the constant term (intercept) of
    the model. Scikit-learn will automatically add the intercept
    to your predictor(s) matrix, therefore, :math:`X` should not include a
    constant term. Pingouin will remove any constant term (e.g column with only
    one unique value), or duplicate columns from :math:`X`.

    Results have been compared against statsmodels, R, and JASP.

    Examples
    --------
    1. Simple binary logistic regression

    >>> import numpy as np
    >>> from pingouin import logistic_regression
    >>> np.random.seed(123)
    >>> x = np.random.normal(size=30)
    >>> y = np.random.randint(0, 2, size=30)
    >>> lom = logistic_regression(x, y)
    >>> lom.round(2)
           names  coef    se     z  pval  CI[2.5%]  CI[97.5%]
    0  Intercept -0.27  0.37 -0.74  0.46     -1.00       0.45
    1         x1  0.07  0.32  0.21  0.84     -0.55       0.68

    2. Multiple binary logistic regression

    >>> np.random.seed(42)
    >>> z = np.random.normal(size=30)
    >>> X = np.column_stack((x, z))
    >>> lom = logistic_regression(X, y)
    >>> print(lom['coef'].values)
    [-0.36736745 -0.04374684 -0.47829392]

    3. Using a Pandas DataFrame

    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    >>> lom = logistic_regression(df[['x', 'z']], df['y'])
    >>> print(lom['coef'].values)
    [-0.36736745 -0.04374684 -0.47829392]

    4. Return only the coefficients

    >>> logistic_regression(X, y, coef_only=True)
    array([-0.36736745, -0.04374684, -0.47829392])

    5. Passing custom parameters to sklearn

    >>> lom = logistic_regression(X, y, solver='sag', max_iter=10000,
    ...                           random_state=42)
    >>> print(lom['coef'].values)
    [-0.36751796 -0.04367056 -0.47841908]
    """
    # Check that sklearn is installed
    from pingouin.utils import _is_sklearn_installed
    _is_sklearn_installed(raise_error=True)
    from sklearn.linear_model import LogisticRegression

    # Extract names if X is a Dataframe or Series
    if isinstance(X, pd.DataFrame):
        names = X.keys().tolist()
    elif isinstance(X, pd.Series):
        names = [X.name]
    else:
        names = []

    # Convert to numpy array
    X = np.asarray(X)
    y = np.asarray(y)
    assert y.ndim == 1, 'y must be one-dimensional.'
    assert 0 < alpha < 1, 'alpha must be between 0 and 1.'

    # Add axis if only one-dimensional array
    if X.ndim == 1:
        X = X[..., np.newaxis]

    # Check for NaN /  Inf
    if remove_na:
        X, y = rm_na(X, y[..., np.newaxis], paired=True, axis='rows')
        y = np.squeeze(y)
    y_gd = np.isfinite(y).all()
    X_gd = np.isfinite(X).all()
    assert y_gd, ("Target (y) contains NaN or Inf. Please remove them "
                  "manually or use remove_na=True.")
    assert X_gd, ("Predictors (X) contain NaN or Inf. Please remove them "
                  "manually or use remove_na=True.")

    # Check that X and y have same length
    assert y.shape[0] == X.shape[0], 'X and y must have same number of samples'

    # Check that y is binary
    if np.unique(y).size != 2:
        raise ValueError('Dependent variable must be binary.')

    if not names:
        names = ['x' + str(i + 1) for i in range(X.shape[1])]

    # We also want to make sure that there is no column
    # with only one unique value, otherwise the regression fails
    # This is equivalent, but much faster, to pd.DataFrame(X).nunique()
    idx_unique = np.where(np.all(X == X[0, :], axis=0))[0]
    if len(idx_unique):
        X = np.delete(X, idx_unique, 1)
        names = np.delete(names, idx_unique).tolist()

    # Finally, we want to remove duplicate columns
    if X.shape[1] > 1:
        idx_duplicate = []
        for pair in itertools.combinations(range(X.shape[1]), 2):
            if np.array_equal(X[:, pair[0]], X[:, pair[1]]):
                idx_duplicate.append(pair[1])
        if len(idx_duplicate):
            X = np.delete(X, idx_duplicate, 1)
            names = np.delete(names, idx_duplicate).tolist()

    # Initialize and fit
    if 'solver' not in kwargs:
        kwargs['solver'] = 'lbfgs'
    if 'multi_class' not in kwargs:
        kwargs['multi_class'] = 'auto'
    if 'penalty' not in kwargs:
        kwargs['penalty'] = 'none'
    lom = LogisticRegression(**kwargs)
    lom.fit(X, y)
    coef = np.append(lom.intercept_, lom.coef_)
    if coef_only:
        return coef

    # Design matrix -- add intercept
    names.insert(0, "Intercept")
    X_design = np.column_stack((np.ones(X.shape[0]), X))
    n, p = X_design.shape

    # Fisher Information Matrix
    denom = (2 * (1 + np.cosh(lom.decision_function(X))))
    denom = np.tile(denom, (p, 1)).T
    fim = np.dot((X_design / denom).T, X_design)
    crao = np.linalg.pinv(fim)

    # Standard error and Z-scores
    se = np.sqrt(np.diag(crao))
    z_scores = coef / se

    # Two-tailed p-values
    pval = 2 * norm.sf(np.fabs(z_scores))

    # Confidence intervals
    crit = norm.ppf(1 - alpha / 2)
    ll = coef - crit * se
    ul = coef + crit * se

    # Rename CI
    ll_name = 'CI[%.1f%%]' % (100 * alpha / 2)
    ul_name = 'CI[%.1f%%]' % (100 * (1 - alpha / 2))

    # Create dict
    stats = {'names': names, 'coef': coef, 'se': se, 'z': z_scores,
             'pval': pval, ll_name: ll, ul_name: ul}
    if as_dataframe:
        return pd.DataFrame(stats)
    else:
        return stats


def _point_estimate(X_val, XM_val, M_val, y_val, idx, n_mediator,
                    mtype='linear'):
    """Point estimate of indirect effect based on bootstrap sample."""
    # Mediator(s) model (M(j) ~ X + covar)
    beta_m = []
    for j in range(n_mediator):
        if mtype == 'linear':
            beta_m.append(linear_regression(X_val[idx], M_val[idx, j],
                                            coef_only=True)[1])
        else:
            beta_m.append(logistic_regression(X_val[idx], M_val[idx, j],
                                              coef_only=True)[1])

    # Full model (Y ~ X + M + covar)
    beta_y = linear_regression(XM_val[idx], y_val[idx],
                               coef_only=True)[2:(2 + n_mediator)]

    # Point estimate
    return beta_m * beta_y


def _bca(ab_estimates, sample_point, n_boot, alpha=0.05):
    """Get (1 - alpha) * 100 bias-corrected confidence interval estimate

    Note that this is similar to the "cper" module implemented in
    :py:func:`pingouin.compute_bootci`.

    Parameters
    ----------
    ab_estimates : 1d array-like
        Array with bootstrap estimates for each sample.
    sample_point : float
        Indirect effect point estimate based on full sample.
    n_boot : int
        Number of bootstrap samples
    alpha : float
        Alpha for confidence interval

    Returns
    -------
    CI : 1d array-like
        Lower limit and upper limit bias-corrected confidence interval
        estimates.
    """
    # Bias of bootstrap estimates
    z0 = norm.ppf(np.sum(ab_estimates < sample_point) / n_boot)

    # Adjusted intervals
    adjusted_ll = norm.cdf(2 * z0 + norm.ppf(alpha / 2)) * 100
    adjusted_ul = norm.cdf(2 * z0 + norm.ppf(1 - alpha / 2)) * 100
    ll = np.percentile(ab_estimates, q=adjusted_ll)
    ul = np.percentile(ab_estimates, q=adjusted_ul)
    return np.array([ll, ul])


def _pval_from_bootci(boot, estimate):
    """Compute p-value from bootstrap distribution.
    Similar to the pval function in the R package mediation.
    Note that this is less accurate than a permutation test because the
    bootstrap distribution is not conditioned on a true null hypothesis.
    """
    if estimate == 0:
        out = 1
    else:
        out = 2 * min(sum(boot > 0), sum(boot < 0)) / len(boot)
    return min(out, 1)


@pf.register_dataframe_method
def mediation_analysis(data=None, x=None, m=None, y=None, covar=None,
                       alpha=0.05, n_boot=500, seed=None, return_dist=False):
    """Mediation analysis using a bias-correct non-parametric bootstrap method.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe.
    x : str
        Column name in data containing the predictor variable.
        The predictor variable must be continuous.
    m : str or list of str
        Column name(s) in data containing the mediator variable(s).
        The mediator(s) can be continuous or binary (e.g. 0 or 1).
        This function supports multiple parallel mediators.
    y : str
        Column name in data containing the outcome variable.
        The outcome variable must be continuous.
    covar : None, str, or list
        Covariate(s). If not None, the specified covariate(s) will be included
        in all regressions.
    alpha : float
        Significance threshold. Used to determine the confidence interval,
        :math:`\\text{CI} = [\\alpha / 2 ; 1 - \\alpha / 2]`.
    n_boot : int
        Number of bootstrap iterations for confidence intervals and p-values
        estimation. The greater, the slower.
    seed : int or None
        Random state seed.
    return_dist : bool
        If True, the function also returns the indirect bootstrapped beta
        samples (size = n_boot). Can be plotted for instance using
        :py:func:`seaborn.distplot()` or :py:func:`seaborn.kdeplot()`
        functions.

    Returns
    -------
    stats : pd.DataFrame
        Mediation summary::

        'path' : regression model
        'coef' : regression estimates
        'se' : standard error
        'CI[2.5%]' : lower confidence interval
        'CI[97.5%]' : upper confidence interval
        'pval' : two-sided p-values
        'sig' : statistical significance

    See also
    --------
    linear_regression, logistic_regression

    Notes
    -----
    Mediation analysis is a "statistical procedure to test
    whether the effect of an independent variable X on a dependent variable
    Y (i.e., X → Y) is at least partly explained by a chain of effects of the
    independent variable on an intervening mediator variable M and of the
    intervening variable on the dependent variable (i.e., X → M → Y)"
    (from Fiedler et al. 2011).

    The **indirect effect** (also referred to as average causal mediation
    effect or ACME) of X on Y through mediator M quantifies the estimated
    difference in Y resulting from a one-unit change in X through a sequence of
    causal steps in which X affects M, which in turn affects Y.
    It is considered significant if the specified confidence interval does not
    include 0. The path 'X --> Y' is the sum of both the indirect and direct
    effect. It is sometimes referred to as total effect. For more details,
    please refer to Fiedler et al 2011 or Hayes and Rockwood 2017.

    A linear regression is used if the mediator variable is continuous and a
    logistic regression if the mediator variable is dichotomous (binary). Note
    that this function also supports parallel multiple mediators: "in such
    models, mediators may be and often are correlated, but nothing in the
    model allows one mediator to causally influence another."
    (Hayes and Rockwood 2017)

    This function wll only work well if the outcome variable is continuous.
    It does not support binary or ordinal outcome variable. For more
    advanced mediation models, please refer to the `lavaan` or `mediation` R
    packages, or the PROCESS macro for SPSS.

    The two-sided p-value of the indirect effect is computed using the
    bootstrap distribution, as in the mediation R package. However, the p-value
    should be interpreted with caution since it is a) not constructed
    conditioned on a true null hypothesis (see Hayes and Rockwood 2017) and b)
    varies depending on the number of bootstrap samples and the random seed.

    Note that rows with NaN are automatically removed.

    Results have been tested against the R mediation package and this tutorial
    https://data.library.virginia.edu/introduction-to-mediation-analysis/

    References
    ----------
    .. [1] Baron, R. M. & Kenny, D. A. The moderator–mediator variable
           distinction in social psychological research: Conceptual, strategic,
           and statistical considerations. J. Pers. Soc. Psychol. 51, 1173–1182
           (1986).

    .. [2] Fiedler, K., Schott, M. & Meiser, T. What mediation analysis can
           (not) do. J. Exp. Soc. Psychol. 47, 1231–1236 (2011).

    .. [3] Hayes, A. F. & Rockwood, N. J. Regression-based statistical
           mediation and moderation analysis in clinical research:
           Observations, recommendations, and implementation. Behav. Res.
           Ther. 98, 39–57 (2017).

    .. [4] https://cran.r-project.org/web/packages/mediation/mediation.pdf

    .. [5] http://lavaan.ugent.be/tutorial/mediation.html

    .. [6] https://github.com/rmill040/pymediation

    Examples
    --------
    1. Simple mediation analysis

    >>> from pingouin import mediation_analysis, read_dataset
    >>> df = read_dataset('mediation')
    >>> mediation_analysis(data=df, x='X', m='M', y='Y', alpha=0.05, seed=42)
           path    coef      se          pval  CI[2.5%]  CI[97.5%]  sig
    0     M ~ X  0.5610  0.0945  4.391362e-08    0.3735     0.7485  Yes
    1     Y ~ M  0.6542  0.0858  1.612674e-11    0.4838     0.8245  Yes
    2     Total  0.3961  0.1112  5.671128e-04    0.1755     0.6167  Yes
    3    Direct  0.0396  0.1096  7.187429e-01   -0.1780     0.2572   No
    4  Indirect  0.3565  0.0833  0.000000e+00    0.2198     0.5377  Yes

    2. Return the indirect bootstrapped beta coefficients

    >>> stats, dist = mediation_analysis(data=df, x='X', m='M', y='Y',
    ...                                  return_dist=True)
    >>> print(dist.shape)
    (500,)

    3. Mediation analysis with a binary mediator variable

    >>> mediation_analysis(data=df, x='X', m='Mbin', y='Y', seed=42)
           path    coef      se      pval  CI[2.5%]  CI[97.5%]  sig
    0  Mbin ~ X -0.0208  0.1159  0.857510   -0.2479     0.2063   No
    1  Y ~ Mbin -0.1354  0.4118  0.743076   -0.9525     0.6818   No
    2     Total  0.3961  0.1112  0.000567    0.1755     0.6167  Yes
    3    Direct  0.3956  0.1117  0.000614    0.1739     0.6173  Yes
    4  Indirect  0.0023  0.0503  0.960000   -0.0724     0.1464   No

    4. Mediation analysis with covariates

    >>> mediation_analysis(data=df, x='X', m='M', y='Y',
    ...                    covar=['Mbin', 'Ybin'], seed=42)
           path    coef      se          pval  CI[2.5%]  CI[97.5%]  sig
    0     M ~ X  0.5594  0.0968  9.394635e-08    0.3672     0.7516  Yes
    1     Y ~ M  0.6660  0.0861  1.017261e-11    0.4951     0.8368  Yes
    2     Total  0.4204  0.1129  3.324252e-04    0.1962     0.6446  Yes
    3    Direct  0.0645  0.1104  5.608583e-01   -0.1548     0.2837   No
    4  Indirect  0.3559  0.0865  0.000000e+00    0.2093     0.5530  Yes

    5. Mediation analysis with multiple parallel mediators

    >>> mediation_analysis(data=df, x='X', m=['M', 'Mbin'], y='Y', seed=42)
                path    coef      se          pval  CI[2.5%]  CI[97.5%]  sig
    0          M ~ X  0.5610  0.0945  4.391362e-08    0.3735     0.7485  Yes
    1       Mbin ~ X -0.0051  0.0290  8.592408e-01   -0.0626     0.0523   No
    2          Y ~ M  0.6537  0.0863  2.118163e-11    0.4824     0.8250  Yes
    3       Y ~ Mbin -0.0640  0.3282  8.456998e-01   -0.7154     0.5873   No
    4          Total  0.3961  0.1112  5.671128e-04    0.1755     0.6167  Yes
    5         Direct  0.0395  0.1102  7.206301e-01   -0.1792     0.2583   No
    6     Indirect M  0.3563  0.0845  0.000000e+00    0.2148     0.5385  Yes
    7  Indirect Mbin  0.0003  0.0097  9.520000e-01   -0.0172     0.0252   No
    """
    # Sanity check
    assert isinstance(x, str), 'y must be a string.'
    assert isinstance(y, str), 'y must be a string.'
    assert isinstance(m, (list, str)), 'Mediator(s) must be a list or string.'
    assert isinstance(covar, (type(None), str, list))
    if isinstance(m, str):
        m = [m]
    n_mediator = len(m)
    assert isinstance(data, pd.DataFrame), 'Data must be a DataFrame.'
    # Check for duplicates
    assert n_mediator == len(set(m)), 'Cannot have duplicates mediators.'
    if isinstance(covar, str):
        covar = [covar]
    if isinstance(covar, list):
        assert len(covar) == len(set(covar)), 'Cannot have duplicates covar.'
        assert set(m).isdisjoint(covar), 'Mediator cannot be in covar.'
    # Check that columns are in dataframe
    columns = _fl([x, m, y, covar])
    keys = data.columns
    assert all([c in keys for c in columns]), 'Column(s) are not in DataFrame.'
    # Check that columns are numeric
    err_msg = "Columns must be numeric or boolean."
    assert all([data[c].dtype.kind in 'bfi' for c in columns]), err_msg

    # Drop rows with NAN Values
    data = data[columns].dropna()
    n = data.shape[0]
    assert n > 5, 'DataFrame must have at least 5 samples (rows).'

    # Check if mediator is binary
    mtype = 'logistic' if all(data[m].nunique() == 2) else 'linear'

    # Name of CI
    ll_name = 'CI[%.1f%%]' % (100 * alpha / 2)
    ul_name = 'CI[%.1f%%]' % (100 * (1 - alpha / 2))

    # Compute regressions
    cols = ['names', 'coef', 'se', 'pval', ll_name, ul_name]

    # For speed, we pass np.array instead of pandas DataFrame
    X_val = data[_fl([x, covar])].values  # X + covar as predictors
    XM_val = data[_fl([x, m, covar])].values  # X + M + covar as predictors
    M_val = data[m].values  # M as target (no covariates)
    y_val = data[y].values  # y as target (no covariates)

    # M(j) ~ X + covar
    sxm = {}
    for idx, j in enumerate(m):
        if mtype == 'linear':
            sxm[j] = linear_regression(X_val, M_val[:, idx],
                                       alpha=alpha).loc[[1], cols]
        else:
            sxm[j] = logistic_regression(X_val, M_val[:, idx],
                                         alpha=alpha).loc[[1], cols]
        sxm[j].at[1, 'names'] = '%s ~ X' % j
    sxm = pd.concat(sxm, ignore_index=True)

    # Y ~ M + covar
    smy = linear_regression(data[_fl([m, covar])], y_val,
                            alpha=alpha).loc[1:n_mediator, cols]

    # Average Total Effects (Y ~ X + covar)
    sxy = linear_regression(X_val, y_val, alpha=alpha).loc[[1], cols]

    # Average Direct Effects (Y ~ X + M + covar)
    direct = linear_regression(XM_val, y_val, alpha=alpha).loc[[1], cols]

    # Rename paths
    smy['names'] = smy['names'].apply(lambda x: 'Y ~ %s' % x)
    direct.at[1, 'names'] = 'Direct'
    sxy.at[1, 'names'] = 'Total'

    # Concatenate and create sig column
    stats = pd.concat((sxm, smy, sxy, direct), ignore_index=True)
    stats['sig'] = np.where(stats['pval'] < alpha, 'Yes', 'No')

    # Bootstrap confidence intervals
    rng = np.random.RandomState(seed)
    idx = rng.choice(np.arange(n), replace=True, size=(n_boot, n))
    ab_estimates = np.zeros(shape=(n_boot, n_mediator))
    for i in range(n_boot):
        ab_estimates[i, :] = _point_estimate(X_val, XM_val, M_val, y_val,
                                             idx[i, :], n_mediator, mtype)

    ab = _point_estimate(X_val, XM_val, M_val, y_val, np.arange(n),
                         n_mediator, mtype)
    indirect = {'names': m, 'coef': ab, 'se': ab_estimates.std(ddof=1, axis=0),
                'pval': [], ll_name: [], ul_name: [], 'sig': []}

    for j in range(n_mediator):
        ci_j = _bca(ab_estimates[:, j], indirect['coef'][j],
                    alpha=alpha, n_boot=n_boot)
        indirect[ll_name].append(min(ci_j))
        indirect[ul_name].append(max(ci_j))
        # Bootstrapped p-value of indirect effect
        # Note that this is less accurate than a permutation test because the
        # bootstrap distribution is not conditioned on a true null hypothesis.
        # For more details see Hayes and Rockwood 2017
        indirect['pval'].append(_pval_from_bootci(ab_estimates[:, j],
                                indirect['coef'][j]))
        indirect['sig'].append('Yes' if indirect['pval'][j] < alpha else 'No')

    # Create output dataframe
    indirect = pd.DataFrame.from_dict(indirect)
    if n_mediator == 1:
        indirect['names'] = 'Indirect'
    else:
        indirect['names'] = indirect['names'].apply(lambda x:
                                                    'Indirect %s' % x)
    stats = stats.append(indirect, ignore_index=True)
    stats = stats.rename(columns={'names': 'path'})

    # Round
    col_to_round = ['coef', 'se', ll_name, ul_name]
    stats[col_to_round] = stats[col_to_round].round(4)

    if return_dist:
        return stats, np.squeeze(ab_estimates)
    else:
        return stats
