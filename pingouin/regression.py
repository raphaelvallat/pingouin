import numpy as np
import pandas as pd
from scipy.stats import t, norm

__all__ = ['linear_regression', 'logistic_regression', 'mediation_analysis']


def linear_regression(X, y, add_intercept=True, coef_only=False, alpha=0.05,
                      as_dataframe=True):
    """(Multiple) Linear regression.

    Parameters
    ----------
    X : np.array or list
        Predictor(s). Shape = (n_samples, n_features) or (n_samples,).
    y : np.array or list
        Dependent variable. Shape = (n_samples).
    add_intercept : bool
        If False, assume that the data are already centered. If True, add a
        constant term to the model. In this case, the first value in the
        output dict is the intercept of the model.
    coef_only : bool
        If True, return only the regression coefficients.
    alpha : float
        Alpha value used for the confidence intervals.
        CI = [alpha / 2 ; 1 - alpha / 2]
    as_dataframe : bool
        If True, returns a pandas DataFrame. If False, returns a dictionnary.

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

    Notes
    -----
    The beta coefficients of the regression are estimated using the
    np.linalg.lstsq function.

    It is generally recommanded to include a constant term (intercept) to the
    model to limit the bias and force the residual mean to equal zero.
    Note that intercept coefficient and p-values are however rarely meaningful.

    The standard error of the estimates is a measure of the accuracy of the
    prediction defined as:

    .. math:: se = \sqrt{MSE \cdot (X^TX)^{-1}}

    where :math:`MSE` is the mean squared error,

    .. math:: MSE = \dfrac{\sum{(true - pred)^2}}{n - p - 1}

    :math:`p` is the total number of explanatory variables in the model
    (excluding the intercept) and :math:`n` is the sample size.

    Using the coefficients and the standard errors, the T-values can be
    obtained:

    .. math:: T = \dfrac{coef}{se}

    and the p-values can then be approximated using a T-distribution
    with :math:`n - p - 1` degrees of freedom.

    The coefficient of determination (:math:`R^2`) is defined as:

    .. math:: R^2 = 1 - (\dfrac{SS_{resid}}{SS_{total}})

    The adjusted :math:`R^2` is defined as:

    .. math:: \overline{R}^2 = 1 - (1 - R^2) \dfrac{n - 1}{n - p - 1}

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
        >>> print(lm['coef'].values)
            [4.39720706 0.39495526]

    2. Multiple linear regression

        >>> np.random.seed(42)
        >>> z = np.random.normal(size=n)
        >>> X = np.column_stack((x, z))
        >>> lm = linear_regression(X, y)
        >>> print(lm['coef'].values)
            [4.54123324 0.36628301 0.17709451]

    3. Using a Pandas DataFrame

        >>> import pandas as pd
        >>> df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        >>> lm = linear_regression(df[['x', 'z']], df['y'])
        >>> print(lm['coef'].values)
            [4.54123324 0.36628301 0.17709451]

    4. No intercept and return coef only

        >>> linear_regression(X, y, add_intercept=False, coef_only=True)
            array([ 1.40935593, -0.2916508 ])

    5. Return a dictionnary instead of a DataFrame

        >>> linear_regression(X, y, as_dataframe=False)
    """
    # Extract names if X is a Dataframe or Series
    if isinstance(X, pd.DataFrame):
        names = X.keys().tolist()
    elif isinstance(X, pd.Series):
        names = [X.name]
    else:
        names = []

    assert 0 < alpha < 1

    # Convert input to numpy array
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 1:
        # Convert to (n_samples, n_features) shape
        X = X[..., np.newaxis]

    if not names:
        names = ['x' + str(i + 1) for i in range(X.shape[1])]

    if add_intercept:
        # Add intercept
        X = np.column_stack((np.ones(X.shape[0]), X))
        names.insert(0, "Intercept")

    # Compute beta coefficient and predictions
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    if coef_only:
        return coef
    pred = np.dot(X, coef)
    resid = np.square(y - pred)
    ss_res = resid.sum()

    n, p = X.shape[0], X.shape[1]
    # Degrees of freedom should not include the intercept
    dof = n - p if add_intercept else n - p - 1
    # Compute mean squared error, variance and SE
    MSE = ss_res / dof
    beta_var = MSE * (np.linalg.inv(np.dot(X.T, X)).diagonal())
    beta_se = np.sqrt(beta_var)

    # Compute R2, adjusted R2 and RMSE
    ss_tot = np.square(y - y.mean()).sum()
    # ss_exp = np.square(pred - y.mean()).sum()
    r2 = 1 - (ss_res / ss_tot)
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof

    # Compute T and p-values
    T = coef / beta_se
    pval = np.array([2 * t.sf(np.abs(i), dof) for i in T])

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
        return pd.DataFrame.from_dict(stats)
    else:
        return stats


def logistic_regression(X, y, coef_only=False, alpha=0.05,
                        as_dataframe=True, **kwargs):
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
        CI = [alpha / 2 ; 1 - alpha / 2]
    as_dataframe : bool
        If True, returns a pandas DataFrame. If False, returns a dictionnary.
    **kwargs : optional
        Optional arguments passed to sklearn.linear_model.LogisticRegression

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

    Notes
    -----
    This is a wrapper around the sklearn.linear_model.LogisticRegression class.

    Results have been compared against statsmodels and JASP.

    Note that the first coefficient is always the constant term (intercept) of
    the model.

    Adapted from a code found at
    https://gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d

    Examples
    --------
    1. Simple binary logistic regression

        >>> import numpy as np
        >>> from pingouin import logistic_regression
        >>> np.random.seed(123)
        >>> x = np.random.normal(size=30)
        >>> y = np.random.randint(0, 2, size=30)
        >>> lom = logistic_regression(x, y)
        >>> print(lom['coef'].values)
            [-0.27122371  0.05927182]

    2. Multiple binary logistic regression

        >>> np.random.seed(42)
        >>> z = np.random.normal(size=30)
        >>> X = np.column_stack((x, z))
        >>> lom = logistic_regression(X, y)
        >>> print(lom['coef'].values)
            [-0.34933805 -0.0226106  -0.39453532]

    3. Using a Pandas DataFrame

        >>> import pandas as pd
        >>> df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        >>> lom = logistic_regression(df[['x', 'z']], df['y'])
        >>> print(lom['coef'].values)
            [-0.34933805 -0.0226106  -0.39453532]

    4. Return only the coefficients

        >>> logistic_regression(X, y, coef_only=True)
            array([-0.34933805, -0.0226106 , -0.39453532])

    4. Passing custom parameters to sklearn

        >>> lom = logistic_regression(X, y, solver='sag', max_iter=10000)
        >>> print(lom['coef'].values)
            [-0.34941889 -0.02261911 -0.39451064]
    """
    # Check that sklearn is installed
    from pingouin.utils import is_sklearn_installed
    is_sklearn_installed(raise_error=True)
    from sklearn.linear_model import LogisticRegression

    # Extract names if X is a Dataframe or Series
    if isinstance(X, pd.DataFrame):
        names = X.keys().tolist()
    elif isinstance(X, pd.Series):
        names = [X.name]
    else:
        names = []

    assert 0 < alpha < 1

    # Convert to numpy array
    X = np.asarray(X)
    y = np.asarray(y)

    if np.unique(y).size != 2:
        raise ValueError('Dependent variable must be binary.')

    # Add axis if only one-dimensional array
    if X.ndim == 1:
        X = X[..., np.newaxis]

    if not names:
        names = ['x' + str(i + 1) for i in range(X.shape[1])]

    # Add intercept in names
    names.insert(0, "Intercept")

    # Initialize and fit
    if 'solver' not in kwargs:
        kwargs['solver'] = 'lbfgs'
    if 'multi_class' not in kwargs:
        kwargs['multi_class'] = 'auto'
    lom = LogisticRegression(**kwargs)
    lom.fit(X, y)
    coef = np.append(lom.intercept_, lom.coef_)
    if coef_only:
        return coef

    # Design matrix -- add intercept
    X_design = np.column_stack((np.ones(X.shape[0]), X))
    n, p = X_design.shape

    # Fisher Information Matrix
    denom = (2 * (1 + np.cosh(lom.decision_function(X))))
    denom = np.tile(denom, (p, 1)).T
    fim = np.dot((X_design / denom).T, X_design)
    crao = np.linalg.inv(fim)

    # Standard error and Z-scores
    se = np.sqrt(np.diag(crao))
    z_scores = coef / se

    # Two-tailed p-values
    pval = np.array([2 * norm.sf(abs(z)) for z in z_scores])

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
        return pd.DataFrame.from_dict(stats)
    else:
        return stats


def _point_estimate(data, x, m, y, idx, mtype='linear'):
    """Point estimate of indirect effect based on bootstrap sample."""
    # Mediator model (M ~ X)
    if mtype == 'linear':
        beta_m = linear_regression(data[x].iloc[idx], data[m].iloc[idx],
                                   add_intercept=True, coef_only=True)
    else:
        beta_m = logistic_regression(data[x].iloc[idx], data[m].iloc[idx],
                                     coef_only=True)
    # Full model (Y ~ X + M)
    beta_y = linear_regression(data[[x, m]].iloc[idx], data[y].iloc[idx],
                               add_intercept=True, coef_only=True)
    # Point estimate
    return beta_m[1] * beta_y[2]


def _bias_corrected_interval(ab_estimates, sample_point, n_boot, alpha=0.05):
    """Get (1 - alpha) * 100 bias-corrected confidence interval estimate

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


def mediation_analysis(data=None, x=None, m=None, y=None, alpha=0.05,
                       n_boot=500, return_dist=False):
    """Mediation analysis using a bias-correct non-parametric bootstrap method.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe.
    x : str
        Column name in data containing the predictor variable.
        The predictor variable must be continuous.
    m : str
        Column name in data containing the mediator variable.
        The mediator can be continuous or binary (e.g. 0 or 1).
    y : str
        Column name in data containing the outcome variable.
        The outcome variable must be continuous.
    alpha : float
        Significance threshold. Used to determine the confidence interval,
        CI = [ alpha / 2 ; 1 -  alpha / 2]
    n_boot : int
        Number of bootstrap iterations. The greater, the slower.
    return_dist : bool
        If True, the function also returns the indirect bootstrapped beta
        samples (size = n_boot). Can be plotted for instance using
        seaborn.distplot() or seaborn.kdeplot() functions.

    Returns
    -------
    stats : pd.DataFrame
        Mediation summary::

        'Path' : regression model
        'Beta' : regression estimates
        'CI[2.5%]' : lower confidence interval
        'CI[97.5%]' : upper confidence interval
        'Sig' : regression statistical significance

    Notes
    -----
    Mediation analysis (MA) is a "statistical procedure to test
    whether the effect of an independent variable X on a dependent variable
    Y (i.e., X → Y) is at least partly explained by a chain of effects of the
    independent variable on an intervening mediator variable M and of the
    intervening variable on the dependent variable (i.e., X → M → Y)"
    (from Fiedler et al. 2011).

    A linear regression is used if the mediator variable is continuous and a
    logistic regression if the mediator variable is dichotomous (binary).

    The indirect effect (also referred to as average causal mediation effect
    or ACME) is considered significant if the specified confidence
    interval does not include 0. The path 'X --> Y' is the sum of both the
    indirect and direct effect. It is sometimes referred to as total effect.

    Results have been tested against the R mediation package and this tutorial
    https://data.library.virginia.edu/introduction-to-mediation-analysis/

    Adapted from a code found at https://github.com/rmill040/pymediation

    References
    ----------
    .. [1] Baron, Reuben M., and David A. Kenny. "The moderator–mediator
           variable distinction in social psychological research: Conceptual,
           strategic, and statistical considerations." Journal of personality
           and social psychology 51.6 (1986): 1173.

    .. [2] Fiedler, Klaus, Malte Schott, and Thorsten Meiser.
           "What mediation analysis can (not) do." Journal of Experimental
           Social Psychology 47.6 (2011): 1231-1236.

    Examples
    --------
    1. Simple mediation analysis

        >>> from pingouin import mediation_analysis
        >>> from pingouin.datasets import read_dataset
        >>> df = read_dataset('mediation')
        >>> mediation_analysis(data=df, x='X', m='M', y='Y', alpha=0.05)

    2. Return the indirect bootstrapped beta coefficients

        >>> stats, dist = mediation_analysis(data=df, x='X', m='M', y='Y',
        >>>                                  return_dist=True)
        >>> print(dist.shape)
            (500,)

    3. Mediation analysis with a binary mediator variable

        >>> from pingouin import mediation_analysis
        >>> from pingouin.datasets import read_dataset
        >>> df = read_dataset('mediation')
        >>> mediation_analysis(data=df, x='X', m='Mbin', y='Y', alpha=0.05)
    """
    n = data.shape[0]

    # Initialize variables
    ab_estimates = np.zeros(n_boot)
    indirect = {}

    # Check if mediator is binary
    mtype = 'logistic' if data[m].unique().size == 2 else 'linear'

    # Bootstrap
    for i in range(n_boot):
        idx = np.random.choice(np.arange(n), replace=True, p=None, size=n)
        ab_estimates[i] = _point_estimate(data, x=x, m=m, y=y, idx=idx,
                                          mtype=mtype)

    # Bootstrap point estimate and confidence interval
    indirect['coef'] = _point_estimate(data, x=x, m=m, y=y, idx=np.arange(n),
                                       mtype=mtype)
    indirect['ci'] = _bias_corrected_interval(ab_estimates, indirect['coef'],
                                              alpha=alpha, n_boot=n_boot)
    # Significance of the mediation effect
    indirect['sig'] = 'Yes' if (np.sign(indirect['ci'][0])
                                == np.sign(indirect['ci'][1])) else 'No'

    # Compute linear regressions
    if mtype == 'linear':
        sxm = linear_regression(data[x], data[m], add_intercept=True,
                                alpha=alpha, as_dataframe=False)
    else:
        sxm = logistic_regression(data[x], data[m], alpha=alpha,
                                  as_dataframe=False)

    smy = linear_regression(data[m], data[y], add_intercept=True, alpha=alpha,
                            as_dataframe=False)
    # sxy = Average Total Effects
    sxy = linear_regression(data[x], data[y], add_intercept=True, alpha=alpha,
                            as_dataframe=False)
    # Average Direct Effects
    direct = linear_regression(data[[x, m]], data[y], add_intercept=True,
                               alpha=alpha, as_dataframe=False)

    # Significance
    sig_sxy = 'Yes' if sxy['pval'][1] < alpha else 'No'
    sig_sxm = 'Yes' if sxm['pval'][1] < alpha else 'No'
    sig_smy = 'Yes' if smy['pval'][1] < alpha else 'No'
    sig_direct = 'Yes' if direct['pval'][1] < alpha else 'No'

    # Name of CI
    ll_name = 'CI[%.1f%%]' % (100 * alpha / 2)
    ul_name = 'CI[%.1f%%]' % (100 * (1 - alpha / 2))

    # Create output dataframe
    stats = pd.DataFrame({'Path': ['X -> M', 'M -> Y', 'X -> Y', 'Direct',
                                   'Indirect'],
                          # Beta coefficients
                          'Beta': [sxm['coef'][1], smy['coef'][1],
                                   sxy['coef'][1], direct['coef'][1],
                                   indirect['coef']],
                          # Lower CI
                          ll_name: [sxm[ll_name][1], smy[ll_name][1],
                                    sxy[ll_name][1], direct[ll_name][1],
                                    min(indirect['ci'])],
                          # Upper CI
                          ul_name: [sxm[ul_name][1], smy[ul_name][1],
                                    sxy[ul_name][1], direct[ul_name][1],
                                    max(indirect['ci'])],
                          # Significance level
                          'Sig': [sig_sxm, sig_smy, sig_sxy, sig_direct,
                                  indirect['sig']],
                          }).round(4)

    if return_dist:
        return stats, ab_estimates
    else:
        return stats
