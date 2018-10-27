import numpy as np
import pandas as pd
from scipy.stats import t, norm

__all__ = ['linear_regression', 'logistic_regression', 'mediation_analysis']


def linear_regression(X, y, add_intercept=True, coef_only=False, alpha=0.05):
    """(Multiple) Linear regression.

    Parameters
    ----------
    X : np.array or list
        Predictor(s). Shape = (n_samples, n_features) or (n_samples,).
    y : np.array or list
        Dependent variable. Shape = (n_samples).
    add_intercept : bool
        If False, assume that the data are already centered. If True, add an
        intercept to the model. In this case, the first value in the
        output dict is the intercept of the model.
    coef_only : bool
        If True, return only the regression coefficients.
    alpha : float
        Alpha value used for the confidence intervals.
        CI = [alpha / 2 ; 1 - alpha / 2]

    Returns
    -------
    stats : dict
        Linear regression summary::

        'coef' : regression coefficients
        'se' : standard error of the estimate
        'tvals' : T-values
        'pvals' : p-values
        'rsquared' : coefficient of determination (R2)
        'adj_rsquared' : adjusted R2
        'll' : lower confidence interval
        'ul' : upper confidence interval

    Notes
    -----
    Results have been compared against the sklearn library.

    The coefficient of determination (:math:`R^2`) is defined as:

    .. math:: R^2 = 1 - (\dfrac{SS_{resid}}{SS_{total}})

    Unlike most other scores, :math:`R^2` score may be negative (it need not
    actually be the square of a quantity R).

    Examples
    --------
    1. Simple linear regression

        >>> import numpy as np
        >>> from pingouin import linear_regression
        >>> np.random.seed(123)
        >>> mean, cov, n = [4, 6], [[1, 0.5], [0.5, 1]], 30
        >>> x, y = np.random.multivariate_normal(mean, cov, n).T
        >>> lm = linear_regression(x, y)
        >>> print(lm['coef'])
            [4.39720706 0.39495526]

    2. Multiple linear regression

        >>> np.random.seed(42)
        >>> z = np.random.normal(size=n)
        >>> X = np.column_stack((x, z))
        >>> lm = linear_regression(X, y)
        >>> print(lm['coef'])
            [4.54123324 0.36628301 0.17709451]

    3. Using a Pandas DataFrame

        >>> import pandas as pd
        >>> df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        >>> lm = linear_regression(df[['x', 'z']], df['y'])
        >>> print(lm['coef'])
            [4.54123324 0.36628301 0.17709451]

    4. No intercept and return coef only

        >>> linear_regression(df[['x', 'z']], df['y'], add_intercept=False,
        >>>                   coef_only=True)
            array([ 1.40935593, -0.2916508 ])
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 1:
        # Convert to (n_samples, n_features) shape
        X = X[..., np.newaxis]

    if add_intercept:
        # Add intercept
        X = np.column_stack((np.ones(X.shape[0]), X))

    # Compute beta coefficient and predictions
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    if coef_only:
        return coef
    pred = np.dot(X, coef)

    # Compute mean squared error, variance and SE
    n, p = X.shape[0], X.shape[1]
    MSE = ((y - pred)**2).sum() / (n - p)
    beta_var = MSE * (np.linalg.inv(np.dot(X.T, X)).diagonal())
    beta_se = np.sqrt(beta_var)

    # Compute R2 and adjusted r-squared
    ss_tot = np.square(y - y.mean()).sum()
    ss_res = np.square(y - pred).sum()
    # ss_exp = np.square(pred - y.mean()).sum()
    rsquared = 1 - (ss_res / ss_tot)
    adj_rsquared = 1 - (1 - rsquared) * (n - 1) / (n - p - 1)

    # Compute T and p-values
    tvals = coef / beta_se
    pvals = [2 * t.sf(np.abs(i), (n - 1)) for i in tvals]

    # Compute confidence intervals
    crit = t.ppf(1 - alpha / 2, n - p - 1)
    marg_error = crit * beta_se
    ll = coef - marg_error
    ul = coef + marg_error

    # Create dict
    stats = {'coef': coef, 'se': beta_se, 'tvals': tvals, 'pvals': pvals,
             'rsquared': rsquared, 'adj_rsquared': adj_rsquared,
             'll': ll, 'ul': ul}
    return stats


def logistic_regression(X, y, coef_only=False, alpha=0.05):
    """Binary logistic regression.

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

    Returns
    -------
    stats : dict
        Logistic regression summary::

        'coef' : regression coefficients
        'se' : standard error
        'z' : z-scores
        'pvals' : two-tailed p-values
        'll' : lower confidence interval
        'ul' : upper confidence interval

    Notes
    -----
    This is a wrapper around the sklearn.linear_model.LogisticRegression class.

    Results have been tested against statsmodels.

    Note that the first coefficient is always the intercept of the model.

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
        >>> print(lom['coef'])
            [-0.27122371  0.05927182]

    2. Multiple binary logistic regression

        >>> np.random.seed(42)
        >>> z = np.random.normal(size=30)
        >>> X = np.column_stack((x, z))
        >>> lom = logistic_regression(X, y)
        >>> print(lom['coef'])
            [-0.34933805 -0.0226106  -0.39453532]

    3. Using a Pandas DataFrame

        >>> import pandas as pd
        >>> df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        >>> lom = logistic_regression(df[['x', 'z']], df['y'])
        >>> print(lom['coef'])
            [-0.34933805 -0.0226106  -0.39453532]

    4. Return only the coefficients

        >>> logistic_regression(df[['x', 'z']], df['y'], coef_only=True)
            array([-0.34933805, -0.0226106 , -0.39453532])
    """
    # Check that sklearn is installed
    from pingouin.utils import is_sklearn_installed
    is_sklearn_installed(raise_error=True)
    from sklearn.linear_model import LogisticRegression

    # Convert to numpy array
    X = np.asarray(X)
    y = np.asarray(y)

    if np.unique(y).size != 2:
        raise ValueError('Dependent variable must be binary.')

    # Add axis if only one-dimensional array
    if X.ndim == 1:
        X = X[..., np.newaxis]

    # Initialize and fit
    lom = LogisticRegression(solver='lbfgs', multi_class='auto')
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
    pvals = [2 * norm.sf(abs(z)) for z in z_scores]

    # Confidence intervals
    crit = norm.ppf(1 - alpha / 2)
    ll = coef - crit * se
    ul = coef + crit * se

    # Create dict
    stats = {'coef': coef, 'se': se, 'z': z_scores, 'pvals': pvals,
             'll': ll, 'ul': ul}
    return stats


def _point_estimate(data, x, m, y, idx):
    """Point estimate of indirect effect based on bootstrap sample."""
    # Mediator model (M ~ X)
    beta_m = linear_regression(data[x].iloc[idx], data[m].iloc[idx],
                               add_intercept=True, coef_only=True)
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
        Column in data containing the predictor variable.
    m : str
        Column in data containing the mediator variable.
    y : str
        Column in data containing the outcome variable.
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
        Mediation summary.

    Notes
    -----
    Mediation analysis (MA) is a "statistical procedure to test
    whether the effect of an independent variable X on a dependent variable
    Y (i.e., X → Y) is at least partly explained by a chain of effects of the
    independent variable on an intervening mediator variable M and of the
    intervening variable on the dependent variable (i.e., X → M → Y)"
    (from Fiedler et al. 2011)

    Note that the current implementation only works with continuous variables.
    The indirect effect is considered significant if the specified confidence
    interval does not include 0.

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
    """
    n = data.shape[0]

    # Initialize variables
    ab_estimates = np.zeros(n_boot)
    indirect = {}

    # Bootstrap
    for i in range(n_boot):
        idx = np.random.choice(np.arange(n), replace=True, p=None, size=n)
        ab_estimates[i] = _point_estimate(data, x=x, m=m, y=y, idx=idx)

    # Bootstrap point estimate and confidence interval
    indirect['coef'] = _point_estimate(data, x=x, m=m, y=y, idx=np.arange(n))
    indirect['ci'] = _bias_corrected_interval(ab_estimates, indirect['coef'],
                                              alpha=alpha, n_boot=n_boot)
    # Significance of the mediation effect
    indirect['sig'] = 'Yes' if (np.sign(indirect['ci'][0])
                                == np.sign(indirect['ci'][1])) else 'No'

    # Compute linear regressions
    sxm = linear_regression(data[x], data[m], add_intercept=True, alpha=alpha)
    smy = linear_regression(data[m], data[y], add_intercept=True, alpha=alpha)
    # sxy = Average Total Effects
    sxy = linear_regression(data[x], data[y], add_intercept=True, alpha=alpha)
    # Average Direct Effects
    direct = linear_regression(data[[x, m]], data[y], add_intercept=True,
                               alpha=alpha)

    # Significance
    sig_sxy = 'Yes' if sxy['pvals'][1] < alpha else 'No'
    sig_sxm = 'Yes' if sxm['pvals'][1] < alpha else 'No'
    sig_smy = 'Yes' if smy['pvals'][1] < alpha else 'No'
    sig_direct = 'Yes' if direct['pvals'][1] < alpha else 'No'

    # Create output dataframe
    stats = pd.DataFrame({'Path': ['X -> M', 'M -> Y', 'X -> Y', 'Direct',
                                   'Indirect'],
                          # Beta coefficients
                          'Beta': [sxm['coef'][1], smy['coef'][1],
                                   sxy['coef'][1], direct['coef'][1],
                                   indirect['coef']],
                          # Lower CI
                          'll': [sxm['ll'][1], smy['ll'][1], sxy['ll'][1],
                                 direct['ll'][1], min(indirect['ci'])],
                          # Upper CI
                          'ul': [sxm['ul'][1], smy['ul'][1], sxy['ul'][1],
                                 direct['ul'][1], max(indirect['ci'])],
                          # Significance level
                          'Sig': [sig_sxm, sig_smy, sig_sxy, sig_direct,
                                  indirect['sig']],
                          }).round(4)
    # Rename CI
    stats.rename(columns={'ll': 'CI[%.1f%%]' % (100 * alpha / 2),
                          'ul': 'CI[%.1f%%]' % (100 * (1 - alpha / 2))},
                 inplace=True)

    if return_dist:
        return stats, ab_estimates
    else:
        return stats
