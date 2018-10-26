import numpy as np
import pandas as pd
from scipy.stats import t, norm

__all__ = ['linear_regression', 'mediation_analysis']


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
        Linear regression output::

        'coef' : regression coefficients
        'se' : standard error of the estimate
        'tvals' : T-values
        'pvals' : p-values
        'rsquared' : coefficient of determination (R2)
        'adj_rsquared' : adjusted R2

    Notes
    -----
    Results have been compared against the sklearn library.

    The coefficient of determination (R2) is defined as:

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

        >>> np.random.seed(123)
        >>> z = np.random.normal(size=n)
        >>> X = np.column_stack((x, z))
        >>> lm = linear_regression(X, y)
        >>> print(lm['coef'])
            [ 4.45443375  0.37561055 -0.22153521]

    3. Using a Pandas DataFrame

        >>> import pandas as pd
        >>> df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        >>> lm = linear_regression(df[['x', 'z']], df['y'])
        >>> print(lm['coef'])
            [ 4.45443375  0.37561055 -0.22153521]

    4. No intercept and return coef only

        >>> linear_regression(df[['x', 'z']], df['y'], add_intercept=False,
        >>>                   coef_only=True)
            array([ 1.41420086, -0.11302849])
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
    The current implementation only works with continuous linear variables.

    Adapted from a code found at https://github.com/rmill040/pymediation

    Results have been tested against the R mediation
    package and this tutorial
    https://data.library.virginia.edu/introduction-to-mediation-analysis/

    The indirect effect is considered significant if the specified confidence
    interval does not include 0.

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
    indirect['sig'] = 'Yes' if (np.sign(indirect['ci'][0]) ==
                                np.sign(indirect['ci'][1])) else 'No'

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
