import numpy as np
from scipy.stats import t
# from scipy.stats import norm


__all__ = ['linear_regression']


def linear_regression(X, y, add_intercept=True, coef_only=False, alpha=0.05):
    """(Multiple) Linear regression.

    Parameters
    ----------
    X : np.array or list
        Predictor array. Shape = (n_samples, n_features) or (n_samples,).
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

    if ll.sum() > ul.sum():
        ll, ul = ul, ll

    # Create dict
    stats = {'coef': coef, 'se': beta_se, 'tvals': tvals, 'pvals': pvals,
             'rsquared': rsquared, 'adj_rsquared': adj_rsquared,
             'll': ll, 'ul': ul}
    return stats
