import itertools
import warnings
import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.stats import t, norm
from scipy.linalg import pinvh, lstsq

from pingouin.config import options
from pingouin.utils import remove_na as rm_na
from pingouin.utils import _flatten_list as _fl
from pingouin.utils import _postprocess_dataframe

__all__ = ['linear_regression', 'logistic_regression', 'mediation_analysis']


def linear_regression(X, y, add_intercept=True, weights=None, coef_only=False,
                      alpha=0.05, as_dataframe=True, remove_na=False, relimp=False):
    """(Multiple) Linear regression.

    Parameters
    ----------
    X : array_like
        Predictor(s), of shape *(n_samples, n_features)* or *(n_samples)*.
    y : array_like
        Dependent variable, of shape *(n_samples)*.
    add_intercept : bool
        If False, assume that the data are already centered. If True, add a
        constant term to the model. In this case, the first value in the
        output dict is the intercept of the model.

        .. note:: It is generally recommended to include a constant term
            (intercept) to the model to limit the bias and force the residual
            mean to equal zero. The intercept coefficient and p-values
            are however rarely meaningful.
    weights : array_like
        An optional vector of sample weights to be used in the fitting
        process, of shape *(n_samples)*. Missing or negative weights are not
        allowed. If not null, a weighted least squares is calculated.

        .. versionadded:: 0.3.5
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
    relimp : bool
        If True, returns the relative importance (= contribution) of
        predictors. This is irrelevant when the predictors are uncorrelated:
        the total :math:`R^2` of the model is simply the sum of each univariate
        regression :math:`R^2`-values. However, this does not apply when
        predictors are correlated. Instead, the total :math:`R^2` of the model
        is partitioned by averaging over all combinations of predictors,
        as done in the `relaimpo
        <https://cran.r-project.org/web/packages/relaimpo/relaimpo.pdf>`_
        R package (``calc.relimp(type="lmg")``).

        .. warning:: The computation time roughly doubles for each
            additional predictor and therefore this can be extremely slow for
            models with more than 12-15 predictors.

        .. versionadded:: 0.3.0

    Returns
    -------
    stats : :py:class:`pandas.DataFrame` or dict
        Linear regression summary:

        * ``'names'``: name of variable(s) in the model (e.g. x1, x2...)
        * ``'coef'``: regression coefficients
        * ``'se'``: standard errors
        * ``'T'``: T-values
        * ``'pval'``: p-values
        * ``'r2'``: coefficient of determination (:math:`R^2`)
        * ``'adj_r2'``: adjusted :math:`R^2`
        * ``'CI[2.5%]'``: lower confidence intervals
        * ``'CI[97.5%]'``: upper confidence intervals
        * ``'relimp'``: relative contribution of each predictor to the final\
                        :math:`R^2` (only if ``relimp=True``).
        * ``'relimp_perc'``: percent relative contribution

        In addition, the output dataframe comes with hidden attributes such as
        the residuals, and degrees of freedom of the model and residuals, which
        can be accessed as follow, respectively:

        >>> lm = pg.linear_regression() # doctest: +SKIP
        >>> lm.residuals_, lm.df_model_, lm.df_resid_ # doctest: +SKIP

        Note that to follow scikit-learn convention, these hidden atributes end
        with an "_". When ``as_dataframe=False`` however, these attributes
        are no longer hidden and can be accessed as any other keys in the
        output dictionary.

        >>> lm = pg.linear_regression() # doctest: +SKIP
        >>> lm['residuals'], lm['df_model'], lm['df_resid'] # doctest: +SKIP

        When ``as_dataframe=False`` the dictionary also contains the
        processed ``X`` and ``y`` arrays (i.e, with NaNs removed if
        ``remove_na=True``) and the model's predicted values ``pred``.

        >>> lm['X'], lm['y'], lm['pred'] # doctest: +SKIP

        For a weighted least squares fit, the weighted ``Xw`` and ``yw``
        arrays are included in the dictionary.

        >>> lm['Xw'], lm['yw'] # doctest: +SKIP

    See also
    --------
    logistic_regression, mediation_analysis, corr

    Notes
    -----
    The :math:`\\beta` coefficients are estimated using an ordinary least
    squares (OLS) regression, as implemented in the
    :py:func:`scipy.linalg.lstsq` function. The OLS method minimizes
    the sum of squared residuals, and leads to a closed-form expression for
    the estimated :math:`\\beta`:

    .. math:: \\hat{\\beta} = (X^TX)^{-1} X^Ty

    It is generally recommended to include a constant term (intercept) to the
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

    The relative importance (``relimp``) column is a partitioning of the
    total :math:`R^2` of the model into individual :math:`R^2` contribution.
    This is calculated by taking the average over average contributions in
    models of different sizes. For more details, please refer to
    `Groemping et al. 2006 <http://dx.doi.org/10.18637/jss.v017.i01>`_
    and the R package `relaimpo
    <https://cran.r-project.org/web/packages/relaimpo/relaimpo.pdf>`_.

    Note that Pingouin will automatically remove any duplicate columns
    from :math:`X`, as well as any column with only one unique value
    (constant), excluding the intercept.

    Results have been compared against sklearn, R, statsmodels and JASP.

    Examples
    --------
    1. Simple linear regression using columns of a pandas dataframe

    In this first example, we'll use the tips dataset to see how well we
    can predict the waiter's tip (in dollars) based on the total bill (also
    in dollars).

    >>> import numpy as np
    >>> import pingouin as pg
    >>> df = pg.read_dataset('tips')
    >>> # Let's predict the tip ($) based on the total bill (also in $)
    >>> lm = pg.linear_regression(df['total_bill'], df['tip'])
    >>> lm.round(2)
            names  coef    se      T  pval    r2  adj_r2  CI[2.5%]  CI[97.5%]
    0   Intercept  0.92  0.16   5.76   0.0  0.46    0.45      0.61       1.23
    1  total_bill  0.11  0.01  14.26   0.0  0.46    0.45      0.09       0.12

    It comes as no surprise that total bill is indeed a significant predictor
    of the waiter's tip (T=14.26, p<0.05). The :math:`R^2` of the model is 0.46
    and the adjusted :math:`R^2` is 0.45, which means that our model roughly
    explains ~45% of the total variance in the tip amount.

    2. Multiple linear regression

    We can also have more than one predictor and run a multiple linear
    regression. Below, we add the party size as a second predictor of tip.

    >>> # We'll add a second predictor: the party size
    >>> lm = pg.linear_regression(df[['total_bill', 'size']], df['tip'])
    >>> lm.round(2)
            names  coef    se      T  pval    r2  adj_r2  CI[2.5%]  CI[97.5%]
    0   Intercept  0.67  0.19   3.46  0.00  0.47    0.46      0.29       1.05
    1  total_bill  0.09  0.01  10.17  0.00  0.47    0.46      0.07       0.11
    2        size  0.19  0.09   2.26  0.02  0.47    0.46      0.02       0.36

    The party size is also a significant predictor of tip (T=2.26, p=0.02).
    Note that adding this new predictor however only improved the :math:`R^2`
    of our model by ~1%.

    This function also works with numpy arrays:

    >>> X = df[['total_bill', 'size']].to_numpy()
    >>> y = df['tip'].to_numpy()
    >>> pg.linear_regression(X, y).round(2)
           names  coef    se      T  pval    r2  adj_r2  CI[2.5%]  CI[97.5%]
    0  Intercept  0.67  0.19   3.46  0.00  0.47    0.46      0.29       1.05
    1         x1  0.09  0.01  10.17  0.00  0.47    0.46      0.07       0.11
    2         x2  0.19  0.09   2.26  0.02  0.47    0.46      0.02       0.36

    3. Get the residuals

    >>> # For clarity, only display the first 9 values
    >>> np.round(lm.residuals_, 2)[:9]
    array([-1.62, -0.55,  0.31,  0.06, -0.11,  0.93,  0.13, -0.81, -0.49])

    Using pandas, we can show a summary of the distribution of the residuals:

    >>> import pandas as pd
    >>> pd.Series(lm.residuals_).describe().round(2)
    count    244.00
    mean      -0.00
    std        1.01
    min       -2.93
    25%       -0.55
    50%       -0.09
    75%        0.51
    max        4.04
    dtype: float64

    5. No intercept and return only the regression coefficients

    Sometimes it may be useful to remove the constant term from the regression,
    or to only return the regression coefficients without calculating the
    standard errors or p-values. This latter can potentially save you a lot of
    time if you need to calculate hundreds of regression and only care about
    the coefficients!

    >>> pg.linear_regression(X, y, add_intercept=False, coef_only=True)
    array([0.1007119 , 0.36209717])

    6. Return a dictionnary instead of a dataframe

    >>> lm_dict = pg.linear_regression(X, y, as_dataframe=False)
    >>> lm_dict.keys()
    dict_keys(['names', 'coef', 'se', 'T', 'pval', 'r2', 'adj_r2', 'CI[2.5%]',
               'CI[97.5%]', 'df_model', 'df_resid', 'residuals', 'X', 'y',
               'pred'])

    7. Remove missing values

    >>> X[4, 1] = np.nan
    >>> y[7] = np.nan
    >>> pg.linear_regression(X, y, remove_na=True, coef_only=True)
    array([0.65749955, 0.09262059, 0.19927529])

    8. Get the relative importance of predictors

    >>> lm = pg.linear_regression(X, y, remove_na=True, relimp=True)
    >>> lm[['names', 'relimp', 'relimp_perc']]
           names    relimp  relimp_perc
    0  Intercept       NaN          NaN
    1         x1  0.342503    73.045583
    2         x2  0.126386    26.954417

    The ``relimp`` column is a partitioning of the total :math:`R^2` of the
    model into individual contribution. Therefore, it sums to the :math:`R^2`
    of the full model. The ``relimp_perc`` is normalized to sum to 100%. See
    `Groemping 2006 <https://www.jstatsoft.org/article/view/v017i01>`_
    for more details.

    >>> lm[['relimp', 'relimp_perc']].sum()
    relimp           0.468889
    relimp_perc    100.000000
    dtype: float64

    9. Weighted linear regression

    >>> X = [1, 2, 3, 4, 5, 6]
    >>> y = [10, 22, 11, 13, 13, 16]
    >>> w = [1, 0.1, 1, 1, 0.5, 1]  # Array of weights. Must be >= 0.
    >>> lm = pg.linear_regression(X, y, weights=w)
    >>> lm.round(2)
           names  coef    se     T  pval    r2  adj_r2  CI[2.5%]  CI[97.5%]
    0  Intercept  9.00  2.03  4.42  0.01  0.51    0.39      3.35      14.64
    1         x1  1.04  0.50  2.06  0.11  0.51    0.39     -0.36       2.44
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
    # 1. Let's remove column(s) with only zero, otherwise the regression fails
    n_nonzero = np.count_nonzero(X, axis=0)
    idx_zero = np.flatnonzero(n_nonzero == 0)  # Find columns that are only 0
    if len(idx_zero):
        X = np.delete(X, idx_zero, 1)
        names = np.delete(names, idx_zero)

    # 2. We also want to make sure that there is no more than one constant
    # column (= intercept), otherwise the regression fails
    # This is equivalent, but much faster, to pd.DataFrame(X).nunique()
    idx_unique = np.where(np.all(X == X[0, :], axis=0))[0]
    if len(idx_unique) > 1:
        # We remove all but the first "Intercept" column.
        X = np.delete(X, idx_unique[1:], 1)
        names = np.delete(names, idx_unique[1:])
    # Is there a constant in our predictor matrix? Useful for dof and R^2.
    constant = 1 if len(idx_unique) > 0 else 0

    # 3. Finally, we want to remove duplicate columns
    if X.shape[1] > 1:
        idx_duplicate = []
        for pair in itertools.combinations(range(X.shape[1]), 2):
            if np.array_equal(X[:, pair[0]], X[:, pair[1]]):
                idx_duplicate.append(pair[1])
        if len(idx_duplicate):
            X = np.delete(X, idx_duplicate, 1)
            names = np.delete(names, idx_duplicate)

    # 4. Check that we have enough samples / features
    n, p = X.shape[0], X.shape[1]
    assert n >= 3, 'At least three valid samples are required in X.'
    assert p >= 1, 'X must have at least one valid column.'

    # 5. Handle weights
    if weights is not None:
        if relimp:
            raise ValueError("relimp = True is not supported when using "
                             "weights.")
        w = np.asarray(weights)
        assert w.ndim == 1, 'weights must be a 1D array.'
        assert w.size == n, 'weights must be of shape n_samples.'
        assert not np.isnan(w).any(), 'Missing weights are not accepted.'
        assert not (w < 0).any(), 'Negative weights are not accepted.'
        # Do not count weights == 0 in dof
        # This gives similar results as R lm() but different from statsmodels
        n = np.count_nonzero(w)
        # Rescale (whitening)
        wts = np.diag(np.sqrt(w))
        Xw = wts @ X
        yw = wts @ y
    else:
        # Set all weights to one, [1, 1, 1, ...]
        w = np.ones(n)
        Xw = X
        yw = y

    # FIT (WEIGHTED) LEAST SQUARES REGRESSION
    coef, ss_res, rank, _ = lstsq(Xw, yw, cond=None)
    ss_res = ss_res[0] if ss_res.shape == (1,) else ss_res
    if coef_only:
        return coef
    calc_ss_res = False
    if rank < Xw.shape[1]:
        # in this case, ss_res is of shape (0,), i.e., an empty array
        warnings.warn('Design matrix supplied with `X` parameter is rank '
                      f'deficient (rank {rank} with {Xw.shape[1]} columns). '
                      'That means that one or more of the columns in `X` '
                      'are a linear combination of one of more of the '
                      'other columns.')
        calc_ss_res = True

    # Degrees of freedom
    df_model = rank - constant
    df_resid = n - rank

    # Calculate predicted values and (weighted) residuals
    pred = Xw @ coef
    resid = yw - pred
    if calc_ss_res:
        # In case we did not get ss_res from lstsq due to rank deficiency
        ss_res = (resid ** 2).sum()

    # Calculate total (weighted) sums of squares and R^2
    ss_tot = yw @ yw
    ss_wtot = np.sum(w * (y - np.average(y, weights=w))**2)
    if constant:
        r2 = 1 - ss_res / ss_wtot
    else:
        r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - constant) / df_resid

    # Compute mean squared error, variance and SE
    mse = ss_res / df_resid
    beta_var = mse * (np.linalg.pinv(Xw.T @ Xw).diagonal())
    beta_se = np.sqrt(beta_var)

    # Compute T and p-values
    T = coef / beta_se
    pval = 2 * t.sf(np.fabs(T), df_resid)

    # Compute confidence intervals
    crit = t.ppf(1 - alpha / 2, df_resid)
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

    # Relative importance
    if relimp:
        data = pd.concat([pd.DataFrame(y, columns=['y']),
                          pd.DataFrame(X, columns=names)], sort=False, axis=1)
        if 'Intercept' in names:
            # Intercept is the first column
            reli = _relimp(data.drop(columns=['Intercept']).cov())
            reli['names'] = ['Intercept'] + reli['names']
            reli['relimp'] = np.insert(reli['relimp'], 0, np.nan)
            reli['relimp_perc'] = np.insert(reli['relimp_perc'], 0, np.nan)
        else:
            reli = _relimp(data.cov())
        stats.update(reli)

    if as_dataframe:
        stats = _postprocess_dataframe(pd.DataFrame(stats))
        stats.df_model_ = df_model
        stats.df_resid_ = df_resid
        stats.residuals_ = 0  # Trick to avoid Pandas warning
        stats.residuals_ = resid  # Residuals is a hidden attribute
    else:
        stats['df_model'] = df_model
        stats['df_resid'] = df_resid
        stats['residuals'] = resid
        stats['X'] = X
        stats['y'] = y
        stats['pred'] = pred
        if weights is not None:
            stats['yw'] = yw
            stats['Xw'] = Xw
    return stats


def _relimp(S):
    """Relative importance of predictors in multiple regression.

    This is an internal function. This function should only be used with a low
    number of predictors. Indeed, the computation time roughly doubles for each
    additional predictor.

    Parameters
    ----------
    S : pd.DataFrame
        Covariance matrix. The target variable MUST be the FIRST column,
        followed by the predictors (excluding the intercept).
    """
    assert isinstance(S, pd.DataFrame)
    cols = S.columns.tolist()

    # Define indices of columns: .iloc is faster than .loc
    predictors = cols[1:]
    npred = len(predictors)
    target_int = 0
    predictors_int = np.arange(1, npred + 1)

    # Calculate total sum of squares and beta coefficients
    # Note that the R^2 that we calculate below is always the R^2 of the model
    # INCLUDING the intercept!
    ss_tot = S.iat[target_int, target_int]
    betas = (np.linalg.pinv(S.iloc[predictors_int, predictors_int])
             @ S.iloc[predictors_int, target_int])
    r2_full = betas @ S.iloc[target_int, predictors_int] / ss_tot

    # Pre-computed SSreg dictionnary
    ss_reg_precomp = {}

    # Start looping over predictors
    all_preds = []
    for pred in predictors_int:
        loo = np.setdiff1d(predictors_int, pred)
        r2_seq_mean = []
        # Loop over number of predictors
        for k in np.arange(0, npred - 1):
            r2_seq = []
            # Loop over combinations of predictors
            for p in itertools.combinations(loo, int(k)):
                p = list(p)
                p_with = p + [pred]

                # To avoid calculating several times the same values
                # we use a trick here: we save the first calculation
                # to a dictionnary where the key is the sorted string
                # (hence the order does not matter)
                if str(sorted(p)) in ss_reg_precomp.keys():
                    ss_reg_without = ss_reg_precomp[str(sorted(p))]
                else:
                    S_without = S.iloc[p, target_int]
                    ss_reg_without = (np.linalg.pinv(S.iloc[p, p]) @ S_without
                                      @ S_without)
                    ss_reg_precomp[str(sorted(p))] = ss_reg_without

                S_with = S.iloc[p_with, target_int]
                ss_reg_with = (pinvh(S.iloc[p_with, p_with]) @ S_with
                               @ S_with)
                ss_reg_precomp[str(sorted(p_with))] = ss_reg_with

                # Calculate R^2
                r2_diff = (ss_reg_with - ss_reg_without) / ss_tot
                # Append the difference
                r2_seq.append(r2_diff)

            # First averaging
            r2_seq_mean.append(np.mean(r2_seq))

        # When Sk(r) = S
        S_without = S.iloc[loo, target_int]
        ss_reg = np.linalg.pinv(S.iloc[loo, loo]) @ S_without @ S_without
        r2_without = ss_reg / ss_tot
        r2_seq = r2_full - r2_without
        r2_seq_mean.append(r2_seq)
        all_preds.append(np.mean(r2_seq_mean))

    stats_relimp = {'names': predictors,
                    'relimp': all_preds,
                    'relimp_perc': all_preds / sum(all_preds) * 100}

    return stats_relimp


def logistic_regression(X, y, coef_only=False, alpha=0.05,
                        as_dataframe=True, remove_na=False, **kwargs):
    """(Multiple) Binary logistic regression.

    Parameters
    ----------
    X : array_like
        Predictor(s), of shape *(n_samples, n_features)* or *(n_samples)*.
    y : array_like
        Dependent variable, of shape *(n_samples)*.
        ``y`` must be binary, i.e. only contains 0 or 1. Multinomial logistic
        regression is not supported.
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
        :py:class:`sklearn.linear_model.LogisticRegression` (see Notes).

    Returns
    -------
    stats : :py:class:`pandas.DataFrame` or dict
        Logistic regression summary:

        * ``'names'``: name of variable(s) in the model (e.g. x1, x2...)
        * ``'coef'``: regression coefficients (log-odds)
        * ``'se'``: standard error
        * ``'z'``: z-scores
        * ``'pval'``: two-tailed p-values
        * ``'CI[2.5%]'``: lower confidence interval
        * ``'CI[97.5%]'``: upper confidence interval

    See also
    --------
    linear_regression

    Notes
    -----
    .. caution:: This function is a wrapper around the
        :py:class:`sklearn.linear_model.LogisticRegression` class. However,
        Pingouin internally disables the L2 regularization and changes the
        default solver in order to get results that are similar to R and
        statsmodels.

    The logistic regression assumes that the log-odds (the logarithm of the
    odds) for the value labeled "1" in the response variable is a linear
    combination of the predictor variables. The log-odds are given by the
    `logit <https://en.wikipedia.org/wiki/Logit>`_ function,
    which map a probability :math:`p` of the response variable being "1"
    from :math:`[0, 1)` to :math:`(-\\infty, +\\infty)`.

    .. math:: \\text{logit}(p) = \\ln \\frac{p}{1 - p} = \\beta_0 + \\beta X

    The odds of the response variable being "1" can be obtained by
    exponentiating the log-odds:

    .. math:: \\frac{p}{1 - p} = e^{\\beta_0 + \\beta X}

    and the probability of the response variable being "1" is given by the
    `logistic function <https://en.wikipedia.org/wiki/Logistic_function>`_:

    .. math:: p = \\frac{1}{1 + e^{-(\\beta_0 + \\beta X})}

    The first coefficient is always the constant term (intercept) of
    the model. Pingouin will automatically add the intercept
    to your predictor(s) matrix, therefore, :math:`X` should not include a
    constant term. Pingouin will remove any constant term (e.g column with only
    one unique value), or duplicate columns from :math:`X`.

    The calculation of the p-values and confidence interval is adapted from a
    `code by Rob Speare
    <https://gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d>`_.
    Results have been compared against statsmodels, R, and JASP.

    Examples
    --------
    1. Simple binary logistic regression.

    In this first example, we'll use the
    `penguins dataset <https://github.com/allisonhorst/palmerpenguins>`_
    to see how well we can predict the sex of penguins based on their
    bodies mass.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import pingouin as pg
    >>> df = pg.read_dataset('penguins')
    >>> # Let's first convert the target variable from string to boolean:
    >>> df['male'] = (df['sex'] == 'male').astype(int)  # male: 1, female: 0
    >>> # Since there are missing values in our outcome variable, we need to
    >>> # set `remove_na=True` otherwise regression will fail.
    >>> lom = pg.logistic_regression(df['body_mass_g'], df['male'],
    ...                              remove_na=True)
    >>> lom.round(2)
             names  coef    se     z  pval  CI[2.5%]  CI[97.5%]
    0    Intercept -5.16  0.71 -7.24   0.0     -6.56      -3.77
    1  body_mass_g  0.00  0.00  7.24   0.0      0.00       0.00

    Body mass is a significant predictor of sex (p<0.001). Here, it
    could be useful to rescale our predictor variable from *g* to *kg*
    (e.g divide by 1000) in order to get more intuitive coefficients and
    confidence intervals:

    >>> df['body_mass_kg'] = df['body_mass_g'] / 1000
    >>> lom = pg.logistic_regression(df['body_mass_kg'], df['male'],
    ...                              remove_na=True)
    >>> lom.round(2)
              names  coef    se     z  pval  CI[2.5%]  CI[97.5%]
    0     Intercept -5.16  0.71 -7.24   0.0     -6.56      -3.77
    1  body_mass_kg  1.23  0.17  7.24   0.0      0.89       1.56

    2. Multiple binary logistic regression

    We'll now add the species as a categorical predictor in our model. To do
    so, we first need to dummy-code our categorical variable, dropping the
    first level of our categorical variable (species = Adelie) which will be
    used as the reference level:

    >>> df = pd.get_dummies(df, columns=['species'], drop_first=True)
    >>> X = df[['body_mass_kg', 'species_Chinstrap', 'species_Gentoo']]
    >>> y = df['male']
    >>> lom = pg.logistic_regression(X, y, remove_na=True)
    >>> lom.round(2)
                   names   coef    se     z  pval  CI[2.5%]  CI[97.5%]
    0          Intercept -26.24  2.84 -9.24  0.00    -31.81     -20.67
    1       body_mass_kg   7.10  0.77  9.23  0.00      5.59       8.61
    2  species_Chinstrap  -0.13  0.42 -0.31  0.75     -0.96       0.69
    3     species_Gentoo  -9.72  1.12 -8.65  0.00    -11.92      -7.52

    3. Using NumPy aray and returning only the coefficients

    >>> pg.logistic_regression(X.to_numpy(), y.to_numpy(), coef_only=True,
    ...                        remove_na=True)
    array([-26.23906892,   7.09826571,  -0.13180626,  -9.71718529])

    4. Passing custom parameters to sklearn

    >>> lom = pg.logistic_regression(X, y, solver='sag', max_iter=10000,
    ...                           random_state=42, remove_na=True)
    >>> print(lom['coef'].to_numpy())
    [-25.98248153   7.02881472  -0.13119779  -9.62247569]

    **How to interpret the log-odds coefficients?**

    We'll use the `Wikipedia example
    <https://en.wikipedia.org/wiki/Logistic_regression#Probability_of_passing_an_exam_versus_hours_of_study>`_
    of the probability of passing an exam
    versus the hours of study:

    *A group of 20 students spends between 0 and 6 hours studying for an
    exam. How does the number of hours spent studying affect the
    probability of the student passing the exam?*

    >>> # First, let's create the dataframe
    >>> Hours = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
    ...          2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]
    >>> Pass = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    >>> df = pd.DataFrame({'HoursStudy': Hours, 'PassExam': Pass})
    >>> # And then run the logistic regression
    >>> lr = pg.logistic_regression(df['HoursStudy'], df['PassExam']).round(3)
    >>> lr
            names   coef     se      z   pval  CI[2.5%]  CI[97.5%]
    0   Intercept -4.078  1.761 -2.316  0.021    -7.529     -0.626
    1  HoursStudy  1.505  0.629  2.393  0.017     0.272      2.737

    The ``Intercept`` coefficient (-4.078) is the log-odds of ``PassExam=1``
    when ``HoursStudy=0``. The odds ratio can be obtained by exponentiating
    the log-odds:

    >>> np.exp(-4.078)
    0.016941314421496552

    i.e. :math:`0.017:1`. Conversely the odds of failing the exam are
    :math:`(1/0.017) \\approx 59:1`.

    The probability can then be obtained with the following equation

    .. math:: p = \\frac{1}{1 + e^{-(-4.078 + 0 * 1.505)}}

    >>> 1 / (1 + np.exp(-(-4.078)))
    0.016659087580814722

    The ``HoursStudy`` coefficient (1.505) means that for each additional hour
    of study, the log-odds of passing the exam increase by 1.505, and the odds
    are multipled by :math:`e^{1.505} \\approx 4.50`.

    For example, a student who studies 2 hours has a probability of passing
    the exam of 25%:

    >>> 1 / (1 + np.exp(-(-4.078 + 2 * 1.505)))
    0.2557836148964987

    The table below shows the probability of passing the exam for several
    values of ``HoursStudy``:

    +----------------+----------+----------------+------------------+
    | Hours of Study | Log-odds | Odds           | Probability      |
    +================+==========+================+==================+
    | 0              | −4.08    | 0.017 ≈ 1:59   | 0.017            |
    +----------------+----------+----------------+------------------+
    | 1              | −2.57    | 0.076 ≈ 1:13   | 0.07             |
    +----------------+----------+----------------+------------------+
    | 2              | −1.07    | 0.34 ≈ 1:3     | 0.26             |
    +----------------+----------+----------------+------------------+
    | 3              | 0.44     | 1.55           | 0.61             |
    +----------------+----------+----------------+------------------+
    | 4              | 1.94     | 6.96           | 0.87             |
    +----------------+----------+----------------+------------------+
    | 5              | 3.45     | 31.4           | 0.97             |
    +----------------+----------+----------------+------------------+
    | 6              | 4.96     | 141.4          | 0.99             |
    +----------------+----------+----------------+------------------+
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
        # https://stats.stackexchange.com/a/204324/253579
        # Updated in Pingouin > 0.3.6 to be consistent with R
        kwargs['solver'] = 'newton-cg'
    if 'penalty' not in kwargs:
        kwargs['penalty'] = 'none'
    lom = LogisticRegression(**kwargs)
    lom.fit(X, y)

    if lom.get_params()['fit_intercept']:
        names.insert(0, "Intercept")
        X_design = np.column_stack((np.ones(X.shape[0]), X))
        coef = np.append(lom.intercept_, lom.coef_)
    else:
        coef = lom.coef_
        X_design = X

    if coef_only:
        return coef

    # Fisher Information Matrix
    n, p = X_design.shape
    denom = (2 * (1 + np.cosh(lom.decision_function(X))))
    denom = np.tile(denom, (p, 1)).T
    fim = (X_design / denom).T @ X_design
    crao = np.linalg.pinv(fim)

    # Standard error and Z-scores
    se = np.sqrt(np.diag(crao))
    z_scores = coef / se

    # Two-tailed p-values
    pval = 2 * norm.sf(np.fabs(z_scores))

    # Wald Confidence intervals
    # In R: this is equivalent to confint.default(model)
    # Note that confint(model) will however return the profile CI
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
        return _postprocess_dataframe(pd.DataFrame(stats))
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
    data : :py:class:`pandas.DataFrame`
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
    stats : :py:class:`pandas.DataFrame`
        Mediation summary:

        * ``'path'``: regression model
        * ``'coef'``: regression estimates
        * ``'se'``: standard error
        * ``'CI[2.5%]'``: lower confidence interval
        * ``'CI[97.5%]'``: upper confidence interval
        * ``'pval'``: two-sided p-values
        * ``'sig'``: statistical significance

    See also
    --------
    linear_regression, logistic_regression

    Notes
    -----
    Mediation analysis [1]_ is a *"statistical procedure to test
    whether the effect of an independent variable X on a dependent variable
    Y (i.e., X → Y) is at least partly explained by a chain of effects of the
    independent variable on an intervening mediator variable M and of the
    intervening variable on the dependent variable (i.e., X → M → Y)"* [2]_.

    The **indirect effect** (also referred to as average causal mediation
    effect or ACME) of X on Y through mediator M quantifies the estimated
    difference in Y resulting from a one-unit change in X through a sequence of
    causal steps in which X affects M, which in turn affects Y.
    It is considered significant if the specified confidence interval does not
    include 0. The path 'X --> Y' is the sum of both the indirect and direct
    effect. It is sometimes referred to as total effect.

    A linear regression is used if the mediator variable is continuous and a
    logistic regression if the mediator variable is dichotomous (binary).
    Multiple parallel mediators are also supported.

    This function will only work well if the outcome variable is continuous.
    It does not support binary or ordinal outcome variable. For more
    advanced mediation models, please refer to the
    `lavaan <http://lavaan.ugent.be/tutorial/mediation.html>`_ or  `mediation
    <https://cran.r-project.org/web/packages/mediation/mediation.pdf>`_ R
    packages, or the `PROCESS macro
    <https://www.processmacro.org/index.html>`_ for SPSS.

    The two-sided p-value of the indirect effect is computed using the
    bootstrap distribution, as in the mediation R package. However, the p-value
    should be interpreted with caution since it is not constructed
    conditioned on a true null hypothesis [3]_ and varies depending on the
    number of bootstrap samples and the random seed.

    Note that rows with missing values are automatically removed.

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

    Code originally adapted from https://github.com/rmill040/pymediation.

    Examples
    --------
    1. Simple mediation analysis

    >>> from pingouin import mediation_analysis, read_dataset
    >>> df = read_dataset('mediation')
    >>> mediation_analysis(data=df, x='X', m='M', y='Y', alpha=0.05,
    ...                    seed=42)
           path      coef        se          pval  CI[2.5%]  CI[97.5%]  sig
    0     M ~ X  0.561015  0.094480  4.391362e-08  0.373522   0.748509  Yes
    1     Y ~ M  0.654173  0.085831  1.612674e-11  0.483844   0.824501  Yes
    2     Total  0.396126  0.111160  5.671128e-04  0.175533   0.616719  Yes
    3    Direct  0.039604  0.109648  7.187429e-01 -0.178018   0.257226   No
    4  Indirect  0.356522  0.083313  0.000000e+00  0.219818   0.537654  Yes

    2. Return the indirect bootstrapped beta coefficients

    >>> stats, dist = mediation_analysis(data=df, x='X', m='M', y='Y',
    ...                                  return_dist=True)
    >>> print(dist.shape)
    (500,)

    3. Mediation analysis with a binary mediator variable

    >>> mediation_analysis(data=df, x='X', m='Mbin', y='Y', seed=42).round(3)
           path   coef     se   pval  CI[2.5%]  CI[97.5%]  sig
    0  Mbin ~ X -0.021  0.116  0.857    -0.248      0.206   No
    1  Y ~ Mbin -0.135  0.412  0.743    -0.952      0.682   No
    2     Total  0.396  0.111  0.001     0.176      0.617  Yes
    3    Direct  0.396  0.112  0.001     0.174      0.617  Yes
    4  Indirect  0.002  0.050  0.960    -0.072      0.146   No

    4. Mediation analysis with covariates

    >>> mediation_analysis(data=df, x='X', m='M', y='Y',
    ...                    covar=['Mbin', 'Ybin'], seed=42).round(3)
           path   coef     se   pval  CI[2.5%]  CI[97.5%]  sig
    0     M ~ X  0.559  0.097  0.000     0.367      0.752  Yes
    1     Y ~ M  0.666  0.086  0.000     0.495      0.837  Yes
    2     Total  0.420  0.113  0.000     0.196      0.645  Yes
    3    Direct  0.064  0.110  0.561    -0.155      0.284   No
    4  Indirect  0.356  0.086  0.000     0.209      0.553  Yes

    5. Mediation analysis with multiple parallel mediators

    >>> mediation_analysis(data=df, x='X', m=['M', 'Mbin'], y='Y',
    ...                    seed=42).round(3)
                path   coef     se   pval  CI[2.5%]  CI[97.5%]  sig
    0          M ~ X  0.561  0.094  0.000     0.374      0.749  Yes
    1       Mbin ~ X -0.005  0.029  0.859    -0.063      0.052   No
    2          Y ~ M  0.654  0.086  0.000     0.482      0.825  Yes
    3       Y ~ Mbin -0.064  0.328  0.846    -0.715      0.587   No
    4          Total  0.396  0.111  0.001     0.176      0.617  Yes
    5         Direct  0.040  0.110  0.721    -0.179      0.258   No
    6     Indirect M  0.356  0.085  0.000     0.215      0.538  Yes
    7  Indirect Mbin  0.000  0.010  0.952    -0.017      0.025   No
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
    assert all([data[c].dtype.kind in 'bfiu' for c in columns]), err_msg

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
    X_val = data[_fl([x, covar])].to_numpy()  # X + covar as predictors
    XM_val = data[_fl([x, m, covar])].to_numpy()  # X + M + covar as predictors
    M_val = data[m].to_numpy()  # M as target (no covariates)
    y_val = data[y].to_numpy()  # y as target (no covariates)

    # For max precision, make sure rounding is disabled
    old_options = options.copy()
    options['round'] = None

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

    # Restore options
    options.update(old_options)

    if return_dist:
        return _postprocess_dataframe(stats), np.squeeze(ab_estimates)
    else:
        return _postprocess_dataframe(stats)
