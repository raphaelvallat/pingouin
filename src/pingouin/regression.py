import warnings

import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.linalg import pinvh
from scipy.stats import norm, t

from .config import _no_rounding
from .utils import _flatten_list as _fl
from .utils import _postprocess_dataframe
from .utils import remove_na as rm_na

__all__ = ["linear_regression", "logistic_regression", "mediation_analysis"]


def linear_regression(
    X,
    y,
    add_intercept=True,
    weights=None,
    coef_only=False,
    alpha=0.05,
    as_dataframe=True,
    remove_na=False,
    relimp=False,
):
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
        * ``'CI2.5'``: lower confidence intervals
        * ``'CI97.5'``: upper confidence intervals
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
    squares (OLS) regression, computed from a singular value decomposition of
    the design matrix. The OLS method minimizes
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

    If one or more columns of :math:`X` are a linear combination of the
    others (e.g. duplicate columns, all-zero columns, or a constant column
    in addition to the intercept), the design matrix is rank deficient. A
    warning is raised, singular values below
    :math:`\\max(n, p) \\cdot \\epsilon \\cdot \\sigma_{\\max}` are treated as zero (the
    tolerance of :py:func:`numpy.linalg.matrix_rank`), and the minimum-norm
    solution is returned, as in statsmodels. The coefficients of the collinear
    columns are then not unique: for example, the effect of a duplicated
    column is split equally between the two copies, with the same T-value
    as when the column is only included once. The T-value and p-value of an
    all-zero column are NaN.

    .. versionchanged:: 0.7.0
        Duplicate, all-zero and extra constant columns are no longer removed
        from :math:`X`.

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
            names  coef    se      T  pval    r2  adj_r2     CI2.5     CI97.5
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
            names  coef    se      T  pval    r2  adj_r2     CI2.5     CI97.5
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
           names  coef    se      T  pval    r2  adj_r2     CI2.5     CI97.5
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
    dict_keys(['names', 'coef', 'se', 'T', 'pval', 'r2', 'adj_r2', 'CI2.5',
               'CI97.5', 'df_model', 'df_resid', 'residuals', 'X', 'y',
               'pred'])

    7. Remove missing values

    >>> X, y = X.copy(), y.copy()  # to_numpy() can return a read-only view
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
           names  coef    se     T  pval    r2  adj_r2     CI2.5     CI97.5
    0  Intercept  9.00  2.03  4.42  0.01  0.51    0.39      3.35      14.64
    1         x1  1.04  0.50  2.06  0.11  0.51    0.39     -0.36       2.44
    """
    assert 0 < alpha < 1
    X, y, names = _prepare_Xy(X, y, remove_na)

    if add_intercept:
        # Add intercept
        X = np.column_stack((np.ones(X.shape[0]), X))
        names.insert(0, "Intercept")

    # Is there a constant (e.g. the intercept) in the design? Used for the dof and R^2. No column
    # is removed: all-zero, duplicate or collinear constant columns make the design rank
    # deficient, which is handled below with a single rank decision.
    is_const = (np.ptp(X, axis=0) == 0) & (X[0] != 0)
    constant = int(is_const.any())

    # Check that we have enough samples / features
    n, p = X.shape[0], X.shape[1]
    assert n >= 3, "At least three valid samples are required in X."
    assert p >= 1, "X must have at least one valid column."

    # Handle weights
    if weights is not None:
        if relimp:
            raise ValueError("relimp = True is not supported when using weights.")
        w = np.asarray(weights)
        assert w.ndim == 1, "weights must be a 1D array."
        assert w.size == n, "weights must be of shape n_samples."
        assert not np.isnan(w).any(), "Missing weights are not accepted."
        assert not (w < 0).any(), "Negative weights are not accepted."
        # Do not count weights == 0 in dof
        # This gives similar results as R lm() but different from statsmodels
        n = np.count_nonzero(w)
        # Rescale (whitening). Broadcasting avoids building a dense (n, n) diagonal matrix.
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
    else:
        # Set all weights to one, [1, 1, 1, ...]
        w = np.ones(n)
        Xw = X
        yw = y

    # FIT (WEIGHTED) LEAST SQUARES REGRESSION
    coef, unscaled_var, rank = _lstsq(Xw, yw)
    if coef_only:
        return coef
    if rank < Xw.shape[1]:
        warnings.warn(
            "Design matrix supplied with `X` parameter is rank "
            f"deficient (rank {rank} with {Xw.shape[1]} columns). "
            "That means that one or more of the columns in `X` "
            "are a linear combination of one of more of the "
            "other columns."
        )

    # Degrees of freedom
    df_model = rank - constant
    df_resid = n - rank

    # Calculate predicted values and (weighted) residuals
    pred = Xw @ coef
    resid = yw - pred
    ss_res = (resid**2).sum()

    # Calculate total (weighted) sums of squares and R^2
    ss_tot = yw @ yw
    ss_wtot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    if constant:
        r2 = 1 - ss_res / ss_wtot
    else:
        r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - constant) / df_resid

    # Compute mean squared error, variance and SE
    mse = ss_res / df_resid
    beta_se = np.sqrt(mse * unscaled_var)

    # Compute T and p-values. The coefficient and SE of an all-zero column are both zero, which
    # gives a NaN T-value and p-value.
    with np.errstate(divide="ignore", invalid="ignore"):
        T = coef / beta_se
    pval = 2 * t.sf(np.fabs(T), df_resid)

    # Compute confidence intervals
    crit = t.ppf(1 - alpha / 2, df_resid)
    marg_error = crit * beta_se
    ll = coef - marg_error
    ul = coef + marg_error

    # Rename CI
    ll_name = "CI%.1f" % (100 * alpha / 2)
    ul_name = "CI%.1f" % (100 * (1 - alpha / 2))

    # Create dict
    stats = {
        "names": names,
        "coef": coef,
        "se": beta_se,
        "T": T,
        "pval": pval,
        "r2": r2,
        "adj_r2": adj_r2,
        ll_name: ll,
        ul_name: ul,
    }

    # Relative importance
    if relimp:
        # Relative importance is computed on the correlation matrix, which makes it
        # invariant to the scale of the predictors. The correlation of a constant
        # column (e.g. the Intercept) is undefined, so the constant columns are
        # excluded and re-inserted afterwards.
        idx_const = np.flatnonzero(np.ptp(X, axis=0) == 0)
        idx_var = np.flatnonzero(np.ptp(X, axis=0) != 0)
        S = pd.DataFrame(
            np.corrcoef(np.column_stack((y, X[:, idx_var])), rowvar=False),
            columns=["y", *[names[i] for i in idx_var]],
        )
        reli = _relimp(S)
        for i in idx_const:
            # The intercept has no relative importance, and a user-defined constant
            # column explains no variance in y.
            fill = np.nan if add_intercept and i == 0 else 0.0
            reli["names"].insert(i, names[i])
            reli["relimp"] = np.insert(reli["relimp"], i, fill)
            reli["relimp_perc"] = np.insert(reli["relimp_perc"], i, fill)
        stats.update(reli)

    if as_dataframe:
        stats = _postprocess_dataframe(pd.DataFrame(stats))
        stats.df_model_ = df_model
        stats.df_resid_ = df_resid
        stats.residuals_ = 0  # Trick to avoid Pandas warning
        stats.residuals_ = resid  # Residuals is a hidden attribute
    else:
        stats["df_model"] = df_model
        stats["df_resid"] = df_resid
        stats["residuals"] = resid
        stats["X"] = X
        stats["y"] = y
        stats["pred"] = pred
        if weights is not None:
            stats["yw"] = yw
            stats["Xw"] = Xw
    return stats


def _prepare_Xy(X, y, remove_na=False):
    """Validate the predictors and target of a regression.

    Returns ``X`` as a (n_samples, n_features) array, ``y`` as a (n_samples,) array and the names
    of the predictors, which are extracted from ``X`` if it is a DataFrame or a Series.
    """
    # Extract names if X is a Dataframe or Series
    if isinstance(X, pd.DataFrame):
        names = X.keys().tolist()
    elif isinstance(X, pd.Series):
        names = [X.name]
    else:
        names = []

    # Convert input to numpy array. X and y are cast to float, e.g. for boolean inputs.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    assert y.ndim == 1, "y must be one-dimensional."

    if X.ndim == 1:
        # Convert to (n_samples, n_features) shape
        X = X[..., np.newaxis]

    # Check for NaN / Inf
    if remove_na:
        X, y = rm_na(X, y[..., np.newaxis], paired=True, axis="rows")
        y = np.squeeze(y)
    assert np.isfinite(y).all(), (
        "Target (y) contains NaN or Inf. Please remove them manually or use remove_na=True."
    )
    assert np.isfinite(X).all(), (
        "Predictors (X) contain NaN or Inf. Please remove them manually or use remove_na=True."
    )

    # Check that X and y have same length
    assert y.shape[0] == X.shape[0], "X and y must have same number of samples"

    if not names:
        names = ["x" + str(i + 1) for i in range(X.shape[1])]
    return X, y, names


def _lstsq(X, y=None):
    """Minimum-norm least squares solution and unscaled variance of the coefficients.

    Returns the coefficients of ``y ~ X`` (None if ``y`` is None), the diagonal of the
    pseudo-inverse of ``X.T @ X`` and the rank of ``X``. The covariance of the coefficients is
    the diagonal multiplied by the residual variance (linear regression) or, if ``X`` is
    whitened by the square root of the IRLS weights, it is the covariance itself (logistic
    regression).

    Inverting ``X.T @ X`` squares the condition number of ``X`` and discards estimable
    directions when the predictors have different units. Instead, singular values of ``X``
    below ``max(n, p) * eps * s_max`` are treated as zero (same tolerance as
    :py:func:`numpy.linalg.matrix_rank`) and the covariance is formed from the others. This
    single rank decision handles duplicate and collinear columns alike.

    All-zero columns are excluded from the decomposition, and their coefficient and variance are
    set to exactly zero. Otherwise, the rounding noise of the SVD leaks into these columns and
    gives a coefficient and a variance of ~1e-17, whose ratio (the T-value) is arbitrary.

    The SVD is computed on the small triangular factor of a QR decomposition of ``X``, which has
    the same singular values and is much cheaper than an SVD of a tall ``X``. Appending ``y`` to
    ``X`` before the QR gives ``Q.T @ y`` without forming ``Q``.
    """
    n, p = X.shape
    coef, unscaled_var = np.zeros(p), np.zeros(p)
    nonzero = X.any(axis=0)
    if not nonzero.any():
        return (None if y is None else coef), unscaled_var, 0
    Xnz = X[:, nonzero] if not nonzero.all() else X
    A = Xnz if y is None else np.column_stack((Xnz, y))
    R = np.linalg.qr(A.astype(float, copy=False), mode="r")
    u, s, vt = np.linalg.svd(R[:, : Xnz.shape[1]], full_matrices=False)
    rank = int(np.sum(s > max(n, p) * np.finfo(float).eps * s[0]))
    scaled_vt = vt[:rank] / s[:rank, np.newaxis]
    unscaled_var[nonzero] = np.sum(scaled_vt**2, axis=0)
    if y is None:
        return None, unscaled_var, rank
    coef[nonzero] = scaled_vt.T @ (u[:, :rank].T @ R[:, -1])
    return coef, unscaled_var, rank


def _duplicate_columns(X):
    """Indices of the columns of ``X`` that are exact duplicates of an earlier column."""
    # Duplicate columns have the same weighted sum, up to a rounding error much smaller than
    # the weighted sum of absolute values. Only the columns whose sums are within that
    # tolerance need to be compared element-wise.
    w = np.linspace(1, 2, X.shape[0])
    fingerprint, tol = w @ X, 1e-8 * (w @ np.abs(X))
    idx_duplicate = []
    for j in range(1, X.shape[1]):
        candidates = np.flatnonzero(np.abs(fingerprint[:j] - fingerprint[j]) <= tol[j])
        if any(np.array_equal(X[:, i], X[:, j]) for i in candidates):
            idx_duplicate.append(j)
    return np.array(idx_duplicate, dtype=int)


def _relimp(S):
    """Relative importance of predictors in multiple regression.

    This is an internal function. This function should only be used with a low
    number of predictors. Indeed, the computation time roughly doubles for each
    additional predictor.

    Parameters
    ----------
    S : pd.DataFrame
        Correlation matrix. The target variable MUST be the FIRST column,
        followed by the predictors (excluding the intercept).
    """
    assert isinstance(S, pd.DataFrame)
    predictors = S.columns[1:].tolist()
    npred = len(predictors)
    S = S.to_numpy()
    ss_tot, r, Rxx = S[0, 0], S[1:, 0], S[1:, 1:]

    # R^2 (including the intercept) of the model fitted on each subset of predictors, where
    # the subset is encoded as a bitmask: bit j is set if predictor j is in the model.
    # Each subset is solved only once.
    masks = np.arange(2**npred)
    r2 = np.zeros(2**npred)
    for mask in masks[1:]:
        idx = np.flatnonzero((mask >> np.arange(npred)) & 1)
        r2[mask] = r[idx] @ pinvh(Rxx[np.ix_(idx, idx)]) @ r[idx] / ss_tot
    size = np.array([bin(mask).count("1") for mask in masks])

    # LMG: increase in R^2 when adding predictor j, averaged first over all the subsets of the
    # other predictors with a given size, and then over the subset sizes.
    all_preds = []
    for j in range(npred):
        without = masks[(masks >> j) & 1 == 0]
        r2_diff = r2[without | (1 << j)] - r2[without]
        all_preds.append(np.mean([r2_diff[size[without] == k].mean() for k in range(npred)]))

    stats_relimp = {
        "names": predictors,
        "relimp": all_preds,
        "relimp_perc": np.array(all_preds) / sum(all_preds) * 100,
    }

    return stats_relimp


def logistic_regression(
    X, y, coef_only=False, alpha=0.05, as_dataframe=True, remove_na=False, **kwargs
):
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
        * ``'CI2.5'``: lower confidence interval
        * ``'CI97.5'``: upper confidence interval

    See also
    --------
    linear_regression

    Notes
    -----
    .. caution:: This function is a wrapper around the
        :py:class:`sklearn.linear_model.LogisticRegression` class. However,
        Pingouin internally disables the L2 regularization and changes the
        default solver to 'newton-cg' to obtain results that are similar to R and
        statsmodels.

    Logistic regression assumes that the log-odds (the logarithm of the
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
    one unique value), or duplicate columns from :math:`X`. If ``fit_intercept=False``
    is passed to scikit-learn, the first non-zero constant column of :math:`X` is kept
    and acts as the intercept.

    .. versionchanged:: 0.7.0
        With ``fit_intercept=False``, the first non-zero constant column is no longer
        removed.

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
    >>> df = pg.read_dataset("penguins")
    >>> # Let's first convert the target variable from string to boolean:
    >>> df["male"] = (df["sex"] == "male").astype(int)  # male: 1, female: 0
    >>> # Since there are missing values in our outcome variable, we need to
    >>> # set `remove_na=True` otherwise regression will fail.
    >>> lom = pg.logistic_regression(df["body_mass_g"], df["male"], remove_na=True)
    >>> lom.round(2)
             names  coef    se     z  pval     CI2.5     CI97.5
    0    Intercept -5.16  0.71 -7.24   0.0     -6.56      -3.77
    1  body_mass_g  0.00  0.00  7.24   0.0      0.00       0.00

    Body mass is a significant predictor of sex (p<0.001). Here, it
    could be useful to rescale our predictor variable from *g* to *kg*
    (e.g divide by 1000) in order to get more intuitive coefficients and
    confidence intervals:

    >>> df["body_mass_kg"] = df["body_mass_g"] / 1000
    >>> lom = pg.logistic_regression(df["body_mass_kg"], df["male"], remove_na=True)
    >>> lom.round(2)
              names  coef    se     z  pval     CI2.5     CI97.5
    0     Intercept -5.16  0.71 -7.24   0.0     -6.56      -3.77
    1  body_mass_kg  1.23  0.17  7.24   0.0      0.89       1.56

    2. Multiple binary logistic regression

    We'll now add the species as a categorical predictor in our model. To do
    so, we first need to dummy-code our categorical variable, dropping the
    first level of our categorical variable (species = Adelie) which will be
    used as the reference level:

    >>> df = pd.get_dummies(df, columns=["species"], dtype=float, drop_first=True)
    >>> X = df[["body_mass_kg", "species_Chinstrap", "species_Gentoo"]]
    >>> y = df["male"]
    >>> lom = pg.logistic_regression(X, y, remove_na=True)
    >>> lom.round(2)
                   names   coef    se     z  pval     CI2.5     CI97.5
    0          Intercept -26.24  2.84 -9.24  0.00    -31.81     -20.67
    1       body_mass_kg   7.10  0.77  9.23  0.00      5.59       8.61
    2  species_Chinstrap  -0.13  0.42 -0.31  0.75     -0.96       0.69
    3     species_Gentoo  -9.72  1.12 -8.65  0.00    -11.92      -7.52

    3. Using NumPy aray and returning only the coefficients

    >>> pg.logistic_regression(X.to_numpy(), y.to_numpy(), coef_only=True, remove_na=True).round(2)
    array([-26.24,   7.1 ,  -0.13,  -9.72])

    4. Passing custom parameters to sklearn

    >>> lom = pg.logistic_regression(
    ...     X, y, solver="sag", max_iter=10000, random_state=42, remove_na=True
    ... )
    >>> print(lom["coef"].to_numpy())
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
    >>> Hours = [
    ...     0.50,
    ...     0.75,
    ...     1.00,
    ...     1.25,
    ...     1.50,
    ...     1.75,
    ...     1.75,
    ...     2.00,
    ...     2.25,
    ...     2.50,
    ...     2.75,
    ...     3.00,
    ...     3.25,
    ...     3.50,
    ...     4.00,
    ...     4.25,
    ...     4.50,
    ...     4.75,
    ...     5.00,
    ...     5.50,
    ... ]
    >>> Pass = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    >>> df = pd.DataFrame({"HoursStudy": Hours, "PassExam": Pass})
    >>> # And then run the logistic regression
    >>> lr = pg.logistic_regression(df["HoursStudy"], df["PassExam"]).round(3)
    >>> lr
            names   coef     se      z   pval     CI2.5     CI97.5
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
    from sklearn.linear_model import LogisticRegression

    assert 0 < alpha < 1, "alpha must be between 0 and 1."
    X, y, names = _prepare_Xy(X, y, remove_na)

    # Check that y is binary
    if np.unique(y).size != 2:
        raise ValueError("Dependent variable must be binary.")

    # Remove the constant columns, which are collinear with the intercept, and the duplicate
    # columns. Unlike linear_regression, scikit-learn does not return the minimum-norm solution
    # for a rank-deficient design. Without the intercept of scikit-learn, the first non-zero
    # constant column is kept, since it is then the intercept of the model.
    idx_unique = np.flatnonzero(np.ptp(X, axis=0) == 0)
    if not kwargs.get("fit_intercept", True):
        idx_unique = np.delete(idx_unique, np.flatnonzero(X[0, idx_unique] != 0)[:1])
    if len(idx_unique):
        X = np.delete(X, idx_unique, 1)
        names = np.delete(names, idx_unique).tolist()

    idx_duplicate = _duplicate_columns(X)
    if len(idx_duplicate):
        X = np.delete(X, idx_duplicate, 1)
        names = np.delete(names, idx_duplicate).tolist()

    # Initialize and fit
    if "solver" not in kwargs:
        # https://stats.stackexchange.com/a/204324/253579
        # Updated in Pingouin > 0.3.6 to be consistent with R
        kwargs["solver"] = "newton-cg"
        # The default tol=1e-4 of scikit-learn stops before convergence to the maximum
        # likelihood estimates (e.g. 3rd decimal of the coefficients)
        kwargs.setdefault("tol", 1e-8)
    if "penalty" not in kwargs and "C" not in kwargs:
        # No regularization. C=np.inf is equivalent to penalty=None, which is deprecated in
        # sklearn 1.8
        kwargs["C"] = np.inf
    lom = LogisticRegression(**kwargs)
    with warnings.catch_warnings():
        # sklearn 1.8 maps C=np.inf to penalty=None and then warns that C is ignored
        warnings.filterwarnings("ignore", message="Setting penalty=None will ignore")
        lom.fit(X, y)

    if lom.get_params()["fit_intercept"]:
        names.insert(0, "Intercept")
        X_design = np.column_stack((np.ones(X.shape[0]), X))
        coef = np.append(lom.intercept_, lom.coef_)
    else:
        # coef_ has shape (1, n_features)
        coef = lom.coef_.ravel()
        X_design = X

    if coef_only:
        return coef

    # The covariance of the coefficients is the inverse of the Fisher information matrix
    # X.T @ W @ X, with W = p * (1 - p) = 1 / (2 * (1 + cosh(logit))). It is computed from the
    # whitened design sqrt(W) @ X to avoid squaring its condition number.
    denom = 2 * (1 + np.cosh(lom.decision_function(X)))
    _, var, _ = _lstsq(X_design / np.sqrt(denom)[:, None])

    # Standard error and Z-scores
    se = np.sqrt(var)
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
    ll_name = "CI%.1f" % (100 * alpha / 2)
    ul_name = "CI%.1f" % (100 * (1 - alpha / 2))

    # Create dict
    stats = {
        "names": names,
        "coef": coef,
        "se": se,
        "z": z_scores,
        "pval": pval,
        ll_name: ll,
        ul_name: ul,
    }
    if as_dataframe:
        return _postprocess_dataframe(pd.DataFrame(stats))
    else:
        return stats


def _ols_coef(X, y):
    """Least squares coefficients of y ~ 1 + X, without the input checks of linear_regression.

    No column is removed from the design matrix, so that the position of each coefficient is
    fixed across bootstrap samples, even when a covariate is constant in a given sample.
    """
    return _lstsq(np.column_stack((np.ones(X.shape[0]), X)), y)[0]


def _point_estimate(X_val, XM_val, M_val, y_val, idx, n_mediator, m_binary, **logreg_kwargs):
    """Point estimate of indirect effect based on bootstrap sample.

    ``m_binary`` is a boolean array indicating, for each mediator, whether it is binary and
    thus modeled with a logistic regression instead of a linear regression.
    """
    # Mediator(s) model (M(j) ~ X + covar)
    beta_m = []
    for j in range(n_mediator):
        if not m_binary[j]:
            beta_m.append(_ols_coef(X_val[idx], M_val[idx, j])[1])
        else:
            beta_m.append(
                logistic_regression(X_val[idx], M_val[idx, j], coef_only=True, **logreg_kwargs)[1]
            )

    # Full model (Y ~ X + M + covar)
    beta_y = _ols_coef(XM_val[idx], y_val[idx])[2 : (2 + n_mediator)]

    # Point estimate
    return beta_m * beta_y


def _bias_corrected_ci(bootdist, sample_point, alpha=0.05):
    """Bias-corrected confidence intervals

    Parameters
    ----------
    bootdist : array-like
        Array with bootstrap estimates for each sample.
    sample_point : float
        Point estimate based on full sample.
    alpha : float
        Alpha for confidence interval.

    Returns
    -------
    CI : 1d array-like
        Lower and upper bias-corrected confidence interval estimates.

    Notes
    -----
    This is what's used in the "cper" method implemented in :py:func:`pingouin.compute_bootci`.

    This differs from the bias-corrected and accelerated method (BCa, default in Matlab and
    SciPy) because it does not correct for skewness. Indeed, the acceleration parameter, a,
    is proportional to the skewness of the bootstrap distribution. The bias-correction parameter,
    z0, is related to the proportion of bootstrap estimates that are less than the observed
    statistic.
    """
    # Bias of bootstrap estimates
    # In Matlab bootci they also count when bootdist == sample_point
    # >>> z0 = norminv(mean(bstat < stat,1) + mean(bstat == stat,1)/2);
    z0 = norm.ppf(np.mean(bootdist < sample_point))
    # Adjusted intervals
    adjusted_ll = norm.cdf(2 * z0 + norm.ppf(alpha / 2)) * 100
    adjusted_ul = norm.cdf(2 * z0 + norm.ppf(1 - alpha / 2)) * 100
    # Adjusted percentiles
    ci = np.percentile(bootdist, q=[adjusted_ll, adjusted_ul])
    return ci


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
def mediation_analysis(
    data=None,
    x=None,
    m=None,
    y=None,
    covar=None,
    alpha=0.05,
    n_boot=500,
    seed=None,
    return_dist=False,
    logreg_kwargs=None,
):
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
    logreg_kwargs : dict or None
        Dictionary with optional arguments passed to :py:func:`pingouin.logistic_regression`
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
        * ``'CI2.5'``: lower confidence interval
        * ``'CI97.5'``: upper confidence interval
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
    Multiple parallel mediators are also supported, and the type of regression is chosen
    separately for each mediator.

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
    >>> df = read_dataset("mediation")
    >>> mediation_analysis(data=df, x="X", m="M", y="Y", alpha=0.05, seed=42)
           path      coef        se          pval     CI2.5     CI97.5  sig
    0     M ~ X  0.561015  0.094480  4.391362e-08  0.373522   0.748509  Yes
    1     Y ~ M  0.654173  0.085831  1.612674e-11  0.483844   0.824501  Yes
    2     Total  0.396126  0.111160  5.671128e-04  0.175533   0.616719  Yes
    3    Direct  0.039604  0.109648  7.187429e-01 -0.178018   0.257226   No
    4  Indirect  0.356522  0.083313  0.000000e+00  0.219818   0.537654  Yes

    2. Return the indirect bootstrapped beta coefficients

    >>> stats, dist = mediation_analysis(data=df, x="X", m="M", y="Y", return_dist=True)
    >>> print(dist.shape)
    (500,)

    3. Mediation analysis with a binary mediator variable

    >>> mediation_analysis(data=df, x="X", m="Mbin", y="Y", seed=42).round(3)
           path   coef     se   pval     CI2.5     CI97.5  sig
    0  Mbin ~ X -0.021  0.116  0.858    -0.248      0.206   No
    1  Y ~ Mbin -0.135  0.412  0.743    -0.952      0.682   No
    2     Total  0.396  0.111  0.001     0.176      0.617  Yes
    3    Direct  0.396  0.112  0.001     0.174      0.617  Yes
    4  Indirect  0.002  0.050  0.960    -0.072      0.146   No

    4. Mediation analysis with covariates

    >>> mediation_analysis(data=df, x="X", m="M", y="Y", covar=["Mbin", "Ybin"], seed=42).round(3)
           path   coef     se   pval     CI2.5     CI97.5  sig
    0     M ~ X  0.559  0.097  0.000     0.367      0.752  Yes
    1     Y ~ M  0.666  0.086  0.000     0.495      0.837  Yes
    2     Total  0.420  0.113  0.000     0.196      0.645  Yes
    3    Direct  0.064  0.110  0.561    -0.155      0.284   No
    4  Indirect  0.356  0.086  0.000     0.209      0.553  Yes

    5. Mediation analysis with multiple parallel mediators

    >>> mediation_analysis(data=df, x="X", m=["M", "Mbin"], y="Y", seed=42).round(3)
                path   coef     se   pval     CI2.5     CI97.5  sig
    0          M ~ X  0.561  0.094  0.000     0.374      0.749  Yes
    1       Mbin ~ X -0.021  0.116  0.858    -0.248      0.206   No
    2          Y ~ M  0.654  0.086  0.000     0.482      0.825  Yes
    3       Y ~ Mbin -0.064  0.328  0.846    -0.715      0.587   No
    4          Total  0.396  0.111  0.001     0.176      0.617  Yes
    5         Direct  0.040  0.110  0.721    -0.179      0.258   No
    6     Indirect M  0.356  0.085  0.000     0.215      0.538  Yes
    7  Indirect Mbin  0.001  0.041  0.952    -0.071      0.108   No
    """
    # Sanity check
    assert isinstance(x, (str, int)), "y must be a string or int."
    assert isinstance(y, (str, int)), "y must be a string or int."
    assert isinstance(m, (list, str, int)), "Mediator(s) must be a list, string or int."
    assert isinstance(covar, (type(None), str, list, int))
    if isinstance(m, (str, int)):
        m = [m]
    n_mediator = len(m)
    assert isinstance(data, pd.DataFrame), "Data must be a DataFrame."
    # Check for duplicates
    assert n_mediator == len(set(m)), "Cannot have duplicates mediators."
    if isinstance(covar, (str, int)):
        covar = [covar]
    if isinstance(covar, list):
        assert len(covar) == len(set(covar)), "Cannot have duplicates covar."
        assert set(m).isdisjoint(covar), "Mediator cannot be in covar."
    # Check that columns are in dataframe
    columns = _fl([x, m, y, covar])
    keys = data.columns
    assert all([c in keys for c in columns]), "Column(s) are not in DataFrame."
    # Check that columns are numeric
    err_msg = "Columns must be numeric or boolean."
    assert all([data[c].dtype.kind in "bfiu" for c in columns]), err_msg

    # Drop rows with NAN Values
    data = data[columns].dropna()
    n = data.shape[0]
    assert n > 5, "DataFrame must have at least 5 samples (rows)."

    # Check which mediator(s) are binary (logistic regression) or continuous (linear regression)
    m_binary = (data[m].nunique() == 2).to_numpy()

    # Check if a dict with kwargs for logistic_regression has been passed
    logreg_kwargs = {} if logreg_kwargs is None else logreg_kwargs

    # Name of CI
    ll_name = "CI%.1f" % (100 * alpha / 2)
    ul_name = "CI%.1f" % (100 * (1 - alpha / 2))

    # Compute regressions
    cols = ["names", "coef", "se", "pval", ll_name, ul_name]

    # For speed, we pass np.array instead of pandas DataFrame
    X_val = data[_fl([x, covar])].to_numpy()  # X + covar as predictors
    XM_val = data[_fl([x, m, covar])].to_numpy()  # X + M + covar as predictors
    M_val = data[m].to_numpy()  # M as target (no covariates)
    y_val = data[y].to_numpy()  # y as target (no covariates)

    with _no_rounding():  # For max precision
        # M(j) ~ X + covar
        sxm = {}
        for idx, j in enumerate(m):
            if not m_binary[idx]:
                sxm[j] = linear_regression(X_val, M_val[:, idx], alpha=alpha).loc[[1], cols]
            else:
                sxm[j] = logistic_regression(
                    X_val, M_val[:, idx], alpha=alpha, **logreg_kwargs
                ).loc[[1], cols]
            sxm[j].at[1, "names"] = "%s ~ X" % j
        sxm = pd.concat(sxm, ignore_index=True)

        # Y ~ M + covar
        smy = linear_regression(data[_fl([m, covar])], y_val, alpha=alpha).loc[1:n_mediator, cols]
        # Average Total Effects (Y ~ X + covar)
        sxy = linear_regression(X_val, y_val, alpha=alpha).loc[[1], cols]
        # Average Direct Effects (Y ~ X + M + covar)
        direct = linear_regression(XM_val, y_val, alpha=alpha).loc[[1], cols]

        # Rename paths
        smy["names"] = smy["names"].apply(lambda x: "Y ~ %s" % x)
        direct.at[1, "names"] = "Direct"
        sxy.at[1, "names"] = "Total"

        # Concatenate and create sig column
        stats = pd.concat((sxm, smy, sxy, direct), ignore_index=True)
        stats["sig"] = np.where(stats["pval"] < alpha, "Yes", "No")

        # Bootstrap confidence intervals
        rng = np.random.RandomState(seed)
        idx = rng.choice(np.arange(n), replace=True, size=(n_boot, n))
        ab_estimates = np.zeros(shape=(n_boot, n_mediator))
        for i in range(n_boot):
            ab_estimates[i, :] = _point_estimate(
                X_val, XM_val, M_val, y_val, idx[i, :], n_mediator, m_binary, **logreg_kwargs
            )

        ab = _point_estimate(
            X_val, XM_val, M_val, y_val, np.arange(n), n_mediator, m_binary, **logreg_kwargs
        )
        indirect = {
            "names": m,
            "coef": ab,
            "se": ab_estimates.std(ddof=1, axis=0),
            "pval": [],
            ll_name: [],
            ul_name: [],
            "sig": [],
        }

        for j in range(n_mediator):
            ci_j = _bias_corrected_ci(ab_estimates[:, j], indirect["coef"][j], alpha=alpha)
            indirect[ll_name].append(min(ci_j))
            indirect[ul_name].append(max(ci_j))
            # Bootstrapped p-value of indirect effect
            # Note that this is less accurate than a permutation test because the
            # bootstrap distribution is not conditioned on a true null hypothesis.
            # For more details see Hayes and Rockwood 2017
            indirect["pval"].append(_pval_from_bootci(ab_estimates[:, j], indirect["coef"][j]))
            indirect["sig"].append("Yes" if indirect["pval"][j] < alpha else "No")

        # Create output dataframe
        indirect = pd.DataFrame.from_dict(indirect)
        if n_mediator == 1:
            indirect["names"] = "Indirect"
        else:
            indirect["names"] = indirect["names"].apply(lambda x: "Indirect %s" % x)
        stats = pd.concat([stats, indirect], axis=0, ignore_index=True, sort=False)
        stats = stats.rename(columns={"names": "path"})

    if return_dist:
        return _postprocess_dataframe(stats), np.squeeze(ab_estimates)
    else:
        return _postprocess_dataframe(stats)
