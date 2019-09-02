import numpy as np
import pandas as pd
from scipy.stats import f

__all__ = ["cronbach_alpha", "intraclass_corr"]


def cronbach_alpha(data=None, items=None, scores=None, subject=None,
                   nan_policy='pairwise', ci=.95):
    """Cronbach's alpha reliability measure.

    Parameters
    ----------
    data : pandas dataframe
        Wide or long-format dataframe.
    items : str
        Column in ``data`` with the items names (long-format only).
    scores : str
        Column in ``data`` with the scores (long-format only).
    subject : str
        Column in ``data`` with the subject identifier (long-format only).
    nan_policy : bool
        If `'listwise'`, remove the entire rows that contain missing values
        (= listwise deletion). If `'pairwise'` (default), only pairwise
        missing values are removed when computing the covariance matrix.
        For more details, please refer to the :py:meth:`pandas.DataFrame.cov`
        method.
    ci : float
        Confidence interval (.95 = 95%)

    Returns
    -------
    alpha : float
        Cronbach's alpha

    Notes
    -----
    This function works with both wide and long format dataframe. If you pass a
    long-format dataframe, you must also pass the ``items``, ``scores`` and
    ``subj`` columns (in which case the data will be converted into wide
    format using the :py:meth:`pandas.DataFrame.pivot` method).

    Internal consistency is usually measured with Cronbach's alpha, a statistic
    calculated from the pairwise correlations between items.
    Internal consistency ranges between negative infinity and one.
    Coefficient alpha will be negative whenever there is greater
    within-subject variability than between-subject variability.

    Cronbach's :math:`\\alpha` is defined as

    .. math::

        \\alpha ={k \\over k-1}\\left(1-{\\sum_{{i=1}}^{k}\\sigma_{{y_{i}}}^{2}
        \\over\\sigma_{x}^{2}}\\right)

    where :math:`k` refers to the number of items, :math:`\\sigma_{x}^{2}`
    is the variance of the observed total scores, and
    :math:`\\sigma_{{y_{i}}}^{2}` the variance of component :math:`i` for
    the current sample of subjects.

    Another formula for Cronbach's :math:`\\alpha` is

    .. math::

        \\alpha = \\frac{k \\times \\bar c}{\\bar v + (k - 1) \\times \\bar c}

    where :math:`\\bar c` refers to the average of all covariances between
    items and :math:`\\bar v` to the average variance of each item.

    95% confidence intervals are calculated using Feldt's method:

    .. math::

        c_L = 1 - (1 - \\alpha) \\cdot F_{(0.025, n-1, (n-1)(k-1))}

        c_U = 1 - (1 - \\alpha) \\cdot F_{(0.975, n-1, (n-1)(k-1))}

    where :math:`n` is the number of subjects and :math:`k` the number of
    items.

    Results have been tested against the R package psych.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cronbach%27s_alpha

    .. [2] http://www.real-statistics.com/reliability/cronbachs-alpha/

    .. [3] https://cran.r-project.org/web/packages/psych/psych.pdf

    .. [4] Feldt, Leonard S., Woodruff, David J., & Salih, Fathi A. (1987).
           Statistical inference for coefficient alpha. Applied Psychological
           Measurement, 11(1):93-103.

    Examples
    --------
    Binary wide-format dataframe (with missing values)

    >>> import pingouin as pg
    >>> data = pg.read_dataset('cronbach_wide_missing')
    >>> # In R: psych:alpha(data, use="pairwise")
    >>> pg.cronbach_alpha(data=data)
    (0.732661, array([0.435, 0.909]))

    After listwise deletion of missing values (remove the entire rows)

    >>> # In R: psych:alpha(data, use="complete.obs")
    >>> pg.cronbach_alpha(data=data, nan_policy='listwise')
    (0.801695, array([0.581, 0.933]))

    After imputing the missing values with the median of each column

    >>> pg.cronbach_alpha(data=data.fillna(data.median()))
    (0.738019, array([0.447, 0.911]))

    Likert-type long-format dataframe

    >>> data = pg.read_dataset('cronbach_alpha')
    >>> pg.cronbach_alpha(data=data, items='Items', scores='Scores',
    ...                   subject='Subj')
    (0.591719, array([0.195, 0.84 ]))
    """
    # Safety check
    assert isinstance(data, pd.DataFrame), 'data must be a dataframe.'
    assert nan_policy in ['pairwise', 'listwise']

    if all([v is not None for v in [items, scores, subject]]):
        # Data in long-format: we first convert to a wide format
        data = data.pivot(index=subject, values=scores, columns=items)

    # From now we assume that data is in wide format
    n, k = data.shape
    assert k >= 2, 'At least two items are required.'
    assert n >= 2, 'At least two raters/subjects are required.'
    err = 'All columns must be numeric.'
    assert all([data[c].dtype.kind in 'bfi' for c in data.columns]), err
    if data.isna().any().any() and nan_policy == 'listwise':
        # In R = psych:alpha(data, use="complete.obs")
        data = data.dropna(axis=0, how='any')

    # Compute covariance matrix and Cronbach's alpha
    C = data.cov()
    cronbach = (k / (k - 1)) * (1 - np.trace(C) / C.sum().sum())
    # which is equivalent to
    # v = np.diag(C).mean()
    # c = C.values[np.tril_indices_from(C, k=-1)].mean()
    # cronbach = (k * c) / (v + (k - 1) * c)

    # Confidence intervals
    alpha = 1 - ci
    df1 = n - 1
    df2 = df1 * (k - 1)
    lower = 1 - (1 - cronbach) * f.isf(alpha / 2, df1, df2)
    upper = 1 - (1 - cronbach) * f.isf(1 - alpha / 2, df1, df2)
    return round(cronbach, 6), np.round([lower, upper], 3)


def intraclass_corr(data=None, items=None, raters=None, scores=None, ci=.95):
    """Intra-class correlation coefficient.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the variables
    items : string
        Name of column in data containing the items (targets).
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

    Shrout and Fleiss (1979) describe six cases of reliability of ratings done
    by :math:`k` raters on :math:`n` targets. Pingouin only returns ICC1,
    which consider that each target is rated by a different rater and the
    raters are selected at random. (This is a one-way ANOVA
    fixed effects model and is found by (MSB - MSW)/(MSB + (nr - 1) * MSW)).
    ICC1 is sensitive to differences in means between raters and is a measure
    of absolute agreement.

    This function has been tested against the ICC function of the R psych
    package.

    References
    ----------
    .. [1] Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations:
           uses in assessing rater reliability. Psychological bulletin, 86(2),
           420.

    .. [2] https://cran.r-project.org/web/packages/psych/psych.pdf

    .. [3] http://www.real-statistics.com/reliability/intraclass-correlation/

    Examples
    --------
    ICC of wine quality assessed by 4 judges.

    >>> import pingouin as pg
    >>> data = pg.read_dataset('icc')
    >>> pg.intraclass_corr(data=data, items='Wine', raters='Judge',
    ...                    scores='Scores', ci=.95)
    (0.727526, array([0.434, 0.927]))
    """
    from pingouin import anova

    # Check dataframe
    if any(v is None for v in [data, items, raters, scores]):
        raise ValueError('Data, items, raters and scores must be specified')
    assert isinstance(data, pd.DataFrame), 'Data must be a pandas dataframe.'
    # Check that scores is a numeric variable
    assert data[scores].dtype.kind in 'fi', 'Scores must be numeric.'
    # Check that data are fully balanced
    if data.groupby(raters)[scores].count().nunique() > 1:
        raise ValueError('Data must be balanced.')

    # Extract sizes
    k = data[raters].nunique()
    # n = data[groups].nunique()

    # ANOVA and ICC
    aov = anova(dv=scores, data=data, between=items, detailed=True)
    icc = (aov.at[0, 'MS'] - aov.at[1, 'MS']) / \
          (aov.at[0, 'MS'] + (k - 1) * aov.at[1, 'MS'])

    # Confidence interval
    alpha = 1 - ci
    df_num, df_den = aov.at[0, 'DF'], aov.at[1, 'DF']
    f_lower = aov.at[0, 'F'] / f.isf(alpha / 2, df_num, df_den)
    f_upper = aov.at[0, 'F'] * f.isf(alpha / 2, df_den, df_num)
    lower = (f_lower - 1) / (f_lower + k - 1)
    upper = (f_upper - 1) / (f_upper + k - 1)

    return round(icc, 6), np.round([lower, upper], 3)
