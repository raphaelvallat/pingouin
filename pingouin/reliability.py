import numpy as np
import pandas as pd
from pingouin.utils import remove_rm_na

__all__ = ["cronbach_alpha", "intraclass_corr"]


def cronbach_alpha(data=None, items=None, scores=None, subject=None):
    """Cronbach's alpha reliability measure.

    Parameters
    ----------
    items : str
        Column in ``data`` with the items names.
    scores : str
        Column in ``data`` with the scores.
    subject : str
        Column in ``data`` with the subject identifier.
    data : pandas dataframe
        Long-format dataframe.

    Returns
    -------
    alpha : float
        Cronbach's alpha

    Notes
    -----
    Data are expected to be in long-format. If your data are in wide-format,
    please use the :py:func:`pandas.melt` function before running
    this function.

    Internal consistency is usually measured with Cronbach's alpha, a statistic
    calculated from the pairwise correlations between items.
    Internal consistency ranges between negative infinity and one.
    Coefficient alpha will be negative whenever there is greater
    within-subject variability than between-subject variability.

    Subject(s) with missing values are automatically removed using
    the :py:func:`pingouin.remove_rm_na` function (listwise deletion).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cronbach%27s_alpha
    """
    # Safety check
    assert isinstance(data, pd.DataFrame), 'data must be a dataframe.'
    assert isinstance(items, str), 'items must be a column name in data.'
    assert isinstance(scores, str), 'scores must be a column name in data.'
    assert isinstance(subject, str), 'subj must be a column name in data.'
    assert items in data.columns, 'items is not in dataframe.'
    assert scores in data.columns, 'scores is not in dataframe.'
    assert subject in data.columns, 'subj is not in dataframe.'

    # Remove missing values
    assert ~data[items].isna().any(), 'Cannot have NaN in items column.'
    assert ~data[subject].isna().any(), 'Cannot have NaN in subject column.'
    if data[scores].isna().any():
        data = remove_rm_na(dv=scores, within=items,
                            subject=subject, data=data)

    # GroupBy
    grp_item = data.groupby(items)[scores]
    grp_subj = data.groupby(subject)[scores]

    # Compute Cronbach's Alpha
    k = grp_item.ngroups
    nsubj = grp_subj.ngroups
    assert k >= 2, 'At least two items are required.'
    assert nsubj >= 2, 'At least two subjects are required.'
    sv1 = grp_item.var().sum()
    sv2 = grp_subj.sum().var()
    alpha = (k / (k - 1)) * (1 - sv1 / sv2)
    return alpha


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
    ICC of wine quality assessed by 4 judges.

    >>> from pingouin import intraclass_corr, read_dataset
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
    if data.groupby(raters)[scores].count().nunique() > 1:
        raise ValueError('Data must be balanced.')

    # Extract sizes
    k = data[raters].nunique()
    # n = data[groups].nunique()

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
