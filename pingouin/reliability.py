import numpy as np
import pandas as pd
# from pingouin.utils import remove_rm_na

__all__ = ["intraclass_corr"]


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
