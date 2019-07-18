# Author: Antoine Weill--Duflos <antoine@weill-duflos.fr>
# Date: July 2019
import numpy as np
import pandas as pd
from pingouin.parametric import ttest
from pingouin.nonparametric import wilcoxon, mwu


__all__ = ["tost"]


def tost(x, y, paired=False, parametric=True, bound=0.3, correction=False):
    """Two one-sided test (TOST) for equivalence.

    Parameters
    ----------
    x : array_like
        First set of observations.
    y : array_like or float
        Second set of observations. If y is a single value, a one-sample T-test
        is computed.
    paired : boolean
        Specify whether the two observations are related (i.e. repeated
        measures) or independent.
    parametric : boolean
        If True (default), use the parametric :py:func:`ttest` function.
        If False, use :py:func:`pingouin.wilcoxon` or :py:func:`pingouin.mwu`
        for paired or unpaired samples, respectively.
    bound : float
        Magnitude of region of similarity (epsilon).
    correction : auto or boolean
        Specify whether or not to correct for unequal variances using Welch
        separate variances T-test. This only applies if ``parametric`` is True.

    Returns
    -------
    stats : pandas DataFrame
        TOST summary ::

        'upper' : upper interval p-value
        'lower' : lower interval p-value
        'p-val' : TOST p-value

    See also
    --------
    ttest, mwu, wilcoxon

    References
    ----------
    .. [1] Schuirmann, D.L. 1981. On hypothesis testing to determine if the
           mean of a normal distribution is contained in a known interval.
           Biometrics 37 617.

    .. [2] https://cran.r-project.org/web/packages/equivalence/equivalence.pdf

    Examples
    --------
    1. TOST with a region of similarity of 1

    >>> import pingouin as pg
    >>> a = [4, 7, 8, 6, 3, 2]
    >>> b = [6, 8, 7, 10, 11, 9]
    >>> pg.tost(a, b, bound=1)
             upper     lower     p-val
    TOST  0.965097  0.002216  0.965097

    2. Non parametric paired TOST

    >>> a = [4, 7, 8, 6, 3, 2, 4, 7, 8, 6, 3, 2, 4, 7, 8, 6, 3, 2]
    >>> b = [6, 8, 7, 10, 11, 9, 6, 8, 7, 10, 11, 9, 6, 8, 7, 10, 11, 9]
    >>> pg.tost(a, b, paired=True, parametric=False)
             upper    lower     p-val
    TOST  0.001117  0.00028  0.001117
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if parametric:
        df_ttesta = ttest(y + bound, x, paired=paired, tail='one-sided',
                          correction=correction)
        df_ttestb = ttest(x + bound, y, paired=paired, tail='one-sided',
                          correction=correction)
        if df_ttestb.at['T-test', 'T'] < 0:
            df_ttestb.at['T-test', 'p-val'] = 1 - df_ttestb.at['T-test',
                                                               'p-val']
        if df_ttesta.at['T-test', 'T'] < 0:
            df_ttesta.at['T-test', 'p-val'] = 1 - df_ttesta.at['T-test',
                                                               'p-val']
        if df_ttestb.at['T-test', 'p-val'] >= df_ttesta.at['T-test',
                                                           'p-val']:
            pval = df_ttestb.at['T-test', 'p-val']
            lpval = df_ttesta.at['T-test', 'p-val']
        else:
            pval = df_ttesta.at['T-test', 'p-val']
            lpval = df_ttestb.at['T-test', 'p-val']
    else:
        if paired:
            df_ttesta = wilcoxon(y + bound, x, tail='one-sided')
            df_ttestb = wilcoxon(x + bound, y, tail='one-sided')
            if df_ttestb.at['Wilcoxon', 'p-val'] >= df_ttesta.at['Wilcoxon',
                                                                 'p-val']:
                pval = df_ttestb.at['Wilcoxon', 'p-val']
                lpval = df_ttesta.at['Wilcoxon', 'p-val']
            else:
                pval = df_ttesta.at['Wilcoxon', 'p-val']
                lpval = df_ttestb.at['Wilcoxon', 'p-val']
        else:
            df_ttesta = mwu(y + bound, x, tail='one-sided')
            df_ttestb = mwu(x + bound, y, tail='one-sided')
            if df_ttestb.at['MWU', 'p-val'] >= df_ttesta.at['MWU', 'p-val']:
                pval = df_ttestb.at['MWU', 'p-val']
                lpval = df_ttesta.at['MWU', 'p-val']
            else:
                pval = df_ttesta.at['MWU', 'p-val']
                lpval = df_ttestb.at['MWU', 'p-val']

    # Create output dataframe
    stats = {'p-val': pval, 'upper': pval, 'lower': lpval}
    stats = pd.DataFrame.from_records(stats, index=['TOST'])
    return stats[['upper', 'lower', 'p-val']]
