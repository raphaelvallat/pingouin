# Author: Antoine Weill--Duflos <antoine@weill-duflos.fr>
# Date: July 2019
import numpy as np
import pandas as pd
from pingouin.parametric import ttest
from pingouin.nonparametric import wilcoxon, mwu
__all__ = ["tost"]


def tost(x, y, paired=False, parametric=True, bound=0.3, correction=False):
    """T-test.

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
        Magnitude of region of similarity
    correction : auto or boolean
        Specify whether or not to correct for unequal variances using Welch
        separate variances T-test

    Returns
    -------
    stats : pandas DataFrame
        TOST summary ::

        'upper' : upper interval p-value
        'lower' : lower interval p-value
        'p-val' : TOST p-value

    Examples
    --------

    1. TOST with a region of similarity of 1
    >>> import pingouin as pg
    >>> a = [4, 7, 8, 6, 3, 2]
    >>> b = [6, 8, 7, 10, 11, 9]
    >>> pg.tost(a, b, bound=1)
             upper     lower     p-val
    TOST  0.965097  0.002216  0.965097

    2. non parametric, paired, TOST with a region of similarity of 10
    >>> import pingouin as pg
    >>> a = [4, 7, 8, 6, 3, 2, 4, 7, 8, 6, 3, 2, 4, 7, 8, 6, 3, 2]
    >>> b = [6, 8, 7, 10, 11, 9, 6, 8, 7, 10, 11, 9, 6, 8, 7, 10, 11, 9]
    >>> pg.tost(a,b,bound=10,paired=True,parametric=False)
             upper     lower     p-val
    TOST  0.000103  0.000103  0.000103

    """
    if parametric:
        df_ttesta = ttest(list(np.asarray(y) + bound), x, paired=paired,
                          tail='one-sided', correction=correction)
        df_ttestb = ttest(list(np.asarray(x) + bound), y, paired=paired,
                          tail='one-sided', correction=correction)
        if df_ttestb.loc['T-test', 'T'] < 0:
            df_ttestb.loc['T-test', 'p-val'] = 1 - df_ttestb.loc['T-test',
                                                                 'p-val']
        if df_ttesta.loc['T-test', 'T'] < 0:
            df_ttesta.loc['T-test', 'p-val'] = 1 - df_ttesta.loc['T-test',
                                                                 'p-val']
        if df_ttestb.loc['T-test', 'p-val'] >= df_ttesta.loc['T-test',
                                                             'p-val']:
            pval = df_ttestb.loc['T-test', 'p-val']
            lpval = df_ttesta.loc['T-test', 'p-val']
        else:
            pval = df_ttesta.loc['T-test', 'p-val']
            lpval = df_ttestb.loc['T-test', 'p-val']
    else:
        if paired:
            df_ttesta = wilcoxon(list(np.asarray(y) + bound), x,
                                 tail='greater')
            df_ttestb = wilcoxon(list(np.asarray(x) + bound), y,
                                 tail='greater')
            if df_ttestb.loc['Wilcoxon', 'p-val'] >= df_ttesta.loc['Wilcoxon',
                                                                   'p-val']:
                pval = df_ttestb.loc['Wilcoxon', 'p-val']
                lpval = df_ttesta.loc['Wilcoxon', 'p-val']
            else:
                pval = df_ttesta.loc['Wilcoxon', 'p-val']
                lpval = df_ttestb.loc['Wilcoxon', 'p-val']
        else:
            df_ttesta = mwu(list(np.asarray(y) + bound), x, tail='greater')
            df_ttestb = mwu(list(np.asarray(x) + bound), y, tail='greater')
            if df_ttestb.loc['MWU', 'p-val'] >= df_ttesta.loc['MWU',
                                                              'p-val']:
                pval = df_ttestb.loc['MWU', 'p-val']
                lpval = df_ttesta.loc['MWU', 'p-val']
            else:
                pval = df_ttesta.loc['MWU', 'p-val']
                lpval = df_ttestb.loc['MWU', 'p-val']
    stats = {'p-val': pval, 'upper': pval, 'lower': lpval}

    # Convert to dataframe
    stats = pd.DataFrame.from_records(stats, index=['TOST'])

    col_order = ['upper', 'lower', 'p-val']
    stats = stats.reindex(columns=col_order)
    stats.dropna(how='all', axis=1, inplace=True)
    return stats
