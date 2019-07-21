# Author: Antoine Weill--Duflos <antoine@weill-duflos.fr>
# Date: July 2019
import numpy as np
import pandas as pd
from pingouin.parametric import ttest


__all__ = ["tost"]


def tost(x, y, bound=1, paired=False, correction=False):
    """Two one-sided test (TOST) for equivalence.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. ``x`` and ``y`` should have the
        same units. One sample tests are not yet supported.
    bound : float
        Magnitude of region of similarity (epsilon). Note that this should be
        expressed in the same unit as ``x`` and ``y``.
    paired : boolean
        Specify whether the two observations are related (i.e. repeated
        measures) or independent.
    correction : auto or boolean
        Specify whether or not to correct for unequal variances using Welch
        separate variances T-test. This only applies if ``parametric`` is True.

    Returns
    -------
    stats : pandas DataFrame
        TOST summary ::

        'bound' : bound (= epsilon, or equivalence margin)
        'upper' : upper interval p-value
        'lower' : lower interval p-value
        'p-val' : TOST p-value

    See also
    --------
    ttest

    References
    ----------
    .. [1] Schuirmann, D.L. 1981. On hypothesis testing to determine if the
           mean of a normal distribution is contained in a known interval.
           Biometrics 37 617.

    .. [2] https://cran.r-project.org/web/packages/equivalence/equivalence.pdf

    Examples
    --------
    1. TOST with a region of similarity of 1 (default)

    >>> import pingouin as pg
    >>> a = [4, 7, 8, 6, 3, 2]
    >>> b = [6, 8, 7, 10, 11, 9]
    >>> pg.tost(a, b)
          bound     upper     lower     p-val
    TOST      1  0.965097  0.002216  0.965097

    2. Paired TOST with a different equivalent region

    >>> pg.tost(a, b, bound=0.5, paired=True)
          bound     upper     lower     p-val
    TOST    0.5  0.954854  0.019138  0.954854
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert isinstance(bound, (int, float)), 'bound must be int or float.'

    # Run the T-tests
    df_a = ttest(y + bound, x, paired=paired, tail='one-sided',
                 correction=correction)
    df_b = ttest(x + bound, y, paired=paired, tail='one-sided',
                 correction=correction)
    if df_a.at['T-test', 'T'] < 0:
        df_a.at['T-test', 'p-val'] = 1 - df_a.at['T-test', 'p-val']
    if df_b.at['T-test', 'T'] < 0:
        df_b.at['T-test', 'p-val'] = 1 - df_b.at['T-test', 'p-val']
    if df_b.at['T-test', 'p-val'] >= df_a.at['T-test', 'p-val']:
        pval = df_b.at['T-test', 'p-val']
        lpval = df_a.at['T-test', 'p-val']
    else:
        pval = df_a.at['T-test', 'p-val']
        lpval = df_b.at['T-test', 'p-val']

    # Create output dataframe
    stats = {'p-val': pval, 'upper': pval, 'lower': lpval, 'bound': bound}
    stats = pd.DataFrame.from_records(stats, index=['TOST'])
    return stats[['bound', 'upper', 'lower', 'p-val']]
