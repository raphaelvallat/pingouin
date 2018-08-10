# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
from six import string_types
from pingouin.external.tabulate import tabulate
import pandas as pd

__all__ = ["mad", "madmedianrule", "mahal", "print_table", "_export_table",
           "reshape_data", "_check_eftype", "_remove_rm_na", "_remove_na",
           "_check_dataframe", "_extract_effects", "is_statsmodels_installed",
           "is_sklearn_installed"]


def mad(a, normalize=True, axis=0):
    """
    Median Absolute Deviation along given axis of an array

    See https://en.wikipedia.org/wiki/Median_absolute_deviation

    Parameters
    ----------
    a : array-like
        Input array.
    normalize : boolean.
        If True, scale by a normalization constant (~0.67)
    axis : int, optional
        The defaul is 0. Can also be None.

    Returns
    -------
    mad : float
        mad = median(abs(a - median(a))) / c

    Examples
    --------

        >>>  a = [1.2, 5.4, 3.2, 7.8, 2.5]
        >>> mad(a)
            2.965
        >>> mad(a, normalize=False)
            2.0
    """
    from scipy.stats import norm
    c = norm.ppf(3 / 4.) if normalize else 1
    return np.median(np.abs(a - np.median(a)) / c, axis=axis)


def madmedianrule(a):
    """Outlier detection based on the MAD-median rule described in
    Wilcox 2012.

    Parameters
    ----------
    a : array-like
        Input array.

    Returns
    -------
    outliers: boolean (same shape as a)
        Boolean array indicating whether each sample is an outlier (True) or
        not (False).

    Examples
    --------

        >>> a = [-1.09, 1., 0.28, -1.51, -0.58, 6.61, -2.43, -0.43]
        >>> madmedianrule(a)
            array([False, False, False, False, False, True, False, False])
    """
    from scipy.stats import chi2
    k = np.sqrt(chi2.ppf(0.975, 1))
    return (np.abs(a - np.median(a)) / mad(a)) > k


def mahal(Y, X):
    """Mahalanobis distance.

    Equivalent to the Matlab mahal function.

    Parameters
    ----------
    Y : ndarray (shape=(n, m))
        Data
    X : ndarray (shape=(p, m))
        Reference samples

    Returns
    -------
    MD : 1D-array (shape=(n,))
        Squared Mahalanobis distance of each observation in Y to the
        reference samples in X.
    """
    rx, cx = X.shape
    ry, cy = Y.shape

    m = np.mean(X, 0)
    M = np.tile(m, ry).reshape(ry, 2)
    C = X - np.tile(m, rx).reshape(rx, 2)
    Q, R = np.linalg.qr(C)
    ri = np.linalg.solve(R.T, (Y - M).T)
    return np.sum(ri**2, 0) * (rx - 1)


def print_table(df, floatfmt=".3f", tablefmt='simple'):
    """Nice display of table.

    See: https://pypi.org/project/tabulate/.

    Parameters
    ----------
    df : DataFrame
        Dataframe to print (e.g. ANOVA summary)
    floatfmt : string
        Decimal number formatting
    tablefmt : string
        Table format (e.g. 'simple', 'plain', 'html', 'latex', 'grid')
    """
    if 'F' in df.keys():
        print('\n=============\nANOVA SUMMARY\n=============\n')
    if 'A' in df.keys():
        print('\n==============\nPOST HOC TESTS\n==============\n')

    print(tabulate(df, headers="keys", showindex=False, floatfmt=floatfmt,
                   tablefmt=tablefmt))
    print('')


def _export_table(table, fname):
    """Export DataFrame to .csv"""
    import os.path as op
    extension = op.splitext(fname.lower())[1]
    if extension == '':
        fname = fname + '.csv'
    table.to_csv(fname, index=None, sep=',', encoding='utf-8',
                 float_format='%.4f', decimal='.')


def reshape_data(df, id, dv='DV', rm='Time'):
    """Reshape data from human-readable to long-format.

    See: https://deparkes.co.uk/2016/10/28/reshape-pandas-data-with-melt/

    Parameters
    ----------
    df : DataFrame
        Dataframe in original shape
    id : string
        Column to use as identifier variables. (e.g. "Subjects")
    dv : string
        Name of the dependant variables (e.g. "DV" or "Scores")
    rm : string
        Name of the measurements (e.g. "Time" or "Weekday")

    Returns
    -------
    df_reshaped : DataFrame
        Reshaped DataFrame

    Examples
    --------

    Human-readable:

    *Values represent scores at a cognitive test at different times of the day.
    Ss = subject*::

        Ss    10am  2pm   6pm
        1     3.5   3.2   2.8
        2     2.1   2.4   2.8

    Pingouin::

        Ss    Score    Time
        1     3.5      10am
        1     3.2      2pm
        1     2.8      6pm
        2     2.1      10am
        2     2.4      2pm
        2     2.8      6pm

    >>> import pandas as pd
    >>> from pingouins import reshape_data
    >>> data = {'Ss': [1, 2, 3],
    >>>        '10am': [12, 6, 5],
    >>>        '2pm': [10, 6, 11],
    >>>        '6pm': [8, 5, 7]}
    >>> df = pd.DataFrame(data, columns=['Ss', '10am', '2pm', '6pm'])
    >>> reshaped = reshape_data(df, 'Ss', dv="Score", rm="Time")
    >>> print(reshaped)
        Ss  Time  Score
        1   10am  12
        1   2pm   10
        1   6pm   8
        2   10am  6
        2   2pm   6
        2   6pm   5
        3  10am   5
        3   2pm   11
        3   6pm   7
    """
    return pd.melt(df, id_vars=id, var_name=rm, value_name=dv).sort_values(
        by=id)


def _remove_na(x, y, paired=False):
    """Remove missing values in paired and independant measurements.
    """
    x_na = np.any(np.isnan(x))
    y_na = np.any(np.isnan(y))
    if (x_na or y_na) and paired:
        ar = np.c_[x, y]
        ar = ar[~np.isnan(ar).any(axis=1)]
        x, y = ar[:, 0], ar[:, 1]
    elif (x_na or y_na) and not paired:
        x = np.array(list(filter(lambda v: v == v, x))) if x_na else x
        y = np.array(list(filter(lambda v: v == v, y))) if y_na else y
    return x, y


def _remove_rm_na(dv=None, within=None, subject=None, data=None):
    """Remove subject(s) with one or more missing values in repeated
    measurements.
    """
    if subject is None:
        rm = list(data[within].dropna().unique())
        n_rm = len(rm)
        n_obs = int(data.groupby(within)[dv].count().max())
        data['ID_Subj'] = np.tile(np.arange(n_obs), n_rm)
        data = data.set_index('ID_Subj')
    else:
        data = data.set_index(subject)

    # Find index with nan
    iloc_nan = pd.isnull(data).any(1).nonzero()[0]
    idx_nan = data.index[iloc_nan].values
    print('\nNote: %i subject(s) removed because of missing value(s).\n'
          % len(idx_nan))
    return data.drop(idx_nan).reset_index(drop=False)


def _check_eftype(eftype):
    """Check validity of eftype"""
    if eftype.lower() in ['none', 'hedges', 'cohen', 'glass', 'r',
                          'eta-square', 'odds-ratio', 'auc']:
        return True
    else:
        return False


def _check_dataframe(dv=None, between=None, within=None, subject=None,
                     effects=None, data=None):
    """Check dataframe"""
    # Check input arguments
    if not isinstance(data, pd.DataFrame):
        raise ValueError('Data must be a pandas dataframe')
    if any(v is None for v in [dv, data]):
        raise ValueError('DV and data must be specified')
    if effects not in ['within', 'between', 'interaction', 'all']:
        raise ValueError('Effects must be: within, between, interaction, all')
    if effects == 'within' and not isinstance(within, string_types):
        raise ValueError('within must be specified when effects=within')
    elif effects == 'between' and not isinstance(between, string_types):
        raise ValueError('between must be specified when effects=between')
    elif effects == 'interaction':
        for input in [within, between]:
            if not isinstance(input, string_types):
                raise ValueError('within and between must be specified when \
                effects=interaction')
    # Check that subject identifier is provided in rm_anova and friedman.
    if effects == 'within' and subject is None:
        raise ValueError('subject must be specified when effects=within')


def _extract_effects(dv=None, between=None, within=None, subject=None,
                     effects=None, data=None):
    """Extract main effects"""
    # Check the dataframe
    _check_dataframe(dv=dv, between=between, within=within, subject=subject,
                     effects=effects, data=data)

    datadic = {}
    nobs = np.array([], dtype=int)

    # Extract number of pairwise comparisons
    if effects.lower() in ['within', 'between']:
        col = within if effects == 'within' else between
        # Extract data
        labels = list(data[col].unique())
        for l in labels:
            datadic[l] = data[data[col] == l][dv]
            nobs = np.append(nobs, len(datadic[l]))

    elif effects.lower() == 'interaction':
        labels_with = list(data[within].unique())
        labels_betw = list(data[between].unique())
        for lw in labels_with:
            for l in labels_betw:
                tmp = data[data[within] == lw]
                datadic[lw, l] = tmp[tmp[between] == l][dv]
                nobs = np.append(nobs, len(datadic[lw, l]))

    dt_array = pd.DataFrame.from_dict(datadic)
    return dt_array, nobs


def is_statsmodels_installed(raise_error=False):
    try:
        import statsmodels  # noqa
        is_installed = True
    except IOError:
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:
        raise IOError("statsmodels is not installed.")
    return is_installed


def is_sklearn_installed(raise_error=False):
    try:
        import sklearn  # noqa
        is_installed = True
    except IOError:
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:
        raise IOError("sklearn is not installed.")
    return is_installed
