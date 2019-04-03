# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import warnings
import collections
import numpy as np
from pingouin.external.tabulate import tabulate
import pandas as pd

__all__ = ["_flatten_list", "_perm_pval", "print_table", "_export_table",
           "_check_eftype", "_remove_rm_na", "_remove_na", "_check_dataframe",
           "_is_sklearn_installed", "_is_statsmodels_installed"]


def _flatten_list(x):
    """Flatten an arbitrarily nested list into a new list.

    This can be useful to select pandas DataFrame columns.

    From https://stackoverflow.com/a/16176969/10581531

    Examples
    --------
    >>> from pingouin.utils import _flatten_list
    >>> x = ['X1', ['M1', 'M2'], 'Y1', ['Y2']]
    >>> _flatten_list(x)
    ['X1', 'M1', 'M2', 'Y1', 'Y2']

    >>> x = ['Xaa', 'Xbb', 'Xcc']
    >>> _flatten_list(x)
    ['Xaa', 'Xbb', 'Xcc']
    """
    result = []
    for el in x:
        x_is_iter = isinstance(x, collections.Iterable)
        if x_is_iter and not isinstance(el, (str, tuple)):
            result.extend(_flatten_list(el))
        else:
            result.append(el)
    return result


def _perm_pval(bootstat, estimate, tail='two-sided'):
    """
    Compute p-values from a permutation test.

    Parameters
    ----------
    bootstat : 1D array
        Permutation distribution.
    estimate : float or int
        Point estimate.
    tail : str
        'upper': one-sided p-value (upper tail)
        'lower': one-sided p-value (lower tail)
        'two-sided': two-sided p-value

    Returns
    -------
    p : float
        P-value.
    """
    assert tail in ['two-sided', 'upper', 'lower'], 'Wrong tail argument.'
    assert isinstance(estimate, (int, float))
    bootstat = np.asarray(bootstat)
    assert bootstat.ndim == 1, 'bootstat must be a 1D array.'
    n_boot = bootstat.size
    assert n_boot >= 1, 'bootstat must have at least one value.'
    if tail == 'upper':
        p = np.greater_equal(bootstat, estimate).sum() / n_boot
    elif tail == 'lower':
        p = np.less_equal(bootstat, estimate).sum() / n_boot
    else:
        p = np.greater_equal(np.fabs(bootstat), abs(estimate)).sum() / n_boot
    return p


def print_table(df, floatfmt=".3f", tablefmt='simple'):
    """Pretty display of table.

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


def _remove_na(x, y, paired=False):
    """Remove missing values in paired and independent measurements.

    Parameters
    ----------
    x, y : 1D arrays
        Data
    paired : bool
        Indicates if the measurements are paired or not.

    Returns
    -------
    x, y : 1D arrays
        Data without NaN
    """
    x_na = np.any(np.isnan(x))
    y_na = np.any(np.isnan(y))
    if (x_na or y_na) and paired:
        ar = np.column_stack((x, y))
        ar = ar[~np.isnan(ar).any(axis=1)]
        x, y = ar[:, 0], ar[:, 1]
    elif (x_na or y_na) and not paired:
        x = np.array(list(filter(lambda v: v == v, x))) if x_na else x
        y = np.array(list(filter(lambda v: v == v, y))) if y_na else y
    return x, y


def _remove_rm_na(dv=None, within=None, subject=None, data=None):
    """Remove subject(s) with one or more missing values in repeated
    measurements.

    Parameters
    ----------
    dv : string
        Dependant variable
    within : string or list
        Within-subject factor
    subject : string
        Subject identifier
    data : dataframe
        Dataframe

    Returns
    -------
    data : dataframe
        Dataframe without the subjects nan values
    """
    if subject is None:
        rm = list(data[within].dropna().unique())
        n_rm = len(rm)
        n_obs = int(data.groupby(within)[dv].count().max())
        data['Subj'] = np.tile(np.arange(n_obs), n_rm)
        data = data.set_index('Subj')
    else:
        data = data.set_index(subject)

    # Find index with nan
    iloc_nan = pd.isnull(data).any(1).values.nonzero()[0]
    idx_nan = data.index[iloc_nan].values
    if len(idx_nan) > 0:
        warnings.warn("\nNote: %i subject(s) removed because of "
                      "missing value(s)." % len(idx_nan))
    return data.drop(idx_nan).reset_index(drop=False)


def _check_eftype(eftype):
    """Check validity of eftype"""
    if eftype.lower() in ['none', 'hedges', 'cohen', 'glass', 'r',
                          'eta-square', 'odds-ratio', 'auc', 'cles']:
        return True
    else:
        return False


def _check_dataframe(dv=None, between=None, within=None, subject=None,
                     effects=None, data=None):
    """Check dataframe"""
    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise ValueError('Data must be a pandas dataframe.')
    # Check that both dv and data are provided.
    if any(v is None for v in [dv, data]):
        raise ValueError('DV and data must be specified')
    # Check that dv is a numeric variable
    if data[dv].dtype.kind not in 'fi':
        raise ValueError('DV must be numeric.')
    # Check that effects is provided
    if effects not in ['within', 'between', 'interaction', 'all']:
        raise ValueError('Effects must be: within, between, interaction, all')
    # Check that within is a string or a list (rm_anova2)
    if effects == 'within' and not isinstance(within, (str, list)):
        raise ValueError('within must be a string or a list.')
    # Check that subject identifier is provided in rm_anova and friedman.
    if effects == 'within' and subject is None:
        raise ValueError('subject must be specified when effects=within')
    # Check that between is a string or a list (anova2)
    if effects == 'between' and not isinstance(between, (str,
                                                         list)):
        raise ValueError('between must be a string or a list.')
    # Check that both between and within are present for interaction
    if effects == 'interaction':
        for input in [within, between]:
            if not isinstance(input, (str, list)):
                raise ValueError('within and between must be specified when '
                                 'effects=interaction')


def _is_statsmodels_installed(raise_error=False):
    try:
        import statsmodels  # noqa
        is_installed = True
    except IOError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise IOError("statsmodels needs to be installed. Please use `pip "
                      "install statsmodels`.")
    return is_installed


def _is_sklearn_installed(raise_error=False):
    try:
        import sklearn  # noqa
        is_installed = True
    except IOError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise IOError("sklearn needs to be installed. Please use `pip "
                      "install scikit-learn`.")
    return is_installed
