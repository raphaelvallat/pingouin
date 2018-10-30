# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
from six import string_types
from pingouin.external.tabulate import tabulate
import pandas as pd

__all__ = ["print_table", "_export_table", "_check_eftype",
           "_remove_rm_na", "_remove_na", "_check_dataframe",
           "is_sklearn_installed", "is_statsmodels_installed"]


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
    if effects == 'within' and not isinstance(within, (string_types, list)):
        raise ValueError('within must be a string or a list.')
    # Check that subject identifier is provided in rm_anova and friedman.
    if effects == 'within' and subject is None:
        raise ValueError('subject must be specified when effects=within')
    # Check that between is a string or a list (anova2)
    if effects == 'between' and not isinstance(between, (string_types,
                                                         list)):
        raise ValueError('between must be a string or a list.')
    # Check that both between and within are present for interaction
    if effects == 'interaction':
        for input in [within, between]:
            if not isinstance(input, string_types):
                raise ValueError('within and between must be specified when '
                                 'effects=interaction')


def is_statsmodels_installed(raise_error=False):
    try:
        import statsmodels  # noqa
        is_installed = True
    except IOError:
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:
        raise IOError("statsmodels needs to be installed. Please use `pip "
                      "install statsmodels`.")
    return is_installed


def is_sklearn_installed(raise_error=False):
    try:
        import sklearn  # noqa
        is_installed = True
    except IOError:
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:
        raise IOError("sklearn needs to be installed. Please use `pip "
                      "install scikit-learn`.")
    return is_installed
