# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import collections
import numpy as np
from pingouin.external.tabulate import tabulate
import pandas as pd

__all__ = ["_perm_pval", "print_table", "_export_table", "_check_eftype",
           "remove_rm_na", "_remove_na", "_flatten_list", "_check_dataframe",
           "_is_sklearn_installed", "_is_statsmodels_installed"]


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

###############################################################################
# PRINT & EXPORT OUTPUT TABLE
###############################################################################


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

###############################################################################
# MISSING VALUES
###############################################################################


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


def remove_rm_na(dv=None, within=None, subject=None, data=None,
                 aggregate='mean'):
    """Remove missing values in long-format repeated-measures dataframe.

    Parameters
    ----------
    dv : string or list
        Dependent variable(s), from which the missing values should be removed.
        If ``dv`` is not specified, all the columns in the dataframe are
        considered. ``dv`` must be numeric.
    within : string or list
        Within-subject factor(s).
    subject : string
        Subject identifier.
    data : dataframe
        Long-format dataframe.
    aggregate : string
        Aggregation method if there are more within-factors in the data than
        specified in the ``within`` argument. Can be `mean`, `median`, `sum`,
        `first`, last`, or any other function accepted by
        :py:func:`pandas.DataFrame.groupby`.

    Returns
    -------
    data : dataframe
        Dataframe without the missing values.

    Notes
    -----
    If multiple factors are specified, the missing values are removed on the
    last factor, so the order of ``within`` is important.

    In addition, if there are more within-factors in the data than specified in
    the ``within`` argument, data will be aggregated using the function
    specified in ``aggregate``. Note that in the default case (aggregation
    using the mean), all the non-numeric column(s) will be dropped.
    """
    # Safety checks
    assert isinstance(aggregate, str), 'aggregate must be a str.'
    assert isinstance(within, (str, list)), 'within must be str or list.'
    assert isinstance(subject, str), 'subject must be a string.'
    assert isinstance(data, pd.DataFrame), 'Data must be a DataFrame.'

    idx_cols = _flatten_list([subject, within])
    all_cols = data.columns

    if data[idx_cols].isnull().any().any():
        raise ValueError("NaN are present in the within-factors or in the "
                         "subject column. Please remove them manually.")

    # Check if more within-factors are present and if so, aggregate
    if (data.groupby(idx_cols).count() > 1).any().any():
        # Make sure that we keep the non-numeric columns when aggregating
        # This is disabled by default to avoid any confusion.
        # all_others = all_cols.difference(idx_cols)
        # all_num = data[all_others].select_dtypes(include='number').columns
        # agg = {c: aggregate if c in all_num else 'first' for c in all_others}
        data = data.groupby(idx_cols).agg(aggregate)
    else:
        # Set subject + within factors as index.
        # Sorting is done to avoid performance warning when dropping.
        data = data.set_index(idx_cols).sort_index()

    # Find index with missing values
    if dv is None:
        iloc_nan = data.isnull().values.nonzero()[0]
    else:
        iloc_nan = data[dv].isnull().values.nonzero()[0]

    # Drop the last within level
    idx_nan = data.index[iloc_nan].droplevel(-1)

    # Drop and re-order
    data = data.drop(idx_nan).reset_index(drop=False)
    return data.reindex(columns=all_cols).dropna(how='all', axis=1)


###############################################################################
# ARGUMENTS CHECK
###############################################################################

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
    # Remove None
    x = list(filter(None.__ne__, x))
    for el in x:
        x_is_iter = isinstance(x, collections.Iterable)
        if x_is_iter and not isinstance(el, (str, tuple)):
            result.extend(_flatten_list(el))
        else:
            result.append(el)
    return result


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

###############################################################################
# DEPENDENCIES
###############################################################################


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
