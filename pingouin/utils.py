"""Helper functions."""
import numbers
import numpy as np
import pandas as pd
import itertools as it
import collections.abc
from tabulate import tabulate
from .config import options

__all__ = ["_perm_pval", "print_table", "_postprocess_dataframe",
           "_check_eftype", "remove_rm_na", "remove_na", "_flatten_list",
           "_check_dataframe", "_is_sklearn_installed",
           "_is_statsmodels_installed", "_is_mpmath_installed"]


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
        Tail for p-value. Can be either `'two-sided'` (default), or `'greater'`
        or `'less'` for directional tests.

    Returns
    -------
    p : float
        P-value.
    """
    assert tail in ['two-sided', 'greater', 'less'], 'Wrong tail argument.'
    assert isinstance(estimate, (int, float))
    bootstat = np.asarray(bootstat)
    assert bootstat.ndim == 1, 'bootstat must be a 1D array.'
    n_boot = bootstat.size
    assert n_boot >= 1, 'bootstat must have at least one value.'
    if tail == 'greater':
        p = np.greater_equal(bootstat, estimate).sum() / n_boot
    elif tail == 'less':
        p = np.less_equal(bootstat, estimate).sum() / n_boot
    else:
        p = np.greater_equal(np.fabs(bootstat), abs(estimate)).sum() / n_boot
    return p

###############################################################################
# PRINT & EXPORT OUTPUT TABLE
###############################################################################


def print_table(df, floatfmt=".3f", tablefmt='simple'):
    """Pretty display of table.

    Parameters
    ----------
    df : :py:class:`pandas.DataFrame`
        Dataframe to print (e.g. ANOVA summary)
    floatfmt : string
        Decimal number formatting
    tablefmt : string
        Table format (e.g. 'simple', 'plain', 'html', 'latex', 'grid', 'rst').
        For a full list of available formats, please refer to
        https://pypi.org/project/tabulate/
    """
    if 'F' in df.keys():
        print('\n=============\nANOVA SUMMARY\n=============\n')
    if 'A' in df.keys():
        print('\n==============\nPOST HOC TESTS\n==============\n')

    print(tabulate(df, headers="keys", showindex=False, floatfmt=floatfmt,
                   tablefmt=tablefmt))
    print('')


def _postprocess_dataframe(df):
    """Apply some post-processing to an ouput dataframe (e.g. rounding).

    Whether and how rounding is applied is governed by options specified in
    `pingouin.options`. The default rounding (number of decimals) is
    determined by `pingouin.options['round']`. You can specify rounding for a
    given column name by the option `'round.column.<colname>'`, e.g.
    `'round.column.CI95%'`. Analogously, `'round.row.<rowname>'` also works
    (where `rowname`) refers to the pandas index), as well as
    `'round.cell.[<rolname>]x[<colname]'`. A cell-based option is used,
    if available; if not, a column-based option is used, if
    available; if not, a row-based option is used, if available; if not,
    the default is used. (Default `pingouin.options['round'] = None`,
    i.e. no rounding is applied.)

    If a round option is `callable` instead of `int`, then it will be called,
    and the return value stored in the cell.

    Post-processing is applied on a copy of the DataFrame, leaving the
    original DataFrame untouched.

    This is an internal function (no public API).

    Parameters
    ----------
    df : :py:class:`pandas.DataFrame`
        Dataframe to apply post-processing to (e.g. ANOVA summary)

    Returns
    ----------
    df : :py:class:`pandas.DataFrame`
        Dataframe with post-processing applied
    """
    df = df.copy()
    for row, col in it.product(df.index, df.columns):
        round_option = _get_round_setting_for(row, col)
        if round_option is None:
            continue
        if callable(round_option):
            newval = round_option(df.at[row, col])
            # ensure that dtype changes are processed
            df[col] = df[col].astype(type(newval))
            df.at[row, col] = newval
            continue
        if isinstance(df.at[row, col], bool):
            # No rounding if value is a boolean
            continue
        is_number = isinstance(df.at[row, col], numbers.Number)
        is_array = isinstance(df.at[row, col], np.ndarray)
        if not any([is_number, is_array]):
            # No rounding if value is not a Number or an array
            continue
        if is_array:
            is_float_array = issubclass(df.at[row, col].dtype.type,
                                        np.floating)
            if not is_float_array:
                # No rounding if value is not a float array
                continue
        df.at[row, col] = np.round(df.at[row, col], decimals=round_option)
    return df


def _get_round_setting_for(row, col):
    keys_to_check = (
        'round.cell.[{}]x[{}]'.format(row, col),
        'round.column.{}'.format(col), 'round.row.{}'.format(row))
    for key in keys_to_check:
        try:
            return options[key]
        except KeyError:
            pass
    return options['round']


###############################################################################
# MISSING VALUES
###############################################################################


def _remove_na_single(x, axis='rows'):
    """Remove NaN in a single array.
    This is an internal Pingouin function.
    """
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
    else:
        # 2D arrays
        ax = 1 if axis == 'rows' else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
    # Check if missing values are present
    if ~x_mask.all():
        ax = 0 if axis == 'rows' else 1
        ax = 0 if x.ndim == 1 else ax
        x = x.compress(x_mask, axis=ax)
    return x


def remove_na(x, y=None, paired=False, axis='rows'):
    """Remove missing values along a given axis in one or more (paired) numpy
    arrays.

    Parameters
    ----------
    x, y : 1D or 2D arrays
        Data. ``x`` and ``y`` must have the same number of dimensions.
        ``y`` can be None to only remove missing values in ``x``.
    paired : bool
        Indicates if the measurements are paired or not.
    axis : str
        Axis or axes along which missing values are removed.
        Can be 'rows' or 'columns'. This has no effect if ``x`` and ``y`` are
        one-dimensional arrays.

    Returns
    -------
    x, y : np.ndarray
        Data without missing values

    Examples
    --------
    Single 1D array

    >>> import numpy as np
    >>> from pingouin import remove_na
    >>> x = [6.4, 3.2, 4.5, np.nan]
    >>> remove_na(x)
    array([6.4, 3.2, 4.5])

    With two paired 1D arrays

    >>> y = [2.3, np.nan, 5.2, 4.6]
    >>> remove_na(x, y, paired=True)
    (array([6.4, 4.5]), array([2.3, 5.2]))

    With two independent 2D arrays

    >>> x = np.array([[4, 2], [4, np.nan], [7, 6]])
    >>> y = np.array([[6, np.nan], [3, 2], [2, 2]])
    >>> x_no_nan, y_no_nan = remove_na(x, y, paired=False)
    """
    # Safety checks
    x = np.asarray(x)
    assert x.size > 1, 'x must have more than one element.'
    assert axis in ['rows', 'columns'], 'axis must be rows or columns.'

    if y is None:
        return _remove_na_single(x, axis=axis)
    elif isinstance(y, (int, float, str)):
        return _remove_na_single(x, axis=axis), y
    else:  # y is list, np.array, pd.Series
        y = np.asarray(y)
        # Make sure that we just pass-through if y have only 1 element
        if y.size == 1:
            return _remove_na_single(x, axis=axis), y
        if x.ndim != y.ndim or paired is False:
            # x and y do not have the same dimension
            x_no_nan = _remove_na_single(x, axis=axis)
            y_no_nan = _remove_na_single(y, axis=axis)
            return x_no_nan, y_no_nan

    # At this point, we assume that x and y are paired and have same dimensions
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
        y_mask = ~np.isnan(y)
    else:
        # 2D arrays
        ax = 1 if axis == 'rows' else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
        y_mask = ~np.any(np.isnan(y), axis=ax)

    # Check if missing values are present
    if ~x_mask.all() or ~y_mask.all():
        ax = 0 if axis == 'rows' else 1
        ax = 0 if x.ndim == 1 else ax
        both = np.logical_and(x_mask, y_mask)
        x = x.compress(both, axis=ax)
        y = y.compress(both, axis=ax)
    return x, y


def remove_rm_na(data=None, dv=None, within=None, subject=None,
                 aggregate='mean'):
    """Remove missing values in long-format repeated-measures dataframe.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Long-format dataframe.
    dv : string or list
        Dependent variable(s), from which the missing values should be removed.
        If ``dv`` is not specified, all the columns in the dataframe are
        considered. ``dv`` must be numeric.
    within : string or list
        Within-subject factor(s).
    subject : string
        Subject identifier.
    aggregate : string
        Aggregation method if there are more within-factors in the data than
        specified in the ``within`` argument. Can be `mean`, `median`, `sum`,
        `first`, `last`, or any other function accepted by
        :py:meth:`pandas.DataFrame.groupby`.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`
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
    if (data.groupby(idx_cols, observed=True).count() > 1).any().any():
        # Make sure that we keep the non-numeric columns when aggregating
        # This is disabled by default to avoid any confusion.
        # all_others = all_cols.difference(idx_cols)
        # all_num = data[all_others].select_dtypes(include='number').columns
        # agg = {c: aggregate if c in all_num else 'first' for c in all_others}
        data = data.groupby(idx_cols, observed=True).agg(aggregate)
    else:
        # Set subject + within factors as index.
        # Sorting is done to avoid performance warning when dropping.
        data = data.set_index(idx_cols).sort_index()

    # Find index with missing values
    if dv is None:
        iloc_nan = data.isnull().to_numpy().nonzero()[0]
    else:
        iloc_nan = data[dv].isnull().to_numpy().nonzero()[0]

    # Drop the last within level
    idx_nan = data.index[iloc_nan].droplevel(-1)

    # Drop and re-order
    data = data.drop(idx_nan).reset_index(drop=False)
    return data.reindex(columns=all_cols).dropna(how='all', axis=1)


###############################################################################
# ARGUMENTS CHECK
###############################################################################

def _flatten_list(x, include_tuple=False):
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

    >>> x = ['Xaa', ('Xbb', 'Xcc'), (1, 2), (1)]
    >>> _flatten_list(x)
    ['Xaa', ('Xbb', 'Xcc'), (1, 2), 1]

    >>> _flatten_list(x, include_tuple=True)
    ['Xaa', 'Xbb', 'Xcc', 1, 2, 1]
    """
    # If x is not iterable, return x
    if not isinstance(x, collections.abc.Iterable):
        return x
    # Remove None
    x = list(filter(None.__ne__, x))
    # Initialize empty output variable
    result = []
    # Loop over items in x
    for el in x:
        # Check if element is iterable
        el_is_iter = isinstance(el, collections.abc.Iterable)
        if el_is_iter:
            if not isinstance(el, (str, tuple)):
                result.extend(_flatten_list(el))
            else:
                if isinstance(el, tuple) and include_tuple:
                    result.extend(_flatten_list(el))
                else:
                    result.append(el)
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


def _check_dataframe(data=None, dv=None, between=None, within=None,
                     subject=None, effects=None,):
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
    """Check if statsmodels is installed."""
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
    """Check if sklearn is installed."""
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


def _is_mpmath_installed(raise_error=False):
    """Check if mpmath is installed."""
    try:
        import mpmath  # noqa
        is_installed = True
    except IOError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise IOError("mpmath needs to be installed. Please use `pip "
                      "install mpmath`.")
    return is_installed
