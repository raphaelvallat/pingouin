"""Helper functions."""
import numbers
import numpy as np
import pandas as pd
import itertools as it
import collections.abc
from tabulate import tabulate
from .config import options

__all__ = [
    "_perm_pval",
    "print_table",
    "_postprocess_dataframe",
    "_check_eftype",
    "remove_na",
    "_flatten_list",
    "_check_dataframe",
    "_is_sklearn_installed",
    "_is_statsmodels_installed",
    "_is_mpmath_installed",
]


def _perm_pval(bootstat, estimate, alternative="two-sided"):
    """
    Compute p-values from a permutation test.

    Parameters
    ----------
    bootstat : 1D array
        Permutation distribution.
    estimate : float or int
        Point estimate.
    alternative : str
        Tail for p-value. Can be either `'two-sided'` (default), `'greater'` or `'less'`.

    Returns
    -------
    p : float
        P-value.
    """
    assert alternative in ["two-sided", "greater", "less"], "Wrong tail argument."
    assert isinstance(estimate, (int, float))
    bootstat = np.asarray(bootstat)
    assert bootstat.ndim == 1, "bootstat must be a 1D array."
    n_boot = bootstat.size
    assert n_boot >= 1, "bootstat must have at least one value."
    if alternative == "greater":
        p = np.greater_equal(bootstat, estimate).sum() / n_boot
    elif alternative == "less":
        p = np.less_equal(bootstat, estimate).sum() / n_boot
    else:
        p = np.greater_equal(np.fabs(bootstat), abs(estimate)).sum() / n_boot
    return p


###############################################################################
# PRINT & EXPORT OUTPUT TABLE
###############################################################################


def print_table(df, floatfmt=".3f", tablefmt="simple"):
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
    if "F" in df.keys():
        print("\n=============\nANOVA SUMMARY\n=============\n")
    if "A" in df.keys():
        print("\n==============\nPOST HOC TESTS\n==============\n")

    print(tabulate(df, headers="keys", showindex=False, floatfmt=floatfmt, tablefmt=tablefmt))
    print("")


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
            is_float_array = issubclass(df.at[row, col].dtype.type, np.floating)
            if not is_float_array:
                # No rounding if value is not a float array
                continue
        df.at[row, col] = np.round(df.at[row, col], decimals=round_option)
    return df


def _get_round_setting_for(row, col):
    keys_to_check = (
        f"round.cell.[{row}]x[{col}]",
        f"round.column.{col}",
        f"round.row.{row}",
    )
    for key in keys_to_check:
        try:
            return options[key]
        except KeyError:
            pass
    return options["round"]


###############################################################################
# MISSING VALUES
###############################################################################


def _remove_na_single(x, axis="rows"):
    """Remove NaN in a single array.
    This is an internal Pingouin function.
    """
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
    else:
        # 2D arrays
        ax = 1 if axis == "rows" else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
    # Check if missing values are present
    if ~x_mask.all():
        ax = 0 if axis == "rows" else 1
        ax = 0 if x.ndim == 1 else ax
        x = x.compress(x_mask, axis=ax)
    return x


def remove_na(x, y=None, paired=False, axis="rows"):
    """Remove missing values along a given axis in one or more (paired) numpy arrays.

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
    assert axis in ["rows", "columns"], "axis must be rows or columns."

    if y is None:
        return _remove_na_single(x, axis=axis)
    elif isinstance(y, (int, float, str)):
        return _remove_na_single(x, axis=axis), y
    else:  # y is list, np.array, pd.Series
        y = np.asarray(y)
        assert y.size != 0, "y cannot be an empty list or array."
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
        ax = 1 if axis == "rows" else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
        y_mask = ~np.any(np.isnan(y), axis=ax)

    # Check if missing values are present
    if ~x_mask.all() or ~y_mask.all():
        ax = 0 if axis == "rows" else 1
        ax = 0 if x.ndim == 1 else ax
        both = np.logical_and(x_mask, y_mask)
        x = x.compress(both, axis=ax)
        y = y.compress(both, axis=ax)
    return x, y


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
    # Remove None from output
    result = [r for r in result if r is not None]
    return result


def _check_eftype(eftype):
    """Check validity of eftype"""
    if eftype.lower() in [
        "none",
        "hedges",
        "cohen",
        "r",
        "pointbiserialr",
        "eta-square",
        "odds-ratio",
        "auc",
        "cles",
    ]:
        return True
    else:
        return False


def _check_dataframe(data=None, dv=None, between=None, within=None, subject=None, effects=None):
    """Checks whether data is a dataframe or can be converted to a dataframe.
    If successful, a dataframe is returned. If not successful, a ValueError is
    raised.
    """
    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        # DataMatrix objects can be safely convert to DataFrame objects. By
        # first checking the name of the class, we avoid having to actually
        # import DataMatrix unless it is necessary.
        if data.__class__.__name__ == "DataMatrix":  # noqa
            try:
                from datamatrix import DataMatrix, convert as cnv  # noqa
            except ImportError:
                raise ValueError(
                    "Failed to convert object to pandas dataframe (DataMatrix not available)"  # noqa
                )
            else:
                if isinstance(data, DataMatrix):
                    data = cnv.to_pandas(data)
                else:
                    raise ValueError("Data must be a pandas dataframe or compatible object.")
        else:
            raise ValueError("Data must be a pandas dataframe or compatible object.")
    # Check that both dv and data are provided.
    if any(v is None for v in [dv, data]):
        raise ValueError("DV and data must be specified")
    # Check that dv is a numeric variable
    if data[dv].dtype.kind not in "fi":
        raise ValueError("DV must be numeric.")
    # Check that effects is provided
    if effects not in ["within", "between", "interaction", "all"]:
        raise ValueError("Effects must be: within, between, interaction, all")
    # Check that within is a string, int or a list (rm_anova2)
    if effects == "within" and not isinstance(within, (str, int, list)):
        raise ValueError("within must be a string, int or a list.")
    # Check that subject identifier is provided in rm_anova and friedman.
    if effects == "within" and subject is None:
        raise ValueError("subject must be specified when effects=within")
    # Check that between is a string or a list (anova2)
    if effects == "between" and not isinstance(between, (str, int, list)):
        raise ValueError("between must be a string, int or a list.")
    # Check that both between and within are present for interaction
    if effects == "interaction":
        for input in [within, between]:
            if not isinstance(input, (str, int, list)):
                raise ValueError("within and between must be specified when effects=interaction")
    return data


###############################################################################
# DEPENDENCIES
###############################################################################


def _is_statsmodels_installed(raise_error=False):
    """Check if statsmodels is installed."""
    try:
        import statsmodels  # noqa

        is_installed = True
    except OSError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise OSError("statsmodels needs to be installed. Please use `pip " "install statsmodels`.")
    return is_installed


def _is_sklearn_installed(raise_error=False):
    """Check if sklearn is installed."""
    try:
        import sklearn  # noqa

        is_installed = True
    except OSError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise OSError("sklearn needs to be installed. Please use `pip " "install scikit-learn`.")
    return is_installed


def _is_mpmath_installed(raise_error=False):
    """Check if mpmath is installed."""
    try:
        import mpmath  # noqa

        is_installed = True
    except OSError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise OSError("mpmath needs to be installed. Please use `pip " "install mpmath`.")
    return is_installed
