"""Helper functions."""

import collections.abc
import numbers

import numpy as np
import pandas as pd
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
    "_check_alternative",
    "_format_pairwise_matrix",
    "_is_mpmath_installed",
]


def _check_alternative(alternative):
    """Check that ``alternative`` is one of 'two-sided', 'greater' or 'less'."""
    assert alternative in [
        "two-sided",
        "greater",
        "less",
    ], "Alternative must be one of 'two-sided' (default), 'greater' or 'less'."


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
    _check_alternative(alternative)
    assert isinstance(estimate, numbers.Real)
    bootstat = np.asarray(bootstat)
    assert bootstat.ndim == 1, "bootstat must be a 1D array."
    n_boot = bootstat.size
    assert n_boot >= 1, "bootstat must have at least one value."
    # Relative tolerance so that permuted values that are theoretically equal to the estimate
    # but differ by floating-point error are counted as ties (same as scipy.stats.permutation_test)
    gamma = abs(estimate) * np.finfo(float).eps * 100
    if alternative == "greater":
        p = np.greater_equal(bootstat, estimate - gamma).sum() / n_boot
    elif alternative == "less":
        p = np.less_equal(bootstat, estimate + gamma).sum() / n_boot
    else:
        p = np.greater_equal(np.fabs(bootstat), abs(estimate) - gamma).sum() / n_boot
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
    `'round.column.CI95'`. Analogously, `'round.row.<rowname>'` also works
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
    # Fast path: without row- or cell-specific options, the rounding only depends on the column
    per_cell = any(k.startswith(("round.cell.", "round.row.")) for k in options)
    for col in df.columns:
        if per_cell:
            round_options = [_get_round_setting_for(row, col) for row in df.index]
        else:
            round_options = [options.get(f"round.column.{col}", options["round"])] * len(df)
        if all(opt is None for opt in round_options):
            continue
        values = [_round_value(val, opt) for val, opt in zip(df[col], round_options)]
        if any(callable(opt) for opt in round_options):
            # A formatter can change the type of the values (e.g. float -> str)
            df[col] = pd.Series(values, index=df.index, dtype=object).infer_objects()
        else:
            df[col] = pd.Series(values, index=df.index, dtype=df[col].dtype)
    return df


def _round_value(val, round_option):
    """Round (int option) or format (callable option) a single value of a DataFrame."""
    if round_option is None:
        return val
    if callable(round_option):
        return round_option(val)
    if isinstance(val, bool):
        # No rounding if value is a boolean
        return val
    if isinstance(val, np.ndarray):
        # Only float arrays are rounded
        return np.round(val, decimals=round_option) if val.dtype.kind == "f" else val
    if isinstance(val, numbers.Number):
        return np.round(val, decimals=round_option)
    return val


def _format_pairwise_matrix(mat, mat_upper, decimals=3, stars=True, pval_stars=None):
    """Combine two square matrices into a single pairwise matrix of str.

    Used by :py:func:`pingouin.rcorr` and :py:func:`pingouin.ptests`. The output has the lower
    triangle of ``mat``, "-" on the diagonal, and the strict upper triangle of ``mat_upper``.

    If ``pval_stars`` is a dict, ``mat_upper`` contains p-values, which are displayed as stars
    (``stars=True``, the smallest matching threshold of ``pval_stars`` is used regardless of the
    dict order), or as str with ``decimals`` decimals (``stars=False``). Otherwise,
    ``mat_upper`` is displayed as-is.
    """
    if pval_stars is not None:
        if stars:
            thresholds = sorted(pval_stars.items())

            def fmt(p):
                return next((value for key, value in thresholds if p < key), "")

        else:

            def fmt(p):
                return np.format_float_positional(p, precision=decimals)

        mat_upper = mat_upper.map(fmt)
    upper = np.triu(np.ones(mat.shape, dtype=bool), k=1)
    diagonal = np.eye(mat.shape[0], dtype=bool)
    return mat.astype(str).where(~upper, mat_upper).where(~diagonal, "-")


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


def _nan_mask(x, axis="rows"):
    """Return the mask of the rows (or columns) of ``x`` without NaN, and the axis to compress."""
    if x.ndim == 1:
        return ~np.isnan(x), 0
    if axis == "rows":
        return ~np.isnan(x).any(axis=1), 0
    return ~np.isnan(x).any(axis=0), 1


def _remove_na_single(x, axis="rows"):
    """Remove NaN in a single array.
    This is an internal Pingouin function.
    """
    mask, ax = _nan_mask(x, axis=axis)
    return x if mask.all() else x.compress(mask, axis=ax)


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
    x_mask, ax = _nan_mask(x, axis=axis)
    y_mask, _ = _nan_mask(y, axis=axis)
    both = x_mask & y_mask
    if not both.all():
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
    >>> x = ["X1", ["M1", "M2"], "Y1", ["Y2"]]
    >>> _flatten_list(x)
    ['X1', 'M1', 'M2', 'Y1', 'Y2']

    >>> x = ["Xaa", "Xbb", "Xcc"]
    >>> _flatten_list(x)
    ['Xaa', 'Xbb', 'Xcc']

    >>> x = ["Xaa", ("Xbb", "Xcc"), (1, 2), (1)]
    >>> _flatten_list(x)
    ['Xaa', ('Xbb', 'Xcc'), (1, 2), 1]

    >>> _flatten_list(x, include_tuple=True)
    ['Xaa', 'Xbb', 'Xcc', 1, 2, 1]
    """
    # If x is not iterable, return x
    if not isinstance(x, collections.abc.Iterable):
        return x
    result = []
    for el in x:
        is_nested = isinstance(el, collections.abc.Iterable) and not isinstance(el, str)
        if is_nested and (include_tuple or not isinstance(el, tuple)):
            result.extend(_flatten_list(el, include_tuple=include_tuple))
        elif el is not None:
            # None are removed from the output
            result.append(el)
    return result


def _check_eftype(eftype):
    """Check validity of eftype"""
    return eftype.lower() in [
        "none",
        "hedges",
        "cohen",
        "cohen_dz",
        "r",
        "pointbiserialr",
        "eta_square",
        "odds_ratio",
        "auc",
        "cles",
    ]


def _check_dataframe(data=None, dv=None, between=None, within=None, subject=None, effects=None):
    """Checks whether data is a dataframe or can be converted to a dataframe.
    If successful, a dataframe is returned. If not successful, a ValueError is
    raised.
    """
    # Check that data is a dataframe. DataMatrix objects can be safely converted to DataFrame
    # objects. By first checking the name of the class, we avoid having to actually import
    # DataMatrix unless it is necessary.
    if not isinstance(data, pd.DataFrame):
        if data.__class__.__name__ != "DataMatrix":
            raise ValueError("Data must be a pandas dataframe or compatible object.")
        try:
            from datamatrix import DataMatrix, convert as cnv  # noqa
        except ImportError:
            raise ValueError(
                "Failed to convert object to pandas dataframe (DataMatrix not available)"
            )
        if not isinstance(data, DataMatrix):
            raise ValueError("Data must be a pandas dataframe or compatible object.")
        data = cnv.to_pandas(data)
    # Check that dv is provided.
    if dv is None:
        raise ValueError("DV and data must be specified")
    # Check that dv is a numeric variable
    if data[dv].dtype.kind not in "fiu":
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


def _is_mpmath_installed(raise_error=False):
    """Check if mpmath is installed."""
    try:
        import mpmath  # noqa

        is_installed = True
    except ImportError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise OSError("mpmath needs to be installed. Please use `pip install mpmath`.")
    return is_installed
