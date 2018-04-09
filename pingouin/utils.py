# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
from six import string_types
import pandas as pd

__all__ = ["print_table", "_export_table", "reshape_data",
           "_check_eftype", "_remove_rm_na", "_check_data", "_check_dataframe",
           "_extract_effects"]


def print_table(df, floatfmt=".4f"):
    """Nice display of table"""
    if 'F' in df.keys():
        print('\n=============\nANOVA SUMMARY\n=============\n')
    elif 'A' in df.keys():
        print('\n==============\nPOST HOC TESTS\n==============\n')

    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", showindex=False, floatfmt=floatfmt))
        print('')
    except BaseException:
        print(df)


def _export_table(table, fname=None):
    """Export DataFrame to .csv"""
    import os.path as op
    extension = op.splitext(fname.lower())[1]
    if extension == '':
        fname = fname + '.csv'
    table.to_csv(fname, index=None, sep=',', encoding='utf-8',
                 float_format='%.4f', decimal='.')

def reshape_data(df, id, dv='DV', rm='Time'):
    """Reshape data from "human-readable" to analysis shape

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


def _remove_rm_na(dv=None, within=None, data=None):
    """Remove subject(s) with one or more missing values in repeated
    measurements.
    """
    rm = list(data[within].dropna().unique())
    n_rm = len(rm)
    n_obs = int(data.groupby(within)[dv].count().max())
    data['ID_Subj'] = np.tile(np.arange(n_obs), n_rm)

    # Efficiently remove subjects with one or more missing values
    data.set_index('ID_Subj', inplace=True)

    # Find index with nan
    iloc_nan = pd.isnull(data).any(1).nonzero()[0]
    idx_nan = data.index[iloc_nan].values
    print('\nNote: %i subject(s) removed because of missing value(s).\n'
          % len(idx_nan))
    return data.drop(idx_nan).reset_index(drop=True)


def _check_eftype(eftype):
    """Check validity of eftype"""
    if eftype.lower() in ['none', 'hedges', 'cohen', 'glass', 'r',
                          'eta-square', 'odds-ratio', 'auc']:
        return True
    else:
        return False


def _check_data(dv=None, group=None, data=None, x=None, y=None):
    """Extract data from dataframe or numpy arrays"""

    # OPTION A: a pandas DataFrame and column names are specified
    # -----------------------------------------------------------
    if all(v is not None for v in [dv, group, data]):
        for input in [dv, group]:
            if not isinstance(input, string_types):
                err = "Could not interpret input '{}'".format(input)
                raise ValueError(err)
        # Check that data is a pandas dataframe
        if not isinstance(data, pd.DataFrame):
            err = "data must be a pandas dataframe"
            raise ValueError(err)
        # Extract group information
        group_keys = data[group].unique()
        if len(group_keys) != 2:
            err = "group must have exactly two unique values."
            raise ValueError(err)
        # Extract data
        x = data[data[group] == group_keys[0]][dv]
        y = data[data[group] == group_keys[1]][dv]

    # OPTION B: x and y vectors are specified
    # ---------------------------------------
    else:
        if all(v is not None for v in [x, y]):
            for input in [x, y]:
                if not isinstance(input, (list, np.ndarray)):
                    err = "x and y must be np.ndarray"
                    raise ValueError(err)
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return x, y, nx, ny, dof


def _check_dataframe(dv=None, between=None, within=None, effects=None,
                     data=None):
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


def _extract_effects(dv=None, between=None, within=None, effects=None,
                     data=None):
    """Extract main effects"""
    # Check the dataframe
    _check_dataframe(dv=dv, between=between, within=within, effects=effects,
                     data=data)

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
