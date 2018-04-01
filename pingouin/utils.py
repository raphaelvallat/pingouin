# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
# GNU license
import numpy as np
from scipy import stats
from six import string_types
import pandas as pd


__all__ = ["_check_eftype", "_check_data"]

# SUB-FUNCTIONS
def _check_eftype(eftype):
    """Check validity of eftype"""
    return True if eftype in ['hedges', 'cohen', 'r', 'eta-square',
                              'odds-ratio', 'AUC'] else False


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
