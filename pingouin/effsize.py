# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
from scipy import stats
import pandas as pd
from pingouin.utils import (_check_data, _check_eftype)


__all__ = ["convert_effsize", "compute_effsize", "compute_effsize_from_T"]


# MAIN FUNCTIONS
def convert_effsize(ef, input_type, output_type, nx=None, ny=None):
    """Conversion between effect sizes
    Parameters
    ----------
    ef: float
        Original effect size
    input_type: string
        Effect size type of ef
    output_type: string
        Desired effect size type
    nx, ny: int, int
        Length of vector x and y.
        nx and ny are required to convert to Hedges g
    Return
    ------
    ef: float
        Desired converted effect size
    """
    # Check input and output type
    for input in [input_type, output_type]:
        if not _check_eftype(input):
            err = "Could not interpret input '{}'".format(input)
            raise ValueError(err)

    if input_type == output_type:
        return ef

    # First convert to Cohen's d
    it = input_type.lower()
    if it == 'r':
        d = (2 * ef) / np.sqrt(1 - ef**2)
    elif it == 'cohen':
        d = ef

    # Then convert to the desired output type
    ot = output_type.lower()
    if ot == 'cohen':
        return d
    elif ot == 'hedges':
        if all(v is not None for v in [nx, ny]):
            return d * (1 - (3 / (4 * (nx + ny) - 9)))
        else:
            # If shapes of x and y are not known, return cohen's d
            print("You need to pass nx and ny arguments to compute Hedges g. Returning Cohen's d instead")
            return d
    elif ot == 'r':
        # McGrath and Meyer 2006
        if all(v is not None for v in [nx, ny]):
            a = ((nx + ny)**2 - 2*(nx + ny)) / (nx*ny)
        else:
            a = 4
        return d / np.sqrt(d**2 + a)
    elif ot == 'eta-square':
        return (d/2)**2 / (1 + (d/2)**2)
    elif ot == 'odds-ratio':
        return np.exp(d * np.pi / np.sqrt(3))
    elif ot == 'auc':
        from scipy.stats import norm
        return norm.cdf(d / np.sqrt(2))
    elif ot == 'none':
        return None


def compute_effsize(dv=None, group=None, data=None, x=None, y=None,
                    eftype=None, paired=False):
    """Compute effect size from pandas dataframe or two numpy arrays
    Case A: pass a DataFrame
        >>>> compute_effsize(dv='Height', 'group='Countries', data=df)
    Case B: pass two vectors
        >>>> x = np.random.normal(loc=172, size=N)
        >>>> y = np.random.normal(loc=175, size=N)
        >>>> compute_effsize(x=x, y=y, data=df)
    Parameters
    ----------
    dv: string
        Column name of dependant variable in data, optional
    group: string
        Column name of group factor in data, optional
    data : DataFrame, optional
        Pandas Dataframe containing columns dv and group
    x, y: vector data, optional
        X and Y are only taken into account if dv, group and data = None
    eftype: string
        Desired output effect size, optional
        Available methods are :
        `none` : no effect size
        `cohen` : Unbiased Cohen's d
        `hedges` : Hedges g
        `eta-square` : Eta-square
        `odds-ratio` : Odds ratio
        `AUC` : Area Under the Curve
    paired : boolean
        If True, use Cohen davg formula (see Cumming 2012)
    Return
    ------
    ef: float
        Effect size
    """
    # Check arguments
    if not _check_eftype(eftype):
        err = "Could not interpret input '{}'".format(eftype)
        raise ValueError(err)

    # Extract data
    x, y, nx, ny, dof = _check_data(dv, group, data, x, y)

    # Compute unbiased Cohen's d effect size
    if not paired:
        # https://en.wikipedia.org/wiki/Effect_size
        d = (np.mean(x) - np.mean(y)) / np.sqrt(((nx - 1) * \
             np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
    else:
        # Report Cohen d-avg (Cumming 2012; Lakens 2013)
        d = (np.mean(x) - np.mean(y)) / (.5 * (np.std(x) + np.std(y)))

    if eftype.lower() == 'cohen':
        return d
    else:
        return convert_effsize(d, 'cohen', eftype, nx=nx, ny=ny)

def compute_effsize_from_T(T, nx=None, ny=None, N=None, eftype='cohen'):
    """Compute effect size from pandas dataframe or two numpy arrays
    Parameters
    ----------
    T: float
        T-value
    nx, ny: int, optional
        Length of vector x and y.
    N: int, optional
        Total sample size (will not be used if nx and ny are specified)
    eftype: string, optional
        desired output effect size
    Return
    ------
    ef: float
        Effect size
    """
    if not _check_eftype(eftype):
        err = "Could not interpret input '{}'".format(eftype)
        raise ValueError(err)

    if not isinstance(T, float):
        err = "T-value must be float"
        raise ValueError(err)

    # Compute Cohen d (Lakens, 2013)
    if nx is not None and ny is not None:
        d = T * np.sqrt(1 / nx + 1 / ny)
    elif N is not None:
        d = 2 * T / np.sqrt(N)
    else:
        raise ValueError('You must specify either nx + ny, or just N')
    return convert_effsize(d, 'cohen', eftype, nx=nx, ny=ny)
