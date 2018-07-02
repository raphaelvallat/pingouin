# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: July 2018
# Translated from the CircStats MATLAB toolbox.
# Berens, Philipp. 2009. CircStat: A MATLAB Toolbox for Circular Statistics.
# Journal of Statistical Software, Articles 31 (10): 1–21.
import numpy as np
from scipy.stats import circmean
from pingouin import _remove_na

__all__ = ["circ_corrcc"]


def circ_corrcc(x, y, tail='two-sided'):
    """Correlation coefficient between two circular variables.

    Parameters
    ----------
    x: np.array
        First circular variable (expressed in radians)
    y: np.array
        Second circular variable (expressed in radians)
    tail: string
        Specify whether to return 'one-sided' or 'two-sided' p-value.

    Returns
    -------
    r: float
        Correlation coefficient
    pval: float
        Uncorrected p-value

    Notes
    -----
    Adapted from the CircStats MATLAB toolbox (Berens 2009).

    Use the np.deg2rad function to convert angles from degrees to radians.

    Please note that NaN are automatically removed.

    Examples
    --------
    Compute the r and p-value of two circular variables

        >>> x = [0.785, 1.570, 3.141, 3.839, 5.934]
        >>> y = [0.593, 1.291, 2.879, 3.892, 6.108]
        >>> r, pval = circ_corrcc(x, y)
        >>> print(r, pval)
            0.942, 0.066
    """
    from scipy.stats import norm
    x = np.asarray(x)
    y = np.asarray(y)

    # Check size
    if x.size != y.size:
        raise ValueError('x and y must have the same length.')

    # Remove NA
    x, y = _remove_na(x, y, paired=True)
    n = x.size

    # Compute correlation coefficient
    x_sin = np.sin(x - circmean(x))
    y_sin = np.sin(y - circmean(y))
    r = np.sum(x_sin * y_sin) / np.sqrt(np.sum(x_sin**2) * np.sum(y_sin**2))

    # Compute T- and p-values
    tval = np.sqrt((n * (x_sin**2).mean() * (y_sin**2).mean()) /
                   np.mean(x_sin**2 * y_sin**2)) * r
    # Approximately distributed as a standard normal
    pval = 2 * norm.sf(abs(tval))
    pval = pval / 2 if tail == 'one-sided' else pval
    return np.round(r, 3), pval
