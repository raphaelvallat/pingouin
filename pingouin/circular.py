# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: July 2018
# Python code inspired from the CircStats MATLAB toolbox (Berens 2009)
# and the brainpipe Python package.
# Reference:
# Berens, Philipp. 2009. CircStat: A MATLAB Toolbox for Circular Statistics.
# Journal of Statistical Software, Articles 31 (10): 1–21.
import numpy as np
from scipy.stats import circmean
from pingouin import _remove_na

__all__ = ["circ_corrcc", "circ_corrcl", "circ_r", "circ_rayleigh"]


def circ_corrcc(x, y, tail='two-sided'):
    """Correlation coefficient between two circular variables.

    Parameters
    ----------
    x : np.array
        First circular variable (expressed in radians)
    y : np.array
        Second circular variable (expressed in radians)
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.

    Returns
    -------
    r : float
        Correlation coefficient
    pval : float
        Uncorrected p-value

    Notes
    -----
    Adapted from the CircStats MATLAB toolbox (Berens 2009).

    Use the np.deg2rad function to convert angles from degrees to radians.

    Please note that NaN are automatically removed.

    Examples
    --------
    Compute the r and p-value of two circular variables

        >>> from pingouin import circ_corrcc
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


def circ_corrcl(x, y, tail='two-sided'):
    """Correlation coefficient between one circular and one linear variable
    random variables.

    Parameters
    ----------
    x : np.array
        First circular variable (expressed in radians)
    y : np.array
        Second circular variable (linear)
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.

    Returns
    -------
    r : float
        Correlation coefficient
    pval : float
        Uncorrected p-value

    Notes
    -----
    Python code borrowed from brainpipe (based on the MATLAB toolbox CircStats)

    Please note that NaN are automatically removed from datasets.

    Examples
    --------
    Compute the r and p-value between one circular and one linear variables.

        >>> from pingouin import circ_corrcl
        >>> x = [0.785, 1.570, 3.141, 0.839, 5.934]
        >>> y = [1.593, 1.291, -0.248, -2.892, 0.102]
        >>> r, pval = circ_corrcl(x, y)
        >>> print(r, pval)
            0.109, 0.971
    """
    from scipy.stats import pearsonr, chi2
    x = np.asarray(x)
    y = np.asarray(y)

    # Check size
    if x.size != y.size:
        raise ValueError('x and y must have the same length.')

    # Remove NA
    x, y = _remove_na(x, y, paired=True)
    n = x.size

    # Compute correlation coefficent for sin and cos independently
    rxs = pearsonr(y, np.sin(x))[0]
    rxc = pearsonr(y, np.cos(x))[0]
    rcs = pearsonr(np.sin(x), np.cos(x))[0]

    # Compute angular-linear correlation (equ. 27.47)
    r = np.sqrt((rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1 - rcs**2))

    # Compute p-value
    pval = chi2.sf(n * r**2, 2)
    pval = pval / 2 if tail == 'one-sided' else pval
    return np.round(r, 3), pval


def circ_r(alpha, w=None, d=None, axis=0):
    """Mean resultant vector length for circular data.

    Parameters
    ----------
    alpha : array
        Sample of angles in radians
    w : array
        Number of incidences in case of binned angle data
    d : float
        Spacing (in radians) of bin centers for binned data. If supplied,
        a correction factor is used to correct for bias in the estimation
        of r.
    axis : int
        Compute along this dimension

    Returns
    -------
    r : float
        Mean resultant length

    Notes
    -----
    The length of the mean resultant vector is a crucial quantity for the
    measurement of circular spread or hypothesis testing in directional
    statistics. The closer it is to one, the more concentrated the data
    sample is around the mean direction (Berens 2009).

    Examples
    --------
    Mean resultant vector length of circular data

        >>> from pingouin import circ_r
        >>> x = [0.785, 1.570, 3.141, 0.839, 5.934]
        >>> circ_r(x)
            0.497
    """
    alpha = np.array(alpha)
    w = np.array(w) if w is not None else np.ones(alpha.shape)
    if alpha.size is not w.size:
        raise ValueError("Input dimensions do not match")

    # Compute weighted sum of cos and sin of angles:
    r = np.multiply(w, np.exp(1j * alpha)).sum(axis=axis)

    # Obtain length:
    r = np.abs(r) / w.sum(axis=axis)

    # For data with known spacing, apply correction factor
    if d is not None:
        c = d / 2 / np.sin(d / 2)
        r = c * r

    return r


def circ_rayleigh(alpha, w=None, d=None):
    """Rayleigh test for non-uniformity of circular data.

    Parameters
    ----------
    alpha : np.array
        Sample of angles in radians.
    w : np.array
        Number of incidences in case of binned angle data.
    d : float
        Spacing (in radians) of bin centers for binned data. If supplied,
        a correction factor is used to correct for bias in the estimation
        of r.

    Returns
    -------
    z : float
        Z-statistic
    pval : float
        P-value

    Notes
    -----
    The Rayleigh test asks how large the resultant vector length R must be
    to indicate a non-uniform  distribution (Fisher 1995).

    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle

    The assumptions for the Rayleigh test are that (1) the distribution has
    only one mode and (2) the data is sampled from a von Mises distribution.

    Examples
    --------
    1. Simple Rayleigh test for non-uniformity of circular data.

        >>> from pingouin import circ_rayleigh
        >>> x = [0.785, 1.570, 3.141, 0.839, 5.934]
        >>> z, pval = circ_rayleigh(x)
        >>> print(z, pval)
            1.236 0.3048

    2. Specifying w and d

        >>> circ_rayleigh(x, w=[.1, .2, .3, .4, .5], d=0.2)
            0.278, 0.807
    """
    alpha = np.array(alpha)
    if w is None:
        r = circ_r(alpha)
        n = len(alpha)
    else:
        if len(alpha) is not len(w):
            raise ValueError("Input dimensions do not match")
        r = circ_r(alpha, w, d)
        n = np.sum(w)

    # Compute Rayleigh's statistic
    R = n * r
    z = (R**2) / n

    # Compute p value using approxation in Zar (1999), p. 617
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))

    return np.round(z, 3), pval
