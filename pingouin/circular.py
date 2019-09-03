# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: July 2018
# Python code inspired from the CircStats MATLAB toolbox (Berens 2009)
# and the brainpipe Python package.
# Reference:
# Berens, Philipp. 2009. CircStat: A MATLAB Toolbox for Circular Statistics.
# Journal of Statistical Software, Articles 31 (10): 1–21.
import numpy as np
from scipy.stats import circmean
from pingouin import remove_na

__all__ = ["circ_axial", "circ_corrcc", "circ_corrcl", "circ_mean", "circ_r",
           "circ_rayleigh", "circ_vtest"]


def circ_axial(alpha, n):
    """Transforms n-axial data to a common scale.

    Parameters
    ----------
    alpha : array
        Sample of angles in radians
    n : int
        Number of modes

    Returns
    -------
    alpha : float
        Transformed angles

    Notes
    -----
    Tranform data with multiple modes (known as axial data) to a unimodal
    sample, for the purpose of certain analysis such as computation of a
    mean resultant vector (see Berens 2009).

    Examples
    --------
    Transform degrees to unimodal radians in the Berens 2009 neuro dataset.

    >>> import numpy as np
    >>> from pingouin import read_dataset
    >>> from pingouin.circular import circ_axial
    >>> df = read_dataset('circular')
    >>> alpha = df['Orientation'].values
    >>> alpha = circ_axial(np.deg2rad(alpha), 2)
    """
    alpha = np.array(alpha)
    return np.remainder(alpha * n, 2 * np.pi)


def circ_corrcc(x, y, tail='two-sided', correction_uniform=False):
    """Correlation coefficient between two circular variables.

    Parameters
    ----------
    x : np.array
        First circular variable (expressed in radians)
    y : np.array
        Second circular variable (expressed in radians)
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.
    correction_uniform : bool
        Use correction for uniform marginals

    Returns
    -------
    r : float
        Correlation coefficient
    pval : float
        Uncorrected p-value

    Notes
    -----
    Adapted from the CircStats MATLAB toolbox (Berens 2009).

    Use the :py:func:`numpy.deg2rad` function to convert angles from degrees
    to radians.

    Please note that NaN are automatically removed.

    If the ``correction_uniform`` is True, an alternative equation from
    Jammalamadaka & Sengupta (2001, p. 177) is used.
    If the marginal distribution of ``x`` or ``y`` is uniform, the mean is
    not well defined, which leads to wrong estimates of the circular
    correlation. The alternative equation corrects for this by choosing the
    means in a way that maximizes the postitive or negative correlation.

    References
    ----------
    .. [1] Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
           Statistics. Journal of Statistical Software, Articles, 31(10), 1–21.
           https://doi.org/10.18637/jss.v031.i10

    .. [2] Jammalamadaka, S. R., & Sengupta, A. (2001). Topics in circular
           statistics (Vol. 5). world scientific.

    Examples
    --------
    Compute the r and p-value of two circular variables

    >>> from pingouin import circ_corrcc
    >>> x = [0.785, 1.570, 3.141, 3.839, 5.934]
    >>> y = [0.593, 1.291, 2.879, 3.892, 6.108]
    >>> r, pval = circ_corrcc(x, y)
    >>> print(r, pval)
    0.942 0.06579836070349088

    With the correction for uniform marginals

    >>> r, pval = circ_corrcc(x, y, correction_uniform=True)
    >>> print(r, pval)
    0.547 0.28585306869206784
    """
    from scipy.stats import norm
    x = np.asarray(x)
    y = np.asarray(y)

    # Check size
    if x.size != y.size:
        raise ValueError('x and y must have the same length.')

    # Remove NA
    x, y = remove_na(x, y, paired=True)
    n = x.size

    # Compute correlation coefficient
    x_sin = np.sin(x - circmean(x))
    y_sin = np.sin(y - circmean(y))

    if not correction_uniform:
        # Similar to np.corrcoef(x_sin, y_sin)[0][1]
        r = np.sum(x_sin * y_sin) / np.sqrt(np.sum(x_sin**2) *
                                            np.sum(y_sin**2))
    else:
        r_minus = np.abs(np.sum(np.exp((x - y) * 1j)))
        r_plus = np.abs(np.sum(np.exp((x + y) * 1j)))
        denom = 2 * np.sqrt(np.sum(x_sin ** 2) * np.sum(y_sin ** 2))
        r = (r_minus - r_plus) / denom

    # Compute T- and p-values
    tval = np.sqrt((n * (x_sin**2).mean() * (y_sin**2).mean())
                   / np.mean(x_sin**2 * y_sin**2)) * r

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
    0.109 0.9708899750629237
    """
    from scipy.stats import pearsonr, chi2
    x = np.asarray(x)
    y = np.asarray(y)

    # Check size
    if x.size != y.size:
        raise ValueError('x and y must have the same length.')

    # Remove NA
    x, y = remove_na(x, y, paired=True)
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


def circ_mean(alpha, w=None, axis=0):
    """Mean direction for circular data.

    Parameters
    ----------
    alpha : array
        Sample of angles in radians
    w : array
        Number of incidences in case of binned angle data
    axis : int
        Compute along this dimension

    Returns
    -------
    mu : float
        Mean direction

    Examples
    --------
    Mean resultant vector of circular data

    >>> from pingouin import circ_mean
    >>> alpha = [0.785, 1.570, 3.141, 0.839, 5.934]
    >>> circ_mean(alpha)
    1.012962445838065
    """
    alpha = np.array(alpha)
    if isinstance(w, (list, np.ndarray)):
        w = np.array(w)
        if alpha.shape != w.shape:
            raise ValueError("w must have the same shape as alpha.")
    else:
        w = np.ones_like(alpha)
    return np.angle(np.multiply(w, np.exp(1j * alpha)).sum(axis=axis))


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
    0.49723034495605356
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
    1.236 0.3048435876500138

    2. Specifying w and d

    >>> circ_rayleigh(x, w=[.1, .2, .3, .4, .5], d=0.2)
    (0.278, 0.8069972000769801)
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


def circ_vtest(alpha, dir=0., w=None, d=None):
    """V test for non-uniformity of circular data with a specified
    mean direction.

    Parameters
    ----------
    alpha : np.array
        Sample of angles in radians.
    dir : float
        Suspected mean direction (angle in radians).
    w : np.array
        Number of incidences in case of binned angle data.
    d : float
        Spacing (in radians) of bin centers for binned data. If supplied,
        a correction factor is used to correct for bias in the estimation
        of r.

    Returns
    -------
    V : float
        V-statistic
    pval : float
        P-value

    Notes
    -----
    H0: the population is uniformly distributed around the circle.
    HA: the population is not distributed uniformly around the circle but
    has a mean of dir.

    Note: Not rejecting H0 may mean that the population is uniformly
    distributed around the circle OR that it has a mode but that this mode
    is not centered at dir.

    The V test has more power than the Rayleigh test and is preferred if
    there is reason to believe in a specific mean direction.

    Adapted from the Matlab Circular Statistics Toolbox.

    Examples
    --------
    1. V-test for non-uniformity of circular data.

    >>> from pingouin import circ_vtest
    >>> x = [0.785, 1.570, 3.141, 0.839, 5.934]
    >>> v, pval = circ_vtest(x, dir=1)
    >>> print(v, pval)
    2.486 0.05794648732225438

    2. Specifying w and d

    >>> circ_vtest(x, dir=0.5, w=[.1, .2, .3, .4, .5], d=0.2)
    (0.637, 0.23086492929174185)
    """
    from scipy.stats import norm
    alpha = np.array(alpha)
    if w is None:
        r = circ_r(alpha)
        mu = circ_mean(alpha)
        n = len(alpha)
    else:
        if len(alpha) is not len(w):
            raise ValueError("Input dimensions do not match")
        r = circ_r(alpha, w, d)
        mu = circ_mean(alpha, w)
        n = np.sum(w)

    # Compute Rayleigh and V statistics
    R = n * r
    v = R * np.cos(mu - dir)

    # Compute p value
    u = v * np.sqrt(2 / n)
    pval = 1 - norm.cdf(u)

    return np.round(v, 3), pval
