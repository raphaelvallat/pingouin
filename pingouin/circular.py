# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: July 2018
# Python code inspired from the CircStats MATLAB toolbox (Berens 2009)
# and the brainpipe Python package.
# Reference:
# Berens, Philipp. 2009. CircStat: A MATLAB Toolbox for Circular Statistics.
# Journal of Statistical Software, Articles 31 (10): 1–21.
import numpy as np
from scipy.stats import norm

from .utils import remove_na

__all__ = ["convert_angles", "circ_axial", "circ_mean", "circ_r", "circ_corrcc", "circ_corrcl",
           "circ_rayleigh", "circ_vtest"]


###############################################################################
# HELPER FUNCTIONS
###############################################################################

def _checkangles(angles, axis=None):
    """Internal function to check that angles are in radians.
    """
    msg = ("Angles are not in unit of radians. Please use the "
           "`pingouin.convert_angles` function to map your angles to "
           "the [-pi, pi] range.")
    ptp_rad = np.nanmax(angles, axis=axis) - np.nanmin(angles, axis=axis)
    ptp_mask = ptp_rad <= 2 * np.pi
    if not ptp_mask.all():
        raise ValueError(msg)


def convert_angles(angles, low=0, high=360, positive=False):
    """Element-wise conversion of arbitrary-unit circular quantities
    to radians.

    .. versionadded:: 0.3.4

    Parameters
    ----------
    angles : array_like
        Circular data.
    low : float or int, optional
        Low boundary for ``angles`` range.  Default is 0.
    high : float or int, optional
        High boundary for ``angles`` range.  Default is 360
        (for degrees to radians conversion).
    positive : boolean
        If True, radians are mapped on the :math:`[0, 2\\pi]`. Otherwise,
        the resulting angles are mapped from :math:`[-\\pi, \\pi)` (default).

    Returns
    -------
    radians : array_like
        Circular data in radians.

    Notes
    -----
    The formula to convert a set of angles :math:`\\alpha` from an arbitrary
    range :math:`[\\text{high},\\text{low}]` to radians
    :math:`[0, 2\\pi]` is:

    .. math::

        \\alpha_r = \\frac{2\\pi\\alpha}{\\text{high} - \\text{low}}

    If ``positive=False`` (default), the resulting angles in
    radians :math:`\\alpha_r` are then wrapped to the :math:`[-\\pi, \\pi)`
    range:

    .. math::

        (\\text{angle} + \\pi) \\mod 2 \\pi - \\pi

    Examples
    --------
    1. Convert degrees to radians

    >>> from pingouin import convert_angles
    >>> a = [0, 360, 180, 90, 45, 270]
    >>> convert_angles(a, low=0, high=360)
    array([ 0.        ,  0.        , -3.14159265,  1.57079633,  0.78539816,
           -1.57079633])

    with ``positive=True``:

    >>> convert_angles(a, low=0, high=360, positive=True)
    array([0.        , 6.28318531, 3.14159265, 1.57079633, 0.78539816,
           4.71238898])

    2. Convert hours (24h-format) to radians

    >>> sleep_onset = [22.5, 23.25, 24, 0.5, 1]
    >>> convert_angles(sleep_onset, low=0, high=24)
    array([-0.39269908, -0.19634954,  0.        ,  0.13089969,  0.26179939])

    3. Convert radians from :math:`[0, 2\\pi]` to :math:`[-\\pi, \\pi)`:

    >>> import numpy as np
    >>> rad = [0.1, 3.14, 5, 2, 6]
    >>> convert_angles(rad, low=0, high=2*np.pi)
    array([ 0.1       ,  3.14      , -1.28318531,  2.        , -0.28318531])

    4. Convert degrees from a 2-D array

    >>> np.random.seed(123)
    >>> deg = np.random.randint(low=0, high=360, size=(3, 4))
    >>> convert_angles(deg)
    array([[-0.66322512,  1.71042267, -2.26892803,  0.29670597],
           [ 1.44862328,  1.85004901,  2.14675498,  0.99483767],
           [-2.54818071, -2.35619449,  1.67551608,  1.97222205]])
    """
    assert isinstance(positive, bool)
    assert isinstance(high, (int, float)), 'high must be numeric'
    assert isinstance(low, (int, float)), 'low must be numeric'
    ptp = high - low
    assert ptp > 0, 'high - low must be strictly positive.'
    angles = np.asarray(angles)
    assert np.nanmin(angles) >= low, 'angles cannot be >= low.'
    assert np.nanmax(angles) <= high, 'angles cannot be <= high.'
    # Map to [0, 2pi] range
    rad = angles * (2 * np.pi) / ptp
    if not positive:
        # https://stackoverflow.com/a/29237626/10581531
        # Map to [-pi, pi) range:
        rad = (rad + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi)
        # Map to (-pi, pi] range:
        # rad = np.angle(np.exp(1j * rad))
        # rad = -1 * ((-rad + np.pi) % (2 * np.pi) - np.pi)
    return rad


def circ_axial(angles, n):
    """Transforms n-axial data to a common scale.

    Parameters
    ----------
    angles : array
        Sample of angles in radians
    n : int
        Number of modes

    Returns
    -------
    angles : float
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
    >>> angles = df['Orientation'].to_numpy()
    >>> angles = circ_axial(np.deg2rad(angles), 2)
    """
    angles = np.asarray(angles)
    return np.remainder(angles * n, 2 * np.pi)


###############################################################################
# DESCRIPTIVE STATISTICS
###############################################################################

def circ_mean(angles, w=None, axis=0):
    """Mean direction for (binned) circular data.

    Parameters
    ----------
    angles : array_like
        Samples of angles in radians. The range of ``angles`` must be either
        :math:`[0, 2\\pi]` or :math:`[-\\pi, \\pi]`. If ``angles`` is not
        expressed in radians (e.g. degrees or 24-hours), please use the
        :py:func:`pingouin.convert_angles` function prior to using the present
        function.
    w : array_like
        Number of incidences per bins (i.e. "weights"), in case of binned angle
        data.
    axis : int or None
        Compute along this dimension. Default is the first axis (0).

    Returns
    -------
    mu : float
        Circular mean, in radians.

    See also
    --------
    scipy.stats.circmean, scipy.stats.circstd, pingouin.circ_r

    Notes
    -----
    From Wikipedia:

    *In mathematics, a mean of circular quantities is a mean which is sometimes
    better-suited for quantities like angles, daytimes, and fractional parts
    of real numbers. This is necessary since most of the usual means may not be
    appropriate on circular quantities. For example, the arithmetic mean of 0°
    and 360° is 180°, which is misleading because for most purposes 360° is
    the same thing as 0°.
    As another example, the "average time" between 11 PM and 1 AM is either
    midnight or noon, depending on whether the two times are part of a single
    night or part of a single calendar day.*

    The circular mean of a set of angles :math:`\\alpha` is defined by:

    .. math::

        \\bar{\\alpha} =  \\text{angle} \\left ( \\sum_{j=1}^n \\exp(i \\cdot
        \\alpha_j) \\right )

    For binned angles with weights :math:`w`, this becomes:

    .. math::

        \\bar{\\alpha} =  \\text{angle} \\left ( \\sum_{j=1}^n w \\cdot
        \\exp(i \\cdot \\alpha_j) \\right )

    Missing values in ``angles`` are omitted from the calculations.

    References
    ----------
    * https://en.wikipedia.org/wiki/Mean_of_circular_quantities

    * Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, Articles, 31(10),
      1–21. https://doi.org/10.18637/jss.v031.i10

    Examples
    --------
    1. Circular mean of a 1-D array of angles, in radians

    >>> import pingouin as pg
    >>> angles = [0.785, 1.570, 3.141, 0.839, 5.934]
    >>> round(pg.circ_mean(angles), 4)
    1.013

    Compare with SciPy:

    >>> from scipy.stats import circmean
    >>> import numpy as np
    >>> round(circmean(angles, low=0, high=2*np.pi), 4)
    1.013

    2. Using a 2-D array of angles in degrees

    >>> np.random.seed(123)
    >>> deg = np.random.randint(low=0, high=360, size=(3, 5))
    >>> deg
    array([[322,  98, 230,  17,  83],
           [106, 123,  57, 214, 225],
           [ 96, 113, 126,  47,  73]])

    We first need to convert from degrees to radians:

    >>> rad = np.round(pg.convert_angles(deg, low=0, high=360), 4)
    >>> rad
    array([[-0.6632,  1.7104, -2.2689,  0.2967,  1.4486],
           [ 1.85  ,  2.1468,  0.9948, -2.5482, -2.3562],
           [ 1.6755,  1.9722,  2.1991,  0.8203,  1.2741]])

    >>> pg.circ_mean(rad)  # On the first axis (default)
    array([1.27532162, 1.94336576, 2.23195927, 0.52110503, 1.80240563])
    >>> pg.circ_mean(rad, axis=-1)  # On the last axis (default)
    array([0.68920819, 2.49334852, 1.5954149 ])
    >>> round(pg.circ_mean(rad, axis=None), 4)  # Across the entire array
    1.6954

    Missing values are omitted from the calculations:

    >>> rad[0, 0] = np.nan
    >>> pg.circ_mean(rad)
    array([1.76275   , 1.94336576, 2.23195927, 0.52110503, 1.80240563])

    3. Using binned angles

    >>> np.random.seed(123)
    >>> nbins = 18  # Number of bins to divide the unit circle
    >>> angles_bins = np.linspace(0, 2 * np.pi, nbins)
    >>> # w represents the number of incidences per bins, or "weights".
    >>> w = np.random.randint(low=0, high=5, size=angles_bins.size)
    >>> round(pg.circ_mean(angles_bins, w), 4)
    0.606
    """
    angles = np.asarray(angles)
    _checkangles(angles)  # Check that angles is in radians
    w = np.asarray(w) if w is not None else np.ones(angles.shape)
    assert angles.shape == w.shape, "Input dimensions do not match"
    return np.angle(np.nansum(np.multiply(w, np.exp(1j * angles)), axis=axis))


def circ_r(angles, w=None, d=None, axis=0):
    """Mean resultant vector length for circular data.

    Parameters
    ----------
    angles : array_like
        Samples of angles in radians. The range of ``angles`` must be either
        :math:`[0, 2\\pi]` or :math:`[-\\pi, \\pi]`. If ``angles`` is not
        expressed in radians (e.g. degrees or 24-hours), please use the
        :py:func:`pingouin.convert_angles` function prior to using the present
        function.
    w : array_like
        Number of incidences per bins (i.e. "weights"), in case of binned angle
        data.
    d : float
        Spacing (in radians) of bin centers for binned data. If supplied,
        a correction factor is used to correct for bias in the estimation
        of r.
    axis : int or None
        Compute along this dimension. Default is the first axis (0).

    Returns
    -------
    r : float
        Circular mean vector length.

    See also
    --------
    pingouin.circ_mean

    Notes
    -----
    The length of the mean resultant vector is a crucial quantity for the
    measurement of circular spread or hypothesis testing in directional
    statistics. The closer it is to one, the more concentrated the data
    sample is around the mean direction (Berens 2009).

    The circular vector length of a set of angles :math:`\\alpha` is defined
    by:

    .. math::

        \\bar{\\alpha} =  \\frac{1}{N}\\left \\| \\sum_{j=1}^n
        \\exp(i \\cdot \\alpha_j) \\right \\|

    Missing values in ``angles`` are omitted from the calculations.

    References
    ----------
    * https://en.wikipedia.org/wiki/Mean_of_circular_quantities

    * Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, Articles, 31(10),
      1–21. https://doi.org/10.18637/jss.v031.i10

    Examples
    --------
    1. Mean resultant vector length of a 1-D array of angles, in radians

    >>> import pingouin as pg
    >>> angles = [0.785, 1.570, 3.141, 0.839, 5.934]
    >>> r = pg.circ_r(angles)
    >>> round(r, 4)
    0.4972

    Note that there is a close relationship between the vector length and the
    circular standard deviation, i.e. :math:`\\sigma = \\sqrt{-2 \\ln R}`:

    >>> import numpy as np
    >>> round(np.sqrt(-2 * np.log(r)), 4)
    1.1821

    which gives similar result as SciPy built-in function:

    >>> from scipy.stats import circstd
    >>> round(circstd(angles), 4)
    1.1821

    Sanity check: if all angles are the same, the vector length should be one:

    >>> angles = [3.14, 3.14, 3.14, 3.14]
    >>> round(pg.circ_r(angles), 4)
    1.0

    2. Using a 2-D array of angles in degrees

    >>> np.random.seed(123)
    >>> deg = np.random.randint(low=0, high=360, size=(3, 5))
    >>> deg
    array([[322,  98, 230,  17,  83],
           [106, 123,  57, 214, 225],
           [ 96, 113, 126,  47,  73]])

    We first need to convert from degrees to radians:

    >>> rad = np.round(pg.convert_angles(deg, low=0, high=360), 4)
    >>> rad
    array([[-0.6632,  1.7104, -2.2689,  0.2967,  1.4486],
           [ 1.85  ,  2.1468,  0.9948, -2.5482, -2.3562],
           [ 1.6755,  1.9722,  2.1991,  0.8203,  1.2741]])

    >>> pg.circ_r(rad)  # On the first axis (default)
    array([0.46695499, 0.98398294, 0.3723287 , 0.31103746, 0.42527149])
    >>> pg.circ_r(rad, axis=-1)  # On the last axis (default)
    array([0.28099998, 0.45456096, 0.88261161])
    >>> round(pg.circ_r(rad, axis=None), 4)  # Across the entire array
    0.4486

    Missing values are omitted from the calculations:

    >>> rad[0, 0] = np.nan
    >>> pg.circ_r(rad)
    array([0.99619613, 0.98398294, 0.3723287 , 0.31103746, 0.42527149])

    3. Using binned angles

    >>> np.random.seed(123)
    >>> nbins = 18  # Number of bins to divide the unit circle
    >>> angles_bins = np.linspace(0, 2 * np.pi, nbins)
    >>> # w represents the number of incidences per bins, or "weights".
    >>> w = np.random.randint(low=0, high=5, size=angles_bins.size)
    >>> round(pg.circ_r(angles_bins, w), 4)
    0.3642
    """
    angles = np.asarray(angles)
    _checkangles(angles)  # Check that angles is in radians
    w = np.asarray(w) if w is not None else np.ones(angles.shape)
    assert angles.shape == w.shape, "Input dimensions do not match."

    # Add np.nan in weight vector (otherwise nansum(w) is wrong)
    w = w.astype(float)
    w[np.isnan(angles)] = np.nan

    # Compute weighted sum of cos and sin of angles:
    r = np.nansum(np.multiply(w, np.exp(1j * angles)), axis=axis)

    # Calculate vector length:
    r = np.abs(r) / np.nansum(w, axis=axis)

    # For data with known spacing, apply correction factor
    if d is not None:
        c = d / 2 / np.sin(d / 2)
        r = c * r
    return r


###############################################################################
# INFERENTIAL STATISTICS
###############################################################################

def circ_corrcc(x, y, correction_uniform=False):
    """Correlation coefficient between two circular variables.

    Parameters
    ----------
    x : 1-D array_like
        First circular variable (expressed in radians).
    y : 1-D array_like
        Second circular variable (expressed in radians).
    correction_uniform : bool
        Use correction for uniform marginals.

    Returns
    -------
    r : float
        Correlation coefficient.
    pval : float
        Uncorrected p-value.

    Notes
    -----
    Adapted from the CircStats MATLAB toolbox [1]_.

    The range of ``x`` and ``y`` must be either
    :math:`[0, 2\\pi]` or :math:`[-\\pi, \\pi]`. If ``angles`` is not
    expressed in radians (e.g. degrees or 24-hours), please use the
    :py:func:`pingouin.convert_angles` function prior to using the present
    function.

    Please note that NaN are automatically removed.

    If the ``correction_uniform`` is True, an alternative equation from
    [2]_ (p. 177) is used. If the marginal distribution of ``x`` or ``y`` is
    uniform, the mean is not well defined, which leads to wrong estimates
    of the circular correlation. The alternative equation corrects for this
    by choosing the means in a way that maximizes the positive or negative
    correlation.

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
    >>> print(round(r, 3), round(pval, 4))
    0.942 0.0658

    With the correction for uniform marginals

    >>> r, pval = circ_corrcc(x, y, correction_uniform=True)
    >>> print(round(r, 3), round(pval, 4))
    0.547 0.2859
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.size == y.size, 'x and y must have the same length.'

    # Remove NA
    x, y = remove_na(x, y, paired=True)
    n = x.size

    # Compute correlation coefficient
    x_sin = np.sin(x - circ_mean(x))
    y_sin = np.sin(y - circ_mean(y))

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
    return r, pval


def circ_corrcl(x, y):
    """Correlation coefficient between one circular and one linear variable
    random variables.

    Parameters
    ----------
    x : 1-D array_like
        First circular variable (expressed in radians).
        The range of ``x`` must be either :math:`[0, 2\\pi]` or
        :math:`[-\\pi, \\pi]`. If ``angles`` is not
        expressed in radians (e.g. degrees or 24-hours), please use the
        :py:func:`pingouin.convert_angles` function prior to using the present
        function.
    y : 1-D array_like
        Second circular variable (linear)

    Returns
    -------
    r : float
        Correlation coefficient
    pval : float
        Uncorrected p-value

    Notes
    -----
    Please note that NaN are automatically removed from datasets.

    Examples
    --------
    Compute the r and p-value between one circular and one linear variables.

    >>> from pingouin import circ_corrcl
    >>> x = [0.785, 1.570, 3.141, 0.839, 5.934]
    >>> y = [1.593, 1.291, -0.248, -2.892, 0.102]
    >>> r, pval = circ_corrcl(x, y)
    >>> print(round(r, 3), round(pval, 3))
    0.109 0.971
    """
    from scipy.stats import pearsonr, chi2
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.size == y.size, 'x and y must have the same length.'

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
    return r, pval


def circ_rayleigh(angles, w=None, d=None):
    """Rayleigh test for non-uniformity of circular data.

    Parameters
    ----------
    angles : 1-D array_like
        Samples of angles in radians. The range of ``angles`` must be either
        :math:`[0, 2\\pi]` or :math:`[-\\pi, \\pi]`. If ``angles`` is not
        expressed in radians (e.g. degrees or 24-hours), please use the
        :py:func:`pingouin.convert_angles` function prior to using the present
        function.
    w : array_like
        Number of incidences per bins (i.e. "weights"), in case of binned angle
        data.
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
    >>> print(round(z, 3), round(pval, 6))
    1.236 0.304844

    2. Specifying w and d

    >>> z, pval = circ_rayleigh(x, w=[.1, .2, .3, .4, .5], d=0.2)
    >>> print(round(z, 3), round(pval, 6))
    0.278 0.806997
    """
    angles = np.asarray(angles)
    _checkangles(angles)  # Check that angles is in radians
    if w is None:
        r = circ_r(angles)
        n = len(angles)
    else:
        assert len(angles) == len(w), "Input dimensions do not match"
        r = circ_r(angles, w, d)
        n = np.sum(w)

    # Compute Rayleigh's statistic
    R = n * r
    z = (R**2) / n

    # Compute p value using approxation in Zar (1999), p. 617
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))
    return z, pval


def circ_vtest(angles, dir=0., w=None, d=None):
    """V test for non-uniformity of circular data with a specified
    mean direction.

    Parameters
    ----------
    angles : 1-D array_like
        Samples of angles in radians. The range of ``angles`` must be either
        :math:`[0, 2\\pi]` or :math:`[-\\pi, \\pi]`. If ``angles`` is not
        expressed in radians (e.g. degrees or 24-hours), please use the
        :py:func:`pingouin.convert_angles` function prior to using the present
        function.
    dir : float
        Suspected mean direction (angle in radians).
    w : array_like
        Number of incidences per bins (i.e. "weights"), in case of binned angle
        data.
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
    >>> print(round(v, 3), pval)
    2.486 0.05794648732225438

    2. Specifying w and d

    >>> v, pval = circ_vtest(x, dir=0.5, w=[.1, .2, .3, .4, .5], d=0.2)
    >>> print(round(v, 3), round(pval, 5))
    0.637 0.23086
    """
    angles = np.asarray(angles)
    if w is None:
        r = circ_r(angles)
        mu = circ_mean(angles)
        n = len(angles)
    else:
        assert len(angles) == len(w), "Input dimensions do not match"
        r = circ_r(angles, w, d)
        mu = circ_mean(angles, w)
        n = np.sum(w)

    # Compute Rayleigh and V statistics
    R = n * r
    v = R * np.cos(mu - dir)

    # Compute p value
    u = v * np.sqrt(2 / n)
    pval = 1 - norm.cdf(u)
    return v, pval
