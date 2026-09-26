# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import warnings

import numpy as np
from scipy import stats
from scipy.optimize import brenth

from .utils import _check_alternative

__all__ = [
    "power_ttest",
    "power_ttest2n",
    "power_anova",
    "power_rm_anova",
    "power_corr",
    "power_chi2",
]

# Bracketing interval of the Cohen d when solving for the effect size of a T-test
_D_BOUNDS = {"two-sided": (1e-07, 10), "less": (-10, 5), "greater": (-5, 10)}


def _solve(func, lower, upper):
    """Find the root of ``func`` in [lower, upper] with Brent's method, or NaN if it fails."""
    try:
        return brenth(func, lower, upper)
    except ValueError:  # pragma: no cover
        return np.nan


def _ncf_sf(fcrit, dof1, dof2, nc):
    """Survival function of the non-central F distribution, i.e. the power of an F-test."""
    return stats.ncf.sf(fcrit, dof1, dof2, nc)


def _power_nct(nc, dof, alpha, alternative):
    """Power of a T-test, given the non-centrality parameter and degrees of freedom."""
    if alternative == "less":
        tcrit = stats.t.ppf(alpha, dof)
        return 1 - stats.nct.sf(tcrit, dof, nc)
    elif alternative == "two-sided":
        tcrit = stats.t.ppf(1 - alpha / 2, dof)
        return stats.nct.sf(tcrit, dof, nc) + (1 - stats.nct.sf(-tcrit, dof, nc))
    else:  # alternative = "greater"
        tcrit = stats.t.ppf(1 - alpha, dof)
        return stats.nct.sf(tcrit, dof, nc)


def power_ttest(
    d=None, n=None, power=None, alpha=0.05, contrast="two-samples", alternative="two-sided"
):
    """
    Evaluate power, sample size, effect size or significance level of a one-sample T-test,
    a paired T-test or an independent two-samples T-test with equal sample sizes.

    Parameters
    ----------
    d : float
        Cohen d effect size
    n : int
        Sample size
        In case of a two-sample T-test, sample sizes are assumed to be equal.
        Otherwise, see the :py:func:`power_ttest2n` function.
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability).
        The default is 0.05.
    contrast : str
        Can be `"one-sample"`, `"two-samples"` or `"paired"`.
        Note that `"one-sample"` and `"paired"` have the same behavior.
    alternative : string
        Defines the alternative hypothesis, or tail of the test. Must be one of
        "two-sided" (default), "greater" or "less".

    Notes
    -----
    Exactly ONE of the parameters ``d``, ``n``, ``power`` and ``alpha`` must be passed as None, and
    that parameter is determined from the others.

    For a paired T-test, the sample size ``n`` corresponds to the number of pairs. For an
    independent two-sample T-test with equal sample sizes, ``n`` corresponds to the sample size of
    each group (i.e. number of observations in one group). If the sample sizes are unequal, please
    use the :py:func:`power_ttest2n` function instead.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to
    compute it.

    This function is a Python adaptation of the `pwr.t.test` function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Statistical power is the likelihood that a study will detect an effect when there is an effect
    there to be detected. A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one. Statistical power is mainly affected by
    the effect size and the sample size.

    The first step is to use the Cohen's d to calculate the non-centrality parameter
    :math:`\\delta` and degrees of freedom :math:`v`. In case of paired groups, this is:

    .. math:: \\delta = d * \\sqrt n
    .. math:: v = n - 1

    and in case of independent groups with equal sample sizes:

    .. math:: \\delta = d * \\sqrt{\\frac{n}{2}}
    .. math:: v = (n - 1) * 2

    where :math:`d` is the Cohen d and :math:`n` the sample size.

    The critical value is then found using the percent point function of the T distribution with
    :math:`q = 1 - alpha` and :math:`v` degrees of freedom.

    Finally, the power of the test is given by the survival function of the non-central
    distribution using the previously calculated critical value, degrees of freedom and
    non-centrality parameter.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables (i.e.
    sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    Results have been tested against GPower and the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Examples
    --------
    1. Compute power of a one-sample T-test given ``d``, ``n`` and ``alpha``

    >>> from pingouin import power_ttest
    >>> print("power: %.4f" % power_ttest(d=0.5, n=20, contrast="one-sample"))
    power: 0.5645

    2. Compute required sample size given ``d``, ``power`` and ``alpha``

    >>> print("n: %.4f" % power_ttest(d=0.5, power=0.80, alternative="greater"))
    n: 50.1508

    3. Compute achieved ``d`` given ``n``, ``power`` and ``alpha`` level

    >>> print("d: %.4f" % power_ttest(n=20, power=0.80, alpha=0.05, contrast="paired"))
    d: 0.6604

    4. Compute achieved alpha level given ``d``, ``n`` and ``power``

    >>> print("alpha: %.4f" % power_ttest(d=0.5, n=20, power=0.80, alpha=None))
    alpha: 0.4430

    5. One-sided tests

    >>> from pingouin import power_ttest
    >>> print("power: %.4f" % power_ttest(d=0.5, n=20, alternative="greater"))
    power: 0.4634

    >>> print("power: %.4f" % power_ttest(d=0.5, n=20, alternative="less"))
    power: 0.0007
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [d, n, power, alpha]])
    if n_none != 1:
        raise ValueError("Exactly one of n, d, power, and alpha must be None.")

    # Safety checks
    _check_alternative(alternative)
    assert contrast.lower() in ["one-sample", "paired", "two-samples"]
    tsample = 2 if contrast.lower() == "two-samples" else 1
    if d is not None and alternative == "two-sided":
        d = abs(d)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    def func(d, n, alpha):
        return _power_nct(d * np.sqrt(n / tsample), (n - 1) * tsample, alpha, alternative)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power given d, n and alpha
        return func(d, n, alpha)
    elif n is None:
        # Compute required sample size given d, power and alpha
        return _solve(lambda n: func(d, n, alpha) - power, 2 + 1e-10, 1e07)
    elif d is None:
        # Compute achieved d given sample size, power and alpha level
        return _solve(lambda d: func(d, n, alpha) - power, *_D_BOUNDS[alternative])
    else:
        # Compute achieved alpha (significance) level given d, n and power
        return _solve(lambda alpha: func(d, n, alpha) - power, 1e-10, 1 - 1e-10)


def power_ttest2n(nx, ny, d=None, power=None, alpha=0.05, alternative="two-sided"):
    """
    Evaluate power, effect size or  significance level of an independent two-samples T-test
    with unequal sample sizes.

    Parameters
    ----------
    nx, ny : int
        Sample sizes. Must be specified. If the sample sizes are equal, you should use the
        :py:func:`power_ttest` function instead.
    d : float
        Cohen d effect size
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability). The default is 0.05.
    alternative : string
        Defines the alternative hypothesis, or tail of the test. Must be one of "two-sided"
        (default), "greater" or "less".

    Notes
    -----
    Exactly ONE of the parameters ``d``, ``power`` and ``alpha`` must be passed as None, and that
    parameter is determined from the others.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to compute
    it.

    This function is a Python adaptation of the `pwr.t2n.test` function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Statistical power is the likelihood that a study will detect an effect when there is an effect
    there to be detected. A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one. Statistical power is mainly affected by
    the effect size and the sample size.

    The first step is to use the Cohen's d to calculate the non-centrality parameter
    :math:`\\delta` and degrees of freedom :math:`v`.cIn case of two independent groups with
    unequal sample sizes, this is:

    .. math:: \\delta = d * \\sqrt{\\frac{n_i * n_j}{n_i + n_j}}
    .. math:: v = n_i + n_j - 2

    where :math:`d` is the Cohen d, :math:`n` the sample size,
    :math:`n_i` the sample size of the first group and
    :math:`n_j` the sample size of the second group,

    The critical value is then found using the percent point function of the T distribution with
    :math:`q = 1 - alpha` and :math:`v` degrees of freedom.

    Finally, the power of the test is given by the survival function of the non-central
    distribution using the previously calculated critical value, degrees of freedom and
    non-centrality parameter.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables (i.e.
    sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    Results have been tested against GPower and the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Examples
    --------
    1. Compute achieved power of a T-test given ``d``, ``n`` and ``alpha``

    >>> from pingouin import power_ttest2n
    >>> print("power: %.4f" % power_ttest2n(nx=20, ny=15, d=0.5, alternative="greater"))
    power: 0.4164

    2. Compute achieved ``d`` given ``n``, ``power`` and ``alpha`` level

    >>> print("d: %.4f" % power_ttest2n(nx=20, ny=15, power=0.80, alpha=0.05))
    d: 0.9859

    3. Compute achieved alpha level given ``d``, ``n`` and ``power``

    >>> print("alpha: %.4f" % power_ttest2n(nx=20, ny=15, d=0.5, power=0.80, alpha=None))
    alpha: 0.5000
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [d, power, alpha]])
    if n_none != 1:
        raise ValueError("Exactly one of d, power, and alpha must be None")

    # Safety checks
    _check_alternative(alternative)
    if d is not None and alternative == "two-sided":
        d = abs(d)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    def func(d, alpha):
        return _power_nct(d * (1 / np.sqrt(1 / nx + 1 / ny)), nx + ny - 2, alpha, alternative)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power given d, n and alpha
        return func(d, alpha)
    elif d is None:
        # Compute achieved d given sample size, power and alpha level
        return _solve(lambda d: func(d, alpha) - power, *_D_BOUNDS[alternative])
    else:
        # Compute achieved alpha (significance) level given d, n and power
        return _solve(lambda alpha: func(d, alpha) - power, 1e-10, 1 - 1e-10)


def power_anova(eta_squared=None, k=None, n=None, power=None, alpha=0.05):
    """
    Evaluate power, sample size, effect size or significance level of a one-way balanced ANOVA.

    Parameters
    ----------
    eta_squared : float
        ANOVA effect size (eta-squared, :math:`\\eta^2`).
    k : int
        Number of groups
    n : int
        Sample size per group. Groups are assumed to be balanced (i.e. same sample size).
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level :math:`\\alpha` (type I error probability). The default is 0.05.

    Notes
    -----
    Exactly ONE of the parameters ``eta_squared``, ``k``, ``n``, ``power`` and ``alpha``
    must be passed as None, and that parameter is determined from the others.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to
    compute it.

    This function is a Python adaptation of the `pwr.anova.test` function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Statistical power is the likelihood that a study will detect an effect when there is an
    effect there to be detected. A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one. Statistical power is mainly affected by
    the effect size and the sample size.

    For one-way ANOVA, eta-squared is the same as partial eta-squared. It can be evaluated from the
    F-value (:math:`F^*`) and the degrees of freedom of the ANOVA (:math:`v_1, v_2`) using the
    following formula:

    .. math:: \\eta^2 = \\frac{v_1 F^*}{v_1 F^* + v_2}

    GPower uses the :math:`f` effect size instead of the :math:`\\eta^2`. The formula to convert
    from one to the other are given below:

    .. math:: f = \\sqrt{\\frac{\\eta^2}{1 - \\eta^2}}

    .. math:: \\eta^2 = \\frac{f^2}{1 + f^2}

    Using :math:`\\eta^2` and the total sample size :math:`N`, the non-centrality parameter is
    defined by:

    .. math:: \\delta = N * \\frac{\\eta^2}{1 - \\eta^2}

    Then the critical value of the non-central F-distribution is computed using the percentile
    point function of the F-distribution with:

    .. math:: q = 1 - \\alpha
    .. math:: v_1 = k - 1
    .. math:: v_2 = N - k

    where :math:`k` is the number of groups.

    Finally, the power of the ANOVA is calculated using the survival function of the non-central
    F-distribution using the previously computed critical value, non-centrality parameter, and
    degrees of freedom.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables (i.e.
    sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    Results have been tested against GPower and the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Examples
    --------
    1. Compute achieved power

    >>> from pingouin import power_anova
    >>> print("power: %.4f" % power_anova(eta_squared=0.1, k=3, n=20))
    power: 0.6082

    2. Compute required number of groups

    >>> print("k: %.4f" % power_anova(eta_squared=0.1, n=20, power=0.80))
    k: 6.0944

    3. Compute required sample size

    >>> print("n: %.4f" % power_anova(eta_squared=0.1, k=3, power=0.80))
    n: 29.9256

    4. Compute achieved effect size

    >>> print("eta-squared: %.4f" % power_anova(n=20, k=4, power=0.80, alpha=0.05))
    eta-squared: 0.1255

    5. Compute achieved alpha (significance)

    >>> print("alpha: %.4f" % power_anova(eta_squared=0.1, n=20, k=4, power=0.80, alpha=None))
    alpha: 0.1085
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [eta_squared, k, n, power, alpha]])
    if n_none != 1:
        err = "Exactly one of eta, k, n, power, and alpha must be None."
        raise ValueError(err)

    # Safety checks
    if eta_squared is not None:
        eta_squared = abs(eta_squared)
        f_sq = eta_squared / (1 - eta_squared)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    def func(f_sq, k, n, alpha):
        nc = (n * k) * f_sq
        dof1 = k - 1
        dof2 = (n * k) - k
        fcrit = stats.f.ppf(1 - alpha, dof1, dof2)
        return _ncf_sf(fcrit, dof1, dof2, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power
        return func(f_sq, k, n, alpha)
    elif k is None:
        # Compute required number of groups
        return _solve(lambda k: func(f_sq, k, n, alpha) - power, 2, 100)
    elif n is None:
        # Compute required sample size
        return _solve(lambda n: func(f_sq, k, n, alpha) - power, 2, 1e07)
    elif eta_squared is None:
        # Compute achieved eta-squared. Solve directly on eta-squared in (0, 1): solving on
        # f_sq in (0, 1) would restrict eta-squared to be lower than 0.5.
        return _solve(lambda eta: func(eta / (1 - eta), k, n, alpha) - power, 1e-10, 1 - 1e-10)
    else:
        # Compute achieved alpha
        return _solve(lambda alpha: func(f_sq, k, n, alpha) - power, 1e-10, 1 - 1e-10)


def power_rm_anova(eta_squared=None, m=None, n=None, power=None, alpha=0.05, corr=0.5, epsilon=1):
    """
    Evaluate power, sample size, effect size or significance level of a balanced one-way
    repeated measures ANOVA.

    Parameters
    ----------
    eta_squared : float
        ANOVA effect size (eta-squared, :math:`\\eta^2`).
    m : int
        Number of repeated measurements.
    n : int
        Sample size per measurement. All measurements must have the same sample size.
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level :math:`\\alpha` (type I error probability). The default is 0.05.
    corr : float
        Average correlation coefficient among repeated measurements. The default is :math:`r=0.5`.
    epsilon : float
        Epsilon adjustement factor for sphericity. This can be calculated using the
        :py:func:`pingouin.epsilon` function.

    Notes
    -----
    Exactly ONE of the parameters ``eta_squared``, ``m``, ``n``, ``power`` and ``alpha`` must be
    passed as None, and that parameter is determined from the others.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to
    compute it.

    Statistical power is the likelihood that a study will detect an effect when there is an effect
    there to be detected. A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one. Statistical power is mainly affected by
    the effect size and the sample size.

    GPower uses the :math:`f` effect size instead of the :math:`\\eta^2`. The formula to convert
    from one to the other are given below:

    .. math:: f = \\sqrt{\\frac{\\eta^2}{1 - \\eta^2}}

    .. math:: \\eta^2 = \\frac{f^2}{1 + f^2}

    Using :math:`\\eta^2`, the sample size :math:`N`, the number of repeated measurements
    :math:`m`, the epsilon correction factor :math:`\\epsilon` (see :py:func:`pingouin.epsilon`),
    and the average correlation between the repeated measures :math:`c`, one can then calculate the
    non-centrality parameter as follow:

    .. math:: \\delta = \\frac{f^2 * N * m * \\epsilon}{1 - c}

    Then the critical value of the non-central F-distribution is computed using the percentile
    point function of the F-distribution with:

    .. math:: q = 1 - \\alpha
    .. math:: v_1 = (m - 1) * \\epsilon
    .. math:: v_2 = (N - 1) * v_1

    Finally, the power of the ANOVA is calculated using the survival function of the non-central
    F-distribution using the previously computed critical value, non-centrality parameter,
    and degrees of freedom.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables
    (i.e. sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    Results have been tested against GPower and the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Examples
    --------
    1. Compute achieved power

    >>> from pingouin import power_rm_anova
    >>> print("power: %.4f" % power_rm_anova(eta_squared=0.1, m=3, n=20))
    power: 0.8913

    2. Compute required number of groups

    >>> print("m: %.4f" % power_rm_anova(eta_squared=0.1, n=20, power=0.90))
    m: 3.1347

    3. Compute required sample size

    >>> print("n: %.4f" % power_rm_anova(eta_squared=0.1, m=3, power=0.80))
    n: 15.9979

    4. Compute achieved effect size

    >>> print("eta-squared: %.4f" % power_rm_anova(n=20, m=4, power=0.80, alpha=0.05))
    eta-squared: 0.0680

    5. Compute achieved alpha (significance)

    >>> print("alpha: %.4f" % power_rm_anova(eta_squared=0.1, n=20, m=4, power=0.80, alpha=None))
    alpha: 0.0081

    Let's take a more concrete example. First, we'll load a repeated measures
    dataset in wide-format. Each row is an observation (e.g. a subject), and
    each column a successive repeated measurements (e.g t=0, t=1, ...).

    >>> import pingouin as pg
    >>> data = pg.read_dataset("rm_anova_wide")
    >>> data.head()
       Before  1 week  2 week  3 week
    0     4.3     5.3     4.8     6.3
    1     3.9     2.3     5.6     4.3
    2     4.5     2.6     4.1     NaN
    3     5.1     4.2     6.0     6.3
    4     3.8     3.6     4.8     6.8

    Note that this dataset has some missing values. We'll simply delete any row with one or more
    missing values, and then compute a repeated measures ANOVA:

    >>> data = data.dropna()
    >>> pg.rm_anova(data, effsize="n2").round(3)
       Source  ddof1  ddof2      F  p_unc     n2    eps
    0  Within      3     24  5.201  0.007  0.346  0.694

    The repeated measures ANOVA is significant at the 0.05 level. Now, we can
    easily compute the power of the ANOVA with the information in the ANOVA table:

    >>> # n is the sample size and m is the number of repeated measures
    >>> n, m = data.shape
    >>> round(pg.power_rm_anova(eta_squared=0.346, m=m, n=n, epsilon=0.694), 3)
    0.99

    Our ANOVA has a very high statistical power. However, to be even more accurate in our power
    calculation, we should also fill in the average correlation among repeated measurements.
    Since our dataframe is in wide-format (with each column being a successive measurement), this
    can be done by taking the mean of the superdiagonal of the correlation matrix, which is similar
    to manually calculating the correlation between each successive pairwise measurements and then
    taking the mean. Since correlation coefficients are not normally distributed, we use the
    *r-to-z* transform prior to averaging (:py:func:`numpy.arctanh`), and then the *z-to-r*
    transform (:py:func:`numpy.tanh`) to convert back to a correlation coefficient. This gives a
    more precise estimate of the mean.

    >>> import numpy as np
    >>> corr = np.diag(data.corr(), k=1)
    >>> avgcorr = np.tanh(np.arctanh(corr).mean())
    >>> round(avgcorr, 4)
    -0.1996

    In this example, we're using a fake dataset and the average correlation is negative. However,
    it will most likely be positive with real data. Let's now compute the final power of the
    repeated measures ANOVA:

    >>> round(pg.power_rm_anova(eta_squared=0.346, m=m, n=n, epsilon=0.694, corr=avgcorr), 3)
    0.771
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [eta_squared, m, n, power, alpha]])
    if n_none != 1:
        msg = "Exactly one of eta, m, n, power, and alpha must be None."
        raise ValueError(msg)

    # Safety checks
    assert 0 < epsilon <= 1, "epsilon must be between 0 and 1."
    assert -1 < corr < 1, "corr must be between -1 and 1."
    if eta_squared is not None:
        eta_squared = abs(eta_squared)
        f_sq = eta_squared / (1 - eta_squared)
    if alpha is not None:
        assert 0 < alpha <= 1, "alpha must be between 0 and 1."
    if power is not None:
        assert 0 < power <= 1, "power must be between 0 and 1."
    if n is not None:
        assert n > 1, "The sample size n must be > 1."
    if m is not None:
        assert m > 1, "The number of repeated measures m must be > 1."

    def func(f_sq, m, n, alpha):
        dof1 = (m - 1) * epsilon
        dof2 = (n - 1) * dof1
        nc = (f_sq * n * m * epsilon) / (1 - corr)
        fcrit = stats.f.ppf(1 - alpha, dof1, dof2)
        return _ncf_sf(fcrit, dof1, dof2, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power
        return func(f_sq, m, n, alpha)
    elif m is None:
        # Compute required number of repeated measures
        return _solve(lambda m: func(f_sq, m, n, alpha) - power, 2, 100)
    elif n is None:
        # Compute required sample size
        return _solve(lambda n: func(f_sq, m, n, alpha) - power, 2, 1e6)
    elif eta_squared is None:
        # Compute achieved eta-squared. Solve directly on eta-squared in (0, 1): solving on
        # f_sq in (0, 1) would restrict eta-squared to be lower than 0.5.
        return _solve(lambda eta: func(eta / (1 - eta), m, n, alpha) - power, 1e-10, 1 - 1e-10)
    else:
        # Compute achieved alpha
        return _solve(lambda alpha: func(f_sq, m, n, alpha) - power, 1e-10, 1 - 1e-10)


def power_corr(r=None, n=None, power=None, alpha=0.05, alternative="two-sided"):
    """
    Evaluate power, sample size, correlation coefficient or significance level of a correlation
    test.

    Parameters
    ----------
    r : float
        Correlation coefficient.
    n : int
        Number of observations (sample size).
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability). The default is 0.05.
    alternative : string
        Defines the alternative hypothesis, or tail of the correlation. Must be one of
        "two-sided" (default), "greater" or "less". Both "greater" and "less" return a one-sided
        p-value. "greater" tests against the alternative hypothesis that the correlation is
        positive (greater than zero), "less" tests against the hypothesis that the correlation is
        negative.

    Notes
    -----
    Exactly ONE of the parameters ``r``, ``n``, ``power`` and ``alpha`` must be passed as None,
    and that parameter is determined from the others.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to
    compute it.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables (i.e.
    sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    This function is a Python adaptation of the `pwr.r.test` function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Examples
    --------
    1. Compute achieved power given ``r``, ``n`` and ``alpha``

    >>> from pingouin import power_corr
    >>> print("power: %.4f" % power_corr(r=0.5, n=20))
    power: 0.6379

    2. Same but one-sided test

    >>> print("power: %.4f" % power_corr(r=0.5, n=20, alternative="greater"))
    power: 0.7510

    >>> print("power: %.4f" % power_corr(r=0.5, n=20, alternative="less"))
    power: 0.0000

    3. Compute required sample size given ``r``, ``power`` and ``alpha``

    >>> print("n: %.4f" % power_corr(r=0.5, power=0.80))
    n: 28.2484

    4. Compute achieved ``r`` given ``n``, ``power`` and ``alpha`` level

    >>> print("r: %.4f" % power_corr(n=20, power=0.80, alpha=0.05))
    r: 0.5822

    5. Compute achieved alpha level given ``r``, ``n`` and ``power``

    >>> print("alpha: %.4f" % power_corr(r=0.5, n=20, power=0.80, alpha=None))
    alpha: 0.1377
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [r, n, power, alpha]])
    if n_none != 1:
        raise ValueError("Exactly one of n, r, power, and alpha must be None")

    # Safety checks
    _check_alternative(alternative)

    if r is not None:
        assert -1 <= r <= 1
        if alternative == "two-sided":
            r = abs(r)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1
    if n is not None:
        if n <= 4:
            warnings.warn("Sample size is too small to estimate power (n <= 4). Returning NaN.")
            return np.nan

    # Define main function. A "less" test is a "greater" test on the negated correlation.
    tside = 2 if alternative == "two-sided" else 1

    def func(r, n, alpha):
        if alternative == "less":
            r = -r
        dof = n - 2
        ttt = stats.t.ppf(1 - alpha / tside, dof)
        rc = np.sqrt(ttt**2 / (ttt**2 + dof))
        zr = np.arctanh(r) + r / (2 * (n - 1))
        zrc = np.arctanh(rc)
        pwr = stats.norm.cdf((zr - zrc) * np.sqrt(n - 3))
        if alternative == "two-sided":
            pwr = pwr + stats.norm.cdf((-zr - zrc) * np.sqrt(n - 3))
        return pwr

    # Evaluate missing variable
    if power is None:
        # Compute achieved power given r, n and alpha
        return func(r, n, alpha)
    elif n is None:
        # Compute required sample size given r, power and alpha
        return _solve(lambda n: func(r, n, alpha) - power, 4 + 1e-10, 1e09)
    elif r is None:
        # Compute achieved r given sample size, power and alpha level
        lower = 1e-10 if alternative == "two-sided" else -1 + 1e-10
        return _solve(lambda r: func(r, n, alpha) - power, lower, 1 - 1e-10)
    else:
        # Compute achieved alpha (significance) level given r, n and power
        return _solve(lambda alpha: func(r, n, alpha) - power, 1e-10, 1 - 1e-10)


def power_chi2(dof, w=None, n=None, power=None, alpha=0.05):
    """
    Evaluate power, sample size, effect size or significance level of chi-squared tests.

    Parameters
    ----------
    dof : float
        Degree of freedom (depends on the chosen test).
    w : float
        Cohen's w effect size [1]_.
    n : int
        Total number of observations.
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability). The default is 0.05.

    Notes
    -----
    Exactly ONE of the parameters ``w``, ``n``, ``power`` and ``alpha`` must be passed as None,
    and that parameter is determined from the others. The degrees of freedom ``dof`` must always
    be specified.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to
    compute it.

    This function is a Python adaptation of the `pwr.chisq.test` function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Statistical power is the likelihood that a study will detect an effect when there is an effect
    there to be detected. A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one. Statistical power is mainly affected by
    the effect size and the sample size.

    The non-centrality parameter is defined by:

    .. math:: \\delta = N * w^2

    Then the critical value is computed using the percentile point function of the :math:`\\chi^2`
    distribution with the alpha level and degrees of freedom.

    Finally, the power of the chi-squared test is calculated using the survival function of the
    non-central :math:`\\chi^2` distribution using the previously computed critical value,
    non-centrality parameter, and the degrees of freedom of the test.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables (i.e.
    sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    Results have been tested against GPower and the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.).

    Examples
    --------
    1. Compute achieved power

    >>> from pingouin import power_chi2
    >>> print("power: %.4f" % power_chi2(dof=1, w=0.3, n=20))
    power: 0.2687

    2. Compute required sample size

    >>> print("n: %.4f" % power_chi2(dof=3, w=0.3, power=0.80))
    n: 121.1396

    3. Compute achieved effect size

    >>> print("w: %.4f" % power_chi2(dof=2, n=20, power=0.80, alpha=0.05))
    w: 0.6941

    4. Compute achieved alpha (significance)

    >>> print("alpha: %.4f" % power_chi2(dof=1, w=0.5, n=20, power=0.80, alpha=None))
    alpha: 0.1630
    """
    assert isinstance(dof, (int, float))
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [w, n, power, alpha]])
    if n_none != 1:
        err = "Exactly one of w, n, power, and alpha must be None."
        raise ValueError(err)

    # Safety checks
    if w is not None:
        w = abs(w)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    def func(w, n, alpha):
        k = stats.chi2.ppf(1 - alpha, dof)
        nc = n * w**2
        return stats.ncx2.sf(k, dof, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power
        return func(w, n, alpha)
    elif n is None:
        # Compute required sample size
        return _solve(lambda n: func(w, n, alpha) - power, 1, 1e07)
    elif w is None:
        # Compute achieved effect size
        return _solve(lambda w: func(w, n, alpha) - power, 1e-10, 100)
    else:
        # Compute achieved alpha
        return _solve(lambda alpha: func(w, n, alpha) - power, 1e-10, 1 - 1e-10)
