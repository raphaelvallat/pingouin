# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
from scipy.integrate import quad

__all__ = ["bayesfactor_ttest", "bayesfactor_pearson", "bayesfactor_binom"]


def _format_bf(bf, precision=3, trim='0'):
    """Format BF10 to floating point or scientific notation.
    """
    if bf >= 1e4 or bf <= 1e-4:
        out = np.format_float_scientific(bf, precision=precision, trim=trim)
    else:
        out = np.format_float_positional(bf, precision=precision, trim=trim)
    return out


def bayesfactor_ttest(t, nx, ny=None, paired=False, tail='two-sided', r=.707):
    """
    Bayes Factor of a T-test.

    Parameters
    ----------
    t : float
        T-value of the T-test
    nx : int
        Sample size of first group
    ny : int
        Sample size of second group (only needed in case of an independent
        two-sample T-test)
    paired : boolean
        Specify whether the two observations are related (i.e. repeated
        measures) or independent.
    tail : string
        Specify whether the test is 'one-sided' or 'two-sided'

        .. warning:: One-sided Bayes Factor (BF) are simply obtained by
            doubling the two-sided BF, which is not exactly the same behavior
            as R or JASP. Be extra careful when interpretating one-sided BF,
            and if you can, always double-check your results.
    r : float
        Cauchy scale factor. Smaller values of r (e.g. 0.5), may be appropriate
        when small effect sizes are expected a priori; larger values of r are
        appropriate when large effect sizes are expected (Rouder et al 2009).
        The default is 0.707.

    Returns
    -------
    bf : str
        Scaled Jeffrey-Zellner-Siow (JZS) Bayes Factor (BF10).
        The Bayes Factor quantifies the evidence in favour of the
        alternative hypothesis.

    Notes
    -----
    Adapted from a Matlab code found at
    https://github.com/anne-urai/Tools/tree/master/stats/BayesFactors

    If you would like to compute the Bayes Factor directly from the raw data
    instead of from the T-value, use the :py:func:`pingouin.ttest` function.

    The JZS Bayes Factor is approximated using the formula described
    in ref [1]_:

    .. math::

        BF_{10} = \\frac{\\int_{0}^{\\infty}(1 + Ngr^2)^{-1/2}
        (1 + \\frac{t^2}{v(1 + Ngr^2)})^{-(v+1) / 2}(2\\pi)^{-1/2}g^
        {-3/2}e^{-1/2g}}{(1 + \\frac{t^2}{v})^{-(v+1) / 2}}

    where **t** is the T-value, **v** the degrees of freedom, **N** the
    sample size and **r** the Cauchy scale factor (i.e. prior on effect size).

    References
    ----------
    .. [1] Rouder, J.N., Speckman, P.L., Sun, D., Morey, R.D., Iverson, G.,
       2009. Bayesian t tests for accepting and rejecting the null hypothesis.
       Psychon. Bull. Rev. 16, 225–237. https://doi.org/10.3758/PBR.16.2.225

    Examples
    --------
    1. Bayes Factor of an independent two-sample T-test

    >>> from pingouin import bayesfactor_ttest
    >>> bf = bayesfactor_ttest(3.5, 20, 20)
    >>> print("Bayes Factor: %s (two-sample independent)" % bf)
    Bayes Factor: 26.743 (two-sample independent)

    2. Bayes Factor of a paired two-sample T-test

    >>> bf = bayesfactor_ttest(3.5, 20, 20, paired=True)
    >>> print("Bayes Factor: %s (two-sample paired)" % bf)
    Bayes Factor: 17.185 (two-sample paired)

    3. Bayes Factor of an one-sided one-sample T-test

    >>> bf = bayesfactor_ttest(3.5, 20, tail='one-sided')
    >>> print("Bayes Factor: %s (one-sample)" % bf)
    Bayes Factor: 34.369 (one-sample)
    """
    one_sample = True if ny is None or ny == 1 else False

    # Function to be integrated
    def fun(g, t, n, r, df):
        return (1 + n * g * r**2)**(-.5) * (1 + t**2 / ((1 + n * g * r**2)
                                            * df))**(-(df + 1) / 2) *  \
               (2 * np.pi)**(-.5) * g**(-3. / 2) * np.exp(-1 / (2 * g))

    # Define n and degrees of freedom
    if one_sample or paired:
        n = nx
        df = n - 1
    else:
        n = nx * ny / (nx + ny)
        df = nx + ny - 2

    # JZS Bayes factor calculation: eq. 1 in Rouder et al. (2009)
    integr = quad(fun, 0, np.inf, args=(t, n, r, df))[0]
    bf10 = 1 / ((1 + t**2 / df)**(-(df + 1) / 2) / integr)

    # Tail
    bf10 = bf10 * (1 / 0.5) if tail == 'one-sided' else bf10

    return _format_bf(bf10)


def bayesfactor_pearson(r, n):
    """
    Bayes Factor of a Pearson correlation.

    Parameters
    ----------
    r : float
        Pearson correlation coefficient
    n : int
        Sample size

    Returns
    -------
    bf : str
        Bayes Factor (BF10).
        The Bayes Factor quantifies the evidence in favour of the alternative
        hypothesis.

    Notes
    -----
    Adapted from a Matlab code found at
    https://github.com/anne-urai/Tools/blob/master/stats/BayesFactors/corrbf.m

    If you would like to compute the Bayes Factor directly from the raw data
    instead of from the correlation coefficient, use the
    :py:func:`pingouin.corr` function.

    The JZS Bayes Factor is approximated using the formula described in
    ref [1]_:

    .. math::

        BF_{10} = \\frac{\\sqrt{n/2}}{\\gamma(1/2)}*
        \\int_{0}^{\\infty}e((n-2)/2)*
        log(1+g)+(-(n-1)/2)log(1+(1-r^2)*g)+(-3/2)log(g)-n/2g

    where **n** is the sample size and **r** is the Pearson correlation
    coefficient.

    .. warning:: This function does not (yet) support directional, one-sided
        test. It only returns the two-sided Bayes Factor.

    References
    ----------
    .. [1] Wetzels, R., Wagenmakers, E.-J., 2012. A default Bayesian
       hypothesis test for correlations and partial correlations.
       Psychon. Bull. Rev. 19, 1057–1064.
       https://doi.org/10.3758/s13423-012-0295-x

    Examples
    --------
    Bayes Factor of a Pearson correlation

    >>> from pingouin import bayesfactor_pearson
    >>> bf = bayesfactor_pearson(0.6, 20)
    >>> print("Bayes Factor: %s" % bf)
    Bayes Factor: 8.221
    """
    from scipy.special import gamma

    # Function to be integrated
    def fun(g, r, n):
        return np.exp(((n - 2) / 2) * np.log(1 + g) + (-(n - 1) / 2)
                      * np.log(1 + (1 - r**2) * g) + (-3 / 2)
                      * np.log(g) + - n / (2 * g))

    # JZS Bayes factor calculation
    integr = quad(fun, 0, np.inf, args=(r, n))[0]
    bf10 = np.sqrt((n / 2)) / gamma(1 / 2) * integr
    return _format_bf(bf10)


def bayesfactor_binom(k, n, p=.5):
    """
    Bayes factor of a binomial test with :math:`k` successes,
    :math:`n` trials and base probability :math:`p`.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    p : float
        Base probability of success (range from 0 to 1).

    Returns
    -------
    bf10 : float
        The Bayes Factor quantifies the evidence in favour of the
        alternative hypothesis, where the null hypothesis is that
        the random variable is binomially distributed with base probability
        :math:`p`.

    Notes
    -----
    Adapted from a Matlab code found at
    https://github.com/anne-urai/Tools/blob/master/stats/BayesFactors/binombf.m

    The Bayes Factor is approximated using the formula below:

    .. math::

        BF_{10} = \\frac{\\int_0^1 \\binom{n}{k}g^k(1-g)^{n-k}}
        {\\binom{n}{k} p^k (1-p)^{n-k}}

    References
    ----------
    .. [1] http://pcl.missouri.edu/bf-binomial

    .. [2] https://en.wikipedia.org/wiki/Bayes_factor

    Examples
    --------
    >>> import pingouin as pg
    >>> bf = pg.bayesfactor_binom(k=16, n=20, p=0.5)
    >>> print("Bayes Factor: %s" % bf)
    Bayes Factor: 10.306

    >>> bf = pg.bayesfactor_binom(k=10, n=20, p=0.5)
    >>> print("Bayes Factor: %s" % bf)
    Bayes Factor: 0.27

    With a different base probability

    >>> bf = pg.bayesfactor_binom(k=100, n=1000, p=0.1)
    >>> print("Bayes Factor: %s" % bf)
    Bayes Factor: 0.024
    """
    from scipy.stats import binom
    assert 0 < p < 1, 'p must be between 0 and 1.'
    assert isinstance(k, int), 'k must be int.'
    assert isinstance(n, int), 'n must be int.'
    assert k <= n, 'k (successes) cannot be higher than n (trials).'

    def fun(g, k, n):
        return binom.pmf(k, n, g)

    bf10 = quad(fun, 0, 1, args=(k, n))[0] / binom.pmf(k, n, p)
    return _format_bf(bf10)
