"""Bayesian functions."""
import warnings
import numpy as np
from scipy.integrate import quad
from math import pi, exp, log, lgamma

__all__ = ["bayesfactor_ttest", "bayesfactor_pearson", "bayesfactor_binom"]


def _format_bf(bf, precision=3, trim='0'):
    """Format BF10 to floating point or scientific notation.
    """
    if type(bf) == str:
        return bf
    if bf >= 1e4 or bf <= 1e-4:
        out = np.format_float_scientific(bf, precision=precision, trim=trim)
    else:
        out = np.format_float_positional(bf, precision=precision, trim=trim)
    return out


def bayesfactor_ttest(t, nx, ny=None, paired=False, alternative='two-sided', r=.707):
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
    alternative : string
        Defines the alternative hypothesis, or tail of the test. Must be one of
        "two-sided" (default), "greater" or "less".

        .. warning:: One-sided Bayes Factor (BF) are simply obtained by
            doubling the two-sided BF, which is not the same behavior
            as R or JASP. Be extra careful when interpretating one-sided BF,
            and if you can, always double-check your results.
    r : float
        Cauchy scale factor. Smaller values of ``r`` (e.g. 0.5), may be
        appropriate when small effect sizes are expected a priori; larger
        values of ``r`` are appropriate when large effect sizes are
        expected (Rouder et al 2009). The default is
        :math:`\\sqrt{2} / 2 \\approx 0.707`.

    Returns
    -------
    bf : float
        Scaled Jeffrey-Zellner-Siow (JZS) Bayes Factor (BF10).
        The Bayes Factor quantifies the evidence in favour of the
        alternative hypothesis.

    See also
    --------
    ttest : T-test
    pairwise_ttest : Pairwise T-tests
    bayesfactor_pearson : Bayes Factor of a correlation
    bayesfactor_binom : Bayes Factor of a binomial test

    Notes
    -----
    Adapted from a Matlab code found at
    https://github.com/anne-urai/Tools/tree/master/stats/BayesFactors

    If you would like to compute the Bayes Factor directly from the raw data
    instead of from the T-value, use the :py:func:`pingouin.ttest` function.

    The JZS Bayes Factor is approximated using the formula described
    in ref [1]_:

    .. math::

        \\text{BF}_{10} = \\frac{\\int_{0}^{\\infty}(1 + Ngr^2)^{-1/2}
        (1 + \\frac{t^2}{v(1 + Ngr^2)})^{-(v+1) / 2}(2\\pi)^{-1/2}g^
        {-3/2}e^{-1/2g}}{(1 + \\frac{t^2}{v})^{-(v+1) / 2}}

    where :math:`t` is the T-value, :math:`v` the degrees of freedom,
    :math:`N` the sample size, :math:`r` the Cauchy scale factor
    (= prior on effect size) and :math:`g` is is an auxiliary variable
    that is integrated out numerically.

    Results have been validated against JASP and the BayesFactor R package.

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
    >>> print("Bayes Factor: %.3f (two-sample independent)" % bf)
    Bayes Factor: 26.743 (two-sample independent)

    2. Bayes Factor of a paired two-sample T-test

    >>> bf = bayesfactor_ttest(3.5, 20, 20, paired=True)
    >>> print("Bayes Factor: %.3f (two-sample paired)" % bf)
    Bayes Factor: 17.185 (two-sample paired)

    3. Now specifying the direction of the test

    >>> tval = -3.5
    >>> bf_greater = bayesfactor_ttest(tval, 20, alternative='greater')
    >>> bf_less = bayesfactor_ttest(tval, 20, alternative='less')
    >>> print("BF10-greater: %.3f | BF10-less: %.3f" % (bf_greater, bf_less))
    BF10-greater: 0.029 | BF10-less: 34.369
    """
    # Check tail
    assert alternative in ['two-sided', 'greater', 'less'], (
        "Alternative must be one of 'two-sided' (default), 'greater' or 'less'.")
    one_sample = True if ny is None or ny == 1 else False

    # Check T-value
    assert isinstance(t, (int, float)), 'The T-value must be a int or a float.'
    if not np.isfinite(t):
        return np.nan

    # Function to be integrated
    def fun(g, t, n, r, df):
        return (1 + n * g * r**2)**(-.5) * (1 + t**2 / ((1 + n * g * r**2)
                                            * df))**(-(df + 1) / 2) *  \
               (2 * pi)**(-.5) * g**(-3. / 2) * exp(-1 / (2 * g))

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
    tail_binary = 'two-sided' if alternative == 'two-sided' else 'one-sided'
    bf10 = bf10 * (1 / 0.5) if tail_binary == 'one-sided' else bf10
    # Now check the direction of the test
    if ((alternative == 'greater' and t < 0) or (alternative == 'less' and t > 0)) and bf10 > 1:
        bf10 = 1 / bf10
    return bf10


def bayesfactor_pearson(r, n, alternative='two-sided', method='ly', kappa=1.):
    """
    Bayes Factor of a Pearson correlation.

    Parameters
    ----------
    r : float
        Pearson correlation coefficient.
    n : int
        Sample size.
    alternative : string
        Defines the alternative hypothesis, or tail of the correlation. Must be one of
        "two-sided" (default), "greater" or "less". Both "greater" and "less" return a one-sided
        p-value. "greater" tests against the alternative hypothesis that the correlation is
        positive (greater than zero), "less" tests against the hypothesis that the correlation is
        negative.
    method : str
        Method to compute the Bayes Factor. Can be "ly" (default) or
        "wetzels". The former has an exact analytical solution, while the
        latter requires integral solving (and is therefore slower). "wetzels"
        was the default in Pingouin <= 0.2.5. See Notes for details.
    kappa : float
        Kappa factor. This is sometimes called the *rscale* parameter, and
        is only used when ``method`` is "ly".

    Returns
    -------
    bf : float
        Bayes Factor (BF10).
        The Bayes Factor quantifies the evidence in favour of the alternative
        hypothesis.

    See also
    --------
    corr : (Robust) correlation between two variables
    pairwise_corr : Pairwise correlation between columns of a pandas DataFrame
    bayesfactor_ttest : Bayes Factor of a T-test
    bayesfactor_binom : Bayes Factor of a binomial test

    Notes
    -----
    To compute the Bayes Factor directly from the raw data, use the
    :py:func:`pingouin.corr` function.

    The two-sided **Wetzels Bayes Factor** (also called *JZS Bayes Factor*)
    is calculated using the equation 13 and associated R code of [1]_:

    .. math::

        \\text{BF}_{10}(n, r) = \\frac{\\sqrt{n/2}}{\\gamma(1/2)}*
        \\int_{0}^{\\infty}e((n-2)/2)*
        log(1+g)+(-(n-1)/2)log(1+(1-r^2)*g)+(-3/2)log(g)-n/2g

    where :math:`n` is the sample size, :math:`r` is the Pearson correlation
    coefficient and :math:`g` is is an auxiliary variable that is integrated
    out numerically. Since the Wetzels Bayes Factor requires solving an
    integral, it is slower than the analytical solution described below.

    The two-sided **Ly Bayes Factor** (also called *Jeffreys
    exact Bayes Factor*) is calculated using equation 25 of [2]_:

    .. math::

        \\text{BF}_{10;k}(n, r) = \\frac{2^{\\frac{k-2}{k}}\\sqrt{\\pi}}
        {\\beta(\\frac{1}{k}, \\frac{1}{k})} \\cdot
        \\frac{\\Gamma(\\frac{2+k(n-1)}{2k})}{\\Gamma(\\frac{2+nk}{2k})}
        \\cdot 2F_1(\\frac{n-1}{2}, \\frac{n-1}{2}, \\frac{2+nk}{2k}, r^2)

    The one-sided version is described in eq. 27 and 28 of Ly et al, 2016.
    Please take note that the one-sided test requires the
    `mpmath <http://mpmath.org/>`_ package.

    Results have been validated against JASP and the BayesFactor R package.

    References
    ----------
    .. [1] Ly, A., Verhagen, J. & Wagenmakers, E.-J. Harold Jeffreys’s default
       Bayes factor hypothesis tests: Explanation, extension, and
       application in psychology. J. Math. Psychol. 72, 19–32 (2016).

    .. [2] Wetzels, R. & Wagenmakers, E.-J. A default Bayesian hypothesis test
       for correlations and partial correlations. Psychon. Bull. Rev. 19,
       1057–1064 (2012).

    Examples
    --------
    Bayes Factor of a Pearson correlation

    >>> from pingouin import bayesfactor_pearson
    >>> r, n = 0.6, 20
    >>> bf = bayesfactor_pearson(r, n)
    >>> print("Bayes Factor: %.3f" % bf)
    Bayes Factor: 10.634

    Compare to Wetzels method:

    >>> bf = bayesfactor_pearson(r, n, method='wetzels')
    >>> print("Bayes Factor: %.3f" % bf)
    Bayes Factor: 8.221

    One-sided test

    >>> bf10pos = bayesfactor_pearson(r, n, alternative='greater')
    >>> bf10neg = bayesfactor_pearson(r, n, alternative='less')
    >>> print("BF-pos: %.3f, BF-neg: %.3f" % (bf10pos, bf10neg))
    BF-pos: 21.185, BF-neg: 0.082
    """
    from scipy.special import gamma, betaln, hyp2f1
    assert method.lower() in ['ly', 'wetzels'], 'Method not recognized.'
    assert alternative in ['two-sided', 'greater', 'less'], (
        "Alternative must be one of 'two-sided' (default), 'greater' or 'less'.")

    # Wrong input
    if not np.isfinite(r) or n < 2:
        return np.nan
    assert -1 <= r <= 1, 'r must be between -1 and 1.'

    if alternative != 'two-sided' and method.lower() == 'wetzels':
        warnings.warn("One-sided Bayes Factor are not supported by the "
                      "Wetzels's method. Switching to method='ly'.")
        method = 'ly'

    if method.lower() == 'wetzels':
        # Wetzels & Wagenmakers, 2012. Integral solving

        def fun(g, r, n):
            return exp(((n - 2) / 2) * log(1 + g) + (-(n - 1) / 2)
                       * log(1 + (1 - r**2) * g) + (-3 / 2)
                       * log(g) + - n / (2 * g))

        integr = quad(fun, 0, np.inf, args=(r, n))[0]
        bf10 = np.sqrt((n / 2)) / gamma(1 / 2) * integr

    else:
        # Ly et al, 2016. Analytical solution.
        k = kappa
        lbeta = betaln(1 / k, 1 / k)
        log_hyperterm = log(hyp2f1(((n - 1) / 2), ((n - 1) / 2),
                                   ((n + 2 / k) / 2), r**2))
        bf10 = exp((1 - 2 / k) * log(2) + 0.5 * log(pi) - lbeta
                   + lgamma((n + 2 / k - 1) / 2) - lgamma((n + 2 / k) / 2) +
                   log_hyperterm)

        if alternative != 'two-sided':
            # Directional test.
            # We need mpmath for the generalized hypergeometric function
            from .utils import _is_mpmath_installed
            _is_mpmath_installed(raise_error=True)
            from mpmath import hyp3f2
            hyper_term = float(hyp3f2(1, n / 2, n / 2, 3 / 2, (2 + k * (n + 1)) / (2 * k), r**2))
            log_term = 2 * (lgamma(n / 2) - lgamma((n - 1) / 2)) - lbeta
            C = 2**((3 * k - 2) / k) * k * r / (2 + (n - 1) * k) * exp(log_term) * hyper_term

            bf10neg = bf10 - C
            bf10pos = 2 * bf10 - bf10neg
            if alternative == 'greater':
                # We expect the correlation to be positive
                bf10 = bf10pos
            else:
                # We expect the correlation to be negative
                bf10 = bf10neg

    return bf10


def bayesfactor_binom(k, n, p=.5, a=1, b=1):
    """
    Bayes factor of a binomial test with :math:`k` successes,
    :math:`n` trials and base probability :math:`p`.  This means that
    the null hypothesis is that the probability is :math:`p`.  It is
    compared against the alternative hypothesis that :math:`p` is from
    the Beta distribution with parameters :math:`(a, b)`.  By default,
    both :math:`a` and :math:`b` are 1, making the alternative
    hypothesis equivalent to the uniform distribution, i.e., we are
    completely uninformed about :math:`p`.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    p : float
        Base probability of success (range from 0 to 1).
    a : float
        The "a" parameter of the Beta distribution.
    b : float
        The "b" parameter of the Beta distribution.

    Returns
    -------
    bf10 : float
        The Bayes Factor quantifies the evidence in favour of the
        alternative hypothesis, where the null hypothesis is that
        the random variable is binomially distributed with base probability
        :math:`p`.

    See also
    --------
    bayesfactor_pearson : Bayes Factor of a correlation
    bayesfactor_ttest : Bayes Factor of a T-test

    Notes
    -----
    Adapted from a Matlab code found at
    https://github.com/anne-urai/Tools/blob/master/stats/BayesFactors/binombf.m

    The Bayes Factor is given by the formula below:

    .. math::

        BF_{10} = \\frac{\\int_0^1 \\binom{n}{k}g^k(1-g)^{n-k}}
        {\\binom{n}{k} p^k (1-p)^{n-k}}

    References
    ----------
    * http://pcl.missouri.edu/bf-binomial

    * https://en.wikipedia.org/wiki/Bayes_factor

    Examples
    --------
    We want to determine if a coin if fair. After tossing the coin 200 times
    in a row, we report 115 heads (hereafter referred to as "successes") and 85
    tails ("failures"). The Bayes Factor can be easily computed using Pingouin:

    >>> import pingouin as pg
    >>> bf = float(pg.bayesfactor_binom(k=115, n=200, p=0.5))
    >>> # Note that Pingouin returns the BF-alt by default.
    >>> # BF-null is simply 1 / BF-alt
    >>> print("BF-null: %.3f, BF-alt: %.3f" % (1 / bf, bf))
    BF-null: 1.197, BF-alt: 0.835

    Since the Bayes Factor of the null hypothesis ("the coin is fair") is
    higher than the Bayes Factor of the alternative hypothesis
    ("the coin is not fair"), we can conclude that there is more evidence to
    support the fact that the coin is indeed fair. However, the strength of the
    evidence in favor of the null hypothesis (1.197) is "barely worth
    mentionning" according to Jeffreys's rule of thumb.

    Interestingly, a frequentist alternative to this test would give very
    different results. It can be performed using the
    :py:func:`scipy.stats.binom_test` function:

    >>> from scipy.stats import binomtest
    >>> result = binomtest(k=115, n=200, p=0.5)
    >>> round(result.pvalue, 5)
    0.04004

    The binomial test rejects the null hypothesis that the coin is fair at the
    5% significance level (p=0.04). Thus, whereas a frequentist hypothesis test
    would yield significant results at the 5% significance level, the Bayes
    factor indicates preference of the null hypothesis to the alternative
    hypothesis that we know nothing about p.

    We can use a more informed alternative hypothesis too, if desirable.  E.g.,
    the original test using Beta(5, 4) as the alternative hypothesis:

    >>> bf = pg.bayesfactor_binom(k=115, n=200, p=0.5, a=5, b=4)
    >>> print("Bayes Factor: %.3f" % bf)
    Bayes Factor: 1.930

    Using a different base probability of successes:

    >>> bf = pg.bayesfactor_binom(k=100, n=1000, p=0.1)
    >>> print("Bayes Factor: %.3f" % bf)
    Bayes Factor: 0.024
    """
    from scipy.stats import beta, binom
    assert 0 < p < 1, 'p must be between 0 and 1.'
    assert isinstance(k, int), 'k must be int.'
    assert isinstance(n, int), 'n must be int.'
    assert k <= n, 'k (successes) cannot be higher than n (trials).'
    assert a > 0, 'a must be positive.'
    assert b > 0, 'b must be positive.'

    def fun(g):
        return beta.pdf(g, a, b) * binom.pmf(k, n, g)

    bf10 = quad(fun, 0, 1)[0] / binom.pmf(k, n, p)
    return bf10
