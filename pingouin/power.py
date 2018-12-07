# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
from scipy import stats
from scipy.optimize import brenth

__all__ = ["power_ttest", "power_ttest2n", "power_anova", "power_corr"]


def power_ttest(d=None, n=None, power=None, alpha=0.05, contrast='two-samples',
                tail='two-sided'):
    """
    Evaluate power, sample size, effect size or
    significance level of a one-sample T-test, a paired T-test or an
    independent two-samples T-test with equal sample sizes.

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
        Can be "one-sample", "two-samples" or "paired".
        Note that "one-sample" and "paired" have the same behavior.
    tail : str
        Indicates whether the test is "two-sided" or "one-sided".

    Notes
    -----
    Exactly ONE of the parameters `d`, `n`, `power` and `alpha` must
    be passed as None, and that parameter is determined from the others.

    For a paired T-test, the sample size `n` corresponds to the number of
    pairs. For an independent two-sample T-test with equal sample sizes, `n`
    corresponds to the sample size of each group (i.e. number of observations
    in one group). If the sample sizes are unequal, please use the
    :py:func:`power_ttest2n` function instead.

    Notice that `alpha` has a default value of 0.05 so None must be explicitly
    passed if you want to compute it.

    This function is a mere Python translation of the original `pwr.t.test`
    function implemented in the `pwr` package. All credit goes to the author,
    Stephane Champely.

    Statistical power is the likelihood that a study will
    detect an effect when there is an effect there to be detected.
    A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one.
    Statistical power is mainly affected by the effect size and the sample
    size.

    The first step is to use the Cohen's d to calculate the non-centrality
    parameter :math:`\delta` and degrees of freedom :math:`v`.
    In case of paired groups, this is:

    .. math:: \delta = d * \sqrt n
    .. math:: v = n - 1

    and in case of independent groups with equal sample sizes:

    .. math:: \delta = d * \sqrt{\dfrac{n}{2}}
    .. math:: v = (n - 1) * 2

    where :math:`d` is the Cohen d and :math:`n` the sample size.

    The critical value is then found using the percent point function of the T
    distribution with :math:`q = 1 - alpha` and :math:`v`
    degrees of freedom.

    Finally, the power of the test is given by the survival function of the
    non-central distribution using the previously calculated critical value,
    degrees of freedom and non-centrality parameter.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other
    variables (i.e. sample size, effect size, or significance level). If the
    solving fails, a nan value is returned.

    Results have been tested against GPower and the R pwr package.

    References
    ----------

    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Hillsdale,NJ: Lawrence Erlbaum.

    .. [2] https://cran.r-project.org/web/packages/pwr/pwr.pdf

    Examples
    --------
    1. Compute achieved power of a one-sample T-test given `d`, `n` and `alpha`

        >>> from pingouin import power_ttest
        >>> print('power: %.4f' % power_ttest(d=0.5, n=20,
        ...                                   contrast='one-sample'))
            power: 0.5645

    2. Compute required sample size given `d`, `power` and `alpha`

        >>> print('n: %.4f' % power_ttest(d=0.5, power=0.80,
        ...                               tail='one-sided'))
            n: 50.1508

    3. Compute achieved `d` given `n`, `power` and `alpha` level

        >>> print('d: %.4f' % power_ttest(n=20, power=0.80, alpha=0.05,
        ...                               contrast='paired'))
            d: 0.6604

    4. Compute achieved alpha (significance) level given `d`, `n` and `power`

        >>> print('alpha: %.4f' % power_ttest(d=0.5, n=20, power=0.80,
        ...                                   alpha=None))
            alpha: 0.4630
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [d, n, power, alpha]])
    if n_none != 1:
        raise ValueError('Exactly one of n, d, power, and alpha must be None.')

    # Safety checks
    assert contrast.lower() in ['one-sample', 'paired', 'two-samples']
    if d is not None:
        d = abs(d)
    if alpha is not None:
        assert 0 < alpha <= 1
        # For simplicity
        alpha = alpha / 2 if tail == 'two-sided' else alpha
    if power is not None:
        assert 0 < power <= 1

    if contrast.lower() in ['one-sample', 'paired']:

        # One-sample or paired T-test

        def func(d, n, power, alpha):
            nc = d * np.sqrt(n)
            dof = n - 1
            tcrit = stats.t.ppf(1 - alpha, dof)
            return stats.nct.sf(tcrit, dof, nc)

    else:

        # Independent two-sample T-test

        def func(d, n, power, alpha):
            nc = d * np.sqrt(n / 2)
            dof = 2 * n - 2
            tcrit = stats.t.ppf(1 - alpha, dof)
            return stats.nct.sf(tcrit, dof, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power given d, n and alpha
        return func(d, n, power=None, alpha=alpha)

    elif n is None:
        # Compute required sample size given d, power and alpha

        def _eval_n(n, d, power, alpha):
            return func(d, n, power, alpha) - power

        try:
            return brenth(_eval_n, 2 + 1e-10, 1e+07, args=(d, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    elif d is None:
        # Compute achieved d given sample size, power and alpha level

        def _eval_d(d, n, power, alpha):
            return func(d, n, power, alpha) - power

        try:
            return brenth(_eval_d, 1e-07, 10, args=(n, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    else:
        # Compute achieved alpha (significance) level given d, n and power

        def _eval_alpha(alpha, d, n, power):
            return func(d, n, power, alpha) - power

        try:
            alpha = brenth(_eval_alpha, 1e-10, 1 - 1e-10, args=(d, n, power))
            if tail == 'one-sided':
                return alpha
            else:
                return 2 * alpha
        except ValueError:  # pragma: no cover
            return np.nan


def power_ttest2n(nx, ny, d=None, power=None, alpha=0.05, tail='two-sided'):
    """
    Evaluate power, effect size or  significance level of an independent
    two-samples T-test with unequal sample sizes.

    Parameters
    ----------
    nx, ny : int
        Sample sizes. Must be specified.
        If the sample sizes are equal, you should use the
        :py:func:`power_ttest` function instead.
    d : float
        Cohen d effect size
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability).
        The default is 0.05.
    tail : str
        Indicates whether the test is "two-sided" or "one-sided".

    Notes
    -----
    Exactly ONE of the parameters `d`, `power` and `alpha` must
    be passed as None, and that parameter is determined from the others.

    Notice that `alpha` has a default value of 0.05 so None must be explicitly
    passed if you want to compute it.

    This function is a mere Python translation of the original `pwr.t2n.test`
    function implemented in the `pwr` package. All credit goes to the author,
    Stephane Champely.

    Statistical power is the likelihood that a study will
    detect an effect when there is an effect there to be detected.
    A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one.
    Statistical power is mainly affected by the effect size and the sample
    size.

    The first step is to use the Cohen's d to calculate the non-centrality
    parameter :math:`\delta` and degrees of freedom :math:`v`.
    In case of two independent groups with unequal sample sizes, this is:

    .. math:: \delta = d * \sqrt{\dfrac{n_i * n_j}{n_i + n_j}}
    .. math:: v = n_i + n_j - 2

    where :math:`d` is the Cohen d, :math:`n` the sample size,
    :math:`n_i` the sample size of the first group and
    :math:`n_j` the sample size of the second group,

    The critical value is then found using the percent point function of the T
    distribution with :math:`q = 1 - alpha` and :math:`v`
    degrees of freedom.

    Finally, the power of the test is given by the survival function of the
    non-central distribution using the previously calculated critical value,
    degrees of freedom and non-centrality parameter.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other
    variables (i.e. sample size, effect size, or significance level). If the
    solving fails, a nan value is returned.

    Results have been tested against GPower and the R pwr package.

    Examples
    --------
    1. Compute achieved power of a T-test given `d`, `n` and `alpha`

        >>> from pingouin import power_ttest2n
        >>> print('power: %.4f' % power_ttest2n(nx=20, ny=15, d=0.5,
        ...                                     tail='one-sided'))
            power: 0.4164

    3. Compute achieved `d` given `n`, `power` and `alpha` level

        >>> print('d: %.4f' % power_ttest2n(nx=20, ny=15, power=0.80,
        ...                                 alpha=0.05))
            d: 0.9859

    4. Compute achieved alpha (significance) level given `d`, `n` and `power`

        >>> print('alpha: %.4f' % power_ttest2n(nx=20, ny=15, d=0.5,
        ...                                     power=0.80, alpha=None))
            alpha: 0.5366
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [d, power, alpha]])
    if n_none != 1:
        raise ValueError('Exactly one of d, power, and alpha must be None')

    # Safety checks
    if d is not None:
        d = abs(d)
    if alpha is not None:
        assert 0 < alpha <= 1
        # For simplicity
        alpha = alpha / 2 if tail == 'two-sided' else alpha
    if power is not None:
        assert 0 < power <= 1

    # Independent two-sample T-test
    def func(d, nx, ny, power, alpha):
        nc = d * np.sqrt((nx * ny) / (nx + ny))
        dof = nx + ny - 2
        tcrit = stats.t.ppf(1 - alpha, dof)
        return stats.nct.sf(tcrit, dof, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power given d, n and alpha
        return func(d, nx, ny, power=None, alpha=alpha)

    elif d is None:
        # Compute achieved d given sample size, power and alpha level

        def _eval_d(d, nx, ny, power, alpha):
            return func(d, nx, ny, power, alpha) - power

        try:
            return brenth(_eval_d, 1e-07, 10, args=(nx, ny, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    else:
        # Compute achieved alpha (significance) level given d, n and power

        def _eval_alpha(alpha, d, nx, ny, power):
            return func(d, nx, ny, power, alpha) - power

        try:
            alpha = brenth(_eval_alpha, 1e-10, 1 - 1e-10, args=(d, nx, ny,
                                                                power))
            if tail == 'one-sided':
                return alpha
            else:
                return 2 * alpha
        except ValueError:  # pragma: no cover
            return np.nan


def power_anova(eta=None, k=None, n=None, power=None, alpha=0.05):
    """
    Evaluate power, sample size, effect size or
    significance level of a one-way balanced ANOVA.

    Parameters
    ----------
    eta : float
        ANOVA effect size (eta-square == :math:`\eta^2`).
    k : int
        Number of groups
    n : int
        Sample size per group. Groups are assumed to be balanced
        (i.e. same sample size).
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability).
        The default is 0.05.

    Notes
    -----
    Exactly ONE of the parameters `eta`, `k`, `n`, `power` and `alpha` must
    be passed as None, and that parameter is determined from the others.

    Notice that `alpha` has a default value of 0.05 so None must be explicitly
    passed if you want to compute it.

    This function is a mere Python translation of the original `pwr.anova.test`
    function implemented in the `pwr` package. All credit goes to the author,
    Stephane Champely.

    Statistical power is the likelihood that a study will
    detect an effect when there is an effect there to be detected.
    A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one.
    Statistical power is mainly affected by the effect size and the sample
    size.

    For one-way ANOVA, eta-square is the same as partial eta-square. It can be
    evaluated from the f-value and degrees of freedom of the ANOVA using
    the following formula:

    .. math::
        \eta^2 = \dfrac{F^* * \mathtt{df_1}}{F^* * \mathtt{df_1} +
        \mathtt{df_2}}

    Using :math:`\eta^2` and the total sample size :math:`N`, the
    non-centrality parameter is defined by:

    .. math:: \delta = N * \dfrac{\eta^2}{1 - \eta^2}

    Then the critical value of the non-central F-distribution is computed using
    the percentile point function of the F-distribution with:

    .. math:: q = 1 - alpha
    .. math:: \mathtt{df_1} = k - 1
    .. math:: \mathtt{df_2} = N - k

    where :math:`k` is the number of groups.

    Finally, the power of the ANOVA is calculated using the survival function
    of the non-central F-distribution using the previously computed critical
    value, non-centrality parameter, and degrees of freedom.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other
    variables (i.e. sample size, effect size, or significance level). If the
    solving fails, a nan value is returned.

    Results have been tested against GPower and the R pwr package.

    References
    ----------

    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Hillsdale,NJ: Lawrence Erlbaum.

    .. [2] https://cran.r-project.org/web/packages/pwr/pwr.pdf

    Examples
    --------
    1. Compute achieved power

        >>> from pingouin import power_anova
        >>> print('power: %.4f' % power_anova(eta=0.1, k=3, n=20))
            power: 0.6804

    2. Compute required number of groups

        >>> print('k: %.4f' % power_anova(eta=0.1, n=20, power=0.80))
            k: 6.0944

    3. Compute required sample size

        >>> print('n: %.4f' % power_anova(eta=0.1, k=3, power=0.80))
            n: 25.5289

    4. Compute achieved effect size

        >>> print('eta: %.4f' % power_anova(n=20, power=0.80, alpha=0.05))
            eta: 0.1255

    5. Compute achieved alpha (significance)

        >>> print('alpha: %.4f' % power_anova(eta=0.1, n=20, power=0.80,
        ...                                   alpha=None))
            alpha: 0.1085
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [eta, k, n, power, alpha]])
    if n_none != 1:
        err = 'Exactly one of eta, k, n, power, and alpha must be None.'
        raise ValueError(err)

    # Safety checks
    if eta is not None:
        eta = abs(eta)
        f_sq = eta / (1 - eta)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    def func(f_sq, k, n, power, alpha):
        nc = (n * k) * f_sq
        dof1 = k - 1
        dof2 = (n * k) - k
        fcrit = stats.f.ppf(1 - alpha, dof1, dof2)
        return stats.ncf.sf(fcrit, dof1, dof2, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power
        return func(f_sq, k, n, power, alpha)

    elif k is None:
        # Compute required number of groups

        def _eval_k(k, eta, n, power, alpha):
            return func(f_sq, k, n, power, alpha) - power

        try:
            return brenth(_eval_k, 2, 100, args=(f_sq, n, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    elif n is None:
        # Compute required sample size

        def _eval_n(n, f_sq, k, power, alpha):
            return func(f_sq, k, n, power, alpha) - power

        try:
            return brenth(_eval_n, 2, 1e+07, args=(f_sq, k, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    elif eta is None:
        # Compute achieved eta

        def _eval_eta(f_sq, k, n, power, alpha):
            return func(f_sq, k, n, power, alpha) - power

        try:
            f_sq = brenth(_eval_eta, 1e-10, 1 - 1e-10, args=(k, n, power,
                                                             alpha))
            return f_sq / (f_sq + 1)  # Return eta-square
        except ValueError:  # pragma: no cover
            return np.nan

    else:
        # Compute achieved alpha

        def _eval_alpha(alpha, f_sq, k, n, power):
            return func(f_sq, k, n, power, alpha) - power

        try:
            return brenth(_eval_alpha, 1e-10, 1 - 1e-10, args=(f_sq, k, n,
                                                               power))
        except ValueError:  # pragma: no cover
            return np.nan


def power_corr(r=None, n=None, power=None, alpha=0.05, tail='two-sided'):
    """
    Evaluate power, sample size, correlation coefficient or
    significance level of a correlation test.

    Parameters
    ----------
    r : float
        Correlation coefficient.
    n : int
        Number of observations (sample size).
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability).
        The default is 0.05.
    tail : str
        Indicates whether the test is "two-sided" or "one-sided".

    Notes
    -----
    Exactly ONE of the parameters `r`, `n`, `power` and `alpha` must
    be passed as None, and that parameter is determined from the others.

    Notice that `alpha` has a default value of 0.05 so None must be explicitly
    passed if you want to compute it.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other
    variables (i.e. sample size, effect size, or significance level). If the
    solving fails, a nan value is returned.

    This function is a mere Python translation of the original `pwr.r.test`
    function implemented in the `pwr` R package.
    All credit goes to the author, Stephane Champely.

    References
    ----------

    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Hillsdale,NJ: Lawrence Erlbaum.

    .. [2] https://cran.r-project.org/web/packages/pwr/pwr.pdf

    Examples
    --------
    1. Compute achieved power given `r`, `n` and `alpha`

        >>> from pingouin import power_corr
        >>> print('power: %.4f' % power_corr(r=0.5, n=20))
            power: 0.6379

    2. Compute required sample size given `r`, `power` and `alpha`

        >>> print('n: %.4f' % power_corr(r=0.5, power=0.80,
        ...                                tail='one-sided'))
            n: 22.6091

    3. Compute achieved `r` given `n`, `power` and `alpha` level

        >>> print('r: %.4f' % power_corr(n=20, power=0.80, alpha=0.05))
            r: 0.5822

    4. Compute achieved alpha (significance) level given `r`, `n` and `power`

        >>> print('alpha: %.4f' % power_corr(r=0.5, n=20, power=0.80,
        ...                                    alpha=None))
            alpha: 0.1377
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [r, n, power, alpha]])
    if n_none != 1:
        raise ValueError('Exactly one of n, r, power, and alpha must be None')

    # Safety checks
    if r is not None:
        assert -1 <= r <= 1
        r = abs(r)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1
    if n is not None:
        assert n > 4

    # Define main function
    if tail == 'two-sided':

        def func(r, n, power, alpha):
            dof = n - 2
            ttt = stats.t.ppf(1 - alpha / 2, dof)
            rc = np.sqrt(ttt**2 / (ttt**2 + dof))
            zr = np.arctanh(r) + r / (2 * (n - 1))
            zrc = np.arctanh(rc)
            power = stats.norm.cdf((zr - zrc) * np.sqrt(n - 3)) + \
                stats.norm.cdf((-zr - zrc) * np.sqrt(n - 3))
            return power

    else:

        def func(r, n, power, alpha):
            dof = n - 2
            ttt = stats.t.ppf(1 - alpha, dof)
            rc = np.sqrt(ttt**2 / (ttt**2 + dof))
            zr = np.arctanh(r) + r / (2 * (n - 1))
            zrc = np.arctanh(rc)
            power = stats.norm.cdf((zr - zrc) * np.sqrt(n - 3))
            return power

    # Evaluate missing variable
    if power is None and n is not None and r is not None:
        # Compute achieved power given r, n and alpha
        return func(r, n, power=None, alpha=alpha)

    elif n is None and power is not None and r is not None:
        # Compute required sample size given r, power and alpha

        def _eval_n(n, r, power, alpha):
            return func(r, n, power, alpha) - power

        try:
            return brenth(_eval_n, 4 + 1e-10, 1e+09, args=(r, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    elif r is None and power is not None and n is not None:
        # Compute achieved r given sample size, power and alpha level

        def _eval_r(r, n, power, alpha):
            return func(r, n, power, alpha) - power

        try:
            return brenth(_eval_r, 1e-10, 1 - 1e-10, args=(n, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    else:
        # Compute achieved alpha (significance) level given r, n and power

        def _eval_alpha(alpha, r, n, power):
            return func(r, n, power, alpha) - power

        try:
            return brenth(_eval_alpha, 1e-10, 1 - 1e-10, args=(r, n, power))
        except ValueError:  # pragma: no cover
            return np.nan
