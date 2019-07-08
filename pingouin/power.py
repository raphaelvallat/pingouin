# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np
from scipy import stats
from scipy.optimize import brenth

__all__ = ["power_ttest", "power_ttest2n", "power_anova", "power_rm_anova",
           "power_corr", "power_chi2"]


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
        Can be `"one-sample"`, `"two-samples"` or `"paired"`.
        Note that `"one-sample"` and `"paired"` have the same behavior.
    tail : str
        Indicates the alternative of the test. Can be either `'two-sided'`,
        `'greater'` or `'less'`.

    Notes
    -----
    Exactly ONE of the parameters ``d``, ``n``, ``power`` and ``alpha`` must
    be passed as None, and that parameter is determined from the others.

    For a paired T-test, the sample size ``n`` corresponds to the number of
    pairs. For an independent two-sample T-test with equal sample sizes, ``n``
    corresponds to the sample size of each group (i.e. number of observations
    in one group). If the sample sizes are unequal, please use the
    :py:func:`power_ttest2n` function instead.

    Notice that ``alpha`` has a default value of 0.05 so None must be
    explicitly passed if you want to compute it.

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
    parameter :math:`\\delta` and degrees of freedom :math:`v`.
    In case of paired groups, this is:

    .. math:: \\delta = d * \\sqrt n
    .. math:: v = n - 1

    and in case of independent groups with equal sample sizes:

    .. math:: \\delta = d * \\sqrt{\\frac{n}{2}}
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
    1. Compute power of a one-sample T-test given ``d``, ``n`` and ``alpha``

    >>> from pingouin import power_ttest
    >>> print('power: %.4f' % power_ttest(d=0.5, n=20, contrast='one-sample'))
    power: 0.5645

    2. Compute required sample size given ``d``, ``power`` and ``alpha``

    >>> print('n: %.4f' % power_ttest(d=0.5, power=0.80, tail='greater'))
    n: 50.1508

    3. Compute achieved ``d`` given ``n``, ``power`` and ``alpha`` level

    >>> print('d: %.4f' % power_ttest(n=20, power=0.80, alpha=0.05,
    ...                               contrast='paired'))
    d: 0.6604

    4. Compute achieved alpha level given ``d``, ``n`` and ``power``

    >>> print('alpha: %.4f' % power_ttest(d=0.5, n=20, power=0.80, alpha=None))
    alpha: 0.4430

    5. One-sided tests

    >>> from pingouin import power_ttest
    >>> print('power: %.4f' % power_ttest(d=0.5, n=20, tail='greater'))
    power: 0.4634

    >>> print('power: %.4f' % power_ttest(d=0.5, n=20, tail='less'))
    power: 0.0007
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [d, n, power, alpha]])
    if n_none != 1:
        raise ValueError('Exactly one of n, d, power, and alpha must be None.')

    # Safety checks
    possible_tails = ['two-sided', 'greater', 'less']
    assert tail in possible_tails, 'Invalid tail argument.'
    assert contrast.lower() in ['one-sample', 'paired', 'two-samples']
    tsample = 2 if contrast.lower() == 'two-samples' else 1
    tside = 2 if tail == 'two-sided' else 1
    if d is not None and tside == 2:
        d = abs(d)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    if tail == 'less':

        def func(d, n, power, alpha):
            dof = (n - 1) * tsample
            nc = d * np.sqrt(n / tsample)
            tcrit = stats.t.ppf(alpha / tside, dof)
            return stats.nct.cdf(tcrit, dof, nc)

    elif tail == 'two-sided':

        def func(d, n, power, alpha):
            dof = (n - 1) * tsample
            nc = d * np.sqrt(n / tsample)
            tcrit = stats.t.ppf(1 - alpha / tside, dof)
            return (stats.nct.sf(tcrit, dof, nc) +
                    stats.nct.cdf(-tcrit, dof, nc))

    else:  # Tail = greater

        def func(d, n, power, alpha):
            dof = (n - 1) * tsample
            nc = d * np.sqrt(n / tsample)
            tcrit = stats.t.ppf(1 - alpha / tside, dof)
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
        if tail == 'two-sided':
            b0, b1 = 1e-07, 10
        elif tail == 'less':
            b0, b1 = -10, 5
        else:
            b0, b1 = -5, 10

        def _eval_d(d, n, power, alpha):
            return func(d, n, power, alpha) - power

        try:
            return brenth(_eval_d, b0, b1, args=(n, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    else:
        # Compute achieved alpha (significance) level given d, n and power

        def _eval_alpha(alpha, d, n, power):
            return func(d, n, power, alpha) - power

        try:
            return brenth(_eval_alpha, 1e-10, 1 - 1e-10, args=(d, n, power))
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
        Indicates the alternative of the test. Can be either `'two-sided'`,
        `'greater'` or `'less'`.

    Notes
    -----
    Exactly ONE of the parameters ``d``, ``power`` and ``alpha`` must
    be passed as None, and that parameter is determined from the others.

    Notice that ``alpha`` has a default value of 0.05 so None must be
    explicitly passed if you want to compute it.

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
    parameter :math:`\\delta` and degrees of freedom :math:`v`.
    In case of two independent groups with unequal sample sizes, this is:

    .. math:: \\delta = d * \\sqrt{\\frac{n_i * n_j}{n_i + n_j}}
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
    1. Compute achieved power of a T-test given ``d``, ``n`` and ``alpha``

    >>> from pingouin import power_ttest2n
    >>> print('power: %.4f' % power_ttest2n(nx=20, ny=15, d=0.5,
    ...                                     tail='greater'))
    power: 0.4164

    2. Compute achieved ``d`` given ``n``, ``power`` and ``alpha`` level

    >>> print('d: %.4f' % power_ttest2n(nx=20, ny=15, power=0.80, alpha=0.05))
    d: 0.9859

    3. Compute achieved alpha level given ``d``, ``n`` and ``power``

    >>> print('alpha: %.4f' % power_ttest2n(nx=20, ny=15, d=0.5,
    ...                                     power=0.80, alpha=None))
    alpha: 0.5000
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [d, power, alpha]])
    if n_none != 1:
        raise ValueError('Exactly one of d, power, and alpha must be None')

    # Safety checks
    possible_tails = ['two-sided', 'greater', 'less']
    assert tail in possible_tails, 'Invalid tail argument.'
    tside = 2 if tail == 'two-sided' else 1
    if d is not None and tside == 2:
        d = abs(d)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    if tail == 'less':

        def func(d, nx, ny, power, alpha):
            dof = nx + ny - 2
            nc = d * (1 / np.sqrt(1 / nx + 1 / ny))
            tcrit = stats.t.ppf(alpha / tside, dof)
            return stats.nct.cdf(tcrit, dof, nc)

    elif tail == 'two-sided':

        def func(d, nx, ny, power, alpha):
            dof = nx + ny - 2
            nc = d * (1 / np.sqrt(1 / nx + 1 / ny))
            tcrit = stats.t.ppf(1 - alpha / tside, dof)
            return (stats.nct.sf(tcrit, dof, nc) +
                    stats.nct.cdf(-tcrit, dof, nc))

    else:  # Tail = greater

        def func(d, nx, ny, power, alpha):
            dof = nx + ny - 2
            nc = d * (1 / np.sqrt(1 / nx + 1 / ny))
            tcrit = stats.t.ppf(1 - alpha / tside, dof)
            return stats.nct.sf(tcrit, dof, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power given d, n and alpha
        return func(d, nx, ny, power=None, alpha=alpha)

    elif d is None:
        # Compute achieved d given sample size, power and alpha level
        if tail == 'two-sided':
            b0, b1 = 1e-07, 10
        elif tail == 'less':
            b0, b1 = -10, 5
        else:
            b0, b1 = -5, 10

        def _eval_d(d, nx, ny, power, alpha):
            return func(d, nx, ny, power, alpha) - power

        try:
            return brenth(_eval_d, b0, b1, args=(nx, ny, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    else:
        # Compute achieved alpha (significance) level given d, n and power

        def _eval_alpha(alpha, d, nx, ny, power):
            return func(d, nx, ny, power, alpha) - power

        try:
            return brenth(_eval_alpha, 1e-10, 1 - 1e-10, args=(d, nx, ny,
                                                               power))
        except ValueError:  # pragma: no cover
            return np.nan


def power_anova(eta=None, k=None, n=None, power=None, alpha=0.05):
    """
    Evaluate power, sample size, effect size or
    significance level of a one-way balanced ANOVA.

    Parameters
    ----------
    eta : float
        ANOVA effect size (eta-square = :math:`\\eta^2`).
    k : int
        Number of groups
    n : int
        Sample size per group. Groups are assumed to be balanced
        (i.e. same sample size).
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level :math:`\\alpha` (type I error probability).
        The default is 0.05.

    Notes
    -----
    Exactly ONE of the parameters ``eta``, ``k``, ``n``, ``power`` and
    ``alpha`` must be passed as None, and that parameter is determined from
    the others.

    Notice that ``alpha`` has a default value of 0.05 so None must be
    explicitly passed if you want to compute it.

    This function is a mere Python translation of the original `pwr.anova.test`
    function implemented in the `pwr` package. All credit goes to the author,
    Stephane Champely.

    Statistical power is the likelihood that a study will
    detect an effect when there is an effect there to be detected.
    A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one.
    Statistical power is mainly affected by the effect size and the sample
    size.

    For one-way ANOVA, eta-square is the same as partial
    eta-square. It can be evaluated from the F-value (:math:`F^*`) and the
    degrees of freedom of the ANOVA (:math:`v_1, v_2`) using the following
    formula:

    .. math:: \\eta^2 = \\frac{v_1 F^*}{v_1 F^* + v_2}

    Note that GPower uses the :math:`f` effect size instead of the
    :math:`\\eta^2`. The formula to convert from one to the other are given
    below:

    .. math:: f = \\sqrt{\\frac{\\eta^2}{1 - \\eta^2}}

    .. math:: \\eta^2 = \\frac{f^2}{1 + f^2}

    Using :math:`\\eta^2` and the total sample size :math:`N`, the
    non-centrality parameter is defined by:

    .. math:: \\delta = N * \\frac{\\eta^2}{1 - \\eta^2}

    Then the critical value of the non-central F-distribution is computed using
    the percentile point function of the F-distribution with:

    .. math:: q = 1 - \\alpha
    .. math:: v_1 = k - 1
    .. math:: v_2 = N - k

    where :math:`k` is the number of groups.

    Finally, the power of the ANOVA is calculated using the survival function
    of the non-central F-distribution using the previously computed critical
    value, non-centrality parameter, and degrees of freedom.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other
    variables (i.e. sample size, effect size, or significance level). If the
    solving fails, a nan value is returned.

    Results have been validated against GPower and the R pwr package.

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
    power: 0.6082

    2. Compute required number of groups

    >>> print('k: %.4f' % power_anova(eta=0.1, n=20, power=0.80))
    k: 6.0944

    3. Compute required sample size

    >>> print('n: %.4f' % power_anova(eta=0.1, k=3, power=0.80))
    n: 29.9255

    4. Compute achieved effect size

    >>> print('eta: %.4f' % power_anova(n=20, k=4, power=0.80, alpha=0.05))
    eta: 0.1255

    5. Compute achieved alpha (significance)

    >>> print('alpha: %.4f' % power_anova(eta=0.1, n=20, k=4, power=0.80,
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

        def _eval_k(k, f_sq, n, power, alpha):
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


def power_rm_anova(eta=None, m=None, n=None, power=None, alpha=0.05,
                   corr=0.5, epsilon=1):
    """
    Evaluate power, sample size, effect size or
    significance level of a balanced one-way repeated measures ANOVA.

    Parameters
    ----------
    eta : float
        ANOVA effect size (eta-square = :math:`\\eta^2`).
    m : int
        Number of repeated measurements.
    n : int
        Sample size per measurement. All measurements must have the same
        sample size.
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level :math:`\\alpha` (type I error probability).
        The default is 0.05.
    corr : float
        Average correlation coefficient among repeated measurements.
        The default is :math:`r=0.5`.
    epsilon : float
        Epsilon adjustement factor for sphericity. This can be
        calculated using the :py:func:`pingouin.epsilon` function.

    Notes
    -----
    Exactly ONE of the parameters ``eta``, ``m``, ``n``, ``power`` and
    ``alpha`` must be passed as None, and that parameter is determined from
    the others.

    Notice that ``alpha`` has a default value of 0.05 so None must be
    explicitly passed if you want to compute it.

    Statistical power is the likelihood that a study will
    detect an effect when there is an effect there to be detected.
    A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one.
    Statistical power is mainly affected by the effect size and the sample
    size.

    For one-way repeated measure ANOVA, eta-square is the same as partial
    eta-square. It can be evaluated from the F-value (:math:`F^*`) and the
    degrees of freedom of the ANOVA (:math:`v_1, v_2`) using the following
    formula:

    .. math:: \\eta^2 = \\frac{v_1 F^*}{v_1 F^* + v_2}

    Note that GPower uses the :math:`f` effect size instead of the
    :math:`\\eta^2`. The formula to convert from one to the other are given
    below:

    .. math:: f = \\sqrt{\\frac{\\eta^2}{1 - \\eta^2}}

    .. math:: \\eta^2 = \\frac{f^2}{1 + f^2}

    Using :math:`\\eta^2`, the sample size :math:`N`, the number of repeated
    measurements :math:`m`, the epsilon correction factor :math:`\\epsilon`
    (see :py:func:`pingouin.epsilon`), and the average correlation between
    the repeated measures :math:`c`, one can then calculate the
    non-centrality parameter as follow:

    .. math:: \\delta = \\frac{f^2 * N * m * \\epsilon}{1 - c}

    Then the critical value of the non-central F-distribution is computed using
    the percentile point function of the F-distribution with:

    .. math:: q = 1 - \\alpha
    .. math:: v_1 = (m - 1) * \\epsilon
    .. math:: v_2 = (N - 1) * v_1

    Finally, the power of the ANOVA is calculated using the survival function
    of the non-central F-distribution using the previously computed critical
    value, non-centrality parameter, and degrees of freedom.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other
    variables (i.e. sample size, effect size, or significance level). If the
    solving fails, a nan value is returned.

    Results have been validated against GPower.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Hillsdale,NJ: Lawrence Erlbaum.

    .. [2] https://cran.r-project.org/web/packages/pwr/pwr.pdf

    Examples
    --------
    1. Compute achieved power

    >>> from pingouin import power_rm_anova
    >>> print('power: %.4f' % power_rm_anova(eta=0.1, m=3, n=20))
    power: 0.8913

    2. Compute required number of groups

    >>> print('m: %.4f' % power_rm_anova(eta=0.1, n=20, power=0.90))
    m: 3.1347

    3. Compute required sample size

    >>> print('n: %.4f' % power_rm_anova(eta=0.1, m=3, power=0.80))
    n: 15.9979

    4. Compute achieved effect size

    >>> print('eta: %.4f' % power_rm_anova(n=20, m=4, power=0.80, alpha=0.05))
    eta: 0.0680

    5. Compute achieved alpha (significance)

    >>> print('alpha: %.4f' % power_rm_anova(eta=0.1, n=20, m=4, power=0.80,
    ...                                   alpha=None))
    alpha: 0.0081

    Let's take a more concrete example. First, we'll load a repeated measures
    dataset in wide-format. Each row is an observation (e.g. a subject), and
    each column a successive repeated measurements (e.g t=0, t=1, ...).

    >>> import pingouin as pg
    >>> data = pg.read_dataset('rm_anova_wide')
    >>> data.head()
       Before  1 week  2 week  3 week
    0     4.3     5.3     4.8     6.3
    1     3.9     2.3     5.6     4.3
    2     4.5     2.6     4.1     NaN
    3     5.1     4.2     6.0     6.3
    4     3.8     3.6     4.8     6.8

    Note that this dataset has some missing values. We'll simply delete any
    row with one or more missing values, and then compute a repeated
    measures ANOVA:

    >>> data = data.dropna()
    >>> pg.rm_anova(data)
       Source  ddof1  ddof2      F     p-unc    np2    eps
    0  Within      3     24  5.201  0.006557  0.394  0.694

    The repeated measures ANOVA is significant at the 0.05 level. Now, we can
    easily compute the power of the ANOVA with the information in the ANOVA
    table:

    >>> # n is the sample size and m is the number of repeated measures
    >>> n, m = data.shape
    >>> pg.power_rm_anova(eta=0.394, m=m, n=n, epsilon=0.694)
    0.9976707714861207

    Our ANOVA has a very high statistical power. However, to be even more
    accurate in our power calculation, we should also fill in the average
    correlation among repeated measurements. Since our dataframe is in
    wide-format (with each column being a successive measurement), this can
    be done by taking the mean of the superdiagonal of the correlation matrix,
    which is similar to manually calculating the correlation between each
    successive pairwise measurements and then taking the mean.
    Since correlation coefficients are not normally distributed, we
    use the *r-to-z* transform prior to averaging (:py:func:`numpy.arctanh`),
    and then the *z-to-r* transform (:py:func:`numpy.tanh`) to convert back to
    a correlation coefficient. This gives a more precise estimate of the mean.

    >>> import numpy as np
    >>> corr = np.diag(data.corr(), k=1)
    >>> avgcorr = np.tanh(np.arctanh(corr).mean())
    >>> avgcorr
    -0.19955358859483566

    In this example, we're using a fake dataset and the average correlation is
    negative. However, it will most likely be positive with real data. Let's
    now compute the final power of the repeated measures ANOVA:

    >>> pg.power_rm_anova(eta=0.394, m=m, n=n, epsilon=0.694, corr=avgcorr)
    0.8545404196391064
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [eta, m, n, power, alpha]])
    if n_none != 1:
        msg = 'Exactly one of eta, m, n, power, and alpha must be None.'
        raise ValueError(msg)

    # Safety checks
    assert 0 < epsilon <= 1, 'epsilon must be between 0 and 1.'
    assert -1 < corr < 1, 'corr must be between -1 and 1.'
    if eta is not None:
        eta = abs(eta)
        f_sq = eta / (1 - eta)
    if alpha is not None:
        assert 0 < alpha <= 1, 'alpha must be between 0 and 1.'
    if power is not None:
        assert 0 < power <= 1, 'power must be between 0 and 1.'
    if n is not None:
        assert n > 1, 'The sample size n must be > 1.'
    if m is not None:
        assert m > 1, 'The number of repeated measures m must be > 1.'

    def func(f_sq, m, n, power, alpha, corr):
        dof1 = (m - 1) * epsilon
        dof2 = (n - 1) * dof1
        nc = (f_sq * n * m * epsilon) / (1 - corr)
        fcrit = stats.f.ppf(1 - alpha, dof1, dof2)
        return stats.ncf.sf(fcrit, dof1, dof2, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power
        return func(f_sq, m, n, power, alpha, corr)

    elif m is None:
        # Compute required number of repeated measures

        def _eval_m(m, f_sq, n, power, alpha, corr):
            return func(f_sq, m, n, power, alpha, corr) - power

        try:
            return brenth(_eval_m, 2, 100, args=(f_sq, n, power, alpha, corr))
        except ValueError:  # pragma: no cover
            return np.nan

    elif n is None:
        # Compute required sample size

        def _eval_n(n, f_sq, m, power, alpha, corr):
            return func(f_sq, m, n, power, alpha, corr) - power

        try:
            return brenth(_eval_n, 5, 1e+6, args=(f_sq, m, power, alpha, corr))
        except ValueError:  # pragma: no cover
            return np.nan

    elif eta is None:
        # Compute achieved eta

        def _eval_eta(f_sq, m, n, power, alpha, corr):
            return func(f_sq, m, n, power, alpha, corr) - power

        try:
            f_sq = brenth(_eval_eta, 1e-10, 1 - 1e-10, args=(m, n, power,
                                                             alpha, corr))
            return f_sq / (f_sq + 1)  # Return eta-square
        except ValueError:  # pragma: no cover
            return np.nan

    else:
        # Compute achieved alpha

        def _eval_alpha(alpha, f_sq, m, n, power, corr):
            return func(f_sq, m, n, power, alpha, corr) - power

        try:
            return brenth(_eval_alpha, 1e-10, 1 - 1e-10, args=(f_sq, m, n,
                                                               power, corr))
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
    Exactly ONE of the parameters ``r``, ``n``, ``power`` and ``alpha`` must
    be passed as None, and that parameter is determined from the others.

    Notice that ``alpha`` has a default value of 0.05 so None must be
    explicitly passed if you want to compute it.

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
    1. Compute achieved power given ``r``, ``n`` and ``alpha``

    >>> from pingouin import power_corr
    >>> print('power: %.4f' % power_corr(r=0.5, n=20))
    power: 0.6379

    2. Compute required sample size given ``r``, ``power`` and ``alpha``

    >>> print('n: %.4f' % power_corr(r=0.5, power=0.80,
    ...                                tail='one-sided'))
    n: 22.6091

    3. Compute achieved ``r`` given ``n``, ``power`` and ``alpha`` level

    >>> print('r: %.4f' % power_corr(n=20, power=0.80, alpha=0.05))
    r: 0.5822

    4. Compute achieved alpha level given ``r``, ``n`` and ``power``

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


def power_chi2(dof, w=None, n=None, power=None, alpha=0.05):
    """
    Evaluate power, sample size, effect size or
    significance level of chi-squared tests.

    Parameters
    ----------
    dof : float
        Degree of freedom (depends on the chosen test).
    w : float
        Effect size.
    n : int
        Total number of observations.
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability).
        The default is 0.05.

    Notes
    -----
    Exactly ONE of the parameters ``w``, ``n``, ``power`` and ``alpha`` must
    be passed as None, and that parameter is determined from the others. The
    degrees of freedom ``dof`` must always be specified.

    Notice that ``alpha`` has a default value of 0.05 so None must be
    explicitly passed if you want to compute it.

    This function is a mere Python translation of the original `pwr.chisq.test`
    function implemented in the `pwr` package. All credit goes to the author,
    Stephane Champely.

    Statistical power is the likelihood that a study will
    detect an effect when there is an effect there to be detected.
    A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one.
    Statistical power is mainly affected by the effect size and the sample
    size.

    The non-centrality parameter is defined by:

    .. math:: \\delta = N * w^2

    Then the critical value is computed using the percentile point function of
    the :math:`\\chi^2` distribution with the alpha level and degrees of
    freedom.

    Finally, the power of the chi-squared test is calculated using the survival
    function of the non-central :math:`\\chi^2` distribution using the
    previously computed critical value, non-centrality parameter, and the
    degrees of freedom of the test.

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

    >>> from pingouin import power_chi2
    >>> print('power: %.4f' % power_chi2(dof=1, w=0.3, n=20))
    power: 0.2687

    2. Compute required sample size

    >>> print('n: %.4f' % power_chi2(dof=3, w=0.3, power=0.80))
    n: 121.1396

    3. Compute achieved effect size

    >>> print('w: %.4f' % power_chi2(dof=2, n=20, power=0.80, alpha=0.05))
    w: 0.6941

    4. Compute achieved alpha (significance)

    >>> print('alpha: %.4f' % power_chi2(dof=1, w=0.5, n=20, power=0.80,
    ...                                   alpha=None))
    alpha: 0.1630
    """
    assert isinstance(dof, (int, float))
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [w, n, power, alpha]])
    if n_none != 1:
        err = 'Exactly one of w, n, power, and alpha must be None.'
        raise ValueError(err)

    # Safety checks
    if w is not None:
        w = abs(w)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    def func(w, n, power, alpha):
        k = stats.chi2.ppf(1 - alpha, dof)
        nc = n * w**2
        return stats.ncx2.sf(k, dof, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power
        return func(w, n, power, alpha)

    elif n is None:
        # Compute required sample size

        def _eval_n(n, w, power, alpha):
            return func(w, n, power, alpha) - power

        try:
            return brenth(_eval_n, 1, 1e+07, args=(w, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    elif w is None:
        # Compute achieved effect size

        def _eval_w(w, n, power, alpha):
            return func(w, n, power, alpha) - power

        try:
            return brenth(_eval_w, 1e-10, 1e+07, args=(n, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    else:
        # Compute achieved alpha

        def _eval_alpha(alpha, w, n, power):
            return func(w, n, power, alpha) - power

        try:
            return brenth(_eval_alpha, 1e-10, 1 - 1e-10, args=(w, n, power))
        except ValueError:  # pragma: no cover
            return np.nan
