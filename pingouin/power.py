# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np

__all__ = ["ttest_power", "anova_power"]


def ttest_power(d, nx, ny=None, paired=False, tail='two-sided',
                alpha=.05):
    """Determine achieved power of a T test given effect size, sample size
    and alpha level.

    Parameters
    ----------
    d : float
        Effect size (Cohen d, Hedges g or Glass delta)
    nx, ny : int
        Sample sizes of the two groups. Must be equal when paired is True.
        If only nx is specified, assumes that the test is a one-sample T-test.
    paired : boolean
        Specify the two groups are independent or related (i.e. repeated).
    tail : string
        Specify whether the test is 'one-sided' or 'two-sided'.
    alpha : float
        Significance level of the test.

    Returns
    -------
    power : float
        Achieved power of the test.

    Notes
    -----
    Statistical power is the likelihood that a study will
    detect an effect when there is an effect there to be detected.
    A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one.
    Statistical power is mainly affected by the effect size and the sample
    size.

    The first step is to use the Cohen's d to calculate the non-centrality
    parameter and degrees of freedom. In case of paired groups, this is:

    .. math:: \delta = d * \sqrt n
    .. math:: \mathtt{df} = n - 1

    and in case of independent groups:

    .. math:: \delta = d * \sqrt{\dfrac{n_i * n_j}{n_i + n_j}}
    .. math:: \mathtt{df} = n_i + n_j - 2

    where :math:`d` is the Cohen d, :math:`n` the sample size,
    :math:`n_i` the sample size of the first group and
    :math:`n_j` the sample size of the second group,

    The critical value is then found using the percent point function of the T
    distribution with :math:`q = 1 - alpha` and :math:`\mathtt{df}`
    degrees of freedom.

    Finally, the power of the test is given by the survival function of the
    non-central distribution using the previously calculated critical value,
    degrees of freedom and non-centrality parameter.

    Results have been tested against GPower.

    Examples
    --------
    1. Achieved power of a paired two-sample T-test.

        >>> nx, ny = 20, 20
        >>> d = 0.5
        >>> power = ttest_power(d, nx, ny, paired=True, tail='one-sided')
        >>> print(power)
            0.695

    2. Achieved power of a one sample T-test.

        >>> nx = 20
        >>> d = 0.6
        >>> power = ttest_power(d, nx)
        >>> print(power)
            0.721
    """
    from scipy.stats import t, nct
    d = np.abs(d)
    if paired is True or ny is None or ny == 1:
        nc = d * np.sqrt(nx)
        dof = nx - 1
    else:
        nc = d * np.sqrt(nx * ny / (nx + ny))
        dof = nx + ny - 2
    # Critical T
    if tail == 'one-sided':
        tcrit = t.ppf(1 - alpha, dof)
    else:
        tcrit = t.ppf(1 - alpha / 2, dof)
    return nct.sf(tcrit, dof, nc).round(3)


def anova_power(eta, ntot, ngroups, alpha=.05):
    """Determine achieved power of a one-way ANOVA given effect size,
    sample size, number of groups and alpha level.

    Parameters
    ----------
    eta : float
        Effect size (eta-square or partial eta-square).
    ntot : int
        Total sample size.
    ngroups : int
        Number of groups.
    alpha : float
        Significance level of the test.

    Returns
    -------
    power : float
        Achieved power of the test.

    Notes
    -----
    For one-way ANOVA, partial eta-square is the same as eta-square. It can be
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
    .. math:: \mathtt{df_1} = r - 1
    .. math:: \mathtt{df_2} = N - r

    where :math:`r` is the number of groups.

    Finally, the power of the ANOVA is calculated using the survival function
    of the non-central F-distribution using the previously computed critical
    value, non-centrality parameter, and degrees of freedom.

    Results have been tested against GPower.

    Examples
    --------
    1. Achieved power of a one-way ANOVA.

        >>> ntot, ngroups = 60, 3
        >>> eta = .2
        >>> power = anova_power(eta, ntot, ngroups)
        >>> print(power)
            0.932
    """
    from scipy.stats import f, ncf
    # Non-centrality parameter
    f_sq = eta / (1 - eta)
    nc = ntot * f_sq
    # Degrees of freedom
    dof1 = ngroups - 1
    dof2 = ntot - ngroups
    # Critical F
    fcrit = f.ppf(1 - alpha, dof1, dof2)
    return ncf.sf(fcrit, dof1, dof2, nc).round(3)
