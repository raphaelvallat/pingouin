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
        Specify the two groups are independant or related (i.e. repeated).
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
    the following formula: eta = (fval * dof1) / (fval * dof1 + dof2)

    Results have been tested against GPower.

    Examples
    --------
    1. Achieved power of a one-way ANOVA.

        >>> ntot, ngroups = 60, 3
        >>> eta = .2
        >>> power = anova_power(d, ntot, ngroups)
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
