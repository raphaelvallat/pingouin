# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import numpy as np

__all__ = ["bayesfactor_ttest", "bayesfactor_pearson"]


def bayesfactor_ttest(t, nx, ny=None, paired=False, r=.707):
    """
    Calculates the Jeffrey-Zellner-Siow (JZS) Bayes Factor for a one or
    two-sample T-test given t-value and sample size(s).

    See Rouder et al. (2009) for details.

    Parameters
    ----------
    t : float
        T-value of the T-test
    nx : int
        Sample size of first group
    ny : int
        Sample size of second group (only needed in case of an independant
        two-sample T-test)
    r : float
        Scale factor. The default is 0.707.

    Return
    ------
    bf : float
        Bayes Factor (BF10).
        The Bayes Factor quantifies the evidence in favour of the
        alternative hypothesis.

    Notes
    -----
    Adapted from a Matlab code found at
    https://github.com/anne-urai/Tools/tree/master/stats/BayesFactors

    If you would like to compute the Bayes Factor directly from the raw data
    instead of from the T-value, use the ttest function.

    Example
    -------
    1. Bayes Factor of an independant two-sample T-test

        >>> from pingouin import bayesfactor_ttest
        >>> bf = bayesfactor_ttest(3.5, 20, 20)
        >>> print("Bayes Factor: %.2f (two-sample independant)" % bf)

    2. Bayes Factor of a paired two-sample T-test

        >>> from pingouin import bayesfactor_ttest
        >>> bf = bayesfactor_ttest(3.5, 20, 20, paired=True)
        >>> print("Bayes Factor: %.2f (two-sample paired)" % bf)

    3. Bayes Factor of an one-sample T-test

        >>> from pingouin import bayesfactor_ttest
        >>> bf = bayesfactor_ttest(3.5, 20)
        >>> print("Bayes Factor: %.2f (one-sample)" % bf)
    """
    from scipy.integrate import quad
    one_sample = True if ny is None or ny == 1 else False

    # Define functions
    def F_ind(g, t, nx, ny, r):
        return (1 + (nx * ny / (nx + ny)) * g * r**2)**(-.5) * \
               (1 + t**2. / ((1 + (nx * ny / (nx + ny)) * g * r**2) *
                             (nx + ny - 2)))**(-(nx + ny - 1) / 2) * \
               (2 * np.pi)**(-1. / 2) * g**(-3 / 2) * np.exp(-1. / (2 * g))

    def F_paired(g, t, n, r):
        return (1 + n * g * r**2)**(-.5) * (1 + t**2 / ((1 + n * g * r**2) *
                                            (n - 1)))**(-n / 2) *  \
               (2 * np.pi)**(-.5) * g**(-3. / 2) * np.exp(-1 / (2 * g))

    # JZS Bayes factor calculation
    if one_sample or paired:
        bf01 = (1 + t**2 / (nx - 1))**(-nx / 2) / \
            quad(F_paired, 0, np.inf, args=(t, nx, r))[0]

    else:
        bf01 = (1 + t**2 / (nx + ny - 2))**(-(nx + ny - 1) / 2) / \
            quad(F_ind, 0, np.inf, args=(t, nx, ny, r))[0]

    # Invert Bayes Factor (alternative hypothesis)
    return np.round(1 / bf01, 3)


def bayesfactor_pearson(r, n):
    """
    Calculates the Jeffrey-Zellner-Siow (JZS) Bayes Factor for
    correlation r and sample size n.

    See Wetzels & Wagemakers (2012) for details.

    Parameters
    ----------
    r : float
        Pearson correlation coefficient
    n : int
        Sample size

    Return
    ------
    bf : float
        Bayes Factor (BF10).
        The Bayes Factor quantifies the evidence in favour of the alternative
        hypothesis.

    Notes
    -----
    Adapted from a Matlab code found at
    https://github.com/anne-urai/Tools/blob/master/stats/BayesFactors/corrbf.m

    If you would like to compute the Bayes Factor directly from the raw data
    instead of from the correlation coefficient, use the corr function.

    Example
    -------
    1. Bayes Factor of a Pearson correlation

        >>> from pingouin import bayesfactor_pearson
        >>> bf = bayesfactor_pearson(0.6, 20)
        >>> print("Bayes Factor: %.3f" % bf)
    """
    from scipy.integrate import quad
    from scipy.special import gamma

    # Function to be integrated
    def F(g, r, n):
        return np.exp(((n - 2) / 2) * np.log(1 + g) + (-(n - 1) / 2) *
                      np.log(1 + (1 - r**2) * g) + (-3 / 2) *
                      np.log(g) + - n / (2 * g))

    # JZS Bayes factor calculation
    bf10 = np.sqrt((n / 2)) / gamma(1 / 2) * quad(F, 0, np.inf, args=(r, n))[0]
    return np.round(bf10, 3)
