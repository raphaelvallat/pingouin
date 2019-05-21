# Author: Arthur Paulino <arthurleonardo.ap@gmail.com>
# Date: May 2019
from scipy import stats as sp_stats
from scipy.stats.contingency import expected_freq
from scipy.stats import power_divergence
import pandas as pd
import numpy as np
import warnings
from .utils import dichotomous_crosstab


__all__ = ['chi2', 'chi2_mcnemar']


def chi2(data, x, y, correction=True):
    """
    Chi-squared tests between two categorical variables.

    The test is computed for different values of :math:`\\lambda`: 1, 2/3, 0,
    -1/2, -1 and -2 (Cressie and Read, 1984).

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the ocurrences for the test.
    x, y : string
        The variables names for the Chi-squared test. Must be names of columns
        in ``data``.
    correction : bool
        Whether to apply Yates' correction when the degree of freedom of the
        observed contingency table is 1 (Yates 1934).

    Returns
    -------
    expected : pd.DataFrame
        The expected contingency table of frequencies.
    observed : pd.DataFrame
        The (corrected or not) observed contingency table of frequencies.
    stats : pd.DataFrame
        The tests summary, containing four columns:

        * ``'test'``: The statistic name
        * ``'lambda'``: The :math:`\\lambda` value used for the power\
                        divergence statistic
        * ``'chi2'``: The test statistic
        * ``'p'``: The p-value of the test

    Notes
    -----
    From Wikipedia:

    *The chi-squared test is used to determine whether there is a significant
    difference between the expected frequencies and the observed frequencies
    in one or more categories.*

    As application examples, this test can be used to *i*) evaluate the
    quality of a categorical variable in a classification problem or to *ii*)
    check the similarity between two categorical variables. In the first
    example, a good categorical predictor and the class column should present
    high :math:`\\chi^2` and low p-value. In the second example, similar
    categorical variables should present low :math:`\\chi^2` and high p-value.

    This function is a wrapper around the
    :py:func:`scipy.stats.power_divergence` function.

    .. warning :: As a general guideline for the consistency of this test, the
        observed and the expected contingency tables should not have cells
        with frequencies lower than 5.

    References
    ----------
    .. [1] Cressie, N., & Read, T. R. (1984). Multinomial goodness‐of‐fit
           tests. Journal of the Royal Statistical Society: Series B
           (Methodological), 46(3), 440-464.

    .. [2] Yates, F. (1934). Contingency Tables Involving Small Numbers and the
           :math:`\\chi^2` Test. Supplement to the Journal of the Royal
           Statistical Society, 1, 217-235.

    Examples
    --------
    Let's see if gender is a good categorical predictor for the presence of
    heart disease.

    >>> import pingouin as pg
    >>> data = pg.read_dataset('chi2')
    >>> data['sex'].value_counts(ascending=True)
    0     96
    1    207
    Name: sex, dtype: int64

    If gender is not a good predictor for heart disease, we should expect the
    same 96:207 ratio across the target classes.

    >>> expected, observed, stats = pg.chi2(data, x='sex', y='target')
    >>> expected
    target          0           1
    sex
    0       43.722772   52.277228
    1       94.277228  112.722772

    Let's see what the data tells us.

    >>> observed
    target      0     1
    sex
    0        24.5  71.5
    1       113.5  93.5

    The proportion is lower on the class 0 and higher on the class 1. The
    tests should be sensitive to this difference.

    >>> stats
                     test  lambda    chi2  dof             p
    0             pearson   1.000  22.717    1  1.876778e-06
    1        cressie-read   0.667  22.931    1  1.678845e-06
    2      log-likelihood   0.000  23.557    1  1.212439e-06
    3       freeman-tukey  -0.500  24.220    1  8.595211e-07
    4  mod-log-likelihood  -1.000  25.071    1  5.525544e-07
    5              neyman  -2.000  27.458    1  1.605471e-07

    Very low p-values indeed. The gender qualifies as a good predictor for the
    presence of heart disease on this dataset.
    """
    # Python code inspired by SciPy's chi2_contingency
    assert isinstance(data, pd.DataFrame), 'data must be a pandas DataFrame.'
    assert isinstance(x, str), 'x must be a string.'
    assert isinstance(y, str), 'y must be a string.'
    assert all(col in data.columns for col in (x, y)),\
        'columns are not in dataframe.'
    assert isinstance(correction, bool), 'correction must be a boolean.'

    observed = pd.crosstab(data[x], data[y])

    if observed.size == 0:
        raise ValueError('No data; observed has size 0.')

    expected = pd.DataFrame(expected_freq(observed), index=observed.index,
                            columns=observed.columns)

    # All count frequencies should be at least 5
    for df, name in zip([observed, expected], ['observed', 'expected']):
        if (df < 5).any(axis=None):
            warnings.warn('Low count on {} frequencies.'.format(name))

    dof = expected.size - sum(expected.shape) + expected.ndim - 1

    if dof == 1 and correction:
        # Adjust `observed` according to Yates' correction for continuity.
        observed = observed + 0.5 * np.sign(expected - observed)

    ddof = observed.size - 1 - dof
    stats = []
    names = ["pearson", "cressie-read", "log-likelihood",
             "freeman-tukey", "mod-log-likelihood", "neyman"]

    for name, lambda_ in zip(names, [1.0, 2 / 3, 0.0, -1 / 2, -1.0, -2.0]):
        if dof == 0:
            chi2, p = 0.0, 1.0
        else:
            chi2, p = power_divergence(observed, expected, ddof=ddof,
                                       axis=None, lambda_=lambda_)

        stats.append({'test': name, 'lambda': round(lambda_, 3),
                      'chi2': round(chi2, 3), 'dof': dof, 'p': p})

    stats = pd.DataFrame(stats)[['test', 'lambda', 'chi2', 'dof', 'p']]
    return expected, observed, stats


def chi2_mcnemar(data, procedure1, procedure2, correction=True):
    # Python code inspired by statsmodel's mcnemar
    procedures = (procedure1, procedure2)
    assert isinstance(data, pd.DataFrame), 'data must be a pandas DataFrame.'
    assert all(isinstance(procedure, str) for procedure in procedures),\
        'procedures must contain strings, only.'
    assert all(procedure in data.columns for procedure in procedures),\
        'columns are not in dataframe.'

    observed = dichotomous_crosstab(data, procedure1, procedure2)
    n1, n2 = observed.at[0, 1], observed.at[1, 0]

    # Exact test
    chi2 = min(n1, n2)
    exact = {
        'test': 'exact',
        'chi2': round(chi2, 3),
        'p': round(min(1, sp_stats.binom.cdf(chi2, n1 + n2, 0.5) * 2), 3)
    }

    # Approximated test
    chi2 = (abs(n1 - n2) - int(correction))**2 / (n1 + n2)
    approximated = {
        'test': 'approximated',
        'chi2': round(chi2, 3),
        'p': round(sp_stats.chi2.sf(chi2, 1), 3)
    }

    stats = pd.DataFrame([exact, approximated])
    stats['dof'] = 1
    stats = stats[['test', 'chi2', 'dof', 'p']]

    return observed, stats
