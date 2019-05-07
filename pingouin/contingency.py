# Author: Arthur Paulino <arthurleonardo.ap@gmail.com>
# Date: May 2019
from scipy.stats.contingency import expected_freq
from scipy.stats import power_divergence
import pandas as pd
import numpy as np
import warnings


__all__ = ['chi2']


def chi2(data, x, y, correction=True):
    """
    Chi-squared tests between two categorical variables for different values of
    :math:`\lambda`: 1, 2/3, 0, -1/2, -1 and -2 [1]_.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the ocurrences for the test.
    x, y : string
        The variables names for the Chi-squared test. Must be names of columns
        in ``data``.
    correction : bool
        Whether to apply Yates' correction when the degree of freedom of the
        observed contingency table is 1 [2]_.

    Returns
    -------
    expected : pd.DataFrame
        The expected contingency table of frequencies.
    observed : pd.DataFrame
        The (corrected or not) observed contingency table of frequencies.
    dof : float
        The degree of freedom of ``observed``.
    stats : pd.DataFrame
        The tests summary, containing three columns:

        * ``'lambda'``: The :math:`\lambda` value used for the power\
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
    high :math:`\chi^2` and low p-value. In the second example, similar
    categorical variables should present low :math:`\chi^2` and high p-value.

    .. warning :: As a general guideline for the consistency of this test, the
        observed and the expected contingency tables should not have cells
        with frequencies lower than 5.

    References
    ----------
    .. [1] Cressie, N., & Timothy R. C. Read. (1984). Multinomial
       Goodness-of-Fit Tests. Journal of the Royal Statistical Society. Series
       B (Methodological), 46(3), 440-464.

    .. [2] Yates, F. (1934). Contingency Tables Involving Small Numbers and the
       :math:`\chi^2` Test. Supplement to the Journal of the Royal Statistical
       Society, 1, 217-235.

    Examples
    --------
    Let's see if gender is a good categorical predictor for the presence of
    heart disease.

    >>> import pingouin as pg
    >>> data = pg.read_dataset('heart')
    >>> data['sex'].value_counts(ascending=True)
    0     96
    1    207
    Name: sex, dtype: int64

    If gender is not a good predictor for heart disease, we should expect the
    same 96:207 ratio across the target classes.

    >>> expected, observed, dof, stats = pg.chi2(data, 'sex', 'target')
    >>> expected
    target          0           1
    sex
    0       43.722772   52.277228
    1       94.277228  112.722772

    Let's see what the data tells us.

    >>> dof
    1

    The observed contingency table was adjusted. We should see fractional
    cells.

    >>> observed
    target      0     1
    sex
    0        24.5  71.5
    1       113.5  93.5

    The proportion is lower on the class 0 and higher on the class 1. The
    tests should be sensitive to this difference.

    >>> stats
         lambda       chi2             p
    0  1.000000  22.717227  1.876778e-06
    1  0.666667  22.931427  1.678845e-06
    2  0.000000  23.557374  1.212439e-06
    3 -0.500000  24.219622  8.595211e-07
    4 -1.000000  25.071078  5.525544e-07
    5 -2.000000  27.457956  1.605471e-07

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

    expected_zero_mask = expected == 0

    if expected_zero_mask.any(axis=None):
        expected_zero_mask_pos = [(i, j) for i in expected_zero_mask.index
                                  for j in expected_zero_mask.columns
                                  if expected_zero_mask.at[i, j]]
        raise ValueError(
            'Expected frequencies has value 0 for the following {} pairs:' +
            '\n\t{}'.format((x, y), expected_zero_mask_pos)
        )

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

    for lambda_ in [1.0, 2 / 3, 0.0, -1 / 2, -1.0, -2.0]:
        if dof == 0:
            chi2, p = 0.0, 1.0
        else:
            chi2, p = power_divergence(observed, expected, ddof=ddof,
                                       axis=None, lambda_=lambda_)

        stats.append({'lambda': lambda_, 'chi2': chi2, 'p': p})

    stats = pd.DataFrame(stats)[['lambda', 'chi2', 'p']]

    return expected, observed, dof, stats
