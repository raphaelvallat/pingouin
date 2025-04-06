# Date: May 2019
import warnings
import numpy as np
import pandas as pd

from scipy.stats.contingency import expected_freq
from scipy.stats import power_divergence, binom, chi2 as sp_chi2

from pingouin import power_chi2, _postprocess_dataframe


__all__ = ["chi2_independence", "chi2_mcnemar", "dichotomous_crosstab", "ransacking"]


###############################################################################
# CHI-SQUARED TESTS
###############################################################################


def chi2_independence(data, x, y, correction=True):
    """
    Chi-squared independence tests between two categorical variables.

    The test is computed for different values of :math:`\\lambda`: 1, 2/3, 0,
    -1/2, -1 and -2 (Cressie and Read, 1984).

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        The dataframe containing the ocurrences for the test.
    x, y : string
        The variables names for the Chi-squared test. Must be names of columns
        in ``data``.
    correction : bool
        Whether to apply Yates' correction when the degree of freedom of the
        observed contingency table is 1 (Yates 1934).

    Returns
    -------
    expected : :py:class:`pandas.DataFrame`
        The expected contingency table of frequencies.
    observed : :py:class:`pandas.DataFrame`
        The (corrected or not) observed contingency table of frequencies.
    stats : :py:class:`pandas.DataFrame`
        The test summary, containing four columns:

        * ``'test'``: The statistic name
        * ``'lambda'``: The :math:`\\lambda` value used for the power\
                        divergence statistic
        * ``'chi2'``: The test statistic
        * ``'pval'``: The p-value of the test
        * ``'cramer'``: The Cramer's V effect size
        * ``'power'``: The statistical power of the test

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
    * Cressie, N., & Read, T. R. (1984). Multinomial goodness‐of‐fit
      tests. Journal of the Royal Statistical Society: Series B
      (Methodological), 46(3), 440-464.

    * Yates, F. (1934). Contingency Tables Involving Small Numbers and the
      :math:`\\chi^2` Test. Supplement to the Journal of the Royal
      Statistical Society, 1, 217-235.

    Examples
    --------
    Let's see if gender is a good categorical predictor for the presence of
    heart disease.

    >>> import pingouin as pg
    >>> data = pg.read_dataset('chi2_independence')
    >>> data['sex'].value_counts(ascending=True)
    sex
    0     96
    1    207
    Name: count, dtype: int64

    If gender is not a good predictor for heart disease, we should expect the
    same 96:207 ratio across the target classes.

    >>> expected, observed, stats = pg.chi2_independence(data, x='sex',
    ...                                                  y='target')
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

    >>> stats.round(3)
                     test  lambda    chi2  dof  pval  cramer  power
    0             pearson   1.000  22.717  1.0   0.0   0.274  0.997
    1        cressie-read   0.667  22.931  1.0   0.0   0.275  0.998
    2      log-likelihood   0.000  23.557  1.0   0.0   0.279  0.998
    3       freeman-tukey  -0.500  24.220  1.0   0.0   0.283  0.998
    4  mod-log-likelihood  -1.000  25.071  1.0   0.0   0.288  0.999
    5              neyman  -2.000  27.458  1.0   0.0   0.301  0.999

    Very low p-values indeed. The gender qualifies as a good predictor for the
    presence of heart disease on this dataset.
    """
    # Python code inspired by SciPy's chi2_contingency
    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame."
    assert isinstance(x, (str, int)), "x must be a string or int."
    assert isinstance(y, (str, int)), "y must be a string or int."
    assert all(col in data.columns for col in (x, y)), "columns are not in dataframe."
    assert isinstance(correction, bool), "correction must be a boolean."

    observed = pd.crosstab(data[x], data[y])

    if observed.size == 0:
        raise ValueError("No data; observed has size 0.")

    expected = pd.DataFrame(expected_freq(observed), index=observed.index, columns=observed.columns)

    # All count frequencies should be at least 5
    for df, name in zip([observed, expected], ["observed", "expected"]):
        if (df < 5).any(axis=None):
            warnings.warn(f"Low count on {name} frequencies.")

    dof = float(expected.size - sum(expected.shape) + expected.ndim - 1)

    if dof == 1 and correction:
        # Adjust `observed` according to Yates' correction for continuity.
        observed = observed + 0.5 * np.sign(expected - observed)

    ddof = observed.size - 1 - dof
    n = data.shape[0]
    stats = []
    names = [
        "pearson",
        "cressie-read",
        "log-likelihood",
        "freeman-tukey",
        "mod-log-likelihood",
        "neyman",
    ]

    for name, lambda_ in zip(names, [1.0, 2 / 3, 0.0, -1 / 2, -1.0, -2.0]):
        if dof == 0:
            chi2, p, cramer, power = 0.0, 1.0, np.nan, np.nan
        else:
            chi2, p = power_divergence(observed, expected, ddof=ddof, axis=None, lambda_=lambda_)
            dof_cramer = min(expected.shape) - 1
            cramer = np.sqrt(chi2 / (n * dof_cramer))
            power = power_chi2(dof=dof, w=cramer, n=n, alpha=0.05)

        stats.append(
            {
                "test": name,
                "lambda": lambda_,
                "chi2": chi2,
                "dof": dof,
                "pval": p,
                "cramer": cramer,
                "power": power,
            }
        )

    stats = pd.DataFrame(stats)[["test", "lambda", "chi2", "dof", "pval", "cramer", "power"]]
    return expected, observed, _postprocess_dataframe(stats)


def chi2_mcnemar(data, x, y, correction=True):
    """
    Performs the exact and approximated versions of McNemar's test.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        The dataframe containing the ocurrences for the test. Each row must
        represent either a subject or a pair of subjects.
    x, y : string
        The variables names for the McNemar's test. Must be names of columns
        in ``data``.

        If each row of ``data`` represents a subject, then ``x`` and ``y`` must
        be columns containing dichotomous measurements in two different
        contexts. For instance: the presence of pain before and after a certain
        treatment.

        If each row of ``data`` represents a pair of subjects, then ``x`` and
        ``y`` must be columns containing dichotomous measurements for each of
        the subjects. For instance: a positive response to a certain drug in
        the control group and in the test group, supposing that each pair
        contains a subject in each group.

        The 2x2 crosstab is created using the
        :py:func:`pingouin.dichotomous_crosstab` function.

        .. warning:: Missing values are not allowed.

    correction : bool
        Whether to apply the correction for continuity (Edwards, A. 1948).

    Returns
    -------
    observed : :py:class:`pandas.DataFrame`
        The observed contingency table of frequencies.
    stats : :py:class:`pandas.DataFrame`
        The test summary:

        * ``'chi2'``: The test statistic
        * ``'dof'``: The degree of freedom
        * ``'p_approx'``: The approximated p-value
        * ``'p_exact'``: The exact p-value

    Notes
    -----
    The McNemar's test is compatible with dichotomous paired data, generally
    used to assert the effectiveness of a certain procedure, such as a
    treatment or the use of a drug. "Dichotomous" means that the values of the
    measurements are binary. "Paired data" means that each measurement is done
    twice, either on the same subject in two different moments or in two
    similar (paired) subjects from different groups (e.g.: control/test). In
    order to better understand the idea behind McNemar's test, let's illustrate
    it with an example.

    Suppose that we wanted to compare the effectiveness of two different
    treatments (X and Y) for athlete's foot on a certain group of `n` people.
    To achieve this, we measured their responses to such treatments on each
    foot. The observed data summary was:

    * Number of people with good responses to X and Y: `a`
    * Number of people with good response to X and bad response to Y: `b`
    * Number of people with bad response to X and good response to Y: `c`
    * Number of people with bad responses to X and Y: `d`

    Now consider the two groups:

    1. The group of people who had good response to X (`a` + `b` subjects)
    2. The group of people who had good response to Y (`a` + `c` subjects)

    If the treatments have the same effectiveness, we should expect the
    probabilities of having good responses to be the same, regardless of the
    treatment. Mathematically, such statement can be translated into the
    following equation:

    .. math::

        \\frac{a+b}{n} = \\frac{a+c}{n} \\Rightarrow b = c

    Thus, this test should indicate higher statistical significances for higher
    distances between `b` and `c` (McNemar, Q. 1947):

    .. math::

        \\chi^2 = \\frac{(b - c)^2}{b + c}

    References
    ----------
    * Edwards, A. L. (1948). Note on the "correction for continuity" in
      testing the significance of the difference between correlated
      proportions. Psychometrika, 13(3), 185-187.

    * McNemar, Q. (1947). Note on the sampling error of the difference
      between correlated proportions or percentages. Psychometrika, 12(2),
      153-157.

    Examples
    --------
    >>> import pingouin as pg
    >>> data = pg.read_dataset('chi2_mcnemar')
    >>> observed, stats = pg.chi2_mcnemar(data, 'treatment_X', 'treatment_Y')
    >>> observed
    treatment_Y   0   1
    treatment_X
    0            20  40
    1             8  12

    In this case, `c` (40) seems to be a significantly greater than `b` (8).
    The McNemar test should be sensitive to this.

    >>> stats
                chi2  dof  p_approx   p_exact
    mcnemar  20.020833    1  0.000008  0.000003
    """
    # Python code initially inspired by statsmodel's mcnemar
    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame."
    assert all(isinstance(column, (str, int)) for column in (x, y)), (
        "column names must be string or int."
    )
    assert all(column in data.columns for column in (x, y)), "columns are not in dataframe."

    for column in (x, y):
        if data[column].isna().any():
            raise ValueError("Null values are not allowed.")

    observed = dichotomous_crosstab(data, x, y)
    # Careful, the order of b and c is inverted compared to wikipedia
    # because the colums / rows of the crosstab is [0, 1] and not [1, 0].
    c, b = observed.at[0, 1], observed.at[1, 0]
    n_discordants = b + c

    if (b, c) == (0, 0):
        raise ValueError(
            "McNemar's test does not work if the secondary "
            + "diagonal of the observed data summary does not "
            + "have values different from 0."
        )

    chi2 = (abs(b - c) - int(correction)) ** 2 / n_discordants
    pexact = min(1, 2 * binom.cdf(min(b, c), n_discordants, 0.5))
    stats = {
        "chi2": chi2,
        "dof": 1,
        "p_approx": sp_chi2.sf(chi2, 1),
        "p_exact": pexact,
        # 'p_mid': pexact - binom.pmf(b, n_discordants, 0.5)
    }

    stats = pd.DataFrame(stats, index=["mcnemar"])

    return observed, _postprocess_dataframe(stats)


###############################################################################
# DICHOTOMOUS CONTINGENCY TABLES
###############################################################################


def _dichotomize_series(data, column):
    """Converts the values of a pd.DataFrame column into 0 or 1"""
    series = data[column]
    if series.dtype == bool:
        return series.astype(int)

    def convert_elem(elem):
        if isinstance(elem, (int, float)) and elem in (0, 1):
            return int(elem)
        if isinstance(elem, str):
            lower = elem.lower()
            if lower in ("n", "no", "absent", "false", "f", "negative"):
                return 0
            elif lower in ("y", "yes", "present", "true", "t", "positive", "p"):
                return 1
        raise ValueError(
            "Invalid value to build a 2x2 contingency table on column {}: {}".format(column, elem)
        )

    return series.apply(convert_elem)


def dichotomous_crosstab(data, x, y):
    """
    Generates a 2x2 contingency table from a :py:class:`pandas.DataFrame` that
    contains only dichotomous entries, which are converted to 0 or 1.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Pandas dataframe
    x, y : string
        Column names in ``data``.

        Currently, Pingouin recognizes the following values as dichotomous
        measurements:

        * ``0``, ``0.0``, ``False``, ``'No'``, ``'N'``, ``'Absent'``,\
        ``'False'``, ``'F'`` or ``'Negative'`` for negative cases;

        * ``1``, ``1.0``, ``True``, ``'Yes'``, ``'Y'``, ``'Present'``,\
        ``'True'``, ``'T'``, ``'Positive'`` or ``'P'``,  for positive cases;

        If strings are used, Pingouin will recognize them regardless of their
        uppercase/lowercase combinations.

    Returns
    -------
    crosstab : :py:class:`pandas.DataFrame`
        The 2x2 crosstab. See :py:func:`pandas.crosstab` for more details.

    Examples
    --------
    >>> import pandas as pd
    >>> import pingouin as pg
    >>> df = pd.DataFrame({'A': ['Yes', 'No', 'No'], 'B': [0., 1., 0.]})
    >>> pg.dichotomous_crosstab(data=df, x='A', y='B')
    B  0  1
    A
    0  1  1
    1  1  0
    """
    crosstab = pd.crosstab(_dichotomize_series(data, x), _dichotomize_series(data, y))
    shape = crosstab.shape
    if shape != (2, 2):
        if shape == (2, 1):
            crosstab.loc[:, int(not bool(crosstab.columns[0]))] = [0, 0]
        elif shape == (1, 2):
            crosstab.loc[int(not bool(crosstab.index[0])), :] = [0, 0]
        else:  # shape = (1, 1) or shape = (>2, >2)
            raise ValueError(
                "Both series contain only one unique value. Cannot build 2x2 contingency table."
            )
    crosstab = crosstab.sort_index(axis=0).sort_index(axis=1)
    return crosstab


###############################################################################
# RANSACKING POST-HOC ANALYSIS
###############################################################################


def ransacking(data, row_var, col_var, alpha=0.05, adjusted=False):
    r"""
    Perform ransacking post-hoc analysis of a larger \(r \times c\) contingency table 
    by focusing on each \(2 \times 2\) subtable of interest and testing whether that 
    subtable exhibits an "interaction" (i.e., departure from independence).

    This implementation follows the method described by Sharpe (2015) and others:

    1. **Log Odds Ratio** \((G)\):
       \[
         G = \ln \Bigl(\frac{\text{odds}_{\text{row1}}}{\text{odds}_{\text{row2}}}\Bigr),
         \quad \text{where } \text{odds}_{\text{row1}} 
         = \frac{\text{cell}(1,1)}{\text{cell}(1,2)}.
       \]

    2. **Standard Error** \((\text{SE}_G)\) is approximated by:
       \[
         \sqrt{\frac{1}{n_{11}} + \frac{1}{n_{12}} + \frac{1}{n_{21}} + \frac{1}{n_{22}}}
       \]
       where \(n_{ij}\) is the observed count in row \(i\), column \(j\) of the \(2 \times 2\) subtable.

    3. **\(Z\)-value**:
       \[
         Z = \frac{G}{\text{SE}_G}.
       \]

    4. **Critical \(Z\)-value** from the global degrees of freedom \(\text{dof} = (r-1)\,(c-1)\).
       - We compute \(\chi^2_{(1-\alpha,\ \text{dof})}\) from the chi-square distribution,
         then use 
         \[
           Z_{\alpha} = \sqrt{\chi^2_{(1-\alpha,\ \text{dof})}}.
         \]
       - This matches the post-hoc logic in Sharpe (2015). For example, in a \(2 \times 3\)
         table, \(\text{dof}=2\), so the critical \(Z\) is 
         \(\sqrt{\chi^2_{(1-\alpha=0.95,\ 2)}} \approx 2.45\).

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        The dataframe containing the data.
    row_var : str
        The name of the row variable.
    col_var : str
        The name of the column variable.
    alpha : float, optional
        Significance level for the test. Default is 0.05.
    adjusted : bool, optional
        Whether to apply a Bonferroni-like correction (divide alpha by the number 
        of cells) to maintain a familywise error rate. Default is False.

    Returns
    -------
    :py:class:`pandas.DataFrame`
        A dataframe with the ransacking test results for each cell, including:

        - **Row** (label)
        - **Column** (label)
        - **Odds Ratio** (ratio of two odds)
        - **Log Odds Ratio** (\(G\))
        - **Standard Error** of \(G\)
        - **Z Value** (test statistic)
        - **Critical Z (global dof)** (unadjusted)
        - **Adjusted Critical Z (global dof)** (Bonferroni-like if ``adjusted=True``)
        - **Unadjusted Result** (reject/fail to reject)
        - **Adjusted Result** (reject/fail to reject)
        - **2x2 Table** (the subtable as a list of lists)
        - **DOF** (global, \((r - 1)(c - 1)\))

    Notes
    -----
    - Sharpe (2015) illustrates an example where a \(2 \times 3\) table yields \(\text{dof}=2\).
      Post-hoc selection of a \(2 \times 2\) subtable uses
      \(\sqrt{\chi^2_{\text{critical},\ 2}}\) as the cutoff for the \(Z\)-test, 
      rather than \(\chi^2_{(1-\alpha,\ 1)}\).
    - If you prefer each \(2 \times 2\) subtable be tested against \(\text{dof}=1\), 
      adapt the code to use `dof_cell = 1` instead. However, Sharpe's example 
      shows why using the full table's \((r - 1)(c - 1)\) may be more appropriate 
      for post-hoc analyses.
    - If multiple sub-tests are performed, consider alpha adjustment (e.g., Bonferroni)
      to curb type I error inflation.

    .. rubric:: Example usage

    .. code-block:: python

       >>> from pingouin import read_dataset
       >>> df = read_dataset('chi2_independence')
       >>> results = ransacking(data=df, row_var='cp', col_var='restecg', alpha=0.05, adjusted=True)
       >>> results  # doctest: +SKIP

    .. code-block:: none

       Row,Column,Odds Ratio,Log Odds Ratio,Standard Error,Z Value,Critical Z (global dof),Adjusted Critical Z (global dof),Unadjusted Result,Adjusted Result,2x2 Table,DOF
       0,0,1.583,0.459,0.232,1.981,3.548,4.359,fail to reject,fail to reject,"[[78.0, 65.0], [69.0, 91.0]]",6
       0,1,0.595,-0.519,0.232,-2.234,3.548,4.359,fail to reject,fail to reject,"[[62.0, 81.0], [90.0, 70.0]]",6
       0,2,3.407,1.226,1.161,1.056,3.548,4.359,fail to reject,fail to reject,"[[3.0, 140.0], [1.0, 159.0]]",6
       1,0,0.599,-0.513,0.317,-1.617,3.548,4.359,fail to reject,fail to reject,"[[19.0, 31.0], [128.0, 125.0]]",6
       1,1,1.78,0.577,0.317,1.817,3.548,4.359,fail to reject,fail to reject,"[[31.0, 19.0], [121.0, 132.0]]",6
       1,2,0.0,-23.026,100000.0,-0.0,3.548,4.359,fail to reject,fail to reject,"[[0.0, 50.0], [4.0, 249.0]]",6
       2,0,0.668,-0.404,0.257,-1.573,3.548,4.359,fail to reject,fail to reject,"[[36.0, 51.0], [111.0, 105.0]]",6
       2,1,1.51,0.412,0.256,1.61,3.548,4.359,fail to reject,fail to reject,"[[50.0, 37.0], [102.0, 114.0]]",6
       2,2,0.826,-0.192,1.162,-0.165,3.548,4.359,fail to reject,fail to reject,"[[1.0, 86.0], [3.0, 213.0]]",6
       3,0,1.719,0.542,0.444,1.221,3.548,4.359,fail to reject,fail to reject,"[[14.0, 9.0], [133.0, 147.0]]",6
       3,1,0.616,-0.485,0.444,-1.093,3.548,4.359,fail to reject,fail to reject,"[[9.0, 14.0], [143.0, 137.0]]",6
       3,2,0.0,-23.026,100000.0,-0.0,3.548,4.359,fail to reject,fail to reject,"[[0.0, 23.0], [4.0, 276.0]]",6

    References
    ----------
    Sharpe, D. (2015). Your chi-square test is statistically significant: Now what?
    *Practical assessment, research & evaluation*, *20*(8), n8.

    """
    # Add assertions for input validation
    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
    assert isinstance(alpha, (int, float)), "alpha must be a number"
    assert 0 < alpha < 1, "alpha must be between 0 and 1"
    assert isinstance(adjusted, bool), "adjusted must be a boolean"
    
    try:
        freq_table = pd.crosstab(data[row_var], data[col_var])
        total = freq_table.values.sum()

        # Number of sub-tests and alpha adjustment
        num_tests = freq_table.shape[0] * freq_table.shape[1]
        adjusted_alpha = alpha / num_tests if adjusted else alpha

        # Global DOF for the full table (r-1)*(c-1)
        r = freq_table.shape[0]
        c = freq_table.shape[1]
        dof = (r - 1) * (c - 1)

        # Compute critical z-values using the global dof
        critical_z = np.sqrt(sp_chi2.ppf(1 - alpha, dof))
        adjusted_critical_z = np.sqrt(sp_chi2.ppf(1 - adjusted_alpha, dof))

        results = []

        # Iterate over each cell in the contingency table
        for (row_label, col_label), cell_value in freq_table.stack().items():
            row_total = freq_table.loc[row_label].sum() - cell_value
            col_total = freq_table[col_label].sum() - cell_value
            remaining_total = total - cell_value - row_total - col_total

            # Build the 2x2 sub-table
            table_2x2 = np.array([
                [cell_value,         row_total],
                [col_total, remaining_total]
            ], dtype=float)

            # Small epsilon to avoid division-by-zero
            epsilon = 1e-10
            odds_1 = table_2x2[0, 0] / (table_2x2[0, 1] + epsilon)
            odds_2 = table_2x2[1, 0] / (table_2x2[1, 1] + epsilon)

            # Odds ratio and log odds ratio
            odds_ratio = odds_1 / (odds_2 + epsilon)
            log_odds_ratio = np.log(odds_ratio + epsilon)

            # Standard error for the log odds ratio (SE_G)
            standard_error = np.sqrt(np.sum(1.0 / (table_2x2 + epsilon)))

            # Z-value for the log odds ratio
            z_value = log_odds_ratio / (standard_error + epsilon)

            # Evaluate significance against global critical z
            unadjusted_result = "reject" if abs(z_value) > critical_z else "fail to reject"
            adjusted_result = "reject" if abs(z_value) > adjusted_critical_z else "fail to reject"

            results.append({
                'Row': row_label,
                'Column': col_label,
                'Odds Ratio': odds_ratio,
                'Log Odds Ratio': log_odds_ratio,
                'Standard Error': standard_error,
                'Z Value': z_value,
                'Critical Z (global dof)': critical_z,
                'Adjusted Critical Z (global dof)': adjusted_critical_z,
                'Unadjusted Result': unadjusted_result,
                'Adjusted Result': adjusted_result,
                '2x2 Table': table_2x2.tolist(),
                'DOF': dof  # global dof
            })

        # Convert results to a DataFrame
        df_results = pd.DataFrame(results)

        # Round numeric columns (except integer DOF) to 3 decimals
        numeric_cols = [
            'Odds Ratio', 'Log Odds Ratio', 'Standard Error', 'Z Value',
            'Critical Z (global dof)', 'Adjusted Critical Z (global dof)'
        ]
        for col in numeric_cols:
            df_results[col] = df_results[col].round(3)

        # Round each entry in the 2x2 Table to 3 decimals
        def round_2x2(tbl):
            return [[round(x, 3) for x in row] for row in tbl]

        df_results['2x2 Table'] = df_results['2x2 Table'].apply(round_2x2)

        return df_results

    except KeyError as e:
        raise KeyError(f"Column not found in the dataframe: {e}")
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"Division by zero encountered: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")
