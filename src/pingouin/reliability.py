import numpy as np
import pandas as pd
from scipy.stats import f

from .config import options
from .utils import _postprocess_dataframe

__all__ = ["cronbach_alpha", "intraclass_corr"]


def cronbach_alpha(
    data=None, items=None, scores=None, subject=None, nan_policy="pairwise", ci=0.95
):
    """Cronbach's alpha reliability measure.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Wide or long-format dataframe.
    items : str
        Column in ``data`` with the items names (long-format only).
    scores : str
        Column in ``data`` with the scores (long-format only).
    subject : str
        Column in ``data`` with the subject identifier (long-format only).
    nan_policy : bool
        If `'listwise'`, remove the entire rows that contain missing values
        (= listwise deletion). If `'pairwise'` (default), only pairwise
        missing values are removed when computing the covariance matrix.
        For more details, please refer to the :py:meth:`pandas.DataFrame.cov`
        method.
    ci : float
        Confidence interval (.95 = 95%)

    Returns
    -------
    alpha : float
        Cronbach's alpha

    Notes
    -----
    This function works with both wide and long format dataframe. If you pass a
    long-format dataframe, you must also pass the ``items``, ``scores`` and
    ``subj`` columns (in which case the data will be converted into wide
    format using the :py:meth:`pandas.DataFrame.pivot` method).

    Internal consistency is usually measured with Cronbach's alpha [1]_,
    a statistic calculated from the pairwise correlations between items.
    Internal consistency ranges between negative infinity and one.
    Coefficient alpha will be negative whenever there is greater
    within-subject variability than between-subject variability.

    Cronbach's :math:`\\alpha` is defined as

    .. math::

        \\alpha ={k \\over k-1}\\left(1-{\\sum_{{i=1}}^{k}\\sigma_{{y_{i}}}^{2}
        \\over\\sigma_{x}^{2}}\\right)

    where :math:`k` refers to the number of items, :math:`\\sigma_{x}^{2}`
    is the variance of the observed total scores, and
    :math:`\\sigma_{{y_{i}}}^{2}` the variance of component :math:`i` for
    the current sample of subjects.

    Another formula for Cronbach's :math:`\\alpha` is

    .. math::

        \\alpha = \\frac{k \\times \\bar c}{\\bar v + (k - 1) \\times \\bar c}

    where :math:`\\bar c` refers to the average of all covariances between
    items and :math:`\\bar v` to the average variance of each item.

    95% confidence intervals are calculated using Feldt's method [2]_:

    .. math::

        c_L = 1 - (1 - \\alpha) \\cdot F_{(0.025, n-1, (n-1)(k-1))}

        c_U = 1 - (1 - \\alpha) \\cdot F_{(0.975, n-1, (n-1)(k-1))}

    where :math:`n` is the number of subjects and :math:`k` the number of
    items.

    Results have been tested against the `psych
    <https://cran.r-project.org/web/packages/psych/psych.pdf>`_ R package.

    References
    ----------
    .. [1] http://www.real-statistics.com/reliability/cronbachs-alpha/

    .. [2] Feldt, Leonard S., Woodruff, David J., & Salih, Fathi A. (1987).
           Statistical inference for coefficient alpha. Applied Psychological
           Measurement, 11(1):93-103.

    Examples
    --------
    Binary wide-format dataframe (with missing values)

    >>> import pingouin as pg
    >>> data = pg.read_dataset("cronbach_wide_missing")
    >>> # In R: psych:alpha(data, use="pairwise")
    >>> pg.cronbach_alpha(data=data)
    (0.732660835214447, array([0.435, 0.909]))

    After listwise deletion of missing values (remove the entire rows)

    >>> # In R: psych:alpha(data, use="complete.obs")
    >>> pg.cronbach_alpha(data=data, nan_policy="listwise")
    (0.8016949152542373, array([0.581, 0.933]))

    After imputing the missing values with the median of each column

    >>> pg.cronbach_alpha(data=data.fillna(data.median()))
    (0.7380191693290734, array([0.447, 0.911]))

    Likert-type long-format dataframe

    >>> data = pg.read_dataset("cronbach_alpha")
    >>> pg.cronbach_alpha(data=data, items="Items", scores="Scores", subject="Subj")
    (0.5917188485995826, array([0.195, 0.84 ]))
    """
    # Safety check
    assert isinstance(data, pd.DataFrame), "data must be a dataframe."
    assert nan_policy in ["pairwise", "listwise"]

    if all([v is not None for v in [items, scores, subject]]):
        # Data in long-format: we first convert to a wide format
        data = data.pivot(index=subject, values=scores, columns=items)

    # From now we assume that data is in wide format
    n, k = data.shape
    assert k >= 2, "At least two items are required."
    assert n >= 2, "At least two raters/subjects are required."
    err = "All columns must be numeric."
    assert all([data[c].dtype.kind in "bfiu" for c in data.columns]), err
    if data.isna().any().any() and nan_policy == "listwise":
        # In R = psych:alpha(data, use="complete.obs")
        data = data.dropna(axis=0, how="any")

    # Compute covariance matrix and Cronbach's alpha
    C = data.cov(numeric_only=True)
    cronbach = (k / (k - 1)) * (1 - np.trace(C) / C.sum().sum())
    # which is equivalent to
    # v = np.diag(C).mean()
    # c = C.to_numpy()[np.tril_indices_from(C, k=-1)].mean()
    # cronbach = (k * c) / (v + (k - 1) * c)

    # Confidence intervals
    alpha = 1 - ci
    df1 = n - 1
    df2 = df1 * (k - 1)
    lower = 1 - (1 - cronbach) * f.isf(alpha / 2, df1, df2)
    upper = 1 - (1 - cronbach) * f.isf(1 - alpha / 2, df1, df2)
    return cronbach, np.round([lower, upper], 3)


def intraclass_corr(data=None, targets=None, raters=None, ratings=None, nan_policy="raise"):
    """
    Compute intraclass correlation (ICC) coefficients to assess measurement reliability.

    This function provides six variants of the ICC to evaluate how consistently targets
    (e.g., patients, samples) are rated across different measurements (e.g., raters, days).
    It follows the practical guidance of Liljequist et al. (2019) [2]_, which suggests
    calculating all ICC types together rather than picking a single statistical
    model upfront.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Long-format dataframe containing the targets, raters, and scores.
    targets : string
        Name of the column containing the subjects or items being measured.
    raters : string
        Name of the column containing the raters, sessions, or conditions.
    ratings : string
        Name of the column containing the numerical scores or values.
    nan_policy : str
        Defines how to handle when input contains missing values (nan).
        `'raise'` (default) throws an error, `'omit'` performs the calculations
        after deleting target(s) with one or more missing values (= listwise
        deletion).

        .. versionadded:: 0.3.0

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`
        Summary table with one row per ICC variant, containing:

        * ``'Type'``: ICC variant (e.g., ``ICC(1,1)``, ``ICC(A,1)``, ``ICC(C,1)``).
        * ``'ICC'``: the intraclass correlation coefficient.
        * ``'F'``, ``'pval'``: F-test results for detecting systematic differences (bias) between
          raters. A significant p-value suggests that rater means differ, indicating non-negligible
          bias.
        * ``'df1'``, ``'df2'``: degrees of freedom for the F-test.
        * ``'CI95'``: 95% confidence interval for the ICC.

    Notes
    -----
    The ICC measures the ratio of between-target variance to total variance [1]_.
    It reflects how consistently targets (e.g., patients, samples) are measured
    across raters or sessions, with values typically ranging from 0 (no
    reliability) to 1 (perfect reliability).

    Pingouin follows the notation of Liljequist et al. (2019) [2]_, based on
    McGraw and Wong (1996) [3]_. Six ICC variants are returned, organized along
    two dimensions.

    **How bias is handled (first index):**

    - **ICC(1,)**: Assumes raters are interchangeable with no systematic
      bias. Only valid when rater means are roughly equal.
    - **ICC(A,)**: Absolute agreement. Penalises systematic differences
      between raters: "do raters give the same scores?"
    - **ICC(C,)**: Consistency. Ignores systematic differences between
      raters: "do raters rank targets in the same order?"
      Equivalent to the reliability expected if rater biases were removed.

    **Single or averaged scores (second index):**

    - **1**: Reliability of one rating by a single rater.
    - **k**: Reliability of the mean across all :math:`k` raters

    **Practical guidance**:

    Liljequist et al. (2019) recommend computing all three single-score ICCs
    together and comparing them, rather than selecting a single model upfront.

    *1. Detecting bias (systematic errors):*

    Start by comparing ICC(1,1), ICC(A,1), and ICC(C,1). When they are
    approximately equal, systematic bias between raters is likely negligible.
    When ICC(C,1) is notably larger than ICC(A,1), non-negligible bias is
    likely present. In that case, ICC(1,1) is invalid and should not be
    reported. The F statistic and p-value in the output provide a formal test
    of whether rater means differ significantly.

    *2. Agreement vs. consistency:*

    When bias is present, both ICC(A,1) and ICC(C,1) should be reported
    together with their confidence intervals. ICC(A,1) reflects absolute
    agreement (do raters assign the same values?), while ICC(C,1) reflects
    consistency (do raters rank targets in the same order?).

    *3. Single vs. average ratings:*

    Use the single-score variants (ICC(1,1), ICC(A,1), ICC(C,1)) when
    reporting the reliability of one rating. Use the average-score variants
    (ICC(1,k), ICC(A,k), ICC(C,k)) when the final measurement will be the
    mean of :math:`k` ratings.

    **Interpretation guidelines:**

    General benchmarks for ICC values:

    * < 0.50: Poor
    * 0.50 - 0.75: Moderate
    * 0.75 - 0.90: Good
    * > 0.90: Excellent

    Whether a given ICC is acceptable depends on the intended clinical or
    practical context, not on these thresholds alone.

    This function has been validated against the ICC function of the R psych
    package. The current implementation uses ANOVA rather than linear mixed
    effects models and requires complete, balanced data.

    References
    ----------
    .. [1] http://www.real-statistics.com/reliability/intraclass-correlation/

    .. [2] Liljequist, D., Elfving, B., & Skavberg Roaldsen, K. (2019).
           Intraclass correlation - A discussion and demonstration of basic
           features. PLOS ONE, 14(7), e0219854.

    .. [3] McGraw, K. O., & Wong, S. P. (1996). Forming inferences about some
           intraclass correlation coefficients. Psychological Methods, 1(1),
           30-46.

    Examples
    --------
    ICCs of wine quality assessed by 4 judges.

    >>> import pingouin as pg
    >>> data = pg.read_dataset("icc")
    >>> icc = pg.intraclass_corr(data=data, targets="Wine", raters="Judge", ratings="Scores").round(
    ...     3
    ... )
    >>> icc.set_index("Type")
              ICC       F  df1  df2  pval          CI95
    Type
    ICC(1,1)  0.728  11.680    7   24   0.0  [0.43, 0.93]
    ICC(A,1)  0.728  11.787    7   21   0.0  [0.43, 0.93]
    ICC(C,1)  0.729  11.787    7   21   0.0  [0.43, 0.93]
    ICC(1,k)  0.914  11.680    7   24   0.0  [0.75, 0.98]
    ICC(A,k)  0.914  11.787    7   21   0.0  [0.75, 0.98]
    ICC(C,k)  0.915  11.787    7   21   0.0  [0.75, 0.98]
    """
    from pingouin import anova

    # Safety check
    assert isinstance(data, pd.DataFrame), "data must be a dataframe."
    assert all([v is not None for v in [targets, raters, ratings]])
    assert all([v in data.columns for v in [targets, raters, ratings]])
    assert nan_policy in ["omit", "raise"]

    # Convert data to wide-format
    data = data.pivot_table(index=targets, columns=raters, values=ratings, observed=True)

    # Listwise deletion of missing values
    nan_present = data.isna().any().any()
    if nan_present:
        if nan_policy == "omit":
            data = data.dropna(axis=0, how="any")
        else:
            raise ValueError(
                "Either missing values are present in data or "
                "data are unbalanced. Please remove them "
                "manually or use nan_policy='omit'."
            )

    # Back to long-format
    # data_wide = data.copy()  # Optional, for PCA
    data = data.reset_index().melt(id_vars=targets, value_name=ratings)

    # Check that ratings is a numeric variable
    assert data[ratings].dtype.kind in "bfiu", "Ratings must be numeric."
    # Check that data are fully balanced
    # This behavior is ensured by the long-to-wide-to-long transformation
    # Unbalanced data will result in rows with missing values.
    # assert data.groupby(raters)[ratings].count().nunique() == 1

    # Extract sizes
    k = data[raters].nunique()
    n = data[targets].nunique()

    # Two-way ANOVA
    with np.errstate(invalid="ignore"):
        # For max precision, make sure rounding is disabled
        old_options = options.copy()
        options["round"] = None
        aov = anova(data=data, dv=ratings, between=[targets, raters], ss_type=2)
        options.update(old_options)  # restore options

    # Extract mean squares
    msb = aov.at[0, "MS"]
    msw = (aov.at[1, "SS"] + aov.at[2, "SS"]) / (aov.at[1, "DF"] + aov.at[2, "DF"])
    msj = aov.at[1, "MS"]
    mse = aov.at[2, "MS"]

    # Calculate ICCs
    icc1 = (msb - msw) / (msb + (k - 1) * msw)
    icc2 = (msb - mse) / (msb + (k - 1) * mse + k * (msj - mse) / n)
    icc3 = (msb - mse) / (msb + (k - 1) * mse)
    icc1k = (msb - msw) / msb
    icc2k = (msb - mse) / (msb + (msj - mse) / n)
    icc3k = (msb - mse) / msb

    # Calculate F, df, and p-values
    f1k = msb / msw
    df1 = n - 1
    df1kd = n * (k - 1)
    p1k = f.sf(f1k, df1, df1kd)

    f2k = f3k = msb / mse
    df2kd = (n - 1) * (k - 1)
    p2k = f.sf(f2k, df1, df2kd)

    # Create output dataframe
    stats = {
        "Type": ["ICC(1,1)", "ICC(A,1)", "ICC(C,1)", "ICC(1,k)", "ICC(A,k)", "ICC(C,k)"],
        "ICC": [icc1, icc2, icc3, icc1k, icc2k, icc3k],
        "F": [f1k, f2k, f2k, f1k, f2k, f2k],
        "df1": n - 1,
        "df2": [df1kd, df2kd, df2kd, df1kd, df2kd, df2kd],
        "pval": [p1k, p2k, p2k, p1k, p2k, p2k],
    }

    stats = pd.DataFrame(stats)

    # Calculate confidence intervals
    alpha = 0.05
    # Case 1 and 3
    f1l = f1k / f.ppf(1 - alpha / 2, df1, df1kd)
    f1u = f1k * f.ppf(1 - alpha / 2, df1kd, df1)
    l1 = (f1l - 1) / (f1l + (k - 1))
    u1 = (f1u - 1) / (f1u + (k - 1))
    f3l = f3k / f.ppf(1 - alpha / 2, df1, df2kd)
    f3u = f3k * f.ppf(1 - alpha / 2, df2kd, df1)
    l3 = (f3l - 1) / (f3l + (k - 1))
    u3 = (f3u - 1) / (f3u + (k - 1))
    # Case 2
    fj = msj / mse
    vn = df2kd * (k * icc2 * fj + n * (1 + (k - 1) * icc2) - k * icc2) ** 2
    vd = df1 * k**2 * icc2**2 * fj**2 + (n * (1 + (k - 1) * icc2) - k * icc2) ** 2
    v = vn / vd
    f2u = f.ppf(1 - alpha / 2, n - 1, v)
    f2l = f.ppf(1 - alpha / 2, v, n - 1)
    l2 = n * (msb - f2u * mse) / (f2u * (k * msj + (k * n - k - n) * mse) + n * msb)
    u2 = n * (f2l * msb - mse) / (k * msj + (k * n - k - n) * mse + n * f2l * msb)

    stats["CI95"] = [
        np.array([l1, u1]),
        np.array([l2, u2]),
        np.array([l3, u3]),
        np.array([1 - 1 / f1l, 1 - 1 / f1u]),
        np.array([l2 * k / (1 + l2 * (k - 1)), u2 * k / (1 + u2 * (k - 1))]),
        np.array([1 - 1 / f3l, 1 - 1 / f3u]),
    ]

    return _postprocess_dataframe(stats)
