"""Convert Pingouin functions into Pandas methods.

Authors
- Raphael Vallat <raphaelvallat9@gmail.com>
- Nicolas Legrand <nicolaslegrand21@gmail.com>
"""
import numpy as np
import pandas as pd
from pingouin.multicomp import multicomp
from pingouin.correlation import partial_corr
from pingouin.parametric import (anova, welch_anova, rm_anova, mixed_anova)
from pingouin.pairwise import (pairwise_corr, pairwise_ttests)
from pingouin.regression import mediation_analysis


__all__ = ['pcorr', 'rcorr']


# ANOVA
def _anova(self, dv=None, between=None, ss_type=2, detailed=False,
           export_filename=None):
    """Return one-way and two-way ANOVA."""
    aov = anova(data=self, dv=dv, between=between, ss_type=ss_type,
                detailed=detailed, export_filename=export_filename)
    return aov


pd.DataFrame.anova = _anova


# Welch ANOVA
def _welch_anova(self, dv=None, between=None, export_filename=None):
    """Return one-way Welch ANOVA."""
    aov = welch_anova(data=self, dv=dv, between=between,
                      export_filename=export_filename)
    return aov


pd.DataFrame.welch_anova = _welch_anova


# Repeated measures ANOVA
def _rm_anova(self, dv=None, within=None, subject=None, detailed=False,
              correction='auto', export_filename=None):
    """One-way and two-way repeated measures ANOVA."""
    aov = rm_anova(data=self, dv=dv, within=within, subject=subject,
                   correction=correction, detailed=detailed,
                   export_filename=export_filename)
    return aov


pd.DataFrame.rm_anova = _rm_anova


# Post-hoc tests corrected for multiple-comparisons
def _pairwise_ttests(self, dv=None, between=None, within=None, subject=None,
                     parametric=True, alpha=.05, tail='two-sided',
                     padjust='none', effsize='hedges', return_desc=False,
                     export_filename=None):
    """Post-hoc tests."""
    posthoc = pairwise_ttests(data=self, dv=dv, between=between,
                              within=within, subject=subject,
                              parametric=parametric, alpha=alpha, tail=tail,
                              padjust=padjust, effsize=effsize,
                              return_desc=return_desc,
                              export_filename=export_filename)
    return posthoc


pd.DataFrame.pairwise_ttests = _pairwise_ttests


# Two-way mixed ANOVA
def _mixed_anova(self, dv=None, between=None, within=None, subject=None,
                 correction=False, export_filename=None):
    """Two-way mixed ANOVA."""
    aov = mixed_anova(data=self, dv=dv, between=between, within=within,
                      subject=subject, correction=correction,
                      export_filename=export_filename)
    return aov


pd.DataFrame.mixed_anova = _mixed_anova


# Pairwise correlations
def _pairwise_corr(self, columns=None, covar=None, tail='two-sided',
                   method='pearson', padjust='none', export_filename=None):
    """Pairwise (partial) correlations."""
    stats = pairwise_corr(data=self, columns=columns, covar=covar,
                          tail=tail, method=method, padjust=padjust,
                          export_filename=export_filename)
    return stats


pd.DataFrame.pairwise_corr = _pairwise_corr


# Partial correlation
def _partial_corr(self, x=None, y=None, covar=None, x_covar=None, y_covar=None,
                  tail='two-sided', method='pearson'):
    """Partial and semi-partial correlation."""
    stats = partial_corr(data=self, x=x, y=y, covar=covar, x_covar=x_covar,
                         y_covar=y_covar, tail=tail, method=method)
    return stats


pd.DataFrame.partial_corr = _partial_corr


# Partial correlation matrix
def pcorr(self):
    """Partial correlation matrix (:py:class:`pandas.DataFrame` method).

    Returns
    ----------
    pcormat : :py:class:`pandas.DataFrame`
        Partial correlation matrix.

    Notes
    -----
    This function calculates the pairwise partial correlations for each pair of
    variables in a :py:class:`pandas.DataFrame` given all the others. It has
    the same behavior as the pcor function in the `ppcor` R package.

    Note that this function only returns the raw Pearson correlation
    coefficient. If you want to calculate the test statistic and p-values, or
    use more robust estimates of the correlation coefficient, please refer to
    the :py:func:`pingouin.pairwise_corr` or :py:func:`pingouin.partial_corr`
    functions. The :py:func:`pingouin.pcorr` function uses the inverse of
    the variance-covariance matrix to calculate the partial correlation matrix
    and is therefore much faster than the two latter functions which are based
    on the residuals.

    References
    ----------
    .. [1] https://cran.r-project.org/web/packages/ppcor/index.html

    Examples
    --------
    >>> import pingouin as pg
    >>> data = pg.read_dataset('mediation')
    >>> data.pcorr()
                 X         M         Y      Mbin      Ybin
    X     1.000000  0.392251  0.059771 -0.014405 -0.149210
    M     0.392251  1.000000  0.545618 -0.015622 -0.094309
    Y     0.059771  0.545618  1.000000 -0.007009  0.161334
    Mbin -0.014405 -0.015622 -0.007009  1.000000 -0.076614
    Ybin -0.149210 -0.094309  0.161334 -0.076614  1.000000

    On a subset of columns

    >>> data[['X', 'Y', 'M']].pcorr().round(3)
           X      Y      M
    X  1.000  0.037  0.413
    Y  0.037  1.000  0.540
    M  0.413  0.540  1.000
    """
    V = self.cov()  # Covariance matrix
    Vi = np.linalg.pinv(V)  # Inverse covariance matrix
    D = np.diag(np.sqrt(1 / np.diag(Vi)))
    pcor = -1 * (D @ Vi @ D)  # Partial correlation matrix
    pcor[np.diag_indices_from(pcor)] = 1
    return pd.DataFrame(pcor, index=V.index, columns=V.columns)


pd.DataFrame.pcorr = pcorr


# Correlation matrix with p-values and sample size.
def rcorr(self, method='pearson', upper='pval', decimals=3, padjust=None,
          stars=True, pval_stars={0.001: '***', 0.01: '**', 0.05: '*'}):
    """
    Correlation matrix of a dataframe with p-values and/or sample size on the
    upper triangle (:py:class:`pandas.DataFrame` method).

    This method is a faster, but less exhaustive, matrix-version of the
    :py:func:`pingouin.pairwise_corr` function. It is based on the
    :py:func:`pandas.DataFrame.corr` method. Missing values are automatically
    removed from each pairwise correlation.

    Parameters
    ----------
    self : :py:class:`pandas.DataFrame`
        Input dataframe.
    method : str
        Correlation method. Can be either 'pearson' or 'spearman'.
    upper : str
        If 'pval', the upper triangle of the output correlation matrix shows
        the p-values. If 'n', the upper triangle is the sample size used in
        each pairwise correlation.
    decimals : int
        Number of decimals to display in the output correlation matrix.
    padjust : string or None
        Method used for adjustment of pvalues.
        Available methods are ::

        'none' : no correction
        'bonferroni' : one-step Bonferroni correction
        'holm' : step-down method using Bonferroni adjustments
        'fdr_bh' : Benjamini/Hochberg FDR correction
        'fdr_by' : Benjamini/Yekutieli FDR correction
    stars : boolean
        If True, only significant p-values are displayed as stars using the
        pre-defined thresholds of ``pval_stars``. If False, all the raw
        p-values are displayed.
    pval_stars : dict
        Significance thresholds. Default is 3 stars for p-values < 0.001,
        2 stars for p-values < 0.01 and 1 star for p-values < 0.05.

    Returns
    -------
    rcorr : :py:class:`pandas.DataFrame`
        Correlation matrix, of type str.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import pingouin as pg
    >>> # Load an example dataset of personality dimensions
    >>> df = pg.read_dataset('pairwise_corr').iloc[:, 1:]
    >>> # Add some missing values
    >>> df.iloc[[2, 5, 20], 2] = np.nan
    >>> df.iloc[[1, 4, 10], 3] = np.nan
    >>> df.head().round(2)
       Neuroticism  Extraversion  Openness  Agreeableness  Conscientiousness
    0         2.48          4.21      3.94           3.96               3.46
    1         2.60          3.19      3.96            NaN               3.23
    2         2.81          2.90       NaN           2.75               3.50
    3         2.90          3.56      3.52           3.17               2.79
    4         3.02          3.33      4.02            NaN               2.85

    >>> # Correlation matrix on the four first columns
    >>> df.iloc[:, 0:4].rcorr()
                  Neuroticism Extraversion Openness Agreeableness
    Neuroticism             -          ***                     **
    Extraversion        -0.35            -      ***
    Openness            -0.01        0.265        -           ***
    Agreeableness      -0.134        0.054    0.161             -

    >>> # Spearman correlation and Holm adjustement for multiple comparisons
    >>> df.iloc[:, 0:4].rcorr(method='spearman', padjust='holm')
                  Neuroticism Extraversion Openness Agreeableness
    Neuroticism             -          ***                     **
    Extraversion       -0.325            -      ***
    Openness           -0.027         0.24        -           ***
    Agreeableness       -0.15         0.06    0.173             -

    >>> # Compare with the pg.pairwise_corr function
    >>> pairwise = df.iloc[:, 0:4].pairwise_corr(method='spearman',
    ...                                          padjust='holm')
    >>> pairwise[['X', 'Y', 'r', 'p-corr']].round(3)  # Do not show all columns
                  X              Y      r  p-corr
    0   Neuroticism   Extraversion -0.325   0.000
    1   Neuroticism       Openness -0.027   0.543
    2   Neuroticism  Agreeableness -0.150   0.002
    3  Extraversion       Openness  0.240   0.000
    4  Extraversion  Agreeableness  0.060   0.358
    5      Openness  Agreeableness  0.173   0.000

    >>> # Display the raw p-values with four decimals
    >>> df.iloc[:, [0, 1, 3]].rcorr(stars=False, decimals=4)
                  Neuroticism Extraversion Agreeableness
    Neuroticism             -       0.0000        0.0028
    Extraversion      -0.3501            -        0.2305
    Agreeableness      -0.134       0.0539             -

    >>> # With the sample size on the upper triangle instead of the p-values
    >>> df.iloc[:, [0, 1, 2]].rcorr(upper='n')
                 Neuroticism Extraversion Openness
    Neuroticism            -          500      497
    Extraversion       -0.35            -      497
    Openness           -0.01        0.265        -
    """
    from numpy import triu_indices_from as tif
    from numpy import format_float_positional as ffp
    from scipy.stats import pearsonr, spearmanr

    # Safety check
    assert isinstance(pval_stars, dict), 'pval_stars must be a dictionnary.'
    assert isinstance(decimals, int), 'decimals must be an int.'
    assert method in ['pearson', 'spearman'], 'Method is not recognized.'
    assert upper in ['pval', 'n'], 'upper must be either `pval` or `n`.'
    mat = self.corr(method=method).round(decimals)
    if upper == 'n':
        mat_upper = self.corr(method=lambda x, y: len(x)).astype(int)
    else:
        if method == 'pearson':
            mat_upper = self.corr(method=lambda x, y: pearsonr(x, y)[1])
        else:
            # Method = 'spearman'
            mat_upper = self.corr(method=lambda x, y: spearmanr(x, y)[1])

        if padjust is not None:
            pvals = mat_upper.values[tif(mat, k=1)]
            mat_upper.values[tif(mat, k=1)] = multicomp(pvals, alpha=0.05,
                                                        method=padjust)[1]

    # Convert r to text
    mat = mat.astype(str)
    np.fill_diagonal(mat.values, '-')  # Inplace modification of the diagonal

    if upper == 'pval':

        def replace_pval(x):
            for key, value in pval_stars.items():
                if x < key:
                    return value
            return ''

        if stars:
            # Replace p-values by stars
            mat_upper = mat_upper.applymap(replace_pval)
        else:
            mat_upper = mat_upper.applymap(lambda x: ffp(x,
                                                         precision=decimals))

    # Replace upper triangle by p-values or n
    mat.values[tif(mat, k=1)] = mat_upper.values[tif(mat, k=1)]
    return mat


pd.DataFrame.rcorr = rcorr


# Mediation analysis
def _mediation_analysis(self, x=None, m=None, y=None, covar=None,
                        alpha=0.05, n_boot=500, seed=None, return_dist=False):
    """Mediation analysis."""
    stats = mediation_analysis(data=self, x=x, m=m, y=y, covar=covar,
                               alpha=alpha, n_boot=n_boot, seed=seed,
                               return_dist=return_dist)
    return stats


pd.DataFrame.mediation_analysis = _mediation_analysis
