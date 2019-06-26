"""Convert Pingouin functions into Pandas methods.

Authors
- Raphael Vallat <raphaelvallat9@gmail.com>
- Nicolas Legrand <nicolaslegrand21@gmail.com>
"""
import numpy as np
import pandas as pd
from pingouin.correlation import partial_corr
from pingouin.parametric import (anova, welch_anova, rm_anova, mixed_anova)
from pingouin.pairwise import (pairwise_corr, pairwise_ttests)
from pingouin.regression import mediation_analysis

__all__ = ['pcorr']


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


# Mediation analysis
def _mediation_analysis(self, x=None, m=None, y=None, covar=None,
                        alpha=0.05, n_boot=500, seed=None, return_dist=False):
    """Mediation analysis."""
    stats = mediation_analysis(data=self, x=x, m=m, y=y, covar=covar,
                               alpha=alpha, n_boot=n_boot, seed=seed,
                               return_dist=return_dist)
    return stats


pd.DataFrame.mediation_analysis = _mediation_analysis
