"""Pingouin to Pandas methods.

Authors
- Raphael Vallat <raphaelvallat9@gmail.com>
- Nicolas Legrand <nicolaslegrand21@gmail.com>
"""

import pandas as pd
import pingouin as pg


# ANOVA
def _anova(self, dv=None, between=None, detailed=False, export_filename=None):
    """Return One-way and two-way ANOVA."""
    aov = pg.anova(data=self, dv=dv, between=between, detailed=detailed,
                   export_filename=export_filename)
    return aov
pd.DataFrame.anova = _anova


# Welch ANOVA
def _welch_anova(self, dv=None, between=None, export_filename=None):
    """Return One-way WELCH ANOVA."""
    aov = pg.welch_anova(data=self, dv=dv, between=between,
                         export_filename=export_filename)
    return aov
pd.DataFrame.welch_anova = _welch_anova


# Repeated measures ANOVA
def _rm_anova(self, dv=None, within=None, subject=None, detailed=True,
              correction='auto', remove_na=True, export_filename=None):
    """One-way and two-way repeated measures ANOVA."""
    aov = pg.rm_anova(data=self, dv=dv, within=within, subject=subject,
                      correction='auto', remove_na=True, detailed=detailed,
                      export_filename=export_filename)
    return aov
pd.DataFrame.rm_anova = _rm_anova


# Post-hoc tests corrected for multiple-comparisons
def _pairwise_ttests(self, dv=None, within=None, subject=None, parametric=True,
                     padjust='fdr_bh', effsize='hedges'):
    """Post-hoc tests."""
    posthoc = pg.pairwise_ttests(data=self, dv=dv, within=within,
                                 subject=subject, parametric=parametric,
                                 padjust=padjust, effsize=effsize)
    return posthoc
pd.DataFrame.pairwise_ttests = _pairwise_ttests


# Two-way mixed ANOVA
def _mixed_anova(self, dv=None, between=None, within=None, subject=None,
                 correction=False, export_filename='mixed_anova'):
    """Two-way mixed ANOVA."""
    aov = pg.mixed_anova(data=self, dv=dv, between=between, within=within,
                         subject=subject, correction=correction,
                         export_filename=export_filename)
    return aov
pd.DataFrame.mixed_anova = _mixed_anova


# Pairwise correlations
def _pairwise_corr(self, columns=None, covar=None, tail='two-sided',
                   method='pearson', padjust='none', export_filename=None):
    """Pairwise (partial) correlations."""
    stats = pg.pairwise_corr(data=self, columns=columns, covar=covar,
                             tail=tail, method=method, padjust=padjust,
                             export_filename=export_filename)
    return stats
pd.DataFrame.pairwise_corr = _pairwise_corr


# Mediation analysis
def _mediation_analysis(self, x='X', m='Z', y='Y', seed=42, n_boot=1000):
    """Pairwise (partial) correlations."""
    stats = pg.mediation_analysis(data=self, x=x, m=m, y=y, seed=seed,
                                  n_boot=n_boot)
    return stats
pd.DataFrame.mediation_analysis = _mediation_analysis
