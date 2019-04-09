"""Convert Pingouin functions into Pandas methods.

Authors
- Raphael Vallat <raphaelvallat9@gmail.com>
- Nicolas Legrand <nicolaslegrand21@gmail.com>
"""
import pandas as pd
from pingouin.correlation import partial_corr
from pingouin.parametric import (anova, welch_anova, rm_anova, mixed_anova)
from pingouin.pairwise import (pairwise_corr, pairwise_ttests)
from pingouin.regression import mediation_analysis


# ANOVA
def _anova(self, dv=None, between=None, detailed=False, export_filename=None):
    """Return one-way and two-way ANOVA."""
    aov = anova(data=self, dv=dv, between=between, detailed=detailed,
                export_filename=export_filename)
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
def _rm_anova(self, dv=None, within=None, subject=None, detailed=True,
              correction='auto', remove_na=True, export_filename=None):
    """One-way and two-way repeated measures ANOVA."""
    aov = rm_anova(data=self, dv=dv, within=within, subject=subject,
                   correction=correction, remove_na=remove_na,
                   detailed=detailed, export_filename=export_filename)
    return aov


pd.DataFrame.rm_anova = _rm_anova


# Post-hoc tests corrected for multiple-comparisons
def _pairwise_ttests(self, dv=None, within=None, subject=None, parametric=True,
                     padjust='fdr_bh', effsize='hedges'):
    """Post-hoc tests."""
    posthoc = pairwise_ttests(data=self, dv=dv, within=within,
                              subject=subject, parametric=parametric,
                              padjust=padjust, effsize=effsize)
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
def _partial_corr(self, x=None, y=None, covar=None, tail='two-sided',
                  method='pearson'):
    """Pairwise (partial) correlations."""
    stats = partial_corr(data=self, x=x, y=y, covar=covar, tail=tail,
                         method=method)
    return stats


pd.DataFrame.partial_corr = _partial_corr


# Mediation analysis
def _mediation_analysis(self, x=None, m=None, y=None, covar=None,
                        alpha=0.05, n_boot=500, seed=None, return_dist=False):
    """Mediation analysis."""
    stats = mediation_analysis(data=self, x=x, m=m, y=y, covar=covar,
                               alpha=alpha, n_boot=n_boot, seed=seed,
                               return_dist=return_dist)
    return stats


pd.DataFrame.mediation_analysis = _mediation_analysis
