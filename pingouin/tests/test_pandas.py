"""Test pingouin to Pandas methods.

Authors
- Nicolas Legrand <nicolaslegrand21@gmail.com>
- Raphael Vallat <raphaelvallat9@gmail.com>
"""
import pingouin
import numpy as np
import pandas as pd
from pingouin.datasets import read_dataset

df = read_dataset('mixed_anova')

# Test the ANOVA (Pandas)
df.anova(dv='Scores', between='Group', detailed=True)

# Test the Welch ANOVA (Pandas)
df.welch_anova(dv='Scores', between='Group')

# Test the repeated measures ANOVA (Pandas)
df.rm_anova(dv='Scores', within='Time', subject='Subject', detailed=True)

# FDR-corrected post hocs with Hedges'g effect size
df.pairwise_ttests(dv='Scores', within='Time', subject='Subject',
                   parametric=True, padjust='fdr_bh', effsize='hedges')

# Test two-way mixed ANOVA
df.mixed_anova(dv='Scores', between='Group', within='Time', subject='Subject',
               correction=False, export_filename='mixed_anova.csv')

# Test parwise correlations
np.random.seed(123)
mean, cov, n = [4, 5], [(1, .6), (.6, 1)], 30
x, y = np.random.multivariate_normal(mean, cov, n).T
z = np.random.normal(5, 1, 30)
data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
data.pairwise_corr(columns=['X', 'Y', 'Z'])

# Test mediation Analysis
data.mediation_analysis(x='X', m='Z', y='Y', seed=42, n_boot=1000)
