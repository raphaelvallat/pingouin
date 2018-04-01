import numpy as np
from numpy.random import normal
import pandas as pd
from pingouin import pairwise_ttests

# Change default display format of pandas
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Generate a fake dataset
# Mixed repeated measures design
# - DV = hours of sleep per night
# - Between = Insomnia (n=10) / Control (n=12)
# - Within = Pre-treatment / Post-treatment
nx, ny, ngr, nrm = 10, 12, 2, 3

between = np.tile(np.r_[np.repeat(['Insomnia'], nx),
                        np.repeat(['Control'], ny)], nrm)
within = np.repeat(['Pre', 'Post-6months', 'Post-12months'], (nx+ny))

# Create DV
i_pre = normal(loc=4, size=nx)
i_posta = normal(loc=7.5, size=nx)
i_postb = normal(loc=7.2, size=nx)
c_pre = normal(loc=7.8, size=ny)
c_posta = normal(loc=7.9, size=ny)
c_postb = normal(loc=7.85, size=ny)
hours_sleep = np.r_[i_pre, c_pre, i_posta, c_posta, i_postb, c_postb]

df = pd.DataFrame({'DV': hours_sleep,
                   'Group': between,
                   'Time': within})
# print(df)

# Pairwise T tests
# ----------------
# 1 - Main effect of group (between)
stats = pairwise_ttests(dv='DV', between='Group', effects='between', data=df,
                        alpha=.05, tail='two-sided', padjust='none',
                        effsize='none')
print(stats)

# 2 - Main effect of measurement (within)
stats = pairwise_ttests(dv='DV', within='Time', effects='within', data=df,
                        alpha=.05, tail='two-sided', padjust='fdr_bh',
                        effsize='hedges')
print(stats)

# 3 - Interaction within * between
stats = pairwise_ttests(dv='DV', within='Time', between='Group',
                        effects='interaction', data=df, alpha=.05,
                        tail='two-sided', padjust='bonf', effsize='eta-square')
print(stats)

# 4 - all = return all three above
stats = pairwise_ttests(dv='DV', within='Time', between='Group',
                        effects='all', data=df, alpha=.05,
                        tail='two-sided', padjust='fdr_by', effsize='hedges')
print(stats)
