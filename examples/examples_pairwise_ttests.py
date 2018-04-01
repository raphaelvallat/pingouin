import numpy as np
from numpy.random import normal
import pandas as pd
import pingouin as pg

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
# Main effect of group (between)
stats = pg.pairwise_ttests(dv='DV', between='Group', effects='between', data=df,
                           alpha=.05, tailed='two-sided', padjust='none',
                           effsize='none')
print(stats)

# Main effect of measurement (within)
stats = pg.pairwise_ttests(dv='DV', within='Time', effects='within', data=df,
                           alpha=.05, tailed='two-sided', padjust='fdr_bh',
                           effsize='hedges')
print(stats)

# Interaction within * between
stats = pg.pairwise_ttests(dv='DV', within='Time', between='Group',
                           effects='interaction', data=df, alpha=.05,
                           tailed='two-sided', padjust='bonf', effsize='hedges')
print(stats)
