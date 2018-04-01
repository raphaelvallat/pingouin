import numpy as np
import pandas as pd
import pingouin as pg

# Generate a fake array of p-values
np.random.seed(123)
nx, ny = 10, 12
x = np.random.normal(loc=174, size=nx)
y = np.random.normal(loc=175, size=ny)
group = np.r_[np.repeat(['France'], nx), np.repeat(['UK'], ny)]
df = pd.DataFrame({'DV': np.r_[x, y], 'Group': group })


# Pairwise T tests
# ----------------

# WORK IN PROGRESS

tvals, pvals = pg.pairwise_ttests(dv='DV', between='Group', within=None, effects='between',
                                  data=df, alpha=.05, tailed='two-sided',
                                  padjust=None)
print(tvals, pvals)
