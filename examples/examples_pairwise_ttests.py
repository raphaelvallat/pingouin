import numpy as np
import pandas as pd
import pingouin as pg

# Generate a fake array of p-values
np.random.seed(123)
N = 20
x = np.random.normal(loc=174, size=N)
y = np.random.normal(loc=175, size=N)
df = pd.DataFrame({'DV': np.r_[x, y], 'Group': np.repeat(['X', 'Y'], N)})


# Pairwise T tests
# ----------------
tvals, pvals = pg.pairwise_ttests(dv='DV', between='Group', within=None, effects='all',
                                  data=df, paired=False, alpha=.05, tailed='two-sided',
                                  padjust=None)
print(tvals, pvals)
