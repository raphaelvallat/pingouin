import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr
from pingouin.effsize import *

# Generate a fake dataset: heights in countries x and y
np.random.seed(123)
nx, ny = 100, 120
x = np.random.normal(loc=174, size=nx)
y = np.random.normal(loc=175, size=ny)
group = np.r_[np.repeat(['France'], nx), np.repeat(['UK'], ny)]
df = pd.DataFrame({'DV': np.r_[x, y], 'Group': group })


# EFFECT SIZE COMPUTATION
# -----------------------
# Define the output eftype
# Possible values are 'cohen', 'hedges', 'r', 'eta-square', 'odds-ratio', 'AUC'
eftype = 'hedges'

# 1 - using bonf
ef = compute_effsize(dv='DV', group='Group', data=df, eftype=eftype)
print(eftype, ': %.3f' % ef)

# 2 - using Numpy
ef = compute_effsize(x=x, y=y, eftype=eftype)
print(eftype, ': %.3f' % ef)

# 3 - using a T-value when nx and ny are known
T, _ = ttest_ind(x, y)
ef = compute_effsize_from_T(T, nx=len(x), ny=len(y), eftype=eftype)
print(eftype, '(from T - nx + ny): %.3f' % ef)

# 3 - using a T-value when only total sample size is known
T, _ = ttest_ind(x, y)
ef = compute_effsize_from_T(T, N=len(x) + len(y), eftype='cohen')
print('cohen (from T - only N): %.3f' % ef)

# EFFECT SIZE CONVERSION
# -----------------------
# 1 - Convert r to Cohen's d
n = 100
x = np.random.normal(loc=174, size=n)
y = np.random.normal(loc=175, size=n)
df = pd.DataFrame({'DV': np.r_[x, y], 'Group': np.repeat(['France', 'UK'], n)})
r, _ = pearsonr(x, y)
eftype = 'cohen'
ef = convert_effsize(r, input_type='r', output_type=eftype)
print('r: : %.3f' % r, 'to', eftype, ': %.3f' % ef)
