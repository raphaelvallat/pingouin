import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr
from pingouin.effsize import *

# Generate a fake dataset: heights in countries x and y
np.random.seed(123)
N = 1000
x = np.random.normal(loc=172, scale=2, size=N)
y = np.random.normal(loc=175, scale=1.8, size=N)
df = pd.DataFrame({'DV': np.r_[x, y], 'Group': np.repeat(['X', 'Y'], N)})


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

# 3 - using a T-value
T, _ = ttest_ind(x, y)
ef = compute_effsize_from_T(T, nx=len(x), ny=len(y), eftype=eftype)
print(eftype, '(from T): %.3f' % ef)

# EFFECT SIZE CONVERSION
# -----------------------
# 1 - Convert r to Cohen's d
r, _ = pearsonr(x, y)
ef = convert_effsize(r, input_type='r', output_type='cohen')
print('r: ', r, '-', eftype, '(from r): %.3f' % ef)
