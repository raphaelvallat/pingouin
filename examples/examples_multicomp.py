import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr
from pingouin.multicomp import *

# Generate a fake array of p-values
np.random.seed(1234)
N = 6
pvals = np.random.rand(N)
pvals[[0]] *= 0.01
pvals[[3]] *= 0.1
print('Uncorrected p-values: ', pvals)

# Multiple comparisons
# -----------------------
reject, pvals_corrected = multicomp(pvals, alpha=.05, method='bonf')
print('Bonf-corrected p-values: ', pvals_corrected)

reject, pvals_corrected = multicomp(pvals, alpha=.05, method='fdr_bh')
print('FDR-corrected (BH) p-values: ', pvals_corrected)

reject, pvals_corrected = multicomp(pvals, alpha=.05, method='fdr_by')
print('FDR-corrected (BY) p-values: ', pvals_corrected)

reject, pvals_corrected = multicomp(pvals, alpha=.05, method='holm')
print('Holm-corrected p-values: ', pvals_corrected)
