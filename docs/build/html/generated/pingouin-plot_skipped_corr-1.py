import numpy as np
import pingouin as pg
np.random.seed(123)
mean, cov, n = [170, 70], [[20, 10], [10, 20]], 30
x, y = np.random.multivariate_normal(mean, cov, n).T
# Introduce two outliers
x[10], y[10] = 160, 100
x[8], y[8] = 165, 90
fig = pg.plot_skipped_corr(x, y, xlabel='Height', ylabel='Weight')
