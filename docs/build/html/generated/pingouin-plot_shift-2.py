import numpy as np
import pingouin as pg
np.random.seed(42)
x = np.random.normal(5.5, 2, 30)
y = np.random.normal(6, 1.5, 30)
fig = pg.plot_shift(x, y, paired=True, n_boot=2000, percentiles=[25, 50, 75],
                    show_median=False, seed=456, violin=False)
