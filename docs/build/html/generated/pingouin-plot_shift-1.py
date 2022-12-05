import numpy as np
import pingouin as pg
np.random.seed(42)
x = np.random.normal(5.5, 2, 50)
y = np.random.normal(6, 1.5, 50)
fig = pg.plot_shift(x, y)
