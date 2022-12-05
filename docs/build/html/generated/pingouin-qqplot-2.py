import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
np.random.seed(123)
x = np.random.normal(size=50)
x_exp = np.random.exponential(size=50)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1 = pg.qqplot(x, dist='norm', ax=ax1, confidence=False)
ax2 = pg.qqplot(x_exp, dist='expon', ax=ax2)
