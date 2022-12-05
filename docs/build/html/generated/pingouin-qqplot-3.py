import numpy as np
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
np.random.seed(123)
x = np.random.normal(size=50)
mean, std = 0, 0.8
sns.set_style('darkgrid')
ax = pg.qqplot(x, dist='norm', sparams=(mean, std))
