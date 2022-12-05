import numpy as np
import pingouin as pg
data = pg.read_dataset('pairwise_corr')

fig = pg.plot_shift(data.Neuroticism, data.Conscientiousness,paired=True, n_boot=2000, 
                    percentiles=[25, 50, 75],show_median=False, seed=456, violin=False)
fig.axes[0].set_xlabel("Groups")
fig.axes[0].set_ylabel("Values",size=15)
fig.axes[0].set_title("Comparing of Neuroticism and  Conscientiousness",size=15)
fig.axes[1].set_xlabel("Neuroticism quantiles",size=12)



fig = pg.plot_shift(x, y, paired=True, n_boot=2000, percentiles=[25, 50, 75],
                    show_median=False, seed=456, violin=False)
