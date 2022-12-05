import pingouin as pg
import seaborn as sns
sns.set(font_scale=1.5, style='white')
ax = pg.plot_circmean([0.8, 1.5, 3.14, 5.2, 6.1, 2.8, 2.6, 3.2],
                      kwargs_markers=dict(marker="None"))
