import pingouin as pg
import seaborn as sns
df = pg.read_dataset('rm_corr')
sns.set(style='darkgrid', font_scale=1.2)
g = pg.plot_rm_corr(data=df, x='pH', y='PacO2',
                    subject='Subject', legend=True,
                    kwargs_facetgrid=dict(height=4.5, aspect=1.5,
                                          palette='Spectral'))
