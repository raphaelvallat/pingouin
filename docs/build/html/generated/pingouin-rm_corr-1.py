import pingouin as pg
df = pg.read_dataset('rm_corr')
g = pg.plot_rm_corr(data=df, x='pH', y='PacO2', subject='Subject')
