import pingouin as pg
df = pg.read_dataset('mixed_anova').query("Time != 'January'")
df = df.query("Group == 'Control'")
ax = pg.plot_paired(data=df, dv='Scores', within='Time',
                    subject='Subject', boxplot_in_front=True)
