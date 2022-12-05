import pingouin as pg
df = pg.read_dataset('mixed_anova').query("Time != 'January'")
df = df.query("Group == 'Meditation' and Subject > 40")
ax = pg.plot_paired(data=df, dv='Scores', within='Time',
                    subject='Subject', dpi=150)
