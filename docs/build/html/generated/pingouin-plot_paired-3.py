import pingouin as pg
import matplotlib.pyplot as plt
df = pg.read_dataset('mixed_anova').query("Group == 'Meditation'")
# df = df.query("Group == 'Meditation' and Subject > 40")
pg.plot_paired(data=df, dv='Scores', within='Time',
               subject='Subject', orient='h')  # doctest: +SKIP
