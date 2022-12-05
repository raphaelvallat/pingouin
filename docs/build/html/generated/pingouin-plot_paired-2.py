import pingouin as pg
import matplotlib.pyplot as plt
df = pg.read_dataset('mixed_anova').query("Time != 'January'")
df = df.query("Group == 'Meditation' and Subject > 40")
fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
pg.plot_paired(data=df, dv='Scores', within='Time',
               subject='Subject', ax=ax1, boxplot=False,
               colors=['grey', 'grey', 'grey'])  # doctest: +SKIP
