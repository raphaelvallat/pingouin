import pingouin as pg
df = pg.read_dataset("blandaltman")
ax = pg.plot_blandaltman(df['A'], df['B'])
plt.tight_layout()
