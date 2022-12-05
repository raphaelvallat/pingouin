import pingouin as pg
ax = pg.plot_circmean([0.05, -0.8, 1.2, 0.8, 0.5, -0.3, 0.3, 0.7],
                      kwargs_markers=dict(color='k', mfc='k'),
                      kwargs_arrow=dict(ec='k', fc='k'))
