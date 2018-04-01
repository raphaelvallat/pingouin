import numpy as np
from numpy.random import normal
import pandas as pd

# Load a fake dataset: the INSOMNIA study
# Goal: evaluate the influence of a treatment on sleep duration in a control
# and insomnia group
# Mixed repeated measures design
# - Dependant variable (DV) = hours of sleep per night
# - Between-factor = two-levels (Insomnia / Control)
# - Within-factor = four levels (Pre, Post1, Post2, Post3)

nx, ny, ngr, nrm = 20, 24, 2, 4
timepoints = ['Pre', 'Post-6months', 'Post-12months', 'Post-24months']
between = np.tile(np.r_[np.repeat(['Insomnia'], nx),
                        np.repeat(['Control'], ny)], nrm)
within = np.repeat(timepoints, (nx+ny))

# Create DV
i_pre = normal(loc=5, size=nx)
i_post1 = normal(loc=7.5, size=nx)
i_post2 = normal(loc=7.3, size=nx)
i_post3 = normal(loc=7.4, size=nx)
c_pre = normal(loc=7.8, size=ny)
c_post1 = normal(loc=7.9, size=ny)
c_post2 = normal(loc=7.85, size=ny)
c_post3 = normal(loc=7.9, size=ny)
hours_sleep = np.r_[i_pre, c_pre, i_post1, c_post1, i_post2, c_post2,
                    i_post3, c_post3]

df = pd.DataFrame({'DV': hours_sleep, 'Group': between, 'Time': within})

df.to_csv('sleep_dataset.csv', index=None)
