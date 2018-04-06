import pandas as pd
from pingouin import pairwise_ttests, print_table
import os
os.system('mode con: cols=200 lines=40')

# Load a fake dataset: the INSOMNIA study
# Goal: evaluate the influence of a treatment on sleep duration in a control
# and insomnia group
#
# Mixed repeated measures design
#   - Dependant variable (DV) = hours of sleep per night
#   - Between-factor = two-levels (Insomnia / Control)
#   - Within-factor = four levels (Pre, Post1, Post2, Post3)
df = pd.read_csv('sleep_dataset.csv')

# Pairwise T tests
# ----------------
# 1 - Main effect of group (between)
# stats = pairwise_ttests(dv='DV', between='Group', effects='between', data=df,
#                         alpha=.05, tail='two-sided', padjust='none',
#                         effsize='none')
# print(stats)

# 2 - Main effect of measurement (within)
# stats = pairwise_ttests(dv='DV', within='Time', effects='within', data=df,
#                         alpha=.05, tail='two-sided', padjust='fdr_bh',
#                         effsize='hedges')
# print(stats)

# 3 - Interaction within * between
# stats = pairwise_ttests(dv='DV', within='Time', between='Group',
#                         effects='interaction', data=df, alpha=.05,
#                         tail='two-sided', padjust='bonf', effsize='eta-square')
# print(stats)

# 4 - all = return all three above
stats = pairwise_ttests(dv='DV', within='Time', between='Group',
                        effects='all', data=df, alpha=.05,
                        tail='two-sided', padjust='fdr_by',
                        effsize='cohen', return_desc=False)
print_table(stats)
