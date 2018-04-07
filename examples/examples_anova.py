"""One-way repeated measures ANOVA
"""
import pandas as pd
from numpy import nan, repeat
from pingouin import anova, rm_anova, mixed_anova, pairwise_ttests, print_table

# Trick to increase windows default size
import os
os.system('mode con: cols=200 lines=50')

###############################################################################
# ONE-WAY ANOVA
###############################################################################

# Generate a dataset
# https://en.wikipedia.org/wiki/One-way_analysis_of_variance#Example
df = pd.DataFrame({'Group': repeat(['A', 'B', 'C'], 6),
                   'DV': [6,8,4,5,3,4,8,12,9,11,6,8,13,9,11,8,7,12]
                   })

# Compute one-way ANOVA
aov = anova(dv='DV', between='Group', data=df, full_table=True)
print_table(aov, floatfmt=".3f")

# Compute pairwise post-hocs with effect size
post_hocs = pairwise_ttests(dv='DV', between='Group', data=df,
                            effects='between', padjust='bonf',
                            effsize='hedges')

# Print the table with 3 decimals
print_table(post_hocs, floatfmt=".2f")

###############################################################################
# ONE-WAY REPEATED MEASURES ANOVA
###############################################################################

# Load dataset
df = pd.read_csv('sleep_dataset.csv')

# Keep only insomnia group
df = df[df['Group'] == 'Insomnia']

# to make it trickier, let's assume that one subject has a missing value.
df.iloc[20, 0] = nan

# Compute one-way repeated measures ANOVA
aov = rm_anova(dv='DV', within='Time', data=df, correction='auto',
                full_table=True)
print_table(aov)

# Compute pairwise post-hocs with effect size
post_hocs = pairwise_ttests(dv='DV', within='Time', data=df, effects='within',
                            padjust='bonf', effsize='hedges')

# Print the table with 3 decimals
print_table(post_hocs, floatfmt=".3f")


###############################################################################
# TWO-WAY MIXED MODEL ANOVA (Within + Between factors)
###############################################################################

# WORK IN PROGRESS

# Load dataset
# df = pd.read_csv('sleep_dataset.csv')
# aov = mixed_anova(dv='DV', within='Time', between='Group', data=df)
# print_table(aov)
