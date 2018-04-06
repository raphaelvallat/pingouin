import pandas as pd
from numpy import nan
from pingouin import rm_anova, pairwise_ttests, print_table

# Trick to increase windows default size
import os
os.system('mode con: cols=200 lines=40')

# Load dataset
df = pd.read_csv('sleep_dataset.csv')

# Keep only insomnia group
df = df[df['Group'] == 'Insomnia']

# to make it trickier, let's assume that one subject has a missing value.
df.iloc[20, 0] = nan

# Compute one-way repeated measures ANOVA
aov = rm_anova(dv='DV', within='Time', data=df, correction='auto',
                full_table=False)
print_table(aov)

# Compute pairwise post-hocs with effect size
post_hocs = pairwise_ttests(dv='DV', within='Time', data=df, effects='within',
                            padjust='bonf', effsize='hedges')

# Print the table with 3 decimals
print_table(post_hocs, floatfmt=".3f")
