import pandas as pd
from numpy import nan
from pingouin import rm_anova, pairwise_ttests

# Change default display format of pandas
# pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load dataset
df = pd.read_csv('sleep_dataset.csv')

# Keep only insomnia group
df = df[df['Group'] == 'Insomnia']

# to make it trickier, let's assume that some random observations are missing
df.iloc[[4,33], 0] = nan

# Compute one-way repeated measures ANOVA
aov = rm_anova(dv='DV', within='Time', data=df, correction='auto')
print(aov)

# Compute pairwise post-hocs with effect size
post_hocs = pairwise_ttests(dv='DV', within='Time', data=df, effects='within',
                            padjust='bonf', effsize='hedges')
print(post_hocs)
