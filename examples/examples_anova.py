"""One-way repeated measures ANOVA
"""
import pandas as pd
from numpy import repeat
from pingouin import anova, rm_anova, mixed_anova, print_table

# Load dataset
df = pd.read_csv('sleep_dataset.csv')

# ONE-WAY ANOVA
aov = anova(dv='DV', between='Group', data=df, detailed=False)
print_table(aov, floatfmt=".3f")

# ONE-WAY REPEATED MEASURES ANOVA
aov = rm_anova(dv='DV', within='Time', data=df, correction='auto',
               remove_na=True, detailed=True)
print_table(aov)

# TWO-WAY MIXED MODEL ANOVA (Within + Between factors)
aov = mixed_anova(dv='DV', within='Time', between='Group', data=df,
                  correction='auto', export_filename='mixed_anova.csv')
print_table(aov)
