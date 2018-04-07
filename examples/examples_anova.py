"""One-way ANOVA
"""
import pandas as pd
from numpy import repeat
from pingouin import anova, pairwise_ttests, print_table

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
