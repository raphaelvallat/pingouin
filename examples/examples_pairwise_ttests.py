import pandas as pd
from pingouin import pairwise_ttests, print_table


# Load a fake dataset: the INSOMNIA study
# Mixed repeated measures design
#   - Dependant variable (DV) = hours of sleep per night
#   - Between-factor = two-levels (Insomnia / Control)
#   - Within-factor = three levels (Pre, Post1, Post2)
df = pd.read_csv('sleep_dataset.csv')

# Pairwise T tests in a mixed-model ANOVA
# ---------------------------------------
stats = pairwise_ttests(dv='DV', within='Time', between='Group',
                        effects='all', data=df, alpha=.05,
                        tail='two-sided', padjust='fdr_by',
                        effsize='cohen', return_desc=True)
print_table(stats)
