import pandas as pd
import numpy as np

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.parametric import (gzscore, test_normality, anova, rm_anova,
                                 mixed_anova)

# Generate random data for ANOVA
n = 30
months = ['August', 'January', 'June']
np.random.seed(1234)
control = np.random.normal(5.5, size=len(months) * n)
meditation = np.r_[np.random.normal(5.5, size=n),
                   np.random.normal(5.8, size=n),
                   np.random.normal(6.4, size=n)]
# Create a dataframe
df = pd.DataFrame({'Scores': np.r_[control, meditation],
                   'Time': np.r_[np.repeat(months, n), np.repeat(months, n)],
                   'Group': np.repeat(['Control', 'Meditation'],
                                      len(months) * n)})

# Create random normal variables
x = np.random.normal(scale=1., size=100)
y = np.random.normal(scale=0.8, size=100)
z = np.random.normal(scale=0.9, size=100)


class TestParametric(_TestPingouin):
    """Test parametric.py."""

    def test_gzscore(self):
        """Test function gzscore."""
        raw = np.random.lognormal(size=100)
        gzscore(raw)

    def test_test_normality(self):
        """Test function test_normality."""
        normal, p = test_normality(x, alpha=.05)
        normal, p = test_normality(x, y, alpha=.05)

    # def test_test_homoscedasticity(self):
    #     """Test function test_homoscedasticity."""
    #     equal_var, p = test_homoscedasticity(x, y, alpha=.05)
    #
    #
    # def test_test_sphericity(self):
    #     """Test function test_sphericity."""
    #     sphericity, W, chi_sq, ddof, p = test_sphericity(np.c_[x, y, z])

    def test_anova(self):
        """Test function anova."""
        anova(dv='Scores', between='Group', data=df, detailed=True)
        anova(dv='Scores', between='Group', data=df, detailed=False)

    def test_rm_anova(self):
        """Test function anova."""
        rm_anova(dv='Scores', within='Time', data=df, correction=False,
                 remove_na=False, detailed=False)
        rm_anova(dv='Scores', within='Time', data=df, correction='auto',
                 remove_na=True, detailed=True)
        rm_anova(dv='Scores', within='Time', data=df, correction=True,
                 remove_na=True, detailed=True)

    def test_mixed_anova(self):
        """Test function anova."""
        mixed_anova(dv='Scores', within='Time', between='Group', data=df,
                    correction='auto', remove_na=False)
        df.iloc[4, 0] = np.nan
        mixed_anova(dv='Scores', within='Time', between='Group', data=df,
                    correction=True, remove_na=True)
