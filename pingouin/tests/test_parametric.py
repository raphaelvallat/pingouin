import pandas as pd
import numpy as np

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.parametric import (gzscore, test_normality, ttest, anova, anova2,
                                 rm_anova, mixed_anova, test_dist)
from pingouin.datasets import read_dataset

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

df_nan = df.copy()
df_nan.iloc[[4, 15], 0] = np.nan

# Create random normal variables
np.random.seed(1234)
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
        test_normality(x, alpha=.05)
        test_normality(x, y, alpha=.05)

    # def test_test_homoscedasticity(self):
    #     """Test function test_homoscedasticity."""
    #     test_homoscedasticity(x, y, alpha=.05)
    #
    # def test_test_sphericity(self):
    #     """Test function test_sphericity."""
    #     test_sphericity(np.c_[x, y, z])

    def test_test_dist(self):
        """Test function test_dist."""
        test_dist(x)

    def test_ttest(self):
        """Test function ttest"""
        h = np.random.normal(scale=0.9, size=95)
        ttest(x, 0.5)
        stats = ttest(x, y, paired=True, tail='one-sided')
        # Compare with JASP
        assert np.allclose(stats.loc['T-test', 'T-val'], 0.616)
        assert np.allclose(stats.loc['T-test', 'p-val'].round(3), .270)
        ttest(x, y, paired=False, correction='auto')
        ttest(x, y, paired=False, correction=True)
        ttest(x, y, paired=False, r=0.5)
        ttest(x, h, paired=True)

    def test_anova(self):
        """Test function anova."""
        # Pain dataset
        df_pain = read_dataset('mcclave1991')
        aov = anova(dv='Pain threshold', between='Hair color', data=df_pain,
                    detailed=True)
        anova(dv='Pain threshold', between=['Hair color'], data=df_pain)
        # Compare with JASP
        assert np.allclose(aov.loc[0, 'F'].round(3), 6.791)
        assert np.allclose(aov.loc[0, 'p-unc'].round(3), .004)
        assert np.allclose(aov.loc[0, 'np2'].round(3), .576)
        # Two-way ANOVA
        anova(dv='Scores', between=['Group', 'Time'], data=df,
              export_filename='test_export.csv')
        anova2(dv='Scores', between=['Group', 'Time'], data=df)
        anova2(dv='Scores', between=['Group'], data=df)
        anova2(dv='Scores', between='Group', data=df)

    def test_rm_anova(self):
        """Test function anova."""
        rm_anova(dv='Scores', within='Time', data=df, correction=False,
                 detailed=False)
        rm_anova(dv='Scores', within='Time', data=df, correction=True,
                 detailed=False)
        aov = rm_anova(dv='Scores', within='Time', data=df, correction='auto',
                       detailed=True)
        # Compare with JASP
        assert np.allclose(aov.loc[0, 'F'].round(3), 3.913)
        assert np.allclose(aov.loc[0, 'p-unc'].round(3), .023)
        assert np.allclose(aov.loc[0, 'np2'].round(3), .062)

        rm_anova(dv='Scores', within='Time', data=df, correction=True,
                 detailed=True)
        rm_anova(dv='Scores', within='Time', data=df_nan,
                 export_filename='test_export.csv')

    def test_mixed_anova(self):
        """Test function anova."""
        aov = mixed_anova(dv='Scores', within='Time', between='Group', data=df,
                          correction='auto', remove_na=False)
        # Compare with JASP
        assert np.allclose(aov.loc[0, 'F'].round(3), 5.052)
        assert np.allclose(aov.loc[1, 'F'].round(3), 4.027)
        assert np.allclose(aov.loc[2, 'F'].round(3), 2.728)

        mixed_anova(dv='Scores', within='Time', between='Group', data=df_nan,
                    correction=True, remove_na=True,
                    export_filename='test_export.csv')
