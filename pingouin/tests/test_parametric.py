import pandas as pd
import numpy as np

from unittest import TestCase
from pingouin.parametric import (ttest, anova, anova2, rm_anova, mixed_anova,
                                 rm_anova2, ancova, welch_anova, ancovan)
from pingouin import read_dataset

# Generate random data for ANOVA
df = read_dataset('mixed_anova.csv')

df_nan = df.copy()
df_nan.iloc[[4, 15], 0] = np.nan

# Create random normal variables
np.random.seed(1234)
x = np.random.normal(scale=1., size=100)
y = np.random.normal(scale=0.8, size=100)


class TestParametric(TestCase):
    """Test parametric.py."""

    def test_ttest(self):
        """Test function ttest"""
        h = np.random.normal(scale=0.9, size=95)
        ttest(x, 0.5)
        stats = ttest(x, y, paired=True, tail='one-sided')
        # Compare with JASP
        assert np.allclose(stats.loc['T-test', 'T'], 0.616)
        assert np.allclose(stats.loc['T-test', 'p-val'].round(3), .270)
        ttest(x, y, paired=False, correction='auto')
        ttest(x, y, paired=False, correction=True)
        ttest(x, y, paired=False, r=0.5)
        ttest(x, h, paired=True)
        # Compare with R t.test
        a = [4, 7, 8, 6, 3, 2]
        b = [6, 8, 7, 10, 11, 9]
        tt = ttest(a, b, paired=False, correction=False, tail='two-sided')
        assert tt.loc['T-test', 'T'] == -2.842
        assert tt.loc['T-test', 'dof'] == 10
        assert round(tt.loc['T-test', 'p-val'], 5) == 0.01749
        np.testing.assert_allclose(tt.loc['T-test', 'CI95%'], [-6.24, -0.76])
        # - Two sample unequal variances
        tt = ttest(a, b, paired=False, correction=True, tail='two-sided')
        assert tt.loc['T-test', 'dof'] == 9.49
        assert round(tt.loc['T-test', 'p-val'], 5) == 0.01837
        np.testing.assert_allclose(tt.loc['T-test', 'CI95%'], [-6.26, -0.74])
        # - Paired
        tt = ttest(a, b, paired=True, correction=False, tail='two-sided')
        assert tt.loc['T-test', 'T'] == -2.445
        assert tt.loc['T-test', 'dof'] == 5
        assert round(tt.loc['T-test', 'p-val'], 5) == 0.05833
        np.testing.assert_allclose(tt.loc['T-test', 'CI95%'], [-7.18, 0.18])
        # - One sample one-sided
        tt = ttest(a, y=0, paired=False, correction=False, tail='one-sided')
        assert tt.loc['T-test', 'T'] == 5.175
        assert tt.loc['T-test', 'dof'] == 5
        assert round(tt.loc['T-test', 'p-val'], 3) == 0.002
        np.testing.assert_allclose(tt.loc['T-test', 'CI95%'], [3.05, 6.95])

    def test_anova(self):
        """Test function anova."""
        # Pain dataset
        df_pain = read_dataset('anova')
        aov = anova(dv='Pain threshold', between='Hair color', data=df_pain,
                    detailed=True, export_filename='test_export.csv')
        anova(dv='Pain threshold', between=['Hair color'], data=df_pain)
        # Compare with JASP
        assert np.allclose(aov.loc[0, 'F'], 6.791)
        assert np.allclose(np.round(aov.loc[0, 'p-unc'], 3), .004)
        assert np.allclose(aov.loc[0, 'np2'], .576)
        # Two-way ANOVA
        anova(dv='Scores', between=['Group', 'Time'], data=df,
              export_filename='test_export.csv')
        anova2(dv='Scores', between=['Group', 'Time'], data=df)
        anova2(dv='Scores', between=['Group'], data=df)
        anova2(dv='Scores', between='Group', data=df)

    def test_welch_anova(self):
        """Test function welch_anova."""
        # Pain dataset
        df_pain = read_dataset('anova')
        aov = welch_anova(dv='Pain threshold', between='Hair color',
                          data=df_pain, export_filename='test_export.csv')
        # Compare with R oneway.test function
        assert aov.loc[0, 'ddof1'] == 3
        assert np.allclose(aov.loc[0, 'ddof2'], 8.330)
        assert np.allclose(aov.loc[0, 'F'], 5.890)
        assert np.allclose(np.round(aov.loc[0, 'p-unc'], 4), .0188)

    def test_rm_anova(self):
        """Test function rm_anova."""
        rm_anova(dv='Scores', within='Time', subject='Subject', data=df,
                 correction=False, detailed=False)
        rm_anova(dv='Scores', within='Time', subject='Subject', data=df,
                 correction=True, detailed=False)
        aov = rm_anova(dv='Scores', within='Time', subject='Subject', data=df,
                       correction='auto', detailed=True)
        # Compare with JASP
        assert np.allclose(aov.loc[0, 'F'], 3.913)
        assert np.allclose(np.round(aov.loc[0, 'p-unc'], 3), .023)
        assert np.allclose(aov.loc[0, 'np2'], .062)

        rm_anova(dv='Scores', within='Time', subject='Subject', data=df,
                 correction=True, detailed=True)
        rm_anova(dv='Scores', within=['Time'], subject='Subject', data=df_nan,
                 export_filename='test_export.csv')
        # Using a wide dataframe with NaN and compare with JASP
        data = read_dataset('rm_anova_wide')
        aov = data.rm_anova(detailed=True, correction=True)
        assert aov.loc[0, 'F'] == 5.201
        assert round(aov.loc[0, 'p-unc'], 3) == .007
        assert aov.loc[0, 'np2'] == .394
        assert aov.loc[0, 'eps'] == .694
        assert aov.loc[0, 'W-spher'] == .307
        assert round(aov.loc[0, 'p-GG-corr'], 3) == .017

    def test_rm_anova2(self):
        """Test function rm_anova2."""
        data = pd.DataFrame({'Subject': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                             'Time': [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                             'Drug': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B',
                                      'C', 'A', 'B', 'C'],
                             'Scores': [3, 4, 5, 7, 2, 4, 5, 8, 5, 7, 3, 8]})
        rm_anova2(dv='Scores', within=['Time', 'Drug'], subject='Subject',
                  data=data, export_filename='test_export.csv')
        rm_anova(dv='Scores', within=['Time', 'Drug'], subject='Subject',
                 data=data)
        # With missing values
        df2 = read_dataset('rm_missing')
        df2.rm_anova(dv='BOLD', within=['Session', 'Time'], subject='Subj')

    def test_mixed_anova(self):
        """Test function anova."""
        aov = mixed_anova(dv='Scores', within='Time', subject='Subject',
                          between='Group', data=df, correction='auto')
        # Compare with JASP
        assert np.allclose(aov.loc[0, 'F'].round(3), 5.052)
        assert np.allclose(aov.loc[1, 'F'].round(3), 4.027)
        assert np.allclose(aov.loc[2, 'F'].round(3), 2.728)

        mixed_anova(dv='Scores', within='Time', subject='Subject',
                    between='Group', data=df_nan, correction=True,
                    export_filename='test_export.csv')

    def test_ancova(self):
        """Test function ancova."""
        df = read_dataset('ancova')
        aov = ancova(data=df, dv='Scores', covar='Income', between='Method')
        # Compare with statsmodels
        assert np.allclose(aov.loc[0, 'F'].round(3), 3.336)
        assert np.allclose(aov.loc[1, 'F'].round(3), 29.419)
        aov, bw = ancova(data=df, dv='Scores', covar='Income',
                         between='Method', export_filename='test_export.csv',
                         return_bw=True)
        ancova(data=df, dv='Scores', covar=['Income'], between='Method')
        ancova(data=df, dv='Scores', covar=['Income', 'BMI'],
               between='Method')

    def test_ancovan(self):
        """Test function ancovan."""
        df = read_dataset('ancova')
        aov = ancovan(data=df, dv='Scores', covar=['Income', 'BMI'],
                      between='Method')
        # Compare with statsmodels
        assert np.allclose(aov.loc[0, 'F'], 3.233)
        assert np.allclose(aov.loc[1, 'F'], 27.637)
        ancovan(data=df, dv='Scores', covar=['Income', 'BMI'],
                between='Method', export_filename='test_export.csv')
        ancovan(data=df, dv='Scores', covar=['Income'], between='Method')
        ancovan(data=df, dv='Scores', covar='Income', between='Method')
