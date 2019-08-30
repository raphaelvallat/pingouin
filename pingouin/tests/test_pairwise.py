import pandas as pd
import numpy as np
import pytest

from unittest import TestCase
from pingouin.pairwise import (pairwise_ttests, pairwise_corr, pairwise_tukey,
                               pairwise_gameshowell)
from pingouin import read_dataset


class TestPairwise(TestCase):
    """Test pairwise.py."""

    def test_pairwise_ttests(self):
        """Test function pairwise_ttests.
        Tested against the pairwise.t.test R function."""
        df = read_dataset('mixed_anova.csv')
        # Within + Between + Within * Between
        pairwise_ttests(dv='Scores', within='Time', between='Group',
                        subject='Subject', data=df, alpha=.01)
        pairwise_ttests(dv='Scores', within=['Time'], between=['Group'],
                        subject='Subject', data=df, padjust='fdr_bh',
                        return_desc=True)
        # Simple within
        # In R:
        # >>> pairwise.t.test(df$Scores, df$Time, pool.sd = FALSE,
        # ...                 p.adjust.method = 'holm', paired = TRUE)
        pt = pairwise_ttests(dv='Scores', within='Time', subject='Subject',
                             data=df, return_desc=True, padjust='holm')
        np.testing.assert_array_equal(pt.loc[:, 'p-corr'].round(3),
                                      [0.174, 0.024, 0.310])
        np.testing.assert_array_equal(pt.loc[:, 'p-unc'].round(3),
                                      [0.087, 0.008, 0.310])
        pairwise_ttests(dv='Scores', within='Time', subject='Subject',
                        data=df, parametric=False, return_desc=True)
        # Simple between
        # In R:
        # >>> pairwise.t.test(df$Scores, df$Group, pool.sd = FALSE)
        pt = pairwise_ttests(dv='Scores', between='Group', data=df).round(3)
        assert pt.loc[0, 'p-unc'] == 0.023
        pairwise_ttests(dv='Scores', between='Group',
                        data=df, padjust='bonf', tail='one-sided',
                        effsize='cohen', parametric=False,
                        export_filename='test_export.csv')

        # Two between factors
        pt = pairwise_ttests(dv='Scores', between=['Time', 'Group'], data=df,
                             padjust='holm').round(3)
        pairwise_ttests(dv='Scores', between=['Time', 'Group'], data=df,
                        padjust='holm', parametric=False)
        # .. with no interaction
        pt_no_inter = df.pairwise_ttests(dv='Scores',
                                         between=['Time', 'Group'],
                                         interaction=False,
                                         padjust='holm').round(3)
        assert pt.drop(columns=['Time']).iloc[0:4, :].equals(pt_no_inter)

        # Two within subject factors
        ptw = pairwise_ttests(data=df, dv='Scores', within=['Group', 'Time'],
                              subject='Subject', padjust='bonf',
                              parametric=False).round(3)
        ptw_no_inter = df.pairwise_ttests(dv='Scores',
                                          within=['Group', 'Time'],
                                          subject='Subject', padjust='bonf',
                                          interaction=False,
                                          parametric=False).round(3)
        assert ptw.drop(columns=['Group']).iloc[0:4, :].equals(ptw_no_inter)

        # Both multiple between and multiple within
        with pytest.raises(ValueError):
            pairwise_ttests(dv='Scores', between=['Time', 'Group'],
                            within=['Time', 'Group'], subject='Subject',
                            data=df)

        # Wrong input argument
        df['Group'] = 'Control'
        with pytest.raises(ValueError):
            pairwise_ttests(dv='Scores', between='Group', data=df)

        # Missing values in repeated measurements
        # 1. Parametric
        df = read_dataset('pairwise_ttests_missing')
        st = pairwise_ttests(dv='Value', within='Condition', subject='Subject',
                             data=df, nan_policy='listwise')
        np.testing.assert_array_equal(st['dof'].values, [7, 7, 7])
        st2 = pairwise_ttests(dv='Value', within='Condition', data=df,
                              subject='Subject', nan_policy='pairwise')
        np.testing.assert_array_equal(st2['dof'].values, [8, 7, 8])
        # 2. Non-parametric
        st = pairwise_ttests(dv='Value', within='Condition', subject='Subject',
                             data=df, parametric=False, nan_policy='listwise')
        np.testing.assert_array_equal(st['W-val'].values, [9, 3, 12])
        st2 = pairwise_ttests(dv='Value', within='Condition', data=df,
                              subject='Subject', nan_policy='pairwise',
                              parametric=False)
        # Tested against a simple for loop on combinations
        np.testing.assert_array_equal(st2['W-val'].values, [9, 3, 21])

        with pytest.raises(ValueError):
            # Unbalanced design in repeated measurements
            df_unbalanced = df.iloc[1:, :].copy()
            pairwise_ttests(data=df_unbalanced, dv='Value', within='Condition',
                            subject='Subject')

        # Two within factors from other datasets and with NaN values
        df2 = read_dataset('rm_anova')
        pairwise_ttests(dv='DesireToKill',
                        within=['Disgustingness', 'Frighteningness'],
                        subject='Subject', padjust='holm', data=df2)

        # Compare with JASP tail / parametric argument
        df = read_dataset('pairwise_ttests')
        # 1. Within
        # 1.1 Parametric
        # 1.1.1 Tail is greater
        pt = pairwise_ttests(dv='Scores', within='Drug', subject='Subject',
                             data=df, tail='greater')
        np.testing.assert_array_equal(pt.loc[:, 'p-unc'].round(3),
                                      [0.907, 0.941, 0.405])
        assert all(pt.loc[:, 'BF10'].astype(float) < 1)
        # 1.1.2 Tail is less
        pt = pairwise_ttests(dv='Scores', within='Drug', subject='Subject',
                             data=df, tail='less')
        np.testing.assert_array_equal(pt.loc[:, 'p-unc'].round(3),
                                      [0.093, 0.059, 0.595])
        assert sum(pt.loc[:, 'BF10'].astype(float) > 1) == 2
        # 1.1.3 Tail is one-sided: smallest p-value
        pt = pairwise_ttests(dv='Scores', within='Drug', subject='Subject',
                             data=df, tail='one-sided')
        np.testing.assert_array_equal(pt.loc[:, 'p-unc'].round(3),
                                      [0.093, 0.059, 0.405])

        # 1.2 Non-parametric
        # 1.2.1 Tail is greater
        pt = pairwise_ttests(dv='Scores', within='Drug', subject='Subject',
                             parametric=False, data=df, tail='greater')
        np.testing.assert_array_equal(pt.loc[:, 'p-unc'].round(3),
                                      [0.910, 0.951, 0.482])
        # 1.2.2 Tail is less
        pt = pairwise_ttests(dv='Scores', within='Drug', subject='Subject',
                             parametric=False, data=df, tail='less')
        np.testing.assert_array_equal(pt.loc[:, 'p-unc'].round(3),
                                      [0.108, 0.060, 0.554])
        # 1.2.3 Tail is one-sided: smallest p-value
        pt = pairwise_ttests(dv='Scores', within='Drug', subject='Subject',
                             parametric=False, data=df, tail='one-sided')
        np.testing.assert_array_equal(pt.loc[:, 'p-unc'].round(3),
                                      [0.108, 0.060, 0.482])

        # Compare the RBC value for wilcoxon
        from pingouin.nonparametric import wilcoxon
        x = df[df['Drug'] == 'A']['Scores'].values
        y = df[df['Drug'] == 'B']['Scores'].values
        assert -0.6 < wilcoxon(x, y).at['Wilcoxon', 'RBC'] < -0.4
        x = df[df['Drug'] == 'B']['Scores'].values
        y = df[df['Drug'] == 'C']['Scores'].values
        assert wilcoxon(x, y).at['Wilcoxon', 'RBC'].round(3) == 0.030

        # 2. Between
        # 2.1 Parametric
        # 2.1.1 Tail is greater
        pt = pairwise_ttests(dv='Scores', between='Gender',
                             data=df, tail='greater')
        assert pt.loc[0, 'p-unc'].round(3) == 0.068
        assert float(pt.loc[0, 'BF10']) > 1
        # 2.1.2 Tail is less
        pt = pairwise_ttests(dv='Scores', between='Gender',
                             data=df, tail='less')
        assert pt.loc[0, 'p-unc'].round(3) == 0.932
        assert float(pt.loc[0, 'BF10']) < 1
        # 2.1.3 Tail is one-sided: smallest p-value
        pt = pairwise_ttests(dv='Scores', between='Gender',
                             data=df, tail='one-sided')
        assert pt.loc[0, 'p-unc'].round(3) == 0.068
        assert float(pt.loc[0, 'BF10']) > 1

        # 2.2 Non-parametric
        # 2.2.1 Tail is greater
        pt = pairwise_ttests(dv='Scores', between='Gender',
                             parametric=False, data=df, tail='greater')
        assert pt.loc[0, 'p-unc'].round(3) == 0.105
        # 2.2.2 Tail is less
        pt = pairwise_ttests(dv='Scores', between='Gender',
                             parametric=False, data=df, tail='less')
        assert pt.loc[0, 'p-unc'].round(3) == 0.901
        # 2.2.3 Tail is one-sided: smallest p-value
        pt = pairwise_ttests(dv='Scores', between='Gender',
                             parametric=False, data=df, tail='one-sided')
        assert pt.loc[0, 'p-unc'].round(3) == 0.105

        # Compare the RBC value for MWU
        from pingouin.nonparametric import mwu
        x = df[df['Gender'] == 'M']['Scores'].values
        y = df[df['Gender'] == 'F']['Scores'].values
        assert abs(mwu(x, y).at['MWU', 'RBC']) == 0.252

    def test_pairwise_tukey(self):
        """Test function pairwise_tukey"""
        df = read_dataset('anova')
        stats = pairwise_tukey(dv='Pain threshold', between='Hair color',
                               data=df)
        assert np.allclose([0.074, 0.435, 0.415, 0.004, 0.789, 0.037],
                           stats.loc[:, 'p-tukey'].values.round(3), atol=0.05)

    def test_pairwise_gameshowell(self):
        """Test function pairwise_gameshowell"""
        df = read_dataset('anova')
        stats = pairwise_gameshowell(dv='Pain threshold', between='Hair color',
                                     data=df)
        # Compare with R package `userfriendlyscience`
        np.testing.assert_array_equal(np.abs(stats['T'].round(2)),
                                      [2.48, 1.42, 1.75, 4.09, 1.11, 3.56])
        np.testing.assert_array_equal(stats['df'].round(2),
                                      [7.91, 7.94, 6.56, 8.0, 6.82, 6.77])
        sig = stats['pval'].apply(lambda x: 'Yes' if x < 0.05 else 'No').values
        np.testing.assert_array_equal(sig, ['No', 'No', 'No', 'Yes', 'No',
                                            'Yes'])

    def test_pairwise_corr(self):
        """Test function pairwise_corr"""
        # Load JASP Big 5 DataSets (remove subject column)
        data = read_dataset('pairwise_corr').iloc[:, 1:]
        stats = pairwise_corr(data=data, method='pearson', tail='two-sided')
        jasp_rval = [-0.350, -0.01, -.134, -.368, .267, .055, .065, .159,
                     -.013, .159]
        assert np.allclose(stats['r'].values, jasp_rval)
        assert stats['n'].values[0] == 500
        # Correct for multiple comparisons
        pairwise_corr(data=data, method='spearman', tail='one-sided',
                      padjust='bonf')
        # Export
        pairwise_corr(data=data, method='spearman', tail='one-sided',
                      export_filename='test_export.csv')
        # Check with a subset of columns
        pairwise_corr(data=data, columns=['Neuroticism', 'Extraversion'])
        with pytest.raises(ValueError):
            pairwise_corr(data=data, columns='wrong')
        # Check with non-numeric columns
        data['test'] = 'test'
        pairwise_corr(data=data, method='pearson')
        # Check different variation of product / combination
        n = data.shape[0]
        data['Age'] = np.random.randint(18, 65, n)
        data['IQ'] = np.random.normal(105, 1, n)
        data['One'] = 1
        data['Gender'] = np.repeat(['M', 'F'], int(n / 2))
        pairwise_corr(data, columns=['Neuroticism', 'Gender'],
                      method='shepherd')
        pairwise_corr(data, columns=['Neuroticism', 'Extraversion', 'Gender'])
        pairwise_corr(data, columns=['Neuroticism'])
        pairwise_corr(data, columns='Neuroticism', method='skipped')
        pairwise_corr(data, columns=[['Neuroticism']], method='spearman')
        pairwise_corr(data, columns=[['Neuroticism'], None], method='percbend')
        pairwise_corr(data, columns=[['Neuroticism', 'Gender'], ['Age']])
        pairwise_corr(data, columns=[['Neuroticism'], ['Age', 'IQ']])
        pairwise_corr(data, columns=[['Age', 'IQ'], []])
        pairwise_corr(data, columns=['Age', 'Gender', 'IQ', 'Wrong'])
        pairwise_corr(data, columns=['Age', 'Gender', 'Wrong'])
        # Test with no good combinations
        with pytest.raises(ValueError):
            pairwise_corr(data, columns=['Gender', 'Gender'])
        # Test when one column has only one unique value
        pairwise_corr(data=data, columns=['Age', 'One', 'Gender'])
        stats = pairwise_corr(data, columns=['Neuroticism', 'IQ', 'One'])
        assert stats.shape[0] == 1
        # Test with covariate
        pairwise_corr(data, covar='Age')
        pairwise_corr(data, covar=['Age', 'Neuroticism'])
        with pytest.raises(AssertionError):
            pairwise_corr(data, covar=['Age', 'Gender'])
        with pytest.raises(ValueError):
            pairwise_corr(data, columns=['Neuroticism', 'Age'], covar='Age')
        # Partial pairwise with missing values
        data.loc[[4, 5, 8, 20, 22], 'Age'] = np.nan
        data.loc[[10, 12], 'Neuroticism'] = np.nan
        pairwise_corr(data)
        pairwise_corr(data, covar='Age')
        # Listwise deletion
        assert (pairwise_corr(data, covar='Age',
                              nan_policy='listwise')['n'].nunique() == 1)
        assert pairwise_corr(data, nan_policy='listwise')['n'].nunique() == 1
        ######################################################################
        # MultiIndex columns
        from numpy.random import random as rdm
        # Create MultiIndex dataframe
        columns = pd.MultiIndex.from_tuples([('Behavior', 'Rating'),
                                             ('Behavior', 'RT'),
                                             ('Physio', 'BOLD'),
                                             ('Physio', 'HR'),
                                             ('Psycho', 'Anxiety')])
        data = pd.DataFrame(dict(Rating=rdm(size=10),
                                 RT=rdm(size=10),
                                 BOLD=rdm(size=10),
                                 HR=rdm(size=10),
                                 Anxiety=rdm(size=10)))
        data.columns = columns
        pairwise_corr(data, method='spearman')
        stats = pairwise_corr(data, columns=[('Behavior', 'Rating')])
        assert stats.shape[0] == data.shape[1] - 1
        pairwise_corr(data, columns=[('Behavior', 'Rating'),
                                     ('Behavior', 'RT')])
        st1 = pairwise_corr(data, columns=[[('Behavior', 'Rating'),
                                            ('Behavior', 'RT')], None])
        st2 = pairwise_corr(data, columns=[[('Behavior', 'Rating'),
                                            ('Behavior', 'RT')]])
        assert st1['X'].equals(st2['X'])
        st3 = pairwise_corr(data, columns=[[('Behavior', 'Rating')],
                                           [('Behavior', 'RT'),
                                            ('Physio', 'BOLD')]])
        assert st3.shape[0] == 2
        # With covar
        pairwise_corr(data, covar=[('Psycho', 'Anxiety')])
        pairwise_corr(data, columns=[('Behavior', 'Rating')],
                      covar=[('Psycho', 'Anxiety')])
        # With missing values
        data.iloc[2, [2, 3]] = np.nan
        data.iloc[[1, 4], [1, 4]] = np.nan
        assert pairwise_corr(data, nan_policy='listwise')['n'].nunique() == 1
        assert pairwise_corr(data, nan_policy='pairwise')['n'].nunique() == 3
        assert (pairwise_corr(data, columns=[('Behavior', 'Rating')],
                covar=[('Psycho', 'Anxiety')],
                nan_policy='listwise')['n'].nunique() == 1)
        assert (pairwise_corr(data, columns=[('Behavior', 'Rating')],
                covar=[('Psycho', 'Anxiety')],
                nan_policy='pairwise')['n'].nunique() == 2)
