import pandas as pd
import numpy as np
import pytest

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.pairwise import (pairwise_ttests, pairwise_corr, pairwise_tukey,
                               pairwise_gameshowell)
from pingouin.datasets import read_dataset

# Dataset for pairwise_ttests
df = read_dataset('mixed_anova.csv')


class TestPairwise(_TestPingouin):
    """Test pairwise.py."""

    def test_pairwise_ttests(self):
        """Test function pairwise_ttests"""
        # Within + Between + Within * Between
        pairwise_ttests(dv='Scores', within='Time', between='Group',
                        subject='Subject', data=df, alpha=.01)
        pairwise_ttests(dv='Scores', within=['Time'], between=['Group'],
                        subject='Subject', data=df, padjust='fdr_bh',
                        return_desc=True)
        # Simple within
        pairwise_ttests(dv='Scores', within='Time', subject='Subject',
                        data=df, return_desc=True)
        # Simple between
        pairwise_ttests(dv='Scores', between='Group',
                        data=df, padjust='bonf', tail='one-sided',
                        effsize='cohen', export_filename='test_export.csv')

        # Two between factors
        pairwise_ttests(dv='Scores', between=['Time', 'Group'], data=df,
                        padjust='holm')

        # Two within subject factors
        pairwise_ttests(dv='Scores', within=['Group', 'Time'],
                        subject='Subject', data=df, padjust='bonf')

        # Wrong tail argument
        with pytest.raises(ValueError):
            pairwise_ttests(dv='Scores', between='Group', data=df,
                            tail='wrong')
        # Wrong alpha argument
        with pytest.raises(ValueError):
            pairwise_ttests(dv='Scores', between='Group', data=df, alpha='.05')

        # Both multiple between and multiple within
        with pytest.raises(ValueError):
            pairwise_ttests(dv='Scores', between=['Time', 'Group'],
                            within=['Time', 'Group'], subject='Subject',
                            data=df)

        # Missing values
        df.iloc[[10, 15], 0] = np.nan
        pairwise_ttests(dv='Scores', within='Time', subject='Subject', data=df)
        # Wrong input argument
        df['Group'] = 'Control'
        with pytest.raises(ValueError):
            pairwise_ttests(dv='Scores', between='Group', data=df)

        # Two within factors from other datasets and with NaN values
        df2 = read_dataset('rm_anova')
        pairwise_ttests(dv='DesireToKill',
                        within=['Disgustingness', 'Frighteningness'],
                        subject='Subject', padjust='holm', data=df2)

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
            pairwise_corr(data=data, tail='wrong')
        # Check with non-numeric columns
        data['test'] = 'test'
        pairwise_corr(data=data, method='pearson')
        # Check different variation of product / combination
        n = data.shape[0]
        data['Age'] = np.random.randint(18, 65, n)
        data['IQ'] = np.random.normal(105, 1, n)
        data['Gender'] = np.repeat(['M', 'F'], int(n / 2))
        pairwise_corr(data, columns=['Neuroticism', 'Gender'])
        pairwise_corr(data, columns=['Neuroticism', 'Extraversion', 'Gender'])
        pairwise_corr(data, columns=['Neuroticism'])
        pairwise_corr(data, columns='Neuroticism')
        pairwise_corr(data, columns=[['Neuroticism']])
        pairwise_corr(data, columns=[['Neuroticism'], None])
        pairwise_corr(data, columns=[['Neuroticism', 'Gender'], ['Age']])
        pairwise_corr(data, columns=[['Neuroticism'], ['Age', 'IQ']])
        pairwise_corr(data, columns=[['Age', 'IQ'], []])
        pairwise_corr(data, columns=['Age', 'Gender', 'IQ', 'Wrong'])
        pairwise_corr(data, columns=['Age', 'Gender', 'Wrong'])
        # Test with more than 1000 columns (BF10 not computed)
        data1500 = pd.concat([data, data, data], ignore_index=True)
        pcor1500 = pairwise_corr(data1500, method='pearson')
        assert 'BF10' not in pcor1500.keys()
