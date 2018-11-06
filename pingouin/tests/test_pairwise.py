import pandas as pd
import numpy as np
import pytest

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.pairwise import (pairwise_ttests, pairwise_corr, pairwise_tukey,
                               pairwise_gameshowell)
from pingouin.datasets import read_dataset

# Dataset for pairwise_ttests
n = 30
months = ['August', 'January', 'June']
# Generate random data
np.random.seed(1234)
control = np.random.normal(5.5, size=len(months) * n)
meditation = np.r_[np.random.normal(5.5, size=n),
                   np.random.normal(5.8, size=n),
                   np.random.normal(6.4, size=n)]

df = pd.DataFrame({'Scores': np.r_[control, meditation],
                   'Time': np.r_[np.repeat(months, n), np.repeat(months, n)],
                   'Group': np.repeat(['Control', 'Meditation'],
                                      len(months) * n),
                   'Subject': np.r_[np.tile(np.arange(n), 3),
                                    np.tile(np.arange(n, n + n), 3)]})


class TestPairwise(_TestPingouin):
    """Test pairwise.py."""

    def test_pairwise_ttests(self):
        """Test function pairwise_ttests"""
        pairwise_ttests(dv='Scores', within='Time', between='Group',
                        subject='Subject', effects='interaction', data=df,
                        padjust='holm', alpha=.01)
        pairwise_ttests(dv='Scores', within='Time', between='Group',
                        subject='Subject', effects='all', data=df,
                        padjust='fdr_bh')
        pairwise_ttests(dv='Scores', within='Time', subject='Subject',
                        effects='within', data=df, padjust='none',
                        return_desc=False)
        pairwise_ttests(dv='Scores', within=None, between='Group',
                        effects='between', data=df, padjust='bonf',
                        tail='one-sided', effsize='cohen')
        pairwise_ttests(dv='Scores', within=None, between='Group',
                        effects='between', data=df,
                        export_filename='test_export.csv')
        # Wrong tail argument
        with pytest.raises(ValueError):
            pairwise_ttests(dv='Scores', between='Group', data=df,
                            tail='wrong')
        # Wrong alpha argument
        with pytest.raises(ValueError):
            pairwise_ttests(dv='Scores', between='Group', data=df, alpha='.05')
        # Missing values
        df.iloc[[10, 15], 0] = np.nan
        pairwise_ttests(dv='Scores', within='Time', effects='within',
                        subject='Subject', data=df)
        # Wrong input argument
        df['Group'] = 'Control'
        with pytest.raises(ValueError):
            pairwise_ttests(dv='Scores', between='Group', data=df)

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
