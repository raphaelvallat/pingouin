import pandas as pd
import numpy as np
import pytest

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.pairwise import pairwise_ttests

# Dataset
n = 30
months = ['August', 'January', 'June']
# Generate random data
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


class TestPairwise(_TestPingouin):
    """Test pairwise.py."""

    def test_pairwise_ttests(self):
        """Test function pairwise_ttests"""
        pairwise_ttests(dv='Scores', within='Time', between='Group',
                        effects='interaction', data=df, padjust='holm',
                        alpha=.01)
        pairwise_ttests(dv='Scores', within='Time', between='Group',
                        effects='all', data=df, padjust='fdr_bh')
        pairwise_ttests(dv='Scores', within='Time', between=None,
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
            pairwise_ttests(dv='Scores', within='Time', data=df, tail='wrong')
        # Wrong alpha argument
        with pytest.raises(ValueError):
            pairwise_ttests(dv='Scores', within='Time', data=df, alpha='.05')
        # Missing values
        df.iloc[[10, 15], 1] = np.nan
        pairwise_ttests(dv='Scores', within='Time', effects='within', data=df)
        # Wrong input argument
        df['Group'] = 'Control'
        with pytest.raises(ValueError):
            pairwise_ttests(dv='Scores', between='Group', data=df)
