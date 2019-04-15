import pytest
import numpy as np
from unittest import TestCase
from pingouin.correlation import cronbach_alpha, intraclass_corr
from pingouin import read_dataset


class TestReliability(TestCase):
    """Test reliability.py.
    Compare against real-statistics.com
    """

    def test_cronbach_alpha(self):
        """Test function cronbach_alpha"""
        df = read_dataset('cronbach_alpha')
        alpha = cronbach_alpha(data=df, items='Items', scores='Scores',
                               subject='Subj')
        assert round(alpha, 3) == 0.592
        # With missing values
        df.loc[2, 'Scores'] = np.nan
        cronbach_alpha(data=df, items='Items', scores='Scores',
                       subject='Subj')

    def test_intraclass_corr(self):
        """Test function intraclass_corr"""
        df = read_dataset('icc')
        intraclass_corr(df, 'Wine', 'Judge', 'Scores', ci=.68)
        icc, ci = intraclass_corr(df, 'Wine', 'Judge', 'Scores')
        assert round(icc, 3) == 0.728
        assert ci[0] == .434
        assert ci[1] == .927
        with pytest.raises(ValueError):
            intraclass_corr(df, None, 'Judge', 'Scores')
        with pytest.raises(ValueError):
            intraclass_corr('error', 'Wine', 'Judge', 'Scores')
        with pytest.raises(ValueError):
            intraclass_corr(df, 'Wine', 'Judge', 'Judge')
        with pytest.raises(ValueError):
            intraclass_corr(df.drop(index=0), 'Wine', 'Judge', 'Scores')
