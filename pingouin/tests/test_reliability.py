import pytest
import numpy as np
from unittest import TestCase
from pingouin.reliability import cronbach_alpha, intraclass_corr
from pingouin import read_dataset


class TestReliability(TestCase):
    """Test reliability.py.
    Compare against real-statistics.com
    """

    def test_cronbach_alpha(self):
        """Test function cronbach_alpha.
        Compare results with the R package psych.
        Note that this function returns slightly different results when
        missing values are present in data.
        """
        df = read_dataset('cronbach_alpha')
        alpha, ci = cronbach_alpha(data=df, items='Items', scores='Scores',
                                   subject='Subj')
        assert np.round(alpha, 3) == 0.592
        assert ci[0] == .195
        assert ci[1] == .840
        # With missing values
        df.loc[2, 'Scores'] = np.nan
        cronbach_alpha(data=df, items='Items', scores='Scores',
                       subject='Subj')
        # In R = psych:alpha(data, use="complete.obs")
        cronbach_alpha(data=df, items='Items', scores='Scores',
                       subject='Subj', nan_policy='listwise')
        # Wide format
        data = read_dataset('cronbach_wide_missing')
        alpha, _ = cronbach_alpha(data=data)
        assert np.round(alpha, 2) == .73
        alpha, _ = cronbach_alpha(data=data, nan_policy='listwise')
        assert np.round(alpha, 2) == .80

    def test_intraclass_corr(self):
        """Test function intraclass_corr"""
        df = read_dataset('icc')
        intraclass_corr(df, 'Wine', 'Judge', 'Scores', ci=.68)
        icc, ci = intraclass_corr(df, 'Wine', 'Judge', 'Scores')
        assert np.round(icc, 3) == 0.728
        assert ci[0] == .434
        assert ci[1] == .927
        with pytest.raises(ValueError):
            intraclass_corr(df, None, 'Judge', 'Scores')
        with pytest.raises(AssertionError):
            intraclass_corr(df, 'Wine', 'Judge', 'Judge')
        with pytest.raises(ValueError):
            intraclass_corr(df.drop(index=0), 'Wine', 'Judge', 'Scores')
